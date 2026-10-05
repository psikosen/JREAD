#!/usr/bin/env python3
"""Frozen local Hy-MT2 comparison; reuse the existing owned-server lifecycle."""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import evaluate_index_translate as rt
import evaluate_translation_ensemble as ensemble

ROOT = Path(__file__).resolve().parents[1]
BASE = Path('/media/psikosen/newvolume/jread-hymt2')
PROTOCOL = ROOT/'tests/fixtures/accuracy-hymt2.json'
LOCK = Path('/media/psikosen/newvolume/jread-verifier/evaluation.lock')
RUNNER = Path(__file__).resolve()
MODEL = BASE/'model/Hy-MT2-1.8B-Q8_0.gguf'
ALIAS = 'jread-eval-hymt2-q8'
sha, write, payload_bytes = rt.sha, rt.write, rt.payload_bytes


def binding(path):
    return {'path':str(path),'sha256':sha(path)}


def read(ref):
    assert binding(Path(ref['path'])) == ref, f'Changed binding: {ref["path"]}'
    return json.loads(Path(ref['path']).read_text())


def matched_terms(text, glossary):
    result=[]
    for entry in glossary['entries']:
        if entry['source'] not in text:
            continue
        anchors=entry.get('require_any_source',[])
        if anchors and not any(anchor in text for anchor in anchors):
            continue
        result.append({'Japanese':entry['source'],'English':entry['english']})
    return result


def request_for(model, case, variant, glossary):
    assert model in ('hymt2','gemma') and variant in ('baseline','glossary','following')
    text=case['text']
    terms=matched_terms(text,glossary) if variant=='glossary' else []
    following=case.get('following_context','') if variant=='following' else ''
    if variant=='following' and not following:
        raise ValueError('Following-line condition requires its frozen Japanese context.')
    if model=='hymt2':
        prompt='Translate the following text into English. Note that you should only output the translated result without any additional explanation:\n\n'+text
        if terms:
            prompt='Reference the following translations:\n'+'\n'.join(x['Japanese']+' translates to '+x['English'] for x in terms)+'\n\n'+prompt
        if following:
            prompt='[Background Information]\nThe following Japanese utterance comes immediately AFTER the source text. Use it only as context; do not translate it:\n'+following+'\n\n[Source Text to Translate]\n'+prompt
        return {'model':ALIAS,'messages':[{'role':'user','content':prompt}],
                'stream':False,'temperature':0,'max_tokens':1024,'cache_prompt':False,
                'chat_template_kwargs':{'enable_thinking':False}}
    source={k:case[k] for k in ('start','end','text')}
    context=('Following Japanese utterance (reference only): '+following) if following else ''
    messages=ensemble.gemma._messages([source],context,None)
    if terms:
        messages[0]['content']+=' Use these source-matched terminology hints only for the quoted Japanese terms. Preserve the source roles, negation and scope; the hints do not add facts: '+json.dumps(terms,ensure_ascii=False,separators=(',',':'))
    return ensemble.payload('4b',messages,0,None,1024)


def ports_free():
    return {str(port):not rt.port_accepting(port) for port in (18767,18768)}


def validate():
    p=json.loads(PROTOCOL.read_text())
    if p.get('status')!='frozen' or p['runner_sha256']!=sha(RUNNER):
        raise ValueError('Frozen runner protocol required.')
    for ref in p['bindings']:
        if binding(Path(ref['path']))!=ref:
            raise ValueError(f'Changed model, helper or data: {ref["path"]}')
    if p['lock']!=str(LOCK) or p['ports']!=[18767,18768]:
        raise ValueError('The global model lock and private ports must remain fixed.')
    if p['models']['gemma']!=rt.current_gemma():
        raise ValueError('Gemma model/runtime changed.')
    if p['models']['hymt2']['path']!=str(MODEL) or p['models']['hymt2']['sha256']!=sha(MODEL):
        raise ValueError('Hy-MT2 model bytes changed.')
    if p['models']['hymt2']['sha256']!='5c3fe0b1408a5ceb0143184ef247b11b579c525f4b02b060e6c851bb76fef1a4':
        raise ValueError('Expected the pinned official Q8_0 model.')
    sources=read(p['sources']);glossary=read(p['glossary']);plan=read(p['plan'])
    cases={row['id']:row for row in sources['cases']}
    if len(cases)!=37 or any(set(c)!={'id','partition','text','start','end','following_context'} for c in cases.values()):
        raise ValueError('Require the frozen 20 development, nine exposed caption and eight fresh source-only cases.')
    if list(plan['stages'])!=p['stage_order']:
        raise ValueError('Stage order changed.')
    for stage, spec in plan['stages'].items():
        if [t['case_id'] for t in spec['tasks']]!=p['stage_ids'][stage]:
            raise ValueError('Case denominator/order changed.')
        if spec['model'] not in ('hymt2','gemma'):
            raise ValueError('Unknown model.')
        for task in spec['tasks']:
            expected=request_for(spec['model'],cases[task['case_id']],spec['variant'],glossary)
            raw=payload_bytes(expected)
            if task['request']!=expected or task['request_sha256']!=hashlib.sha256(raw).hexdigest():
                raise ValueError('Frozen source/prompt request differs.')
    return p,plan


def checked_report(path,p,stage):
    r=json.loads(path.read_text())
    spec=read(p['plan'])['stages'][stage]
    if (r['protocol']!=binding(PROTOCOL) or r['stage']!=stage or
        r['model']!=spec['model'] or r['variant']!=spec['variant'] or
        r['status'] not in ('complete','complete_with_errors') or
        not r['cleanup'].get('released') or not all(r['cleanup'].get('private_ports_free',{}).values()) or
        set(r['cleanup'].get('private_ports_free',{}))!={'18767','18768'} or
        [s['case_id'] for s in r['rows']]!=p['stage_ids'][stage]):
        raise ValueError(f'Invalid prerequisite report: {stage}')
    if stage=='readiness' and any(s['status']!='ok' for s in r['rows']):
        raise ValueError('Hy-MT2 readiness did not pass the output contract.')
    for row,task in zip(r['rows'],spec['tasks'],strict=True):
        if row['request']!=task['request']:
            raise ValueError('Prior request differs from the frozen task.')
        if row['request_sha256']!=hashlib.sha256(payload_bytes(row['request'])).hexdigest():
            raise ValueError('Prior request bytes differ.')
        if row['status']=='ok':
            raw=row['raw_response_json_utf8']
            if (sha_text(raw)!=row['response_sha256'] or
                    len(raw.encode('utf-8'))!=row['response_bytes'] or
                    json.loads(raw)!=row['raw_response'] or
                    rt.require_translation_response(json.loads(raw),strict_json=spec['model']=='gemma')!=row['text']):
                raise ValueError('Prior response bytes differ.')
    return r


def sha_text(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def run(stage):
    p,plan=validate()
    if stage not in p['stage_order']:raise ValueError('Unknown phase.')
    spec=plan['stages'][stage];out=BASE/'runs'/stage
    cache={};deps=[]
    for prior in p['stage_order'][:p['stage_order'].index(stage)]:
        path=BASE/'runs'/prior/'report.json';report=checked_report(path,p,prior)
        deps.append(binding(path))
        for row in report['rows']:
            if row['status']=='ok':
                cache[(report['model'],payload_bytes(row['request']))]=(row,binding(path))
    if out.exists():raise ValueError(f'Refusing existing output: {out}')
    report={'status':'preparing','stage':stage,'model':spec['model'],'variant':spec['variant'],
            'protocol':binding(PROTOCOL),'dependencies':deps,'expected_ids':p['stage_ids'][stage],
            'started_at_utc':None,'rows':[],'cleanup':{},'production_changed':False,'heldout_rows_used':0,
            'runtime':{'device':'CPU','threads':8,'context_tokens':4096,'temperature':0,'max_tokens':1024},
            'model_provenance':p['models'][spec['model']]}
    old_term=signal.getsignal(signal.SIGTERM)
    def terminate(_signum,_frame):raise KeyboardInterrupt('SIGTERM')
    old_port=(rt.gemma.PORT,rt.gemma.BASE)
    old_index=(rt.INDEX_MODEL,rt.ALIAS,rt.DEADLINE)
    process=None;log=None;server=None;created=False
    LOCK.parent.mkdir(parents=True,exist_ok=True)
    with LOCK.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            if not all(ports_free().values()):raise ValueError('Private model port already occupied.')
            out.mkdir(parents=True,exist_ok=False);created=True
            signal.signal(signal.SIGTERM,terminate)
            rt.DEADLINE=time.monotonic()+1800
            report.update(status='running',started_at_utc=datetime.now(timezone.utc).isoformat())
            write(out/'report.json',report)
            needs_model=any((spec['model'],payload_bytes(t['request'])) not in cache for t in spec['tasks'])
            if needs_model:
                if spec['model']=='hymt2':
                    rt.INDEX_MODEL=MODEL;rt.ALIAS=ALIAS
                    process,log,seconds,cmd,identity=rt.start_index(out)
                    report.update(model_load_seconds=seconds,server_command=cmd,server_identity=identity)
                else:
                    server,_=rt.start_gemma()
                    report.update(model_load_seconds=server['load_seconds'],server_identity=server)
            for task in spec['tasks']:
                row={'case_id':task['case_id'],'request':task['request'],'request_sha256':task['request_sha256']}
                saved=cache.get((spec['model'],payload_bytes(task['request'])))
                if saved:
                    old,origin=saved
                    row.update(old,reused=True,reuse_origin=origin,seconds=0.0,
                               historical_seconds=old.get('historical_seconds',old['seconds']))
                    row['case_id']=task['case_id']
                else:
                    result=rt.call('http://127.0.0.1:18767',task['request'],
                                   auth=spec['model']=='gemma',strict_json=spec['model']=='gemma')
                    row.update(result,reused=False)
                    if row['status']=='ok' and '<｜' in row['text']:
                        row.update(status='error',error='Exposed Hunyuan protocol marker.')
                report['rows'].append(row);write(out/'report.json',report)
            report['status']='complete_with_errors' if any(x['status']!='ok' for x in report['rows']) else 'complete'
        except BaseException as error:
            report.update(status='failed',error=f'{type(error).__name__}: {error}')
            raise
        finally:
            if created:
                try:
                    if spec['model']=='gemma':
                        cleanup=rt.stop_gemma(old_port)
                    else:
                        cleanup=rt.stop_owned(process)
                        if log:log.close()
                    cleanup['private_ports_free']=ports_free()
                    cleanup['released']=cleanup['released'] and not cleanup.get('group_remaining') and all(cleanup['private_ports_free'].values())
                    report['cleanup']=cleanup
                    if not cleanup['released']:report.update(status='failed',error='Owned model cleanup failed.')
                except BaseException as error:
                    report.update(status='failed',cleanup={'released':False,'error':repr(error)})
                report['finished_at_utc']=datetime.now(timezone.utc).isoformat()
                report['inference_seconds']=sum(x.get('seconds',0) for x in report['rows'])
                report['reused_historical_seconds']=sum(x.get('historical_seconds',0) for x in report['rows'] if x.get('reused'))
                write(out/'report.json',report)
            rt.INDEX_MODEL,rt.ALIAS,rt.DEADLINE=old_index
            rt.gemma.PORT,rt.gemma.BASE=old_port
            signal.signal(signal.SIGTERM,old_term)
    return report


def self_test():
    case={'id':'check','text':'動画を見た。','start':0.0,'end':1.0,'following_context':'テレビで見た。'}
    glossary={'entries':[{'source':'動画','english':'in-between drawings','require_any_source':['原画']}]}
    for model in ('hymt2','gemma'):
        assert request_for(model,case,'baseline',glossary)==request_for(model,case,'glossary',glossary)
        assert request_for(model,case,'baseline',glossary)!=request_for(model,case,'following',glossary)
    assert not matched_terms('動画を見た。',glossary)
    assert matched_terms('原画の間をつなぐ動画',glossary)==[{'Japanese':'動画','English':'in-between drawings'}]
    assert 'テレビ' not in json.dumps(request_for('hymt2',case,'baseline',glossary),ensure_ascii=False)
    source={k:case[k] for k in ('start','end','text')}
    assert request_for('gemma',case,'baseline',glossary)==ensemble.payload('4b',ensemble.gemma._messages([source],'',None),0,None,1024)
    import tempfile
    from unittest.mock import patch
    request=request_for('hymt2',case,'baseline',glossary)
    raw={'choices':[{'finish_reason':'stop','message':{'role':'assistant','content':'A video was watched.'}}]}
    encoded=json.dumps(raw)
    row={'case_id':'check','request':request,'request_sha256':hashlib.sha256(payload_bytes(request)).hexdigest(),
         'status':'ok','text':'A video was watched.','raw_response':raw,'raw_response_json_utf8':encoded,
         'response_sha256':sha_text(encoded),'response_bytes':len(encoded.encode())}
    mock_binding={'path':'test','sha256':'test'}
    report={'protocol':mock_binding,'stage':'readiness','model':'hymt2','variant':'baseline','status':'complete',
            'cleanup':{'released':True,'private_ports_free':{'18767':True,'18768':True}},'rows':[row]}
    plan={'stages':{'readiness':{'model':'hymt2','variant':'baseline','tasks':[{'case_id':'check','request':request}]}}}
    p={'plan':{},'stage_ids':{'readiness':['check']}}
    with tempfile.TemporaryDirectory() as temp, patch(__name__+'.binding',return_value=mock_binding), patch(__name__+'.read',return_value=plan):
        path=Path(temp)/'report.json';path.write_text(json.dumps(report))
        checked_report(path,p,'readiness')
        report['rows'][0]['text']='An edited result.';path.write_text(json.dumps(report))
        try:
            checked_report(path,p,'readiness')
            raise AssertionError('Edited cached text must be rejected.')
        except ValueError:
            pass
    print('Hy-MT2 prompt/source matching self-test passed; no inference.')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage');parser.add_argument('--self-test',action='store_true');parser.add_argument('--validate',action='store_true')
    args=parser.parse_args()
    if args.self_test:self_test()
    elif args.validate:validate();print('Frozen Hy-MT2 plan validated; no inference.')
    elif args.stage:
        result=run(args.stage);print(json.dumps({'stage':args.stage,'status':result['status'],'rows':len(result['rows']),'cleanup':result['cleanup']}))
    else:parser.error('Specify --stage, --validate or --self-test.')
