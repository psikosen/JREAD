from __future__ import annotations
import hashlib, json, os, pathlib, subprocess, sys, tempfile

REPO='google/translategemma-4b-it'
REV='10042cb0e6e7fdce748996a71dc3dc432a4e0c89'
DEST=pathlib.Path('/media/psikosen/newvolume/jread-translategemma/model')
HF='/media/psikosen/newvolume/jread-finetune/runtime/venv/bin/hf'
THREAD='01a0febb-0589-7460-bd4c-0dff152bdff6'
CODEX='/home/psikosen/.local/bin/codex'
EXPECTED={
'.gitattributes':(1570,'52373fe24473b1aa44333d318f578ae6bf04b49b',None),
'README.md':(17508,'44f3690778da1770bf19d64bace73480c5825bcc',None),
'added_tokens.json':(35,'e17bde03d42feda32d1abfca6d3b598b9a020df7',None),
'chat_template.jinja':(16982,'d571968f9770221512d29627697012202451983c',None),
'config.json':(2618,'84ed56cf42ac645cf9e34f99548f0c1db6c2ee10',None),
'generation_config.json':(156,'2fde3a7918391229df4dc54a3c57b7bed4fd332a',None),
'model-00001-of-00002.safetensors':(4961251752,'2db68b512c4c1467ce217bfb1216335d87c256e4','6187566ec8d00990103cd9243b306aceaa352697abfa3bb6fcbc5ce7a64ce142'),
'model-00002-of-00002.safetensors':(3639026128,'88f46a314ece2a62e34d0374385adc99bbecec0f','40c6a896b2275866e8bf56abc1da85f37a0d1d8fee824ad3008ea3231f4a3b70'),
'model.safetensors.index.json':(90594,'63f4efcf7f2f61a304bde053a8c55b8a59eb7355',None),
'preprocessor_config.json':(570,'cbd4f0cd77e39566f11921f9995bb3a4d008a83e',None),
'processor_config.json':(70,'453c7966d4b5d0b4a317c585989f64c58c2a6bf0',None),
'special_tokens_map.json':(662,'1a6193244714d3d78be48666cb02cdbfac62ad86',None),
'tokenizer.json':(33384570,'8d5bbd14966010dddf92b58a87bb9f58cb53d7b6','7d4046bf0505a327dd5a0abbb427ecd4fc82f99c2ceaa170bc61ecde12809b0c'),
'tokenizer.model':(4689074,'14f810a829755bae3fafd6f97096dbd2eac556bd','1299c11d7cf632ef3b4e11937501358ada021bbdf7c47638d13c0ee982f2e79c'),
'tokenizer_config.json':(1155415,'ad0c34f8dd76c7138b35b3973f3b4a422d5136a9',None),
}

def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()

def git_blob_sha1(path):
    n=path.stat().st_size
    h=hashlib.sha1(); h.update(f'blob {n}\0'.encode())
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()

def verified_file(path, size, blob, lfs):
    if not path.is_file() or path.is_symlink() or path.stat().st_size != size:
        raise RuntimeError(f'expected file missing/type/size mismatch: {path.name}')
    actual_sha = sha256(path)
    if lfs is not None:
        if actual_sha != lfs:
            raise RuntimeError(f'Hub LFS SHA256 mismatch: {path.name}')
        # Hub blob_id hashes the Git LFS pointer, not the downloaded payload.
        pointer = f'version https://git-lfs.github.com/spec/v1\noid sha256:{lfs}\nsize {size}\n'.encode()
        actual_blob = hashlib.sha1(f'blob {len(pointer)}\0'.encode() + pointer).hexdigest()
    else:
        actual_blob = git_blob_sha1(path)
    if actual_blob != blob:
        raise RuntimeError(f'Hub Git blob/pointer mismatch: {path.name}')
    return {'size_bytes': size, 'sha256': actual_sha, 'hub_git_blob_sha1': blob,
            'hub_git_blob_scope': 'lfs_pointer' if lfs else 'file_content',
            'hub_lfs_sha256': lfs, 'verified': True}

def self_test():
    with tempfile.TemporaryDirectory() as directory:
        path = pathlib.Path(directory) / 'payload'
        path.write_bytes(b'payload')
        digest = sha256(path)
        pointer = f'version https://git-lfs.github.com/spec/v1\noid sha256:{digest}\nsize 7\n'.encode()
        blob = hashlib.sha1(f'blob {len(pointer)}\0'.encode() + pointer).hexdigest()
        assert verified_file(path, 7, blob, digest)['hub_git_blob_scope'] == 'lfs_pointer'
        assert verified_file(path, 7, git_blob_sha1(path), None)['hub_git_blob_scope'] == 'file_content'
        for bad_blob, bad_lfs in [('0' * 40, digest), (blob, '0' * 64)]:
            try:
                verified_file(path, 7, bad_blob, bad_lfs)
            except RuntimeError:
                pass
            else:
                raise AssertionError('Incorrect pin accepted.')
        path.write_bytes(b'PAYLOAD')
        try:
            verified_file(path, 7, blob, digest)
        except RuntimeError:
            pass
        else:
            raise AssertionError('Same-size corrupt payload accepted.')
    print('Passed: regular Git content, LFS payload/pointer distinction, bad pins and same-size corruption.')

def atomic_json(path, obj):
    raw=(json.dumps(obj,ensure_ascii=False,indent=2)+'\n').encode()
    fd,name=tempfile.mkstemp(prefix='.provenance-',dir=path.parent)
    with os.fdopen(fd,'wb') as f: f.write(raw); f.flush(); os.fsync(f.fileno())
    os.replace(name,path)
    return hashlib.sha256(raw).hexdigest()

def notify(message):
    if '--verify-only' in sys.argv:
        return  # Foreground repair reports directly; no duplicate completion callback.
    try: subprocess.run([CODEX,'queue','--thread',THREAD,'--message',message],timeout=30,check=False,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    except Exception: pass

if set(sys.argv[1:]) - {'--verify-only', '--self-test'}:
    raise SystemExit('Supported flags: --verify-only or --self-test')
if '--self-test' in sys.argv:
    self_test()
    raise SystemExit(0)

DEST.parent.mkdir(parents=True,exist_ok=True)
DEST.mkdir(parents=True,exist_ok=True)
(DEST/'.tmp').mkdir(exist_ok=True)
env=os.environ.copy()
# Keep existing HF_HOME/token lookup intact; relocate blob/Xet caches and temp files to the external drive.
env['HF_HUB_CACHE']=str(DEST/'.hub-cache')
env['HF_XET_CACHE']=str(DEST/'.xet-cache')
env['TMPDIR']=str(DEST/'.tmp')
cmd=[HF,'download',REPO,'--revision',REV,'--local-dir',str(DEST),'--max-workers','4']
try:
    download_exit = 0
    if '--verify-only' not in sys.argv:
        with open(DEST/'download.log','ab',buffering=0) as log:
            proc=subprocess.run(cmd,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=env,check=False)
        download_exit = proc.returncode
    if download_exit:
        tail=(DEST/'download.log').read_text(errors='replace')[-3000:]
        status={'status':'failed','repository':REPO,'revision':REV,'exit_code':download_exit,'log':str(DEST/'download.log'),'tail':tail}
        atomic_json(DEST/'download-status.json',status)
        notify(f'TranslateGemma download FAILED (exit {download_exit}); no verified marker written. See {DEST}/download-status.json. Do not use partial files.')
        sys.exit(download_exit)
    files={}
    for name,(size,blob,lfs) in EXPECTED.items():
        files[name] = verified_file(DEST / name, size, blob, lfs)
    provenance={
      'schema_version':1,'status':'verified','repo_id':REPO,'revision':REV,
      'license':'gemma','license_link':'https://ai.google.dev/gemma/docs/gemma_3_license',
      'gated':'manual','terms_acknowledged_by_task':False,
      'source':'Official Hugging Face model repository metadata pinned before download.',
      'verification_method':'LFS files: full content SHA256/size plus canonical pointer Git blob SHA1. Non-LFS files: content Git blob SHA1/size. All files retain content SHA256.',
      'verifier_sha256':sha256(pathlib.Path(__file__)),
      'base_architecture':'Gemma3ForConditionalGeneration','parameter_count':4300079472,
      'translation_template':{'source_lang_code':'ja','target_lang_code':'en','supported_in_pinned_template':True},
      'total_bytes':sum(x[0] for x in EXPECTED.values()),'files':files,
      'intended_use':'isolated local challenger staging; no inference or training performed'
    }
    prov_sha=atomic_json(DEST/'provenance.json',provenance)
    marker=f'provenance_sha256={prov_sha}\nrepo_revision={REV}\nfiles_verified={len(files)}\n'
    (DEST/'.verified.tmp').write_text(marker)
    os.replace(DEST/'.verified.tmp',DEST/'.verified')
    status={'status':'verified','repository':REPO,'revision':REV,'provenance_sha256':prov_sha,'files_verified':len(files),'total_bytes':provenance['total_bytes'],'path':str(DEST)}
    atomic_json(DEST/'download-status.json',status)
    print(json.dumps(status))
    notify(f'TranslateGemma model VERIFIED at {DEST}; revision {REV}; {len(files)} files, {provenance["total_bytes"]} bytes; provenance SHA256 {prov_sha}. Ja→en stock template supported. No model was loaded.')
except BaseException as e:
    status={'status':'verification_failed','repository':REPO,'revision':REV,'error':f'{type(e).__name__}: {e}','log':str(DEST/'download.log')}
    atomic_json(DEST/'download-status.json',status)
    notify(f'TranslateGemma post-download verification FAILED: {type(e).__name__}: {e}. No verified marker written; inspect {DEST}/download-status.json.')
    raise
