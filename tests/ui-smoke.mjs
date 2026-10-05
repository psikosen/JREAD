import { chromium } from '@playwright/test';
import { readFile, mkdir, writeFile } from 'node:fs/promises';
import assert from 'node:assert/strict';
const token = (await readFile(`${process.env.HOME}/.local/share/JREAD/token`,'utf8')).trim();
const browser = await chromium.launch({executablePath:'/usr/bin/brave-browser',headless:true});
const page = await browser.newPage({viewport:{width:1440,height:1050},reducedMotion:'reduce'});
let createdId, initialSettings; const testTitle='UI regression fixture '+Date.now();
const report={started_at:new Date().toISOString(),browser:'Disposable Brave',turbo_controls:false,settings_restored:false,passed:false};
const errors=[];page.on('pageerror',error=>errors.push(error.message));
async function restoreSettings() {
 const response=await fetch('http://127.0.0.1:8765/api/settings',{method:'PATCH',headers:{Authorization:`Bearer ${token}`,'Content-Type':'application/json'},body:JSON.stringify(initialSettings)});
 assert.equal(response.status,200,'Initial settings must be restored');
 assert.deepEqual(await response.json(),initialSettings);
}
async function selectSetting(id,value) {
 const changed=page.waitForResponse(response=>response.url().endsWith('/api/settings')&&response.request().method()==='PATCH');
 await page.locator('#'+id).selectOption(value);
 assert.equal((await changed).status(),200);
 await page.reload();
 await page.getByText('Ready',{exact:true}).first().waitFor();
 assert.equal(await page.locator('#'+id).inputValue(),value);
}
try {
 await page.goto(`http://127.0.0.1:8765/#token=${token}`);
 await page.getByText('Ready',{exact:true}).first().waitFor();
 const settings=initialSettings=await (await fetch('http://127.0.0.1:8765/api/settings',{headers:{Authorization:`Bearer ${token}`}})).json();
 report.initial_settings=initialSettings;
 if(settings.translator==='whisper') await selectSetting('translator-select','gemma4');
 await selectSetting('model-select','large-v3-turbo');
 assert.match(await page.locator('#engine-model').textContent(),/Large v3 Turbo.*Gemma 4/);
 assert.equal(await page.locator('#translator-select option[value="whisper"]').evaluate(option=>option.disabled),true);
 assert.equal(await page.locator('#device-select').inputValue(),settings.device);
 const persisted=await (await fetch('http://127.0.0.1:8765/api/settings',{headers:{Authorization:`Bearer ${token}`}})).json();
 assert.equal(persisted.model,'large-v3-turbo');
 assert.equal(persisted.translator,settings.translator==='whisper'?'gemma4':settings.translator);
 await selectSetting('model-select','medium');
 await selectSetting('translator-select','whisper');
 assert.equal(await page.locator('#model-select option[value="large-v3-turbo"]').evaluate(option=>option.disabled),true);
 report.turbo_controls=true;
 await restoreSettings();
 await page.reload();
 await page.getByText('Ready',{exact:true}).first().waitFor();
 const captionSettings={caption_glossary:settings.caption_glossary||'none',caption_context:Boolean(settings.caption_context),caption_boundaries:Boolean(settings.caption_boundaries)};
 for(const [id,key,value] of [['glossary-select','caption_glossary',captionSettings.caption_glossary==='none'?'aikatsu_pripara':'none'],['caption-context','caption_context',!captionSettings.caption_context],['caption-boundaries','caption_boundaries',!captionSettings.caption_boundaries]]) {
  const changed=page.waitForResponse(response=>response.url().endsWith('/api/settings')&&response.request().method()==='PATCH');
  if(typeof value==='boolean') await page.locator('#'+id).setChecked(value); else await page.locator('#'+id).selectOption(value);
  assert.equal((await changed).status(),200);
  await page.reload();
  await page.getByText('Ready',{exact:true}).first().waitFor();
  assert.equal(typeof value==='boolean'?await page.locator('#'+id).isChecked():await page.locator('#'+id).inputValue(),value);
 }
 assert.equal(await page.locator('#translator-select').inputValue(),settings.translator);
 if(settings.translator==='gemma4') assert.match(await page.locator('#engine-model').textContent(),/Gemma 4 E4B/);
 if(settings.translator==='gemma4-12b') assert.match(await page.locator('#engine-model').textContent(),/Gemma 4 12B/);
 assert.equal(new URL(page.url()).hash,'');
 assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false);
 await page.locator('#subs-tab').click();
 await page.locator('#file-input').setInputFiles({name:'ui-check.srt',mimeType:'text/plain',buffer:Buffer.from('1\n00:00:01,125 --> 00:00:03,250\nUI regression original\n')});
 await page.locator('#title-input').fill(testTitle);
 const imported=page.waitForResponse(response=>response.url().endsWith('/api/import') && response.request().method()==='POST');
 await page.locator('#generate-button').click();
 createdId=(await (await imported).json()).id;
 await page.locator('#detail-title').filter({hasText:testTitle}).waitFor();
 await page.locator('.cue-text').fill('Edited <script>alert(1)</script>');
 await page.getByRole('button',{name:'Save edit',exact:true}).click();
 await page.getByRole('button',{name:'Undo',exact:true}).waitFor({state:'visible'});
 await page.waitForFunction(()=>!document.querySelector('.cue-actions button:nth-child(2)').disabled);
 assert.equal(await page.locator('.cue-text').inputValue(),'Edited <script>alert(1)</script>');
 await page.getByRole('button',{name:'Undo',exact:true}).click();
 await page.waitForFunction(()=>document.querySelector('.cue-text').value==='UI regression original');
 const download = page.waitForEvent('download');await page.locator('#export-vtt').click();
 assert.match((await download).suggestedFilename(),/\.vtt$/);
 await page.locator('#pair-button').click();
 await page.locator('#pair-code').waitFor({state:'visible'});
 assert.match(await page.locator('#pair-value').textContent(),/^[A-F0-9]{8}$/);
 await page.locator('#close-detail').click();
 const card=page.locator('.job-card').filter({hasText:testTitle});
 await card.getByRole('button',{name:'Delete',exact:true}).click();
 await page.locator('#confirm-delete').click();
 await card.waitFor({state:'detached'});
 const demo=(await (await fetch('http://127.0.0.1:8765/api/jobs',{headers:{Authorization:`Bearer ${token}`}})).json()).find(job=>job.has_media && job.status==='complete' && job.title.startsWith('Japanese demo'));
 assert.ok(demo,'Run the real inference fixture first so native playback can be checked.');
 const demoDetail=await (await fetch(`http://127.0.0.1:8765/api/jobs/${demo.id}`,{headers:{Authorization:`Bearer ${token}`}})).json();
 const playbackCue=demoDetail.cues.find(cue=>cue.end-cue.start>.5);
 assert.ok(playbackCue,'The fixture needs a playable subtitle cue.');
 const cueTime=playbackCue.start+.1;
 await page.locator('#search-input').fill(demo.title);
 const demoCard=page.locator('.job-card').filter({hasText:demo.title}).first();
 for(let reopen=0;reopen<2;reopen++) {
  await demoCard.getByRole('button',{name:'Open player',exact:true}).click();
  await page.waitForFunction(()=>document.querySelector('#video').readyState>=2);
  await page.locator('#video').evaluate(async (video,cueTime)=>{video.muted=true;video.currentTime=cueTime;await video.play();},cueTime);
  await page.waitForFunction(cueTime=>document.querySelector('#video').currentTime>cueTime+.1,cueTime);
  assert.equal(await page.locator('#video').evaluate(video=>video.error),null);
  assert.ok(await page.locator('#video').evaluate(video=>Array.from(video.textTracks).some(track=>track.activeCues?.length)));
  await page.locator('#close-detail').click();
 }
 await page.locator('#search-input').fill('');
 await page.evaluate(()=>window.scrollTo(0,0));
 await mkdir('test-results',{recursive:true});
 await page.screenshot({path:'test-results/companion-desktop.png',fullPage:true});
 await page.setViewportSize({width:390,height:844});
 assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false);
 await page.screenshot({path:'test-results/companion-mobile.png',fullPage:true});
 assert.deepEqual(errors,[]);
 report.passed=true;
 console.log('PASS: Turbo/Gemma selection and reload, incompatible options disabled, import, edit, undo, export, pairing, deletion, native playback/captions and reopen, responsive layout, no JavaScript errors');
} finally {
 try {
  if(initialSettings) { await restoreSettings(); report.settings_restored=true; }
 } finally {
  try {
   if(createdId)await fetch(`http://127.0.0.1:8765/api/jobs/${createdId}`,{method:'DELETE',headers:{Authorization:`Bearer ${token}`}});
  } finally {
   await browser.close();
   await mkdir('test-results',{recursive:true});
   await writeFile('test-results/ui-smoke.json',JSON.stringify({...report,completed_at:new Date().toISOString(),page_errors:errors},null,2)+'\n');
  }
 }
}
