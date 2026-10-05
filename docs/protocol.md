# JREAD local API v1

Backend binds only http://127.0.0.1:8765. JSON errors use `detail`. Every `/api` request requires authentication except POST /api/pair. Extension requests use `Authorization: Bearer TOKEN`; same-origin companion requests may also use the session cookie. Desktop launcher opens `/#token=TOKEN`; UI removes fragment immediately and stores token in sessionStorage. POST /api/session (bearer token, empty body) establishes a same-origin HttpOnly cookie for media playback. Session token never goes into page DOM. Local web UI is served at `/` from companion/. Extension needs user pairing.

## Companion
- GET /api/status => {version, engine:{model,device,ready,downloading,error}, active_job, live, disk_free_bytes}
- GET /api/settings => {model:"small",device:"cpu",font_size:28,show_japanese:false,caption_background:true,caption_position:10}
- PATCH /api/settings JSON matching fields, bounded server validation.
- POST /api/models/download {model:"small"|"medium"} => status; explicit download only. Poll status for readiness.
- POST /api/models/remove {model} => result; reject busy engines.
- GET /api/jobs => array of Job (most recent first)
- POST /api/jobs multipart file + optional language (ja default), title => Job, starts recognition/English translation. Upload cap 2GB; no arbitrary filesystem path or remote fetch endpoint.
- POST /api/import multipart file (SRT/VTT) + optional title => complete Job.
- GET /api/jobs/{id} => Job with cues
- POST /api/jobs/{id}/cancel => Job
- POST /api/jobs/{id}/retry => Job (resume)
- DELETE /api/jobs/{id} => {ok:true}
- GET /api/jobs/{id}/media => uploaded media, Range supported (cookie auth for native video).
- GET /api/jobs/{id}/export?format=srt|vtt => subtitle file
- PATCH /api/jobs/{id}/cues/{cue_id} {text,start,end} => Cue, saves undo revision
- POST /api/jobs/{id}/cues/{cue_id}/undo => Cue
- POST /api/pair/new => {code,expires_in:300}; explicit companion button displays code
- POST /api/pair {code} => {token,expires_in:86400}; no auth but exact allowed extension Origin, short-lived code, attempt limit. Extension stores token in chrome.storage.session.

Job: {id,title,status:"uploading"|"queued"|"preparing"|"recognizing"|"translating"|"complete"|"failed"|"cancelled"|"live",progress:0..1,created_at,language,duration,error,has_media,model,device,cue_count,cancellation_pending?:boolean,cues?:Cue[]}. Times are seconds; created_at is Unix time. Cancel during inference is cooperative; retry/delete wait until cancellation_pending clears. Uploaded media becomes resumable only after its upload and validation finish.
Cue: {id,start,end,text,source_text,status:"final"|"user_edited"|"needs_review"}, seconds media timeline, 0<=start<end. Caption text is untrusted plain text. Render using textContent/VTTCue only.

Generated English is wrapped at 42 characters and split into at most two lines per cue. Long-segment internal boundaries are proportional estimates, flagged needs_review. The original Japanese segment may be repeated across its English splits; this is not word alignment. Edited/imported cues retain the user's layout.

## Extension
MV3 TypeScript compiled with esbuild (root provides npm build tool). Optional site host permissions, activeTab/scripting/tabCapture/offscreen/storage/nativeMessaging. Never pass companion credentials to content scripts. All network calls in extension-owned context; validate sender tab/frame. No arbitrary remote fetch proxy or shell API. Native host name `com.jread.companion`, request `{action:"start"}` => `{base_url:"http://127.0.0.1:8765",version:1}`; host starts companion and opens local UI. Pairing still requires manual code from companion. JREAD claims native video tracks on accessible HTML5 video and permitted frames; report unsupported frames honestly. Imported/generated jobs explicitly chosen and applied to selected video; duration mismatch prompts confirmation.

## Live capture
WebSocket ws://127.0.0.1:8765/api/live. Exact extension Origin required. First JSON frame within 5 sec: {type:"hello",token,session_id,media_id,epoch:0,language:"ja",title}. Server => {type:"ready",job_id}. Only one session, no audio retained to disk.
Each audio chunk: JSON {type:"audio",epoch,seq,media_time,sample_rate:16000}, then binary PCM16 little-endian mono <=30 seconds. media_time is media time of first sample. Suggested 8-second chunks, 1-second overlap; account for playbackRate in packet `rate` (default 1, 0.25..4). Do not capture samples while video paused. Server emits zero or more {type:"cue",epoch,cue:Cue}, then {type:"ack",seq} after inference. Keep at most 3 chunks waiting locally; stop with clear backpressure error instead of dropping speech. On seek/rate change: {type:"reset",epoch:new}; discard old buffered audio and stale results. A replaced media source requires a fresh selection. Server may send {type:"error",detail}.

On stop, flush remaining audio then send {type:"stop"}. The server drains queued inference, sends {type:"stopped",job_id}, persists completion, and closes. An unexpected disconnect marks the job incomplete and preserves saved cues. Clean up the browser audio graph and capture tracks on completion, failure, and tab close. Native source-timed cues support replay; late live cues display sequentially on a separate temporary track, labeled with delay. Clear that display queue on timeline reset.
