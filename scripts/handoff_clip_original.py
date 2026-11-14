#!/usr/bin/env python3
import sys,os,time,json,http.client,datetime,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];os.chdir(ROOT);sys.path.insert(0,str(ROOT/'scripts'))
from artifact_transfer_client import Client
run=Path(Path('overnight_logs/clip_direct/original_run.txt').read_text().strip())
if '--wait' in sys.argv:
 while True:
  s=json.loads((run/'pipeline_status.json').read_text())
  if s['status']=='done':break
  if s['status']=='failed':raise RuntimeError(str(s))
  p=Path(f"/proc/{s['pid']}/cmdline")
  if not p.exists() or b'run_clip_original.py' not in p.read_bytes():raise RuntimeError('Pipeline missing; no completion claim')
  time.sleep(10)
 with Path('plan/1103_clip_original_images.md').open('a') as f:f.write('\n## 完整結果已完成\n\n兩組原圖正式訓練、自然背景validation評估及配對統計已完成，最終報告：'+str(run/'REPORT.md')+'。結果依REPORT.md判讀，不以smoke推論正式效果。\n')
c=Client('https://100.104.52.34:40115',Path('/root/.config/rgb-sfm-transfer/token'),Path('/root/.config/rgb-sfm-transfer/server.crt'))
def conn():
 x=http.client.HTTPSConnection('127.0.0.1',41081,context=c.context,timeout=30);x.set_tunnel(c.host,c.port);return x
c.connection=conn;stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ');out=Path('overnight_logs/clip_original')/stamp;out.mkdir(parents=True)
l=c.request('GET','/files');last=sorted(e['path'] for e in l if e['path'].startswith('notes/local/'))[-1:]
for name in ['notes/README.md']+last:c.download(name,out/Path(name).name)
files=[Path(p) for p in ['plan/1103_clip_original_images.md','dataloader/ObjectScalePairs.py','dataloader/OriginalImagePairs.py','scripts/train_object_scale.py','scripts/evaluate_object_scale.py','scripts/run_clip_original.py','scripts/report_clip_original.py']]
for name in ['pipeline_status.json','REPORT.md','results.json','comparison.png']:
 if (run/name).exists():files.append((run/name).relative_to(ROOT))
entries=[]
for p in files:
 q=out/p;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(p.read_bytes());remote=f'notes/remote140165/documents/{stamp}/{p}';c.upload(q,remote);entries.append(dict(path=str(p),remote=remote,sha256=hashlib.sha256(q.read_bytes()).hexdigest()))
m=out/'manifest.json';m.write_text(json.dumps(dict(files=entries,handoff='E: inbox delivery is not local project import; local root unverified. Verify SHA256 and preserve differing versions.'),indent=2));c.upload(m,f'notes/remote140165/documents/{stamp}/manifest.json')
s=json.loads((run/'pipeline_status.json').read_text());p=out/(stamp+'_clip_original.md');p.write_text(f'author: remote140165\nutc: {stamp}\ntask: clip-original-images\nstatus: '+('done' if s['status']=='done' else 'info' if s['status']=='paused' else 'working')+f'\n\nRun {run}；版本／SHA256 notes/remote140165/documents/{stamp}/manifest.json。目前狀態見pipeline_status.json；若paused，佇列已停止且不會自動接續。E inbox交付未確認本機專案匯入。\n');c.upload(p,'notes/remote140165/'+p.name);print('Verified E inbox handoff; local project import unverified')
