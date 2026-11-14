#!/usr/bin/env python3
"""Snapshot current depth experiment or wait for its live pipeline and hand off final report."""
import sys,json,http.client,datetime,hashlib,time,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];os.chdir(ROOT);sys.path.insert(0,str(ROOT/'scripts'))
from artifact_transfer_client import Client
run=ROOT/'runs/train/clip_depth_ablation_1103'
if '--wait' in sys.argv:
 while True:
  state=json.loads((run/'pipeline_status.json').read_text())
  if state['status']=='done':break
  if state['status']=='failed':raise RuntimeError(str(state))
  proc=Path(f"/proc/{state['pid']}/cmdline")
  if not proc.exists() or b'run_clip_depth_ablation.py' not in proc.read_bytes():raise RuntimeError('Pipeline process missing; no completion claim')
  time.sleep(10)
 c=json.loads((run/'results.json').read_text())
 with (ROOT/'plan/1103_clip_depth_ablation.md').open('a') as f:
  f.write('\n## 完整消融完成\n\n三個新增組100epochs與14條件評估、held-out語意代理及配對統計已完成。最終報告 runs/train/clip_depth_ablation_1103/REPORT.md。\n'+c['verdict']['depth_comparison']+'\n')
c=Client('https://100.104.52.34:40115',Path('/root/.config/rgb-sfm-transfer/token'),Path('/root/.config/rgb-sfm-transfer/server.crt'))
def conn():
 x=http.client.HTTPSConnection('127.0.0.1',41081,context=c.context,timeout=30);x.set_tunnel(c.host,c.port);return x
c.connection=conn
stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ');root=ROOT/'overnight_logs/clip_depth'/stamp;root.mkdir(parents=True)
listing=c.request('GET','/files');local=sorted(e['path'] for e in listing if e['path'].startswith('notes/local/'))
for name in ['notes/README.md']+local[-1:]:c.download(name,root/Path(name).name)
names=['plan/1103_clip_depth_ablation.md','scripts/train_object_scale.py','scripts/run_clip_depth_ablation.py','scripts/report_clip_depth.py','scripts/probe_clip_stage_semantics.py']
for f in ['REPORT.md','results.json','stage_semantic_probe.json','pipeline_status.json']:
 if (run/f).exists():names.append(str((run/f).relative_to(ROOT)))
entries=[]
for name in names:
 p=ROOT/name;q=root/name;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(p.read_bytes());remote=f'notes/remote140165/documents/{stamp}/{name}';c.upload(q,remote);entries.append(dict(path=name,remote=remote,sha256=hashlib.sha256(q.read_bytes()).hexdigest()))
m=root/'manifest.json';m.write_text(json.dumps(dict(files=entries,handoff='E: inbox delivery is not local project import. Local root unverified. Verify SHA256 and preserve differing versions.'),indent=2));c.upload(m,f'notes/remote140165/documents/{stamp}/manifest.json')
state=json.loads((run/'pipeline_status.json').read_text());n=root/(stamp+'_clip_depth_'+state['status']+'.md');n.write_text(f'author: remote140165\nutc: {stamp}\ntask: deep-only-clip-ablation\nstatus: '+('done' if state['status']=='done' else 'working')+f'\n\n遠端run {run}；版本與SHA256：notes/remote140165/documents/{stamp}/manifest.json。本機project root未確認；E inbox交付不表示匯入。\n');c.upload(n,'notes/remote140165/'+n.name)
print('Verified versioned E inbox delivery',stamp,'; local project import unverified',flush=True)
