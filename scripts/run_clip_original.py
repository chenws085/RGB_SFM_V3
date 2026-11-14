#!/usr/bin/env python3
import os,json,time,subprocess,datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];os.chdir(ROOT)
root=ROOT/'runs/train'/('clip_original_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ'));root.mkdir()
(ROOT/'overnight_logs/clip_direct/original_run.txt').write_text(str(root)+'\n')
def state(**values):
 p=root/'pipeline_status.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(dict(pid=os.getpid(),**values),indent=2));tmp.replace(p)
try:
 depth=ROOT/'runs/train/clip_depth_ablation_1103/pipeline_status.json'
 while depth.exists():
  s=json.loads(depth.read_text())
  if s['status']=='done':break
  if s['status']=='failed':raise RuntimeError('Inspect failed depth experiment before scheduling')
  p=Path(f"/proc/{s['pid']}/cmdline")
  if not p.exists() or b'run_clip_depth_ablation.py' not in p.read_bytes():raise RuntimeError('Depth process missing')
  state(status='waiting_for_live_depth_pipeline',waiting_pid=s['pid']);time.sleep(10)
 for name in ['CE','CLIP']:
  state(status='training',branch=name)
  cmd=[str(ROOT/'.venv/bin/python'),'-u','scripts/train_object_scale.py','--output',str(root/name),'--epochs','100','--per-class','100000','--valid-per-class','100000','--batch-size','16','--original-images','--clip-stages','2','3','--pred-weight','0','--region-weight','0','--device','cuda:1']
  if name=='CLIP':cmd+=['--clip-cache',str(ROOT/'datasets/clip_bbox_vitb32_1101.pt'),'--clip-weight','.2']
  with (root/(name+'.log')).open('w') as log:subprocess.run(cmd,check=True,stdout=log,stderr=subprocess.STDOUT)
 state(status='reporting')
 with (root/'report.log').open('w') as log:subprocess.run([str(ROOT/'.venv/bin/python'),'scripts/report_clip_original.py',str(root)],check=True,stdout=log,stderr=subprocess.STDOUT)
 state(status='done',report=str(root/'REPORT.md'))
except BaseException as exc:state(status='failed',error=str(exc));raise
