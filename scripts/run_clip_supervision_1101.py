#!/usr/bin/env python3
"""Verified cache -> smoke training -> full CLIP-supervised run -> scale evaluation."""
import os,json,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];os.chdir(ROOT)
root=ROOT/'runs/train/clip_supervision_1101';root.mkdir(exist_ok=False)
def status(**values):
 p=root/'pipeline_status.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(dict(pid=os.getpid(),**values),indent=2));t.replace(p)
try:
 status(status='waiting_for_teacher_cache')
 while True:
  live=False
  for p in Path('/proc').glob('[0-9]*/cmdline'):
   try:cmd=p.read_bytes().split(b'\0')
   except (FileNotFoundError,PermissionError):continue
   if b'python' in cmd[0] and b'scripts/cache_clip_bbox.py' in cmd:live=True
  if not live:break
  time.sleep(2)
 cache=ROOT/'datasets/clip_bbox_vitb32_1101.pt'
 if not cache.exists():raise RuntimeError('Teacher cache did not complete')
 for name,epochs,limit in [('smoke',1,2),('full',100,100000)]:
  status(status='training',branch=name)
  command=[str(ROOT/'.venv/bin/python'),'-u','scripts/train_object_scale.py','--output',str(root/name),
   '--epochs',str(epochs),'--per-class',str(limit),'--valid-per-class','100000' if name=='full' else '4',
   '--clip-cache',str(cache),'--clip-weight','.2','--pred-weight','0','--region-weight','0','--device','cuda:1']
  with (root/(name+'.log')).open('w') as log:subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT)
 status(status='evaluating')
 with (root/'evaluation.log').open('w') as log:
  subprocess.run([str(ROOT/'.venv/bin/python'),'scripts/evaluate_object_scale.py',str(root/'full'),'--device','cuda:1'],check=True,stdout=log,stderr=subprocess.STDOUT)
 status(status='done')
except BaseException as exc:
 status(status='failed',error=str(exc));raise
