#!/usr/bin/env python3
import os,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];os.chdir(ROOT)
root=ROOT/'runs/train/clip_depth_ablation_1103';root.mkdir(exist_ok=False)
def status(**values):
 p=root/'pipeline_status.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(dict(pid=os.getpid(),**values),indent=2));t.replace(p)
try:
 for name,stages,weight in [('deep',['2','3'],'.2'),('shallow',['0','1'],'.2'),('deep_half',['2','3'],'.1')]:
  status(status='training',branch=name)
  cmd=[str(ROOT/'.venv/bin/python'),'-u','scripts/train_object_scale.py','--output',str(root/name),'--epochs','100','--per-class','100000','--valid-per-class','100000','--batch-size','16','--clip-cache',str(ROOT/'datasets/clip_bbox_vitb32_1101.pt'),'--clip-stages',*stages,'--clip-weight',weight,'--pred-weight','0','--region-weight','0','--device','cuda:1']
  with (root/(name+'.log')).open('w') as f:subprocess.run(cmd,check=True,stdout=f,stderr=subprocess.STDOUT)
  status(status='evaluating',branch=name)
  with (root/(name+'_evaluation.log')).open('w') as f:subprocess.run([str(ROOT/'.venv/bin/python'),'scripts/evaluate_object_scale.py',str(root/name),'--device','cuda:1'],check=True,stdout=f,stderr=subprocess.STDOUT)
 status(status='reporting')
 with (root/'report.log').open('w') as f:subprocess.run([str(ROOT/'.venv/bin/python'),'scripts/report_clip_depth.py',str(root)],check=True,stdout=f,stderr=subprocess.STDOUT)
 status(status='done',report=str(root/'REPORT.md'))
except BaseException as exc:status(status='failed',error=str(exc));raise
