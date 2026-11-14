#!/usr/bin/env python3
"""Finish full-original-image MHA ablation after its identified live trainer."""
import json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
 run=Path(sys.argv[1]).resolve()
 def status(**values):
  p=run/'pipeline_status.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(dict(pid=os.getpid(),**values),indent=2));tmp.replace(p)
 try:
  while True:
   s=json.loads((run/'window/status.json').read_text())
   if s['status']=='done':break
   if s['status']!='running':raise RuntimeError('Trainer terminal state: '+str(s))
   proc=Path(f"/proc/{s['pid']}/cmdline")
   if not proc.exists():raise RuntimeError('Identified trainer missing')
   command=proc.read_bytes()
   if b'train_object_scale.py' not in command or str(run.relative_to(ROOT)).encode() not in command:raise RuntimeError('Trainer PID command mismatch')
   status(status='waiting_for_live_trainer',epoch=s['epoch'],trainer_pid=s['pid']);time.sleep(10)
  status(status='evaluating_classification_and_spherical')
  with (run/'report.log').open('x') as log:
   subprocess.run([str(ROOT/'.venv/bin/python'),'-u','scripts/report_original_window_attention.py',str(run)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
  status(status='done',report=str(run/'spherical/REPORT.md'))
 except BaseException as exc:
  status(status='failed',error=type(exc).__name__+': '+str(exc));raise

if __name__=='__main__':main()
