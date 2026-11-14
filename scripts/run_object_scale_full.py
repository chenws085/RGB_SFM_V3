#!/usr/bin/env python3
"""Run matched full-source A/B/hybrid training sequentially on GPU 1."""
import datetime,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.chdir(ROOT)
output=ROOT/'runs/train'/('object_scale_full_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
output.mkdir()
(Path('overnight_logs/object_scale_training_1031/full_run.txt')).write_text(str(output)+'\n')
def status(**values):
 p=output/'queue_status.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(dict(pid=os.getpid(),**values),indent=2));tmp.replace(p)
# Only wait for the specifically identified existing pilot; do not infer liveness from status files.
pilot=Path('runs/train/object_scale_1031_hybrid_pilot/status.json')
if pilot.exists():
 prior=json.loads(pilot.read_text());pid=prior.get('pid');proc=Path(f'/proc/{pid}/cmdline')
 while proc.exists():
  try:command=proc.read_bytes()
  except FileNotFoundError:break
  if b'train_object_scale.py' not in command or b'object_scale_1031_hybrid_pilot' not in command:break
  status(status='waiting_for_live_pilot',waiting_pid=pid);time.sleep(5)
 if json.loads(pilot.read_text())['status']!='done':
  status(status='failed',reason='Prior hybrid pilot did not finish successfully');sys.exit(1)
commands=[('A',['--pred-weight','0','--region-weight','0']),('B',[]),('H',['--hybrid-merging'])]
try:
 for branch,extra in commands:
  command=[str(ROOT/'.venv/bin/python'),'-u','scripts/train_object_scale.py','--output',str(output/branch),
    '--epochs','100','--per-class','100000','--valid-per-class','100000','--batch-size','16','--device','cuda:1','--seed','42']+extra
  status(status='running',branch=branch,command=command)
  with (output/(branch+'.log')).open('w') as log:
   subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
  status(status='evaluating',branch=branch)
  with (output/(branch+'_evaluation.log')).open('w') as log:
   subprocess.run([str(ROOT/'.venv/bin/python'),'scripts/evaluate_object_scale.py',str(output/branch),'--device','cuda:1'],stdout=log,stderr=subprocess.STDOUT,check=True)
 status(status='done')
except BaseException as exc:
 status(status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',error=str(exc));raise
