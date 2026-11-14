#!/usr/bin/env python3
"""Wait for the live training queue, then validate and summarize its saved results."""
import json,os,subprocess,time,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];os.chdir(ROOT)
root=Path(Path('overnight_logs/object_scale_training_1031/full_run.txt').read_text().strip())
state=root/'report_status.json'
def save(**values):state.write_text(json.dumps(dict(pid=os.getpid(),**values),indent=2))
try:
 while True:
  queue=json.loads((root/'queue_status.json').read_text())
  if queue['status']=='done':break
  if queue['status'] in ['failed','interrupted']:raise RuntimeError('Queue terminated: '+str(queue))
  proc=Path(f"/proc/{queue['pid']}/cmdline")
  try:command=proc.read_bytes()
  except FileNotFoundError:raise RuntimeError('Queue process disappeared without terminal status')
  if b'run_object_scale_full.py' not in command:raise RuntimeError('Queue PID identity changed')
  save(status='waiting_for_live_queue',queue_pid=queue['pid']);time.sleep(10)
 save(status='reporting')
 with (root/'report.log').open('w') as log:
  subprocess.run([str(ROOT/'.venv/bin/python'),'scripts/report_object_scale_full.py',str(root)],stdout=log,stderr=subprocess.STDOUT,check=True)
 save(status='done',report=str(root/'REPORT.md'))
except BaseException as exc:
 save(status='failed',error=str(exc));raise
