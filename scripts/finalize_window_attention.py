#!/usr/bin/env python3
"""Wait only for the identified live trainer, then run both matched evaluations."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    run=Path(sys.argv[1]).resolve()
    def status(**kwargs):
        p=run/'pipeline_status.json';tmp=p.with_suffix('.tmp')
        tmp.write_text(json.dumps(dict(pid=os.getpid(),**kwargs),indent=2));tmp.replace(p)
    try:
        while True:
            state=json.loads((run/'local/status.json').read_text())
            if state['status']=='done':break
            if state['status'] not in ['running']:
                raise RuntimeError('Trainer terminal state: '+str(state))
            proc=Path(f"/proc/{state['pid']}/cmdline")
            if not proc.exists():raise RuntimeError('Identified trainer no longer exists')
            command=proc.read_bytes()
            if b'train_object_scale.py' not in command or str(run.relative_to(ROOT)).encode() not in command:
                raise RuntimeError('Trainer PID command does not match intended run')
            status(status='waiting_for_live_trainer',epoch=state['epoch'],trainer_pid=state['pid'])
            time.sleep(10)
        for name,cmd in [('classification',['scripts/evaluate_object_scale.py',str(run/'local'),'--device','cuda:1']),
                         ('spherical',['scripts/report_window_attention.py',str(run)])]:
            status(status='evaluating',phase=name)
            with (run/(name+'.log')).open('x') as f:
                subprocess.run([str(ROOT/'.venv/bin/python'),'-u',*cmd],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
        status(status='done',report=str(run/'spherical/REPORT.md'))
    except BaseException as exc:
        status(status='failed',error=type(exc).__name__+': '+str(exc))
        raise


if __name__=='__main__':
    main()
