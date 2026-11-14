#!/usr/bin/env python3
"""Serial GPU1 screening; reuse frozen global/window7, evaluate each new variant."""
import hashlib,json,os,subprocess,sys,traceback
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PYTHON=str(ROOT/'.venv/bin/python')
VARIANTS=[('window4',['--attention-mode','window','--window-size','4']),
          ('window14',['--attention-mode','window','--window-size','14']),
          ('window7_offset3',['--attention-mode','window','--window-size','7','--window-offset','3']),
          ('window7_stage0',['--attention-mode','window','--window-size','7','--attention-stages','0']),
          ('sliding7_stage01',['--attention-mode','sliding','--window-size','7','--attention-stages','0','1'])]
SOURCES=['scripts/run_local_attention_sweep.py','scripts/train_caltech_mergingvit.py','scripts/evaluate_attention_translation.py','scripts/report_attention_translation.py','scripts/summarize_local_attention_sweep.py','models/MergingViT.py','models/WindowAttention.py','models/ObjectScaleViT.py','dataloader/Caltech101.py','dataloader/OriginalImageViews.py','dataloader/OriginalImagePairs.py']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def utc():return datetime.now(timezone.utc).isoformat()
def write(p,value):
 temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2));temp.replace(p)
def main():
 run=Path(sys.argv[1]).resolve();run.mkdir(parents=True,exist_ok=False)
 state={'started_utc':utc(),'pid':os.getpid(),'status':'running','variants':[], 'source_sha256':{p:sha(ROOT/p) for p in SOURCES},'baselines':{p:sha(ROOT/p) for p in ['runs/train/mergingvit_caltech101_letterbox160/MergingViT_best.pth','runs/train/window_letterbox160_20261109/MergingViT_best.pth']}}
 for name,args in VARIANTS:state['variants'].append({'name':name,'args':args,'status':'pending','training':str(run/name),'evaluation':str(run/(name+'_translation'))})
 snap=run/'source_snapshot'
 for p in SOURCES:
  dest=snap/p;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((ROOT/p).read_bytes())
 write(run/'status.json',state)
 for item in state['variants']:
  try:
   assert all(sha(ROOT/p)==h for p,h in state['source_sha256'].items()),'Source changed during sweep'
   assert all(sha(ROOT/p)==h for p,h in state['baselines'].items()),'Baseline changed during sweep'
   item.update(status='training',started_utc=utc());write(run/'status.json',state)
   with (run/(item['name']+'.log')).open('x') as log:
    command=[PYTHON,'scripts/train_caltech_mergingvit.py','--output-dir',item['training'],'--device','cuda:1','--epochs','100','--batch-size','32','--workers','8','--lr','0.0003','--patience','20','--seed','42','--content-size','160']+item['args']
    subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    item['status']='evaluating';write(run/'status.json',state)
    subprocess.run([PYTHON,'scripts/evaluate_attention_translation.py',item['evaluation'],'--letterbox-run',item['training']],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    subprocess.run([PYTHON,'scripts/report_attention_translation.py',item['evaluation']],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
   item.update(status='complete',completed_utc=utc());write(run/'status.json',state)
   subprocess.run([PYTHON,'scripts/summarize_local_attention_sweep.py',str(run)],cwd=ROOT,check=True)
  except Exception:
   item.update(status='failed',error=traceback.format_exc());write(run/'status.json',state)
 state.update(status='complete' if all(v['status']=='complete' for v in state['variants']) else 'completed_with_errors',finished_utc=utc());write(run/'status.json',state)
 subprocess.run([PYTHON,'scripts/summarize_local_attention_sweep.py',str(run)],cwd=ROOT,check=True)
 # Versioned document/results handoff after the queue; upload failure does not relabel completed experiments.
 try:subprocess.run([sys.executable,'scripts/publish_local_attention_sweep.py',str(run)],cwd=ROOT,check=True)
 except Exception:write(run/'delivery_error.json',{'error':traceback.format_exc(),'utc':utc()})
if __name__=='__main__':main()
