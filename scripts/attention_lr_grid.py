#!/usr/bin/env python3
"""Fixed-stage 5-method comparison; LR sweep paused with resumable status and matrix reports."""
import hashlib,json,os,subprocess,sys,traceback
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1];PYTHON=str(ROOT/'.venv/bin/python')
METHODS=['global','window4','window7','sliding_window4','sliding_window7']
LRS=[3e-4]  # User paused the LR sweep; no other LR is queued.
SOURCES=['scripts/attention_lr_grid.py','scripts/train_caltech_mergingvit.py','scripts/evaluate_attention_translation.py','scripts/report_attention_translation.py','models/MergingViT.py','models/ObjectScaleViT.py','models/WindowAttention.py','dataloader/Caltech101.py','dataloader/OriginalImageViews.py','dataloader/OriginalImagePairs.py']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def utc():return datetime.now(timezone.utc).isoformat()
def write(p,d):
 tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2));tmp.replace(p)
def summarize(run):
 import numpy as np
 s=json.loads((run/'status.json').read_text());lines=['# Window大小 × sliding與否（lr固定3e-4）','主比較僅改window_size與sliding與否；lr固定3e-4，lr掃描暫緩；Absolute position保留，所有局部方法僅Stage0/1，Stage2/3全局，offset=0。','window_size以tokens為單位。Sliding4每軸offset為[-1,0,1,2]；Sliding7為[-3,…,3]。偶數鄰域具有半格方向偏置，不能當完全對稱置中。',f'狀態：{s["status"]}。— 表示尚無完成評估，不是0。','原位及平移使用同一模型best checkpoint；lr是AdamW初始值，皆搭配100epoch cosine、最低1e-6及patience20。']
 rows=[]
 for cell in s['cells']:
  if cell['status'] not in ['complete','reused']:continue
  a=np.load(Path(cell['evaluation'])/'paired.npz');prefix=cell['prefix']
  def values(stage,shifts):return np.mean([a[k] for k in a.files if k.startswith(f'{prefix}_{stage}_') and int(k.split('_')[2]) in shifts],axis=0)
  v=dict(small_consistency=float(values(0,list(range(1,8)))[:,2].mean()),shift8_accuracy=float(values(0,[8])[:,3].mean()),shift8_consistency=float(values(0,[8])[:,2].mean()),original_accuracy=float(a[f'{prefix}_0_0_original'][:,3].mean()),small_accuracy=float(values(0,list(range(1,8)))[:,3].mean()),shift32_accuracy=float(values(0,[32])[:,3].mean()),shift32_consistency=float(values(0,[32])[:,2].mean()),one_token_agreement=float(np.mean([values(st,[px])[:,0].mean() for st,px in enumerate([8,16,32])])),one_token_cosine=float(np.mean([values(st,[px])[:,1].mean() for st,px in enumerate([8,16,32])])))
  rows.append(dict(method=cell['method'],lr=cell['lr'],evaluation=cell['evaluation'],values=v))
 for lr in LRS:
  lines+=['',f'## 五種方法比較（lr={lr:.0e}）','','| 方法 | accuracy（原位） | 1–7px_acc | 8px_acc | 1–7px一致性 | 8px一致性 |','|---|---:|---:|---:|---:|---:|']
  for method in METHODS:
   found=[r for r in rows if r['lr']==lr and r['method']==method]
   vals=[f"{100*found[0]['values'][k]:.2f}%" if found else '—' for k in ['original_accuracy','small_accuracy','shift8_accuracy','small_consistency','shift8_consistency']]
   label={'window4':'window 4','window7':'window 7','sliding_window4':'sliding_window 4','sliding_window7':'sliding_window 7'}.get(method,method)
   lines.append('|'+label+'|'+'|'.join(vals)+'|')
 lines+=['','一致性＝平移後預測類別與同張原位影像預測相同的比例（分類一致率），不是K-means群一致率。Accuracy以真實類別為準。1–7px為7種位移量×四方向共28條件平均；8px為四方向平均，每張來源圖等權。原位僅計一次，平移均保留完整來源內容。所有數值均以百分比呈現。其他位移、token分群與cosine保留在各方法完整評估與results.json，不混入此主表。']
 lines+=['','## 固定設定與限制','','Caltech101 6907/1770 split、letterbox160完整自然原圖、seed42、相同augmentation/AdamW wd.01/CE smoothing.1/batch32。本輪以global為基準，局部方法只比較範圍4或7、互斥窗或sliding鄰域；lr固定3e-4。global沒有window_size。所有局部配置均為Stage0/1；固定窗口實際4/4或7/7，深層不再隨window_size改變。訓練預算及停止規則相同，實際停止epoch可能不同。','單seed、多方法使用validation篩選，不是獨立final test；本表先報點估計，正式選型須額外seed與保留評估。one-token平均混合Stage0/1/2=8/16/32px，不能當純深度效果。逐stage/方向、airplanes、CI與來源hash見各cell evaluation。','舊window4（Stage2也局部）、window14、offset3、stage0-only均保留在歷史報告，不混進本表。Absolute position/stage/offset暫不掃描；低通濾波只列未來規劃，不在此輪執行。']
 (run/'REPORT.md').write_text('\n'.join(lines));write(run/'results.json',rows)
def main():
 if sys.argv[1]=='report':summarize(Path(sys.argv[2]).resolve());return
 run=Path(sys.argv[1]).resolve();run.mkdir(parents=True,exist_ok=False)
 old=ROOT/'runs/train/local_attention_sweep_20261109T180400Z'
 reuse={'global':(ROOT/'runs/train/mergingvit_caltech101_letterbox160',ROOT/'plots/reports/attention_translation_letterbox160_20261109','global'), 'window7':(ROOT/'runs/train/window_letterbox160_20261109',ROOT/'plots/reports/attention_translation_letterbox160_20261109','window'),'sliding_window7':(old/'sliding7_stage01',old/'sliding7_stage01_translation','window')}
 state=dict(lr_sweep='paused_by_user; only lr=3e-4 runs are authorized in this queue',status='running',started_utc=utc(),pid=os.getpid(),source_sha256={p:sha(ROOT/p) for p in SOURCES},cells=[])
 for lr in LRS:
  for method in METHODS:
   name=f'{method}_lr{lr:.0e}';item=dict(name=name,method=method,lr=lr,status='pending',training=str(run/name),evaluation=str(run/(name+'_translation')),prefix='window')
   if lr==3e-4 and method in reuse:
    tr,ev,prefix=reuse[method];cfg=json.loads((tr/'run_config.json').read_text());assert cfg['lr']==lr and cfg['seed']==42 and cfg['epochs']==100 and cfg['batch_size']==32 and cfg['patience']==20
    mode='sliding' if method.startswith('sliding') else 'window' if method.startswith('window') else 'global';assert cfg.get('attention_mode','global')==mode
    if mode!='global':assert cfg['window_size']==7 and cfg.get('window_offset',0)==0
    if mode=='sliding':assert cfg['attention_stages']==[0,1]
    manifest=json.loads((ev/'manifest.json').read_text());assert manifest['status']=='complete';assert sha(tr/'MergingViT_best.pth')==manifest['checkpoints'][prefix]['sha256']
    assert all(sha(ev/p)==h for p,h in manifest['artifacts_sha256'].items())
    item.update(status='reused',training=str(tr),evaluation=str(ev),prefix=prefix,checkpoint_sha256=sha(tr/'MergingViT_best.pth'),note='Verified historical result; window7 stages2/3 are already global-equivalent')
   state['cells'].append(item)
 for p in SOURCES:
  q=run/'source_snapshot'/p;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes((ROOT/p).read_bytes())
 write(run/'status.json',state);summarize(run)
 for item in sorted(state['cells'],key=lambda c:0 if c['lr']==3e-4 else 1):
  if item['status']=='reused':continue
  try:
   assert all(sha(ROOT/p)==h for p,h in state['source_sha256'].items()),'Source changed during grid'
   item.update(status='training',started_utc=utc());write(run/'status.json',state)
   mode='sliding' if item['method'].startswith('sliding') else 'window' if item['method'].startswith('window') else 'global'
   ws=4 if item['method'].endswith('4') else 7
   args=['--attention-mode',mode,'--window-size',str(ws),'--window-offset','0']+(['--attention-stages','0','1'] if mode!='global' else [])
   with (run/(item['name']+'.log')).open('x') as log:
    subprocess.run([PYTHON,'scripts/train_caltech_mergingvit.py','--output-dir',item['training'],'--device','cuda:1','--epochs','100','--batch-size','32','--workers','8','--lr',str(item['lr']),'--patience','20','--seed','42','--content-size','160']+args,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    item['status']='evaluating';write(run/'status.json',state)
    subprocess.run([PYTHON,'scripts/evaluate_attention_translation.py',item['evaluation'],'--letterbox-run',item['training']],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    subprocess.run([PYTHON,'scripts/report_attention_translation.py',item['evaluation']],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
   item.update(status='complete',checkpoint_sha256=sha(Path(item['training'])/'MergingViT_best.pth'),completed_utc=utc())
  except Exception:item.update(status='failed',error=traceback.format_exc())
  write(run/'status.json',state);summarize(run)
 state.update(status='complete' if all(c['status'] in ['complete','reused'] for c in state['cells']) else 'completed_with_errors',finished_utc=utc());write(run/'status.json',state);summarize(run)
 try:subprocess.run([PYTHON,'scripts/publish_attention_lr_grid.py',str(run)],cwd=ROOT,check=True)
 except Exception:write(run/'delivery_error.json',{'error':traceback.format_exc()})
if __name__=='__main__':main()
