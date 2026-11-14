#!/usr/bin/env python3
"""Sliding k=1,2,3,4,5,6,7,9,11,15; fixed lr/stages; explicit CUDA preflight."""
import hashlib,json,os,subprocess,sys,traceback
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import attention_lr_grid as grid
SIZES=[1,2,3,4,5,6,7,9,11,15]
PYTHON=str(ROOT/'.venv/bin/python')
SOURCES=['scripts/sliding_size_sweep.py','scripts/attention_lr_grid.py','scripts/train_caltech_mergingvit.py','scripts/evaluate_attention_translation.py','scripts/report_attention_translation.py','models/WindowAttention.py','models/ObjectScaleViT.py','models/MergingViT.py','dataloader/Caltech101.py','dataloader/OriginalImageViews.py','dataloader/OriginalImagePairs.py']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def utc():return datetime.now(timezone.utc).isoformat()
def report(run):
 grid.METHODS=['global']+[f'sliding_window{k}' for k in SIZES];grid.summarize(run)
 lines=['','## Window size明確定義','','k代表每個query的k×k token鄰域，並非pixels。Stage0/1均使用同一k，Stage2/3維持global；不截斷k為stage網格大小，出界keys以mask排除。','| k | 每軸token offsets | 最多keys | Stage0名義寬度(px) | Stage1名義寬度(px) |','|---|---|---:|---:|---:|']
 for k in SIZES:
  before=(k-1)//2;after=k-1-before;lines.append(f'|{k}|[{ -before}, …, {after}]|{k*k}|{k*8}|{k*16}|')
 lines+=['','偶數k的中心在半格位置，採既有k4方向慣例，不另掃offset。k1只有自己，為局部注意力不作跨token交流的控制；不是整個模型完全沒有跨token操作（merging與深層global仍存在）。表中pixels是token stride乘k的名義跨度，不是有效RF。網格邊緣有效keys較少；例如Stage1 k15即使寬度大於14，角落仍不是全局attention。','RF分析依使用者要求放在實驗全部完成後：區分名義窗口範圍、attention/merging理論連接範圍與實測effective RF。先完成分類/平移結果，再以相同來源/位置/target比較ERF，記錄padding/邊界與gradient聚集尺度；不直接用k×stride推論完整RF。','甜蜜點候選以原位accuracy、1–7px_acc、8px_acc與兩組分類一致率的共同取捨判斷，附配對來源區間；不臨時混合權重選單一最高分。單seed、曾選模validation探索，不宣稱最終最優；多seed驗證未自動排程。lr掃描、位置編碼/stage/offset與低通濾波仍暫緩。']
 p=run/'REPORT.md';s=p.read_text().replace('只比較範圍4或7','比較範圍1/2/3/4/5/6/7/9/11/15').replace('固定窗口實際4/4或7/7，深層不再隨window_size改變。','本輪全部為sliding，深層不隨window_size改變。').replace('五種方法比較','Sliding尺寸比較').replace('# Window大小 × sliding與否（lr固定3e-4）','# Sliding window size細掃（lr固定3e-4）').replace('Sliding4每軸offset為[-1,0,1,2]；Sliding7為[-3,…,3]。','各k的精確offsets見下表。')
 p.write_text(s+'\n'+'\n'.join(lines))
def initialize(run):
 run.mkdir(parents=True,exist_ok=False);old=ROOT/'runs/train/attention_size_sliding_20261110_lr3e4';prior=json.loads((old/'status.json').read_text());reuse={c['method']:c for c in prior['cells'] if c['method'] in ['global','sliding_window4','sliding_window7']}
 state=dict(status='prepared',started_utc=utc(),lr_sweep='paused',source_sha256={q:sha(ROOT/q) for q in SOURCES},cells=[])
 for method in ['global']+[f'sliding_window{k}' for k in SIZES]:
  item=dict(name=method,method=method,lr=3e-4,status='pending',training=str(run/method),evaluation=str(run/(method+'_translation')),prefix='window')
  if method in reuse:
   c=reuse[method];tr=Path(c['training']);ev=Path(c['evaluation']);cfg=json.loads((tr/'run_config.json').read_text());assert cfg['lr']==3e-4 and cfg['seed']==42
   if method!='global':assert cfg['attention_mode']=='sliding' and cfg['attention_stages']==[0,1] and cfg['window_size']==int(method.removeprefix('sliding_window'))
   m=json.loads((ev/'manifest.json').read_text());assert m['status']=='complete';assert sha(tr/'MergingViT_best.pth')==m['checkpoints'][c['prefix']]['sha256'];assert all(sha(ev/p)==h for p,h in m['artifacts_sha256'].items())
   item.update(status='reused',training=str(tr),evaluation=str(ev),prefix=c['prefix'],checkpoint_sha256=sha(tr/'MergingViT_best.pth'))
  state['cells'].append(item)
 for q in SOURCES:
  dest=run/'source_snapshot'/q;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((ROOT/q).read_bytes())
 grid.write(run/'status.json',state);report(run)
def main():
 run=Path(sys.argv[1]).resolve()
 if not (run/'status.json').exists():initialize(run)
 if '--prepare-only' in sys.argv:return
 import torch
 state=json.loads((run/'status.json').read_text());assert all(sha(ROOT/p)==h for p,h in state['source_sha256'].items()),'Source changed; review before resuming'
 if not torch.cuda.is_available() or torch.cuda.device_count()<2:
  state.update(status='waiting_for_gpu',cuda_available=bool(torch.cuda.is_available()),device_count=torch.cuda.device_count(),last_check_utc=utc(),reason='GPU1 unavailable; no new-size training started');grid.write(run/'status.json',state);report(run);print(state['reason'],flush=True);return
 # Confirm a real allocation before entering the training queue.
 try:torch.zeros(1,device='cuda:1')
 except Exception:
  state.update(status='waiting_for_gpu',reason=traceback.format_exc());grid.write(run/'status.json',state);report(run);return
 state.update(status='running',pid=os.getpid());grid.write(run/'status.json',state)
 for item in state['cells']:
  if item['status'] in ['complete','reused']:continue
  if item['status']!='pending':raise RuntimeError('Review interrupted/failed item before resuming: '+item['name'])
  assert all(sha(ROOT/p)==h for p,h in state['source_sha256'].items()),'Source changed during sweep'
  try:
   k=int(item['method'].removeprefix('sliding_window'));item.update(status='training',started_utc=utc());grid.write(run/'status.json',state)
   with (run/(item['name']+'.log')).open('x') as log:
    subprocess.run([PYTHON,'scripts/train_caltech_mergingvit.py','--output-dir',item['training'],'--device','cuda:1','--epochs','100','--batch-size','32','--workers','8','--lr','0.0003','--patience','20','--seed','42','--content-size','160','--attention-mode','sliding','--window-size',str(k),'--attention-stages','0','1','--window-offset','0'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    item['status']='evaluating';grid.write(run/'status.json',state)
    subprocess.run([PYTHON,'scripts/evaluate_attention_translation.py',item['evaluation'],'--letterbox-run',item['training']],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    subprocess.run([PYTHON,'scripts/report_attention_translation.py',item['evaluation']],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
   item.update(status='complete',completed_utc=utc(),checkpoint_sha256=sha(Path(item['training'])/'MergingViT_best.pth'))
  except Exception:item.update(status='failed',error=traceback.format_exc())
  grid.write(run/'status.json',state);report(run)
 state.update(status='complete' if all(c['status'] in ['complete','reused'] for c in state['cells']) else 'completed_with_errors',finished_utc=utc());grid.write(run/'status.json',state);report(run)
if __name__=='__main__':main()
