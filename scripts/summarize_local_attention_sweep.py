#!/usr/bin/env python3
import sys,json,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
run=Path(sys.argv[1]);state=json.loads((run/'status.json').read_text())
base=ROOT/'plots/reports/attention_translation_letterbox160_20261109'
a=np.load(base/'paired.npz');ids=a['source_indices'];boot=np.random.default_rng(42).integers(len(ids),size=(2000,len(ids)));rows=[]
def score(name,path,prefix):
 data=np.load(path/'paired.npz');assert np.array_equal(data['source_indices'],ids)
 def mean(s,shifts):return np.mean([data[k] for k in data.files if k.startswith(f'{prefix}_{s}_') and int(k.split('_')[2]) in shifts],axis=0)
 original=data[f'{prefix}_0_0_original'][:,3];aligned=np.mean([mean(s,[px])[:,0] for s,px in enumerate([8,16,32])],axis=0);unaligned=mean(0,list(range(1,8)));moved=mean(0,[32]);values={'original_accuracy':original,'one_token_agreement':aligned,'unaligned_accuracy':unaligned[:,3],'unaligned_consistency':unaligned[:,2],'shift32_accuracy':moved[:,3],'shift32_consistency':moved[:,2]}
 rows.append({'name':name,'path':str(path),'values':{k:float(v.mean()) for k,v in values.items()},'per_source':values})
score('global',base,'global');score('window7',base,'window')
for item in state['variants']:
 if item['status']=='complete':score(item['name'],Path(item['evaluation']),'window')
ref=rows[0]['per_source']
for row in rows:
 row['vs_global']={k:{'delta':float((v-ref[k]).mean()),'ci95':np.quantile((v-ref[k])[boot].mean(1),[.025,.975]).tolist()} for k,v in row.pop('per_source').items()}
(run/'comparison.json').write_text(json.dumps(rows,indent=2))
lines=['# letterbox160局部attention變體篩選',f"Queue status: {state['status']}; completed new variants: {sum(v['status']=='complete' for v in state['variants'])}/5",'相同seed42、6907/1770 split、100epoch上限及patience20；沿用global/window7，其餘從頭訓練。無新增平移增強。','|Variant|原位accuracy|one-token群一致平均|1–7px accuracy|32px accuracy|32px分類一致|','|---|---|---|---|---|---|']
for r in rows:
 v=r['values'];lines.append('|'+r['name']+'|'+ '|'.join(f'{v[k]:.2%}' for k in ['original_accuracy','one_token_agreement','unaligned_accuracy','shift32_accuracy','shift32_consistency'])+'|')
lines+=['','單seed、多變體探索，validation曾選epoch與比較變體，未校正多重比較；CI只反映來源抽樣，不含training/分群seed變異。不能僅以原位accuracy最高決定attention。候選應同時保住原位分類、局部表示及平移分類；one-token平均混合Stage0/1/2的8/16/32px，不是純深度效果。各組逐stage/方向、cosine、CI與airplanes見各自REPORT.md及summary.csv；配對來源CI見comparison.json。','offset3是固定網格原點偏移，不是完整Swin交替移窗；sliding為重疊中心鄰域，無循環連接。Stage為0-based。']
(run/'REPORT.md').write_text('\n'.join(lines))
