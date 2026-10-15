#!/usr/bin/env python3
"""Fixed 1–7px translation protocol with true FAISS spherical K-means."""
import argparse
import json
from pathlib import Path
from datetime import datetime,timezone
import faiss
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import run_caltech_corrected_center as c
import test_caltech_translation_kmeans as l

ROOT=l.ROOT
OUT=ROOT/'plots/kmeans/caltech_non_aligned_1015'
CHECKPOINTS={'baseline':'runs/train/mergingvit_caltech101_letterbox160/MergingViT_best.pth',
             'aligned_8_16_32':'runs/train/mergingvit_caltech101_aligned_translation_1014/MergingViT_best.pth',
             'augment_1_7':'runs/train/mergingvit_caltech101_non_aligned_1015/MergingViT_best.pth'}
CLASSES=['airplanes','chandelier','brain']


def unit(x):
    x=np.ascontiguousarray(x,dtype=np.float32)
    if not np.isfinite(x).all() or np.any(np.linalg.norm(x,axis=1)<1e-10):
        raise ValueError('Invalid feature for spherical clustering')
    return x/np.linalg.norm(x,axis=1,keepdims=True)


def write_json(path,obj):
    path.write_text(json.dumps(obj,indent=2))


def report():
    available=[name for name in CHECKPOINTS if (OUT/name/'summary.csv').exists()]
    lines=['# 1–7 px 非對齊平移：分類與 spherical K-means','',
           '## 設定','',
           '160-in-224，固定 seed-42 split。訓練 6,907 張、101 類；評估固定 airplanes 160／chandelier 22／brain 20 張。全來源影像平移，沒有物件裁切或平移插值。',
           '測試原位及四方向 × 1–7 px；此輪增強模型從頭訓練，training 每次讀取均勻抽原位或四方向 × 1–7 px，共 29 種。沿用原有 flip／color jitter／erasing 與模型／優化設定。這是見過位移範圍的 robustness，不是 unseen-shift 泛化。',
           '每模型獨立以未平移全類別 training 的修正中心 token fitting 每 stage 共享 codebook。FAISS spherical=True；L2-normalized features；centroids 每次更新正規化；K=31、niter=100、nredo=50、seed=42。內積最高者為指派；margin=最高與次高 cosine 分數差。與舊 sklearn 歐式 margin 數值不可直接互比。',
           '每個模型在全部測試變換中固定 codebook。不同模型 cluster ID 不對應；前後皆錯亦可能 prediction 一致。',
           '1–7 px 不對齊初始 8 px patch，中心 token 涵蓋內容可能改變。Cosine 與群變動同時包含內容涵蓋與位置敏感性，不能當成相同 crop 的純漂移。Stage 3 是觀察 64 px 網格，不是平移 64 px。',
           '', '## 執行狀態','']
    for name in CHECKPOINTS:
        lines.append(f"- {name}: {'已評估完成' if name in available else '尚未完成評估'}")
    lines+=['','既有 validation 用於選模；僅一個 training seed，三類探索性比較，不是 untouched final test 或全類別結論。','']
    allrows={}
    for name in available:
        import csv
        rows=list(csv.DictReader((OUT/name/'summary.csv').open()));allrows[name]=rows
        lines += [f'## {name}','', '### 分類：四方向平均','', '| 位移 | 飛機 accuracy | 吊燈 accuracy | 腦部 accuracy | 三類合併 accuracy | 預測一致率（三類） |','|---|---:|---:|---:|---:|---:|']
        def pick(category,shift,stage=0):
            return next(r for r in rows if r['category']==category and int(r['shift_pixels'])==shift and int(r['stage'])==stage)
        for shift in range(8):
            a=pick('all_selected',shift)
            values=[float(pick(cat,shift)['accuracy']) for cat in CLASSES]+[float(a['accuracy']),float(a['prediction_consistency'])]
            lines.append(f"| {shift} px | "+' | '.join(f'{v:.2%}' for v in values)+' |')
        lines += ['', '### 內部表徵：三類合併、四方向平均','']
        for stage in range(4):
            lines += [f'#### Stage {stage}（網格 {8*2**stage} px）','', '| 位移 | 群一致率 | Token cosine | Spherical margin | 使用群數 |','|---|---:|---:|---:|---:|']
            for shift in range(8):
                r=pick('all_selected',shift,stage)
                lines.append(f"| {shift} px | {float(r['cluster_agreement']):.2%} | {float(r['token_cosine']):.4f} | {float(r['spherical_margin']):.4f} | {r['used_clusters']} |")
        lines += ['',f'![{name}]({name}/translation_stability.png)','']
    if len(available)>=2:
        lines+=['## 三類合併：1–7 px 整體描述','',
                '各位移、方向等權；同張圖片反覆使用，下列平均不是獨立樣本推論。','',
                '| 模型 | 原圖 accuracy | 平移後平均 accuracy | 平均預測一致率 |','|---|---:|---:|---:|']
        for name,rows in allrows.items():
            a=[r for r in rows if r['category']=='all_selected' and int(r['stage'])==0]
            orig=next(r for r in a if int(r['shift_pixels'])==0)
            moved=[r for r in a if int(r['shift_pixels'])>0]
            lines.append(f"| {name} | {float(orig['accuracy']):.2%} | {np.mean([float(r['accuracy']) for r in moved]):.2%} | {np.mean([float(r['prediction_consistency']) for r in moved]):.2%} |")
    if 'augment_1_7' in available and 'baseline' in available:
        lines += ['', '## 增強後的逐 stage 改善（1–7 px 平均）', '',
                  '| Stage | 原模型群一致率 | 1–7 增強群一致率 | 原模型 cosine | 1–7 增強 cosine |',
                  '|---|---:|---:|---:|---:|']
        for stage in range(4):
            values=[]
            for model in ['baseline','augment_1_7']:
                sub=[r for r in allrows[model] if r['category']=='all_selected' and int(r['stage'])==stage and int(r['shift_pixels'])>0]
                values.append((np.mean([float(r['cluster_agreement']) for r in sub]),np.mean([float(r['token_cosine']) for r in sub])))
            lines.append(f'| Stage {stage} | {values[0][0]:.2%} | {values[1][0]:.2%} | {values[0][1]:.4f} | {values[1][1]:.4f} |')
        def classification_means(model):
            sub=[r for r in allrows[model] if r['category']=='all_selected' and int(r['stage'])==0 and int(r['shift_pixels'])>0]
            return np.mean([float(r['accuracy']) for r in sub]),np.mean([float(r['prediction_consistency']) for r in sub])
        before=classification_means('baseline');after=classification_means('augment_1_7')
        lines += ['', f'相較原模型，1–7 px 增強後三類合併的平均平移 accuracy 改變 {(after[0]-before[0])*100:+.2f} 個百分點；平均預測一致率改變 {(after[1]-before[1])*100:+.2f} 個百分點。是否改善須搭配每類別與原圖 accuracy 檢查，不只看總平均。']
    if 'augment_1_7' in available:
        lines+=['','## 解讀','',
                '應同時閱讀原圖 accuracy、平移後 accuracy 與預測一致率，不能只因一致率提高就認定分類變好。逐 stage 的群一致率／cosine／margin 可判斷內部變動是否同步改善，但單一中心 token 不能代表整張圖片。群數與群熵另見 CSV，以排查只因集中到少數群而產生的表面一致性。',
                '與 aligned_8_16_32 的對照用以檢查小位移增強的增益；三模型皆重新使用相同規格 spherical codebooks，避免混入 sklearn／FAISS 的方法差異。未做內容對齊 ROI 或機制消融，不將結果宣稱為嚴格平移不變性。']
    lines+=['','## 重現與資料','',
            '執行：`bash scripts/run_non_aligned_translation_1015.sh`（正在執行時不要重複啟動）。',
            '每模型目錄保存 manifest、逐筆 paired_metrics.csv、summary.csv、分群中心與圖。訓練紀錄另見 `overnight_logs/non_aligned_translation_1015/`。']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',choices=list(CHECKPOINTS));p.add_argument('--device',default='cpu');p.add_argument('--workers',type=int,default=4);p.add_argument('--report-only',action='store_true');a=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    if a.report_only:report();return
    if not a.model:p.error('--model required')
    torch.set_num_threads(4);faiss.omp_set_num_threads(4)
    name=a.model;out=OUT/name;out.mkdir(exist_ok=True)
    path=ROOT/CHECKPOINTS[name]
    ckpt=torch.load(path,map_location='cpu',weights_only=False)
    model=l.MergingViT(**ckpt['model_args']).to(a.device).eval();model.load_state_dict(ckpt['model_weights'])
    base=l.Caltech101(str(ROOT/'data'),target_type=['category','annotation'],download=False)
    train,valid=l.split_indices(str(ROOT/'data'),42);valid=sorted(valid)
    ids=[i for i in valid if base.categories[base.y[i]] in CLASSES];assert len(ids)==202
    assert not set(train)&set(valid)
    config=dict(model=name,checkpoint_sha256=c.sha(path),runner_sha256=c.sha(__file__),extractor_sha256=c.sha(l.__file__),coordinate_sha256=c.sha(c.__file__),
                train_indices=train,valid_indices=valid,evaluation_indices=ids,classes=CLASSES,
                coordinate_version=c.COORDINATE_VERSION,content_max=160,shifts=list(range(8)),K=31,niter=100,nredo=50,seed=42,faiss=faiss.__version__,spherical=True,
                device=a.device,torch=torch.__version__)
    ident=out/'cache_identity.json'
    if ident.exists() and json.loads(ident.read_text())!=config:raise RuntimeError('Cache identity mismatch')
    write_json(ident,config)
    manifest=dict(config,status='running',started_utc=datetime.now(timezone.utc).isoformat());write_json(out/'run_manifest.json',manifest)
    def extract(tag,indices,dx=0,dy=0):
        dest=out/f'{tag}.npz'
        if dest.exists():return c.read_result(dest)
        # Reuse earlier exact-protocol unshifted cache only after identity checks.
        previous=ROOT/'plots/kmeans/caltech_aligned_translation_1014/case_analysis'/f"{'baseline' if name=='baseline' else 'augmented'}_training.npz"
        if tag=='training' and name!='augment_1_7' and previous.exists():
            meta=json.loads(previous.with_suffix('.identity.json').read_text())
            if meta==dict(checkpoint_sha256=c.sha(path),indices=indices,extractor_sha256=c.sha(l.__file__),coordinate_sha256=c.sha(c.__file__),content_max=160):
                result=c.read_result(previous);c.save_result(dest,result);return result
        result=l.extract(model,c.CorrectedSubset(base,indices,dx,dy),32,a.workers,torch.device(a.device),8)
        c.save_result(dest,result);return result
    tr=extract('training',train);orig=extract('original',ids)
    norms=[unit(f) for f in orig['features']];centers=[];origlabels=[]
    for stage,f in enumerate(tr['features']):
        x=unit(f)
        km=faiss.Kmeans(x.shape[1],31,niter=100,nredo=50,seed=42,spherical=True,gpu=False,max_points_per_centroid=len(x))
        km.train(x);centroid=km.centroids
        assert np.allclose(np.linalg.norm(centroid,axis=1),1,atol=1e-5)
        centers.append(centroid.copy());origlabels.append((norms[stage]@centroid.T).argmax(1))
        print(name,'fitted spherical stage',stage,flush=True)
    np.savez(out/'spherical_centers.npz',**{f'stage{s}':v for s,v in enumerate(centers)})
    del tr
    rows=[];summaries=[]
    for shift in range(8):
        directions={'original':(0,0)} if shift==0 else c.DIRECTIONS
        for direction,(ux,uy) in directions.items():
            moved=orig if shift==0 else extract(f'shift_{shift}_{direction}',ids,ux*shift,uy*shift)
            assert np.array_equal(moved['source_indices'],orig['source_indices'])
            for stage in range(4):
                f=unit(moved['features'][stage]);scores=f@centers[stage].T;labels=scores.argmax(1);sorted_scores=np.sort(scores,axis=1)
                cosine=np.sum(norms[stage]*f,axis=1)
                if not shift:
                    assert np.all(labels==origlabels[stage]) and np.allclose(cosine,1,atol=1e-5)
                    assert np.array_equal(moved['predictions'],orig['predictions'])
                for i,source in enumerate(ids):
                    old,new=orig['metadata'][i],moved['metadata'][i]
                    assert np.isclose(new['center_x']-old['center_x'],ux*shift)
                    assert np.isclose(new['center_y']-old['center_y'],uy*shift)
                    truth=int(orig['labels'][i]);before=int(orig['predictions'][i]);after=int(moved['predictions'][i])
                    rows.append(dict(model=name,category=base.categories[truth],source_index=source,stage=stage,shift_pixels=shift,direction=direction,
                                     original_cluster=int(origlabels[stage][i]),moved_cluster=int(labels[i]),same_cluster=bool(origlabels[stage][i]==labels[i]),
                                     cosine=float(cosine[i]),spherical_margin=float(sorted_scores[i,-1]-sorted_scores[i,-2]),
                                     original_prediction=before,moved_prediction=after,original_correct=before==truth,moved_correct=after==truth,prediction_consistent=before==after,
                                     token_changed=c.position(old,8*2**stage)!=c.position(new,8*2**stage),
                                     padding_contact=l.token_touches_padding(new,new['center_x'],new['center_y'],8*2**stage)))
            print(name,shift,direction,flush=True)
        for stage in range(4):
            for cat in ['all_selected']+CLASSES:
                sub=[r for r in rows if r['stage']==stage and r['shift_pixels']==shift and (cat=='all_selected' or r['category']==cat)]
                counts=np.bincount([r['moved_cluster'] for r in sub],minlength=31);prob=counts[counts>0]/len(sub)
                summaries.append(dict(model=name,category=cat,stage=stage,shift_pixels=shift,pairs=len(sub),source_images=len({r['source_index'] for r in sub}),
                                      accuracy=float(np.mean([r['moved_correct'] for r in sub])),prediction_consistency=float(np.mean([r['prediction_consistent'] for r in sub])),
                                      cluster_agreement=float(np.mean([r['same_cluster'] for r in sub])),token_cosine=float(np.mean([r['cosine'] for r in sub])),
                                      spherical_margin=float(np.mean([r['spherical_margin'] for r in sub])),used_clusters=int(sum(counts>0)),cluster_entropy=float(-sum(prob*np.log(prob)))))
        c.write_csv(out/'paired_metrics.csv',rows)
    assert len(rows)==202*29*4
    assert len({(r['source_index'],r['stage'],r['shift_pixels'],r['direction']) for r in rows})==len(rows)
    c.write_csv(out/'summary.csv',summaries)
    fig,axes=plt.subplots(4,3,figsize=(13,13),sharex=True)
    for stage in range(4):
        sub=[r for r in summaries if r['stage']==stage and r['category']=='all_selected']
        for col,(key,title) in enumerate([('accuracy','Accuracy'),('cluster_agreement','Cluster agreement'),('token_cosine','Token cosine')]):
            axes[stage,col].plot([r['shift_pixels'] for r in sub],[r[key] for r in sub],marker='o')
            axes[stage,col].set_title(f'Stage {stage}: {title}');axes[stage,col].set_ylim(0,1.02);axes[stage,col].set_xticks(range(8));axes[stage,col].grid(alpha=.3)
    for ax in axes[-1]:ax.set_xlabel('Translation (px)')
    fig.tight_layout();fig.savefig(out/'translation_stability.png',dpi=160);plt.close(fig)
    manifest.update(status='complete',completed_utc=datetime.now(timezone.utc).isoformat(),verified_pairs=len(rows),artifacts_sha256={f.name:c.sha(f) for f in out.iterdir() if f.suffix in ['.csv','.png']})
    write_json(out/'run_manifest.json',manifest);report();print('COMPLETE',name,flush=True)

if __name__=='__main__':main()
