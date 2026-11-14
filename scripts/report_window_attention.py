#!/usr/bin/env python3
"""Compare CE window/global models using identical frozen spherical protocols."""
import hashlib
import itertools
import json
import sys
from pathlib import Path

import faiss
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from compare_clip_spherical_scale import RegionViews, assign, fit, mapping, unit
from dataloader.ObjectScalePairs import ObjectScalePairs, feature_grid
from models.ObjectScaleViT import ObjectScaleViT, corresponding_regions

BASE=ROOT/'runs/train/object_scale_full_20261031T093902Z/A'
CACHE=ROOT/'plots/kmeans/clip_support_matched_20261109T060934Z'
CONDITIONS=[(96,0,0),(80,0,0),(112,0,0),(80,7,-5),(64,0,0),(128,0,0),(192,0,0),(128,-16,-16),(192,-16,-16)]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    torch.set_num_threads(4);faiss.omp_set_num_threads(4)
    run=Path(sys.argv[1]).resolve()
    local=run/'local';out=run/'spherical';out.mkdir(exist_ok=False)
    cfg=json.loads((local/'config.json').read_text());bcfg=json.loads((BASE/'config.json').read_text())
    for key in ['splits','seed','epochs','batch_size','model_args','pred_weight','region_weight','annotation_sha256']:
        assert cfg[key]==bcfg[key],key
    assert cfg['attention_mode']=='window' and cfg['clip_cache'] is None
    assert cfg['pred_weight']==cfg['region_weight']==0
    assert json.loads((local/'status.json').read_text())['status']=='done'
    prior=json.loads((CACHE/'manifest.json').read_text())
    assert sha(BASE/'best.pth')==prior['checkpoints']['CE']['sha256']
    expected={x['path']:x['sha256'] for x in json.loads((CACHE/'feature_cache_manifest.json').read_text())['files']}
    reused={}
    def load(path):
        digest=sha(path);relative=str(path.relative_to(ROOT))
        if relative in expected:assert digest==expected[relative],relative
        reused[relative]=digest
        return np.load(path)
    checkpoint=torch.load(local/'best.pth',map_location='cpu',weights_only=False)
    model=ObjectScaleViT(**cfg['model_args'],attention_mode=cfg['attention_mode'],window_size=cfg['window_size']).to('cuda:1').eval()
    model.load_state_dict(checkpoint['model'])
    assert [b[0].attn.window_size for b in model.stages]==[7,7,7,4]
    values={};labels=None;indices=None
    base=ObjectScalePairs(str(ROOT/'data'),[]).base
    with torch.inference_mode():
        for split,condition in [('train',(96,0,0))]+[('valid',c) for c in CONDITIONS]:
            length,dx,dy=condition;buckets=[[] for _ in range(4)];got=[];ys=[];pred=[]
            ds=RegionViews(base,cfg['splits'][split],length,(dx,dy))
            for batch in DataLoader(ds,batch_size=32,num_workers=2):
                logits,features=model(batch['a'].to('cuda:1'),True)
                points,mask=batch['points_a'].to('cuda:1'),batch['region'].to('cuda:1')
                for s,f in enumerate(features):
                    buckets[s].append(corresponding_regions(f,feature_grid(points,f,8*2**s),mask).cpu().numpy())
                got.extend(batch['index'].tolist());ys.extend(batch['label'].tolist());pred.append(logits.cpu().numpy())
            assert got==cfg['splits'][split]
            data={f'stage{s}':unit(np.concatenate(v)) for s,v in enumerate(buckets)}
            data.update(source_indices=np.array(got),labels=np.array(ys),logits=np.concatenate(pred))
            np.savez_compressed(out/f'window_{split}_L{length}_{dx}_{dy}_features.npz',**data)
            values[('window',split,condition)]=data
            if split=='valid':
                if indices is None:indices,labels=np.array(got),np.array(ys)
                else:assert np.array_equal(indices,got) and np.array_equal(labels,ys)
            print('window',split,condition,'features saved',flush=True)
    del model,checkpoint;torch.cuda.empty_cache()
    for split,condition in [('train',(96,0,0))]+[('valid',c) for c in CONDITIONS]:
        length,dx,dy=condition
        if (dx,dy)==(7,-5):
            data={}
            for s in range(4):
                cached=load(ROOT/f'plots/kmeans/clip_roi_1103/A_raw_stage{s}.npz')
                assert np.array_equal(cached['source_indices'],indices)
                data[f'stage{s}']=cached['b']
            data.update(source_indices=indices,labels=labels)
        else:
            path=(CACHE/f'CE_{split}_L{length}_features.npz' if dx==0 else
                  CACHE/'phase_control'/f'CE_valid_L{length}_shift_minus16_features.npz')
            cached=load(path);data={k:cached[k] for k in cached.files}
            if 'source_indices' in data:assert np.array_equal(data['source_indices'],cfg['splits'][split])
        values[('global',split,condition)]=data
    centers,maps,assignments,occupancy={},{},{},{}
    for name in ['global','window']:
        train_ids=[]
        for s in range(4):
            tr=values[(name,'train',(96,0,0))][f'stage{s}']
            center=(load(ROOT/f'plots/kmeans/clip_roi_1103/A_raw_stage{s}.npz')['centers'] if name=='global' else fit(tr))
            assert np.allclose(np.linalg.norm(center,axis=1),1,atol=1e-4)
            centers[f'{name}_{s}']=center
            train_ids.append(assign(tr,center))
            for c in CONDITIONS:
                tag=f'{name}_{s}_{c}'
                a=assign(values[(name,'valid',c)][f'stage{s}'],center);assignments[tag]=a
                p=np.bincount(a,minlength=31)/len(a)
                occupancy[tag]=dict(occupied=int((p>0).sum()),max_fraction=float(p.max()),
                    effective_clusters=float(np.exp(-(p[p>0]*np.log(p[p>0])).sum())))
        for i,j in itertools.combinations(range(4),2):maps[f'{name}_{i}_{j}']=mapping(train_ids[i],train_ids[j])
    # Exact historical cluster assignment reproduction is the baseline contract.
    for s in range(4):
        old=load(ROOT/f'plots/kmeans/clip_roi_1103/A_raw_stage{s}.npz')
        assert np.array_equal(assignments[f'global_{s}_{(96,0,0)}'],old['cluster_a'])
        assert np.array_equal(assignments[f'global_{s}_{(80,7,-5)}'],old['cluster_b'])
    permutation=np.arange(len(labels))
    for label in np.unique(labels):
        ix=np.flatnonzero(labels==label);assert len(ix)>1;permutation[ix]=np.roll(ix,1)
    rows,per_source=[],{}
    def add(name,metric,stages,condition,same,control):
        same,control=np.asarray(same,dtype='float32'),np.asarray(control,dtype='float32')
        key=f'{name}_{metric}_{stages}_{condition}'
        per_source[key],per_source[key+'_control']=same,control
        rows.append(dict(model=name,metric=metric,stages=stages,condition=condition,
            agreement=float(same.mean()),same_class_other=float(control.mean()),gap=float((same-control).mean()),key=key))
    for name in ['global','window']:
        def a(s,c):return assignments[f'{name}_{s}_{c}']
        for s in range(4):
            for c in [(80,0,0),(112,0,0),(80,7,-5)]:
                x,y=a(s,(96,0,0)),a(s,c)
                add(name,'within',[s+1],str(c),x==y,x==y[permutation])
        for i,j in itertools.combinations(range(4),2):
            m=maps[f'{name}_{i}_{j}']
            for length in [80,112]:
                x,y=a(i,(96,0,0)),m[a(j,(length,0,0))]
                xx,yy=a(i,(length,0,0)),m[a(j,(96,0,0))]
                same=((x==y).astype(float)+(xx==yy))/2
                control=((x==y[permutation]).astype(float)+(xx==yy[permutation]))/2
                add(name,'cross_historical',[i+1,j+1],str(length),same,control)
            if j-i==1:
                for big in [128,192]:
                    for metric in ['support_center','support_phase']:
                        shift=-16 if metric=='support_phase' and j>=2 else 0
                        x,y=a(i,(big//2,0,0)),m[a(j,(big,shift,shift))]
                        add(name,metric,[i+1,j+1],str(big),x==y,x==y[permutation])
    boot=np.random.default_rng(42).integers(len(labels),size=(2000,len(labels)))
    summaries,effects=[],[]
    specifications=[('within',[[1],[2],[3],[4],'mean'],[str((80,0,0)),str((112,0,0))]),
                    ('within_shift',[[1],[2],[3],[4],'mean'],[str((80,7,-5))]),
                    ('cross_historical',[[i,j] for i,j in itertools.combinations(range(1,5),2)]+['mean'],['80','112']),
                    ('support_center',[[1,2],[2,3],[3,4],'mean'],['128','192']),
                    ('support_phase',[[1,2],[2,3],[3,4],'mean'],['128','192'])]
    for metric,stage_options,conditions in specifications:
        actual_metric='within' if metric=='within_shift' else metric
        for stages in stage_options:
            scores,controls={},{}
            for name in ['global','window']:
                selected=[r for r in rows if r['model']==name and r['metric']==actual_metric and r['condition'] in conditions and
                          (stages=='mean' or r['stages']==stages)]
                scores[name]=np.mean([per_source[r['key']] for r in selected],axis=0)
                controls[name]=np.mean([per_source[r['key']+'_control'] for r in selected],axis=0)
                summaries.append(dict(model=name,metric=metric,stages=stages,agreement=float(scores[name].mean()),
                    same_class_other=float(controls[name].mean()),gap=float((scores[name]-controls[name]).mean())))
            d=scores['window']-scores['global']
            gap=d-(controls['window']-controls['global'])
            effects.append(dict(metric=metric,stages=stages,delta=float(d.mean()),
                source_bootstrap_95ci=np.quantile(d[boot].mean(1),[.025,.975]).tolist(),gap_delta=float(gap.mean())))
    classification={};classification_scores={}
    for name,folder in [('global',BASE),('window',local)]:
        z=np.load(folder/'scale_predictions.npz');assert np.array_equal(z['source_indices'],indices)
        correct=np.mean([z[f'length_{l}_shift_{dx}_{dy}'].argmax(1)==labels for l in [80,112] for dx,dy in [(0,0),(7,-5)]],axis=0)
        classification_scores[name]=correct
        classification[name]=dict(reference=float((z['length_96_shift_0_0'].argmax(1)==labels).mean()),unseen=float(correct.mean()))
    d=classification_scores['window']-classification_scores['global']
    classification['effect']=dict(delta=float(d.mean()),source_bootstrap_95ci=np.quantile(d[boot].mean(1),[.025,.975]).tolist())
    assert all(np.isfinite(v).all() for v in per_source.values())
    np.savez_compressed(out/'per_source.npz',source_indices=indices,labels=labels,**per_source)
    np.savez_compressed(out/'dictionaries_assignments.npz',**centers,**maps,**assignments)
    result=dict(rows=rows,summaries=summaries,effects=effects,classification=classification,occupancy=occupancy)
    (out/'results.json').write_text(json.dumps(result,indent=2))
    manifest=dict(project=str(ROOT),n_train=len(cfg['splits']['train']),n_valid=len(indices),seed=42,
        checkpoints={n:dict(path=str(p/'best.pth'),sha256=sha(p/'best.pth')) for n,p in [('global',BASE),('window',local)]},
        reused_cache_sha256=reused,script_sha256=sha(Path(__file__)),training_script_sha256=cfg['source_sha256']['scripts/train_object_scale.py'],
        attention='unshifted windows7/7/7/4 on grids28/14/7/4; final two stages cover whole grid; no relative position bias or additional blocks',
        spherical='FAISS spherical K31 niter100 nredo50 seed42; training96 codebooks and Hungarian alignment; no validation fitting')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    lines=['# 原版全局MHA vs 局部窗口MHA：純CE控制實驗', '',
        '相同初始化張量、參數量1304261、原merging、絕對位置編碼、stage深度與head數；純CE無CLIP，完整6902training／1768validation、seed42、100epochs、batch16、AdamW3e-4。學生皆去背物件與合成背景、雙view縮放64/96/128與每軸−16..16平移。', '',
        '窗口7×7，四stage實際窗口7/7/7/4；後兩stage7×7與4×4網格的attention仍覆蓋全層。沒有shifted windows或relative position bias，故不是完整Swin；本次主要隔離前兩stage全局／窗口attention。兩組相同初始權重及資料，attention dropout張量形狀不同，不能聲稱逐位置dropout亂數完全匹配。', '',
        '各模型原始pre-merge整物件ROI 8×8 mask加權pool後分群；不是逐token分群。FAISS spherical K31、100iter、50restarts、seed42，training96建字典及Hungarian群號配對，validation只評估。global重用歷史原版字典並逐筆重現舊96／80平移群指派。', '',
        '| 分類指標 | 全局MHA | 窗口MHA |','|---|---:|---:|',
        f"| 原位96 | {classification['global']['reference']:.2%} | {classification['window']['reference']:.2%} |",
        f"| 未見80/112（含原位與平移） | {classification['global']['unseen']:.2%} | {classification['window']['unseen']:.2%} |"]
    for metric,title,stage_options in [('within','同層跨尺度80/112',[[1],[2],[3],[4],'mean']),
        ('within_shift','舊條件96→80+(7,-5)',[[1],[2],[3],[4],'mean']),
        ('cross_historical','跨層＋跨尺度96↔80/112',[[i,j] for i,j in itertools.combinations(range(1,5),2)]+['mean']),
        ('support_center','相鄰層名義支援比例匹配64→128／96→192',[[1,2],[2,3],[3,4],'mean']),
        ('support_phase','相鄰層支援比例＋網格位置校正',[[1,2],[2,3],[3,4],'mean'])]:
        lines += ['',f'## {title}', '', '| 層／配對 | 全局MHA | 窗口MHA | 差值 |','|---|---:|---:|---:|']
        for stages in stage_options:
            label='平均' if stages=='mean' else '↔'.join(map(str,stages))
            v=[next(r['agreement'] for r in summaries if r['metric']==metric and r['stages']==stages and r['model']==n) for n in ['global','window']]
            lines.append('| '+label+f' | {v[0]:.2%} | {v[1]:.2%} | {100*(v[1]-v[0]):+.2f}pp |')
    lines += ['', '## 配對來源bootstrap：窗口−全局，平均指標', '']
    for e in effects:
        if e['stages']=='mean':
            ci=e['source_bootstrap_95ci']
            lines.append(f"- {e['metric']}: {100*e['delta']:+.2f}pp，95% CI [{100*ci[0]:+.2f}, {100*ci[1]:+.2f}]，同類異圖差距改變 {100*e['gap_delta']:+.2f}pp。")
    lines += ['', '單seed、探索／選模validation；bootstrap2000次先在來源內平均，再重抽來源，不含訓練或K-means seed變異。支援比例及位置校正不是實測有效RF或逐token支援框匹配；位置校正也改變絕對位置。群佔用與同類異圖控制見results.json。', '',
        '遠端專案 /home/ref/RGB_SFM_V3；E: inbox版本交付不代表本機專案匯入，本機root未確認，匯入需驗SHA256並保留不同版本。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print((out/'REPORT.md').read_text(),flush=True)


if __name__=='__main__':
    main()
