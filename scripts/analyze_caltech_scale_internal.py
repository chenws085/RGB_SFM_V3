#!/usr/bin/env python3
"""Fixed-case, pre-merge token spherical maps and controlled intervention pilot."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch
import faiss
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from evaluate_caltech_scale import (ScaleSubset, SIZES, unit, sha, dump, csv_write,
    Caltech101, split_indices, MergingViT, CenterLetterbox, area_weights)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inbox', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--train-images', type=int, default=512)
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args()
    torch.set_num_threads(4); faiss.omp_set_num_threads(4)
    out = args.output_dir
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    dump(out/'status.json', {'status':'running'})
    manifests = list((args.inbox/'weights/local').glob('completed_*manifest.json'))
    assert len(manifests) == 1
    entries = json.loads(manifests[0].read_text())
    for entry in entries:
        for f in entry['files']:
            assert sha(args.inbox/f['path']) == f['sha256'], f['path']
    base = Caltech101(str(ROOT/'data'), target_type=['category','annotation'], download=False)
    reference_manifest = json.loads((ROOT/'plots/kmeans/caltech_scale_1015/S0/run_manifest.json').read_text())
    train, valid = reference_manifest['train_indices'], reference_manifest['valid_indices']
    assert len(train)==6907 and len(valid)==1770 and not set(train)&set(valid)
    assert sorted(train+valid)==list(range(len(base)))
    for entry in entries:
        config=json.loads((args.inbox/'weights/local'/entry['run']/'run_config.json').read_text())
        assert sorted(config['train_indices'])==train and sorted(config['valid_indices'])==valid
    cases = [next(i for i in valid if base.categories[base.y[i]] == cat)
             for cat in ('airplanes','chandelier','brain')]
    rng = np.random.default_rng(42)
    train_ids = sorted(rng.choice(train, min(args.train_images,len(train)), replace=False).tolist())
    geometry_grids = [(28,28),(14,14),(7,7),(4,4)]
    metrics, effects, edges, checks = [], [], [], []
    for entry in entries:
        rng = np.random.default_rng(1042)  # Same training token draws and controls across models.
        name = entry['model']; folder = out/name; folder.mkdir()
        checkpoint = args.inbox/next(f['path'] for f in entry['files'] if f['path'].endswith('.pth'))
        old = json.loads((ROOT/'plots/kmeans/caltech_scale_1015'/name/'run_manifest.json').read_text())
        assert sha(checkpoint) == old['checkpoint_sha256']
        ckpt = torch.load(checkpoint, map_location='cpu', weights_only=False)
        assert ckpt['epoch'] == entry['best_epoch']
        model = MergingViT(**ckpt['model_args']).to(args.device).eval()
        model.load_state_dict(ckpt['model_weights'])
        captured = {}
        handles = [stage[-1].register_forward_hook(lambda m,i,o,s=s: captured.__setitem__(s,o.detach()))
                   for s,stage in enumerate(model.stages)]
        def forward(images):
            with torch.inference_mode():
                logits = model(images.to(args.device))
            return logits.cpu().numpy(), [captured[s].cpu().numpy().copy() for s in range(4)]
        samples = [[] for _ in range(4)]
        position_sum = [np.zeros((h*w, ckpt['model_args']['embed_dims'][s]),np.float64)
                        for s,(h,w) in enumerate(geometry_grids)]
        # Eight uniformly selected valid canvas cells per training image and stage.
        # This is a new global token codebook, distinct from the old ROI dictionaries.
        for images,_,_,_ in DataLoader(ScaleSubset(base,train_ids,160),batch_size=32,num_workers=0):
            _, feats = forward(images)
            for s, f in enumerate(feats):
                position_sum[s] += f.sum(axis=0)
                for row in f:
                    samples[s].append(row[rng.choice(len(row),8,replace=False)])
        means = [x/len(train_ids) for x in position_sum]
        centers=[]
        for s in range(4):
            x=unit(np.concatenate(samples[s]))
            km=faiss.Kmeans(x.shape[1],31,niter=100,nredo=50,seed=42,spherical=True,
                            max_points_per_centroid=len(x),gpu=False)
            km.train(x); centers.append(km.centroids.copy())
            assert np.allclose(np.linalg.norm(centers[-1],axis=1),1,atol=1e-5)
        np.savez(folder/'token_codebooks.npz', **{f's{s}':x for s,x in enumerate(centers)})
        states={}
        for size in SIZES:
            ds=ScaleSubset(base,cases,size)
            batch=[ds[i] for i in range(len(ds))]
            images=torch.stack([b[0] for b in batch]); geo=np.stack([b[3] for b in batch])
            logits,feats=forward(images)
            scores=[unit(f.reshape(-1,f.shape[-1])).reshape(*f.shape)@c.T for f,c in zip(feats,centers)]
            assignments=[x.argmax(-1) for x in scores]
            states[size]=(images,geo,logits,feats,assignments,scores)
            np.savez_compressed(folder/f'cases_{size}.npz', source_indices=cases,geometry=geo,logits=logits,
                                **{f's{s}_tokens':f for s,f in enumerate(feats)},
                                **{f's{s}_clusters':f for s,f in enumerate(assignments)})
        for ci,source in enumerate(cases):
            cat=base.categories[base.y[source]]
            fig,axes=plt.subplots(5,7,figsize=(19,13))
            ref=states[160]
            target=int(ref[2][ci].argmax()) # Fixed reference predicted class at every scale.
            for col,size in enumerate(SIZES):
                images,geo,logits,feats,assign,scores=states[size]
                pred=int(logits[ci].argmax())
                axes[0,col].imshow(CenterLetterbox(size)(base[source][0]))
                axes[0,col].set_title(f'{size/160:.1f}x\n{base.categories[pred]}',fontsize=8)
                for s,(h,w) in enumerate(geometry_grids):
                    ax=axes[s+1,col]
                    ax.imshow(CenterLetterbox(size)(base[source][0]),extent=(0,224,224,0))
                    ax.imshow(assign[s][ci].reshape(h,w),cmap=plt.get_cmap('gist_rainbow',31),vmin=0,vmax=30,alpha=.5,
                              extent=(0,w*8*2**s,h*8*2**s,0),interpolation='nearest')
                    box=geo[ci,2:6];x0,y0,x1,y1=box
                    ax.add_patch(plt.Rectangle((x0,y0),x1-x0,y1-y0,fill=False,color='red'))
                    ax.set(xlim=(0,224),ylim=(224,0),title=f'Stage {s}')
                    # Inverse-map cell centers into original image, then to reference canvas.
                    stride=8*2**s
                    yy,xx=np.mgrid[:h,:w];xx=np.minimum((xx.ravel()+.5)*stride,223.5);yy=np.minimum((yy.ravel()+.5)*stride,223.5)
                    original_x=(xx-geo[ci,14])/geo[ci,16];original_y=(yy-geo[ci,15])/geo[ci,17]
                    rx=original_x*ref[1][ci,16]+ref[1][ci,14];ry=original_y*ref[1][ci,17]+ref[1][ci,15]
                    rid=(np.clip(ry//stride,0,h-1).astype(int)*w+np.clip(rx//stride,0,w-1).astype(int))
                    weights,_=area_weights(torch.tensor(box[None],dtype=torch.float32),h,w,stride)
                    weight=weights.numpy()[0]
                    current=unit(feats[s][ci]);reference=unit(ref[3][s][ci])[rid]
                    agreement=assign[s][ci]==ref[4][s][ci][rid]
                    top=np.sort(scores[s][ci],axis=1)
                    metrics.append(dict(model=name,source_index=source,category=cat,content_size=size,stage=s,
                        prediction=base.categories[pred],correct=int(pred==base.y[source]),
                        mapped_bbox_agreement=float(weight@agreement),mapped_bbox_cosine=float(weight@(current*reference).sum(1)),
                        bbox_margin=float(weight@(top[:,-1]-top[:,-2]))))
                    if s<3:
                        ph,pw=geometry_grids[s+1]
                        for token,child in enumerate(assign[s][ci]):
                            y,x=divmod(token,w);parent=(y//2)*pw+x//2
                            edges.append(dict(model=name,source_index=source,content_size=size,child_stage=s,
                                child_token=token,parent_token=parent,child_cluster=int(child),
                                parent_cluster=int(assign[s+1][ci][parent]),child_bbox_weight=float(weight[token])))
                # Stage-2 candidate: reference center token concept, intervened only inside current bbox.
                s=2;h,w=geometry_grids[s];stride=32
                rcx,rcy=ref[1][ci,:2];ref_token=int(rcy//stride)*w+int(rcx//stride)
                concept=int(ref[4][s][ci,ref_token])
                weights,_=area_weights(torch.tensor(geo[ci,2:6][None],dtype=torch.float32),h,w,stride)
                eligible=np.flatnonzero(weights.numpy()[0]>0)
                selected=eligible[assign[s][ci,eligible]==concept]
                image=images[ci:ci+1].to(args.device)
                baseline=logits[ci]
                def intervene(ids,identity=False):
                    def hook(module,inputs,output):
                        value=output.clone()
                        if not identity and len(ids):
                            value[:,ids]=torch.as_tensor(means[s][ids],device=value.device,dtype=value.dtype)
                        return value
                    handle=model.stages[s][-1].register_forward_hook(hook)
                    try:
                        with torch.inference_mode():return model(image)[0].cpu().numpy()
                    finally:handle.remove()
                ident=intervene(selected,True)
                checks.append(float(np.max(np.abs(ident-baseline))))
                treated=intervene(selected)
                random_drops=[]
                for repeat in range(20):
                    control=rng.choice(eligible,len(selected),replace=False)
                    control_logits=intervene(control)
                    random_drops.append(float(baseline[target]-control_logits[target]))
                drop=float(baseline[target]-treated[target])
                effects.append(dict(model=name,source_index=source,category=cat,content_size=size,stage=s,
                    reference_concept=concept,target_class=base.categories[target],selected_tokens=len(selected),
                    eligible_bbox_tokens=len(eligible),logit_drop=drop,random_mean_drop=float(np.mean(random_drops)),
                    effect_gap=drop-float(np.mean(random_drops)),prediction_changed=int(treated.argmax()!=pred)))
            for ax in axes.flat:ax.axis('off')
            fig.suptitle(f'{name} / {cat} / source {source}: token spherical maps (IDs local to model and stage)')
            fig.tight_layout();fig.savefig(folder/f'{cat}_token_maps.png',dpi=140);plt.close(fig)
        for handle in handles:handle.remove()
        dump(folder/'manifest.json',dict(model=name,checkpoint_sha256=sha(checkpoint),best_epoch=ckpt['epoch'],
            train_ids=train_ids,cases=cases,K=31,niter=100,nredo=50,seed=42,
            grid_shapes=geometry_grids,source_manifest_sha256=sha(manifests[0]),runner_sha256=sha(__file__),
            codebook='global per-stage; 8 canvas cells/image, training-160 only; 512-image pilot',
            intervention='stage2 reference-center concept within bbox; per-position raw training mean; 20 bbox random controls'))
        print(name,'complete',flush=True)
    assert max(checks)<1e-4, max(checks)
    csv_write(out/'token_metrics.csv',metrics);csv_write(out/'interventions.csv',effects);csv_write(out/'merge_edges.csv',edges)
    dump(out/'verification.json',dict(identity_max_abs_error=max(checks),checks=len(checks),
        reference_cosine_min=min(r['mapped_bbox_cosine'] for r in metrics if r['content_size']==160),
        reference_agreement_min=min(r['mapped_bbox_agreement'] for r in metrics if r['content_size']==160)))
    lines=['# 縮放案例：逐 token 中間狀態與介入試驗','',
        '三個固定案例（airplanes／chandelier／brain）、三模型、七尺度。此為描述性 pilot，不做群體顯著性推論。',
        '每模型每 stage 分別以固定 512 張 training-160 影像、每張 8 個隨機 token 建立 FAISS spherical K=31 字典（100 iterations、50 restarts）。不同模型／stage 群號不可比較；這不是原報告的 ROI 字典。',
        '顏色對應該模型該層的群號，精確群號請讀 cases_*.npz；紅框為 bbox。對應以 cell 中心逆映射至原圖再映射至原尺度，nearest-cell 並非精確語意對應。Stage 3 含 merging padding。',
        '', '| Model | 未見尺度 bbox token agreement | bbox token cosine | 有支持的介入案例數 | 平均 effect gap |', '|---|---:|---:|---:|---:|']
    for name in ('S0','S1','T1'):
        rows=[r for r in metrics if r['model']==name and r['stage']==2 and r['content_size'] in (112,144,176)]
        erows=[r for r in effects if r['model']==name and r['selected_tokens']>0 and r['content_size'] in (112,144,176)]
        gap=np.mean([r['effect_gap'] for r in erows]) if erows else float('nan')
        lines.append(f"| {name} | {np.mean([r['mapped_bbox_agreement'] for r in rows]):.2%} | {np.mean([r['mapped_bbox_cosine'] for r in rows]):.4f} | {len(erows)}/9 | {gap:.4f} |")
    lines += ['', '介入固定原尺度預測類別作 target；替換該參考中心概念在當前 bbox 內所有 token，使用同位置 raw training mean。對照匹配 stage、token 數與 bbox 範圍，未匹配精確位置、norm 或概念分布；可包含被處理 token。無支持列不作介入證據。effect gap 正值不代表 logit drop 必定為正；替換亦可能偏離 activation 分布。',
        'merge_edges.csv 僅記錄真實 child→parent 結構及群號，不是已驗證的跨層因果路徑。本輪尚未做 A/B/A+B、donor 敏感度或介入後下游群追蹤。',
        '身份介入與原輸出比對、原尺度自對應及 checkpoint hashes 見 verification.json 與各模型 manifest。未重新訓練 backbone。']
    for name in ('S0','S1','T1'):
        lines+=['',f'## {name}']
        for cat in ('airplanes','chandelier','brain'):
            lines+=['',f'![{name} {cat}](./{name}/{cat}_token_maps.png)']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    dump(out/'status.json',{'status':'complete'})

if __name__=='__main__':main()
