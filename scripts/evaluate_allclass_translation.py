#!/usr/bin/env python3
"""Stratified all-class translation evaluation with verified frozen spherical codebooks."""
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
OUT=ROOT/'plots/kmeans/caltech_allclass_translation_1020'
CHECKPOINTS={'baseline':'runs/train/mergingvit_caltech101_letterbox160/MergingViT_best.pth',
             'aligned_8_16_32':'runs/train/mergingvit_caltech101_aligned_translation_1014/MergingViT_best.pth',
             'augment_1_7':'runs/train/mergingvit_caltech101_non_aligned_restart_1015/MergingViT_best.pth'}
CLASSES=['airplanes','chandelier','brain']


def unit(x):
    x=np.ascontiguousarray(x,dtype=np.float32)
    if not np.isfinite(x).all() or np.any(np.linalg.norm(x,axis=1)<1e-10):
        raise ValueError('Invalid feature for spherical clustering')
    return x/np.linalg.norm(x,axis=1,keepdims=True)


def write_json(path,obj):
    path.write_text(json.dumps(obj,indent=2))


def dataset_fingerprint(base, indices):
    import hashlib
    checksum=hashlib.sha256()
    for source in sorted(indices):
        label=base.y[source]
        image=Path(base.root)/'101_ObjectCategories'/base.categories[label]/f'image_{base.index[source]:04d}.jpg'
        ann=Path(base.root)/'Annotations'/base.annotation_categories[label]/f'annotation_{base.index[source]:04d}.mat'
        for path in (image,ann):
            checksum.update(f'{source}:{path.relative_to(base.root)}\n'.encode())
            checksum.update(path.read_bytes())
    return checksum.hexdigest()


def main():
    global OUT, CLASSES
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',choices=list(CHECKPOINTS));p.add_argument('--device',default='cpu');p.add_argument('--workers',type=int,default=4);p.add_argument('--shifts',type=int,nargs='+',default=[0,1,2,3,4,5,6,7,8,16,32])
    p.add_argument('--checkpoint',type=Path,help='Override selected model checkpoint for a separate training run')
    p.add_argument('--output-dir',type=Path,default=OUT,help='Separate evaluation/cache/report directory')
    p.add_argument('--per-class',type=int,default=5)
    p.add_argument('--sample-seed',type=int,default=42)
    p.add_argument('--codebook-dir',type=Path,default=ROOT/'plots/kmeans/caltech_mixed_translation_1020')
    a=p.parse_args()
    if a.per_class<1:p.error('--per-class must be positive')
    OUT=ROOT/a.output_dir
    if a.checkpoint:
        if not a.model:p.error('--checkpoint requires --model')
        CHECKPOINTS[a.model]=str(a.checkpoint)
    OUT.mkdir(parents=True,exist_ok=True)
    if len(set(a.shifts))!=len(a.shifts) or 0 not in a.shifts or min(a.shifts)<0 or max(a.shifts)>32:p.error('Unique shifts must include 0 and stay within 0..32')
    if not a.model:p.error('--model required')
    torch.set_num_threads(4);faiss.omp_set_num_threads(4)
    name=a.model;out=OUT/name;out.mkdir(exist_ok=True)
    path=ROOT/CHECKPOINTS[name]
    ckpt=torch.load(path,map_location='cpu',weights_only=False)
    model=l.MergingViT(**ckpt['model_args']).to(a.device).eval();model.load_state_dict(ckpt['model_weights'])
    base=l.Caltech101(str(ROOT/'data'),target_type=['category','annotation'],download=False)
    train,valid=l.split_indices(str(ROOT/'data'),42);valid=sorted(valid)
    CLASSES=list(base.categories);assert len(CLASSES)==101
    rng=np.random.default_rng(a.sample_seed);ids=[];selection=[]
    for category,label in enumerate(CLASSES):
        candidates=[i for i in valid if base.y[i]==category]
        chosen=sorted(rng.choice(candidates,size=min(a.per_class,len(candidates)),replace=False).tolist())
        assert chosen
        ids.extend(chosen);selection.append(dict(category=label,valid_available=len(candidates),selected=len(chosen),source_indices=chosen))
    ids=sorted(ids);n_sources=len(ids)
    selection_manifest=dict(seed=a.sample_seed,per_class_cap=a.per_class,classes=101,total_images=n_sources,selection=selection)
    selection_path=OUT/'selection_manifest.json'
    if selection_path.exists():assert json.loads(selection_path.read_text())==selection_manifest
    else:write_json(selection_path,selection_manifest)
    print('Selected',n_sources,'validation images across',len(CLASSES),'classes',flush=True)
    assert not set(train)&set(valid)
    config=dict(model=name,checkpoint_sha256=c.sha(path),runner_sha256=c.sha(__file__),extractor_sha256=c.sha(l.__file__),coordinate_sha256=c.sha(c.__file__),
                train_indices=train,valid_indices=valid,evaluation_indices=ids,classes=CLASSES,
                coordinate_version=c.COORDINATE_VERSION,content_max=160,shifts=a.shifts,K=31,niter=100,nredo=50,seed=42,faiss=faiss.__version__,spherical=True,
                device=a.device,torch=torch.__version__)
    config['protocol']='allclass_translation_1020_v1'
    config['sampling']=selection_manifest
    config['data_fingerprint']=dataset_fingerprint(base,train+valid)
    source_dir=a.codebook_dir/name
    source_manifest=json.loads((source_dir/'run_manifest.json').read_text())
    assert source_manifest['status']=='complete'
    for key in ['checkpoint_sha256','data_fingerprint','train_indices','valid_indices','extractor_sha256','coordinate_sha256']:
        assert source_manifest[key]==config[key],key
    assert c.sha(source_dir/'spherical_centers.npz')==source_manifest['centers_sha256']
    config['codebook_source']=str(source_dir.relative_to(ROOT))
    config['codebook_source_manifest_sha256']=c.sha(source_dir/'run_manifest.json')
    config['centers_sha256']=source_manifest['centers_sha256']
    ident=out/'cache_identity.json'
    if ident.exists() and json.loads(ident.read_text())!=config:raise RuntimeError('Cache identity mismatch')
    write_json(ident,config)
    manifest=dict(config,status='running',started_utc=datetime.now(timezone.utc).isoformat());write_json(out/'run_manifest.json',manifest)
    def extract(tag,indices,dx=0,dy=0):
        dest=out/f'{tag}.npz'
        if dest.exists():return c.read_result(dest)
        result=l.extract(model,c.CorrectedSubset(base,indices,dx,dy),32,a.workers,torch.device(a.device),8)
        c.save_result(dest,result);return result
    orig=extract('original',ids)
    norms=[unit(f) for f in orig['features']]
    with np.load(source_dir/'spherical_centers.npz') as raw:
        centers=[raw[f'stage{stage}'].copy() for stage in range(4)]
    for centroid in centers:assert np.allclose(np.linalg.norm(centroid,axis=1),1,atol=1e-5)
    origlabels=[(norms[stage]@centers[stage].T).argmax(1) for stage in range(4)]
    np.savez(out/'spherical_centers.npz',**{f'stage{stage}':v for stage,v in enumerate(centers)})
    rows=[];summaries=[]
    for shift in a.shifts:
        shift_start=len(rows)
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
        shift_rows=rows[shift_start:]
        for stage in range(4):
            stage_rows=[r for r in shift_rows if r['stage']==stage]
            for cat in ['all_selected']+CLASSES:
                sub=stage_rows if cat=='all_selected' else [r for r in stage_rows if r['category']==cat]
                counts=np.bincount([r['moved_cluster'] for r in sub],minlength=31);prob=counts[counts>0]/len(sub)
                summaries.append(dict(model=name,category=cat,stage=stage,shift_pixels=shift,pairs=len(sub),source_images=len({r['source_index'] for r in sub}),
                                      accuracy=float(np.mean([r['moved_correct'] for r in sub])),prediction_consistency=float(np.mean([r['prediction_consistent'] for r in sub])),
                                      cluster_agreement=float(np.mean([r['same_cluster'] for r in sub])),token_cosine=float(np.mean([r['cosine'] for r in sub])),
                                      spherical_margin=float(np.mean([r['spherical_margin'] for r in sub])),used_clusters=int(sum(counts>0)),cluster_entropy=float(-sum(prob*np.log(prob)))))
        c.write_csv(out/'paired_metrics.csv',rows)
    assert len(rows)==n_sources*(1+4*(len(a.shifts)-1))*4
    assert len({(r['source_index'],r['stage'],r['shift_pixels'],r['direction']) for r in rows})==len(rows)
    c.write_csv(out/'summary.csv',summaries)
    fig,axes=plt.subplots(4,3,figsize=(13,13),sharex=True)
    for stage in range(4):
        sub=[r for r in summaries if r['stage']==stage and r['category']=='all_selected']
        for col,(key,title) in enumerate([('accuracy','Accuracy'),('cluster_agreement','Cluster agreement'),('token_cosine','Token cosine')]):
            axes[stage,col].plot([r['shift_pixels'] for r in sub],[r[key] for r in sub],marker='o')
            axes[stage,col].set_title(f'Stage {stage}: {title}');axes[stage,col].set_ylim(0,1.02);axes[stage,col].set_xticks(a.shifts);axes[stage,col].grid(alpha=.3)
    for ax in axes[-1]:ax.set_xlabel('Translation (px)')
    fig.tight_layout();fig.savefig(out/'translation_stability.png',dpi=160);plt.close(fig)
    manifest.update(status='complete',completed_utc=datetime.now(timezone.utc).isoformat(),verified_pairs=len(rows),artifacts_sha256={f.name:c.sha(f) for f in out.iterdir() if f.suffix in ['.csv','.png']})
    manifest['centers_sha256']=c.sha(out/'spherical_centers.npz')
    write_json(out/'run_manifest.json',manifest);print('COMPLETE',name,flush=True)

if __name__=='__main__':main()
