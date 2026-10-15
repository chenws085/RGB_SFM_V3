#!/usr/bin/env python3
"""Compare original and translation-augmented models on held-out airplanes."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits
import run_caltech_corrected_center as corrected
import test_caltech_translation_kmeans as legacy
from plot_aligned_translation_examples import save_examples


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--augmented', required=True)
    p.add_argument('--baseline', default='runs/train/mergingvit_caltech101_letterbox160/MergingViT_best.pth')
    p.add_argument('--output-dir', required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--workers', type=int, default=4)
    a = p.parse_args()
    torch.set_num_threads(4)
    threadpool_limits(4)
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    legacy.CONTENT_MAX_SIZE = 160
    base = legacy.Caltech101('data', target_type=['category', 'annotation'], download=False)
    train, valid = legacy.split_indices('data', 42)
    airplane = base.categories.index('airplanes')
    ids = sorted(i for i in valid if base.y[i] == airplane)
    assert not set(train) & set(valid)
    manifest = dict(status='running', shifts=[0,8,16,32], content_max=160, canvas=224,
                    coordinate_version=corrected.COORDINATE_VERSION, train_indices=train,
                    evaluation_indices=ids, seed=42, K=31, n_init=50, models={},
                    code_sha256={str(f):corrected.sha(f) for f in [Path(__file__),Path(corrected.__file__),Path(legacy.__file__)]},
                    note='Validation used for checkpoint selection; exploratory comparison, one training seed.')
    summaries, samples = [], []
    for name, checkpoint in [('baseline',a.baseline),('augmented',a.augmented)]:
        ckpt = torch.load(checkpoint, map_location='cpu', weights_only=False)
        model = legacy.MergingViT(**ckpt['model_args']).to(a.device).eval()
        model.load_state_dict(ckpt['model_weights'])
        manifest['models'][name] = dict(checkpoint=checkpoint, sha256=corrected.sha(checkpoint), epoch=ckpt.get('epoch'))
        (out/'run_manifest.json').write_text(json.dumps(manifest,indent=2))
        def extract(indices, dx=0, dy=0):
            return legacy.extract(model, corrected.CorrectedSubset(base,indices,dx,dy),32,a.workers,torch.device(a.device),8)
        training = extract(train)
        original = extract(ids)
        centers = [KMeans(n_clusters=31,n_init=50,random_state=42).fit(normalize(f)) for f in training['features']]
        del training
        for shift in [0,8,16,32]:
            directions = {'original':(0,0)} if shift==0 else corrected.DIRECTIONS
            for direction,(dx,dy) in directions.items():
                moved = original if not shift else extract(ids,dx*shift,dy*shift)
                for stage in range(4):
                    before,after = normalize(original['features'][stage]),normalize(moved['features'][stage])
                    c0,c1 = centers[stage].predict(before),centers[stage].predict(after)
                    rows=[]
                    for i,source in enumerate(ids):
                        old,new=original['metadata'][i],moved['metadata'][i]
                        assert np.isclose(new['center_x']-old['center_x'],dx*shift)
                        assert np.isclose(new['center_y']-old['center_y'],dy*shift)
                        row=dict(model=name,stage=stage,shift_pixels=shift,direction=direction,source_index=source,
                                 same_cluster=bool(c0[i]==c1[i]),cosine=float(before[i]@after[i]),
                                 prediction_consistent=bool(original['predictions'][i]==moved['predictions'][i]),
                                 original_prediction=int(original['predictions'][i]),moved_prediction=int(moved['predictions'][i]),
                                 original_correct=bool(original['predictions'][i]==airplane),moved_correct=bool(moved['predictions'][i]==airplane),
                                 padding_contact=legacy.token_touches_padding(new,new['center_x'],new['center_y'],8*2**stage),
                                 token_changed=corrected.position(old,8*2**stage)!=corrected.position(new,8*2**stage),
                                 original_cluster=int(c0[i]),moved_cluster=int(c1[i]))
                        rows.append(row)
                    if shift==0:
                        assert all(r['prediction_consistent'] and r['same_cluster'] and abs(r['cosine']-1)<1e-5 for r in rows)
                    samples.extend(rows)
                    summaries.append(dict(model=name,**corrected.aggregate(rows,stage,shift,direction,'all')))
                corrected.write_csv(out/'translation_samples.csv',samples)
                corrected.write_csv(out/'translation_summary.csv',summaries)
                print(f'{name}: {shift}px {direction}',flush=True)
        save_examples(base,ids,out/'examples',samples,name)
        del model
        torch.cuda.empty_cache()
    specs=[('cluster_agreement','Cluster agreement'),('matched_token_cosine','Token cosine'),('prediction_consistency','Prediction consistency'),('classification_accuracy','Accuracy')]
    for name in ['baseline','augmented']:
        fig,axes=plt.subplots(4,4,figsize=(17,14),sharex=True)
        for stage in range(4):
            for col,(metric,title) in enumerate(specs):
                for direction in corrected.DIRECTIONS:
                    rows=[r for r in summaries if r['model']==name and r['stage']==stage and r['direction']==direction]
                    zero=next(r for r in summaries if r['model']==name and r['stage']==stage and r['shift_pixels']==0)
                    axes[stage,col].plot([0]+[r['shift_pixels'] for r in rows],[zero[metric]]+[r[metric] for r in rows],marker='o',label=direction)
                axes[stage,col].set_title(f'Stage {stage}: {title}')
                axes[stage,col].set_xticks([0,8,16,32]); axes[stage,col].set_ylim(0,1.02); axes[stage,col].grid(alpha=.3)
        axes[0,-1].legend(); fig.tight_layout(); fig.savefig(out/f'{name}_translation_stability.png',dpi=160); plt.close(fig)
    fig,axes=plt.subplots(4,4,figsize=(17,14),sharex=True)
    for stage in range(4):
        for col,(metric,title) in enumerate(specs):
            for name in ['baseline','augmented']:
                values=[np.mean([r[metric] for r in summaries if r['model']==name and r['stage']==stage and r['shift_pixels']==s]) for s in [0,8,16,32]]
                axes[stage,col].plot([0,8,16,32],values,marker='o',label=name)
            axes[stage,col].set_title(f'Stage {stage}: {title}'); axes[stage,col].set_xticks([0,8,16,32]); axes[stage,col].set_ylim(0,1.02); axes[stage,col].grid(alpha=.3)
    axes[0,-1].legend(); fig.tight_layout(); fig.savefig(out/'augmentation_comparison.png',dpi=160); plt.close(fig)
    manifest['status']='complete'
    manifest['artifacts_sha256']={f.name:corrected.sha(f) for f in out.iterdir() if f.suffix in ['.csv','.png']}
    (out/'run_manifest.json').write_text(json.dumps(manifest,indent=2))
    report=['# All-class aligned translation augmentation; airplanes evaluation','',
            '160-in-224; uniform training choice over identity and four directions × 8/16/32 px. 64 px deferred.',
            'Original training augmentations retained. Models use seed 42 and original split. Codebooks fitted separately on unshifted training activations; cluster IDs are not shared.',
            'Validation was used for checkpoint selection. One training seed; exploratory evidence.', '',
            '| Model | Shift | Accuracy | Prediction consistency |','|---|---:|---:|---:|']
    for name in ['baseline','augmented']:
        for shift in [0,8,16,32]:
            rows=[r for r in summaries if r['model']==name and r['stage']==0 and r['shift_pixels']==shift]
            report.append(f"| {name} | {shift} | {np.mean([r['classification_accuracy'] for r in rows]):.4f} | {np.mean([r['prediction_consistency'] for r in rows]):.4f} |")
    report+=['','![comparison](augmentation_comparison.png)']
    for name in ['baseline','augmented']:
        for stage,shift in enumerate([8,16,32]):
            report += ['', f'![{name} {shift}px](examples/{name}_airplanes_stage{stage}_{shift}px_montage.png)']
    (out/'REPORT.md').write_text('\n'.join(report))

if __name__=='__main__':
    main()
