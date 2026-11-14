#!/usr/bin/env python3
"""Adjacent-stage nominal support: add image-grid phase control, no refitting."""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluate_clip_support_matched import FOLDERS, sha, stats
from compare_clip_spherical_scale import RegionViews, assign, unit
from dataloader.ObjectScalePairs import ObjectScalePairs, feature_grid
from models.ObjectScaleViT import ObjectScaleViT, corresponding_regions


def main():
    torch.set_num_threads(4)
    run=Path(sys.argv[1]).resolve()
    out=run/'phase_control'
    out.mkdir(exist_ok=False)
    manifest=json.loads((run/'manifest.json').read_text())
    dictionaries=np.load(run/'dictionaries.npz')
    per_source,rows,geometry=[],[],[]
    values={}
    for name,folder in FOLDERS.items():
        assert sha(folder/'best.pth')==manifest['checkpoints'][name]['sha256']
        cfg=json.loads((folder/'config.json').read_text())
        checkpoint=torch.load(folder/'best.pth',map_location='cpu',weights_only=False)
        model=ObjectScaleViT(**cfg['model_args']).to('cuda:1').eval()
        model.load_state_dict(checkpoint['model'])
        base=ObjectScalePairs(str(ROOT/'data'),[]).base
        with torch.inference_mode():
            for length in [128,192]:
                buckets=[[] for _ in range(4)]
                got=[]
                for batch in DataLoader(RegionViews(base,cfg['splits']['valid'],length,(-16,-16)),batch_size=32,num_workers=2):
                    _,features=model(batch['a'].to('cuda:1'),True)
                    points,mask=batch['points_a'].to('cuda:1'),batch['region'].to('cuda:1')
                    for s,f in enumerate(features):
                        buckets[s].append(corresponding_regions(f,feature_grid(points,f,8*2**s),mask).cpu().numpy())
                    got.extend(batch['index'].tolist())
                assert got==cfg['splits']['valid']
                data={f'stage{s}':unit(np.concatenate(v)) for s,v in enumerate(buckets)}
                values[(name,length)]=data
                np.savez_compressed(out/f'{name}_valid_L{length}_shift_minus16_features.npz',**data)
                print('phase',name,length,'saved',flush=True)
        del model,checkpoint
        torch.cuda.empty_cache()
    labels=np.load(run/'CE_valid_L96_features.npz')['labels']
    permutation=np.arange(len(labels))
    for label in np.unique(labels):
        ix=np.flatnonzero(labels==label)
        permutation[ix]=np.roll(ix,1)
    scores,controls,stored={},{},{}
    for i,j in [(0,1),(1,2),(2,3)]:
        for big in [128,192]:
            small=big//2
            small_geom=np.load(run/f'CE_valid_L{small}_features.npz')['geometry_wh']
            big_geom=np.load(run/f'CE_valid_L{big}_features.npz')['geometry_wh']
            offset_small=np.floor((224-small_geom)/2)
            offset_big=np.floor((224-big_geom)/2)
            stride=8*2**j
            shift=0 if j==1 else -16
            def phase_error(offset):
                return np.abs((2*offset_small-offset+stride/2)%stride-stride/2)/stride
            before=phase_error(offset_big)
            after=phase_error(offset_big+shift)
            geometry.append(dict(stages=[i+1,j+1],small=small,big=big,target_shift=[shift,shift],
                phase_cycle_before_p95=float(np.quantile(before,.95)),phase_cycle_after_p95=float(np.quantile(after,.95)),
                phase_cycle_before_max=float(before.max()),phase_cycle_after_max=float(after.max())))
            assert float(after.max())<=1/stride+1e-6
            for name in FOLDERS:
                a=assign(np.load(run/f'{name}_valid_L{small}_features.npz')[f'stage{i}'],dictionaries[f'{name}_{i}'])
                target=(np.load(run/f'{name}_valid_L{big}_features.npz') if shift==0 else values[(name,big)])
                b=dictionaries[f'{name}_{i}_{j}'][assign(target[f'stage{j}'],dictionaries[f'{name}_{j}'])]
                same,control=stats(a,b,permutation)
                key=f'{name}_{i+1}_{j+1}_{small}_{big}'
                stored[key],stored[key+'_control']=same,control
                scores[(name,i,j,big)],controls[(name,i,j,big)]=same,control
                rows.append(dict(model=name,stages=[i+1,j+1],small=small,big=big,agreement=float(same.mean()),
                    same_class_other=float(control.mean()),gap=float((same-control).mean()),per_source_key=key))
    boot=np.random.default_rng(42).integers(len(labels),size=(2000,len(labels)))
    summaries,effects=[],[]
    for stages in [[1,2],[2,3],[3,4],'adjacent_mean']:
        means,gaps={},{}
        for name in FOLDERS:
            selected=[r for r in rows if r['model']==name and (stages=='adjacent_mean' or stages==r['stages'])]
            mean=np.mean([stored[r['per_source_key']] for r in selected],axis=0)
            control=np.mean([stored[r['per_source_key']+'_control'] for r in selected],axis=0)
            means[name],gaps[name]=mean,mean-control
            summaries.append(dict(stages=stages,model=name,agreement=float(mean.mean()),same_class_other=float(control.mean()),gap=float((mean-control).mean())))
        for first,second in [('all','CE'),('deep','CE'),('deep','all')]:
            d=means[first]-means[second]
            gd=gaps[first]-gaps[second]
            effects.append(dict(stages=stages,first=first,second=second,delta=float(d.mean()),
                source_bootstrap_95ci=np.quantile(d[boot].mean(1),[.025,.975]).tolist(),gap_delta=float(gd.mean())))
    assert all(np.isfinite(v).all() for v in stored.values())
    np.savez_compressed(out/'per_source.npz',labels=labels,source_indices=np.array(manifest['splits']['valid']),**stored)
    result=dict(rows=rows,summaries=summaries,effects=effects,geometry=geometry)
    (out/'results.json').write_text(json.dumps(result,indent=2))
    (out/'manifest.json').write_text(json.dumps(dict(parent_manifest_sha256=sha(run/'manifest.json'),
        script_sha256=sha(Path(__file__)),checkpoint_hashes=manifest['checkpoints'],
        target_shift='0,0 for Stage2; -16,-16 for Stage3/4; shallow remains centered',
        fit='No codebook or mapping refit; same training96 dictionaries and Hungarian mappings'),indent=2))
    lines=['# 補充：相鄰層網格相位校正', '',
        '配對大小仍為64→128、96→192。淺層物件置中；目標Stage2保持置中，Stage3/4的大物件平移(-16,-16)。因224畫布的網格起點，這能消除主要名義網格相位差（2↔3約半格、3↔4約四分之一格），剩餘由短邊取整引起的相位差至多一個目標像素。所有物件仍在畫布內，沒有裁切；不重新建字典或群號對應。', '',
        '此控制同時改變物件絕對位置，不可單獨將差異歸因相位；全局attention有效RF、padding與ROI雙線性插值仍不等同嚴格原生token支援框匹配。', '',
        '| 層配對 | 純CE | 全層CLIP | 深層CLIP |','|---|---:|---:|---:|']
    for stages in [[1,2],[2,3],[3,4],'adjacent_mean']:
        label='平均' if isinstance(stages,str) else f'{stages[0]}↔{stages[1]}'
        vals=[next(r['agreement'] for r in summaries if r['stages']==stages and r['model']==n) for n in FOLDERS]
        lines.append('| '+label+' | '+' | '.join(f'{v:.2%}' for v in vals)+' |')
    lines += ['', '來源配對bootstrap2000次（單seed探索validation，非final test）：']
    for e in effects:
        if e['stages']=='adjacent_mean':
            ci=e['source_bootstrap_95ci']
            lines.append(f"- {e['first']}−{e['second']}: {100*e['delta']:+.2f}pp，95% CI [{100*ci[0]:+.2f}, {100*ci[1]:+.2f}]，同類異圖差距改變 {100*e['gap_delta']:+.2f}pp。")
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print((out/'REPORT.md').read_text(),flush=True)


if __name__=='__main__':
    main()
