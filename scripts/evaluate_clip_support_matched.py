#!/usr/bin/env python3
"""Three frozen models, source-relative nominal support matched across stages."""
import datetime
import hashlib
import itertools
import json
import sys
from pathlib import Path

import faiss
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from compare_clip_spherical_scale import RegionViews, assign, fit, mapping, unit
from dataloader.ObjectScalePairs import ObjectScalePairs, feature_grid
from models.ObjectScaleViT import ObjectScaleViT, corresponding_regions

LENGTHS = [16, 24, 32, 48, 64, 80, 96, 112, 128, 192]
FOLDERS = {'CE': ROOT/'runs/train/object_scale_full_20261031T093902Z/A',
           'all': ROOT/'runs/train/clip_supervision_1101/full',
           'deep': ROOT/'runs/train/clip_depth_ablation_1103/deep'}


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def stats(a, b, permutation):
    same = (a == b).astype('float32')
    control = (a == b[permutation]).astype('float32')
    return same, control


def main():
    torch.set_num_threads(4)
    faiss.omp_set_num_threads(4)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out = ROOT/'plots/kmeans'/('clip_support_matched_'+stamp)
    out.mkdir(parents=True, exist_ok=False)
    print('OUTPUT', out, flush=True)
    configs = {n: json.loads((p/'config.json').read_text()) for n,p in FOLDERS.items()}
    for key in ['splits','seed','epochs','batch_size','model_args','pred_weight','region_weight']:
        assert all(c[key] == configs['CE'][key] for c in configs.values()), key
    splits = configs['CE']['splits']
    assert not set(splits['train']) & set(splits['valid'])
    assert configs['deep']['clip_stages'] == [2,3]
    manifest = dict(utc=stamp, machine='remote140165', project=str(ROOT), splits=splits,
        lengths=LENGTHS, codebooks='train96 only, FAISS spherical K31 niter100 nredo50 seed42',
        mapping='Hungarian training96 same-source assignments, fixed across all evaluation conditions',
        checkpoints={n: dict(path=str(p/'best.pth'),sha256=sha(p/'best.pth')) for n,p in FOLDERS.items()},
        script_sha256=sha(Path(__file__)), sources={p:sha(ROOT/p) for p in
            ['scripts/compare_clip_spherical_scale.py','scripts/evaluate_object_scale.py',
             'models/ObjectScaleViT.py','dataloader/ObjectScalePairs.py']})
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2))
    cache, dictionaries, maps, classification = {}, {}, {}, {}
    for name, folder in FOLDERS.items():
        assert json.loads((folder/'status.json').read_text())['status']=='done'
        checkpoint = torch.load(folder/'best.pth', map_location='cpu', weights_only=False)
        model = ObjectScaleViT(**configs[name]['model_args']).to('cuda:1').eval()
        model.load_state_dict(checkpoint['model'])
        base = ObjectScalePairs(str(ROOT/'data'), []).base
        classification[name] = {}
        with torch.inference_mode():
            for split, length in [('train',96)]+[('valid',l) for l in LENGTHS]:
                ds = RegionViews(base, splits[split], length, (0,0))
                buckets, indices, labels, geometry, logits = [[] for _ in range(4)], [], [], [], []
                for batch in DataLoader(ds, batch_size=32, num_workers=2):
                    pred, features = model(batch['a'].to('cuda:1'), True)
                    points, mask = batch['points_a'].to('cuda:1'), batch['region'].to('cuda:1')
                    for s, f in enumerate(features):
                        pooled = corresponding_regions(f, feature_grid(points,f,8*2**s),mask)
                        buckets[s].append(pooled.cpu().numpy())
                    points_cpu = batch['points_a'].numpy()
                    geometry.append(np.stack([8*(points_cpu[:,0,1,0]-points_cpu[:,0,0,0]),
                                              8*(points_cpu[:,1,0,1]-points_cpu[:,0,0,1])],axis=1))
                    indices.extend(batch['index'].tolist())
                    labels.extend(batch['label'].tolist())
                    logits.append(pred.cpu().numpy())
                assert indices == splits[split]
                data = dict(source_indices=np.array(indices), labels=np.array(labels),
                            geometry_wh=np.concatenate(geometry), logits=np.concatenate(logits),
                            **{f'stage{s}':unit(np.concatenate(v)) for s,v in enumerate(buckets)})
                np.savez_compressed(out/f'{name}_{split}_L{length}_features.npz', **data)
                cache[(name,split,length)] = data
                if split=='valid':
                    classification[name][str(length)] = float((data['logits'].argmax(1)==data['labels']).mean())
                print(name, split, length, 'features saved', flush=True)
        del model, checkpoint
        torch.cuda.empty_cache()
        oldname = 'A' if name=='CE' else name
        for s in range(4):
            if name in ['CE','all']:
                oldname = 'A' if name=='CE' else 'CLIP'
                old = np.load(ROOT/f'plots/kmeans/clip_roi_1103/{oldname}_raw_stage{s}.npz')
                assert np.allclose(cache[(name,'valid',96)][f'stage{s}'],old['a'],atol=3e-5)
                centers = old['centers'].copy()
            else:
                centers = fit(cache[(name,'train',96)][f'stage{s}'])
            assert np.allclose(np.linalg.norm(centers,axis=1),1,atol=1e-4)
            dictionaries[f'{name}_{s}'] = centers
        train_ids = [assign(cache[(name,'train',96)][f'stage{s}'],dictionaries[f'{name}_{s}']) for s in range(4)]
        for i,j in itertools.combinations(range(4),2):
            maps[f'{name}_{i}_{j}'] = mapping(train_ids[i],train_ids[j])
        np.savez_compressed(out/'dictionaries.partial.npz', **dictionaries, **maps)
    np.savez_compressed(out/'dictionaries.npz', **dictionaries, **maps)
    labels = cache[('CE','valid',96)]['labels']
    permutation = np.arange(len(labels))
    for label in np.unique(labels):
        ix = np.flatnonzero(labels==label)
        assert len(ix)>1
        permutation[ix] = np.roll(ix,1)
    assignments, occupancy = {}, {}
    for name in FOLDERS:
        for length in LENGTHS:
            for s in range(4):
                key = f'{name}_L{length}_S{s}'
                a = assign(cache[(name,'valid',length)][f'stage{s}'],dictionaries[f'{name}_{s}'])
                assignments[key] = a
                p = np.bincount(a,minlength=31)/len(a)
                occupancy[key] = dict(occupied=int((p>0).sum()),max_fraction=float(p.max()),
                    effective_clusters=float(np.exp(-(p[p>0]*np.log(p[p>0])).sum())))
    per_source, rows, geometry_checks = {}, [], []
    for i,j in itertools.combinations(range(4),2):
        ratio = 2**(j-i)
        for big in [128,192]:
            small = big//ratio
            gsmall = cache[('CE','valid',small)]['geometry_wh']
            gbig = cache[('CE','valid',big)]['geometry_wh']
            error = np.abs(gbig/(gsmall*ratio)-1)
            geometry_checks.append(dict(stages=[i+1,j+1],small=small,big=big,
                nominal_stride_ratio=ratio,max_relative_axis_rounding_error=float(error.max()),
                p95_relative_axis_rounding_error=float(np.quantile(error,.95))))
            for condition, li, lj in [('matched',small,big),('reversed',big,small),('same_size_large',big,big)]:
                for name in FOLDERS:
                    a = assignments[f'{name}_L{li}_S{i}']
                    b = maps[f'{name}_{i}_{j}'][assignments[f'{name}_L{lj}_S{j}']]
                    same, control = stats(a,b,permutation)
                    tag = f'{name}_{condition}_S{i+1}_{j+1}_L{small}_{big}'
                    per_source[tag],per_source[tag+'_control'] = same,control
                    rows.append(dict(model=name,condition=condition,stages=[i+1,j+1],
                        shallow_length=li,deep_length=lj,matched_small=small,matched_big=big,
                        agreement=float(same.mean()),same_class_other=float(control.mean()),
                        gap=float((same-control).mean()),per_source_key=tag))
        # Preserve the historical non-support-matched protocol for the new deep group too.
        for name in FOLDERS:
            m = maps[f'{name}_{i}_{j}']
            for length in [80,112]:
                a = assignments[f'{name}_L96_S{i}']
                b = m[assignments[f'{name}_L{length}_S{j}']]
                aa = assignments[f'{name}_L{length}_S{i}']
                bb = m[assignments[f'{name}_L96_S{j}']]
                x,xc = stats(a,b,permutation)
                y,yc = stats(aa,bb,permutation)
                same,control = (x+y)/2,(xc+yc)/2
                tag = f'{name}_historical_S{i+1}_{j+1}_L{length}'
                per_source[tag],per_source[tag+'_control'] = same,control
                rows.append(dict(model=name,condition='historical',stages=[i+1,j+1],
                    matched_big=length,agreement=float(same.mean()),same_class_other=float(control.mean()),
                    gap=float((same-control).mean()),per_source_key=tag))
    assert all(np.isfinite(v).all() for v in per_source.values())
    boot = np.random.default_rng(42).integers(len(labels),size=(2000,len(labels)))
    summaries, effects = [], []
    for condition, bigs in [('matched',[128]),('matched',[192]),('matched',[128,192]),
                             ('reversed',[128,192]),('same_size_large',[128,192]),('historical',[80,112])]:
        for stages in [[1,2],[1,3],[1,4],[2,3],[2,4],[3,4],'adjacent_mean','all_pairs_mean']:
            scores,controls = {},{}
            for name in FOLDERS:
                selected=[r for r in rows if r['condition']==condition and r['model']==name and r['matched_big'] in bigs and
                    (r['stages']==stages or stages=='all_pairs_mean' or
                     (stages=='adjacent_mean' and r['stages'][1]-r['stages'][0]==1))]
                score = np.mean([per_source[r['per_source_key']] for r in selected],axis=0)
                control = np.mean([per_source[r['per_source_key']+'_control'] for r in selected],axis=0)
                scores[name],controls[name] = score,control
                summaries.append(dict(condition=condition,big_lengths=bigs,stages=stages,model=name,
                    agreement=float(score.mean()),same_class_other=float(control.mean()),gap=float((score-control).mean())))
            for first,second in [('all','CE'),('deep','CE'),('deep','all')]:
                d = scores[first]-scores[second]
                gap = d-(controls[first]-controls[second])
                effects.append(dict(condition=condition,big_lengths=bigs,stages=stages,first=first,second=second,
                    delta=float(d.mean()),source_bootstrap_95ci=np.quantile(d[boot].mean(1),[.025,.975]).tolist(),
                    gap_delta=float(gap.mean()),gap_delta_95ci=np.quantile(gap[boot].mean(1),[.025,.975]).tolist()))
    np.savez_compressed(out/'assignments.npz', **assignments)
    np.savez_compressed(out/'per_source.npz',source_indices=np.array(splits['valid']),labels=labels,**per_source)
    result=dict(rows=rows,summaries=summaries,effects=effects,classification=classification,
                geometry_checks=geometry_checks,occupancy=occupancy)
    (out/'results.json').write_text(json.dumps(result,indent=2))
    lines=['# 跨層名義支援尺度對應：CE／全層CLIP／深層CLIP', '',
        '既有三模型、原始pre-merge整物件ROI表示，不使用事後probe或alignment head。學生皆去背cutout＋合成純色背景，224畫布；6902training／1768validation、seed42、100epochs。', '',
        '| 訓練設定 | 純CE | CE＋全四層CLIP | CE＋深層CLIP |','|---|---|---|---|',
        '| 直接CLIP監督 | 無 | Stage1/2/3/4 | Stage3/4 |',
        '| CLIP總權重 | 0 | 0.2 | 0.2 |',
        '| 單層相對係數（view平均後） | 0 | 0.05 | 0.10 |',
        '| 訓練大小／平移 | 64/96/128；每軸−16..16 | 相同 | 相同 |',
        '| 學生／教師 | 去背學生，無教師 | 去背學生；凍結CLIP原始bbox教師 | 相同 |', '',
        '全層與深層總權重相同，但深層每層係數為兩倍。因此此三模型比較不能單獨識別監督位置或單層loss強度的作用。', '',
        '## 工程定義與限制', '',
        'Stage名義stride 8/16/32/64；配對令stride_i/L_i=stride_j/L_j，較深層配較大物件。固定置中純色背景，僅改物件大小。這是相對名義網格支援範圍匹配，不是量到有效receptive field：全局attention已混合全圖；網格相位、短邊整數resize、padding與bilinear ROI採樣也未完全相等。geometry_checks保存實際長寬取整誤差。8×8 mask加權ROI只匹配來源區域，不是逐token或真實merging父子支援框的一對一匹配。', '',
        '字典皆training原位96的FAISS spherical K31／100iter／50restarts／seed42。CE及all沿用舊字典；deep新增相同規格字典。跨層群號對應只在training96同物件擬合Hungarian後固定，所有尺度共用，不在validation重配群或fit。故同群率是training對齊後的群指派一致率，不是已知語意相同。validation曾選best checkpoint；非final test。', '',
        '## 相鄰層主比較：兩個支援尺度匹配條件平均', '',
        '| 層配對 | 淺層物件→深層物件 | 純CE | 全層CLIP | 深層CLIP |','|---|---|---:|---:|---:|']
    def value(condition,bigs,stages,name):
        return next(r['agreement'] for r in summaries if r['condition']==condition and r['big_lengths']==bigs and r['stages']==stages and r['model']==name)
    for stages in [[1,2],[2,3],[3,4],'adjacent_mean']:
        title = '平均' if isinstance(stages,str) else f'{stages[0]}↔{stages[1]}'
        vals=[value('matched',[128,192],stages,n) for n in FOLDERS]
        lines.append('| '+title+' | 64→128、96→192 | '+' | '.join(f'{v:.2%}' for v in vals)+' |')
    for big,title in [(128,'64→128（兩個大小皆曾訓練）'),(192,'96→192（192超出訓練長邊）')]:
        lines += ['',f'## 分條件：{title}', '', '| 層配對 | 純CE | 全層CLIP | 深層CLIP |','|---|---:|---:|---:|']
        for stages in [[1,2],[2,3],[3,4],'adjacent_mean']:
            label='平均' if isinstance(stages,str) else f'{stages[0]}↔{stages[1]}'
            lines.append('| '+label+' | '+' | '.join(f"{value('matched',[big],stages,n):.2%}" for n in FOLDERS)+' |')
    lines += ['', '## 非相鄰層探索：小物件資訊退化混雜', '',
        '| 層配對 | 淺層→深層大小 | 純CE | 全層CLIP | 深層CLIP |','|---|---|---:|---:|---:|']
    for stages in [[1,3],[1,4],[2,4]]:
        ratio=2**(stages[1]-stages[0])
        sizes=f'{128//ratio}→128、{192//ratio}→192'
        lines.append('| '+f'{stages[0]}↔{stages[1]}'+' | '+sizes+' | '+' | '.join(f"{value('matched',[128,192],stages,n):.2%}" for n in FOLDERS)+' |')
    lines += ['', '## 舊協定對照：96↔80/112，兩大小方向影像內平均', '',
        '| 層配對 | 純CE | 全層CLIP | 深層CLIP |','|---|---:|---:|---:|']
    for i,j in itertools.combinations(range(1,5),2):
        lines.append('| '+f'{i}↔{j}'+' | '+' | '.join(f"{value('historical',[80,112],[i,j],n):.2%}" for n in FOLDERS)+' |')
    lines += ['', '## 主比較的配對來源信賴區間', '']
    for e in effects:
        if e['condition']=='matched' and e['big_lengths']==[128,192] and e['stages']=='adjacent_mean':
            ci=e['source_bootstrap_95ci']
            lines.append(f"- {e['first']}−{e['second']}: {100*e['delta']:+.2f}pp，95% CI [{100*ci[0]:+.2f}, {100*ci[1]:+.2f}]；同物件−同類異圖差距改變 {100*e['gap_delta']:+.2f}pp。")
    lines += ['', 'results.json另含每配對的同類異圖對照、同群率差距、配對CI、反向大小／同大物件對照、群佔用與分類accuracy。bootstrap2000次先在來源內平均各條件再抽來源，不把尺度與層當獨立樣本；不含training seed、K-means seed或字典重fit變異，CI未多重比較校正。', '',
        '遠端 /home/ref/RGB_SFM_V3；報告／資料／SHA256版本交付E: artifact inbox不代表本機專案匯入，本機root未核實，匯入需驗hash並保留不同版本。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print((out/'REPORT.md').read_text(),flush=True)


if __name__=='__main__':
    main()
