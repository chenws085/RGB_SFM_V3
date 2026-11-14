#!/usr/bin/env python3
"""Frozen CE/CLIP raw vs matched CLIP probes: spherical scale/stage agreement."""
import datetime
import hashlib
import itertools
import json
import sys
from pathlib import Path

import faiss
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dataloader.ObjectScalePairs import ObjectScalePairs, object_crop, feature_grid
from models.ObjectScaleViT import ObjectScaleViT, corresponding_regions
from evaluate_object_scale import Views

PROBE = ROOT/'plots/reports/clip_cross_stage_20261109T053032Z'
CONDITIONS = {'96_center': (96, (0, 0)), '80_center': (80, (0, 0)),
              '112_center': (112, (0, 0)), '80_shift': (80, (7, -5))}


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def unit(x):
    return np.ascontiguousarray(x/np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12), dtype='float32')


class RegionViews(Views):
    def __getitem__(self, item):
        x, label, index = super().__getitem__(item)
        image, mask, _, _ = object_crop(self.base, index)
        scale = self.length/max(image.size)
        w, h = max(1, round(image.width*scale)), max(1, round(image.height*scale))
        ox, oy = (224-w)//2+self.shift[0], (224-h)//2+self.shift[1]
        yy, xx = np.meshgrid((np.arange(8)+.5)/8, (np.arange(8)+.5)/8, indexing='ij')
        points = (np.stack([xx, yy], -1)*[w, h]+[ox, oy]).astype('float32')
        from PIL import Image
        region = np.asarray(mask.resize((8, 8), Image.Resampling.BOX), dtype='float32')[None]/255
        return dict(a=x, label=label, index=index, points_a=points, region=region)


def fit(x):
    x = unit(x)
    km = faiss.Kmeans(x.shape[1], 31, niter=100, nredo=50, seed=42,
                      spherical=True, max_points_per_centroid=len(x))
    km.train(x)
    assert np.allclose(np.linalg.norm(km.centroids, axis=1), 1, atol=1e-4)
    return km.centroids.copy()


def assign(x, centers):
    return (unit(x)@centers.T).argmax(1)


def mapping(a, b):
    """Training-only bijection: map b cluster labels onto a labels."""
    counts = np.zeros((31, 31), dtype='int64')
    np.add.at(counts, (a, b), 1)
    rows, cols = linear_sum_assignment(-counts)
    result = np.empty(31, dtype='int64')
    result[cols] = rows
    assert len(np.unique(result)) == 31
    return result


def main():
    torch.set_num_threads(4)
    faiss.omp_set_num_threads(4)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out = ROOT/'plots/kmeans'/('clip_cross_stage_scale_'+stamp)
    out.mkdir(parents=True, exist_ok=False)
    print('OUTPUT', out, flush=True)
    folders = {'CE': ROOT/'runs/train/object_scale_full_20261031T093902Z/A',
               'CLIP': ROOT/'runs/train/clip_supervision_1101/full'}
    probes = np.load(PROBE/'probes.npz')
    prior = json.loads((PROBE/'manifest.json').read_text())
    old = ROOT/'plots/kmeans/clip_roi_1103'
    old_manifest = json.loads((old/'manifest.json').read_text())
    all_labels, all_train, centers_out, mappings_out = {}, {}, {}, {}
    diagnostics, ids, labels = {}, None, None
    for name, folder in folders.items():
        cfg = json.loads((folder/'config.json').read_text())
        assert cfg['splits'] == prior['splits']
        assert sha(folder/'best.pth') == prior['checkpoints'][name]['sha256']
        oldname = 'A' if name == 'CE' else name
        assert sha(folder/'best.pth') == old_manifest[oldname]['checkpoint_sha256']
        checkpoint = torch.load(folder/'best.pth', map_location='cpu', weights_only=False)
        model = ObjectScaleViT(**cfg['model_args']).to('cuda:1').eval()
        model.load_state_dict(checkpoint['model'])
        base = ObjectScalePairs(str(ROOT/'data'), []).base
        values = {}
        with torch.inference_mode():
            for split, condition in [('train', '96_center')]+[('valid', c) for c in CONDITIONS]:
                length, shift = CONDITIONS[condition]
                ds = RegionViews(base, cfg['splits'][split], length, shift)
                buckets = [[] for _ in range(4)]
                got_ids, got_labels = [], []
                for batch in DataLoader(ds, batch_size=32, num_workers=2):
                    _, features = model(batch['a'].to('cuda:1'), True)
                    points, mask = batch['points_a'].to('cuda:1'), batch['region'].to('cuda:1')
                    for s, f in enumerate(features):
                        r = corresponding_regions(f, feature_grid(points, f, 8*2**s), mask)
                        buckets[s].append(r.cpu().numpy())
                    got_ids.extend(batch['index'].tolist())
                    got_labels.extend(batch['label'].tolist())
                assert got_ids == cfg['splits'][split]
                values[(split, condition)] = [np.concatenate(b) for b in buckets]
                if split == 'valid':
                    if ids is None:
                        ids, labels = np.array(got_ids), np.array(got_labels)
                    else:
                        assert np.array_equal(ids, got_ids) and np.array_equal(labels, got_labels)
                print(name, split, condition, 'extracted', flush=True)
        del model, checkpoint
        torch.cuda.empty_cache()
        for kind in ['raw', 'probe']:
            key = name+'_'+kind
            train, valid = [], {c: [] for c in CONDITIONS}
            for s in range(4):
                def transform(x):
                    if kind == 'raw':
                        return unit(x)
                    prefix = f'{name}_{s}_'
                    x = ((x.astype('float64')-probes[prefix+'mean'])@probes[prefix+'basis'])/probes[prefix+'std']
                    return unit(np.c_[x, np.ones(len(x))]@probes[prefix+'weights'])
                tr = transform(values[('train', '96_center')][s])
                train.append(tr)
                for c in CONDITIONS:
                    valid[c].append(transform(values[('valid', c)][s]))
                # Exact earlier readout/geometry reproduction on both historical conditions.
                if kind == 'raw':
                    cached = np.load(old/f'{oldname}_raw_stage{s}.npz')
                    for c, v in [('96_center', 'a'), ('80_shift', 'b')]:
                        assert np.allclose(valid[c][s], cached[v], atol=3e-5), (name, s, c)
                    centers = cached['centers'].copy()
                    # Historical dictionary is the exact direct-middle-state baseline.
                else:
                    centers = fit(tr)
                centers_out[f'{key}_{s}'] = centers
                print(key, 'stage', s+1, 'dictionary ready', flush=True)
            train_ids = [assign(t, centers_out[f'{key}_{s}']) for s, t in enumerate(train)]
            val_ids = {c: [assign(valid[c][s], centers_out[f'{key}_{s}']) for s in range(4)] for c in CONDITIONS}
            for i, j in itertools.combinations(range(4), 2):
                mappings_out[f'{key}_{i}_{j}'] = mapping(train_ids[i], train_ids[j])
            all_labels[key], all_train[key] = val_ids, train_ids
            diagnostics[key] = []
            for s in range(4):
                a = val_ids['96_center'][s]
                counts = np.bincount(a, minlength=31)
                p = counts/counts.sum()
                diagnostics[key].append(dict(stage=s+1, occupied=int((p>0).sum()),
                    max_cluster_fraction=float(p.max()), effective_clusters=float(np.exp(-(p[p>0]*np.log(p[p>0])).sum()))))
            # A shared dictionary is meaningful only in the common CLIP coordinate system.
            if kind == 'probe':
                shared = fit(np.concatenate(train))
                centers_out[key+'_shared'] = shared
                all_labels[key+'_shared'] = {c: [assign(v, shared) for v in valid[c]] for c in CONDITIONS}
            # Persist expensive dictionaries before statistical/report operations.
            partial = dict(centers_out)
            partial.update(mappings_out)
            for branch, branch_values in all_labels.items():
                for c, stages in branch_values.items():
                    for s, a in enumerate(stages):
                        partial[f'assign_{branch}_{c}_{s}'] = a
            np.savez_compressed(out/'dictionaries_assignments.partial.npz', **partial)
    permutation = np.arange(len(labels))
    for label in np.unique(labels):
        ix = np.flatnonzero(labels == label)
        assert len(ix)>1
        permutation[ix] = np.roll(ix, 1)
    bootstrap = np.random.default_rng(42).integers(len(labels), size=(2000, len(labels)))
    per_source, rows = {}, []
    def add(key, metric, condition, stages, same, control):
        same, control = same.astype('float64'), control.astype('float64')
        tag = f'{key}__{metric}__{condition}__'+str(stages)
        per_source[tag] = same.astype('float32')
        per_source[tag+'__control'] = control.astype('float32')
        row = dict(branch=key, metric=metric, condition=condition, stages=stages,
                   agreement=float(same.mean()), same_class_other=float(control.mean()),
                   gap=float((same-control).mean()), per_source_key=tag)
        rows.append(row)
    for key, v in all_labels.items():
        shared = key.endswith('_shared')
        for condition in CONDITIONS:
            for s in range(4):
                a, b = v['96_center'][s], v[condition][s]
                add(key, 'within_stage_scale', condition, [s+1], a==b, a==b[permutation])
            for i, j in itertools.combinations(range(4), 2):
                m = np.arange(31) if shared else mappings_out[f'{key}_{i}_{j}']
                a, b = v['96_center'][i], m[v[condition][j]]
                aa, bb = v[condition][i], m[v['96_center'][j]]
                # Average the two scale orientations within each source image.
                same = ((a==b).astype(float)+(aa==bb))/2
                control = ((a==b[permutation]).astype(float)+(aa==bb[permutation]))/2
                add(key, 'cross_stage_scale', condition, [i+1, j+1], same, control)
    effects = []
    for first, second in [('CE_probe', 'CE_raw'), ('CLIP_probe', 'CLIP_raw'),
                          ('CLIP_raw', 'CE_raw'), ('CLIP_probe', 'CE_probe')]:
        for row in [r for r in rows if r['branch']==first]:
            other = next(r for r in rows if r['branch']==second and
                         all(r[k]==row[k] for k in ['metric', 'condition', 'stages']))
            a, b = row['per_source_key'], other['per_source_key']
            d = per_source[a]-per_source[b]
            gap = d-(per_source[a+'__control']-per_source[b+'__control'])
            effects.append(dict(first=first, second=second, metric=row['metric'],
                condition=row['condition'], stages=row['stages'], delta=float(d.mean()),
                source_bootstrap_95ci=np.quantile(d[bootstrap].mean(1), [.025, .975]).tolist(),
                gap_delta=float(gap.mean())))
    for key, v in all_labels.items():
        for c, stages in v.items():
            for s, a in enumerate(stages):
                centers_out[f'assign_{key}_{c}_{s}'] = a
    np.savez_compressed(out/'dictionaries_assignments.npz', **centers_out, **mappings_out)
    np.savez_compressed(out/'per_source.npz', source_indices=ids, labels=labels, **per_source)
    results = dict(rows=rows, effects=effects, diagnostics=diagnostics)
    (out/'results.json').write_text(json.dumps(results, indent=2))
    manifest = dict(utc=stamp, project=str(ROOT), machine='remote140165', conditions=CONDITIONS,
        n_train=6902, n_valid=len(ids), codebook='FAISS spherical K31 niter100 nredo50 seed42',
        raw_codebooks='reused exact historical per-stage dictionaries after validating normalized readouts',
        cross_stage='per-stage dictionaries aligned with training-only Hungarian bijection; no validation fit; two scale orientations averaged within source',
        probes=dict(path=str(PROBE/'probes.npz'), sha256=sha(PROBE/'probes.npz')),
        checkpoints=prior['checkpoints'], splits=prior['splits'], script_sha256=sha(Path(__file__)),
        data_sources={p: sha(ROOT/p) for p in ['scripts/evaluate_object_scale.py','dataloader/ObjectScalePairs.py','models/ObjectScaleViT.py']})
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2))
    lines = ['# Spherical K-means：投影前後跨層／跨尺度同群率', '',
        'CE與CLIP各自比較raw中間狀態與前次同容量train-only CLIP probe。相同6902 training／1768 validation，固定背景物件cutout、mask加權8×8 ROI。不是逐token或真實merging父子評估。', '',
        '字典為FAISS spherical K31、100 iterations、50 restarts、seed42；只用training原位96 fit，所有validation變換共用固定字典。raw直接重用歷史字典，並驗證重新抽取的96原位與80平移特徵重現舊cache。', '',
        '跨層原始維度不同，各層字典先以training同物件群指派做Hungarian一對一群號對齊，validation不可重配；probe也使用相同對齊規則。跨尺度跨層取兩個尺度方向的影像內平均。額外shared probe字典四層共用，數值見results.json，不能與獨立字典對齊結果混稱同一協定。', '',
        '## 同層跨尺度：80/112置中平均', '',
        '| 層 | CE raw | CE probe | CLIP raw | CLIP probe | CLIP probe−CLIP raw |', '|---|---:|---:|---:|---:|---:|']
    def mean_agreement(key, metric, stages, conditions):
        return np.mean([r['agreement'] for r in rows if r['branch']==key and r['metric']==metric and r['stages']==stages and r['condition'] in conditions])
    branches = ['CE_raw', 'CE_probe', 'CLIP_raw', 'CLIP_probe']
    for s in range(1, 5):
        vals = [mean_agreement(k, 'within_stage_scale', [s], ['80_center','112_center']) for k in branches]
        lines.append('| '+str(s)+' | '+' | '.join(f'{v:.2%}' for v in vals)+f' | {(vals[3]-vals[2])*100:+.2f} pp |')
    lines += ['', '## 跨層＋跨尺度：80/112置中平均（training對齊群號）', '',
        '| 層配對 | CE raw | CE probe | CLIP raw | CLIP probe | CLIP probe−CLIP raw |', '|---|---:|---:|---:|---:|---:|']
    for i, j in itertools.combinations(range(1, 5), 2):
        vals = [mean_agreement(k, 'cross_stage_scale', [i,j], ['80_center','112_center']) for k in branches]
        lines.append('| '+f'{i}↔{j}'+' | '+' | '.join(f'{v:.2%}' for v in vals)+f' | {(vals[3]-vals[2])*100:+.2f} pp |')
    lines += ['', '## 重現舊條件：96原位 → 80+(7,-5)，同層同群率', '',
        '| 層 | CE raw | CE probe | CLIP raw | CLIP probe |', '|---|---:|---:|---:|---:|']
    for s in range(1, 5):
        vals = [mean_agreement(k, 'within_stage_scale', [s], ['80_shift']) for k in branches]
        lines.append('| '+str(s)+' | '+' | '.join(f'{v:.2%}' for v in vals)+' |')
    lines += ['', '## 限制及產物', '',
        '所有同類異圖對照、差距、來源影像配對bootstrap 2000次CI、群佔用與有效群數見results.json。CI未做多重比較校正、不含訓練seed、probe重擬合與K-means seed變異。validation已用於選模，非final test；單seed合成背景結果不可推成部件語意或自然影像泛化。跨層Hungarian只能定義群對應，不能證明兩層群具同一語意。', '',
        '遠端專案 /home/ref/RGB_SFM_V3；版本副本與SHA256 manifest交付E: artifact inbox不代表本機專案匯入，本機root未確認；匯入需驗hash並保留不同版本。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print((out/'REPORT.md').read_text(), flush=True)


if __name__ == '__main__':
    main()
