#!/usr/bin/env python3
"""Matched frozen, train-only rank32/ridge10 probes for CE vs CLIP ROI alignment."""
import datetime
import hashlib
import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dataloader.ObjectScalePairs import ObjectScalePairs, feature_grid
from models.ObjectScaleViT import ObjectScaleViT, corresponding_regions


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def cka(x, y):
    x, y = x - x.mean(0), y - y.mean(0)
    return float(np.square(x.T @ y).sum() /
                 np.sqrt(np.square(x.T @ x).sum() * np.square(y.T @ y).sum()))


def main():
    torch.set_num_threads(4)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out = ROOT / 'plots/reports' / ('clip_cross_stage_' + stamp)
    out.mkdir(parents=True, exist_ok=False)
    print('OUTPUT', out, flush=True)
    folders = {'CE': ROOT/'runs/train/object_scale_full_20261031T093902Z/A',
               'CLIP': ROOT/'runs/train/clip_supervision_1101/full'}
    configs = {n: json.loads((p/'config.json').read_text()) for n, p in folders.items()}
    for key in ['splits', 'seed', 'epochs', 'batch_size', 'model_args', 'pred_weight', 'region_weight']:
        assert configs['CE'][key] == configs['CLIP'][key], key
    assert not set(configs['CE']['splits']['train']) & set(configs['CE']['splits']['valid'])
    teacher_path = ROOT/'datasets/clip_bbox_vitb32_1101.pt'
    teacher = torch.load(teacher_path, map_location='cpu', weights_only=False)['features'].numpy().astype('float64')
    splits = configs['CE']['splits']
    target, target_valid = unit(teacher[splits['train']]), unit(teacher[splits['valid']])
    predictions, raw, diagnostics, labels, probe_weights = {}, {}, {}, None, {}
    for name, folder in folders.items():
        cfg = configs[name]
        checkpoint = torch.load(folder/'best.pth', map_location='cpu', weights_only=False)
        model = ObjectScaleViT(**cfg['model_args']).to('cuda:1').eval()
        model.load_state_dict(checkpoint['model'])
        values = {}
        with torch.inference_mode():
            for split in ['train', 'valid']:
                ds = ObjectScalePairs(str(ROOT/'data'), cfg['splits'][split], training=False)
                buckets = [[] for _ in range(4)]
                split_labels, indices = [], []
                for batch in DataLoader(ds, batch_size=32, num_workers=2):
                    _, features = model(batch['a'].to('cuda:1'), True)
                    points, mask = batch['points_a'].to('cuda:1'), batch['region'].to('cuda:1')
                    for s, f in enumerate(features):
                        r = corresponding_regions(f, feature_grid(points, f, 8*2**s), mask)
                        buckets[s].append(r.cpu().numpy())
                    split_labels.extend(batch['label'].tolist())
                    indices.extend(batch['index'].tolist())
                assert indices == splits[split]
                values[split] = [np.concatenate(v).astype('float64') for v in buckets]
                if split == 'valid':
                    if labels is None:
                        labels = np.array(split_labels)
                    else:
                        assert np.array_equal(labels, split_labels)
                print(name, split, len(ds), 'extracted', flush=True)
        raw[name] = values['valid']
        predictions[name], diagnostics[name] = [], []
        for s in range(4):
            train, valid = values['train'][s], values['valid'][s]
            mean = train.mean(0)
            _, _, vt = np.linalg.svd(train-mean, full_matrices=False)
            basis = vt[:32].T
            x = (train-mean) @ basis
            std = np.maximum(x.std(0), 1e-8)
            x = np.c_[x/std, np.ones(len(train))]
            xv = np.c_[((valid-mean)@basis)/std, np.ones(len(valid))]
            penalty = np.eye(33)*10
            penalty[-1, -1] = 0
            weights = np.linalg.solve(x.T@x+penalty, x.T@target)
            z = unit(xv@weights)
            # Train-prediction mean removes the shared teacher direction, fitted without validation.
            centered = unit(xv@weights-(x@weights).mean(0))
            predictions[name].append((z, centered))
            probe_weights.update({f'{name}_{s}_'+k: v for k, v in
                                 dict(mean=mean, basis=basis, std=std, weights=weights).items()})
            diagnostics[name].append(dict(stage=s+1, teacher_cosine=float((z*target_valid).sum(1).mean())))
        del model, checkpoint
        torch.cuda.empty_cache()
    permutation = np.arange(len(labels))
    for label in np.unique(labels):
        ix = np.flatnonzero(labels == label)
        assert len(ix) > 1
        permutation[ix] = np.roll(ix, 1)
    rng = np.random.default_rng(42)
    bootstrap = rng.integers(len(labels), size=(2000, len(labels)))
    rows, arrays = [], dict(source_indices=np.array(splits['valid']), labels=labels)
    for i, j in itertools.combinations(range(4), 2):
        row = dict(stages=[i+1, j+1])
        for metric, k in [('cosine', 0), ('centered_cosine', 1)]:
            scores = {}
            for name in folders:
                a, b = predictions[name][i][k], predictions[name][j][k]
                same = (a*b).sum(1)
                control = (a*b[permutation]).sum(1)
                scores[name] = dict(same=same, same_class_other=control, gap=same-control)
            row[metric] = {}
            for stat in ['same', 'same_class_other', 'gap']:
                d = scores['CLIP'][stat]-scores['CE'][stat]
                ci = np.quantile(d[bootstrap].mean(1), [.025, .975])
                row[metric][stat] = dict(CE=float(scores['CE'][stat].mean()),
                                        CLIP=float(scores['CLIP'][stat].mean()),
                                        delta=float(d.mean()), source_bootstrap_95ci=ci.tolist())
                for name in folders:
                    arrays[f'{name}_{i}_{j}_{metric}_{stat}'] = scores[name][stat]
        row['raw_linear_cka'] = {n: cka(raw[n][i], raw[n][j]) for n in folders}
        rows.append(row)
    for name in folders:
        for s in range(4):
            arrays[f'{name}_stage{s}_projected'] = predictions[name][s][0]
            arrays[f'{name}_stage{s}_centered'] = predictions[name][s][1]
    np.savez_compressed(out/'per_source.npz', **arrays)
    np.savez_compressed(out/'probes.npz', **probe_weights)
    manifest = dict(utc=stamp, machine='remote140165', project=str(ROOT), n_train=len(target),
                    n_valid=len(labels), rank=32, ridge=10, bootstrap_repeats=2000,
                    checkpoints={n: dict(path=str(p/'best.pth'), sha256=sha(p/'best.pth')) for n,p in folders.items()},
                    teacher_sha256=sha(teacher_path), script_sha256=sha(Path(__file__)),
                    source_sha256={p: sha(ROOT/p) for p in ['models/ObjectScaleViT.py', 'models/MergingViT.py', 'dataloader/ObjectScalePairs.py']},
                    splits=splits)
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2))
    (out/'results.json').write_text(json.dumps(dict(rows=rows, diagnostics=diagnostics), indent=2))
    lines = ['# CE 與 CLIP：同物件跨層表示比較', '',
             f'日期 {stamp}；remote140165；6902 training／1768 validation；同來源、同 seed42 與100epochs既有 checkpoint。', '',
             '凍結兩組 backbone；每層僅在 training 原位長邊96的 mask 加權8×8物件ROI fit PCA rank32，再標準化、ridge10（含不懲罰截距）映射至同一 frozen CLIP bbox 教師512維。validation不擬合、不調參。兩組都重新 fit 同容量 probe，不用CLIP組既有alignment heads作不對稱比較。', '',
             '這是整物件 ROI pooling 的共同空間 cosine，不是原生逐token／部件 cosine，也不等同真實merging父子對應。', '',
             '| 層配對 | CE cosine | CLIP cosine | 差值 CLIP−CE | 來源配對95% CI | CE同類異圖差距 | CLIP同類異圖差距 |',
             '|---|---:|---:|---:|---|---:|---:|']
    for r in rows:
        v, gap = r['cosine']['same'], r['cosine']['gap']
        ci = v['source_bootstrap_95ci']
        lines.append(f"| {r['stages'][0]}↔{r['stages'][1]} | {v['CE']:.4f} | {v['CLIP']:.4f} | {v['delta']:+.4f} | [{ci[0]:+.4f}, {ci[1]:+.4f}] | {gap['CE']:.4f} | {gap['CLIP']:.4f} |")
    lines += ['', '## 去除共同均值方向後的 cosine', '',
              '各層減去其training預測均值再L2正規化；避免共同教師均值使所有來源看似接近。', '',
              '| 層配對 | CE | CLIP | 差值 | CE同類異圖差距 | CLIP同類異圖差距 |', '|---|---:|---:|---:|---:|---:|']
    for r in rows:
        v, g = r['centered_cosine']['same'], r['centered_cosine']['gap']
        lines.append(f"| {r['stages'][0]}↔{r['stages'][1]} | {v['CE']:.4f} | {v['CLIP']:.4f} | {v['delta']:+.4f} | {g['CE']:.4f} | {g['CLIP']:.4f} |")
    lines += ['', '## 不使用 CLIP probe 的原始表示對照', '',
              'linear CKA 衡量跨來源的兩層表示幾何相似性，可比較不同維度；不是逐物件cosine，且CKA較高不自動等於語意改善。', '',
              '| 層配對 | CE raw CKA | CLIP raw CKA |', '|---|---:|---:|']
    for r in rows:
        v = r['raw_linear_cka']
        lines.append(f"| {r['stages'][0]}↔{r['stages'][1]} | {v['CE']:.4f} | {v['CLIP']:.4f} |")
    lines += ['', '## 限制與交接', '',
              '單seed、探索／選模validation；bootstrap只重抽來源影像，不重新擬合probe、不含訓練seed變異。六個配對CI為未校正的描述性區間；同類異圖為類內循環錯配。教師也提供共同對齊錨點，因此共同空間相似不能等同原始token語意或部件因果證據。', '',
              '此報告及SHA256版本副本應送E:\\RGB_SFM_artifacts inbox；本機專案root未確認。收件匣交付不代表本機專案匯入，匯入需驗hash並保留不同版本。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print((out/'REPORT.md').read_text(), flush=True)


if __name__ == '__main__':
    main()
