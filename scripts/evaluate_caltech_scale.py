#!/usr/bin/env python3
"""All-class scale evaluation with separate center/bbox/central-ROI codebooks."""
import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import faiss
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import ImageDraw
from scipy.io import loadmat
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import Caltech101
from torchvision.transforms import functional as TF

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from train_caltech_mergingvit import CenterLetterbox, IMAGENET_MEAN, IMAGENET_STD
from test_caltech_translation_kmeans import split_indices
from models.MergingViT import MergingViT
from run_caltech_corrected_center import absolute_annotation

SIZES = (96, 112, 128, 144, 160, 176, 192)
READOUTS = ('center', 'bbox', 'central_roi')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n')


def csv_write(path, rows):
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class ScaleSubset(Dataset):
    def __init__(self, base, ids, size):
        self.base, self.ids, self.size = base, list(ids), size

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, item):
        source = self.ids[item]
        image, (label, annotation) = self.base[source]
        width, height = image.size
        factor = self.size / max(width, height)
        rw, rh = max(1, round(width * factor)), max(1, round(height * factor))
        ox, oy = (224-rw)//2, (224-rh)//2
        # Use actual per-axis resize factors, including integer rounding.
        sx, sy = rw/width, rh/height
        contour = absolute_annotation(self.base, source, annotation)
        center = np.array([contour[0].mean()*sx+ox, contour[1].mean()*sy+oy])
        category = self.base.y[source]
        mat = Path(self.base.root)/'Annotations'/self.base.annotation_categories[category]/f'annotation_{self.base.index[source]:04d}.mat'
        box = loadmat(mat)['box_coord'].ravel()
        # MATLAB inclusive pixel indices -> zero-based pixel-edge interval.
        original = np.array([max(0, box[2]-1), max(0, box[0]-1), min(width, box[3]), min(height, box[1])], dtype=float)
        if original[2] <= original[0] or original[3] <= original[1]:
            raise ValueError(f'Invalid annotation bbox: {source}')
        bbox = original*np.array([sx, sy, sx, sy])+np.array([ox, oy, ox, oy])
        mid = (bbox[:2]+bbox[2:])/2
        half = (bbox[2:]-bbox[:2])/4
        roi = np.concatenate([mid-half, mid+half])
        tensor = TF.normalize(TF.to_tensor(CenterLetterbox(self.size)(image)), IMAGENET_MEAN, IMAGENET_STD)
        geometry = np.concatenate([center, bbox, roi, [width, height, rw, rh, ox, oy, sx, sy]])
        return tensor, label, source, geometry


def area_weights(boxes, height, width, stride):
    """Exact ROI/token-cell overlap; last cells clip to the 224px canvas."""
    device = boxes.device
    ys, xs = torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device), indexing='ij')
    x0, y0 = xs.flatten()*stride, ys.flatten()*stride
    x1, y1 = (x0+stride).clamp(max=224), (y0+stride).clamp(max=224)
    wx = (torch.minimum(boxes[:, 2:3], x1)-torch.maximum(boxes[:, 0:1], x0)).clamp(min=0)
    wy = (torch.minimum(boxes[:, 3:4], y1)-torch.maximum(boxes[:, 1:2], y0)).clamp(min=0)
    weights = wx*wy
    area = (boxes[:, 2]-boxes[:, 0])*(boxes[:, 3]-boxes[:, 1])
    coverage = weights.sum(1)/area.clamp(min=1e-9)
    return weights/weights.sum(1, keepdim=True).clamp(min=1e-9), coverage


@torch.inference_mode()
def extract(model, dataset, device, workers):
    # A single real forward supplies both classification and pre-merge features.
    captured = {}
    handles = [stage[-1].register_forward_hook(lambda module, inputs, output, s=s: captured.__setitem__(s, output))
               for s, stage in enumerate(model.stages)]
    parts = {f's{s}_{r}': [] for s in range(4) for r in READOUTS}
    geometry, labels, predictions, sources = [], [], [], []
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=workers, pin_memory=True)
    try:
        for images, target, ids, geo in loader:
            logits = model(images.to(device))
            predictions.append(logits.argmax(1).cpu().numpy())
            labels.append(target.numpy()); sources.append(ids.numpy()); geometry.append(geo.numpy())
            points = geo.to(device=device, dtype=torch.float32)
            h, w = model.patch_embed.grid_h, model.patch_embed.grid_w
            for s in range(4):
                feat = captured[s]
                assert feat.shape[1] == h*w
                stride = 8*2**s
                x, y = (points[:, 0]//stride).long(), (points[:, 1]//stride).long()
                assert torch.all((x >= 0) & (x < w) & (y >= 0) & (y < h))
                center = feat[torch.arange(len(feat), device=device), y*w+x]
                parts[f's{s}_center'].append(center.cpu().numpy())
                for r, offset in (('bbox', 2), ('central_roi', 6)):
                    weights, coverage = area_weights(points[:, offset:offset+4], h, w, stride)
                    assert torch.all(coverage > .999), f'Incomplete ROI coverage: stage {s}'
                    pooled = torch.einsum('bn,bnc->bc', weights, feat)
                    parts[f's{s}_{r}'].append(pooled.cpu().numpy())
                merge = model.merges[s]
                if hasattr(merge, 'm_h'):
                    h = (h+merge.m_h-1)//merge.m_h
                    w = (w+merge.m_w-1)//merge.m_w
    finally:
        for handle in handles:
            handle.remove()
        captured.clear()
    result = {key: np.concatenate(value) for key, value in parts.items()}
    result.update(geometry=np.concatenate(geometry), labels=np.concatenate(labels), predictions=np.concatenate(predictions), source_indices=np.concatenate(sources))
    assert len(np.unique(result['source_indices'])) == len(dataset)
    return result


def unit(features):
    norm = np.linalg.norm(features, axis=1, keepdims=True)
    if not np.isfinite(features).all() or np.any(norm < 1e-10):
        raise ValueError('Invalid spherical feature')
    return np.ascontiguousarray(features/norm, dtype=np.float32)


def bootstrap(values):
    rng = np.random.default_rng(42)
    estimates = [values[rng.integers(len(values), size=len(values))].mean() for _ in range(1000)]
    return [float(x) for x in np.percentile(estimates, [2.5, 97.5])]


def evaluate(name, checkpoint, args, base, train, valid):
    out = args.output_dir/name
    out.mkdir(parents=True, exist_ok=True)
    identity = dict(checkpoint_sha256=sha(checkpoint), runner_sha256=sha(__file__),
                    model_sha256=sha(ROOT/'models/MergingViT.py'),
                    trainer_sha256=sha(ROOT/'scripts/train_caltech_mergingvit.py'),
                    coordinate_sha256=sha(ROOT/'scripts/run_caltech_corrected_center.py'),
                    coordinate_version='box_offset_minus1_actual_resize_axes_bbox_edges_v1',
                    train_indices=train, valid_indices=valid, content_sizes=SIZES,
                    K=args.clusters, nredo=args.nredo, niter=100, seed=42,
                    readouts=READOUTS, faiss=faiss.__version__, torch=torch.__version__, device=args.device)
    identity = json.loads(json.dumps(identity))
    ipath = out/'cache_identity.json'
    if ipath.exists() and json.loads(ipath.read_text()) != identity:
        raise RuntimeError(f'Incompatible existing cache: {out}')
    dump(ipath, identity)
    dump(out/'run_manifest.json', dict(identity, status='running', started_utc=datetime.now(timezone.utc).isoformat()))
    ckpt = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert ckpt['model_args']['patch_size'] == 8
    assert ckpt['model_args']['merge_size'] == [(2, 2)]*3
    model = MergingViT(**ckpt['model_args']).to(args.device).eval()
    model.load_state_dict(ckpt['model_weights'])

    def cached(tag, ids, size):
        path = out/f'{tag}.npz'
        if path.exists():
            with np.load(path) as raw:
                result = {key: raw[key] for key in raw.files}
        else:
            result = extract(model, ScaleSubset(base, ids, size), args.device, args.workers)
            np.savez_compressed(path, **result)
        assert np.array_equal(result['source_indices'], ids)
        print(name, tag, len(ids), flush=True)
        return result

    reference = cached('valid_160', valid, 160)
    fitting = cached('training_160', train, 160)
    centers, refassign = {}, {}
    for s in range(4):
        for r in READOUTS:
            key = f's{s}_{r}'
            x = unit(fitting[key])
            km = faiss.Kmeans(x.shape[1], args.clusters, niter=100, nredo=args.nredo, seed=42,
                              spherical=True, gpu=False, max_points_per_centroid=len(x))
            km.train(x)
            centers[key] = km.centroids.copy()
            assert np.allclose(np.linalg.norm(centers[key], axis=1), 1, atol=1e-5)
            refassign[key] = (unit(reference[key])@centers[key].T).argmax(1)
    np.savez(out/'spherical_centers.npz', **centers)
    del fitting
    classification, summary, paired = [], [], []
    assignments = {}
    negative_ids = np.arange(len(valid))
    negative_valid = np.zeros(len(valid), dtype=bool)
    for label in np.unique(reference['labels']):
        ids = np.flatnonzero(reference['labels'] == label)
        if len(ids) > 1:
            negative_ids[ids] = np.roll(ids, 1)
            negative_valid[ids] = True
    for size in SIZES:
        current = reference if size == 160 else cached(f'valid_{size}', valid, size)
        assert np.array_equal(current['labels'], reference['labels'])
        correct = current['predictions'] == current['labels']
        consistent = current['predictions'] == reference['predictions']
        for label in [-1, *np.unique(current['labels']).tolist()]:
            mask = np.ones(len(valid), dtype=bool) if label == -1 else current['labels'] == label
            classification.append(dict(model=name, content_size=size, scale=size/160,
                category='all' if label == -1 else base.categories[label], samples=int(mask.sum()),
                accuracy=float(correct[mask].mean()), prediction_consistency=float(consistent[mask].mean()),
                macro_accuracy=float(np.mean([correct[current['labels'] == c].mean() for c in np.unique(current['labels'][mask])]))))
        for s in range(4):
            for r in READOUTS:
                key = f's{s}_{r}'
                z, ref = unit(current[key]), unit(reference[key])
                scores = z@centers[key].T
                assignment = scores.argmax(1)
                agreement = assignment == refassign[key]
                cosine = (z*ref).sum(1).clip(-1, 1)
                negative_cosine = (z*ref[negative_ids]).sum(1).clip(-1, 1)
                top = np.partition(scores, -2, axis=1)[:, -2:]
                margin = top[:, 1]-top[:, 0]
                counts = np.bincount(assignment, minlength=args.clusters)
                p = counts[counts > 0]/len(valid)
                summary.append(dict(model=name, content_size=size, scale=size/160, stage=s, readout=r,
                    samples=len(valid), cluster_agreement=float(agreement.mean()), cosine=float(cosine.mean()),
                    same_class_other_image_cosine=float(negative_cosine[negative_valid].mean()),
                    correspondence_gap=float((cosine-negative_cosine)[negative_valid].mean()),
                    spherical_margin=float(margin.mean()), used_clusters=int((counts > 0).sum()),
                    largest_cluster_fraction=float(p.max()), cluster_entropy=float(-(p*np.log(p)).sum())))
                assignments[(size, key)] = assignment
                for i, source in enumerate(valid):
                    paired.append(dict(source_index=source, category=base.categories[int(current['labels'][i])],
                        content_size=size, scale=size/160, stage=s, readout=r, correct=int(correct[i]),
                        prediction_consistent=int(consistent[i]), cluster=int(assignment[i]),
                        reference_cluster=int(refassign[key][i]), cluster_agreement=int(agreement[i]),
                        cosine=float(cosine[i]), spherical_margin=float(margin[i])))
        geo = current['geometry']
        csv_write(out/f'transforms_{size}.csv', [dict(source_index=source, content_size=size,
            center_x=float(g[0]), center_y=float(g[1]), bbox=json.dumps(g[2:6].tolist()),
            central_roi=json.dumps(g[6:10].tolist()), original_width=g[10], original_height=g[11],
            resized_width=g[12], resized_height=g[13], offset_x=g[14], offset_y=g[15],
            actual_scale_x=g[16], actual_scale_y=g[17], source_pixels_retained=1.0,
            interpolation='bilinear', antialias=True, roi_grid_coverage=1.0)
            for source, g in zip(valid, geo)])
        if size == 160:
            assert np.allclose([v['cosine'] for v in summary if v['content_size'] == 160], 1, atol=1e-5)
            assert all(v['cluster_agreement'] == 1 for v in summary if v['content_size'] == 160)
    csv_write(out/'classification.csv', classification)
    csv_write(out/'summary.csv', summary)
    csv_write(out/'paired_metrics.csv', paired)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    rows = [r for r in classification if r['category'] == 'all']
    axes[0].plot([r['scale'] for r in rows], [r['accuracy'] for r in rows], 'o-', label='accuracy')
    axes[0].plot([r['scale'] for r in rows], [r['prediction_consistency'] for r in rows], 'o-', label='prediction consistency')
    for r in READOUTS:
        rows = [v for v in summary if v['stage'] == 2 and v['readout'] == r]
        axes[1].plot([v['scale'] for v in rows], [v['cluster_agreement'] for v in rows], 'o-', label=r)
        axes[2].plot([v['scale'] for v in rows], [v['cosine'] for v in rows], 'o-', label=r)
    for ax, title in zip(axes, ('Classification', 'Stage 2 agreement', 'Stage 2 cosine')):
        ax.set(xlabel='Scale relative to 160px', title=title); ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(out/'scale_curves.png', dpi=160); plt.close(fig)
    # Fixed pilot categories; no selection by whether the result improved.
    selected = [next((i for i in valid if base.categories[base.y[i]] == cat), valid[0])
                for cat in ('airplanes', 'chandelier', 'brain')]
    fig, axes = plt.subplots(3, 7, figsize=(17, 8))
    for row, source in enumerate(selected):
        image, _ = base[source]
        for col, size in enumerate(SIZES):
            _, _, _, geo = ScaleSubset(base, [source], size)[0]
            canvas = CenterLetterbox(size)(image)
            draw = ImageDraw.Draw(canvas); draw.rectangle(tuple(geo[2:6]), outline='red', width=2)
            draw.rectangle(tuple(geo[6:10]), outline='yellow', width=2)
            axes[row, col].imshow(canvas); axes[row, col].axis('off')
            cluster = assignments[(size, 's2_bbox')][valid.index(source)]
            axes[row, col].set_title(f'{size/160:.1f}x / C{cluster}')
    fig.tight_layout(); fig.savefig(out/'scale_cases.png', dpi=130); plt.close(fig)
    dump(out/'run_manifest.json', dict(identity, status='complete', completed_utc=datetime.now(timezone.utc).isoformat(),
         best_epoch=ckpt['epoch'], best_original_validation_accuracy=ckpt['valid_acc']))
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return classification, summary, paired


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--s0', type=Path, required=True); p.add_argument('--s1', type=Path, required=True)
    p.add_argument('--translation-checkpoint', type=Path)
    p.add_argument('--output-dir', type=Path, default=ROOT/'plots/kmeans/caltech_scale_1015')
    p.add_argument('--device', default='cpu'); p.add_argument('--workers', type=int, default=4)
    p.add_argument('--clusters', type=int, default=31); p.add_argument('--nredo', type=int, default=50)
    p.add_argument('--max-train', type=int); p.add_argument('--max-valid', type=int)
    args = p.parse_args()
    if args.clusters < 2 or args.nredo < 1:
        p.error('Need clusters >= 2 and nredo >= 1')
    torch.set_num_threads(4); faiss.omp_set_num_threads(4)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    base = Caltech101(str(ROOT/'data'), target_type=['category', 'annotation'], download=False)
    train, valid = split_indices(str(ROOT/'data'), 42)
    train, valid = sorted(train), sorted(valid)
    if args.max_train: train = train[:args.max_train]
    if args.max_valid: valid = valid[:args.max_valid]
    assert not set(train)&set(valid)
    results = {}
    models = [('S0', args.s0), ('S1', args.s1)]
    if args.translation_checkpoint: models.append(('T1', args.translation_checkpoint))
    for name, path in models:
        results[name] = evaluate(name, path, args, base, train, valid)
    lines = ['# 尺度穩定性：首輪比較', '',
        '完整來源影像在固定畫布置中縮放，背景一起縮放；非物件獨立縮放。',
        'S0 固定長邊 160；S1 訓練長邊 96/128/160/192。T1（若有）為平移增強參考，不替代 S0。',
        '每模型／stage／readout 獨立 fitting training-160 spherical codebook；不同模型群編號不對應。',
        f'Train {len(train)} / validation {len(valid)}；K={args.clusters}, niter=100, nredo={args.nredo}, seed=42。',
        'Stage 3 為 4×4 網格；末端 token 包含 merging padding，ROI 權重僅計 224 畫布內像素。',
        '原位 validation 用於選模；單一 training seed，屬探索性結果。', '',
        '| 模型 | 1.0 accuracy | 未見倍率平均 accuracy | 未見倍率 macro accuracy | Stage 2 bbox agreement |',
        '|---|---:|---:|---:|---:|']
    for name, (classification, summary, _) in results.items():
        orig = next(r for r in classification if r['category'] == 'all' and r['content_size'] == 160)
        unseen = [r for r in classification if r['category'] == 'all' and r['content_size'] in (112, 144, 176)]
        roi = [r for r in summary if r['stage'] == 2 and r['readout'] == 'bbox' and r['content_size'] in (112, 144, 176)]
        lines.append(f"| {name} | {orig['accuracy']:.2%} | {np.mean([r['accuracy'] for r in unseen]):.2%} | {np.mean([r['macro_accuracy'] for r in unseen]):.2%} | {np.mean([r['cluster_agreement'] for r in roi]):.2%} |")
    deltas = {}
    for metric in ('correct', 'cluster_agreement'):
        values = []
        for name in ('S0', 'S1'):
            sub = [r for r in results[name][2] if r['stage'] == 2 and r['readout'] == 'bbox' and r['content_size'] in (112, 144, 176)]
            bysource = {i: [] for i in valid}
            for r in sub: bysource[r['source_index']].append(r[metric])
            assert all(len(v) == 3 for v in bysource.values())
            values.append(np.array([np.mean(bysource[i]) for i in valid]))
        delta = values[1]-values[0]
        deltas[metric] = dict(mean=float(delta.mean()), image_bootstrap_95ci=bootstrap(delta))
    dump(args.output_dir/'primary_comparison.json', deltas)
    lines += ['', '主要差值 S1−S0：每張來源先平均三個未見倍率，再做配對影像 bootstrap（1000 次）；CI 不涵蓋 training-seed 不確定性。', '', '```json', json.dumps(deltas, indent=2), '```', '',
        '群一致率須配合群熵／最大群占比、同圖對應 cosine 與同類異圖 cosine 一起判讀；ROI pooling 的穩定不等於原始 token 不變。']
    for name in results:
        lines += ['', f'## {name}', '', f'![curves]({name}/scale_curves.png)', '', f'![cases]({name}/scale_cases.png)']
    (args.output_dir/'REPORT.md').write_text('\n'.join(lines)+'\n')
    dump(args.output_dir/'status.json', dict(status='complete', completed_utc=datetime.now(timezone.utc).isoformat()))


if __name__ == '__main__':
    main()
