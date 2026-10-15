#!/usr/bin/env python3
"""Corrected static object-center codebooks, then phase-controlled translations.

Legacy safe_letterbox is intentionally unchanged: only the annotation passed to
it changes. Existing experiments/caches keep their original interpretation.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, silhouette_samples
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits
import torch
from torch.utils.data import Dataset

from test_caltech_translation_kmeans import (
    ROOT, Caltech101, MergingViT, safe_letterbox, extract, split_indices,
    token_touches_padding, CANVAS_RGB,
)

COORDINATE_VERSION = 'caltech_box_offset_minus1_v1'
DIRECTIONS = {'left': (-1, 0), 'right': (1, 0), 'up': (0, -1), 'down': (0, 1)}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(path, rows):
    with Path(path).open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def absolute_annotation(base, source, annotation):
    category = base.y[source]
    path = Path(base.root) / 'Annotations' / base.annotation_categories[category] / f'annotation_{base.index[source]:04d}.mat'
    box = loadmat(path)['box_coord'].ravel()
    return np.asarray(annotation, dtype=np.float64) + np.array([[box[2]], [box[0]]]) - 1


class CorrectedSubset(Dataset):
    def __init__(self, base, indices, dx=0, dy=0):
        self.base, self.indices, self.dx, self.dy = base, list(indices), dx, dy

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        source = self.indices[item]
        image, (label, annotation) = self.base[source]
        corrected = absolute_annotation(self.base, source, annotation)
        tensor, metadata, _ = safe_letterbox(image, corrected, self.dx, self.dy)
        _, legacy, _ = safe_letterbox(image, annotation, self.dx, self.dy)
        metadata.update(legacy_center_x=legacy['center_x'], legacy_center_y=legacy['center_y'])
        if not (0 <= metadata['center_x'] < 224 and 0 <= metadata['center_y'] < 224):
            raise ValueError(f'Corrected center outside canvas: {source}')
        return tensor, label, source, metadata


def save_result(path, result):
    arrays = {f'stage{s}': x for s, x in enumerate(result['features'])}
    arrays.update({k: result[k] for k in ['predictions', 'confidences', 'labels', 'source_indices']})
    arrays['metadata_json'] = np.array(json.dumps(result['metadata']))
    np.savez_compressed(path, **arrays)


def read_result(path):
    with np.load(path) as raw:
        return dict(features=[raw[f'stage{s}'] for s in range(4)],
                    metadata=json.loads(str(raw['metadata_json'])),
                    **{k: raw[k] for k in ['predictions', 'confidences', 'labels', 'source_indices']})


def position(meta, size):
    return int(meta['center_x'] // size), int(meta['center_y'] // size)


def boundary_distance(meta, size):
    x, y = meta['center_x'] % size, meta['center_y'] % size
    return min(x, size-x, y, size-y)


def aggregate(rows, stage, shift, direction, stratum):
    return dict(stage=stage, shift_pixels=shift, direction=direction, stratum=stratum, samples=len(rows),
                cluster_agreement=float(np.mean([r['same_cluster'] for r in rows])),
                matched_token_cosine=float(np.mean([r['cosine'] for r in rows])),
                prediction_consistency=float(np.mean([r['prediction_consistent'] for r in rows])),
                classification_accuracy=float(np.mean([r['moved_correct'] for r in rows])),
                padding_contact_rate=float(np.mean([r['padding_contact'] for r in rows])),
                token_changed_rate=float(np.mean([r['token_changed'] for r in rows])),
                ami=float(adjusted_mutual_info_score([r['original_cluster'] for r in rows], [r['moved_cluster'] for r in rows])),
                ari=float(adjusted_rand_score([r['original_cluster'] for r in rows], [r['moved_cluster'] for r in rows])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', default='plots/kmeans/caltech_corrected_center_1012')
    parser.add_argument('--checkpoint', default='runs/train/mergingvit_caltech101_letterbox160/MergingViT_best.pth')
    parser.add_argument('--data-root', default='data')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--clusters', type=int, default=31)
    parser.add_argument('--n-init', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--boundary-px', type=float, default=2.)
    parser.add_argument('--max-train', type=int)
    parser.add_argument('--max-valid', type=int)
    args = parser.parse_args()
    torch.set_num_threads(4)
    threadpool_limits(4)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out = ROOT / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    checkpoint_path = ROOT / args.checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model = MergingViT(**checkpoint['model_args'])
    model.load_state_dict(checkpoint['model_weights'], strict=True)
    patch = checkpoint['model_args']['patch_size']
    assert patch == 8
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device).eval()
    base = Caltech101(str(ROOT / args.data_root), target_type=['category', 'annotation'], download=False)
    train_ids, valid_ids = split_indices(str(ROOT / args.data_root), args.seed)
    # Preserve the previous all-class held-out ordering for direct source-ID comparisons.
    valid_ids = sorted(valid_ids)
    if args.max_train:
        train_ids = train_ids[:args.max_train]
    if args.max_valid:
        valid_ids = valid_ids[:args.max_valid]
    assert not set(train_ids) & set(valid_ids)
    airplane = base.categories.index('airplanes')
    test_ids = [i for i in valid_ids if base.y[i] == airplane]
    if not test_ids:
        raise ValueError('No validation airplanes selected')
    annotation_hash = hashlib.sha256()
    # Fingerprint annotations and source pixels, with source IDs delimiting records.
    image_hash = hashlib.sha256()
    for source in sorted(train_ids + valid_ids):
        category = base.y[source]
        anno = Path(base.root) / 'Annotations' / base.annotation_categories[category] / f'annotation_{base.index[source]:04d}.mat'
        img = Path(base.root) / '101_ObjectCategories' / base.categories[category] / f'image_{base.index[source]:04d}.jpg'
        annotation_hash.update(str(source).encode() + anno.read_bytes())
        image_hash.update(str(source).encode() + img.read_bytes())
    identity = dict(coordinate_version=COORDINATE_VERSION, checkpoint_sha256=sha(checkpoint_path),
                    annotations_sha256=annotation_hash.hexdigest(), images_sha256=image_hash.hexdigest(),
                    runner_sha256=sha(__file__), legacy_preprocess_sha256=sha(ROOT / 'scripts/test_caltech_translation_kmeans.py'),
                    seed=args.seed, train_indices=train_ids, valid_indices=valid_ids,
                    canvas=224, content_max=160, canvas_rgb=CANVAS_RGB)
    identity_path = out / 'cache_identity.json'
    # JSON conversion makes tuple/list comparison stable on restart.
    identity = json.loads(json.dumps(identity))
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise RuntimeError('Cache identity mismatch; use a new output directory')
    identity_path.write_text(json.dumps(identity, indent=2))
    manifest = dict(identity, checkpoint=str(checkpoint_path), model_args=checkpoint['model_args'],
                    epoch=checkpoint.get('epoch'), arguments=vars(args), command=sys.argv,
                    started_utc=datetime.now(timezone.utc).isoformat(), status='running_static',
                    device=str(device), python=sys.version, torch=torch.__version__, numpy=np.__version__,
                    codebook_method='L2 normalized features + sklearn Euclidean KMeans',
                    shifts=[0, 4, 8, 16, 32], translation_categories=['airplanes'], translation_indices=test_ids,
                    boundary_definition='min distance to any token grid edge <= boundary_px; paired near if either endpoint near',
                    center_definition='mean of contour vertices; not area centroid',
                    coordinate_formula='obj_contour + [[box_coord[2]], [box_coord[0]]] - 1; legacy nominal scale retained')
    manifest_path = out / 'run_manifest.json'
    def status(value):
        manifest['status'] = value
        manifest_path.write_text(json.dumps(manifest, indent=2))
    status('running_static')

    def cached(name, ids, dx=0, dy=0):
        path = out / f'{name}_features.npz'
        if path.exists():
            result = read_result(path)
        else:
            print(f'Extracting {name}: N={len(ids)}, dx={dx}, dy={dy}', flush=True)
            result = extract(model, CorrectedSubset(base, ids, dx, dy), args.batch_size, args.workers, device, patch)
            save_result(path, result)
        assert np.array_equal(result['source_indices'], ids)
        return result

    # Prove that correcting annotation leaves model input pixels unchanged.
    checks = []
    for source in train_ids + valid_ids:
        image, (_, annotation) = base[source]
        old_tensor, old_meta, old_canvas = safe_letterbox(image, annotation)
        new_tensor, new_meta, new_canvas = safe_letterbox(image, absolute_annotation(base, source, annotation))
        assert torch.equal(old_tensor, new_tensor) and old_canvas.tobytes() == new_canvas.tobytes()
        checks.append(dict(source_index=source, pixel_identical=True))
    write_csv(out / 'pixel_invariance_checks.csv', checks)
    train = cached('train', train_ids)
    valid = cached('validation', valid_ids)
    static, assignments, audit, centers = [], [], [], []
    for split, result in [('train', train), ('validation', valid)]:
        for source, meta in zip(result['source_indices'], result['metadata']):
            for stage in range(3):
                size = patch * 2**stage
                x, y = position(meta, size)
                legacy_x, legacy_y = int(meta['legacy_center_x']//size), int(meta['legacy_center_y']//size)
                audit.append(dict(split=split, source_index=int(source), stage=stage,
                                  center_x=meta['center_x'], center_y=meta['center_y'],
                                  legacy_center_x=meta['legacy_center_x'], legacy_center_y=meta['legacy_center_y'],
                                  token_x=x, token_y=y, legacy_token_x=legacy_x, legacy_token_y=legacy_y,
                                  same_token=(x,y)==(legacy_x,legacy_y),
                                  boundary_distance=boundary_distance(meta,size),
                                  padding_contact=token_touches_padding(meta,meta['center_x'],meta['center_y'],size)))
    for stage in range(3):
        x = normalize(train['features'][stage])
        kmeans = KMeans(n_clusters=args.clusters, n_init=args.n_init, random_state=args.seed).fit(x)
        centers.append(kmeans)
        np.savez_compressed(out / f'codebook_stage{stage}.npz', centers=kmeans.cluster_centers_, train_labels=kmeans.labels_,
                            coordinate_version=COORDINATE_VERSION)
        for split, result in [('train', train), ('validation', valid)]:
            features = normalize(result['features'][stage])
            labels = kmeans.predict(features)
            sil = silhouette_samples(features, labels, metric='cosine') if 1 < len(np.unique(labels)) < len(labels) else np.full(len(labels), np.nan)
            counts = np.bincount(labels, minlength=args.clusters)
            static.append(dict(stage=stage, split=split, samples=len(labels), cosine_silhouette=float(np.nanmean(sil)),
                               negative_silhouette_fraction=float(np.mean(sil<0)), used_clusters=int(sum(counts>0)),
                               category_ami=float(adjusted_mutual_info_score(result['labels'],labels)),
                               category_ari=float(adjusted_rand_score(result['labels'],labels)),
                               classification_accuracy=float(np.mean(result['predictions']==result['labels'])),
                               kmeans_inertia=float(kmeans.inertia_), kmeans_iterations=int(kmeans.n_iter_)))
            assignments.extend(dict(stage=stage, split=split, source_index=int(source), category=int(category),
                                    cluster=int(cluster), cosine_silhouette=float(score))
                               for source, category, cluster, score in zip(result['source_indices'],result['labels'],labels,sil))
        print(f'Fitted/evaluated static stage {stage}: {static[-1]}', flush=True)
    write_csv(out / 'static_metrics.csv', static)
    write_csv(out / 'static_assignments.csv', assignments)
    write_csv(out / 'coordinate_audit.csv', audit)
    status('static_complete_running_translation')
    valid_lookup = {int(source): i for i, source in enumerate(valid['source_indices'])}
    offsets = [valid_lookup[source] for source in test_ids]
    original = dict(features=[x[offsets] for x in valid['features']],
                    metadata=[valid['metadata'][i] for i in offsets],
                    **{k:valid[k][offsets] for k in ['predictions','confidences','labels','source_indices']})
    sample_rows, aggregates = [], []
    for shift in [0, 4, 8, 16, 32]:
        for direction, (ux,uy) in ({'original':(0,0)} if shift==0 else DIRECTIONS).items():
            moved = original if shift==0 else cached(f'translation_{shift}_{direction}', test_ids, ux*shift, uy*shift)
            for stage in range(3):
                size = patch * 2**stage
                before, after = normalize(original['features'][stage]), normalize(moved['features'][stage])
                km = centers[stage]
                a, b = km.predict(before), km.predict(after)
                normalized_centers = normalize(km.cluster_centers_)
                d = np.sort(km.transform(after),axis=1)
                rows = []
                for i, source in enumerate(test_ids):
                    old, new = original['metadata'][i], moved['metadata'][i]
                    assert np.isclose(new['center_x']-old['center_x'],ux*shift)
                    assert np.isclose(new['center_y']-old['center_y'],uy*shift)
                    ox,oy = position(old,size)
                    nx,ny = position(new,size)
                    if shift % size == 0:
                        assert nx-ox == ux*shift//size and ny-oy == uy*shift//size
                    original_padding = token_touches_padding(old,old['center_x'],old['center_y'],size)
                    padding = token_touches_padding(new,new['center_x'],new['center_y'],size)
                    row = dict(source_index=source, stage=stage, shift_pixels=shift, direction=direction,
                               dx_pixels=ux*shift, dy_pixels=uy*shift, stage_grid_pixels=size,
                               stage_grid_aligned=shift%size==0, initial_patch_aligned=shift%8==0,
                               original_x=old['center_x'], original_y=old['center_y'], moved_x=new['center_x'], moved_y=new['center_y'],
                               original_token_x=ox, original_token_y=oy, moved_token_x=nx, moved_token_y=ny,
                               token_changed=(ox,oy)!=(nx,ny), token_dx=nx-ox, token_dy=ny-oy,
                               original_boundary_distance=boundary_distance(old,size), moved_boundary_distance=boundary_distance(new,size),
                               boundary_stratum='near' if min(boundary_distance(old,size),boundary_distance(new,size))<=args.boundary_px else 'interior',
                               original_padding_contact=original_padding, padding_contact=padding,
                               pair_padding_contact=original_padding or padding, source_pixels_retained=new['source_pixels_retained'],
                               original_cluster=int(a[i]), moved_cluster=int(b[i]), same_cluster=bool(a[i]==b[i]),
                               cosine=float(before[i]@after[i]), moved_cluster_margin=float(d[i,1]-d[i,0]),
                               original_centroid_cosine_distance=float(1-before[i]@normalized_centers[a[i]]),
                               moved_to_original_centroid_cosine_distance=float(1-after[i]@normalized_centers[a[i]]),
                               moved_centroid_cosine_distance=float(1-after[i]@normalized_centers[b[i]]),
                               original_prediction=int(original['predictions'][i]), moved_prediction=int(moved['predictions'][i]),
                               prediction_consistent=bool(original['predictions'][i]==moved['predictions'][i]),
                               moved_correct=bool(moved['predictions'][i]==airplane))
                    for grid in [8,16,32]:
                        for name, meta in [('original',old),('moved',new)]:
                            row[f'{name}_phase_x_mod{grid}']=meta['center_x']%grid
                            row[f'{name}_phase_y_mod{grid}']=meta['center_y']%grid
                    rows.append(row)
                sample_rows.extend(rows)
                selections = {'all':rows}
                for boundary in ['near','interior']:
                    selections[f'boundary_{boundary}']=[r for r in rows if r['boundary_stratum']==boundary]
                    for contact in [False,True]:
                        selections[f'boundary_{boundary}_padding_{contact}']=[r for r in rows if r['boundary_stratum']==boundary and r['pair_padding_contact']==contact]
                for contact in [False,True]:
                    selections[f'pair_padding_{contact}']=[r for r in rows if r['pair_padding_contact']==contact]
                for name, selected in selections.items():
                    if selected:
                        aggregates.append(aggregate(selected,stage,shift,direction,name))
            write_csv(out / 'translation_samples.csv',sample_rows)
            write_csv(out / 'translation_summary.csv',aggregates)
            print(f'Finished translation {shift}px {direction}', flush=True)
    fig,axes=plt.subplots(3,3,figsize=(13,11),sharex=True)
    for stage in range(3):
        for col,(metric,title) in enumerate([('cluster_agreement','Cluster agreement'),('matched_token_cosine','Token cosine'),('prediction_consistency','Prediction consistency')]):
            for direction in DIRECTIONS:
                selected=[r for r in aggregates if r['stage']==stage and r['stratum']=='all' and r['direction']==direction]
                axes[stage,col].plot([0]+[r['shift_pixels'] for r in selected],[1]+[r[metric] for r in selected],marker='o',label=direction)
            axes[stage,col].set_title(f'Stage {stage}: {title}')
            axes[stage,col].set_xticks([0,4,8,16,32])
            axes[stage,col].grid(alpha=.3)
            axes[stage,col].set_ylim(0,1.02)
    axes[0,-1].legend()
    for ax in axes[-1]:
        ax.set_xlabel('Translation (canvas px)')
    fig.tight_layout()
    fig.savefig(out/'translation_stability.png',dpi=160)
    plt.close(fig)
    manifest['completed_utc']=datetime.now(timezone.utc).isoformat()
    manifest['artifacts_sha256']={p.name:sha(p) for p in out.iterdir() if p.suffix in ['.npz','.csv','.png']}
    status('complete')
    print(f'Complete: {out}', flush=True)


if __name__=='__main__':
    main()
