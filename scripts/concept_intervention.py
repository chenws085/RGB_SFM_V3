#!/usr/bin/env python3
"""Fit position-conditioned concepts on training tokens and intervene on held-out images."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.MergingViT import MergingViT
from scripts.test_caltech_translation_kmeans import LetterboxSubset, split_indices
from torchvision.datasets import Caltech101


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


@torch.inference_mode()
def capture(model, images, stage):
    captured = []
    handle = model.stages[stage][-1].register_forward_hook(
        lambda module, inputs, output: captured.append(output.detach().clone()))
    try:
        logits = model(images)
    finally:
        handle.remove()
    return logits, captured[0]


@torch.inference_mode()
def intervene(model, images, stage, mask, replacement):
    def replace(module, inputs, output):
        if mask.shape != output.shape[:2]:
            raise ValueError('Intervention mask must have shape [batch, tokens]')
        return torch.where(mask.to(output.device)[..., None], replacement.to(output), output)
    handle = model.stages[stage][-1].register_forward_hook(replace)
    try:
        return model(images)
    finally:
        handle.remove()


def unit_rows(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def parse_group(value):
    try:
        group = tuple(tuple(map(int, part.split(':'))) for part in value.split('+'))
        if not group or any(len(pair) != 2 or min(pair) < 0 for pair in group):
            raise ValueError()
        if len({p for p, c in group}) != len(group):
            raise ValueError()
        return group
    except ValueError as error:
        raise argparse.ArgumentTypeError('Use position:cluster+position:cluster; positions must be distinct') from error


def margin(logits, target):
    competitors = logits.clone()
    competitors[target] = -torch.inf
    return float(logits[target] - competitors.max())


def save_csv(path, rows):
    with path.open('w', newline='') as stream:
        if rows:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', default='runs/train/mergingvit_caltech101_letterbox160/MergingViT_best.pth')
    parser.add_argument('--data-root', default='data')
    parser.add_argument('--output-dir', default='plots/concept_intervention/stage2')
    parser.add_argument('--stage', type=int, default=2)
    parser.add_argument('--positions', type=int, nargs='+', default=[24])
    parser.add_argument('--groups', type=parse_group, nargs='+', help='Joint presence required, e.g. 24:0+25:2; default: every single concept')
    parser.add_argument('--clusters', type=int, default=31)
    parser.add_argument('--n-init', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--split-seed', type=int, default=42)
    parser.add_argument('--random-repeats', type=int, default=5)
    parser.add_argument('--replacement', choices=['mean', 'zero'], default='mean')
    parser.add_argument('--target', choices=['label', 'prediction'], default='label')
    parser.add_argument('--category', help='Evaluate only this category; training remains all-category')
    parser.add_argument('--max-train-samples', type=int)
    parser.add_argument('--max-eval-samples', type=int)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--workers', type=int, default=0)
    args = parser.parse_args()
    for name in ['clusters', 'n_init', 'random_repeats', 'batch_size', 'max_train_samples', 'max_eval_samples']:
        value = getattr(args, name)
        if value is not None and value < 1:
            parser.error(f'{name} must be positive')
    if args.workers < 0:
        parser.error('workers must be nonnegative')
    torch.set_num_threads(4)
    threadpool_limits(4)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    checkpoint_path = ROOT / args.checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model = MergingViT(**checkpoint['model_args'])
    model.load_state_dict(checkpoint['model_weights'], strict=True)
    if not 0 <= args.stage < len(model.stages):
        parser.error('stage outside model')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device).eval()
    base = Caltech101(str(ROOT / args.data_root), target_type=['category', 'annotation'], download=False)
    train_ids, eval_ids = split_indices(str(ROOT / args.data_root), args.split_seed)
    train_ids, eval_ids = list(train_ids), list(eval_ids)
    # Seeded subsampling avoids selecting a category-ordered prefix for smoke runs.
    if args.max_train_samples and len(train_ids) > args.max_train_samples:
        train_ids = sorted(rng.choice(train_ids, args.max_train_samples, replace=False).tolist())
    if args.category:
        if args.category not in base.categories:
            parser.error('unknown category')
        category = base.categories.index(args.category)
        eval_ids = [i for i in eval_ids if base.y[i] == category]
    if args.max_eval_samples and len(eval_ids) > args.max_eval_samples:
        eval_ids = sorted(rng.choice(eval_ids, args.max_eval_samples, replace=False).tolist())
    if not train_ids or not eval_ids:
        parser.error('empty training or evaluation split')
    def loader(ids):
        return DataLoader(LetterboxSubset(base, ids), batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    positions = sorted(set(args.positions))
    if args.groups:
        positions = sorted(set(positions) | {p for group in args.groups for p, c in group})
    chunks = []
    sums = None
    count = 0
    for images, labels, sources, metadata in loader(train_ids):
        _, features = capture(model, images.to(device), args.stage)
        if any(p < 0 or p >= features.shape[1] for p in positions):
            parser.error(f'positions outside token grid of size {features.shape[1]}')
        if features.shape[1] < 2:
            parser.error('at least two tokens required for spatial random control')
        chunks.append(features[:, positions].cpu().numpy())
        batch_sum = features.double().sum(0).cpu()
        sums = batch_sum if sums is None else sums + batch_sum
        count += len(images)
    training = np.concatenate(chunks)
    mean = (sums / count).float().to(device)
    centers, occupied_counts = {}, {}
    for column, position in enumerate(positions):
        x = unit_rows(training[:, column].astype(np.float64))
        k = min(args.clusters, len(x), len(np.unique(x, axis=0)))
        fit = KMeans(n_clusters=k, random_state=args.seed, n_init=args.n_init).fit(x)
        centers[position] = fit.cluster_centers_
        occupied_counts[position] = np.bincount(fit.labels_, minlength=k).tolist()
    groups = args.groups or [((p, c),) for p in positions for c in range(len(centers[p]))]
    for group in groups:
        if any(c >= len(centers[p]) for p, c in group):
            parser.error('group cluster exceeds fitted codebook size')
    groups = list(dict.fromkeys(groups))
    output = ROOT / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / 'codebooks.npz', **{f'position{p}': c for p, c in centers.items()}, training_mean=mean.cpu().numpy())
    rows = []
    for images, labels, sources, metadata in loader(eval_ids):
        images = images.to(device)
        original, features = capture(model, images, args.stage)
        assignments = {}
        for p in positions:
            x = unit_rows(features[:, p].cpu().numpy().astype(np.float64))
            distances = ((x[:, None] - centers[p][None]) ** 2).sum(-1)
            assignments[p] = distances.argmin(1)
        for group in groups:
            active = np.ones(len(images), dtype=bool)
            for p, c in group:
                active &= assignments[p] == c
            selected = np.flatnonzero(active)
            if not len(selected):
                continue
            selected_tensor = torch.as_tensor(selected, device=device)
            subset = images[selected_tensor]
            mask = torch.zeros((len(selected), features.shape[1]), dtype=torch.bool, device=device)
            mask[:, [p for p, c in group]] = True
            replacement = mean if args.replacement == 'mean' else torch.zeros_like(mean)
            changed = intervene(model, subset, args.stage, mask, replacement)
            controls = []
            control_positions = []
            for repeat in range(args.random_repeats):
                random_mask = torch.zeros_like(mask)
                chosen = [rng.choice(features.shape[1], len(group), replace=False).tolist() for _ in selected]
                for row, indices in enumerate(chosen):
                    random_mask[row, indices] = True
                controls.append(intervene(model, subset, args.stage, random_mask, replacement))
                control_positions.append(chosen)
            for row, source_row in enumerate(selected):
                before = original[source_row]
                target = int(labels[source_row]) if args.target == 'label' else int(before.argmax())
                after = changed[row]
                drop = float(before[target] - after[target])
                for repeat, control in enumerate(controls):
                    random_after = control[row]
                    random_drop = float(before[target] - random_after[target])
                    rows.append(dict(source_index=int(sources[source_row]), label=int(labels[source_row]), target=target,
                        group='+'.join(f'{p}:{c}' for p, c in group), token_count=len(group), random_repeat=repeat,
                        random_positions=json.dumps(control_positions[repeat][row]), original_prediction=int(before.argmax()),
                        intervention_prediction=int(after.argmax()), random_prediction=int(random_after.argmax()),
                        original_logit=float(before[target]), intervention_logit=float(after[target]),
                        logit_drop=drop, random_logit_drop=random_drop, effect_gap=drop-random_drop,
                        margin_drop=margin(before, target)-margin(after, target),
                        probability_drop=float(before.softmax(0)[target]-after.softmax(0)[target])))
        print(f'evaluated batch: {len(images)} images; {len(rows)} paired records', flush=True)
    save_csv(output / 'sample_effects.csv', rows)
    summary = []
    for group in groups:
        key = '+'.join(f'{p}:{c}' for p, c in group)
        matching = [r for r in rows if r['group'] == key]
        supported = sorted(set(r['source_index'] for r in matching))
        result = dict(group=key, supporting_images=len(supported), evaluated_images=len(eval_ids), support_rate=len(supported)/len(eval_ids))
        for field in ['logit_drop', 'random_logit_drop', 'effect_gap', 'margin_drop', 'probability_drop']:
            result[f'mean_{field}'] = float(np.mean([r[field] for r in matching])) if matching else None
        first_repeats = [r for r in matching if r['random_repeat'] == 0]
        result['prediction_change_rate'] = float(np.mean([r['original_prediction'] != r['intervention_prediction'] for r in first_repeats])) if matching else None
        summary.append(result)
    save_csv(output / 'concept_summary.csv', summary)
    manifest = dict(arguments=vars(args), checkpoint_sha256=sha256(checkpoint_path), model_args=checkpoint['model_args'],
        train_indices=train_ids, eval_indices=eval_ids, categories=base.categories, training_cluster_counts=occupied_counts,
        device=str(device), torch_version=torch.__version__, numpy_version=np.__version__,
        intervention_point='last block output at requested stage, before merging',
        assignment='L2-normalized tokens + Euclidean KMeans, independent per position',
        coordinate_protocol='full token grid; annotation center unused; no old feature cache',
        replacement='raw training mean per position' if args.replacement == 'mean' else 'zero',
        random_control='same stage and token count, uniformly sampled positions without replacement; may overlap concept positions',
        limitations=['validation split used for checkpoint selection, not untouched test',
                     'random control does not match position, foreground, padding or feature norms',
                     'joint group requires all constituent concepts; no causal path or interaction inference',
                     'interventions may be off-manifold; effects do not establish human semantic concepts',
                     'random repeats are not independent images; summaries conditional on concept presence'])
    (output / 'run_manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f'Saved {output}', flush=True)


if __name__ == '__main__':
    main()
