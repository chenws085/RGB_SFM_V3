#!/usr/bin/env python3
"""Reuse the Caltech safe-letterbox protocol and compare centroid updates."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.cluster import KMeans, kmeans_plusplus
from sklearn.metrics import adjusted_rand_score
from threadpoolctl import threadpool_limits

from spherical_kmeans import distances, fit_lloyd, unit_rows
from test_caltech_translation_kmeans import (
    ROOT, Caltech101, LetterboxSubset, MergingViT, extract, split_indices,
)


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="plots/kmeans/spherical_comparison_1005")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--n-init", type=int, default=10)
    parser.add_argument("--sklearn-n-init", type=int, default=20)
    parser.add_argument("--clusters", type=int, default=31)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-airplanes", type=int)
    args = parser.parse_args()
    threadpool_limits(4)
    torch.set_num_threads(4)
    output = ROOT / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = ROOT / "runs/train/mergingvit_caltech101_letterbox160/MergingViT_best.pth"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    base = Caltech101(str(ROOT / "data"), target_type=["category", "annotation"], download=False)
    train, valid = split_indices(str(ROOT / "data"), 42)
    train = train[:args.max_train_samples] if args.max_train_samples else train
    airplanes = [i for i in valid if base.y[i] == base.categories.index("airplanes")]
    airplanes = airplanes[:args.max_airplanes] if args.max_airplanes else airplanes
    patch_size = int(checkpoint["model_args"]["patch_size"])
    manifest = dict(checkpoint=str(checkpoint_path), checkpoint_sha256=hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
                    train_indices=train, test_indices=airplanes, split_seed=42,
                    seeds=args.seeds, n_init=args.n_init, clusters=args.clusters,
                    max_iter=300, paired_absolute_tol=1e-5,
                    sklearn_n_init=args.sklearn_n_init, sklearn_tol=1e-4,
                    protocol="original training only; 160-in-224 safe letterbox; centroid token; stages 0-2",
                    note="Paired restarts share normalized kmeans++ centers; each method selects its own best objective.")
    cache_identity = {k: manifest[k] for k in ["checkpoint_sha256", "train_indices", "test_indices", "protocol"]}
    identity_path = output / "feature_cache_identity.json"
    cache = output / "features.npz"
    if cache.exists() and identity_path.exists() and json.loads(identity_path.read_text()) == cache_identity:
        features = dict(np.load(cache))
        print("reused feature cache", flush=True)
    else:
        model = MergingViT(**checkpoint["model_args"])
        model.load_state_dict(checkpoint["model_weights"], strict=True)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device).eval()
        features = {}
        conditions = [("train", train, 0, 0), ("original", airplanes, 0, 0)]
        for stage in range(3):
            shift = patch_size * 2 ** stage
            for direction, dx, dy in [("left", -shift, 0), ("right", shift, 0), ("up", 0, -shift), ("down", 0, shift)]:
                conditions.append((f"s{stage}_{direction}", airplanes, dx, dy))
        for name, indices, dx, dy in conditions:
            result = extract(model, LetterboxSubset(base, indices, dx, dy), 32, 4, device, patch_size)
            for stage in range(3):
                features[f"{name}_{stage}"] = result["features"][stage]
            print(f"extracted {name}: {len(indices)} images", flush=True)
        np.savez_compressed(cache, **features)
        identity_path.write_text(json.dumps(cache_identity))
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    summaries, metrics, history, samples, restart_summaries = [], [], [], [], []
    for stage in range(3):
        x = unit_rows(features[f"train_{stage}"].astype(np.float64))
        original = unit_rows(features[f"original_{stage}"].astype(np.float64))
        for seed in args.seeds:
            best = {}
            elapsed = {"euclidean": 0., "spherical": 0.}
            for restart in range(args.n_init):
                initial, _ = kmeans_plusplus(x, n_clusters=args.clusters, random_state=seed * 1000 + restart)
                for method in elapsed:
                    start = time.perf_counter()
                    fit = fit_lloyd(x, initial, spherical=method == "spherical")
                    elapsed[method] += time.perf_counter() - start
                    restart_summaries.append(dict(stage=stage, seed=seed, restart=restart,
                        method=method, objective=fit["objective"], iterations=fit["iterations"],
                        converged=fit["converged"]))
                    history.extend(dict(stage=stage, seed=seed, restart=restart, method=method, **row) for row in fit["history"])
                    if method not in best or fit["objective"] < best[method]["objective"]:
                        best[method] = dict(fit, restart=restart)
                if (restart + 1) % 10 == 0:
                    print(f"stage={stage} seed={seed} restarts={restart+1}/{args.n_init}", flush=True)
            start = time.perf_counter()
            baseline = KMeans(n_clusters=args.clusters, n_init=args.sklearn_n_init, random_state=seed).fit(x)
            elapsed["sklearn_existing"] = time.perf_counter() - start
            best["sklearn_existing"] = dict(centers=baseline.cluster_centers_, labels=baseline.labels_,
                objective=baseline.inertia_, iterations=int(baseline.n_iter_), converged=None, restart=-1)
            for method, fit in best.items():
                centers = fit["centers"]
                counts = np.bincount(fit["labels"], minlength=args.clusters)
                norms = np.linalg.norm(centers, axis=1)
                cosine_distances = 1 - x @ unit_rows(centers).T
                summaries.append(dict(stage=stage, seed=seed, method=method, selected_restart=fit["restart"],
                    iterations=fit["iterations"], converged=fit["converged"], fit_seconds=elapsed[method],
                    objective=fit["objective"], mean_nearest_cosine_distance=float(cosine_distances.min(axis=1).mean()),
                    centroid_norm_min=float(norms.min()), centroid_norm_mean=float(norms.mean()),
                    centroid_norm_max=float(norms.max()), smallest_cluster=int(counts.min()), largest_cluster=int(counts.max()),
                    ari_vs_sklearn=float(adjusted_rand_score(baseline.labels_, fit["labels"]))))
                original_labels = distances(original, centers).argmin(axis=1)
                for direction in ["left", "right", "up", "down"]:
                    moved = unit_rows(features[f"s{stage}_{direction}_{stage}"].astype(np.float64))
                    moved_labels = distances(moved, centers).argmin(axis=1)
                    similarities = np.sort(moved @ unit_rows(centers).T, axis=1)
                    metrics.append(dict(stage=stage, seed=seed, method=method, direction=direction,
                        samples=len(moved), same_cluster_rate=float(np.mean(original_labels == moved_labels)),
                        mean_cosine_margin=float(np.mean(similarities[:, -1] - similarities[:, -2]))))
                    samples.extend(dict(stage=stage, seed=seed, method=method, direction=direction,
                        source_index=int(source), original_cluster=int(a), moved_cluster=int(b), same_cluster=bool(a == b))
                        for source, a, b in zip(airplanes, original_labels, moved_labels))
                np.savez_compressed(output / f"centers_stage{stage}_seed{seed}_{method}.npz", centers=centers, labels=fit["labels"])
            print(f"fit stage={stage} seed={seed} complete", flush=True)
    write_csv(output / "fit_summary.csv", summaries)
    write_csv(output / "translation_metrics.csv", metrics)
    write_csv(output / "iteration_history.csv", history)
    write_csv(output / "sample_assignments.csv", samples)
    write_csv(output / "restart_summary.csv", restart_summaries)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for stage, axis in enumerate(axes):
        for method in ["sklearn_existing", "euclidean", "spherical"]:
            values = [np.mean([r["same_cluster_rate"] for r in metrics if r["stage"] == stage and r["seed"] == seed and r["method"] == method]) for seed in args.seeds]
            axis.errorbar(method, np.mean(values), yerr=np.std(values), marker="o", capsize=5)
        axis.set_title(f"Stage {stage}")
        axis.set_ylim(0, 1)
        axis.tick_params(axis="x", rotation=20)
        axis.set_ylabel("Translation same-cluster rate")
    fig.tight_layout()
    fig.savefig(output / "translation_comparison.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for stage in range(3):
        for method in ["euclidean", "spherical"]:
            seed = args.seeds[0]
            selected = next(r["selected_restart"] for r in summaries if r["stage"] == stage and r["seed"] == seed and r["method"] == method)
            rows = [r for r in history if r["stage"] == stage and r["seed"] == seed and r["method"] == method and r["restart"] == selected]
            iterations = sorted(set(r["iteration"] for r in rows))
            for axis, field in [(axes[0, stage], "raw_mean_norm"), (axes[1, stage], "centroid_norm")]:
                axis.plot(iterations, [np.mean([r[field] for r in rows if r["iteration"] == i]) for i in iterations], label=method)
                axis.set_title(f"Stage {stage}: {field}")
                axis.set_xlabel("Iteration")
                axis.legend()
    fig.tight_layout()
    fig.savefig(output / "centroid_norm_history.png", dpi=160)
    plt.close(fig)
    print(f"Saved results to {output}", flush=True)


if __name__ == "__main__":
    main()
