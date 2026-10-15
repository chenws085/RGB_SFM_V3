#!/usr/bin/env python3
"""Stage-aligned one-token Caltech-101 translation/KMeans experiment."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
from sklearn.preprocessing import normalize
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import Caltech101
from torchvision.transforms import functional as TF

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dataloader.Caltech101 import Caltech101Dataset  # noqa: E402
from mergingViT_plot_tool.Kmeans_analysis_padding_repr import (  # noqa: E402
    get_all_features_at_checkpoints,
)
from models.MergingViT import MergingViT  # noqa: E402


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CANVAS_SIZE = 224
CONTENT_MAX_SIZE = 160  # 32 px safe margin for Stage 0-2 shifts (8/16/32 px).
CANVAS_RGB = tuple(round(value * 255) for value in IMAGENET_MEAN)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--checkpoint", default="runs/train/mergingvit_caltech101_letterbox160/MergingViT_best.pth")
    parser.add_argument("--output-dir", default="plots/kmeans/caltech101_stage_aligned_one_patch")
    parser.add_argument("--clusters", type=int, default=31)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-airplanes", type=int, default=None)
    return parser.parse_args()


def safe_letterbox(image, annotation, dx=0, dy=0):
    image = image.convert("RGB")
    width, height = image.size
    scale = min(CONTENT_MAX_SIZE / width, CONTENT_MAX_SIZE / height)
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    resized = TF.resize(image, [resized_height, resized_width], antialias=True)
    base_x = (CANVAS_SIZE - resized_width) // 2
    base_y = (CANVAS_SIZE - resized_height) // 2
    x0, y0 = base_x + dx, base_y + dy
    x1, y1 = x0 + resized_width, y0 + resized_height
    if x0 < 0 or y0 < 0 or x1 > CANVAS_SIZE or y1 > CANVAS_SIZE:
        raise ValueError("Translation exceeds the safe letterbox margin")
    canvas = Image.new("RGB", (CANVAS_SIZE, CANVAS_SIZE), CANVAS_RGB)
    canvas.paste(resized, (x0, y0))
    annotation = np.asarray(annotation, dtype=np.float64)
    center_x = float(annotation[0].mean() * scale + x0)
    center_y = float(annotation[1].mean() * scale + y0)
    tensor = TF.normalize(TF.to_tensor(canvas), IMAGENET_MEAN, IMAGENET_STD)
    metadata = {
        "center_x": center_x,
        "center_y": center_y,
        "content_x0": x0,
        "content_y0": y0,
        "content_x1": x1,
        "content_y1": y1,
        "resized_width": resized_width,
        "resized_height": resized_height,
        "source_pixels_retained": 1.0,
    }
    return tensor, metadata, canvas


class LetterboxSubset(Dataset):
    def __init__(self, base, indices, dx=0, dy=0):
        self.base = base
        self.indices = list(indices)
        self.dx = dx
        self.dy = dy

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        source_index = self.indices[item]
        image, (label, annotation) = self.base[source_index]
        tensor, metadata, _ = safe_letterbox(image, annotation, self.dx, self.dy)
        return tensor, label, source_index, metadata


def split_indices(root, seed):
    wrapper_train = Caltech101Dataset(root, train=True, transform=None, seed=seed)
    wrapper_valid = Caltech101Dataset(root, train=False, transform=None, seed=seed)
    return wrapper_train.indices, wrapper_valid.indices


@torch.inference_mode()
def extract(model, dataset, batch_size, workers, device, patch_size):
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
        pin_memory=True, persistent_workers=workers > 0,
    )
    stage_features = [[] for _ in model.stages]
    predictions, confidences, labels, source_indices = [], [], [], []
    metadata_rows = []
    for images, batch_labels, batch_indices, metadata in loader:
        images = images.to(device, non_blocking=True)
        checkpoints = get_all_features_at_checkpoints(model, images)
        logits = model(images)
        probabilities = logits.softmax(1)
        batch_predictions = probabilities.argmax(1)
        rows = torch.arange(len(images), device=device)
        predictions.append(batch_predictions.cpu().numpy())
        confidences.append(probabilities[rows, batch_predictions].cpu().numpy())
        labels.append(batch_labels.numpy())
        source_indices.append(batch_indices.numpy())

        centers_x = metadata["center_x"].numpy()
        centers_y = metadata["center_y"].numpy()
        for stage_idx in range(len(model.stages)):
            features, height, width = checkpoints[(stage_idx, len(model.stages[stage_idx]) - 1)]
            receptive_size = patch_size * (2 ** stage_idx)
            xs = np.clip((centers_x // receptive_size).astype(int), 0, width - 1)
            ys = np.clip((centers_y // receptive_size).astype(int), 0, height - 1)
            positions = torch.as_tensor(ys * width + xs, device=device)
            stage_features[stage_idx].append(features[rows, positions].cpu().numpy())

        for row_idx in range(len(images)):
            metadata_rows.append({key: float(value[row_idx]) for key, value in metadata.items()})
    return {
        "features": [np.concatenate(parts) for parts in stage_features],
        "predictions": np.concatenate(predictions),
        "confidences": np.concatenate(confidences),
        "labels": np.concatenate(labels),
        "source_indices": np.concatenate(source_indices),
        "metadata": metadata_rows,
    }


def token_touches_padding(metadata, center_x, center_y, receptive_size):
    token_x0 = int(center_x // receptive_size) * receptive_size
    token_y0 = int(center_y // receptive_size) * receptive_size
    token_x1, token_y1 = token_x0 + receptive_size, token_y0 + receptive_size
    return not (
        token_x0 >= metadata["content_x0"]
        and token_y0 >= metadata["content_y0"]
        and token_x1 <= metadata["content_x1"]
        and token_y1 <= metadata["content_y1"]
    )


def save_montage(base, source_indices, output_path, shift_pixels):
    selected = source_indices[: min(6, len(source_indices))]
    figure, axes = plt.subplots(len(selected), 5, figsize=(13, 2.7 * len(selected)))
    if len(selected) == 1:
        axes = axes[None, :]
    conditions = [(0, 0), (-shift_pixels, 0), (shift_pixels, 0), (0, -shift_pixels), (0, shift_pixels)]
    titles = ["original", "left", "right", "up", "down"]
    for row, source_index in enumerate(selected):
        image, (_, annotation) = base[int(source_index)]
        for column, ((dx, dy), title) in enumerate(zip(conditions, titles)):
            _, meta, canvas = safe_letterbox(image, annotation, dx, dy)
            draw = ImageDraw.Draw(canvas)
            cx, cy = meta["center_x"], meta["center_y"]
            draw.ellipse((cx - 3, cy - 3, cx + 3, cy + 3), outline="red", width=2)
            axes[row, column].imshow(canvas)
            axes[row, column].axis("off")
            if row == 0:
                axes[row, column].set_title(title)
            if column == 0:
                axes[row, column].set_ylabel(f"source {source_index}")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def save_metric_plot(aggregate_rows, output_path):
    directions = ["left", "right", "up", "down"]
    figure, axes = plt.subplots(1, 3, figsize=(14, 4), sharex=True)
    metric_specs = [
        ("cluster_agreement", "Cluster agreement"),
        ("matched_token_cosine", "Matched-token cosine"),
        ("prediction_consistency", "Prediction consistency"),
    ]
    for axis, (metric, title) in zip(axes, metric_specs):
        for direction in directions:
            rows = sorted(
                [row for row in aggregate_rows if row["condition"] == direction],
                key=lambda row: row["stage"],
            )
            axis.plot(
                [row["stage"] for row in rows],
                [row[metric] for row in rows],
                marker="o",
                label=direction,
            )
        axis.set_title(title)
        axis.set_xlabel("Stage")
        axis.set_xticks([0, 1, 2, 3])
        axis.set_ylim(-0.05, 1.05)
        axis.grid(alpha=0.3)
    axes[0].set_ylabel("Score")
    axes[-1].legend()
    figure.suptitle("Caltech-101 airplanes: one stage-aligned token translation")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def representative_canvas(base, source_index):
    image, (_, annotation) = base[int(source_index)]
    _, metadata, canvas = safe_letterbox(image, annotation)
    return metadata, canvas


def token_box(metadata, receptive_size):
    x0 = int(metadata["center_x"] // receptive_size) * receptive_size
    y0 = int(metadata["center_y"] // receptive_size) * receptive_size
    return x0, y0, x0 + receptive_size, y0 + receptive_size


def save_cluster_representatives(
    output_dir, base, train_result, kmeans_models, patch_size, top_k=5,
):
    root = output_dir / "representatives"
    index_rows = []
    for stage_idx in range(3):
        stage_dir = root / f"stage{stage_idx}"
        stage_dir.mkdir(parents=True, exist_ok=True)
        features = normalize(train_result["features"][stage_idx])
        kmeans = kmeans_models[stage_idx]
        assignments = kmeans.predict(features)
        distances = kmeans.transform(features)
        receptive_size = patch_size * (2 ** stage_idx)
        overview, overview_axes = plt.subplots(6, 6, figsize=(13, 13))
        overview_axes = overview_axes.ravel()

        for cluster_idx in range(kmeans.n_clusters):
            members = np.flatnonzero(assignments == cluster_idx)
            nearest = members[np.argsort(distances[members, cluster_idx])[:top_k]]
            figure, axes = plt.subplots(2, top_k, figsize=(2.4 * top_k, 5.0))
            if top_k == 1:
                axes = axes[:, None]
            for rank in range(top_k):
                axes[0, rank].axis("off")
                axes[1, rank].axis("off")
                if rank >= len(nearest):
                    continue
                feature_index = int(nearest[rank])
                source_index = int(train_result["source_indices"][feature_index])
                metadata, canvas = representative_canvas(base, source_index)
                box = token_box(metadata, receptive_size)
                crop = canvas.crop(box)
                context = canvas.copy()
                ImageDraw.Draw(context).rectangle(box, outline="red", width=3)
                distance = float(distances[feature_index, cluster_idx])
                class_idx = int(base.y[source_index])
                class_name = base.categories[class_idx]
                axes[0, rank].imshow(crop, interpolation="nearest")
                axes[0, rank].set_title(
                    f"#{source_index} {class_name}\nd={distance:.3f}", fontsize=8
                )
                axes[1, rank].imshow(context)
                if rank == 0:
                    overview_axes[cluster_idx].imshow(crop, interpolation="nearest")
                    overview_axes[cluster_idx].set_title(f"C{cluster_idx:02d}", fontsize=9)
                    overview_axes[cluster_idx].axis("off")
                index_rows.append({
                    "stage": stage_idx,
                    "cluster": cluster_idx,
                    "rank": rank + 1,
                    "source_index": source_index,
                    "class_index": class_idx,
                    "class_name": class_name,
                    "distance_to_centroid": distance,
                    "receptive_field_pixels": receptive_size,
                    "token_x": box[0] // receptive_size,
                    "token_y": box[1] // receptive_size,
                })
            axes[0, 0].set_ylabel("token crop")
            axes[1, 0].set_ylabel("source context")
            figure.suptitle(
                f"Stage {stage_idx} / Cluster {cluster_idx}: nearest {top_k} "
                f"(RF={receptive_size}×{receptive_size}px)"
            )
            figure.tight_layout()
            figure.savefig(stage_dir / f"cluster_{cluster_idx:02d}.png", dpi=160)
            plt.close(figure)

        for axis in overview_axes[kmeans.n_clusters:]:
            axis.axis("off")
        overview.suptitle(
            f"Stage {stage_idx}: KMeans cluster medoids (RF={receptive_size}×{receptive_size}px)"
        )
        overview.tight_layout()
        overview.savefig(root / f"stage{stage_idx}_overview.png", dpi=180)
        plt.close(overview)

    with (root / "representatives.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=index_rows[0].keys())
        writer.writeheader()
        writer.writerows(index_rows)
    return root


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = ROOT / args.checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_args = checkpoint["model_args"]
    patch_size = int(model_args["patch_size"])
    stage_shifts = {stage: patch_size * (2 ** stage) for stage in range(3)}
    direction_vectors = {
        "left": (-1, 0), "right": (1, 0), "up": (0, -1), "down": (0, 1),
    }
    model = MergingViT(**model_args)
    model.load_state_dict(checkpoint["model_weights"], strict=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    base = Caltech101(args.data_root, target_type=["category", "annotation"], download=False)
    train_indices, valid_indices = split_indices(args.data_root, args.seed)
    if args.max_train_samples:
        train_indices = train_indices[:args.max_train_samples]
    airplane_label = base.categories.index("airplanes")
    airplane_indices = [index for index in valid_indices if base.y[index] == airplane_label]
    if args.max_airplanes:
        airplane_indices = airplane_indices[:args.max_airplanes]
    if not airplane_indices:
        raise RuntimeError("No held-out airplanes were found")

    print(f"device={device} patch_size={patch_size} train={len(train_indices)} airplanes={len(airplane_indices)}")
    train_result = extract(
        model, LetterboxSubset(base, train_indices), args.batch_size, args.workers,
        device, patch_size,
    )
    kmeans_models = []
    for stage_idx, features in enumerate(train_result["features"]):
        kmeans = KMeans(n_clusters=args.clusters, n_init=20, random_state=args.seed)
        kmeans.fit(normalize(features))
        kmeans_models.append(kmeans)
        print(f"fit stage={stage_idx} shape={features.shape}")

    representative_root = save_cluster_representatives(
        output_dir, base, train_result, kmeans_models, patch_size,
    )
    print(f"saved representatives to {representative_root}")

    original = extract(
        model, LetterboxSubset(base, airplane_indices), args.batch_size,
        args.workers, device, patch_size,
    )
    results = {}
    for stage_idx, shift_pixels in stage_shifts.items():
        for direction, (unit_x, unit_y) in direction_vectors.items():
            dx, dy = unit_x * shift_pixels, unit_y * shift_pixels
            results[(stage_idx, direction)] = extract(
                model, LetterboxSubset(base, airplane_indices, dx, dy), args.batch_size,
                args.workers, device, patch_size,
            )
            print(f"extracted stage={stage_idx} {direction} dx={dx} dy={dy}")

    sample_rows = []
    aggregate_rows = []
    for stage_idx, shift_pixels in stage_shifts.items():
        kmeans = kmeans_models[stage_idx]
        for condition, (unit_x, unit_y) in direction_vectors.items():
            dx, dy = unit_x * shift_pixels, unit_y * shift_pixels
            moved = results[(stage_idx, condition)]
            original_features = normalize(original["features"][stage_idx])
            moved_features = normalize(moved["features"][stage_idx])
            original_assignments = kmeans.predict(original_features)
            moved_assignments = kmeans.predict(moved_features)
            original_distances = kmeans.transform(original_features)
            moved_distances = kmeans.transform(moved_features)
            normalized_centers = normalize(kmeans.cluster_centers_)
            original_centroid_cosine = np.sum(
                original_features * normalized_centers[original_assignments], axis=1
            )
            moved_centroid_cosine = np.sum(
                moved_features * normalized_centers[moved_assignments], axis=1
            )
            moved_to_original_centroid_cosine = np.sum(
                moved_features * normalized_centers[original_assignments], axis=1
            )
            original_centroid_cosine_distance = 1.0 - original_centroid_cosine
            moved_centroid_cosine_distance = 1.0 - moved_centroid_cosine
            moved_to_original_centroid_cosine_distance = 1.0 - moved_to_original_centroid_cosine
            sample_indices = np.arange(len(original_assignments))
            original_to_original_centroid_euclidean = original_distances[
                sample_indices, original_assignments
            ]
            moved_to_original_centroid_euclidean = moved_distances[
                sample_indices, original_assignments
            ]
            moved_to_assigned_centroid_euclidean = moved_distances[
                sample_indices, moved_assignments
            ]
            cosine = np.sum(original_features * moved_features, axis=1)
            euclidean = np.linalg.norm(original_features - moved_features, axis=1)
            sorted_moved_distances = np.sort(moved_distances, axis=1)
            margins = sorted_moved_distances[:, 1] - sorted_moved_distances[:, 0]
            receptive_size = patch_size * (2 ** stage_idx)
            padding_contacts = []
            for metadata in moved["metadata"]:
                padding_contacts.append(token_touches_padding(
                    metadata, metadata["center_x"], metadata["center_y"], receptive_size
                ))
            padding_contacts = np.asarray(padding_contacts)

            for sample_idx, source_index in enumerate(airplane_indices):
                sample_rows.append({
                    "source_index": source_index,
                    "class_name": "airplanes",
                    "condition": condition,
                    "dx_pixels": dx,
                    "dy_pixels": dy,
                    "stage_patch_pixels": shift_pixels,
                    "stage_token_displacement": 1.0,
                    "stage": stage_idx,
                    "source_pixels_retained": moved["metadata"][sample_idx]["source_pixels_retained"],
                    "padding_contact": bool(padding_contacts[sample_idx]),
                    "original_cluster": int(original_assignments[sample_idx]),
                    "moved_cluster": int(moved_assignments[sample_idx]),
                    "same_cluster": bool(original_assignments[sample_idx] == moved_assignments[sample_idx]),
                    "cosine": float(cosine[sample_idx]),
                    "original_centroid_cosine": float(original_centroid_cosine[sample_idx]),
                    "moved_centroid_cosine": float(moved_centroid_cosine[sample_idx]),
                    "original_centroid_cosine_distance": float(original_centroid_cosine_distance[sample_idx]),
                    "moved_centroid_cosine_distance": float(moved_centroid_cosine_distance[sample_idx]),
                    "moved_to_original_centroid_cosine_distance": float(moved_to_original_centroid_cosine_distance[sample_idx]),
                    "original_to_original_centroid_euclidean": float(original_to_original_centroid_euclidean[sample_idx]),
                    "moved_to_original_centroid_euclidean": float(moved_to_original_centroid_euclidean[sample_idx]),
                    "moved_to_assigned_centroid_euclidean": float(moved_to_assigned_centroid_euclidean[sample_idx]),
                    "normalized_euclidean": float(euclidean[sample_idx]),
                    "moved_cluster_margin": float(margins[sample_idx]),
                    "original_prediction": int(original["predictions"][sample_idx]),
                    "moved_prediction": int(moved["predictions"][sample_idx]),
                    "prediction_consistent": bool(original["predictions"][sample_idx] == moved["predictions"][sample_idx]),
                    "moved_correct": bool(moved["predictions"][sample_idx] == airplane_label),
                    "confidence_change": float(moved["confidences"][sample_idx] - original["confidences"][sample_idx]),
                })
            aggregate_rows.append({
                "condition": condition,
                "dx_pixels": dx,
                "dy_pixels": dy,
                "stage_patch_pixels": shift_pixels,
                "stage_token_displacement": 1.0,
                "stage": stage_idx,
                "samples": len(airplane_indices),
                "cluster_agreement": float(np.mean(original_assignments == moved_assignments)),
                "ami": float(adjusted_mutual_info_score(original_assignments, moved_assignments)),
                "ari": float(adjusted_rand_score(original_assignments, moved_assignments)),
                "matched_token_cosine": float(cosine.mean()),
                "original_centroid_cosine_distance": float(original_centroid_cosine_distance.mean()),
                "moved_centroid_cosine_distance": float(moved_centroid_cosine_distance.mean()),
                "moved_to_original_centroid_cosine_distance": float(moved_to_original_centroid_cosine_distance.mean()),
                "original_to_original_centroid_euclidean": float(original_to_original_centroid_euclidean.mean()),
                "moved_to_original_centroid_euclidean": float(moved_to_original_centroid_euclidean.mean()),
                "moved_to_assigned_centroid_euclidean": float(moved_to_assigned_centroid_euclidean.mean()),
                "normalized_euclidean": float(euclidean.mean()),
                "moved_cluster_margin": float(margins.mean()),
                "padding_contact_rate": float(padding_contacts.mean()),
                "prediction_consistency": float(np.mean(original["predictions"] == moved["predictions"])),
                "classification_accuracy": float(np.mean(moved["predictions"] == airplane_label)),
                "confidence_change": float(np.mean(moved["confidences"] - original["confidences"])),
            })

    for filename, rows in (("sample_metrics.csv", sample_rows), ("aggregate_metrics.csv", aggregate_rows)):
        with (output_dir / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    for stage_idx, shift_pixels in stage_shifts.items():
        save_montage(
            base, airplane_indices,
            output_dir / f"airplanes_stage{stage_idx}_{shift_pixels}px_montage.png",
            shift_pixels,
        )
    save_metric_plot(aggregate_rows, output_dir / "translation_stability.png")
    manifest = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_valid_accuracy": checkpoint.get("valid_acc"),
        "checkpoint_airplanes_valid_accuracy": checkpoint.get("airplanes_valid_acc"),
        "device": str(device),
        "seed": args.seed,
        "patch_size": patch_size,
        "stage_shifts_pixels": stage_shifts,
        "directions": direction_vectors,
        "excluded_stage": 3,
        "excluded_stage_reason": "one Stage-3 token requires 64 px, beyond the common 32 px safe margin",
        "canvas_size": CANVAS_SIZE,
        "content_max_size": CONTENT_MAX_SIZE,
        "canvas_rgb": CANVAS_RGB,
        "kmeans_fit_split": "original Caltech-101 training images, safe-letterbox, no translations",
        "kmeans_train_samples": len(train_indices),
        "held_out_airplanes": len(airplane_indices),
        "clusters": args.clusters,
        "representatives": str(representative_root),
        "representatives_per_cluster": 5,
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote results to {output_dir}")


if __name__ == "__main__":
    main()
