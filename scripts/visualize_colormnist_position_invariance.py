#!/usr/bin/env python3
"""Visual test of Stage-0 token invariance across absolute positions."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mergingViT_plot_tool.Kmeans_analysis_padding_repr import (  # noqa: E402
    get_all_features_at_checkpoints,
)
from models.MergingViT import MergingViT  # noqa: E402
from scripts.test_colormnist_translation_kmeans import (  # noqa: E402
    foreground_centroids,
    load_run_config,
    translate_zero_fill,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", default="runs/train/mergingvit_colorful_mnist2")
    parser.add_argument("--train-samples", type=int, default=6000)
    parser.add_argument("--test-samples", type=int, default=1000)
    parser.add_argument("--clusters", type=int, default=31)
    parser.add_argument("--examples", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--stage", type=int, default=0)
    parser.add_argument("--shift-x", type=int, default=None)
    parser.add_argument(
        "--controlled-patch-pairs", action="store_true",
        help="Paste the same real token crop into two grid cells without object clipping.",
    )
    parser.add_argument("--output-dir", default="plots/kmeans/colorful_mnist_position_invariance")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


@torch.inference_mode()
def extract_stage_features(model, images, stage, batch_size, device):
    parts = []
    for start in range(0, len(images), batch_size):
        batch_np = images[start:start + batch_size]
        batch = torch.from_numpy(batch_np).permute(0, 3, 1, 2).float().div_(255).to(device)
        checkpoints = get_all_features_at_checkpoints(model, batch)
        feat, height, width = checkpoints[(stage, len(model.stages[stage]) - 1)]
        parts.append(feat.cpu().numpy())
    return np.concatenate(parts), height, width


def token_positions(images, receptive_size, height, width):
    centroids, visible = foreground_centroids(images)
    xs = np.clip((centroids[:, 0] // receptive_size).astype(int), 0, width - 1)
    ys = np.clip((centroids[:, 1] // receptive_size).astype(int), 0, height - 1)
    return ys * width + xs, xs, ys, visible


def token_crops(images, xs, ys, receptive_size):
    crops = []
    for image, token_x, token_y in zip(images, xs, ys):
        x0, y0 = token_x * receptive_size, token_y * receptive_size
        crop = np.zeros(
            (receptive_size, receptive_size, image.shape[-1]), dtype=image.dtype
        )
        source = image[y0:y0 + receptive_size, x0:x0 + receptive_size]
        crop[:source.shape[0], :source.shape[1]] = source
        crops.append(crop)
    return np.stack(crops)


def build_controlled_patch_pairs(images, receptive_size):
    """Place one real crop in left/right cells while respecting right/bottom padding."""
    image_height, image_width = images.shape[1:3]
    grid_height = (image_height + receptive_size - 1) // receptive_size
    grid_width = (image_width + receptive_size - 1) // receptive_size
    if grid_width < 2:
        raise ValueError("selected stage has fewer than two horizontal token positions")
    _, source_xs, source_ys, _ = token_positions(
        images, receptive_size, grid_height, grid_width
    )
    source_crops = token_crops(images, source_xs, source_ys, receptive_size)
    destination_visible_width = image_width - receptive_size
    original_pairs, shifted_pairs, source_indices = [], [], []
    for source_idx, (crop, token_y) in enumerate(zip(source_crops, source_ys)):
        # The destination cell is partially padded. Use the real visible
        # 12-pixel-wide content and apply the same zero padding to both cells,
        # producing pixel-identical controlled stimuli at two positions.
        canonical_crop = crop.copy()
        canonical_crop[:, destination_visible_width:] = 0
        if canonical_crop.max() == 0:
            continue
        y0 = int(token_y) * receptive_size
        visible_height = min(receptive_size, image_height - y0)
        left = np.zeros_like(images[0])
        right = np.zeros_like(images[0])
        left[y0:y0 + visible_height, :receptive_size] = canonical_crop[:visible_height]
        right[y0:y0 + visible_height, receptive_size:image_width] = (
            canonical_crop[:visible_height, :destination_visible_width]
        )
        original_pairs.append(left)
        shifted_pairs.append(right)
        source_indices.append(source_idx)
    return np.stack(original_pairs), np.stack(shifted_pairs), np.asarray(source_indices)


def select_rows(features, positions):
    return features[np.arange(len(features)), positions]


def add_context(axis, image, boxes):
    axis.imshow(image, interpolation="nearest")
    for x, y, size, color, label in boxes:
        axis.add_patch(Rectangle(
            (x * size - 0.5, y * size - 0.5), size, size,
            fill=False, edgecolor=color, linewidth=2, label=label,
        ))
    axis.set_xticks([])
    axis.set_yticks([])
    if boxes:
        axis.legend(loc="lower right", fontsize=6, framealpha=0.7)


def representative_indices(km, normalized_train_features, cluster, count=5):
    labels = km.labels_
    members = np.flatnonzero(labels == cluster)
    distances = km.transform(normalized_train_features[members])[:, cluster]
    return members[np.argsort(distances)[:count]]


def save_case(
    path, category, source_idx, original_image, shifted_image,
    original_crop, aligned_crop, original_xy, aligned_xy,
    receptive_size, original_cluster, aligned_cluster,
    aligned_cosine, km,
    normalized_train_features, train_images, train_xs, train_ys,
):
    # Each row is self-contained: source context -> selected crop -> its five
    # nearest cluster representatives. This keeps evidence next to its query.
    figure, axes = plt.subplots(2, 7, figsize=(14, 5.5))
    add_context(axes[0, 0], original_image, [
        (*original_xy, receptive_size, "red", "original token")
    ])
    axes[0, 0].set_title("Original image")
    axes[0, 1].imshow(original_crop, interpolation="nearest")
    axes[0, 1].set_title(f"Token crop\ncluster {original_cluster}")
    axes[0, 1].axis("off")
    add_context(axes[1, 0], shifted_image, [
        (*aligned_xy, receptive_size, "lime", "content-aligned"),
    ])
    axes[1, 0].set_title("Shifted image\ncontent-aligned")
    axes[1, 1].imshow(aligned_crop, interpolation="nearest")
    axes[1, 1].set_title(f"Token crop\ncluster {aligned_cluster}")
    axes[1, 1].axis("off")
    for row, (cluster, row_label) in enumerate([
        (original_cluster, "Original"),
        (aligned_cluster, "Content-aligned"),
    ]):
        nearest = representative_indices(km, normalized_train_features, cluster)
        for col, image_idx in enumerate(nearest):
            crop = token_crops(
                train_images[image_idx:image_idx + 1],
                train_xs[image_idx:image_idx + 1],
                train_ys[image_idx:image_idx + 1],
                receptive_size,
            )[0]
            axis = axes[row, col + 2]
            axis.imshow(crop, interpolation="nearest")
            axis.set_title(
                f"Cluster {cluster} rep {col + 1}\ntrain #{image_idx}", fontsize=8
            )
            axis.axis("off")
        axes[row, 0].set_ylabel(row_label, fontsize=10, fontweight="bold")

    figure.suptitle(
        f"{category} / test #{source_idx} / identical aligned crops\n"
        f"original cluster={original_cluster}, aligned={aligned_cluster}; "
        f"cos(aligned)={aligned_cosine:.4f}"
    )
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    run_dir = ROOT / args.checkpoint_dir
    cfg = load_run_config(run_dir)
    model_args = dict(cfg["model"]["args"])
    patch_size = int(model_args["patch_size"])
    if not 0 <= args.stage < len(model_args["depths"]):
        raise ValueError(f"stage must be in [0, {len(model_args['depths']) - 1}]")
    receptive_size = patch_size * (2 ** args.stage)
    shift_x = receptive_size if args.shift_x is None else args.shift_x
    if shift_x % receptive_size != 0:
        raise ValueError("shift-x must be an integer multiple of the selected stage patch size")

    checkpoint_path = run_dir / "MergingViT_best.pth"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = MergingViT(**model_args)
    model.load_state_dict(checkpoint["model_weights"], strict=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    data_dir = ROOT / "data" / "Colorful_MNIST"
    train_images = np.load(data_dir / "Train_images.npy")[:args.train_samples]
    test_images = np.load(data_dir / "Test_images.npy")[:args.test_samples]
    if args.controlled_patch_pairs:
        test_images, shifted_images, controlled_source_indices = (
            build_controlled_patch_pairs(test_images, receptive_size)
        )
    else:
        shifted_images = translate_zero_fill(test_images, shift_x, 0)
        controlled_source_indices = np.arange(len(test_images))

    train_all, height, width = extract_stage_features(
        model, train_images, args.stage, args.batch_size, device
    )
    original_all, _, _ = extract_stage_features(
        model, test_images, args.stage, args.batch_size, device
    )
    shifted_all, _, _ = extract_stage_features(
        model, shifted_images, args.stage, args.batch_size, device
    )

    train_pos, train_xs, train_ys, _ = token_positions(
        train_images, receptive_size, height, width
    )
    original_pos, original_xs, original_ys, original_visible = token_positions(
        test_images, receptive_size, height, width
    )
    aligned_pos, aligned_xs, aligned_ys, shifted_visible = token_positions(
        shifted_images, receptive_size, height, width
    )

    train_features = select_rows(train_all, train_pos)
    original_features = select_rows(original_all, original_pos)
    aligned_features = select_rows(shifted_all, aligned_pos)
    norm_train = normalize(train_features)
    norm_original = normalize(original_features)
    norm_aligned = normalize(aligned_features)

    km = KMeans(n_clusters=args.clusters, n_init=20, random_state=args.seed).fit(norm_train)
    original_clusters = km.predict(norm_original)
    aligned_clusters = km.predict(norm_aligned)
    aligned_cosine = np.sum(norm_original * norm_aligned, axis=1)

    original_crops = token_crops(test_images, original_xs, original_ys, receptive_size)
    aligned_crops = token_crops(shifted_images, aligned_xs, aligned_ys, receptive_size)
    identical = np.all(original_crops == aligned_crops, axis=(1, 2, 3))
    fully_visible = shifted_visible == original_visible
    moved_token = aligned_pos != original_pos
    eligible = identical & fully_visible & moved_token
    same_cluster = original_clusters == aligned_clusters

    records = []
    for idx in range(len(test_images)):
        records.append({
            "source_index": int(controlled_source_indices[idx]),
            "eligible_identical_content": bool(eligible[idx]),
            "original_token_x": int(original_xs[idx]),
            "original_token_y": int(original_ys[idx]),
            "aligned_token_x": int(aligned_xs[idx]),
            "aligned_token_y": int(aligned_ys[idx]),
            "original_cluster": int(original_clusters[idx]),
            "aligned_cluster": int(aligned_clusters[idx]),
            "aligned_same_cluster": bool(same_cluster[idx]),
            "aligned_cosine": float(aligned_cosine[idx]),
        })

    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "position_invariance_metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)

    stable_candidates = np.flatnonzero(eligible & same_cluster)
    sensitive_candidates = np.flatnonzero(eligible & ~same_cluster)
    stable_candidates = stable_candidates[np.argsort(-aligned_cosine[stable_candidates])]
    sensitive_candidates = sensitive_candidates[np.argsort(aligned_cosine[sensitive_candidates])]

    selected = {
        "stable": stable_candidates[:args.examples],
        "position_sensitive": sensitive_candidates[:args.examples],
    }
    for category, indices in selected.items():
        category_dir = output_dir / category
        category_dir.mkdir(parents=True, exist_ok=True)
        for rank, idx in enumerate(indices, start=1):
            save_case(
                category_dir / f"case_{rank:02d}_source_{controlled_source_indices[idx]}.png",
                category, int(controlled_source_indices[idx]), test_images[idx], shifted_images[idx],
                original_crops[idx], aligned_crops[idx],
                (int(original_xs[idx]), int(original_ys[idx])),
                (int(aligned_xs[idx]), int(aligned_ys[idx])),
                receptive_size, int(original_clusters[idx]), int(aligned_clusters[idx]),
                float(aligned_cosine[idx]), km, norm_train, train_images,
                train_xs, train_ys,
            )

    eligible_count = int(eligible.sum())
    print(f"device={device}, checkpoint={checkpoint_path}")
    print(f"stage={args.stage}, receptive_size={receptive_size}px, shift_x={shift_x}px")
    print(f"controlled_patch_pairs={args.controlled_patch_pairs}")
    print(f"eligible identical-content pairs={eligible_count}/{len(test_images)}")
    if eligible_count:
        print(f"aligned cluster agreement={same_cluster[eligible].mean():.4f}")
        print(f"aligned cosine={aligned_cosine[eligible].mean():.4f}")
    else:
        print("aligned cluster agreement=N/A (no eligible pairs)")
        print("aligned cosine=N/A (no eligible pairs)")
    print(f"stable examples={len(selected['stable'])}")
    print(f"position-sensitive examples={len(selected['position_sensitive'])}")
    print(f"wrote {metrics_path}")


if __name__ == "__main__":
    main()
