#!/usr/bin/env python3
"""Analyze why stage-aligned translations change Caltech airplane predictions."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.datasets import Caltech101

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.MergingViT import MergingViT  # noqa: E402
from scripts.test_caltech_translation_kmeans import (  # noqa: E402
    LetterboxSubset,
    split_indices,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data")
    parser.add_argument(
        "--checkpoint",
        default="runs/train/mergingvit_caltech101_letterbox160/MergingViT_best.pth",
    )
    parser.add_argument(
        "--results-dir",
        default="plots/kmeans/caltech101_stage_aligned_one_patch",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


@torch.inference_mode()
def forward_outputs(model, images, zero_position=False):
    x = model.patch_embed(images)
    height, width = model.patch_embed.grid_h, model.patch_embed.grid_w
    for stage_idx in range(len(model.stages)):
        if not zero_position:
            x = x + model.pos_embeds[stage_idx]
        for block in model.stages[stage_idx]:
            x = block(x)
        if not isinstance(model.merges[stage_idx], torch.nn.Identity):
            x, height, width = model.merges[stage_idx](x, height, width)
    pooled = model.norm_final(x.mean(dim=1))
    logits = model.head(pooled)
    return pooled, logits


@torch.inference_mode()
def infer(model, dataset, batch_size, workers, device, airplane_label, zero_position=False):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
    )
    features, logits, predictions, airplane_probabilities, confidences = [], [], [], [], []
    for images, _, _, _ in loader:
        images = images.to(device, non_blocking=True)
        batch_features, batch_logits = forward_outputs(model, images, zero_position)
        probabilities = batch_logits.softmax(1)
        batch_predictions = probabilities.argmax(1)
        rows = torch.arange(len(images), device=device)
        features.append(batch_features.cpu().numpy())
        logits.append(batch_logits.cpu().numpy())
        predictions.append(batch_predictions.cpu().numpy())
        airplane_probabilities.append(probabilities[:, airplane_label].cpu().numpy())
        confidences.append(probabilities[rows, batch_predictions].cpu().numpy())
    return {
        "features": np.concatenate(features),
        "logits": np.concatenate(logits),
        "predictions": np.concatenate(predictions),
        "airplane_probability": np.concatenate(airplane_probabilities),
        "confidence": np.concatenate(confidences),
    }


def normalized_cosine(first, second):
    first = first / np.maximum(np.linalg.norm(first, axis=1, keepdims=True), 1e-12)
    second = second / np.maximum(np.linalg.norm(second, axis=1, keepdims=True), 1e-12)
    return np.sum(first * second, axis=1)


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def cluster_prediction_association(sample_metrics_path):
    with sample_metrics_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    output = []
    for stage in range(3):
        stage_rows = [row for row in rows if int(row["stage"]) == stage]
        counts = Counter(
            (row["same_cluster"] == "False", row["prediction_consistent"] == "False")
            for row in stage_rows
        )
        cluster_changed = counts[(True, True)] + counts[(True, False)]
        cluster_same = counts[(False, True)] + counts[(False, False)]
        output.append({
            "stage": stage,
            "cluster_changed_prediction_changed": counts[(True, True)],
            "cluster_changed_prediction_same": counts[(True, False)],
            "cluster_same_prediction_changed": counts[(False, True)],
            "cluster_same_prediction_same": counts[(False, False)],
            "prediction_change_rate_given_cluster_changed": counts[(True, True)] / cluster_changed,
            "prediction_change_rate_given_cluster_same": counts[(False, True)] / cluster_same,
        })
    return output


def main():
    args = parse_args()
    results_dir = ROOT / args.results_dir
    analysis_dir = results_dir / "classification_analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(ROOT / args.checkpoint, map_location="cpu", weights_only=False)
    model = MergingViT(**checkpoint["model_args"])
    model.load_state_dict(checkpoint["model_weights"], strict=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    base = Caltech101(args.data_root, target_type=["category", "annotation"], download=False)
    _, valid_indices = split_indices(args.data_root, args.seed)
    airplane_label = base.categories.index("airplanes")
    airplane_indices = [index for index in valid_indices if base.y[index] == airplane_label]
    patch_size = int(checkpoint["model_args"]["patch_size"])
    directions = {"left": (-1, 0), "right": (1, 0), "up": (0, -1), "down": (0, 1)}

    aggregate_rows = []
    transition_rows = []
    sample_rows = []
    for zero_position in (False, True):
        ablation = "standard" if not zero_position else "zero_position_embedding"
        original = infer(
            model, LetterboxSubset(base, airplane_indices), args.batch_size,
            args.workers, device, airplane_label, zero_position,
        )
        for stage in range(3):
            shift_pixels = patch_size * (2 ** stage)
            for direction, (unit_x, unit_y) in directions.items():
                dx, dy = unit_x * shift_pixels, unit_y * shift_pixels
                moved = infer(
                    model, LetterboxSubset(base, airplane_indices, dx, dy),
                    args.batch_size, args.workers, device, airplane_label, zero_position,
                )
                feature_cosine = normalized_cosine(original["features"], moved["features"])
                logit_cosine = normalized_cosine(original["logits"], moved["logits"])
                prediction_changed = original["predictions"] != moved["predictions"]
                for row_index, source_index in enumerate(airplane_indices):
                    sample_rows.append({
                        "ablation": ablation,
                        "source_index": source_index,
                        "stage": stage,
                        "direction": direction,
                        "shift_pixels": shift_pixels,
                        "original_prediction": int(original["predictions"][row_index]),
                        "moved_prediction": int(moved["predictions"][row_index]),
                        "prediction_changed": bool(prediction_changed[row_index]),
                        "final_feature_cosine": float(feature_cosine[row_index]),
                        "logit_cosine": float(logit_cosine[row_index]),
                        "original_airplane_probability": float(original["airplane_probability"][row_index]),
                        "moved_airplane_probability": float(moved["airplane_probability"][row_index]),
                        "airplane_probability_change": float(
                            moved["airplane_probability"][row_index]
                            - original["airplane_probability"][row_index]
                        ),
                    })
                aggregate_rows.append({
                    "ablation": ablation,
                    "stage": stage,
                    "direction": direction,
                    "shift_pixels": shift_pixels,
                    "samples": len(airplane_indices),
                    "original_accuracy": float(np.mean(original["predictions"] == airplane_label)),
                    "moved_accuracy": float(np.mean(moved["predictions"] == airplane_label)),
                    "prediction_consistency": float(np.mean(~prediction_changed)),
                    "final_feature_cosine": float(feature_cosine.mean()),
                    "logit_cosine": float(logit_cosine.mean()),
                    "original_airplane_probability": float(original["airplane_probability"].mean()),
                    "moved_airplane_probability": float(moved["airplane_probability"].mean()),
                    "airplane_probability_change": float(
                        np.mean(moved["airplane_probability"] - original["airplane_probability"])
                    ),
                    "changed_subset_final_feature_cosine": float(feature_cosine[prediction_changed].mean())
                    if prediction_changed.any() else float("nan"),
                    "changed_subset_airplane_probability_change": float(
                        np.mean(
                            moved["airplane_probability"][prediction_changed]
                            - original["airplane_probability"][prediction_changed]
                        )
                    ) if prediction_changed.any() else float("nan"),
                })
                wrong_predictions = moved["predictions"][moved["predictions"] != airplane_label]
                for predicted_label, count in Counter(wrong_predictions.tolist()).most_common(10):
                    transition_rows.append({
                        "ablation": ablation,
                        "stage": stage,
                        "direction": direction,
                        "shift_pixels": shift_pixels,
                        "predicted_class_index": predicted_label,
                        "predicted_class_name": base.categories[predicted_label],
                        "count": count,
                    })

    association_rows = cluster_prediction_association(results_dir / "sample_metrics.csv")
    write_csv(analysis_dir / "classification_metrics.csv", aggregate_rows)
    write_csv(analysis_dir / "classification_sample_metrics.csv", sample_rows)
    write_csv(analysis_dir / "prediction_transitions.csv", transition_rows)
    write_csv(analysis_dir / "cluster_prediction_association.csv", association_rows)
    summary = {
        "checkpoint": str(ROOT / args.checkpoint),
        "samples": len(airplane_indices),
        "airplane_class_index": airplane_label,
        "note": "zero_position_embedding is an inference-time diagnostic, not a causal architecture comparison",
    }
    (analysis_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    standard = [row for row in aggregate_rows if row["ablation"] == "standard"]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    directions_order = ["left", "right", "up", "down"]
    x = np.arange(len(directions_order))
    for stage in range(3):
        rows = [row for row in standard if row["stage"] == stage]
        rows.sort(key=lambda row: directions_order.index(row["direction"]))
        label = f"Stage {stage} ({patch_size * 2 ** stage}px)"
        axes[0].plot(x, [row["prediction_consistency"] for row in rows], marker="o", label=label)
        axes[1].plot(x, [row["final_feature_cosine"] for row in rows], marker="o", label=label)
        axes[2].plot(x, [row["moved_airplane_probability"] for row in rows], marker="o", label=label)
    for axis, title, ylabel in zip(
        axes,
        ["Prediction consistency", "Final feature cosine", "Moved airplane probability"],
        ["Ratio", "Cosine similarity", "Probability"],
    ):
        axis.set_xticks(x, directions_order)
        axis.set_ylim(0, 1.02)
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(analysis_dir / "classification_translation_analysis.png", dpi=180)
    plt.close(fig)
    print(f"wrote {analysis_dir}")


if __name__ == "__main__":
    main()
