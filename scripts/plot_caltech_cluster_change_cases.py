#!/usr/bin/env python3
"""Plot deterministic cluster-change examples for Caltech Stage 0-2."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import ImageDraw
from torchvision.datasets import Caltech101

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.test_caltech_translation_kmeans import safe_letterbox, token_box  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        default="plots/kmeans/caltech101_stage_aligned_one_patch",
    )
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--representatives", type=int, default=3)
    return parser.parse_args()


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def select_cases(rows, same_cluster, count=3):
    """Select confident cases per stage while keeping source images unique."""
    selected = {}
    expected = "True" if same_cluster else "False"
    for stage in range(3):
        candidates = [
            row for row in rows
            if int(row["stage"]) == stage and row["same_cluster"] == expected
        ]
        candidates.sort(
            key=lambda row: (
                -float(row["moved_cluster_margin"]),
                -float(row["cosine"]),
                int(row["source_index"]),
                row["condition"],
            )
        )
        stage_rows = []
        used_sources = set()
        for row in candidates:
            source_index = int(row["source_index"])
            if source_index in used_sources:
                continue
            used_sources.add(source_index)
            stage_rows.append(row)
            if len(stage_rows) == count:
                break
        if len(stage_rows) < count:
            raise RuntimeError(
                f"Stage {stage} only has {len(stage_rows)} unique-source cases for same_cluster={expected}"
            )
        selected[stage] = stage_rows
    return selected


def direction_shift(row):
    return int(row["dx_pixels"]), int(row["dy_pixels"])


def context_and_crop(base, source_index, stage, dx=0, dy=0):
    image, (_, annotation) = base[int(source_index)]
    _, metadata, canvas = safe_letterbox(image, annotation, dx, dy)
    receptive_size = 8 * (2 ** stage)
    box = token_box(metadata, receptive_size)
    context = canvas.copy()
    ImageDraw.Draw(context).rectangle(box, outline="red", width=3)
    return context, canvas.crop(box), box


def representative_crops(base, representative_rows, stage, cluster, count):
    rows = sorted(
        [
            row for row in representative_rows
            if int(row["stage"]) == stage and int(row["cluster"]) == cluster
        ],
        key=lambda row: int(row["rank"]),
    )[:count]
    output = []
    receptive_size = 8 * (2 ** stage)
    for row in rows:
        source_index = int(row["source_index"])
        image, (_, annotation) = base[source_index]
        _, _, canvas = safe_letterbox(image, annotation)
        x0 = int(row["token_x"]) * receptive_size
        y0 = int(row["token_y"]) * receptive_size
        crop = canvas.crop((x0, y0, x0 + receptive_size, y0 + receptive_size))
        output.append((crop, row))
    return output


def annotate_shift(axis, dx, dy):
    """Draw the translation vector in image-pixel coordinates."""
    length = 42
    unit_x = 0 if dx == 0 else (1 if dx > 0 else -1)
    unit_y = 0 if dy == 0 else (1 if dy > 0 else -1)
    start_x = 112 - unit_x * length / 2
    start_y = 28 - unit_y * length / 2
    end_x = 112 + unit_x * length / 2
    end_y = 28 + unit_y * length / 2
    pixels = abs(dx) + abs(dy)
    axis.annotate(
        f"{pixels} px",
        xy=(end_x, end_y),
        xytext=(start_x, start_y),
        ha="center",
        va="center",
        color="red",
        fontsize=11,
        fontweight="bold",
        arrowprops={"arrowstyle": "-|>", "color": "red", "lw": 2.5},
        bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "red", "alpha": 0.9},
    )


def centroid_cosine_distance(row, original):
    key = "original_centroid_cosine_distance" if original else "moved_centroid_cosine_distance"
    return float(row[key])


def plot_case(base, representative_rows, row, output_path, count):
    stage = int(row["stage"])
    source_index = int(row["source_index"])
    original_cluster = int(row["original_cluster"])
    moved_cluster = int(row["moved_cluster"])
    dx, dy = direction_shift(row)
    original_context, original_crop, original_box = context_and_crop(
        base, source_index, stage,
    )
    moved_context, moved_crop, moved_box = context_and_crop(
        base, source_index, stage, dx, dy,
    )
    original_reps = representative_crops(
        base, representative_rows, stage, original_cluster, count,
    )
    moved_reps = representative_crops(
        base, representative_rows, stage, moved_cluster, count,
    )

    figure, axes = plt.subplots(2, 5, figsize=(15, 7.3))
    axes[0, 0].imshow(original_context)
    axes[0, 0].set_title(f"Original image (not moved)\ntoken={original_box[:2]}")
    axes[0, 1].imshow(original_crop, interpolation="nearest")
    axes[0, 1].set_title(
        f"Original patch\nCluster {original_cluster}\n"
        f"centroid cosine dist={centroid_cosine_distance(row, True):.4f}"
    )
    axes[1, 0].imshow(moved_context)
    annotate_shift(axes[1, 0], dx, dy)
    axes[1, 0].set_title(f"Moved image ({row['condition']})\ntoken={moved_box[:2]}")
    axes[1, 1].imshow(moved_crop, interpolation="nearest")
    axes[1, 1].set_title(
        f"Moved patch\nCluster {moved_cluster}\n"
        f"centroid cosine dist={centroid_cosine_distance(row, False):.4f}"
    )
    for column in range(3):
        for plot_row, (items, cluster, label) in enumerate((
            (original_reps, original_cluster, "Before"),
            (moved_reps, moved_cluster, "After"),
        )):
            axis = axes[plot_row, column + 2]
            if column < len(items):
                crop, representative = items[column]
                axis.imshow(crop, interpolation="nearest")
                axis.set_title(
                    f"{label} C{cluster} repr {column + 1}\n"
                    f"source #{representative['source_index']}",
                    fontsize=9,
                )
    for axis in axes.ravel():
        axis.axis("off")
    axes[0, 0].set_ylabel("NOT MOVED", fontsize=12, fontweight="bold")
    axes[1, 0].set_ylabel("MOVED", fontsize=12, fontweight="bold")

    status = "UNCHANGED" if original_cluster == moved_cluster else "CHANGED"
    arrow = "=" if original_cluster == moved_cluster else "→"
    figure.suptitle(
        f"Stage {stage} cluster {status} — source #{source_index}, "
        f"C{original_cluster} {arrow} C{moved_cluster}, cosine={float(row['cosine']):.4f}, "
        f"margin={float(row['moved_cluster_margin']):.4f}"
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_combined(base, representative_rows, selected, output_path):
    figure, axes = plt.subplots(6, 5, figsize=(14, 15.5))
    for stage, row in enumerate(selected):
        source_index = int(row["source_index"])
        original_cluster = int(row["original_cluster"])
        moved_cluster = int(row["moved_cluster"])
        dx, dy = direction_shift(row)
        original_context, original_crop, _ = context_and_crop(base, source_index, stage)
        moved_context, moved_crop, _ = context_and_crop(base, source_index, stage, dx, dy)
        original_reps = representative_crops(base, representative_rows, stage, original_cluster, 3)
        moved_reps = representative_crops(base, representative_rows, stage, moved_cluster, 3)
        top_row, bottom_row = stage * 2, stage * 2 + 1
        axes[top_row, 0].imshow(original_context)
        axes[top_row, 0].set_title(f"Original #{source_index} (not moved)", fontsize=9)
        axes[top_row, 1].imshow(original_crop, interpolation="nearest")
        axes[top_row, 1].set_title(
            f"Patch C{original_cluster}\ncos dist={centroid_cosine_distance(row, True):.4f}",
            fontsize=9,
        )
        axes[bottom_row, 0].imshow(moved_context)
        annotate_shift(axes[bottom_row, 0], dx, dy)
        axes[bottom_row, 0].set_title(f"Moved {row['condition']}", fontsize=9)
        axes[bottom_row, 1].imshow(moved_crop, interpolation="nearest")
        axes[bottom_row, 1].set_title(
            f"Patch C{moved_cluster}\ncos dist={centroid_cosine_distance(row, False):.4f}",
            fontsize=9,
        )
        for column, (crop, representative) in enumerate(original_reps, start=2):
            axes[top_row, column].imshow(crop, interpolation="nearest")
            axes[top_row, column].set_title(
                f"C{original_cluster} repr #{representative['rank']}", fontsize=9
            )
        for column, (crop, representative) in enumerate(moved_reps, start=2):
            axes[bottom_row, column].imshow(crop, interpolation="nearest")
            axes[bottom_row, column].set_title(
                f"C{moved_cluster} repr #{representative['rank']}", fontsize=9
            )
        axes[top_row, 0].set_ylabel(
            f"Stage {stage}\nNOT MOVED\nC{original_cluster}", fontsize=10, fontweight="bold"
        )
        axes[bottom_row, 0].set_ylabel(
            f"Stage {stage}\nMOVED\nC{moved_cluster}\ncos={float(row['cosine']):.3f}",
            fontsize=10,
            fontweight="bold",
        )
    for axis in axes.ravel():
        axis.axis("off")
    figure.suptitle("Stage-aligned one-token translations that change KMeans cluster")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_case_group(base, representative_rows, rows, output_path, status):
    """Create one six-row sheet: three cases, before row then moved row."""
    figure, axes = plt.subplots(6, 5, figsize=(14, 15.5))
    stage = int(rows[0]["stage"])
    for case_index, row in enumerate(rows):
        source_index = int(row["source_index"])
        original_cluster = int(row["original_cluster"])
        moved_cluster = int(row["moved_cluster"])
        dx, dy = direction_shift(row)
        original_context, original_crop, _ = context_and_crop(base, source_index, stage)
        moved_context, moved_crop, _ = context_and_crop(base, source_index, stage, dx, dy)
        original_reps = representative_crops(base, representative_rows, stage, original_cluster, 3)
        moved_reps = representative_crops(base, representative_rows, stage, moved_cluster, 3)
        top_row, bottom_row = case_index * 2, case_index * 2 + 1
        axes[top_row, 0].imshow(original_context)
        axes[top_row, 0].set_title(f"Case {case_index + 1}: original #{source_index}", fontsize=9)
        axes[top_row, 1].imshow(original_crop, interpolation="nearest")
        axes[top_row, 1].set_title(
            f"Patch C{original_cluster}\ncos dist={centroid_cosine_distance(row, True):.4f}",
            fontsize=9,
        )
        axes[bottom_row, 0].imshow(moved_context)
        annotate_shift(axes[bottom_row, 0], dx, dy)
        axes[bottom_row, 0].set_title(f"Moved {row['condition']}", fontsize=9)
        axes[bottom_row, 1].imshow(moved_crop, interpolation="nearest")
        axes[bottom_row, 1].set_title(
            f"Patch C{moved_cluster}\ncos dist={centroid_cosine_distance(row, False):.4f}",
            fontsize=9,
        )
        for column, (crop, representative) in enumerate(original_reps, start=2):
            axes[top_row, column].imshow(crop, interpolation="nearest")
            axes[top_row, column].set_title(
                f"C{original_cluster} repr #{representative['rank']}", fontsize=9
            )
        for column, (crop, representative) in enumerate(moved_reps, start=2):
            axes[bottom_row, column].imshow(crop, interpolation="nearest")
            axes[bottom_row, column].set_title(
                f"C{moved_cluster} repr #{representative['rank']}", fontsize=9
            )
        axes[top_row, 0].set_ylabel(
            f"Case {case_index + 1}\nNOT MOVED\nC{original_cluster}",
            fontsize=10,
            fontweight="bold",
        )
        axes[bottom_row, 0].set_ylabel(
            f"MOVED\nC{moved_cluster}\ncos={float(row['cosine']):.3f}",
            fontsize=10,
            fontweight="bold",
        )
    for axis in axes.ravel():
        axis.axis("off")
    figure.suptitle(f"Stage {stage}: three cluster-{status} cases")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main():
    args = parse_args()
    results_dir = ROOT / args.results_dir
    case_dir = results_dir / "cluster_change_cases"
    case_dir.mkdir(parents=True, exist_ok=True)
    sample_rows = read_csv(results_dir / "sample_metrics.csv")
    representative_rows = read_csv(results_dir / "representatives" / "representatives.csv")
    changed = select_cases(sample_rows, same_cluster=False, count=3)
    unchanged = select_cases(sample_rows, same_cluster=True, count=3)
    base = Caltech101(args.data_root, target_type=["category", "annotation"], download=False)
    indexed_rows = []
    for status, groups in (("changed", changed), ("unchanged", unchanged)):
        for stage, rows in groups.items():
            stage_dir = case_dir / f"stage{stage}" / status
            stage_dir.mkdir(parents=True, exist_ok=True)
            for rank, row in enumerate(rows, start=1):
                plot_case(
                    base, representative_rows, row,
                    stage_dir / f"case_{rank:02d}_source_{row['source_index']}.png",
                    args.representatives,
                )
                indexed_rows.append({"case_status": status, "case_rank": rank, **row})
            plot_case_group(
                base, representative_rows, rows,
                case_dir / f"stage{stage}_{status}_three_cases.png",
                status,
            )

    # Keep the original three-stage overview using the first changed case per stage.
    plot_combined(
        base, representative_rows, [changed[stage][0] for stage in range(3)],
        case_dir / "stage0-2_cluster_changes.png",
    )
    with (case_dir / "selected_cases.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=indexed_rows[0].keys())
        writer.writeheader()
        writer.writerows(indexed_rows)
    for row in indexed_rows:
        print(
            f"{row['case_status']} rank={row['case_rank']} stage={row['stage']} "
            f"source={row['source_index']} {row['condition']} "
            f"{row['original_cluster']}->{row['moved_cluster']} "
            f"cosine={float(row['cosine']):.4f} margin={float(row['moved_cluster_margin']):.4f}"
        )
    print(f"wrote {case_dir}")


if __name__ == "__main__":
    main()
