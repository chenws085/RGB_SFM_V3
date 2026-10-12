#!/usr/bin/env python3
"""Draw standard per-cluster silhouettes from saved held-out sample scores."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

METHODS = ["sklearn_existing", "spherical", "faiss_native"]
TITLES = ["Existing Euclidean K-means", "Manual spherical K-means", "FAISS spherical K-means (native init)"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="plots/kmeans/quality_evaluation_1005")
    args = parser.parse_args()
    root = Path(args.input_dir)
    output = root / "silhouette_plots"
    output.mkdir(parents=True, exist_ok=True)
    grouped = defaultdict(list)
    with (root / "heldout_sample_metrics.csv").open() as f:
        for row in csv.DictReader(f):
            if row["method"] in METHODS:
                grouped[(int(row["stage"]), int(row["seed"]), row["method"])].append(row)
    keys = sorted({(stage, seed) for stage, seed, _ in grouped})
    values = np.array([float(r["cosine_silhouette"]) for rows in grouped.values() for r in rows])
    x_min = max(-1, np.floor(values.min() * 10) / 10 - .05)
    x_max = min(1, max(.5, np.ceil(values.max() * 10) / 10 + .05))
    colors = list(plt.get_cmap("tab20").colors) + list(plt.get_cmap("tab20b").colors)
    summary = []
    with PdfPages(output / "all_stages_seeds_silhouettes.pdf") as pdf:
        for stage, seed in keys:
            fig, axes = plt.subplots(1, 3, figsize=(17, 13), sharex=True)
            for axis, method, title in zip(axes, METHODS, TITLES):
                rows = grouped[(stage, seed, method)]
                clusters = defaultdict(list)
                for row in rows:
                    clusters[int(row["cluster"])].append(float(row["cosine_silhouette"]))
                scores = np.array([float(r["cosine_silhouette"]) for r in rows])
                mean = float(scores.mean())
                negative = float(np.mean(scores < 0))
                summary.append(dict(stage=stage, seed=seed, method=method, samples=len(scores),
                    clusters=len(clusters), mean_silhouette=mean, negative_fraction=negative,
                    min_silhouette=float(scores.min()), max_silhouette=float(scores.max())))
                cursor, ticks, labels = 10, [], []
                for cluster in sorted(clusters):
                    ordered = np.sort(clusters[cluster])
                    y = np.arange(cursor, cursor + len(ordered))
                    axis.fill_betweenx(y, 0, ordered, facecolor=colors[cluster % len(colors)], alpha=.88)
                    ticks.append(cursor + (len(ordered)-1)/2)
                    labels.append(f"{cluster:02d} / {len(ordered)}")
                    cursor += len(ordered) + 10
                axis.axvline(0, color="gray", linewidth=.9)
                axis.axvline(mean, color="crimson", linestyle="--", linewidth=1.5,
                             label=f"Mean = {mean:.4f}")
                axis.set_xlim(x_min, x_max)
                axis.set_ylim(0, cursor)
                axis.set_yticks(ticks, labels, fontsize=8)
                axis.set_ylabel("Cluster ID / sample count (band height = count)", fontsize=10)
                axis.set_xlabel("Cosine silhouette coefficient", fontsize=11)
                axis.set_title(f"{title}\nN = {len(scores)}, K = {len(clusters)}, negative = {negative:.1%}", fontsize=12)
                axis.grid(axis="x", alpha=.2)
                axis.set_axisbelow(True)
                axis.legend(loc="upper right", fontsize=10)
            fig.suptitle(f"Caltech-101 held-out silhouettes | Stage {stage} | Seed {seed}\n"
                         "Fixed centers selected from 50 restarts; all held-out samples; scores sorted within each cluster", fontsize=15)
            fig.text(.5, .015, "Cluster IDs and colors are local to each fitted model; matching IDs across panels do not imply matching clusters.",
                     ha="center", fontsize=10)
            fig.tight_layout(rect=(0, .035, 1, .94))
            stem = output / f"stage{stage}_seed{seed}_silhouette_comparison"
            fig.savefig(stem.with_suffix(".png"), dpi=180)
            pdf.savefig(fig)
            plt.close(fig)
            print(f"Saved {stem.name}", flush=True)
    with (output / "plot_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    metadata = dict(input=str(root / "heldout_sample_metrics.csv"), methods=METHODS,
                    panels=keys, x_limits=[x_min, x_max], score="cosine silhouette",
                    order="cluster ID ascending; score ascending within each cluster",
                    note="Each plot uses one seed; no sample scores averaged across different partitions.")
    (output / "plot_manifest.json").write_text(json.dumps(metadata, indent=2))
    lines = ["# K-means 輪廓圖（Silhouette Plot）", "",
        "使用品質評估已保存的全部 1,770 張全類別 held-out 原圖 silhouette，中心來自每個 seed 的 50 次初始化最佳結果。每張圖並排顯示現有 Euclidean K-means、手寫球面 K-means、FAISS native 球面 K-means。", "",
        "## 圖中標示", "",
        "- 橫軸：每筆樣本的 cosine silhouette，越靠右表示群內距離相對較小、最近其他群的距離相對較大。",
        "- 每個色帶：一個群；色帶高度與群內樣本數成比例，帶內樣本依 silhouette 從小到大排序。色帶之間的空白是視覺間距，並非樣本。",
        "- 縱軸標籤：cluster ID / 該群樣本數。群依 ID 排列；不同方法與 seeds 的 ID、顏色不代表相同群。",
        "- 紅色虛線：該方法、該 seed 的全體樣本平均 silhouette。它是單一 seed 平均；先前簡表是三個 seeds 平均，所以數字不一定相同。",
        "- 灰色直線：silhouette = 0；左側負值表示該樣本到另一群的平均距離比到自身群更小。",
        "- negative：負 silhouette 樣本的比例。圖中全部樣本均顯示；所有圖採相同橫軸範圍。", "",
        "每個 seed 分開畫，避免把不同分群的 cluster ID 與逐樣本 silhouette 混合成不存在的平均分群。", "",
        "## 圖片", "", "| Stage | Seed 42 | Seed 43 | Seed 44 |", "|---|---|---|---|"]
    for stage in sorted({stage for stage, _ in keys}):
        links = [f"[輪廓圖](../plots/kmeans/quality_evaluation_1005/silhouette_plots/stage{stage}_seed{seed}_silhouette_comparison.png)" for seed in [42,43,44]]
        lines.append("| " + " | ".join([str(stage)] + links) + " |")
    lines.extend(["", "[全部九張圖的 PDF](../plots/kmeans/quality_evaluation_1005/silhouette_plots/all_stages_seeds_silhouettes.pdf)", "",
        "重現：`.venv/bin/python scripts/plot_kmeans_silhouettes.py`。逐圖摘要：`plots/kmeans/quality_evaluation_1005/silhouette_plots/plot_summary.csv`。", "",
        "判讀時同時觀察平均線、負值比例、各群的厚度與形狀；silhouette 衡量方向空間中的幾何分群，不等同於部件語義或平移穩定性。",
        "資料限制：這批 held-out 圖片屬於原模型選 checkpoint 的 validation split。",
    ])
    Path("plan/1005_kmeans_silhouette_plots.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
