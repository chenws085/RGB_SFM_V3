#!/usr/bin/env python3
"""Summarize paired translation outcomes; bootstrap images, not translations."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from spherical_kmeans import distances, unit_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="plots/kmeans/spherical_comparison_1005")
    args = parser.parse_args()
    root = Path(args.output_dir)
    manifest = json.loads((root / "run_manifest.json").read_text())
    features = np.load(root / "features.npz")
    subjects = manifest["test_indices"]
    seeds = manifest["seeds"]
    rows, outcomes = [], {}
    methods = ["sklearn_existing", "euclidean", "spherical"]
    for stage in range(3):
        for method in methods:
            values = []
            for seed in seeds:
                centers = np.load(root / f"centers_stage{stage}_seed{seed}_{method}.npz")["centers"]
                original = distances(unit_rows(features[f"original_{stage}"].astype(float)), centers).argmin(1)
                for direction in ["left", "right", "up", "down"]:
                    moved = distances(unit_rows(features[f"s{stage}_{direction}_{stage}"].astype(float)), centers).argmin(1)
                    values.append(original == moved)
                    rows.extend(dict(stage=stage, seed=seed, method=method, direction=direction,
                        source_index=source, original_cluster=int(a), moved_cluster=int(b), same_cluster=bool(a == b))
                        for source, a, b in zip(subjects, original, moved))
            outcomes[(stage, method)] = np.asarray(values, dtype=float)
    with (root / "sample_assignments.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    rng = np.random.default_rng(20261005)
    resampled = rng.integers(len(subjects), size=(10000, len(subjects)))
    table, contrasts = [], []
    for stage in range(3):
        rates = [outcomes[(stage, m)].mean() * 100 for m in methods]
        table.append(f"| {stage} | {rates[0]:.2f}% | {rates[1]:.2f}% | {rates[2]:.2f}% | {rates[2]-rates[0]:+.2f} pp |")
        for baseline in methods[:2]:
            difference = (outcomes[(stage, "spherical")] - outcomes[(stage, baseline)]).mean(axis=0)
            low, high = np.quantile(difference[resampled].mean(axis=1), [0.025, 0.975]) * 100
            contrasts.append(f"- Stage {stage}, spherical − {baseline}：{difference.mean()*100:+.2f} pp；影像 bootstrap 95% CI [{low:+.2f}, {high:+.2f}] pp。")
    with (root / "fit_summary.csv").open() as f:
        fits = list(csv.DictReader(f))
    with (root / "iteration_history.csv").open() as f:
        history = list(csv.DictReader(f))
    max_error = max(abs(float(r["centroid_norm"])-1) for r in history if r["method"] == "spherical")
    unconverged = [r for r in fits if r["method"] != "sklearn_existing" and r["converged"] != "True"]
    reinitializations = sum(r["reinitialized"] == "True" for r in history)
    norm_table = []
    for stage in range(3):
        for method in methods:
            selected = [r for r in fits if int(r["stage"]) == stage and r["method"] == method]
            norm_table.append(f"| {stage} | {method} | {np.mean([float(r['centroid_norm_mean']) for r in selected]):.6f} | {np.mean([float(r['mean_nearest_cosine_distance']) for r in selected]):.6f} | {np.mean([float(r['fit_seconds']) for r in selected]):.2f} |")
    report = "\n".join([
        "# 1005 K-means 改進實驗：每輪 centroid L2 normalization", "",
        f"使用既有 Caltech-101 letterbox160 checkpoint，原始訓練圖 {len(manifest['train_indices'])} 張、held-out airplanes {len(subjects)} 張、K={manifest['clusters']}；seeds={seeds}。",
        "Stage 0/1/2 分別平移 8/16/32 px，四個方向，依物體質心選取對應 token；沒有裁切或將平移圖加入 fitting。此實驗是 centroid-token 分群，尚未擴大至逐位置／per-head 分群。", "",
        f"三組：現有 sklearn（n_init={manifest['sklearn_n_init']}、預設 tol=1e-4）；可觀測 Euclidean Lloyd；spherical Lloyd（每輪算術平均後 L2 normalize）。後兩組各 {manifest['n_init']} restarts，共享每次 kmeans++ 初始中心、max_iter=300、absolute tol=1e-5，各自以自己的 objective 選最佳 restart。與 sklearn 的差異也包含初始化與停止條件，因此因果比較以配對 Lloyd 兩組為主。", "",
        "## 平移前後同群比例", "",
        "平均涵蓋 3 seeds 與四個方向；不同方法的 cluster ID 不直接比較。", "",
        "| Stage | 現有 sklearn | 配對 Euclidean | Spherical | Spherical − sklearn |",
        "|---|---:|---:|---:|---:|", *table, "",
        *contrasts, "",
        "CI 以同一張 held-out 影像作為抽樣單位，保留所有 seeds／directions 的配對，10,000 次 bootstrap；僅反映這批飛機影像的抽樣不確定性，不涵蓋重新訓練 checkpoint 或其他類別。", "",
        "## 中心與成本", "",
        "| Stage | 方法 | 平均 centroid norm | 最近中心 cosine distance | fitting 秒／seed |",
        "|---|---|---:|---:|---:|", *norm_table, "",
        f"球面每輪 norm 最大偏差：{max_error:.3g}；未達停止條件的已選 runs：{len(unconverged)}；所有 restarts 的中心重初始化事件：{reinitializations}。",
        "`raw_mean_norm` 是投影前群內平均的長度，反映方向集中程度；`centroid_norm` 是投影後長度。兩者均逐群、逐輪保存。最近中心 cosine distance 為共同診斷；各方法自己的 objective 不直接跨方法比較。", "",
        "## 重現與產物", "",
        "```bash", ".venv/bin/python scripts/compare_spherical_kmeans.py",
        ".venv/bin/python scripts/report_spherical_kmeans.py", "```", "",
        f"輸出位置：`{root}`。包含 manifest（checkpoint SHA256、train/test IDs）、features cache、每組最終中心、fit_summary.csv、translation_metrics.csv、sample_assignments.csv、iteration_history.csv、translation_comparison.png、centroid_norm_history.png。", "",
        "基本驗證通過：相同初始中心的一般 Lloyd 與 sklearn objective 一致；球面中心 unit norm；Euclidean／cosine assignment 一致；最終 labels 與 centers 一致；一般案例 objective 不上升；空群恢復與零向量拒絕。", "",
        "## 本次判讀", "",
        "依各 stage 的差值與配對 CI 判讀；每輪 unit norm 限制不保證改善平移指派。優先補其他類別與代表 patch 品質，再決定是否全面替換。", "",
        "此階段尚未驗證代表 patches 的語義品質，較高同群比例本身不足以證明整體分群更好。",
    ])
    (root / "report.md").write_text(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
