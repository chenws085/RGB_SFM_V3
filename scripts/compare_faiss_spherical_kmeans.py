#!/usr/bin/env python3
"""Test FAISS spherical K-means on the frozen 1005 feature cache."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
import faiss
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import kmeans_plusplus
from sklearn.metrics import adjusted_rand_score
from threadpoolctl import threadpool_limits
from spherical_kmeans import unit_rows


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", default="plots/kmeans/spherical_comparison_1005")
    parser.add_argument("--output-dir", default="plots/kmeans/faiss_spherical_comparison_1005")
    args = parser.parse_args()
    source, output = Path(args.source_dir), Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((source / "run_manifest.json").read_text())
    cache = np.load(source / "features.npz")
    k, seeds, restarts = manifest["clusters"], manifest["seeds"], manifest["n_init"]
    threadpool_limits(4)
    faiss.omp_set_num_threads(4)
    stats, fits, metrics, samples = [], [], [], []
    methods = ["sklearn_existing", "spherical", "faiss_native", "faiss_shared_init"]
    outcomes = {}
    for stage in range(3):
        x64 = unit_rows(cache[f"train_{stage}"].astype(np.float64))
        x = np.ascontiguousarray(x64, dtype=np.float32)
        faiss.normalize_L2(x)
        original = np.ascontiguousarray(unit_rows(cache[f"original_{stage}"].astype(np.float64)), dtype=np.float32)
        for seed in seeds:
            centers_by_method = {m: np.load(source / f"centers_stage{stage}_seed{seed}_{m}.npz")["centers"].astype(np.float32)
                                 for m in methods[:2]}
            for method in methods[2:]:
                best = None
                start = time.perf_counter()
                count = 1 if method == "faiss_native" else restarts
                for restart in range(count):
                    km = faiss.Kmeans(x.shape[1], k, spherical=True, niter=300,
                        nredo=restarts if method == "faiss_native" else 1, seed=seed,
                        max_points_per_centroid=len(x), verbose=False, gpu=False)
                    initial = None
                    if method == "faiss_shared_init":
                        initial, _ = kmeans_plusplus(x64, n_clusters=k, random_state=seed * 1000 + restart)
                        initial = np.ascontiguousarray(initial, dtype=np.float32)
                    km.train(x, init_centroids=initial)
                    similarity, labels = km.index.search(x, 1)
                    final_similarity = float(similarity.sum(dtype=np.float64))
                    stats.extend(dict(stage=stage, seed=seed, method=method, restart=restart,
                        iteration=i + 1, **row) for i, row in enumerate(km.iteration_stats))
                    candidate = dict(centers=km.centroids.copy(), labels=labels[:, 0],
                        similarity=final_similarity, restart=restart, stats_count=len(km.iteration_stats))
                    if best is None or final_similarity > best["similarity"]:
                        best = candidate
                    if method == "faiss_shared_init" and (restart + 1) % 10 == 0:
                        print(f"FAISS stage={stage} seed={seed} shared restarts={restart+1}/{restarts}", flush=True)
                elapsed = time.perf_counter() - start
                centers = best["centers"]
                norms = np.linalg.norm(centers, axis=1)
                assert np.allclose(norms, 1, atol=1e-6)
                reference = centers_by_method["spherical"]
                reference_labels = (x @ reference.T).argmax(axis=1)
                counts = np.bincount(best["labels"], minlength=k)
                fits.append(dict(stage=stage, seed=seed, method=method, fit_seconds=elapsed,
                    selected_restart=best["restart"], iteration_stats_count=best["stats_count"],
                    final_cosine_similarity_sum=best["similarity"],
                    final_mean_cosine_distance=1-best["similarity"]/len(x),
                    centroid_norm_min=float(norms.min()), centroid_norm_max=float(norms.max()),
                    smallest_cluster=int(counts.min()), largest_cluster=int(counts.max()),
                    ari_vs_manual_spherical=float(adjusted_rand_score(reference_labels,best["labels"]))))
                centers_by_method[method] = centers
                np.savez_compressed(output / f"centers_stage{stage}_seed{seed}_{method}.npz", centers=centers, labels=best["labels"])
            for method, centers in centers_by_method.items():
                index = faiss.IndexFlatIP(x.shape[1]) if method != "sklearn_existing" else faiss.IndexFlatL2(x.shape[1])
                index.add(np.ascontiguousarray(centers))
                _, original_labels = index.search(original, 1)
                for direction in ["left", "right", "up", "down"]:
                    moved = np.ascontiguousarray(unit_rows(cache[f"s{stage}_{direction}_{stage}"].astype(np.float64)), dtype=np.float32)
                    _, moved_labels = index.search(moved, 1)
                    equal = original_labels[:,0] == moved_labels[:,0]
                    outcomes.setdefault((stage,method), []).append(equal)
                    metrics.append(dict(stage=stage, seed=seed, method=method, direction=direction,
                        samples=len(equal), same_cluster_rate=float(equal.mean())))
                    samples.extend(dict(stage=stage, seed=seed, method=method, direction=direction,
                        source_index=source_id, original_cluster=int(a), moved_cluster=int(b), same_cluster=bool(a==b))
                        for source_id,a,b in zip(manifest["test_indices"],original_labels[:,0],moved_labels[:,0]))
            print(f"FAISS stage={stage} seed={seed} finished", flush=True)
    write_csv(output / "fit_summary.csv", fits)
    write_csv(output / "translation_metrics.csv", metrics)
    write_csv(output / "sample_assignments.csv", samples)
    write_csv(output / "faiss_iteration_stats.csv", stats)
    experiment = dict(faiss_version=faiss.__version__, numpy_version=np.__version__,
        source_manifest=manifest, source_features_sha256=hashlib.sha256((source / "features.npz").read_bytes()).hexdigest(),
        parameters=dict(spherical=True,niter=300,nredo=restarts,cpu_threads=4,gpu=False,
                        max_points_per_centroid=len(x),dtype="float32"),
        note=f"native uses FAISS random initialization; shared_init uses {restarts} external sklearn kmeans++ starts identical to manual experiment; fixed 300 iterations, no manual tolerance stopping",
        stats_note="FAISS exposes objectives/timing/imbalance, but not per-iteration centroids or raw mean norms. Final centroid norms are verified; earlier manual norm history is separate.")
    (output / "run_manifest.json").write_text(json.dumps(experiment,indent=2))
    rng = np.random.default_rng(20261005)
    bootstrap = rng.integers(len(manifest["test_indices"]),size=(10000,len(manifest["test_indices"])))
    table, contrasts = [], []
    fig, axes = plt.subplots(1,3,figsize=(14,4))
    for stage in range(3):
        values = {m: np.asarray(outcomes[(stage,m)],dtype=float) for m in methods}
        table.append("| " + " | ".join([str(stage)]+[f"{values[m].mean()*100:.2f}%" for m in methods])+" |")
        for method in methods[2:]:
            difference = (values[method]-values["sklearn_existing"]).mean(axis=0)
            low, high = np.quantile(difference[bootstrap].mean(axis=1),[.025,.975])*100
            contrasts.append(f"- Stage {stage}, {method} − sklearn：{difference.mean()*100:+.2f} pp，影像 bootstrap 95% CI [{low:+.2f}, {high:+.2f}] pp。")
        for m in methods:
            per_seed = values[m].reshape(len(seeds),4,-1).mean(axis=(1,2))
            axes[stage].errorbar(m,per_seed.mean(),yerr=per_seed.std(),marker="o",capsize=4)
        axes[stage].set_title(f"Stage {stage}")
        axes[stage].set_ylim(0,1)
        axes[stage].tick_params(axis="x",rotation=25)
    fig.tight_layout()
    fig.savefig(output / "translation_comparison.png",dpi=160)
    plt.close(fig)
    improvements = sum(np.mean(outcomes[(stage,method)]) > np.mean(outcomes[(stage,"sklearn_existing")])
                       for stage in range(3) for method in methods[2:])
    conclusion = ("FAISS native 與共享初始化兩組、三個 stages 的同群比例均未高於現有 sklearn；本次套件測試也未觀察到平移穩定性改善，先保留現行方法。"
                  if improvements == 0 else
                  f"六個 stage／FAISS 設定的比較中，{improvements} 個同群比例高於現有 sklearn；需結合配對信賴區間判讀是否支持改善。")
    report = "\n".join([
        "# 1005 FAISS 球面 K-means 套件比較", "",
        f"套件 faiss-cpu=={faiss.__version__}；沿用前次特徵：6,907 張 Caltech-101 原始訓練圖、160 張 held-out airplanes、K={k}、seeds={seeds}、Stage 0–2 四方向平移。",
        f"使用 `faiss.Kmeans(..., spherical=True, niter=300, nredo={restarts})`，CPU 4 threads，float32；max_points_per_centroid 設為訓練樣本數，確保不抽樣。",
        f"FAISS native：套件自身 random 初始化、{restarts} restarts。FAISS shared init：{restarts} 組與手寫版本完全相同的 kmeans++ 初始中心，分別跑 nredo=1，依最終 cosine objective 選最佳。FAISS 固定 300 輪，手寫版以中心變化停止；數值精度也不同，因此不要求逐 bit 一致。", "",
        "| Stage | 現有 sklearn | 手寫 spherical | FAISS native | FAISS shared init |",
        "|---|---:|---:|---:|---:|", *table, "", *contrasts, "",
        "bootstrap 以影像為單位，保留 seeds 與方向的配對；僅反映此批 held-out 飛機影像的抽樣不確定性。", "",
        conclusion + "此結果不等同於所有資料集／類別的結論。", "",
        "已檢查最終 centroid unit norm、群大小、與手寫版的 ARI；詳見 fit_summary.csv。FAISS Python API 有每輪 objective、time、imbalance 等統計，但沒有每輪 centroid／raw mean norm；不將手寫版的 norm 曲線當作套件量測。", "",
        "FAISS spherical 的每輪 objective 是內積總和，越大越好；不直接與 sklearn inertia 比較。另以最終中心重新计算 cosine distance。", "",
        "來源：[FAISS 官方 FAQ](https://github.com/facebookresearch/faiss/wiki/FAQ)、[ClusteringParameters](https://faiss.ai/cpp_api/struct/structfaiss_1_1ClusteringParameters.html)。", "",
        "```bash", ".venv/bin/python -m pip install faiss-cpu==1.12.0 --no-deps",
        ".venv/bin/python scripts/compare_faiss_spherical_kmeans.py", "```", "",
        f"產物：`{output}`，包含 manifest、fit_summary.csv、translation_metrics.csv、sample_assignments.csv、faiss_iteration_stats.csv、各組最終中心及比較圖。",
    ])
    (output / "report.md").write_text(report+"\n")
    print(report,flush=True)


if __name__ == "__main__":
    main()
