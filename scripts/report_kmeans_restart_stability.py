#!/usr/bin/env python3
"""Compare seed variability before/after increasing restart budgets."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def main():
    base = Path("plots/kmeans")
    old = read(base / "faiss_spherical_comparison_1005/translation_metrics.csv")
    new_root = base / "faiss_spherical_comparison_1005_n50"
    new = read(new_root / "translation_metrics.csv")
    manifest = json.loads((base / "spherical_comparison_1005_n50/run_manifest.json").read_text())
    seeds = manifest["seeds"]
    methods = ["sklearn_existing", "spherical", "faiss_native", "faiss_shared_init"]
    summaries, table = [], []
    fig, axes = plt.subplots(1,3,figsize=(14,4))
    for stage in range(3):
        for idx, method in enumerate(methods):
            per_budget = []
            for budget, rows in [("previous", old),("50",new)]:
                rates = [np.mean([float(r["same_cluster_rate"]) for r in rows
                         if int(r["stage"])==stage and int(r["seed"])==seed and r["method"]==method])*100 for seed in seeds]
                summaries.append(dict(stage=stage, method=method,budget=budget,
                    mean_percent=float(np.mean(rates)),seed_std_pp=float(np.std(rates,ddof=1)),
                    seed_min_percent=min(rates),seed_max_percent=max(rates),
                    **{f"seed_{seed}_percent": value for seed,value in zip(seeds,rates)}))
                per_budget.append((np.mean(rates),np.std(rates,ddof=1)))
                axes[stage].errorbar(idx+(-.12 if budget=="previous" else .12),np.mean(rates),
                    yerr=np.std(rates,ddof=1),fmt="o",capsize=4,
                    color="tab:blue" if budget=="previous" else "tab:orange",
                    label=budget if idx==0 else None)
            table.append(f"| {stage} | {method} | {per_budget[0][0]:.2f} ± {per_budget[0][1]:.2f} | {per_budget[1][0]:.2f} ± {per_budget[1][1]:.2f} |")
        axes[stage].set_xticks(range(4),methods,rotation=25)
        axes[stage].set_title(f"Stage {stage}")
        axes[stage].set_ylabel("Same-cluster % (mean ± seed SD)")
        axes[stage].legend()
    fig.tight_layout()
    fig.savefig(new_root / "restart_stability.png",dpi=160)
    with (new_root / "seed_stability.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    restart_rows=read(base / "spherical_comparison_1005_n50/restart_summary.csv")
    progress=[]
    for stage in range(3):
        for method in ["euclidean","spherical"]:
            for seed in seeds:
                selected=[r for r in restart_rows if int(r["stage"])==stage and r["method"]==method and int(r["seed"])==seed]
                best={budget:min(float(r["objective"]) for r in selected if int(r["restart"])<budget) for budget in [10,20,50]}
                progress.append(dict(stage=stage,method=method,seed=seed,**{f"best_objective_{b}":best[b] for b in best},
                    objective_improvement_percent=(best[10]-best[50])/best[10]*100))
    with (new_root / "objective_progress.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(progress[0]))
        writer.writeheader()
        writer.writerows(progress)
    fig, axes=plt.subplots(2,3,figsize=(13,7))
    for row,method in enumerate(["euclidean","spherical"]):
        for stage in range(3):
            for seed in seeds:
                selected=sorted([r for r in restart_rows if int(r["stage"])==stage and r["method"]==method and int(r["seed"])==seed],
                    key=lambda r:int(r["restart"]))
                curve=np.minimum.accumulate([float(r["objective"]) for r in selected])
                axes[row,stage].plot(range(1,len(curve)+1),curve/curve[0],label=f"seed {seed}")
            axes[row,stage].set_title(f"Stage {stage}: {method}")
            axes[row,stage].set_xlabel("Restarts tried")
            axes[row,stage].set_ylabel("Best objective / first objective")
            axes[row,stage].legend()
    fig.tight_layout()
    fig.savefig(new_root / "objective_restart_progress.png",dpi=160)
    plt.close(fig)
    reduced=sum(summaries[i+1]["seed_std_pp"]<summaries[i]["seed_std_pp"] for i in range(0,len(summaries),2))
    fits=read(new_root / "fit_summary.csv")
    ari=[float(r["ari_vs_manual_spherical"]) for r in fits if r["method"]=="faiss_shared_init"]
    comparisons=[]
    for stage in range(3):
        spherical=next(r for r in summaries if r["stage"]==stage and r["method"]=="spherical" and r["budget"]=="50")
        baseline=next(r for r in summaries if r["stage"]==stage and r["method"]=="sklearn_existing" and r["budget"]=="50")
        comparisons.append(f"Stage {stage} 球面 − sklearn：{spherical['mean_percent']-baseline['mean_percent']:+.2f} pp")
    text="\n".join([
        "# 1005 增加 K-means 初始化次數：3 seeds × 50 restarts", "",
        "固定 seeds 42、43、44，每個 seed／stage／方法採 50 次初始化，依 training objective 選最佳，不以測試平移穩定性選模型。",
        "沿用相同 checkpoint、6,907 張原始訓練圖、160 張 held-out airplanes、K=31、Stage 0–2 四方向平移。既有 sklearn n_init 從 20 增至 50；其他方法從 10 增至 50。手寫 Euclidean 與 spherical 共享初始中心；FAISS 同時測 native 與 shared init。",
        "共 3 seeds × 3 stages × 50 restarts × 5 methods = 2,250 次初始化擬合（含配對 Euclidean）。這是增加每個 seed 內的初始化搜尋預算，最終每組仍只有 3 個 seed 結果，不能視為 150 次獨立測試。", "",
        "## 平移同群比例與 seed 波動", "",
        "數值為百分比平均 ± 三個 seeds 的樣本標準差（pp），每個 seed 先平均四方向。", "",
        "| Stage | 方法 | 前次平均 ± SD | 50 次平均 ± SD |",
        "|---|---|---:|---:|", *table, "",
        f"12 個 stage／方法比較中，{reduced} 個的 seed SD 降低；增加 restarts 不保證每個指標的跨 seed 波動均減少，最佳 training objective 也不保證最佳 translation stability。",
        f"FAISS shared-init 與手寫 spherical 的 training ARI：{ari}。", "",
        "；".join(comparisons)+"。本次方法排名隨 stage 不同；影像 bootstrap 的各方法對 sklearn 差值 CI 均跨過 0，尚不足以宣稱球面法整體更好，先保留現行方法。", "",
        "`objective_progress.csv` 記錄相同 seed／初始化序列在 10、20、50 次時的最佳 training objective，僅在各自方法內比較。`seed_stability.csv` 保留每個 seed 的結果。影像 bootstrap CI 見同目錄 report.md；它不估計額外 seed 的抽樣不確定性。", "",
        "## 重現", "",
        "```bash",
        "mkdir -p plots/kmeans/spherical_comparison_1005_n50",
        "cp plots/kmeans/spherical_comparison_1005/features.npz plots/kmeans/spherical_comparison_1005/feature_cache_identity.json plots/kmeans/spherical_comparison_1005_n50/",
        ".venv/bin/python scripts/compare_spherical_kmeans.py --output-dir plots/kmeans/spherical_comparison_1005_n50 --n-init 50 --sklearn-n-init 50",
        ".venv/bin/python scripts/report_spherical_kmeans.py --output-dir plots/kmeans/spherical_comparison_1005_n50",
        ".venv/bin/python scripts/compare_faiss_spherical_kmeans.py --source-dir plots/kmeans/spherical_comparison_1005_n50 --output-dir plots/kmeans/faiss_spherical_comparison_1005_n50",
        ".venv/bin/python scripts/report_kmeans_restart_stability.py",
        "```", "",
        "產物位於 `plots/kmeans/spherical_comparison_1005_n50/` 與 `plots/kmeans/faiss_spherical_comparison_1005_n50/`；包含所有逐輪紀錄、逐 seed 指標、每次 restart objective、最終中心、逐圖指派、manifest 與比較圖。",
    ])
    (new_root / "restart_stability_report.md").write_text(text+"\n")
    Path("plan/1005_kmeans_50_restarts_experiment_results.md").write_text(text+"\n")
    print(text)


if __name__=="__main__":
    main()
