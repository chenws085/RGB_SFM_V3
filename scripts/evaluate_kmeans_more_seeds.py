#!/usr/bin/env python3
"""Fixed 20-seed replication on legacy features; never stop on significance."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
import csv, json, hashlib
from pathlib import Path
import numpy as np
import faiss
from scipy.stats import ttest_1samp, norm
from sklearn.cluster import KMeans, kmeans_plusplus
from threadpoolctl import threadpool_limits
from spherical_kmeans import unit_rows, distances, fit_lloyd
from evaluate_kmeans_quality import query_silhouette

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'plots/kmeans/quality_evaluation_1005_n20seeds'
SOURCE=ROOT/'plots/kmeans/spherical_comparison_1005_n50'
LIB=ROOT/'plots/kmeans/faiss_spherical_comparison_1005_n50'
HELD=ROOT/'plots/kmeans/quality_evaluation_1005/heldout_features.npz'
METHODS=['sklearn_existing','spherical','faiss_native']
SEEDS=list(range(42,62))
def write_csv(path, rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    threadpool_limits(4);faiss.omp_set_num_threads(4)
    manifest=dict(seeds=SEEDS,n_init=50,k=31,max_iter=300,legacy_coordinates=True,
        source_sha256=hashlib.sha256((SOURCE/'features.npz').read_bytes()).hexdigest(),
        heldout_sha256=hashlib.sha256(HELD.read_bytes()).hexdigest(),
        policy='Fixed 20 seeds, all stages/methods; no significance-based stopping; reused pilot seeds 42-44, exploratory inference.',
        inference='paired seed t-test with Holm correction across six comparisons; crossed seed/image bootstrap, frozen pairwise distances')
    (OUT/'run_manifest.json').write_text(json.dumps(manifest,indent=2))
    train=np.load(SOURCE/'features.npz');held=np.load(HELD)
    scores={};rows=[]
    for stage in range(3):
        x=unit_rows(train[f'train_{stage}'].astype(np.float64))
        y=unit_rows(held[f'stage{stage}'].astype(np.float64))
        d=np.clip(1-y@y.T,0,2);np.fill_diagonal(d,0)
        for seed in SEEDS:
            for method in METHODS:
                target=OUT/f'centers_stage{stage}_seed{seed}_{method}.npz'
                old=(LIB if method=='faiss_native' else SOURCE)/target.name
                if target.exists(): centers=np.load(target)['centers']
                elif old.exists():
                    centers=np.load(old)['centers'];np.savez_compressed(target,centers=centers)
                else:
                    if method=='sklearn_existing':
                        centers=KMeans(n_clusters=31,n_init=50,random_state=seed).fit(x).cluster_centers_
                    elif method=='spherical':
                        best=None
                        for restart in range(50):
                            initial,_=kmeans_plusplus(x,n_clusters=31,random_state=seed*1000+restart)
                            fit=fit_lloyd(x,initial,spherical=True)
                            if best is None or fit['objective']<best['objective']:best=fit
                        centers=best['centers']
                    else:
                        xf=np.ascontiguousarray(x,dtype=np.float32);faiss.normalize_L2(xf)
                        km=faiss.Kmeans(x.shape[1],31,spherical=True,niter=300,nredo=50,seed=seed,
                            max_points_per_centroid=len(x),verbose=False,gpu=False)
                        km.train(xf);centers=km.centroids.copy()
                    np.savez_compressed(target,centers=centers)
                labels=distances(y,centers).argmin(axis=1)
                s=query_silhouette(d,labels,np.arange(len(y)))
                scores[stage,seed,method]=s
                rows.append(dict(stage=stage,seed=seed,method=method,silhouette=float(s.mean()),negative_fraction=float((s<0).mean())))
            write_csv(OUT/'seed_metrics.csv',rows)
            print(f'stage={stage} seed={seed} complete',flush=True)
    rng=np.random.default_rng(20261006);contrasts=[]
    for stage in range(3):
        base=np.stack([scores[stage,seed,'sklearn_existing'] for seed in SEEDS])
        for method in METHODS[1:]:
            delta=np.stack([scores[stage,seed,method] for seed in SEEDS])-base
            perseed=delta.mean(axis=1);boot=[]
            for _ in range(5000):
                si=rng.integers(20,size=20);ii=rng.integers(delta.shape[1],size=delta.shape[1])
                boot.append(delta[si][:,ii].mean())
            lo,hi=np.quantile(boot,[.025,.975])
            pilot=perseed[:3];effect=pilot.mean();sd=pilot.std(ddof=1)
            approximate_n=float((norm.ppf(1-.05/12)+norm.ppf(.8))**2*(sd/max(abs(effect),1e-12))**2)
            contrasts.append(dict(stage=stage,method=method,mean_difference=float(perseed.mean()),seed_difference_sd=float(perseed.std(ddof=1)),
                p_seed=float(ttest_1samp(perseed,0).pvalue),crossed_ci_low=float(lo),crossed_ci_high=float(hi),
                pilot_approximate_seeds_for_80pct_power=approximate_n,
                new_seeds_only_mean_difference=float(perseed[3:].mean()),new_seeds_only_p=float(ttest_1samp(perseed[3:],0).pvalue)))
    order=np.argsort([r['p_seed'] for r in contrasts]);previous=0
    for rank,idx in enumerate(order):
        previous=max(previous,min(1,(6-rank)*contrasts[idx]['p_seed']));contrasts[idx]['p_holm']=previous
    write_csv(OUT/'paired_contrasts.csv',contrasts)
    np.savez_compressed(OUT/'sample_silhouettes.npz',**{f's{st}_seed{se}_{m}':v for (st,se,m),v in scores.items()})
    lines=['# 增加 K-means seeds 的固定次數實驗','',
        '固定 seeds 42–61，共 20 個；每次 50 次初始化，K=31，資料與方法設定沿用先前實驗。保留原 3 seeds，新增 17 seeds；不依顯著性提前停止。',
        '', '**有效範圍：沿用舊座標特徵，不能推論修正後物體質心的分群品質。原 3 seeds 已用於選擇實驗，本輪屬探索性重複評估。**','',
        '## 分數與 seed 波動','', '| Stage | 方法 | 3 seeds 平均 ± SD | 20 seeds 平均 ± SD |','|---|---|---|---|']
    for stage in range(3):
        for method in METHODS:
            v=np.array([scores[stage,s,method].mean() for s in SEEDS]);p=v[:3]
            lines.append(f'| {stage} | {method} | {p.mean():.5f} ± {p.std(ddof=1):.5f} | {v.mean():.5f} ± {v.std(ddof=1):.5f} |')
    lines+=['','## 相對原方法的差值','',
        '| Stage | 方法 | 平均差值 | seed 配對檢定 p | Holm 校正 p | seed × image bootstrap 95% CI | 新增 17 seeds 平均差值 |',
        '|---|---|---|---|---|---|---|']
    for r in contrasts:
        lines.append(f"| {r['stage']} | {r['method']} | {r['mean_difference']:+.5f} | {r['p_seed']:.4g} | {r['p_holm']:.4g} | [{r['crossed_ci_low']:+.5f}, {r['crossed_ci_high']:+.5f}] | {r['new_seeds_only_mean_difference']:+.5f} |")
    lines+=['','## 統計解讀','',
        '增加独立 seeds 能減少初始化隨機性造成的平均差值不確定性；增加 n_init 並不提供更多独立重複。若真實差值很小或接近零，增加次數不保證顯著。',
        'seed 配對 t 檢定以每個 seed 的全體影像平均差值為一筆觀測，檢驗固定評估資料上跨初始化的期望差值；六組比較使用 Holm 校正。',
        '交叉 bootstrap 同時重抽 seeds 與影像，兩方法共用抽樣索引，重複 5,000 次。CI 為未做多重比較校正的逐比較百分位區間。樣本 silhouette 的距離與參考群成員保持固定，因此不包含重新抽訓練資料、重新訓練模型或重算 bootstrap 群幾何的不確定性。',
        '新增 17 seeds 的差值另列，供檢查提升是否只來自原先 3 seeds。p 值不代表改善幅度；即使顯著，仍應檢查差值大小與群的可解釋性。',
        'CSV 另附原 3 seeds 的粗略 80% power 樣本數估計：使用常態近似與六組 Bonferroni 門檻。只有 3 筆 pilot，估計非常不穩定，僅供規劃參考；沒有據此追跑到顯著。','']
    (OUT/'report.md').write_text('\n'.join(lines).replace('独立','獨立'))
    dest=ROOT/'plan/1005_kmeans_20_seeds_experiment_results.md';dest.write_text('\n'.join(lines).replace('独立','獨立'))
    print(json.dumps(contrasts,indent=2),flush=True)
if __name__=='__main__':main()
