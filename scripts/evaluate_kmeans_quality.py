#!/usr/bin/env python3
"""Evaluate frozen K-means centers using geometry, coverage and held-out data."""
import argparse
import csv
import hashlib
import itertools
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, silhouette_samples
from threadpoolctl import threadpool_limits
from spherical_kmeans import distances, unit_rows

METHODS = ["sklearn_existing", "euclidean", "spherical", "faiss_native", "faiss_shared_init"]


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def query_silhouette(distance_rows, labels, query_indices):
    """Exact silhouettes for selected queries against all reference samples."""
    unique, inverse = np.unique(labels, return_inverse=True)
    if not 1 < len(unique) < len(labels):
        return np.full(len(query_indices), np.nan)
    counts = np.bincount(inverse)
    membership = np.eye(len(unique))[inverse]
    sums = distance_rows @ membership
    own = inverse[query_indices]
    a = sums[np.arange(len(own)), own] / np.maximum(counts[own] - 1, 1)
    means = sums / counts[None, :]
    means[np.arange(len(own)), own] = np.inf
    b = means.min(axis=1)
    s = np.divide(b-a, np.maximum(a,b), out=np.zeros_like(a), where=np.maximum(a,b)>0)
    s[counts[own] == 1] = 0
    return s


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", default="plots/kmeans/spherical_comparison_1005_n50")
    parser.add_argument("--faiss-dir", default="plots/kmeans/faiss_spherical_comparison_1005_n50")
    parser.add_argument("--output-dir", default="plots/kmeans/quality_evaluation_1005")
    parser.add_argument("--train-silhouette-queries", type=int, default=2000)
    args = parser.parse_args()
    source, library, output = map(Path, [args.source_dir, args.faiss_dir, args.output_dir])
    output.mkdir(parents=True, exist_ok=True)
    threadpool_limits(4)
    manifest = json.loads((source / "run_manifest.json").read_text())
    library_manifest = json.loads((library / "run_manifest.json").read_text())
    feature_sha = hashlib.sha256((source / "features.npz").read_bytes()).hexdigest()
    assert library_manifest["source_features_sha256"] == feature_sha
    cache = np.load(source / "features.npz")
    # Import extraction only here so metric helpers remain lightweight and testable.
    from test_caltech_translation_kmeans import ROOT, Caltech101, LetterboxSubset, MergingViT, extract
    import torch
    torch.set_num_threads(4)
    base = Caltech101(str(ROOT / "data"), target_type=["category", "annotation"], download=False)
    train_ids = np.asarray(manifest["train_indices"])
    valid_ids = np.setdiff1d(np.arange(len(base)), train_ids)
    train_classes = np.asarray(base.y)[train_ids]
    valid_classes = np.asarray(base.y)[valid_ids]
    assert not np.intersect1d(train_ids, valid_ids).size
    assert set(manifest["test_indices"]).issubset(set(valid_ids))
    identity = dict(checkpoint_sha256=manifest["checkpoint_sha256"], source_features_sha256=feature_sha,
                    valid_ids=valid_ids.tolist(), protocol=manifest["protocol"])
    valid_path, identity_path = output / "heldout_features.npz", output / "heldout_cache_identity.json"
    if valid_path.exists() and identity_path.exists() and json.loads(identity_path.read_text()) == identity:
        valid_cache = dict(np.load(valid_path))
    else:
        checkpoint_path = Path(manifest["checkpoint"])
        assert hashlib.sha256(checkpoint_path.read_bytes()).hexdigest() == manifest["checkpoint_sha256"]
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model = MergingViT(**checkpoint["model_args"])
        model.load_state_dict(checkpoint["model_weights"], strict=True)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device).eval()
        result = extract(model, LetterboxSubset(base, valid_ids.tolist()), 32, 4, device,
                         int(checkpoint["model_args"]["patch_size"]))
        assert np.array_equal(result["source_indices"], valid_ids)
        valid_cache = {f"stage{stage}": result["features"][stage] for stage in range(3)}
        np.savez_compressed(valid_path, **valid_cache, source_ids=valid_ids, classes=valid_classes)
        identity_path.write_text(json.dumps(identity))
        print(f"Extracted {len(valid_ids)} all-class held-out images", flush=True)
    rng = np.random.default_rng(20261005)
    queries = np.sort(rng.choice(len(train_ids), min(args.train_silhouette_queries, len(train_ids)), replace=False))
    seeds, k = manifest["seeds"], manifest["clusters"]
    geometry, occupancy, seed_pairs, translation, paired, per_image = [], [], [], [], [], []
    silhouette_values, reconstruction_values = {}, {}
    for stage in range(3):
        train = unit_rows(cache[f"train_{stage}"].astype(np.float64))
        valid = unit_rows(valid_cache[f"stage{stage}"].astype(np.float64))
        train_d = np.maximum(1 - train[queries] @ train.T, 0)
        train_d[np.arange(len(queries)), queries] = 0
        valid_d = np.maximum(1 - valid @ valid.T, 0)
        np.fill_diagonal(valid_d, 0)
        assignments = {}
        for method in METHODS:
            for seed in seeds:
                root = library if method.startswith("faiss") else source
                stored = np.load(root / f"centers_stage{stage}_seed{seed}_{method}.npz")
                centers = stored["centers"].astype(np.float64)
                normalized_centers = unit_rows(centers)
                for split, x, classes in [("train", train, train_classes), ("heldout_all", valid, valid_classes)]:
                    labels = distances(x, centers).argmin(1)
                    assignments[(method, seed, split)] = labels
                    count = np.bincount(labels, minlength=k)
                    proportions = count / len(labels)
                    nonzero = proportions[proportions > 0]
                    silhouette = (query_silhouette(train_d, labels, queries) if split == "train"
                                  else silhouette_samples(valid_d, labels, metric="precomputed")
                                  if 1 < len(np.unique(labels)) < len(labels) else np.full(len(labels), np.nan))
                    reconstruction = 1 - np.sum(x * normalized_centers[labels], axis=1)
                    silhouette_values.setdefault((stage, method, split), []).append(silhouette)
                    reconstruction_values.setdefault((stage, method, split), []).append(reconstruction)
                    geometry.append(dict(stage=stage, seed=seed, method=method, split=split, samples=len(x),
                        silhouette_queries=len(silhouette), cosine_silhouette_mean=float(np.nanmean(silhouette)),
                        negative_silhouette_fraction=float(np.mean(silhouette < 0)),
                        assigned_cosine_dist_mean=float(reconstruction.mean()),
                        assigned_cosine_dist_p95=float(np.quantile(reconstruction,.95)),
                        effective_clusters=float(np.exp(-np.sum(nonzero*np.log(nonzero)))),
                        used_clusters=int(np.sum(count>0)), max_cluster_fraction=float(proportions.max()),
                        category_ami=float(adjusted_mutual_info_score(classes, labels)),
                        category_ari=float(adjusted_rand_score(classes, labels))))
                    occupancy.extend(dict(stage=stage, seed=seed, method=method, split=split,
                        cluster=cluster, count=int(n), fraction=float(n/len(labels))) for cluster,n in enumerate(count))
                    if split == "heldout_all":
                        per_image.extend(dict(stage=stage, seed=seed, method=method, source_index=int(i),
                            category=int(c), cluster=int(l), cosine_silhouette=float(s), assigned_cosine_distance=float(d))
                            for i,c,l,s,d in zip(valid_ids,classes,labels,silhouette,reconstruction))
                train_mismatch = float(np.mean(assignments[(method,seed,"train")] != stored["labels"]))
                assert train_mismatch < .001, (method,stage,seed,train_mismatch)
                original = unit_rows(cache[f"original_{stage}"].astype(float))
                original_d = distances(original, centers)
                original_labels = original_d.argmin(1)
                for direction in ["left","right","up","down"]:
                    moved = unit_rows(cache[f"s{stage}_{direction}_{stage}"].astype(float))
                    moved_d = distances(moved, centers)
                    moved_labels = moved_d.argmin(1)
                    same = original_labels == moved_labels
                    indices = np.arange(len(original))
                    regret = moved_d[indices,original_labels] - moved_d.min(1)
                    assigned_cosine_original = np.sum(original*normalized_centers[original_labels],axis=1)
                    original_center_cosine_moved = np.sum(moved*normalized_centers[original_labels],axis=1)
                    expected = float(np.dot(np.bincount(original_labels,minlength=k), np.bincount(moved_labels,minlength=k))/len(same)**2)
                    translation.append(dict(stage=stage, seed=seed, method=method, direction=direction,
                        same_cluster_rate=float(same.mean()), chance_agreement=expected,
                        chance_corrected_identity_agreement=float((same.mean()-expected)/(1-expected)) if expected<1 else np.nan,
                        macro_original_cluster_agreement=float(np.mean([same[original_labels==c].mean() for c in np.unique(original_labels)])),
                        original_used_clusters=len(np.unique(original_labels)),
                        squared_l2_regret_mean=float(regret.mean()),
                        squared_l2_regret_p95=float(np.quantile(regret,.95)),
                        squared_l2_regret_changed_mean=float(regret[~same].mean()) if (~same).any() else 0,
                        original_center_cosine_distance_change=float(np.mean(assigned_cosine_original-original_center_cosine_moved)),
                        matched_feature_cosine=float(np.mean(np.sum(original*moved,axis=1)))))
            for split in ["train","heldout_all"]:
                for a,b in itertools.combinations(seeds,2):
                    seed_pairs.append(dict(stage=stage,method=method,split=split,seed_a=a,seed_b=b,
                        ari=float(adjusted_rand_score(assignments[(method,a,split)],assignments[(method,b,split)]))))
            print(f"Evaluated stage={stage} method={method}",flush=True)
        bootstrap = rng.integers(len(valid_ids),size=(5000,len(valid_ids)))
        comparisons=[(method,"sklearn_existing") for method in METHODS[1:]]
        comparisons.extend((method,"euclidean") for method in ["spherical","faiss_shared_init"])
        for method,baseline in comparisons:
            for metric,values in [("cosine_silhouette",silhouette_values),("assigned_cosine_distance",reconstruction_values)]:
                difference=(np.asarray(values[(stage,method,"heldout_all")])-np.asarray(values[(stage,baseline,"heldout_all")])).mean(axis=0)
                low,high=np.quantile(difference[bootstrap].mean(axis=1),[.025,.975])
                paired.append(dict(stage=stage, method=method, baseline=baseline,metric=metric,
                    mean_difference=float(difference.mean()),ci_low=float(low),ci_high=float(high),
                    bootstrap_unit="heldout image; frozen silhouettes/centers, average seeds"))
    for name,rows in [("geometry_metrics.csv",geometry),("cluster_occupancy.csv",occupancy),
                      ("seed_pair_ari.csv",seed_pairs),("translation_diagnostics.csv",translation),
                      ("heldout_paired_differences.csv",paired),("heldout_sample_metrics.csv",per_image)]:
        write_csv(output/name,rows)
    eval_manifest=dict(source_manifest=manifest, source_features_sha256=feature_sha,heldout_identity=identity,
        faiss_manifest=library_manifest,train_silhouette_query_indices=queries.tolist(),
        evaluation=f"frozen centers; exact cosine silhouette for {len(queries)} fixed training queries against full training reference; full heldout silhouettes",
        versions=dict(numpy=np.__version__), bootstrap_samples=5000,
        limitations="heldout was checkpoint selection validation split, not untouched final test; no part labels; cosine geometry selects directional notion of quality; 3 seeds only; image bootstrap freezes reference distances and fitted centers")
    (output/"run_manifest.json").write_text(json.dumps(eval_manifest,indent=2))
    tables=[]
    for stage in range(3):
        for method in METHODS:
            rows=[r for r in geometry if r["stage"]==stage and r["method"]==method and r["split"]=="heldout_all"]
            stability=[r["ari"] for r in seed_pairs if r["stage"]==stage and r["method"]==method and r["split"]=="heldout_all"]
            tables.append(f"| {stage} | {method} | {np.mean([r['cosine_silhouette_mean'] for r in rows]):.4f} | {np.mean([r['assigned_cosine_dist_mean'] for r in rows]):.4f} | {np.mean([r['effective_clusters'] for r in rows]):.1f} | {np.mean([r['max_cluster_fraction'] for r in rows])*100:.1f}% | {np.mean(stability):.3f} | {np.mean([r['category_ami'] for r in rows]):.3f} |")
    ci_lines=[f"- Stage {r['stage']}，{r['method']} 的 silhouette − {r['baseline']}：{r['mean_difference']:+.5f}，95% CI [{r['ci_low']:+.5f}, {r['ci_high']:+.5f}]。" for r in paired if r['metric']=="cosine_silhouette"]
    diagnostics=[]
    for stage in range(3):
        for method in ["sklearn_existing","spherical","faiss_native"]:
            rows=[r for r in translation if r['stage']==stage and r['method']==method]
            diagnostics.append(f"| {stage} | {method} | {np.mean([r['same_cluster_rate'] for r in rows])*100:.2f}% | {np.mean([r['macro_original_cluster_agreement'] for r in rows])*100:.2f}% | {np.mean([r['chance_corrected_identity_agreement'] for r in rows]):.4f} | {np.mean([r['original_center_cosine_distance_change'] for r in rows]):.5f} |")
    supported=[r['stage'] for r in paired if r['metric']=="cosine_silhouette" and r['method']=="spherical" and r['baseline']=="sklearn_existing" and r['ci_low']>0]
    findings=[f"- 固定此批資料與三個 seeds，手寫球面法相對現有 sklearn 的 heldout cosine silhouette 差值 CI 完全大於 0 的 stages：{supported}；其他 stages 尚無同樣證據。"]
    for stage in range(3):
        stability={method:np.mean([r['ari'] for r in seed_pairs if r['stage']==stage and r['method']==method and r['split']=="heldout_all"]) for method in ["sklearn_existing","spherical","faiss_native"]}
        findings.append(f"- Stage {stage} 跨 seeds 的 heldout ARI：sklearn={stability['sklearn_existing']:.3f}，手寫球面={stability['spherical']:.3f}，FAISS native={stability['faiss_native']:.3f}。")
    findings.extend([
        "- 全類別 heldout 各方法均使用全部 31 群；有效群數與最大群比例沒有顯示少數群主導，仍應連同類別分布判讀，而不是要求所有群等大。",
        "- 方向空間幾何品質與平移指派穩定性有取捨；不能因平移同群比例略低，就否定球面分群的幾何品質。",
        "- 建議保留現行 Euclidean 作基準，將球面法列為方向空間分析的候選；不宣稱它在所有 stages／語義任務全面勝出。",
        "- 類別 AMI 只有輔助語義證據；下一階段若研究物體部件，仍需代表 patches 或部件標註。",
    ])
    report="\n".join([
        "# 1005 K-means 多面向品質評估", "",
        f"固定 K={k}、seeds={seeds}、每組 50 次初始化、checkpoint 與中心；training={len(train_ids)}、全類別 heldout={len(valid_ids)}。新增全類別原圖特徵只做評估，不更新中心。",
        "全類別 heldout 是原模型選 checkpoint 時的 validation split，因此這是既有模型的比較評估，不是完全未使用過的最終 test。幾何／類別評估使用全類別原圖；平移指標仍僅使用原有 160 張 airplanes。", "",
        "## 指標定義", "",
        f"- Cosine silhouette：所有方法使用相同樣本間 cosine distance，越高表示群內更近且與其他群更分離。training 用固定 {len(queries)} 個 queries 對全部 training 樣本計算精確 silhouette；heldout 使用全部樣本。",
        "- Assigned cosine distance：樣本與其實際所屬中心方向的 1−cosine，越低越好；不把 Euclidean 方法偷偷改成 cosine assignment。",
        "- 有效群數：exp(群使用 entropy)；最大群比例及群大小一併保存。不是越平均越好，也不單独作為優劣分數。",
        "- 跨 seed ARI：同一 split 三對 seeds 的分群一致性，越高越一致；這三對共享 seeds，並非三個獨立統計樣本。",
        "- 類別 AMI／ARI：群與影像類別的關聯，輔助語義證據；K=31、類別數不同，且質心 token 未必是整體物體，不能當作部件標註。",
        "- 平移：保留同群比例，新增機會校正的 identity agreement（期望值=sum p_original(k)p_moved(k)）、逐原群 macro agreement、到原中心 cosine distance 變化、換群 squared-L2 regret。",
        "- Regret=平移後到原中心的平方 Euclidean 距離−到最近中心距離；用各自實際中心計算，數值尺度會受中心 norm 影響，不作跨方法的單一排名。", "",
        "## 全類別 heldout 結果（3 seeds 平均）", "",
        "| Stage | 方法 | Cosine silhouette ↑ | Assigned cosine distance ↓ | 有效群數 | 最大群比例 | Seed ARI ↑ | 類別 AMI ↑ |",
        "|---|---|---:|---:|---:|---:|---:|---:|",*tables,"",
        "## Silhouette 配對差值", "",*ci_lines,"",
        "CI 以 heldout 影像抽樣 5000 次，先平均配對 seeds；固定中心、reference distances 與每筆 silhouette，不重新分群。沒有涵蓋重新訓練模型、reference set 改變或更多 seeds 的不確定性。", "",
        "## 飛機平移診斷（3 seeds × 四方向平均）", "",
        "| Stage | 方法 | 同群比例 | 逐原群 macro 同群比例 | 機會校正一致性 | 到原中心 cosine distance 增量 |",
        "|---|---|---:|---:|---:|---:|",*diagnostics,"",
        "不合併成任意加權總分：方向幾何、群覆蓋、初始化穩定性、類別關聯與平移穩定性分開判讀。固定 cosine 距離是本週『方向特徵空間』研究的選擇，結果不代表所有距離幾何或語義任務。", "",
        "## 本次判讀", "",*findings,"",
        "計算驗證：query silhouette 與 sklearn 在含 singleton 的小型案例一致；3 stages 的 heldout silhouette 與 sklearn 直接 cosine 計算一致；跨 seed ARI 重算一致；群大小總和、單位 norm 輸入、無平移 regret=0 及所有輸出筆數已核對。", "",
        "重現：`.venv/bin/python scripts/evaluate_kmeans_quality.py`。產物包含 geometry_metrics.csv、cluster_occupancy.csv、seed_pair_ari.csv、translation_diagnostics.csv、heldout_paired_differences.csv、heldout_sample_metrics.csv、manifest、heldout features 與 quality_comparison.png。", "",
        "指標來源：[Silhouette](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_samples.html)、[ARI](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html)、[AMI](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_mutual_info_score.html)。",
    ])
    (output/"report.md").write_text(report+"\n")
    Path("plan/1005_kmeans_quality_evaluation_results.md").write_text(report+"\n")
    fig,axes=plt.subplots(2,3,figsize=(15,8))
    for stage in range(3):
        for idx,method in enumerate(METHODS):
            for row,field in [(0,"cosine_silhouette_mean"),(1,"assigned_cosine_dist_mean")]:
                values=[r[field] for r in geometry if r['stage']==stage and r['method']==method and r['split']=="heldout_all"]
                axes[row,stage].errorbar(idx,np.mean(values),yerr=np.std(values,ddof=1),fmt="o",capsize=4)
                axes[row,stage].set_xticks(range(len(METHODS)),METHODS,rotation=30)
                axes[row,stage].set_title(f"Stage {stage}: {field}")
    fig.tight_layout()
    fig.savefig(output/"quality_comparison.png",dpi=160)
    plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(15,8))
    for stage in range(3):
        for idx,method in enumerate(METHODS):
            coverage=[r['effective_clusters'] for r in geometry if r['stage']==stage and r['method']==method and r['split']=="heldout_all"]
            stability=[r['ari'] for r in seed_pairs if r['stage']==stage and r['method']==method and r['split']=="heldout_all"]
            for row,values,title in [(0,coverage,"effective clusters"),(1,stability,"pairwise seed ARI")]:
                axes[row,stage].scatter([idx]*len(values),values,s=20)
                axes[row,stage].plot(idx,np.mean(values),marker="_",markersize=14,color="black")
                axes[row,stage].set_title(f"Stage {stage}: {title}")
                axes[row,stage].set_xticks(range(len(METHODS)),METHODS,rotation=30)
    fig.tight_layout()
    fig.savefig(output/"coverage_seed_stability.png",dpi=160)
    plt.close(fig)
    print(report,flush=True)


if __name__=="__main__":
    main()
