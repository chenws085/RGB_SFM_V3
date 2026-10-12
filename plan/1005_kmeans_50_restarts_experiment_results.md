# 1005 增加 K-means 初始化次數：3 seeds × 50 restarts

> 2026-10-05 座標稽核更新：既有抽特徵流程使用 Caltech-101 的相對輪廓座標，漏加 bounding-box 偏移。本文件數值對應原取樣位置；不能直接作為物體質心或物體部件分群的結論。修正座標後需重新抽取特徵與 fitting。詳見 `plan/1005_caltech101_annotation_coordinate_audit.md` 與代表影像圖庫。

固定 seeds 42、43、44，每個 seed／stage／方法採 50 次初始化，依 training objective 選最佳，不以測試平移穩定性選模型。
沿用相同 checkpoint、6,907 張原始訓練圖、160 張 held-out airplanes、K=31、Stage 0–2 四方向平移。既有 sklearn n_init 從 20 增至 50；其他方法從 10 增至 50。手寫 Euclidean 與 spherical 共享初始中心；FAISS 同時測 native 與 shared init。
共 3 seeds × 3 stages × 50 restarts × 5 methods = 2,250 次初始化擬合（含配對 Euclidean）。這是增加每個 seed 內的初始化搜尋預算，最終每組仍只有 3 個 seed 結果，不能視為 150 次獨立測試。

## 平移同群比例與 seed 波動

數值為百分比平均 ± 三個 seeds 的樣本標準差（pp），每個 seed 先平均四方向。

| Stage | 方法 | 前次平均 ± SD | 50 次平均 ± SD |
|---|---|---:|---:|
| 0 | sklearn_existing | 96.04 ± 1.21 | 95.47 ± 0.87 |
| 0 | spherical | 93.59 ± 0.62 | 95.36 ± 0.79 |
| 0 | faiss_native | 94.64 ± 0.86 | 94.48 ± 0.45 |
| 0 | faiss_shared_init | 93.59 ± 0.62 | 95.36 ± 0.79 |
| 1 | sklearn_existing | 91.46 ± 1.66 | 91.46 ± 0.33 |
| 1 | spherical | 90.52 ± 1.74 | 92.66 ± 0.16 |
| 1 | faiss_native | 91.41 ± 2.82 | 91.41 ± 1.51 |
| 1 | faiss_shared_init | 90.52 ± 1.74 | 92.66 ± 0.16 |
| 2 | sklearn_existing | 89.53 ± 0.62 | 88.91 ± 1.13 |
| 2 | spherical | 88.54 ± 0.94 | 87.60 ± 1.33 |
| 2 | faiss_native | 88.23 ± 1.21 | 88.70 ± 1.67 |
| 2 | faiss_shared_init | 88.54 ± 0.94 | 87.60 ± 1.33 |

12 個 stage／方法比較中，6 個的 seed SD 降低；增加 restarts 不保證每個指標的跨 seed 波動均減少，最佳 training objective 也不保證最佳 translation stability。
FAISS shared-init 與手寫 spherical 的 training ARI：[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]。

Stage 0 球面 − sklearn：-0.10 pp；Stage 1 球面 − sklearn：+1.20 pp；Stage 2 球面 − sklearn：-1.30 pp。本次方法排名隨 stage 不同；影像 bootstrap 的各方法對 sklearn 差值 CI 均跨過 0，尚不足以宣稱球面法整體更好，先保留現行方法。

`objective_progress.csv` 記錄相同 seed／初始化序列在 10、20、50 次時的最佳 training objective，僅在各自方法內比較。`seed_stability.csv` 保留每個 seed 的結果。影像 bootstrap CI 見同目錄 report.md；它不估計額外 seed 的抽樣不確定性。

## 重現

```bash
mkdir -p plots/kmeans/spherical_comparison_1005_n50
cp plots/kmeans/spherical_comparison_1005/features.npz plots/kmeans/spherical_comparison_1005/feature_cache_identity.json plots/kmeans/spherical_comparison_1005_n50/
.venv/bin/python scripts/compare_spherical_kmeans.py --output-dir plots/kmeans/spherical_comparison_1005_n50 --n-init 50 --sklearn-n-init 50
.venv/bin/python scripts/report_spherical_kmeans.py --output-dir plots/kmeans/spherical_comparison_1005_n50
.venv/bin/python scripts/compare_faiss_spherical_kmeans.py --source-dir plots/kmeans/spherical_comparison_1005_n50 --output-dir plots/kmeans/faiss_spherical_comparison_1005_n50
.venv/bin/python scripts/report_kmeans_restart_stability.py
```

產物位於 `plots/kmeans/spherical_comparison_1005_n50/` 與 `plots/kmeans/faiss_spherical_comparison_1005_n50/`；包含所有逐輪紀錄、逐 seed 指標、每次 restart objective、最終中心、逐圖指派、manifest 與比較圖。
