# 1005 FAISS 球面 K-means 套件比較

> 2026-10-05 座標稽核更新：既有抽特徵流程使用 Caltech-101 的相對輪廓座標，漏加 bounding-box 偏移。本文件數值對應原取樣位置；不能直接作為物體質心或物體部件分群的結論。修正座標後需重新抽取特徵與 fitting。詳見 `plan/1005_caltech101_annotation_coordinate_audit.md` 與代表影像圖庫。

套件 faiss-cpu==1.12.0；沿用前次特徵：6,907 張 Caltech-101 原始訓練圖、160 張 held-out airplanes、K=31、seeds=[42, 43, 44]、Stage 0–2 四方向平移。
使用 `faiss.Kmeans(..., spherical=True, niter=300, nredo=10)`，CPU 4 threads，float32；max_points_per_centroid 設為訓練樣本數，確保不抽樣。
FAISS native：套件自身 random 初始化、10 restarts。FAISS shared init：10 組與手寫版本完全相同的 kmeans++ 初始中心，分別跑 nredo=1，依最終 cosine objective 選最佳。FAISS 固定 300 輪，手寫版以中心變化停止；數值精度也不同，因此不要求逐 bit 一致。

| Stage | 現有 sklearn | 手寫 spherical | FAISS native | FAISS shared init |
|---|---:|---:|---:|---:|
| 0 | 96.04% | 93.59% | 94.64% | 93.59% |
| 1 | 91.46% | 90.52% | 91.41% | 90.52% |
| 2 | 89.53% | 88.54% | 88.23% | 88.54% |

- Stage 0, faiss_native − sklearn：-1.41 pp，影像 bootstrap 95% CI [-3.59, +0.73] pp。
- Stage 0, faiss_shared_init − sklearn：-2.45 pp，影像 bootstrap 95% CI [-4.95, -0.05] pp。
- Stage 1, faiss_native − sklearn：-0.05 pp，影像 bootstrap 95% CI [-2.08, +1.98] pp。
- Stage 1, faiss_shared_init − sklearn：-0.94 pp，影像 bootstrap 95% CI [-3.44, +1.61] pp。
- Stage 2, faiss_native − sklearn：-1.30 pp，影像 bootstrap 95% CI [-4.06, +1.61] pp。
- Stage 2, faiss_shared_init − sklearn：-0.99 pp，影像 bootstrap 95% CI [-3.85, +1.93] pp。

bootstrap 以影像為單位，保留 seeds 與方向的配對；僅反映此批 held-out 飛機影像的抽樣不確定性。

FAISS native 與共享初始化兩組、三個 stages 的同群比例均低於現有 sklearn；本次套件測試也未觀察到平移穩定性改善，先保留現行方法。此結果不等同於所有資料集／類別的結論。

已檢查最終 centroid unit norm、群大小、與手寫版的 ARI；詳見 fit_summary.csv。FAISS Python API 有每輪 objective、time、imbalance 等統計，但沒有每輪 centroid／raw mean norm；不將手寫版的 norm 曲線當作套件量測。

FAISS spherical 的每輪 objective 是內積總和，越大越好；不直接與 sklearn inertia 比較。另以最終中心重新计算 cosine distance。

來源：[FAISS 官方 FAQ](https://github.com/facebookresearch/faiss/wiki/FAQ)、[ClusteringParameters](https://faiss.ai/cpp_api/struct/structfaiss_1_1ClusteringParameters.html)。

```bash
.venv/bin/python -m pip install faiss-cpu==1.12.0 --no-deps
.venv/bin/python scripts/compare_faiss_spherical_kmeans.py
```

產物：`plots/kmeans/faiss_spherical_comparison_1005`，包含 manifest、fit_summary.csv、translation_metrics.csv、sample_assignments.csv、faiss_iteration_stats.csv、各組最終中心及比較圖。

額外驗證：9 組共享初始化的最終 training assignments 與手寫版逐筆完全一致，ARI 全為 1；18 組 FAISS 最終中心 norm 誤差均小於 1e-6。
