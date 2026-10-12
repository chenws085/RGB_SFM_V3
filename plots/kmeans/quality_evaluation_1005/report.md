# 1005 K-means 多面向品質評估

> 2026-10-05 座標稽核更新：既有抽特徵流程使用 Caltech-101 的相對輪廓座標，漏加 bounding-box 偏移。本文件數值對應原取樣位置；不能直接作為物體質心或物體部件分群的結論。修正座標後需重新抽取特徵與 fitting。詳見 `plan/1005_caltech101_annotation_coordinate_audit.md` 與代表影像圖庫。

固定 K=31、seeds=[42, 43, 44]、每組 50 次初始化、checkpoint 與中心；training=6907、全類別 heldout=1770。新增全類別原圖特徵只做評估，不更新中心。
全類別 heldout 是原模型選 checkpoint 時的 validation split，因此這是既有模型的比較評估，不是完全未使用過的最終 test。幾何／類別評估使用全類別原圖；平移指標仍僅使用原有 160 張 airplanes。

## 指標定義

- Cosine silhouette：所有方法使用相同樣本間 cosine distance，越高表示群內更近且與其他群更分離。training 用固定 2000 個 queries 對全部 training 樣本計算精確 silhouette；heldout 使用全部樣本。
- Assigned cosine distance：樣本與其實際所屬中心方向的 1−cosine，越低越好；不把 Euclidean 方法偷偷改成 cosine assignment。
- 有效群數：exp(群使用 entropy)；最大群比例及群大小一併保存。不是越平均越好，也不單独作為優劣分數。
- 跨 seed ARI：同一 split 三對 seeds 的分群一致性，越高越一致；這三對共享 seeds，並非三個獨立統計樣本。
- 類別 AMI／ARI：群與影像類別的關聯，輔助語義證據；K=31、類別數不同，且質心 token 未必是整體物體，不能當作部件標註。
- 平移：保留同群比例，新增機會校正的 identity agreement（期望值=sum p_original(k)p_moved(k)）、逐原群 macro agreement、到原中心 cosine distance 變化、換群 squared-L2 regret。
- Regret=平移後到原中心的平方 Euclidean 距離−到最近中心距離；用各自實際中心計算，數值尺度會受中心 norm 影響，不作跨方法的單一排名。

## 全類別 heldout 結果（3 seeds 平均）

| Stage | 方法 | Cosine silhouette ↑ | Assigned cosine distance ↓ | 有效群數 | 最大群比例 | Seed ARI ↑ | 類別 AMI ↑ |
|---|---|---:|---:|---:|---:|---:|---:|
| 0 | sklearn_existing | 0.1996 | 0.2309 | 29.2 | 7.8% | 0.691 | 0.079 |
| 0 | euclidean | 0.1988 | 0.2303 | 29.3 | 7.9% | 0.705 | 0.079 |
| 0 | spherical | 0.2024 | 0.2294 | 28.7 | 9.0% | 0.676 | 0.080 |
| 0 | faiss_native | 0.2010 | 0.2307 | 28.8 | 8.6% | 0.683 | 0.078 |
| 0 | faiss_shared_init | 0.2024 | 0.2294 | 28.7 | 9.0% | 0.676 | 0.080 |
| 1 | sklearn_existing | 0.1320 | 0.3506 | 29.9 | 6.6% | 0.652 | 0.095 |
| 1 | euclidean | 0.1248 | 0.3521 | 30.0 | 6.0% | 0.653 | 0.096 |
| 1 | spherical | 0.1339 | 0.3504 | 29.5 | 7.2% | 0.682 | 0.098 |
| 1 | faiss_native | 0.1324 | 0.3504 | 29.5 | 7.2% | 0.680 | 0.096 |
| 1 | faiss_shared_init | 0.1339 | 0.3504 | 29.5 | 7.2% | 0.682 | 0.098 |
| 2 | sklearn_existing | 0.0903 | 0.3952 | 30.0 | 5.0% | 0.601 | 0.147 |
| 2 | euclidean | 0.0908 | 0.3953 | 30.0 | 5.0% | 0.622 | 0.148 |
| 2 | spherical | 0.1000 | 0.3943 | 29.8 | 5.3% | 0.654 | 0.152 |
| 2 | faiss_native | 0.1012 | 0.3940 | 29.8 | 5.1% | 0.668 | 0.156 |
| 2 | faiss_shared_init | 0.1000 | 0.3943 | 29.8 | 5.3% | 0.654 | 0.152 |

## Silhouette 配對差值

- Stage 0，euclidean 的 silhouette − sklearn_existing：-0.00084，95% CI [-0.00424, +0.00261]。
- Stage 0，spherical 的 silhouette − sklearn_existing：+0.00282，95% CI [-0.00221, +0.00789]。
- Stage 0，faiss_native 的 silhouette − sklearn_existing：+0.00137，95% CI [-0.00268, +0.00526]。
- Stage 0，faiss_shared_init 的 silhouette − sklearn_existing：+0.00282，95% CI [-0.00221, +0.00789]。
- Stage 0，spherical 的 silhouette − euclidean：+0.00366，95% CI [-0.00137, +0.00860]。
- Stage 0，faiss_shared_init 的 silhouette − euclidean：+0.00366，95% CI [-0.00137, +0.00860]。
- Stage 1，euclidean 的 silhouette − sklearn_existing：-0.00718，95% CI [-0.01011, -0.00412]。
- Stage 1，spherical 的 silhouette − sklearn_existing：+0.00191，95% CI [-0.00076, +0.00462]。
- Stage 1，faiss_native 的 silhouette − sklearn_existing：+0.00037，95% CI [-0.00254, +0.00318]。
- Stage 1，faiss_shared_init 的 silhouette − sklearn_existing：+0.00191，95% CI [-0.00076, +0.00462]。
- Stage 1，spherical 的 silhouette − euclidean：+0.00909，95% CI [+0.00656, +0.01169]。
- Stage 1，faiss_shared_init 的 silhouette − euclidean：+0.00909，95% CI [+0.00656, +0.01169]。
- Stage 2，euclidean 的 silhouette − sklearn_existing：+0.00047，95% CI [-0.00177, +0.00276]。
- Stage 2，spherical 的 silhouette − sklearn_existing：+0.00976，95% CI [+0.00727, +0.01230]。
- Stage 2，faiss_native 的 silhouette − sklearn_existing：+0.01094，95% CI [+0.00857, +0.01335]。
- Stage 2，faiss_shared_init 的 silhouette − sklearn_existing：+0.00976，95% CI [+0.00727, +0.01230]。
- Stage 2，spherical 的 silhouette − euclidean：+0.00929，95% CI [+0.00702, +0.01158]。
- Stage 2，faiss_shared_init 的 silhouette − euclidean：+0.00929，95% CI [+0.00702, +0.01158]。

CI 以 heldout 影像抽樣 5000 次，先平均配對 seeds；固定中心、reference distances 與每筆 silhouette，不重新分群。沒有涵蓋重新訓練模型、reference set 改變或更多 seeds 的不確定性。

## 飛機平移診斷（3 seeds × 四方向平均）

| Stage | 方法 | 同群比例 | 逐原群 macro 同群比例 | 機會校正一致性 | 到原中心 cosine distance 增量 |
|---|---|---:|---:|---:|---:|
| 0 | sklearn_existing | 95.47% | 93.94% | 0.9506 | 0.00317 |
| 0 | spherical | 95.36% | 93.18% | 0.9491 | 0.00363 |
| 0 | faiss_native | 94.48% | 92.47% | 0.9394 | 0.00265 |
| 1 | sklearn_existing | 91.46% | 89.51% | 0.9060 | 0.00586 |
| 1 | spherical | 92.66% | 91.39% | 0.9185 | 0.00618 |
| 1 | faiss_native | 91.41% | 88.46% | 0.9049 | 0.00625 |
| 2 | sklearn_existing | 88.91% | 84.01% | 0.8717 | 0.02104 |
| 2 | spherical | 87.60% | 86.81% | 0.8535 | 0.02129 |
| 2 | faiss_native | 88.70% | 84.44% | 0.8668 | 0.02099 |

不合併成任意加權總分：方向幾何、群覆蓋、初始化穩定性、類別關聯與平移穩定性分開判讀。固定 cosine 距離是本週『方向特徵空間』研究的選擇，結果不代表所有距離幾何或語義任務。

## 本次判讀

- 固定此批資料與三個 seeds，手寫球面法相對現有 sklearn 的 heldout cosine silhouette 差值 CI 完全大於 0 的 stages：[2]；其他 stages 尚無同樣證據。
- Stage 0 跨 seeds 的 heldout ARI：sklearn=0.691，手寫球面=0.676，FAISS native=0.683。
- Stage 1 跨 seeds 的 heldout ARI：sklearn=0.652，手寫球面=0.682，FAISS native=0.680。
- Stage 2 跨 seeds 的 heldout ARI：sklearn=0.601，手寫球面=0.654，FAISS native=0.668。
- 全類別 heldout 各方法均使用全部 31 群；有效群數與最大群比例沒有顯示少數群主導，仍應連同類別分布判讀，而不是要求所有群等大。
- 方向空間幾何品質與平移指派穩定性有取捨；不能因平移同群比例略低，就否定球面分群的幾何品質。
- 建議保留現行 Euclidean 作基準，將球面法列為方向空間分析的候選；不宣稱它在所有 stages／語義任務全面勝出。
- 類別 AMI 只有輔助語義證據；下一階段若研究物體部件，仍需代表 patches 或部件標註。

計算驗證：query silhouette 與 sklearn 在含 singleton 的小型案例一致；3 stages 的 heldout silhouette 與 sklearn 直接 cosine 計算一致；跨 seed ARI 重算一致；群大小總和、單位 norm 輸入、無平移 regret=0 及所有輸出筆數已核對。

重現：`.venv/bin/python scripts/evaluate_kmeans_quality.py`。產物包含 geometry_metrics.csv、cluster_occupancy.csv、seed_pair_ari.csv、translation_diagnostics.csv、heldout_paired_differences.csv、heldout_sample_metrics.csv、manifest、heldout features 與 quality_comparison.png。

指標來源：[Silhouette](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_samples.html)、[ARI](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html)、[AMI](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_mutual_info_score.html)。
