# 1005 K-means 改進實驗：每輪 centroid L2 normalization

> 2026-10-05 座標稽核更新：既有抽特徵流程使用 Caltech-101 的相對輪廓座標，漏加 bounding-box 偏移。本文件數值對應原取樣位置；不能直接作為物體質心或物體部件分群的結論。修正座標後需重新抽取特徵與 fitting。詳見 `plan/1005_caltech101_annotation_coordinate_audit.md` 與代表影像圖庫。

使用既有 Caltech-101 letterbox160 checkpoint，原始訓練圖 6907 張、held-out airplanes 160 張、K=31；seeds=[42, 43, 44]。
Stage 0/1/2 分別平移 8/16/32 px，四個方向，依物體質心選取對應 token；沒有裁切或將平移圖加入 fitting。此實驗是 centroid-token 分群，尚未擴大至逐位置／per-head 分群。

三組：現有 sklearn（n_init=50、預設 tol=1e-4）；可觀測 Euclidean Lloyd；spherical Lloyd（每輪算術平均後 L2 normalize）。後兩組各 50 restarts，共享每次 kmeans++ 初始中心、max_iter=300、absolute tol=1e-5，各自以自己的 objective 選最佳 restart。與 sklearn 的差異也包含初始化與停止條件，因此因果比較以配對 Lloyd 兩組為主。

## 平移前後同群比例

平均涵蓋 3 seeds 與四個方向；不同方法的 cluster ID 不直接比較。

| Stage | 現有 sklearn | 配對 Euclidean | Spherical | Spherical − sklearn |
|---|---:|---:|---:|---:|
| 0 | 95.47% | 94.53% | 95.36% | -0.10 pp |
| 1 | 91.46% | 92.19% | 92.66% | +1.20 pp |
| 2 | 88.91% | 88.70% | 87.60% | -1.30 pp |

- Stage 0, spherical − sklearn_existing：-0.10 pp；影像 bootstrap 95% CI [-1.98, +1.77] pp。
- Stage 0, spherical − euclidean：+0.83 pp；影像 bootstrap 95% CI [-1.15, +2.86] pp。
- Stage 1, spherical − sklearn_existing：+1.20 pp；影像 bootstrap 95% CI [-1.25, +3.54] pp。
- Stage 1, spherical − euclidean：+0.47 pp；影像 bootstrap 95% CI [-1.51, +2.34] pp。
- Stage 2, spherical − sklearn_existing：-1.30 pp；影像 bootstrap 95% CI [-3.75, +1.09] pp。
- Stage 2, spherical − euclidean：-1.09 pp；影像 bootstrap 95% CI [-3.96, +1.77] pp。

CI 以同一張 held-out 影像作為抽樣單位，保留所有 seeds／directions 的配對，10,000 次 bootstrap；僅反映這批飛機影像的抽樣不確定性，不涵蓋重新訓練 checkpoint 或其他類別。

## 中心與成本

| Stage | 方法 | 平均 centroid norm | 最近中心 cosine distance | fitting 秒／seed |
|---|---|---:|---:|---:|
| 0 | sklearn_existing | 0.760840 | 0.222059 | 0.91 |
| 0 | euclidean | 0.760959 | 0.222189 | 11.68 |
| 0 | spherical | 1.000000 | 0.222263 | 11.89 |
| 1 | sklearn_existing | 0.647783 | 0.339154 | 1.17 |
| 1 | euclidean | 0.651640 | 0.339512 | 16.57 |
| 1 | spherical | 1.000000 | 0.338802 | 16.08 |
| 2 | sklearn_existing | 0.616479 | 0.386440 | 1.77 |
| 2 | euclidean | 0.616942 | 0.386274 | 29.55 |
| 2 | spherical | 1.000000 | 0.386003 | 29.46 |

球面每輪 norm 最大偏差：3.33e-16；未達停止條件的已選 runs：0；所有 restarts 的中心重初始化事件：0。
`raw_mean_norm` 是投影前群內平均的長度，反映方向集中程度；`centroid_norm` 是投影後長度。兩者均逐群、逐輪保存。最近中心 cosine distance 為共同診斷；各方法自己的 objective 不直接跨方法比較。

## 重現與產物

```bash
.venv/bin/python scripts/compare_spherical_kmeans.py
.venv/bin/python scripts/report_spherical_kmeans.py
```

輸出位置：`plots/kmeans/spherical_comparison_1005_n50`。包含 manifest（checkpoint SHA256、train/test IDs）、features cache、每組最終中心、fit_summary.csv、translation_metrics.csv、sample_assignments.csv、iteration_history.csv、translation_comparison.png、centroid_norm_history.png。

基本驗證通過：相同初始中心的一般 Lloyd 與 sklearn objective 一致；球面中心 unit norm；Euclidean／cosine assignment 一致；最終 labels 與 centers 一致；一般案例 objective 不上升；空群恢復與零向量拒絕。

## 本次判讀

依各 stage 的差值與配對 CI 判讀；每輪 unit norm 限制不保證改善平移指派。優先補其他類別與代表 patch 品質，再決定是否全面替換。

此階段尚未驗證代表 patches 的語義品質，較高同群比例本身不足以證明整體分群更好。
