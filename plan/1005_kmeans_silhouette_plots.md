# K-means 輪廓圖（Silhouette Plot）

> 2026-10-05 座標稽核更新：既有抽特徵流程使用 Caltech-101 的相對輪廓座標，漏加 bounding-box 偏移。本文件數值對應原取樣位置；不能直接作為物體質心或物體部件分群的結論。修正座標後需重新抽取特徵與 fitting。詳見 `plan/1005_caltech101_annotation_coordinate_audit.md` 與代表影像圖庫。

使用品質評估已保存的全部 1,770 張全類別 held-out 原圖 silhouette，中心來自每個 seed 的 50 次初始化最佳結果。每張圖並排顯示現有 Euclidean K-means、手寫球面 K-means、FAISS native 球面 K-means。

## 圖中標示

- 橫軸：每筆樣本的 cosine silhouette，越靠右表示群內距離相對較小、最近其他群的距離相對較大。
- 每個色帶：一個群；色帶高度與群內樣本數成比例，帶內樣本依 silhouette 從小到大排序。色帶之間的空白是視覺間距，並非樣本。
- 縱軸標籤：cluster ID / 該群樣本數。群依 ID 排列；不同方法與 seeds 的 ID、顏色不代表相同群。
- 紅色虛線：該方法、該 seed 的全體樣本平均 silhouette。它是單一 seed 平均；先前簡表是三個 seeds 平均，所以數字不一定相同。
- 灰色直線：silhouette = 0；左側負值表示該樣本到另一群的平均距離比到自身群更小。
- negative：負 silhouette 樣本的比例。圖中全部樣本均顯示；所有圖採相同橫軸範圍。

每個 seed 分開畫，避免把不同分群的 cluster ID 與逐樣本 silhouette 混合成不存在的平均分群。

## 圖片

| Stage | Seed 42 | Seed 43 | Seed 44 |
|---|---|---|---|
| 0 | [輪廓圖](../plots/kmeans/quality_evaluation_1005/silhouette_plots/stage0_seed42_silhouette_comparison.png) | [輪廓圖](../plots/kmeans/quality_evaluation_1005/silhouette_plots/stage0_seed43_silhouette_comparison.png) | [輪廓圖](../plots/kmeans/quality_evaluation_1005/silhouette_plots/stage0_seed44_silhouette_comparison.png) |
| 1 | [輪廓圖](../plots/kmeans/quality_evaluation_1005/silhouette_plots/stage1_seed42_silhouette_comparison.png) | [輪廓圖](../plots/kmeans/quality_evaluation_1005/silhouette_plots/stage1_seed43_silhouette_comparison.png) | [輪廓圖](../plots/kmeans/quality_evaluation_1005/silhouette_plots/stage1_seed44_silhouette_comparison.png) |
| 2 | [輪廓圖](../plots/kmeans/quality_evaluation_1005/silhouette_plots/stage2_seed42_silhouette_comparison.png) | [輪廓圖](../plots/kmeans/quality_evaluation_1005/silhouette_plots/stage2_seed43_silhouette_comparison.png) | [輪廓圖](../plots/kmeans/quality_evaluation_1005/silhouette_plots/stage2_seed44_silhouette_comparison.png) |

[全部九張圖的 PDF](../plots/kmeans/quality_evaluation_1005/silhouette_plots/all_stages_seeds_silhouettes.pdf)

重現：`.venv/bin/python scripts/plot_kmeans_silhouettes.py`。逐圖摘要：`plots/kmeans/quality_evaluation_1005/silhouette_plots/plot_summary.csv`。

判讀時同時觀察平均線、負值比例、各群的厚度與形狀；silhouette 衡量方向空間中的幾何分群，不等同於部件語義或平移穩定性。
資料限制：這批 held-out 圖片屬於原模型選 checkpoint 的 validation split。
