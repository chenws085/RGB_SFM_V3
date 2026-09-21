# Caltech-101：不同 Stage 平移一個 Token 的 KMeans 穩定性實驗

## 1. 實驗目的

本實驗檢查同一個局部影像內容在移動至相鄰 token 位置後，是否仍會被固定的 KMeans 指派至相同 cluster。為確保不同 stage 都恰好移動一個 token，位移量依有效 receptive-field size 調整：

| Stage | 一個 token 對應的輸入距離 | 實驗位移 |
|---|---:|---:|
| Stage 0 | 8×8 pixels | 8 pixels |
| Stage 1 | 16×16 pixels | 16 pixels |
| Stage 2 | 32×32 pixels | 32 pixels |
| Stage 3 | 64×64 pixels | 本輪不執行 |

Stage 3 若要在 224×224 畫布內完整平移 64 pixels，必須將完整原圖縮小到最多 96×96，會引入明顯尺度差異，因此不與 Stage 0–2 合併比較。

## 2. 資料與模型設定

- 資料集：Caltech-101；案例類別為 `airplanes`。
- 模型輸入：224×224。
- 前處理：將完整原圖等比例縮放至長邊不超過 160 pixels，再置中於 224×224 safe-letterbox 畫布。
- 四周至少保留 32 pixels，Stage 0–2 平移後不裁切任何原始內容。
- 模型 checkpoint：`mergingvit_caltech101_letterbox160` epoch 66。
- 整體 validation accuracy：62.09%。
- airplanes validation accuracy：98.75%。
- KMeans：`K=31`，各 stage 分開建立。
- KMeans fitting：只使用 6,907 張未平移的 training images；平移圖與測試圖均未加入 fitting。
- 測試：160 張 held-out airplanes。
- 每個 stage 測試左、右、上、下四個方向。
- 所有配對的 `source_pixels_retained=1.0`。
- 每個 stage × direction 的 160 張影像均驗證 token index 精確移動一格。

## 3. 指標定義

- **Cluster agreement**：同一來源影像平移前後落在相同 KMeans cluster 的比例。
- **AMI / ARI**：整組測試影像平移前後 cluster assignments 的一致程度。
- **Matched-token cosine**：平移前後對應 token feature 的 cosine similarity。
- **Centroid cosine distance**：`1 − cosine(token, assigned cluster centroid)`；越小表示 token 越接近所屬 cluster 中心。平移前後分別以各自被指派的 cluster 計算。
- **Cluster margin**：最近與次近 centroid 距離之差；越小表示越靠近 KMeans 決策邊界。
- **Prediction consistency**：平移前後模型預測類別相同的比例。
- **Classification accuracy**：平移後影像仍正確分類為 airplane 的比例。
- **Padding contact rate**：被分析 token 的 receptive field 接觸人工 letterbox padding 的比例。

## 4. 四方向完整結果

| Stage | 位移 | 方向 | Agreement | Cosine | AMI | ARI | Margin | Prediction consistency | Accuracy | Padding contact |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 8 px | 左 | 98.12% | 0.9956 | 0.9693 | 0.9702 | 0.1500 | 97.50% | 97.50% | 0.00% |
| 0 | 8 px | 右 | 98.75% | 0.9929 | 0.9791 | 0.9773 | 0.1481 | 97.50% | 98.12% | 0.00% |
| 0 | 8 px | 上 | 93.75% | 0.9920 | 0.8997 | 0.8824 | 0.1453 | 96.25% | 96.25% | 0.00% |
| 0 | 8 px | 下 | 95.00% | 0.9929 | 0.9200 | 0.9091 | 0.1500 | 97.50% | 98.12% | 0.00% |
| 1 | 16 px | 左 | 92.50% | 0.9797 | 0.8783 | 0.8515 | 0.1133 | 93.75% | 93.13% | 0.00% |
| 1 | 16 px | 右 | 93.13% | 0.9775 | 0.8989 | 0.8789 | 0.1123 | 95.63% | 95.63% | 0.00% |
| 1 | 16 px | 上 | 92.50% | 0.9764 | 0.8773 | 0.8599 | 0.1099 | 71.25% | 71.25% | 0.00% |
| 1 | 16 px | 下 | 93.75% | 0.9736 | 0.9061 | 0.8996 | 0.1092 | 55.63% | 56.25% | 0.00% |
| 2 | 32 px | 左 | 86.25% | 0.9550 | 0.7508 | 0.7134 | 0.0914 | 84.38% | 84.38% | 3.12% |
| 2 | 32 px | 右 | 87.50% | 0.9676 | 0.7901 | 0.7866 | 0.0945 | 81.87% | 81.87% | 3.12% |
| 2 | 32 px | 上 | 92.50% | 0.9723 | 0.8675 | 0.8349 | 0.0940 | 45.00% | 45.00% | 3.12% |
| 2 | 32 px | 下 | 89.38% | 0.9538 | 0.8064 | 0.7886 | 0.0909 | 32.50% | 32.50% | 3.12% |

## 5. Stage 平均結果

| Stage | Agreement | Cosine | AMI | ARI | Margin | Prediction consistency | Accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| Stage 0 | 96.41% | 0.9934 | 0.9420 | 0.9347 | 0.1484 | 97.19% | 97.50% |
| Stage 1 | 92.97% | 0.9768 | 0.8902 | 0.8725 | 0.1112 | 79.06% | 79.06% |
| Stage 2 | 88.91% | 0.9622 | 0.8037 | 0.7809 | 0.0927 | 60.94% | 60.94% |

隨 stage 加深及對應位移量增加，cluster agreement、cosine、AMI、ARI 與 cluster margin 皆下降。KMeans cluster 的下降幅度仍小於分類預測穩定性的下降幅度。由於三個 stage 同時使用不同 pixel 位移量，本實驗描述的是「各 stage 移動一個 token」的整體效應，不能單獨歸因於網路深度。

### 5.1 與所屬 Cluster 中心的 Cosine 距離

| Stage | 方向 | 平移前 centroid cosine distance | 平移後 centroid cosine distance | 變化量 |
|---|---|---:|---:|---:|
| Stage 0 | 左 | 0.1986 | 0.1998 | +0.0012 |
| Stage 0 | 右 | 0.1986 | 0.2009 | +0.0023 |
| Stage 0 | 上 | 0.1986 | 0.2021 | +0.0035 |
| Stage 0 | 下 | 0.1986 | 0.2014 | +0.0028 |
| Stage 1 | 左 | 0.3009 | 0.3013 | +0.0004 |
| Stage 1 | 右 | 0.3009 | 0.3032 | +0.0023 |
| Stage 1 | 上 | 0.3009 | 0.3067 | +0.0058 |
| Stage 1 | 下 | 0.3009 | 0.3085 | +0.0076 |
| Stage 2 | 左 | 0.3211 | 0.3330 | +0.0119 |
| Stage 2 | 右 | 0.3211 | 0.3234 | +0.0023 |
| Stage 2 | 上 | 0.3211 | 0.3369 | +0.0158 |
| Stage 2 | 下 | 0.3211 | 0.3583 | +0.0372 |

Stage 越深，token 到所屬 cluster centroid 的距離整體越大。Stage 2 向下平移的平均距離增加最多（+0.0372），與其較低的 cosine、prediction consistency 和分類 accuracy 相符。

### 5.2 換群與同群樣本的分組統計

每張測試圖有四個平移方向，因此每個 stage 共 640 個 paired observations。下表不是只統計展示的 3+3 張案例，而是使用該 stage 的全部換群／同群 observations。

| Stage | 組別 | N | 平移前後 token cosine | 平移前 centroid distance | 平移後 centroid distance | 平移後 cluster margin | Prediction consistency |
|---|---|---:|---:|---:|---:|---:|---:|
| Stage 0 | 換群 | 23 | 0.9903 | 0.2838 | 0.2819 | 0.0210 | 95.65% |
| Stage 0 | 同群 | 617 | 0.9935 | 0.1954 | 0.1980 | 0.1531 | 97.24% |
| Stage 1 | 換群 | 45 | 0.9760 | 0.4512 | 0.4451 | 0.0127 | 73.33% |
| Stage 1 | 同群 | 595 | 0.9768 | 0.2895 | 0.2943 | 0.1186 | 79.50% |
| Stage 2 | 換群 | 71 | 0.9570 | 0.4371 | 0.4523 | 0.0176 | 64.79% |
| Stage 2 | 同群 | 569 | 0.9628 | 0.3066 | 0.3236 | 0.1021 | 60.46% |

主要差異不是平移前後 token cosine 大幅下降，而是換群樣本距離 centroid 較遠且 margin 很小：

- Stage 0 換群與同群的平移後 centroid distance 為 0.2819 對 0.1980；margin 為 0.0210 對 0.1531。
- Stage 1 為 0.4451 對 0.2943；margin 為 0.0127 對 0.1186。
- Stage 2 為 0.4523 對 0.3236；margin 為 0.0176 對 0.1021。

因此換群現象較符合「樣本位於 cluster 的外圍或 KMeans 決策邊界附近」，而不是平移後 feature 完全變成另一種不相似表徵。這是描述性統計；尚未進行跨 checkpoint 的顯著性檢定。

### 5.3 Token 與 Cluster Centroid 距離比較

以下將使用者關心的三種距離分開統計：

1. `原 token ↔ 移動後 token`：衡量平移前後表徵本身改變多少。
2. `原 token ↔ 原 cluster centroid` 與 `移動後 token ↔ 原 cluster centroid`：衡量平移後是否遠離原本群中心。
3. 只在換群案例中計算 `移動後 token ↔ 新 cluster centroid`：衡量換群後與新群中心的接近程度。

以下統一使用 cosine distance，定義為 `1 − cosine similarity`；數值越小代表越相似。

#### 全部案例（每個 Stage N=640）

| Stage | 原↔移動 Cosine distance | 原 token→原中心 | 移動 token→原中心 | 平移後遠離原中心的變化量 |
|---|---:|---:|---:|---:|
| Stage 0 | 0.00664 | 0.19860 | 0.20195 | +0.00335 |
| Stage 1 | 0.02322 | 0.30090 | 0.30673 | +0.00583 |
| Stage 2 | 0.03782 | 0.32109 | 0.34222 | +0.02113 |

結果顯示 Stage 越深：

- 平移前後 token 的 cosine distance 由 0.00664 增至 0.03782；
- 移動後 token 到原 cluster centroid 的距離逐漸增加；
- Stage 2 平移後到原中心的 cosine distance 比平移前增加 0.02113，是三層中最大。

#### 僅換群案例

| Stage | 換群 N | 原↔移動 Cosine distance | 原 token→原中心 | 移動 token→原中心 | 移動 token→新中心 | 新中心距離改善 |
|---|---:|---:|---:|---:|---:|---:|
| Stage 0 | 23 | 0.00975 | 0.28383 | 0.30676 | 0.28193 | 0.02483 |
| Stage 1 | 45 | 0.02395 | 0.45116 | 0.47075 | 0.44514 | 0.02561 |
| Stage 2 | 71 | 0.04297 | 0.43707 | 0.49130 | 0.45232 | 0.03898 |

「新中心距離改善」定義為 `移動後到原中心 cosine distance − 移動後到新中心 cosine distance`，全部為正值，符合 KMeans 將樣本指派給較近 centroid 的規則。例如 Stage 2 換群後，移動 token 到原中心為 0.49130，到新中心降為 0.45232。

這組統計可解讀為：

- Stage 0 換群時，平移前後 token 的 cosine distance 只有 0.00975，表示 feature 只小幅變化；但它到新中心的距離已略小於原中心，因此發生換群。
- Stage 1、2 的換群樣本本身離 cluster centroid 較遠，且平移後進一步遠離原中心。
- 新中心相對原中心只近約 0.025–0.039 cosine distance，差距不大，支持這些案例多位於兩群邊界附近，而不是強烈轉變成另一個 cluster 的典型樣本。

### 5.4 平移後分類為何改變

模型分類使用最後一層所有 token 的平均表徵，而前述 KMeans 只觀察物體中心附近的一個局部 token。下表因此另外比較最終分類表徵；每個 stage 均合併四個方向，共 640 筆。

| Stage | 位移 | Prediction consistency | 最終特徵 cosine | Logit cosine | 飛機機率平均變化 |
|---|---:|---:|---:|---:|---:|
| Stage 0 | 8 px | 97.19% | 0.9322 | 0.8497 | −0.0394 |
| Stage 1 | 16 px | 79.06% | 0.7766 | 0.6816 | −0.2598 |
| Stage 2 | 32 px | 60.94% | 0.6856 | 0.5781 | −0.4674 |

真正換分類的 observations，其最終特徵漂移明顯較大：

| Stage | 換分類 N | 換分類：最終特徵 cosine | 未換分類：最終特徵 cosine | 換分類：飛機機率變化 | 未換分類：飛機機率變化 |
|---|---:|---:|---:|---:|---:|
| Stage 0 | 18 | 0.6636 | 0.9399 | −0.2942 | −0.0320 |
| Stage 1 | 134 | 0.4025 | 0.8757 | −0.7783 | −0.1225 |
| Stage 2 | 250 | 0.4062 | 0.8648 | −0.8253 | −0.2380 |

但局部 token 是否換 KMeans cluster，不能充分解釋分類是否改變：

| Stage | P(換分類｜局部 token 換群) | P(換分類｜局部 token 同群) |
|---|---:|---:|
| Stage 0 | 4.35% | 2.76% |
| Stage 1 | 26.67% | 20.50% |
| Stage 2 | 35.21% | 39.54% |

Stage 2 即使局部 token 沒有換群，仍有 39.54% 的分類改變，略高於換群組的 35.21%。因此不能解讀成「KMeans 換群造成分類錯誤」。更合理的解釋是：平移使整張影像的多個 token 與最終表徵一起漂移；局部換群和分類改變是同一個平移敏感性的兩種結果。

造成此現象的可能機制如下：

1. 本模型在每一個 stage 加入 absolute positional embedding，同一物體移到新位置後會與不同的位置向量相加。
2. 2×2 patch merging 使用固定網格；平移後內容相對 merge 邊界的位置可能改變，差異會逐層傳播。
3. 訓練影像全部置中，沒有平移增強；上下移動尤其偏離訓練分布。Stage 2 向上、向下的 prediction consistency 分別只有 45.0% 與 32.5%，明顯低於左右的 84.38% 與 81.87%。
4. 背景裁切不是主要原因，因為所有原始影像內容均完整保留；但內容與 letterbox padding 的空間配置仍然改變。

推論時直接將 positional embedding 歸零並沒有一致改善：Stage 0、1 的 prediction consistency 反而下降，Stage 2 由 60.94% 提升至 66.88%。因為模型並非在此設定下訓練，這只能說明 absolute position 並非唯一因素，不能當作位置編碼的因果消融。正式驗證需要重新訓練「無 absolute position」或使用相對位置編碼的模型。

![平移後分類表徵分析](../plots/kmeans/caltech101_stage_aligned_one_patch/classification_analysis/classification_translation_analysis.png)

## 6. 換群案例圖

測試資料中不存在「同一張來源圖、同一方向」在 Stage 0、1、2 同時換群的案例，因此各 stage 分開選例。每個 stage 輸出三個換群案例及三個同群案例，且同一組內不重複使用來源影像。選擇規則為：先依平移後 cluster margin 由大到小排序，再依 cosine 與 source index 固定順序，避免人工挑選。

![Stage 0–2 cluster-change cases](../plots/kmeans/caltech101_stage_aligned_one_patch/cluster_change_cases/stage0-2_cluster_changes.png)

### Stage 0

- Source index：2140。
- 向右平移 8 pixels。
- Cluster 16 → Cluster 18。
- Cosine：0.9884。
- 平移後 cluster margin：0.0508。

![Stage 0 cluster change](../plots/kmeans/caltech101_stage_aligned_one_patch/cluster_change_cases/stage0_cluster_change.png)

### Stage 1

- Source index：2157。
- 向左平移 16 pixels。
- Cluster 30 → Cluster 27。
- Cosine：0.9769。
- 平移後 cluster margin：0.0492。

![Stage 1 cluster change](../plots/kmeans/caltech101_stage_aligned_one_patch/cluster_change_cases/stage1_cluster_change.png)

### Stage 2

- Source index：2072。
- 向右平移 32 pixels。
- Cluster 24 → Cluster 4。
- Cosine：0.9617。
- 平移後 cluster margin：0.0675。

![Stage 2 cluster change](../plots/kmeans/caltech101_stage_aligned_one_patch/cluster_change_cases/stage2_cluster_change.png)

每個 stage 固定使用兩個橫排：第一排為未移動的完整影像及 token 框、原 patch、原 cluster 代表 patches；第二排為移動後的完整影像、移動後 patch、新 cluster 代表 patches。第二排完整影像上以紅色箭頭與文字標記移動方向及 pixel 數。原 patch 與移動後 patch 的標題均標示其與 assigned cluster centroid 的 cosine distance。代表 patches 均為 training set 中距離固定 KMeans centroid 最近的樣本，沒有人工挑選。

### 每個 Stage 的三個換群／同群案例

- [Stage 0：三個換群案例](../plots/kmeans/caltech101_stage_aligned_one_patch/cluster_change_cases/stage0_changed_three_cases.png)
- [Stage 0：三個同群案例](../plots/kmeans/caltech101_stage_aligned_one_patch/cluster_change_cases/stage0_unchanged_three_cases.png)
- [Stage 1：三個換群案例](../plots/kmeans/caltech101_stage_aligned_one_patch/cluster_change_cases/stage1_changed_three_cases.png)
- [Stage 1：三個同群案例](../plots/kmeans/caltech101_stage_aligned_one_patch/cluster_change_cases/stage1_unchanged_three_cases.png)
- [Stage 2：三個換群案例](../plots/kmeans/caltech101_stage_aligned_one_patch/cluster_change_cases/stage2_changed_three_cases.png)
- [Stage 2：三個同群案例](../plots/kmeans/caltech101_stage_aligned_one_patch/cluster_change_cases/stage2_unchanged_three_cases.png)

每個案例也另外輸出獨立高解析度圖片，路徑為 `cluster_change_cases/stage<stage>/<changed|unchanged>/`。完整案例及選取順序記錄於 `cluster_change_cases/selected_cases.csv`。

## 7. 判讀

1. **局部內容大致穩定，但不是完全不換群。** Stage 0–2 的平均 cluster agreement 分別為 96.41%、92.97%、88.91%。
2. **換群不一定表示像素內容明顯改變。** 案例圖中的平移前後 patch 幾乎相同，但仍可能跨越 KMeans 邊界。
3. **移動各 stage 的一個 token 後，表徵穩定性逐步下降。** Cosine 與 cluster margin 皆變小；但因 pixel 位移同時由 8 增至 32，不能只歸因於網路深度。
4. **Cluster 穩定不代表分類穩定。** Stage 2 仍有約 89% cluster agreement，但 prediction consistency 平均只有約 61%；垂直向下平移時只有 32.5%。
5. **方向具有明顯影響。** Stage 1–2 的上下平移對分類結果影響遠大於左右平移，但 cluster agreement 沒有同比例下降。

## 8. 限制

- KMeans 使用所有 Caltech-101 training categories 建立，因此 cluster 代表圖可能來自其他類別，不一定都是飛機；這是全資料集共同特徵空間下的正常結果。
- Stage 2 有 3.12% 的目標 token receptive fields 接觸 padding，需另做排除 padding-contact 樣本的敏感度分析。
- 本輪只使用一個 checkpoint；正式統計推論仍需多個訓練 seeds。
- Stage 3 尚未完成 64-pixel 一 token 平移測試。
