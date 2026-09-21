# Caltech-101 平移實驗數據彙整

本文件集中記錄 Caltech-101 `airplanes` 平移實驗的模型設定、總體結果、換群／同群差異，以及後續可補充的表格。詳細方法、方向別結果與案例圖見 [完整實驗報告](0921_stage_translation_results.md)。

## 1. 實驗模型與資料設定

### 1.1 模型與訓練數據

| 項目 | 數值／設定 |
|---|---|
| 資料集 | Caltech-101 |
| 類別數 | 101 |
| 全資料集影像數 | 8,677 |
| Training images | 6,907 |
| Validation images | 1,770 |
| 平移實驗類別 | `airplanes` |
| 平移實驗 validation airplanes | 160 |
| 模型 | Hierarchical MergingViT |
| 模型輸入 | 224×224 RGB |
| 原圖內容尺寸 | 等比例縮放，長邊不超過 160 pixels |
| 背景處理 | 置中放入 224×224 safe-letterbox 畫布 |
| 初始 patch size | 8×8 pixels |
| Stage 數 | 4（本輪平移分析使用 Stage 0–2） |
| Embedding dimensions | 32、64、128、256 |
| Attention heads | 2、4、8、16 |
| Transformer depth | 每個 stage 1 block |
| Patch merging | Stage 間使用固定 2×2 merging |
| 位置編碼 | 每個 stage 使用 absolute positional embedding |
| 模型參數量 | 1,304,261 |
| Optimizer | AdamW，learning rate 3×10⁻⁴，weight decay 0.01 |
| Loss | Cross-entropy，label smoothing 0.1 |
| 設定訓練上限 | 100 epochs，early-stopping patience 20 |
| 最佳 checkpoint epoch | 66 |
| Random seed | 42 |
| 整體 validation accuracy | 62.09% |
| Airplanes validation accuracy | 98.75% |

整體準確度與 airplanes 準確度應分開報告：本次平移實驗只分析 airplanes，因此 98.75% 是實驗類別的原始分類基準；62.09% 則反映模型在全部 101 類上的整體能力。

### 1.2 各 Stage 平移設定

| Stage | Token 對應輸入範圍 | 平移距離 | Token 位移 | 測試方向 | 每方向樣本數 |
|---|---:|---:|---:|---|---:|
| Stage 0 | 8×8 px | 8 px | 1 token | 左、右、上、下 | 160 |
| Stage 1 | 16×16 px | 16 px | 1 token | 左、右、上、下 | 160 |
| Stage 2 | 32×32 px | 32 px | 1 token | 左、右、上、下 | 160 |

每個 stage 共形成 640 個 paired observations。所有影像的原始內容保留率均為 100%，因此物體被裁切不是本輪結果的主要原因。Stage 2 有 3.12% 的被分析 token receptive fields 接觸 letterbox padding，需在後續敏感度分析中另外分層。

KMeans 設為 `K=31`，各 stage 分開建立；只使用 6,907 張未平移 training images 的特徵 fitting。測試原圖及平移圖都只使用同一組固定 centroids 進行指派。

## 2. 總體平移前後比較

以下結果將四個方向合併；每個 stage 有 640 個 paired observations。

| Stage | 位移 | Cluster agreement | Matched-token cosine | AMI | ARI | Cluster margin | Prediction consistency | 平移後分類 accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Stage 0 | 8 px | 96.41% | 0.9934 | 0.9420 | 0.9347 | 0.1484 | 97.19% | 97.50% |
| Stage 1 | 16 px | 92.97% | 0.9768 | 0.8902 | 0.8725 | 0.1112 | 79.06% | 79.06% |
| Stage 2 | 32 px | 88.91% | 0.9622 | 0.8037 | 0.7809 | 0.0927 | 60.94% | 60.94% |

指標解讀：

- `Cluster agreement` 越高，代表平移前後仍被指派到相同 cluster 的比例越高。
- `Matched-token cosine` 越接近 1，代表平移前後對應 token 的表徵越相似。
- `AMI / ARI` 衡量整批樣本平移前後 cluster assignments 的一致性。
- `Cluster margin` 是最近與次近中心的距離差，越小代表越靠近 KMeans 決策邊界。
- `Prediction consistency` 衡量平移前後模型預測類別是否相同。
- `平移後分類 accuracy` 衡量平移後是否仍正確預測為 airplane。

從 Stage 0 到 Stage 2，cluster agreement 由 96.41% 降到 88.91%，但 prediction consistency 由 97.19% 降到 60.94%。這表示局部 KMeans cluster 的不變性下降較緩，最終分類結果對平移更敏感。

此表比較的是「每個 stage 各移動一個 token」。因 pixel 位移同時由 8、16 增至 32 pixels，因此下降不能單獨歸因於網路深度，也包含位移距離增加的影響。

## 3. Cluster 有變化／無變化的指標差異

### 3.1 局部 Token 指標

| Stage | 組別 | N | 平移前後 token cosine | 平移前 assigned-centroid cosine distance | 平移後 assigned-centroid cosine distance | 平移後 cluster margin | Prediction consistency |
|---|---|---:|---:|---:|---:|---:|---:|
| Stage 0 | Cluster 有變化 | 23 | 0.9903 | 0.2838 | 0.2819 | 0.0210 | 95.65% |
| Stage 0 | Cluster 無變化 | 617 | 0.9935 | 0.1954 | 0.1980 | 0.1531 | 97.24% |
| Stage 1 | Cluster 有變化 | 45 | 0.9760 | 0.4512 | 0.4451 | 0.0127 | 73.33% |
| Stage 1 | Cluster 無變化 | 595 | 0.9768 | 0.2895 | 0.2943 | 0.1186 | 79.50% |
| Stage 2 | Cluster 有變化 | 71 | 0.9570 | 0.4371 | 0.4523 | 0.0176 | 64.79% |
| Stage 2 | Cluster 無變化 | 569 | 0.9628 | 0.3066 | 0.3236 | 0.1021 | 60.46% |

最明顯的差異不是平移前後 token cosine，而是 centroid distance 與 cluster margin：

- Cluster 有變化的樣本原本就離所屬 centroid 較遠。
- 平移後 margin 約為 0.013–0.021，明顯小於無變化樣本的 0.102–0.153。
- 這表示多數換群案例靠近兩個 cluster 的決策邊界；feature 小幅移動便可能改由另一個 centroid 指派。
- Stage 2 的換群組 prediction consistency 反而高於同群組，因此不能將換群解讀為分類改變的直接原因。

### 3.2 僅限 Cluster 有變化案例的中心距離

以下全部使用 cosine distance（`1 − cosine similarity`），越小代表越接近。

| Stage | 換群 N | 原 token↔移動 token | 原 token→原中心 | 移動 token→原中心 | 移動 token→新中心 | 新中心距離改善 |
|---|---:|---:|---:|---:|---:|---:|
| Stage 0 | 23 | 0.00975 | 0.28383 | 0.30676 | 0.28193 | 0.02483 |
| Stage 1 | 45 | 0.02395 | 0.45116 | 0.47075 | 0.44514 | 0.02561 |
| Stage 2 | 71 | 0.04297 | 0.43707 | 0.49130 | 0.45232 | 0.03898 |

`新中心距離改善 = 移動 token→原中心 − 移動 token→新中心`。改善幅度只有約 0.025–0.039，表示換群後的新中心雖然較近，但優勢不大，再次支持「邊界附近換群」的解釋。

## 4. 其他表格提案

### 4.1 可直接由現有結果產生

| 優先度 | 表格 | 建議欄位 | 用途 |
|---:|---|---|---|
| 1 | 方向別穩定性 | Stage、方向、位移、agreement、cosine、margin、prediction consistency、accuracy | 呈現上下平移明顯比左右平移敏感 |
| 2 | 最終分類表徵漂移 | Stage、final-feature cosine、logit cosine、airplane probability change、prediction consistency | 說明分類改變與整體表徵漂移的關係 |
| 3 | 換群與換分類交叉表 | 換群且換分類、換群但未換分類、同群但換分類、同群且未換分類 | 驗證局部換群不是分類改變的充分或必要條件 |
| 4 | 錯誤類別轉移 | Stage、方向、錯誤預測類別、次數、比例 | 觀察平移後是否固定偏向 helicopter、ferry 等特定類別 |
| 5 | Cluster transition matrix | 原 cluster、新 cluster、轉移次數、平均 cosine、平均 margin | 找出不穩定的 cluster pair，而非只報告總 agreement |
| 6 | Padding 接觸分層 | Stage、padding contact、N、agreement、cosine、accuracy | 確認 Stage 2 結果是否受 padding 接觸影響 |

### 4.2 需要新增實驗

| 優先度 | 表格 | 需要的新增實驗 | 可回答的問題 |
|---:|---|---|---|
| 1 | 固定 pixel 位移的跨 Stage 比較 | Stage 0–2 都測 8、16、32 px | 分離「stage 深度」與「位移距離」效應 |
| 2 | 位移量曲線 | 增加 0、4、8、16、24、32 px | 穩定性是否隨位移量單調下降，是否存在臨界點 |
| 3 | KMeans K 敏感度 | 比較 K=8、16、31、64 | 換群率是否只是 cluster 數量造成的邊界效應 |
| 4 | 多 random seeds 與信賴區間 | 重訓至少 3–5 個 seeds | 結果是否跨模型初始化穩定，並提供 mean±SD／95% CI |
| 5 | 平移增強消融 | 重訓有／無 translation augmentation | 置中訓練偏差是否是分類不穩定的主因 |
| 6 | 位置編碼消融 | 重訓 absolute、none、relative position 三種模型 | 檢驗 absolute positional embedding 的因果影響 |
| 7 | Patch-merging 消融 | 改變 merging phase 或使用 anti-aliasing／overlap merging | 固定 2×2 merging 邊界是否造成深層表徵漂移 |

### 4.3 建議報告主文保留的最小表格組合

若報告篇幅有限，建議正文保留四張表：

1. 模型與資料設定表。
2. Stage 0–2 總體平移比較表。
3. Cluster 有變化／無變化分組表。
4. 方向別 prediction consistency 與 accuracy 表。

其餘中心距離、錯誤類別轉移、cluster transition matrix 及逐樣本數據可放入附錄。

## 5. 對應檔案

- 完整彙總指標：`plots/kmeans/caltech101_stage_aligned_one_patch/aggregate_metrics.csv`
- 逐樣本 KMeans 指標：`plots/kmeans/caltech101_stage_aligned_one_patch/sample_metrics.csv`
- 分類彙總：`plots/kmeans/caltech101_stage_aligned_one_patch/classification_analysis/classification_metrics.csv`
- 逐樣本分類指標：`plots/kmeans/caltech101_stage_aligned_one_patch/classification_analysis/classification_sample_metrics.csv`
- 換群與換分類關聯：`plots/kmeans/caltech101_stage_aligned_one_patch/classification_analysis/cluster_prediction_association.csv`
- 分類錯誤轉移：`plots/kmeans/caltech101_stage_aligned_one_patch/classification_analysis/prediction_transitions.csv`
