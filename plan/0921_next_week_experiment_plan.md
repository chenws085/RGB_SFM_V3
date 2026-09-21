# 0921 本週實驗規劃：Caltech-101 影像平移下的 KMeans Cluster 穩定性

## 一、研究範圍

本週使用 Caltech-101 進行「完整影像平移」實驗，觀察階層式 Vision Transformer 在輸入影像平移後的 token 表徵與固定 KMeans cluster 指派是否穩定。結果展示先以 `airplanes` 類別作為主要案例，並保留其他類別的整體統計作為參考。

「將完全相同的像素 patch 放到另一個 token 位置」的位置控制實驗延後，不列入本週工作與主要結論。

## 二、主要研究問題

將完整影像平移後，物體質心所在位置的 token feature，是否仍會被「僅以原始訓練影像建立」的固定 KMeans 模型指派至相同 cluster？此穩定性是否會隨模型 stage、平移方向、平移距離及資料集複雜度而改變？

本週實驗不能單獨解釋純粹的絕對位置效應，因為完整影像平移可能同時改變：

- 物體在 token grid 中的位置；
- 物體與 patch boundary 的相對關係；
- zero padding 與邊界裁切內容；
- 各 stage 的 receptive field 內容；
- 深層 patch merging 的輸入組合。

因此本週結論應描述為「完整影像平移穩定性」，不可描述為「相同局部內容換位置後的純位置敏感性」。

## 三、KMeans 訓練與測試流程

### 3.1 正式流程

1. 使用未平移的原始訓練圖片擷取各 stage 的 token features。
2. 對各 stage features 做 L2 normalization。
3. 各 stage 分別建立 KMeans，主實驗使用 `K=31`。
4. KMeans 擬合完成後固定 centroids，不再更新。
5. 將未平移的原始測試圖與平移後的測試圖分別輸入同一模型。
6. 取各影像「物體質心所在位置」的 token feature。
7. 使用同一組固定 centroids 指派原圖與平移圖的 cluster。
8. 以相同來源影像配對，比較平移前後的 cluster、feature 與分類結果。

簡化流程：

```text
原始訓練圖 → feature extraction → KMeans fit → 固定 centroids
                                             ├→ 原始測試圖 → cluster A
                                             └→ 平移測試圖 → cluster B
                                                                ↓
                                                         配對比較 A、B
```

### 3.2 資料洩漏限制

- 平移後圖片不得加入 KMeans fitting data。
- 測試圖片（包括未平移版本）不得加入 KMeans fitting data。
- normalization、PCA（若使用）與 KMeans 都只能在原始 training split 上擬合。
- 原圖與其所有平移版本必須保留相同 `source_id`，並位於同一資料切分。
- KMeans random seed、模型 checkpoint 與資料索引必須寫入執行紀錄。

## 四、本週實驗設計

### 4.1 實驗 A：Caltech-101 平移測試

使用專案現有 Caltech-101 dataloader 與訓練完成的 MergingViT checkpoint。模型輸入大小為 `224×224`；正式執行前先記錄 checkpoint 的 validation accuracy，並確認 `airplanes` 類別在 validation/test split 的樣本數與原始影像索引。

KMeans fitting data 使用 Caltech-101 原始 training split 的所有類別，不只使用飛機圖片，以維持模型整體特徵空間的 cluster 定義。平移穩定性先在 held-out `airplanes` 圖片上做案例分析，再補全 validation/test split 所有類別的整體結果。若研究目的後續改成只研究飛機內部的局部結構，才另開一個明確命名的 airplane-only KMeans 消融，不能和主要結果混用。

本輪對每個 stage 平移「該 stage 的一個 token」，不做多距離比較。由於每次 `2×2` patch merging 都使有效 patch size 加倍，平移條件為：

- 基準：`(dx, dy)=(0, 0)`；
- Stage 0：水平／垂直移動 `±8 pixels`；
- Stage 1：水平／垂直移動 `±16 pixels`；
- Stage 2：水平／垂直移動 `±32 pixels`；
- Stage 3 理論上需移動 `±64 pixels`，但在 224×224 輸入及統一影像尺度下無法同時保留完整內容，因此不列入本輪主實驗。

程式必須由 checkpoint 的 `patch_size` 與 stage index 計算 `stage_patch_size = patch_size × 2^stage`，不可分別寫死位移值。平移後比較的 token index 也必須沿平移方向恰好移動一格。

分析 stage：程式索引 `stage=0,1,2,3`。報告全程使用同一套 0-based stage 命名，避免程式 Stage 0 與報告 Stage 1 混淆。

### 4.2 避免平移造成背景裁切

目前 Caltech-101 測試前處理為 `Resize(256) → CenterCrop(224)`，CenterCrop 在平移前就會丟失一部分原圖背景，因此不適合作為本實驗的配對影像產生方式。主實驗改用「完整影像等比例縮放＋安全邊界 letterbox」：

1. 將原始 RGB 圖片等比例縮放，使長邊不超過 `160 pixels`。
2. 不裁切原圖，把縮放後的完整圖片置中貼到 `224×224` 畫布。
3. 置中後至少有一個方向具備 `32 pixels` 邊界，因此 Stage 0–2 在 `±8/16/32 pixels` 平移時，所有原始影像像素都留在畫布內。較短邊會有更大的邊界。
4. 先在未 normalize 的 RGB 空間完成 letterbox 與平移，再套用模型使用的 ImageNet normalization。
5. 原圖與所有平移版本使用完全相同的縮放比例、插值法、畫布值與 normalization，只改變貼入畫布的 `(x, y)` offset。
6. 每一對影像以像素 mask 驗證內容守恆：`source_pixels_retained=1.0`、非 padding 像素數一致；不符合者中止或排除，不進入主結果。

主畫布填充值採用「ImageNet mean 對應的 RGB 值」，normalize 後接近 0，可避免純黑邊框造成過強的人造邊界。另做兩個小型敏感度條件：

- `edge-color padding`：以原圖四邊像素的穩健平均色填滿畫布；
- `reflection padding`：只作敏感度比較，不當主結果，因為會生成原圖不存在的鏡射內容。

不要使用 circular shift 作為主實驗，因為它雖不丟失像素，卻會把右側背景繞到左側，產生不自然的空間關係。也不使用生成式擴圖或 inpainting，因為新生成背景會引入另一個不可控變因。

必須注意：安全邊界方案會讓物體與原始背景完整保留，但畫布中仍包含人工 padding。為確認結果不是 padding 邊界造成，需額外記錄 token receptive field 是否碰到 padding，主表優先報告未碰到 padding 的物體 token，全部 token 結果放入敏感度分析。

### 4.3 前處理一致性與基準檢查

- KMeans fitting 使用的原始 training images 也必須採用同一套 `160-in-224 letterbox`，但不施加平移。
- 原始與平移測試圖皆使用該 letterbox 流程，不能以 CenterCrop 原圖對照 letterbox 平移圖。
- 先比較既有 `Resize→CenterCrop` 與新 letterbox 的未平移分類 accuracy。若 letterbox 造成明顯 accuracy 降低，需將此 distribution shift 列為限制，並考慮以相同 letterbox 前處理微調模型；不能直接把 accuracy 降低解釋為平移效應。
- 固定 resize interpolation、antialias 設定及 rounding 規則，保存縮放後尺寸、四邊 margin 與實際 offset。
- 每個 airplane case 都保存原圖、未平移 letterbox 圖、平移圖與 padding mask 的 montage，供人工核對。

注意事項：

- Caltech-101 的 224×224 模型在 Stage 3 輸出約為 `4×4`；其一個 token 對應 64 pixels。由於本輪統一尺度只預留 32-pixel 邊界，Stage 3 暫不做一個 token 的平移比較。
- 每筆樣本記錄原始內容保留比例、padding 比例及目標 token 是否碰到 padding。
- 主分析只接受原始內容保留比例 `=1.0` 的樣本；裁切版本另列為對照，不混入主表。
- 各 shift 必須回報有效樣本數與有效比例，不能只呈現百分比指標。

### 4.4 裁切對照實驗

為量化「不預留邊界」會造成多少影響，另保留一組次要對照：

- 使用既有 `Resize(256) → CenterCrop(224)` 影像後再 zero-fill 平移；
- 逐張計算保留像素或可用 annotation mask 時的物體保留率；
- 與安全邊界主實驗使用相同 source IDs、shifts、checkpoint 與固定 KMeans；
- 結果標記為 `cropped-control`，只用於顯示背景／物體裁切對穩定性指標的影響。

主結論來自 `safe-letterbox`；`cropped-control` 不與主結果合併平均。

## 五、評估指標

### 5.1 主要指標

- `cluster_agreement`：同一來源影像平移前後被指派至相同 cluster 的比例；
- `matched_token_cosine`：配對 token features 的 cosine similarity；
- `prediction_consistency`：平移前後模型預測類別相同的比例；
- `classification_accuracy`：各平移條件下的分類正確率。

### 5.2 輔助指標

- AMI 與 ARI：比較一個條件內所有來源影像平移前後的 cluster assignments；
- normalized Euclidean distance；
- 原 cluster centroid 距離的變化；
- 最近與次近 centroid 的距離差（cluster margin）；
- confidence change；
- source-content retained fraction；
- padding-contact rate；
- paired bootstrap 95% confidence interval。

AMI、ARI 是整組 assignments 的一致性指標，不是單筆樣本指標；逐樣本結果仍以 same-cluster、cosine、距離與 margin 為主。

## 六、結果判讀規則

- cosine 高、cluster 改變且 margin 小：較支持 KMeans 邊界敏感，而非表徵大幅改變。
- cosine 與 cluster agreement 同時下降：支持 token representation 對完整影像平移敏感。
- prediction consistency 或 accuracy 同時下降：表示問題不只在解釋方法，分類模型本身也不穩定。
- 只在 cropped-control 惡化、safe-letterbox 穩定：結果主要來自背景或物體裁切。
- safe-letterbox 仍惡化，但碰到 padding 的 token 特別明顯：可能是人工畫布邊界效應。
- safe-letterbox 中未碰 padding 的 token 仍惡化：較支持模型表徵對完整內容平移敏感。
- Stage 3 結果只解讀為全域表徵穩定性，不與 Stage 0–2 的空間 token 對應作完全相同的因果解讀。

## 七、KMeans 敏感度分析

主結果固定使用 `K=31`。完成主實驗後，再於相同 checkpoint 與資料切分上測試：

```text
K ∈ {8, 16, 31, 64}
```

每一個 K 都必須重新只用原始訓練圖擬合，並保持測試時 centroids 固定。比較不同 K 下的趨勢是否一致，不以其中表現最好的一個 K 取代主結果。

## 八、本週執行順序

| 優先級 | 工作 | 完成條件 |
|---|---|---|
| P0 | 統一 stage 命名、加入 `source_id` 與逐樣本輸出 | 可從結果追溯原圖、shift、stage 與 cluster |
| P0 | 建立 Caltech-101 safe-letterbox 配對資料 | ±32 px 內原始內容保留率皆為 1.0，像素守恆測試通過 |
| P0 | 篩選 held-out `airplanes` 並保存 source IDs | 每張案例可追溯至原始 Caltech-101 index |
| P0 | 執行 Stage 0/1/2 的 ±8/16/32 px 測試 | 各 stage 左、右、上、下皆恰好移動一個 token |
| P0 | 增加 padding-contact 與 cluster margin | 分開輸出未碰 padding及所有 token 結果 |
| P0 | 執行 Stage 0–2 | 每個條件均有有效樣本數、cluster、feature、prediction 指標 |
| P1 | 執行 K 敏感度分析 | 完成 K={8,16,31,64} 趨勢比較 |
| P1 | 執行 cropped-control | 量化傳統平移中裁切造成的額外下降 |
| P1 | 擴充到 Caltech-101 全類別 | 確認 airplanes 案例是否符合整體趨勢 |

## 九、本週不執行項目

- 不做「完全相同像素 patch 放到另一 token 位置」的 controlled-position 實驗；
- 不以 controlled-position 的因果語言解讀完整影像平移結果；
- 不將平移圖片加入 KMeans fitting；
- 不在本週加入 PartImageNet、SAM、VLM 或 head specialization；
- 不進行所有資料集、K、padding 與架構消融的完整笛卡兒積。

上述項目保留為後續研究規劃。

## 十、交付物

- `sample_metrics.csv`：每個 source × class × preprocessing × shift × stage 的逐樣本結果；
- `aggregate_metrics.csv`：各 dataset × stage × shift × threshold × K 的彙總結果；
- `run_manifest.json`：checkpoint、seed、資料索引、KMeans fitting 範圍與參數；
- stage × shift magnitude 的 cluster agreement 曲線；
- stage × shift magnitude 的 cosine 與 prediction consistency 曲線；
- cluster margin 對換群機率的分析圖；
- 依固定規則自動選出的換群案例，不人工挑選；
- Caltech-101 airplanes 案例圖與全類別對照表；
- safe-letterbox 與 cropped-control 對照表；
- 一段清楚區分「表徵改變、KMeans 邊界敏感、分類不穩定、輸入裁切」的結論。

## 十一、預期結論格式

報告時不只寫「是否落在同一 cluster」，而依序回答：

1. 平移距離增加時，各 stage 的 cluster agreement 是否下降？
2. 換群時 token cosine 與 cluster margin 是否也明顯下降？
3. safe-letterbox 保留全部原始內容後，下降是否仍然存在？
4. 模型分類結果是否同步改變？
5. airplanes 案例是否符合 Caltech-101 全類別的整體趨勢？

只有在後續完成「相同像素 patch 換位置」的控制實驗後，才進一步回答純粹的絕對位置與模型架構效應。

## 十二、實驗版本紀錄

- 第一輪使用所有 stages 固定平移 8 pixels。該設計只有 Stage 0 等於移動一個 stage patch；Stage 1–3 分別只移動 0.5、0.25、0.125 個 stage patch，因此結果保留為「固定輸入位移 8 px」探索性紀錄，不納入新的逐 stage 一個 patch 主表。
- 修正版使用統一的 `160-in-224 safe-letterbox`，Stage 0/1/2 分別平移 8/16/32 pixels。Stage 3 的 64-pixel 條件延後規劃。

## 十三、修正版正式實驗紀錄

執行設定：

- checkpoint：`runs/train/mergingvit_caltech101_letterbox160/MergingViT_best.pth`；
- checkpoint epoch：66；
- Caltech-101 整體 validation accuracy：62.09%；
- airplanes validation accuracy：98.75%；
- KMeans：只用 6,907 張未平移 training images 擬合，`K=31`；
- 測試：160 張 held-out airplanes；
- 所有配對的 `source_pixels_retained=1.0`；
- Stage 0/1/2 × 四方向的 token index 驗證皆為 160/160 精確移動一格。

Cluster agreement：

| Stage | 位移 | 左 | 右 | 上 | 下 | 四方向平均 |
|---|---:|---:|---:|---:|---:|---:|
| Stage 0 | 8 px | 98.12% | 98.75% | 93.75% | 95.00% | 96.41% |
| Stage 1 | 16 px | 92.50% | 93.13% | 92.50% | 93.75% | 92.97% |
| Stage 2 | 32 px | 86.25% | 87.50% | 92.50% | 89.38% | 88.91% |

Matched-token cosine：

| Stage | 左 | 右 | 上 | 下 |
|---|---:|---:|---:|---:|
| Stage 0 | 0.9956 | 0.9929 | 0.9920 | 0.9929 |
| Stage 1 | 0.9797 | 0.9775 | 0.9764 | 0.9736 |
| Stage 2 | 0.9550 | 0.9676 | 0.9723 | 0.9538 |

Prediction consistency：

| Stage | 左 | 右 | 上 | 下 |
|---|---:|---:|---:|---:|
| Stage 0 | 97.50% | 97.50% | 96.25% | 97.50% |
| Stage 1 | 93.75% | 95.63% | 71.25% | 55.63% |
| Stage 2 | 84.38% | 81.87% | 45.00% | 32.50% |

初步判讀：對齊各 stage 的有效 patch size 後，Stage 0–2 的 matched token 表徵與 KMeans cluster 大致穩定，且穩定性隨 stage 加深而緩慢下降；但分類預測對較大的垂直平移明顯不穩定。這表示「局部 token cluster 穩定」不等於「整體分類平移穩定」，兩者需分開報告。
