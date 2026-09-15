# 第八階段：正確逐字稿統整與執行路線圖

本文件根據以下資料整理：

- `summaries/2026-09-02 21-30-47.pdf`
- 目前 `merge_VIT` 衍生分支的程式碼與既有摘要

這份逐字稿比先前版本更接近專案的實際交接內容，應作為後續理解與改進的主要依據。

## 一、專案目前的研究定位

本專案不是要重做 Swin Transformer 或 PVT，而是實作一個較簡化的階層式 ViT，並研究其可解釋方式。

模型的核心是：

1. 將輸入影像切成 patch。
2. 依序經過多個 Transformer stage。
3. 在 stage 之間將相鄰 patch 合併。
4. 取出各 stage、位置或 attention head 的 token 表示。
5. 對 token 做 K-means，找出 representative patch。
6. 用人工判讀、Grad-CAM 與 VLM 協助解釋模型逐層形成的特徵。

研究主張應是：模型能否提供由低階局部特徵，逐步組合到高階語義的可追蹤證據，而不只是最後分類正確。

## 二、目前模型與程式碼已實作的內容

### MergingViT

`models/MergingViT.py` 已經實作：

- patch embedding
- 多 stage Transformer block
- stage-specific positional embedding
- flexible patch merging
- global average pooling 與分類 head

目前設計與逐字稿相符的地方：

- 預設概念是四個 stage。
- `merge_size` 可用 `(2, 2)`，也可設定為各 stage 不同的 `(1, 4)`、`(4, 1)` 等組合。
- 合併後會沿 channel 串接，再經 LayerNorm 和 Linear reduction。
- attention 會保留 `individual_heads_output` 與 `pre_projection_features`，供 token / head 分析使用。

### 訓練與資料

- `train.py` 會由 `config.py` 建立模型、資料集、optimizer、scheduler 和 checkpoint。
- 每次訓練會保存模型程式與當次 config 到 `runs/train/exp*/`。
- `dataloader/` 已經有 CIFAR10、Caltech101、MedMNIST、NonclassicFace 等資料集。
- 非典型人臉資料集的資料載入器已存在，但逐字稿顯示其 K-means 與完整解釋分析尚未完成。

### 訓練後分析

`run_experiment_pipeline.py` 已把下列流程串接：

1. K-means 與 representative patch / Grad-CAM trace 產生。
2. 使用 VLM 對輸出圖像產生 caption。
3. 匯出 JSONL 與 CSV 摘要。

## 三、逐字稿確認的關鍵研究結論

### 1. 這不是 Swin 或 PVT 的直接實作

模型只借用 hierarchical patch merging 的概念：

- 沒有 Swin 的 shifted-window attention。
- 沒有 PVT 的 Spatial Reduction Attention。
- 因此比較時應稱為「受 hierarchical ViT 啟發的自訂 MergingViT」，不應宣稱等同 Swin 或 PVT。

### 2. Attention Rollout 不是主解釋方法

模型使用 global average pooling，不使用 CLS token。  
因此依賴 CLS token 回溯的 Attention Rollout 不適合直接當主要證據。

目前較合理的主線是：

- token / head 特徵分群
- patch merge 的父子對應
- representative patch
- Grad-CAM 作為輔助排序

### 3. merge 形狀影響語義視野，但暫未顯著影響分類準確率

逐字稿的觀察是：

- `(2, 2)`、`(1, 4)`、`(4, 1)` 等 merge 組合在 accuracy 上差異不大。
- 不同 merge 形狀會改變 patch 的長寬比與 receptive field。
- 因此它更可能影響可見特徵和後續語義分析，而不是單純分類性能。

這應成為後續實驗的假說，而不是已被驗證的結論。

### 4. 目前的 K-means 是 position-wise clustering

在某一 stage 的固定位置，收集所有樣本該位置的 token 後再分群。

這種方法在物體多半置中的簡單資料集上可行，但有明顯限制：

- 同一物體若出現在不同位置，相關 token 可能無法被分到同一群。
- 背景或位置偏差可能主導 cluster。
- 在複雜場景、多物件、強平移變化資料集上會失效或難以解釋。

### 5. 現有解釋仍高度依賴人工判讀

目前存在以下問題：

- head clustering 常只能用肉眼做主觀解讀。
- cluster 數量多半是經驗設定，尚無選擇依據。
- Grad-CAM 有時會選到背景，只適合協助挑選優先檢視的 patch。
- VLM 在低解析度 representative patch 上描述偏籠統；Qwen 效果不穩，較大型模型的描述較好但成本更高。

### 6. 真正要展示的是 bottom-up 的組合式解釋

逐字稿中老師期待的不是只說「最後一層看到飛機」，而是可以說明：

1. 低階 stage 捕捉到色塊、邊緣、曲線、紋理。
2. 中階 stage 將局部特徵組合成部件。
3. 高階 stage 將部件組合為物體結構。
4. 這條組合路徑如何支持最終分類。

因此，後續最重要的產出不應只是圖片牆，而應是可追溯的 patch lineage。

## 四、目前最重要的缺口

### A. 分群缺少量化與評估

尚未回答：

- 為何每個 stage / head 要設定這個 cluster 數？
- cluster 是否緊密且可分？
- 不同 random seed、樣本子集或模型 checkpoint 下，cluster 是否穩定？
- cluster 是否與類別、物件位置或語義部件有關？

### B. 空間位置偏差

position-wise clustering 預設物件具有位置一致性。  
這不適合一般自然影像，也限制了對 Caltech101 或非典型人臉的解釋品質。

### C. 低解析度造成後段分析失真

28x28 輸入經多次 merge 後，token grid 很快縮小。  
這會使 representative patch、反推區域和 VLM 輸入都過於粗糙。

### D. 實驗可重現性不足

逐字稿直接指出：

- config 常手動改動。
- inference 抽樣數量 / seed 未完整記錄。
- cluster 數與其他分析參數缺少系統性實驗紀錄。
- wandb 容量有限，且未被當作唯一可回溯的實驗帳本。

### E. 程式碼仍偏研究原型

核心流程可用，但資料、訓練、K-means、VLM 與圖像輸出之間的設定仍有分散情況。  
對接手者而言，重現某張圖需要知道 checkpoint、run config、dataset split、seed、抽樣樣本與分析設定。

## 五、建議未來規劃

## Phase 0：先建立可重現基線

目標：任何一份實驗結果都能被重新產生。

工作：

1. 為每次訓練與分析建立 `experiment_manifest.json`。
2. 固定保存 Git commit、checkpoint 路徑、config 快照、dataset split、seed、樣本索引與 K-means / VLM 參數。
3. 將 analysis output 依 run id 分目錄保存。
4. 對每張輸出圖加入 image id、stage、block、head、position 與 parent token 資訊。

完成標準：指定一個結果目錄後，可在不猜參數的情況下重跑。

## Phase 1：量化 K-means 與 head 分析

目標：把「看起來有差異」改成可檢驗的主張。

工作：

1. 比較不同 cluster 數，例如 silhouette score、Davies-Bouldin index、Calinski-Harabasz score。
2. 用多個 seed 和樣本子集衡量 cluster 穩定度，例如 ARI 或 NMI。
3. 報告 cluster 與類別的關聯，例如 purity、NMI 或 class entropy。
4. 對不同 attention head 使用一致的量化指標，而不是人工挑選案例。

完成標準：每個 stage / head 的 cluster 數都有明確選擇依據與穩定度報告。

## Phase 2：建立 merge lineage 與 bottom-up 解釋

目標：讓高階 token 能回溯到低階組成 patch。

工作：

1. 在每次 `FlexiblePatchMerging` 保存 child-to-parent 的索引對應。
2. 將最終 stage 的候選 token 展開成前一 stage 的子 token。
3. 一路回溯到原始 patch grid，產生樹狀或路徑式輸出。
4. 將 representative patch、cluster id、分類 logit 變化與 lineage 放在同一份報表。

完成標準：能對單一預測展示「低階特徵 -> 中階部件 -> 高階物體」的完整圖與資料表。

## Phase 3：處理位置偏差

目標：避免同一語義因位置不同而被切成不同 cluster。

優先嘗試：

1. 對齊或裁切物件後再做 position-wise clustering。
2. 加入資料增強後，比較 cluster 是否仍受位置主導。
3. 新增 non-position-wise token clustering 作為對照。
4. 以 object-level feature 或 proposal / segmentation 對齊後再分群。

完成標準：在物件非置中的資料上，能比較 position-wise 與 position-agnostic 方法的差異。

## Phase 4：改善資料與 VLM 證據品質

目標：讓分析輸入具有足夠語義資訊。

工作：

1. 以較高解析度資料集建立主實驗，例如 Caltech101 或可公開重現的細粒度資料集。
2. 保留 28x28 資料集作為易讀的機制驗證，不作為唯一主結論。
3. 建立小型人工標註集，評估 VLM caption 是否與 representative patch 的語義一致。
4. 讓 VLM 只處理經量化挑選的候選，而不是全量輸出。

完成標準：VLM 成為可評估的輔助證據，而非不可驗證的敘述來源。

## Phase 5：模型與消融實驗

目標：分清模型分類能力與解釋能力的貢獻。

至少比較：

- `MergingViT` 與 plain ViT
- `MergingViT` 與 Swin / PVT 類 baseline
- `(2, 2)`、`(1, 4)`、`(4, 1)` 及混合 merge 設定
- token mode 與 head mode clustering
- 不同 stage depth、head 數與 patch size

主要報告：

- accuracy / macro-F1
- 參數量、訓練成本與分析成本
- cluster 品質與穩定度
- lineage 可視化案例
- 人工或半自動解釋評估

## 六、推薦的近期執行順序

1. 建立實驗 manifest 與固定抽樣紀錄。
2. 補 K-means 的 cluster 數選擇與穩定度量化。
3. 在 MergingViT 補完整的 merge lineage 輸出。
4. 先以一個低解析度資料集驗證 bottom-up 解釋格式。
5. 再轉到較高解析度、物件位置更多變的資料集。
6. 最後比較 merge 設定與 hierarchical baseline。

## 七、現階段應避免的做法

- 不要只用幾張代表圖和人工描述就宣稱 cluster 具語義。
- 不要只報分類 accuracy 就宣稱 merge 設計有效。
- 不要把 VLM caption 當作唯一或主要的可解釋性證據。
- 不要在缺少 seed、config 和樣本索引的情況下保存分析結果。
- 不要把自訂 MergingViT 直接描述為 Swin 或 PVT。

## 結論

這個 repo 的最佳延伸方向不是再堆更多模型，而是將既有的 MergingViT、K-means、Grad-CAM 與 VLM 工具鏈，轉為可重現、可量化、可回溯的階層式解釋框架。

短期最有價值的工程工作是實驗紀錄、cluster 量化和 merge lineage。  
這三件事完成後，才有足夠證據去討論不同 merge 形狀、head 行為與階層式語義是否真的成立。
