# 第七階段：Hierarchical ViT 統整與未來規劃

這份摘要是根據：

- `summaries/Hierarchical_ViT_Transcript_and_Outline.pdf`
- 目前 repo 的 `MergingViT`、資料層、訓練流程與分析腳本

整理出來的整體理解與後續方向。

## 一、目前統整

這份逐字稿的核心想法很清楚：  
把 ViT 從單一平面式 token 流程，改成「分層式、可解釋、可分析」的 hierarchical ViT。

### 1. 架構主軸

1. `Patch Embedding`
2. 多個 stage 的 Transformer block
3. stage 間做 `Patch Merging`
4. 最後用 `GAP + Linear` 做分類

這和目前 repo 的 `MergingViT` 很接近。

### 2. 為什麼要這樣做

逐字稿裡反覆強調幾個問題：

- 傳統 ViT 太依賴 `CLS token`
- 層級結構不足，語義與空間關係不夠明確
- 需要更好地看見每個 stage 的 token 變化
- 需要把模型內部表示和可解釋性工具串起來

### 3. 目前 repo 已經對上的部分

現在這個 repo 已經具備幾個關鍵元素：

- `MergingViT` 的 stage-wise 架構
- 彈性的 `FlexiblePatchMerging`
- attention 內部輸出可取
- `run_experiment_pipeline.py` 可做 K-means / Grad-CAM / VLM 分析
- `summaries/` 已經可以記錄分階段理解

## 二、逐字稿中最有價值的概念

### 1. 不只看分類，還要看 stage 表示

重點不是只看最後準確率，而是要理解：

- 每個 stage 的 token 長什麼樣
- merge 前後的資訊如何變化
- 哪些 patch 變成代表性 patch

### 2. Position-wise clustering

逐字稿提到把 token 依位置與 stage 做 clustering。  
這表示分析單位不是整張圖，而是：

- 某個 stage
- 某個位置
- 某個 head 或 token 群

這很適合接在現有的 K-means 工具上。

### 3. Representative patch

每群挑代表 patch，可以把抽象 token 變成可看的影像片段。  
這會是後續把「模型理解」轉成「人能理解」的核心步驟。

### 4. Receptive field 與語義遞進

逐字稿想處理的問題是：

- 低階 stage 偏局部紋理
- 高階 stage 偏語義與結構
- merge 後 receptive field 會變大

這正是 hierarchical design 的價值。

## 三、目前缺口

現在 repo 雖然已經有訓練和分析雛形，但還缺幾個完整鏈條：

1. stage-by-stage 的可視化還不夠系統化
2. token / head 的聚類結果還沒完全形成統一報表
3. 不同解釋方法之間的對照還不夠完整
4. hierarchical 語義是否真的比 baseline 更清楚，還需要更正式的評估

## 四、未來規劃

### Phase 1：把結構看清楚

目標：

- 固定每個 stage 的 token 形狀與輸出
- 明確記錄 merge 前後的 resolution
- 把 stage-wise feature dump 出來

建議動作：

- 在 `MergingViT` 加更完整的中間層輸出開關
- 讓分析腳本能直接讀 stage 變化

### Phase 2：把分析鏈補齊

目標：

- K-means 只做一次，但輸出要完整
- 每個 stage 都有 representative patches
- 每個 head / token 群都能對應到影像區域

建議動作：

- 統一 `run_experiment_pipeline.py` 的輸出格式
- 增加 stage-level summary CSV
- 補上可直接讀的圖像索引與位置標記

### Phase 3：把解釋方法串成比較框架

目標：

- Attention Rollout
- Grad-CAM
- token clustering
- VLM caption

都能針對同一批樣本比較。

建議動作：

- 建立同一份 sample index 的對照輸出
- 每種方法輸出同樣的 stage / image / position 標記

### Phase 4：做模型與資料的對照實驗

目標：

- 比較 `MergingViT` 和 `Swin` / `PVT` 類型模型
- 比較不同資料集上的 stage 表現
- 看 hierarchical design 是否真的提升可解釋性

建議動作：

- 固定訓練設定做 ablation
- 比較不同 `merge_size`
- 比較不同 dataset 的 stage 分布

## 五、建議的下一步

如果要最有效率地往下做，順序建議是：

1. 固定 `MergingViT` 的 stage 輸出格式
2. 補 stage-level 的分析報表
3. 把 K-means / representative patch 整成單一輸出規格
4. 再做 Grad-CAM / VLM 的交叉比較

## 結論

這份逐字稿想推的是一個方向：  
把 ViT 做成「分層、可追蹤、可視覺化、可對照」的模型，而不是只看最後分類結果。

就目前 repo 來看，`MergingViT` 已經是最接近這個方向的實作，  
接下來最重要的是把中間表徵的輸出與分析流程正式化。
