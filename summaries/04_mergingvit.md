# 第四階段：MergingViT 模型

這一階段看的是 `models/MergingViT.py`。

## 這一層在做什麼

`MergingViT` 是這個 repo 裡最重要的 Transformer 模型。  
它把影像切成 patch 後，經過多個 stage 的 Transformer block，再做彈性 patch merging，最後輸出分類結果。

## 模型組成

### 1. `PatchEmbed`

- 用 `Conv2d` 把影像切成 patch
- 轉成 token 序列
- 做 `LayerNorm`

### 2. `Attention`

標準 multi-head attention，但額外保留兩個中間結果：

- `individual_heads_output`
- `pre_projection_features`

這兩個輸出是後面 K-means / head analysis 會用到的重點。

### 3. `TransformerBlock`

結構是典型 ViT 的 pre-norm block：

- `LayerNorm`
- `Attention`
- `LayerNorm`
- `MLP`
- `DropPath`

### 4. `FlexiblePatchMerging`

這是這個模型最特別的地方。

- 支援非固定大小的 merge
- 可對 `H x W` 做 padding
- 合併鄰近 patch 後把 channel 拉高
- 再用 Linear 做降維

它不只支援 `(2, 2)`，也支援 per-stage 的自訂 merge size。

## 整體流程

1. `PatchEmbed` 產生 token
2. 每個 stage 加 positional embedding
3. 跑若干個 `TransformerBlock`
4. 中間插入 `FlexiblePatchMerging`
5. 最後做 global average pooling
6. `LayerNorm` + dropout + linear head

## 重要參數

- `img_size`
- `patch_size`
- `in_chans`
- `num_classes`
- `embed_dims`
- `num_heads`
- `depths`
- `merge_size`
- `drop_rate`
- `drop_path_rate`

## 這個模型的價值

`MergingViT` 不只是分類模型，也是一個方便分析 token 與 head 行為的骨架。  
它的中間表示會被 `run_experiment_pipeline.py` 和 `mergingViT_plot_tool/` 直接拿來做 K-means、Grad-CAM trace 和後續解讀。
