# 第六階段：實驗分析流程

這一階段看的是 `run_experiment_pipeline.py`。

## 這一層在做什麼

這支腳本把訓練後的分析流程串起來，主要是三段：

1. K-means 分析
2. VLM caption 產生
3. 結果彙整與輸出

## 主流程

### Step 1：K-means

- 讀 checkpoint 與對應的 config
- 從 `MergingViT` 抽出分析需要的特徵
- 跑 K-means
- 輸出 `gradcam_trace` 與其他分析結果

### Step 2：VLM Caption

- 找出 Grad-CAM 或 VLM 分析輸出中的單列影像
- 依 stage 篩選資料
- 呼叫 `vit_analysis_vlm/run_caption_analysis.py`
- 輸出 JSONL

### Step 3：Summary

- 把前面步驟的結果整理成 CSV
- 方便後續人工看表或再做視覺化

## 常用參數

- `--skip-kmeans`
- `--skip-caption`
- `--skip-summary`
- `--dataset`
- `--caption-source`
- `--caption-stage-start`
- `--caption-stage-end`
- `--output-jsonl`
- `--output-csv`

## 重點觀察

- 這支腳本依賴訓練輸出的 checkpoint。
- `MergingViT` 的中間特徵是分析流程的核心來源。
- `config.py` 也會影響分析行為，不只是訓練。

## 結論

這是整個 repo 的分析主控腳本。  
如果 `train.py` 是訓練入口，那 `run_experiment_pipeline.py` 就是訓練後解讀入口。
