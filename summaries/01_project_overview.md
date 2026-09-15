# 專案概要

## 這個倉庫是什麼

這是一個以「可解釋影像分類」為核心的研究型深度學習專案。  
目前主要有兩條線：

1. 以 `RGB_SFMCNN_V2` 為主的 CNN 可解釋流程。
2. 以 `MergingViT` 為主的新式 Vision Transformer 流程，搭配 patch merging、K-means、Grad-CAM 和 VLM caption 分析。

目前這個 branch 主要對應 `MergingViT` 的工作流。

## 核心流程

這個專案實際上大致是：

1. 在指定資料集上訓練模型。
2. 將最佳 checkpoint 存到 `runs/train/...`。
3. 做訓練後分析，例如 K-means 分群、Grad-CAM trace、VLM caption 產生。
4. 彙整或視覺化分析結果。

## 主要入口

- `train.py`：主要訓練流程。
- `train_kfold.py`：K-fold 訓練版本。
- `run_experiment_pipeline.py`：整合 K-means、VLM caption 與摘要輸出。
- `eval_checkpoint.py`：評估已儲存的 checkpoint。
- `eval_images.py`：單張或批次影像推論工具。
- `display_gui.py`：圖形介面檢視工具。

## 模型層

模型主要放在 `models/`。

重要檔案：

- `models/MergingViT.py`：支援彈性 patch merging 的 Transformer 模型。
- `models/RGB_SFMCNN_V3.py`、`models/RGB_SFMCNN_V2.py`、`models/RGB_SFMCNN.py`：CNN 型可解釋模型。
- `models/ResNet.py`、`models/AlexNet.py`、`models/DenseNet.py`、`models/GoogLeNet.py`、`models/PVTv2.py`、`models/Swin_tiny.py`、`models/VIT.py`：比較用或 baseline 架構。

`MergingViT` 會暴露 attention 內部資訊，供後續分析腳本使用，像是 head-level output 和 pre-projection features。

## 資料層

資料集載入集中在 `dataloader/get_dataloader.py`。

支援的資料集包含：

- `CIFAR10`
- `Caltech101`
- `BloodMNIST`、`PathMNIST`、`DermaMNIST`、`RetinaMNIST`
- `Colored_MNIST`、`Colored_FashionMNIST`
- `MultiColor_Shapes_Database`、`MultiGrayShapesDataset`、`MultiEdgeShapes`
- `FaceDataset`、`NonclassicFace`
- `HeartCalcification_*`
- 其他自訂或預處理過的醫療 / 合成資料集

其中 `Caltech101` 使用的資料增強比多數資料集更強。

## 分析與實驗工具

這個 repo 的訓練後工具很多：

- `mergingViT_plot_tool/`：patch merging、K-means 和代表 token 的分析。
- `vit_analysis_vlm/`：VLM caption 生成與匯出工具。
- `monitor/`：指標與分布監控工具。
- `plot_tool/`、`plot_example_V2.py`、`plot_CI_V2.py`、`plot_stats_metrics.py`、`plot_every_graph.py`：繪圖與報告。
- `scripts/`：實驗輔助腳本與資料集摘要工具。
- `research/`：筆記本與探索性腳本。

`run_experiment_pipeline.py` 是整個分析流程的主控腳本。

## 設定檔

- `config_example.py`：設定範本。
- `config.md`：設定說明。
- `config.py`：訓練時實際使用的 runtime 設定。

設定檔主要控制：

- 模型名稱與架構參數
- 資料集與輸入尺寸
- optimizer / scheduler
- 訓練 loss 選擇
- 輸出路徑
- K-means 與 VLM 分析設定

## 輸出位置

常見輸出會落在：

- `runs/train/exp*/`：訓練 checkpoint 與複製的 config
- `plots/kmeans/...`：K-means 與 trace 結果
- `inference/` 或 `vlm_analysis/`：依 pipeline 設定而定的分析輸出

## 依賴套件

`requirements.txt` 的重點依賴包含：

- PyTorch、torchvision
- timm
- grad-cam
- medmnist
- scikit-learn
- pandas、numpy、matplotlib、seaborn、plotly
- wandb

## 快速閱讀順序

如果要快速理解這個 repo，建議照這個順序看：

1. `config_example.py`
2. `train.py`
3. `models/MergingViT.py`
4. `dataloader/get_dataloader.py`
5. `run_experiment_pipeline.py`
