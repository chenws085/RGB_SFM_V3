# 第二階段：設定檔結構

這一階段主要看 `config_example.py`、`config.md` 和實際執行用的 `config.py`。

## 這一層在做什麼

這個專案幾乎所有訓練與分析流程都由設定檔驅動。  
訓練時先讀 `config.py`，再把 model、dataset、optimizer、loss、輸出路徑等參數帶進 `train.py` 與後續分析腳本。

## 核心內容

- `config_example.py` 是範本，展示整體設定格式。
- `config.py` 是實際執行時讀取的設定來源。
- `config.md` 用來說明各欄位的意義。

## 主要區塊

### 1. 基本資訊

- `project`
- `name`
- `group`
- `tags`
- `description`
- `device`
- `load_model_name`

### 2. 模型設定

`arch` 定義模型名稱與參數。

目前這個 repo 最重要的兩類是：

- `RGB_SFMCNN_V2`
- `MergingViT`

其中 `MergingViT` 會用到：

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

### 3. 訓練設定

常用欄位：

- `save_dir`
- `dataset`
- `input_shape`
- `batch_size`
- `epoch`
- `early_stop`
- `patience`
- `lr`
- `lr_scheduler`
- `optimizer`
- `loss_fn`
- `training_loss_fn`
- `use_metric_based_loss`
- `use_preprocessed_image`

### 4. 其他專用設定

這個 repo 還有醫療影像相關的特化設定，像是心臟鈣化流程：

- `grid_size`
- `resize_height`
- `threshold`
- `enhance_method`
- `use_vessel_mask`
- `augment_positive`

## 實作重點

1. 改模型時，先改 `arch`。
2. 換資料集時，先改 `dataset` 和 `input_shape`。
3. 改輸出位置時，改 `save_dir`。
4. K-means / VLM 分析也會從這份設定讀參數，不只訓練會用。

## 結論

這個專案的設定中心很明確：  
`config.py` 是整個訓練與分析流程的入口，`config_example.py` 則是最值得先讀的模板。
