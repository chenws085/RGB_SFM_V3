# 第三階段：訓練流程

這一階段看的是 `train.py`。

## 這一層在做什麼

`train.py` 是整個專案的訓練入口。  
它會讀設定檔、建立資料載入器、初始化模型、設定 optimizer / scheduler，然後跑訓練與驗證。

## 主流程

1. 讀 `config.py`
2. 建立 `save_dir`
3. 初始化 `wandb`
4. 透過 `get_dataloader()` 載入資料
5. 用 `models/` 裡對應名稱建立模型
6. 建立 loss、optimizer、scheduler
7. 執行 `train()`
8. 執行 `eval()`
9. 儲存 checkpoint 與模型副本

## 主要函式

### `train(...)`

訓練迴圈的核心。

特點：

- 每個 batch 做 forward / backward / step
- 算 train loss 與 accuracy
- 每個 epoch 後跑一次 validation
- 依 validation accuracy 存 `best_epoch.pth`
- 支援 early stopping
- 支援 metric-based loss

如果模型需要額外統計，這裡也會預留 RM / layer stats 的掛鉤。

### `eval(...)`

驗證與測試流程。

特點：

- `model.eval()`
- `torch.no_grad()`
- 計算 loss、accuracy
- 可輸出 binary / multiclass 指標
- 可選擇建立 `wandb.Table`

## 重要行為

### checkpoint

訓練中最佳驗證表現會存成：

- `best_epoch.pth`

訓練結束後還會再輸出：

- `{model_name}_best.pth`
- `pth/{dataset}/{load_model_name}.pth`

### 檔案複製

每次訓練會把：

- 對應的 `models/{model_name}.py`
- `config.py`

複製到當次 `save_dir`，方便回溯實驗。

### wandb

這份腳本整合了 wandb 記錄：

- train / valid loss
- train / valid accuracy
- learning rate
- final test metrics
- model artifact

## 與設定檔的關係

`train.py` 幾乎完全依賴 `config.py`：

- `dataset` 決定資料集
- `model.name` 決定模型
- `model.args` 決定模型參數
- `loss_fn` / `training_loss_fn` 決定損失
- `optimizer` / `lr_scheduler` 決定優化策略
- `save_dir` 決定輸出路徑

## 結論

這一層的重點很單純：  
`train.py` 是訓練和第一次驗證的主入口，也是後續分析流程最重要的 checkpoint 來源。
