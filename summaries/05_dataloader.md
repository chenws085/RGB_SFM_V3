# 第五階段：資料載入

這一階段看的是 `dataloader/get_dataloader.py`。

## 這一層在做什麼

資料載入器負責把不同資料集包成 `train_dataloader` 和 `test_dataloader`。  
這個 repo 的資料層不是單一資料集，而是很多自訂資料集的統一入口。

## 核心結構

### `dataset_classes`

這是一個名稱到 dataset 類別的對照表，集中管理所有資料集實作。

常見項目包含：

- `CIFAR10`
- `Caltech101`
- `BloodMNIST`
- `PathMNIST`
- `DermaMNIST`
- `RetinaMNIST`
- `Colored_MNIST`
- `Colored_FashionMNIST`
- `FaceDataset`
- `NonclassicFace`
- `HeartCalcification_*`

## `get_dataloader(...)`

主要功能：

1. 根據 `dataset` 找對應 class
2. 建立 transform
3. 建立 train / test dataset
4. 用 PyTorch `DataLoader` 包裝

## 兩種常見轉換

### 一般資料集

大多數資料集走：

- `Resize`
- `ToTensor`
- `ConvertImageDtype(torch.float)`

### `Caltech101`

這個資料集有較強的 augmentation：

- `RandomResizedCrop`
- `RandomHorizontalFlip`
- `ColorJitter`
- `RandomErasing`
- `Normalize`

測試集則用：

- `Resize`
- `CenterCrop`
- `Normalize`

## 重點觀察

- 這裡決定資料集名稱是否能被訓練流程辨識。
- `input_size` 會直接影響 transform 和模型輸入。
- `Caltech101` 是少數特別客製 augmentation 的資料集。

## 結論

這一層的角色很明確：  
它把「設定檔中的 dataset 名稱」轉成真正可供訓練的 PyTorch dataloader。
