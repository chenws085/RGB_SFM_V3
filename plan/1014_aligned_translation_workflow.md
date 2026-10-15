# 目前流程：平移增強、飛機評估、換群案例

## 訓練與原本的飛機比較

```mermaid
flowchart TD
    A["Caltech-101 全部 101 類"] --> B["固定 split：seed 42<br/>train 6,907；validation 1,770"]
    B --> C["Training：完整圖片等比例縮至長邊 160<br/>置中於 224×224 畫布"]
    C --> D["基準：不加入平移<br/>保留既有 flip / color jitter / erasing"]
    C --> E["增強：所有 training 類別隨機取 13 種之一<br/>原位；上下左右 × 8 / 16 / 32 px<br/>完整保留內容；64 px 暫緩"]
    E --> F["保留相同既有 flip / color jitter / erasing"]
    D --> G["原有 baseline checkpoint<br/>從頭訓練，best epoch 66"]
    F --> H["增強模型從頭訓練<br/>相同架構、seed、超參數；best epoch 84"]
    B --> I["原位 validation accuracy 選模<br/>不是未使用過的 final test"]
    I -.-> G
    I -.-> H
    G --> J["各模型以未平移全類別 training activations<br/>分 stage 建立 codebook"]
    H --> J
    J --> K["修正 annotation offset 後的中心 token<br/>L2 normalize + Euclidean K-means<br/>K=31，seed 42，n_init=50"]
    B --> L["固定 160 張 validation airplanes<br/>原圖 + 四方向 × 8 / 16 / 32 px"]
    K --> M["原模型、增強模型分別評估<br/>stage 0–3：8 / 16 / 32 / 64 px 網格"]
    L --> M
    M --> N["cluster agreement / token cosine<br/>prediction consistency / accuracy"]
    M --> O["同一張圖：原位與四方向並排<br/>標示 token、cluster、cosine、預測"]
    N --> P["對照圖、逐筆 CSV、manifest、報告"]
    O --> P
```

注意：8/16/32 px 都對齊初始 8 px patch，但不一定對齊所有更深 stage 的 merging 網格。原本飛機評估會觀察各個 stage；下方案例分析則特別限制位移等於被分析 stage 的網格，避免混入該層的非對齊 crop。

## 換群／不換群案例分析

```mermaid
flowchart TD
    A["兩個固定 checkpoint<br/>不再訓練"] --> B["各自評估所有 validation 原圖<br/>依 class 記錄樣本數與 accuracy"]
    B --> C["保留 airplanes<br/>另選增強模型原圖 accuracy 最低的兩類<br/>每類至少 20 張；探索性篩選"]
    A --> D["各自使用未平移 training 中心 token<br/>按 stage fitting 相同規格 codebook"]
    D --> E["每群取離 Euclidean centroid 最近的<br/>五個 training token 作代表"]
    C --> F["Stage 0：四方向 8 px<br/>Stage 1：四方向 16 px<br/>Stage 2：四方向 32 px"]
    D --> G["同一來源影像：追蹤移動後中心 token<br/>使用同一模型、同一 stage codebook 指派"]
    F --> G
    G --> H{"cluster 改變？<br/>prediction 改變？"}
    H --> I["不換群、預測不變"]
    H --> J["換群、預測不變"]
    H --> K["不換群、預測改變"]
    H --> L["換群、預測改變"]
    I --> M["完整組別計數<br/>每 model / class / stage / group 選案例<br/>優先高 cosine；同 stage 內 source 不重複"]
    J --> M
    K --> M
    L --> M
    M --> N["驗證平移前後對齊 token crop 逐像素相同"]
    E --> O["兩列七欄案例圖<br/>整圖 → token crop → 五個群代表<br/>原圖一列；平移圖一列"]
    N --> O
    O --> P["案例報告、selected cases、paired metrics<br/>representatives、分組計數、manifest"]
```

同一模型／stage 的 codebook 跨位置使用；不是每個位置各自建立 codebook。不同模型的 cluster ID 沒有天然對應。局部 crop 相同不代表完整 token activation 必須相同，因為位置編碼與 self-attention 仍可能影響表徵。換群也不直接表示人類語意改變。

Prediction consistency 是相對原圖的預測是否改變，並不等於分類正確；兩次都錯也可能一致。四方向重用來源影像，不能把配對數當成獨立圖片數。

相關產物：[原評估報告](../plots/kmeans/caltech_aligned_translation_1014/REPORT.md)、[案例報告](../plots/kmeans/caltech_aligned_translation_1014/case_analysis/REPORT.md)。
