# 1005 Caltech-101 輪廓座標稽核

## 發現與影響

製作每群 silhouette 代表影像時，發現既有 Caltech-101 特徵抽取流程漏加輪廓座標的 bounding-box 偏移。因此先前所稱的「物體質心 token」，實際上常落在另一個位置，部分樣本落在背景。

既有 K-means、silhouette、ARI 與平移指標仍是對已保存特徵的計算結果，但它們描述的是**原取樣位置**的特徵；不能直接當作物體中心或物體部件分群的證據。先前相關報告已加入此註記。

## 原因與依據

torchvision 的 `Caltech101(..., target_type=['category', 'annotation'])` 回傳 `.mat` 檔中的 `obj_contour`，沒有套用 `box_coord`。

資料集附帶的 [show_annotation.m](../data/caltech101/caltech-101/show_annotation.m) 在畫輪廓時明確使用：

```matlab
x = obj_contour(1, :) + box_coord(3);
y = obj_contour(2, :) + box_coord(1);
```

`box_coord` 的排列為 `[y_min, y_max, x_min, x_max]`。輪廓點是相對 box 的座標，而非完整圖片座標。

既有 [safe_letterbox](../scripts/test_caltech_translation_kmeans.py) 直接以 `annotation[0].mean()`、`annotation[1].mean()` 作為原圖上的位置，漏掉了這兩個 offset。

轉成 Python／PIL 的 0-based 圖片座標時，本次圖庫使用：

```python
absolute_contour = obj_contour + [[box_x_min], [box_y_min]] - 1
center_x = absolute_contour[0].mean()
center_y = absolute_contour[1].mean()
```

這個中心是輪廓點的平均位置，並不是物體 mask 的面積質心。

## 全類別 Held-out 稽核

對同一批 1,770 張 held-out 圖片，在原有 `160-in-224 letterbox` 下，將原位置與修正位置分別映射到各 stage 網格。

| Stage | Token 網格大小 | 修正前後仍是同一 token 的比例 | 取樣 token 不同的比例 |
|---|---|---:|---:|
| 0 | 8 × 8 px | 8.87% | 91.13% |
| 1 | 16 × 16 px | 15.42% | 84.58% |
| 2 | 32 × 32 px | 35.99% | 64.01% |

位置位移的中位數為 **23.06 個 letterbox canvas pixels**。這是標註座標轉換差異，不是模型特徵的不穩定程度。

完整逐圖片紀錄：[coordinate_audit.csv](../plots/kmeans/quality_evaluation_1005/cluster_meaning_seed42/coordinate_audit.csv)。

## 代表案例

Stage 2、手寫球面 K-means、seed 42 的 C00，最高正 silhouette 的三張圖片均為 Motorbikes，但原取樣的 token 多是白色背景；不能將此群直接命名為摩托車部件群。

例如 source 1124：

- 原輪廓點平均位置約為 `(101.00, 56.83)`。
- `box_coord = [25, 144, 31, 233]`。
- 加 offset 並轉成 0-based 後，位置約為 `(131.00, 80.83)`。
- 在同一張 letterbox 圖上，兩個位置會落在不同 Stage-2 tokens。

[C00 代表影像對照圖](../plots/kmeans/quality_evaluation_1005/cluster_meaning_seed42/stage2/spherical/cluster_00.png)。

## 本次圖庫如何呈現

分群標籤及 silhouette 沿用舊特徵，不把修正後的位置假裝成產生這些分數的位置。每張代表圖標出：

- **紅十字**：原實驗實際取樣位置。
- **綠十字**：補上 bounding-box offset 後的輪廓中心。
- **黃框**：原實驗取樣 token 的網格範圍。
- 全圖、修正中心附近 64px 內容，以及原取樣 token 的放大內容。

Transformer 使用 self-attention，因此 token 網格框不等於完整感受野。

[圖庫與逐群索引](1005_kmeans_cluster_meaning_seed42.md)。

## 後續重跑要求

要評估物體中心的特徵與群語義，需要先修正 annotation 座標轉換，再重新抽取 training／held-out／translated features，並重新建立分群中心與評估指標。

新舊 features cache 與 manifest 必須使用不同的座標版本，避免混用；不能只修改代表圖標示，就沿用舊群作為修正後物體中心的實驗結果。

本次僅完成代表圖、中心對照與稽核紀錄，尚未重跑修正座標後的分群實驗。
