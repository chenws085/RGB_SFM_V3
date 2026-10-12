# Foreground token mask overlay 開發規劃

## 1. 目標

在每張輸入影像上，將 foreground filtering 產生的 stage-level token mask 映射回原始影像，輸出可人工檢查的 overlay：

- 保留的前景區域維持原圖色彩；
- 被過濾的背景區域轉成灰階並變暗；
- 顯示該 stage 的 token boundary；
- 前景 token 使用較醒目的邊框；
- 圖上與 metadata 清楚標示 stage、mask 門檻、保留 token 數量與比例。

這一版只完成視覺化 A，用來先驗證 foreground mask 是否合理。**不在本階段直接改變 K-means 的輸入樣本，也不實作第二種代表 patch 視覺化。**

---

## 2. 最小化流程

```text
輸入影像
   ↓
既有最後 stage Grad-CAM
   ↓
依累積 saliency mass 轉成 boolean token mask
   ↓
利用 merge mapping 追溯各 token 的 stage-0 source patches
   ↓
建立與原圖同尺寸的 pixel mask
   ↓
背景灰階／變暗 + token grid + foreground 邊框
   ↓
PNG overlay + JSON metadata
```

MVP 的 foreground 訊號使用現有 `forward_for_gradcam()` 產生的最後 stage class-conditioned Grad-CAM，暫不引入 DINO attention、SAM、額外 segmentation model 或人工標註。

---

## 3. Mask 的定義

### 3.1 不使用固定 top-k 作為主要門檻

固定 top-k 在物體大小不同時會強制每張圖保留相同數量的 token，不適合作為 foreground mask。MVP 改採累積 saliency mass：

1. 將非負 CAM score 攤平並由大到小排序；
2. 將 score 除以總和，得到 saliency mass；
3. 選取能使累積 mass 首次達到 `foreground_mass` 的最少 token；
4. 預設 `foreground_mass = 0.60`，並至少保留 1 個 token。

例如 16 個 token 中，前 4 個已涵蓋 62% CAM mass，則 mask 保留這 4 個 token，而不是固定保留 5 個。

### 3.2 例外處理

- CAM 總和接近 0：標記為 `invalid_cam`，不把任意 token 誤當前景；overlay 仍輸出，但在標題警告。
- CAM 只有一個非零 token：允許只保留該 token。
- `foreground_mass` 限制在 `(0, 1]`。
- metadata 同時記錄 CAM score、normalized mass、threshold 與 mask，方便之後重現。

### 3.3 各 stage 的語意

目前 Grad-CAM 只在最後 stage 計算，因此較早 stage 的 mask 定義為：

> 被選中之最後-stage token 所包含的全部真實 child tokens。

這是同一組高階前景區域的階層式展開，不代表每個 stage 都獨立計算了 saliency。圖名與 metadata 必須標示 `mask_source: last_stage_gradcam`，避免誤讀。

---

## 4. Token mask 映射回原圖

### 4.1 規則 patch grid

若 stage token 與原圖是無 padding 的規則網格，可使用題目中的 nearest-neighbor 放大：

```python
pixel_mask = F.interpolate(
    stage_mask[None, None].float(),
    size=(image_h, image_w),
    mode="nearest",
)[0, 0].bool()
```

### 4.2 本專案的正式作法：source-patch union

MergingViT 在 merge 時可能出現 padding，而且 coarse token 的真實範圍不一定等於把 `H × W` 規則均分到整張圖。正式實作應重用：

- `ViTAnalyzer._source_positions_at_stage()`：把 coarse token 展開至 stage-0 真實來源 token；
- `ViTAnalyzer._stage_patch_bounds()`：取得 token 對應的原圖 pixel bounding box；
- `stage_resolutions` 中的 merge／padding metadata。

對 mask 中每個 foreground token，將其所有非-padding stage-0 source patch 對應的 pixel 區域設為 `True`。這在無 padding 時與 nearest-neighbor 結果一致，在有 padding 時則不會把虛構區域畫入影像。

token grid 也應依 `_stage_patch_bounds()` 畫出每個真實 token 的邊界，而不是用 `image_h / stage_h` 推測。

---

## 5. 圖像設計

### 5.1 主要輸出：單張 overlay

- foreground pixel：保留原圖 RGB；
- background pixel：先轉灰階，再乘 `background_brightness = 0.30`；
- 全部 token boundary：細白線，`alpha = 0.35`；
- foreground token boundary：較粗的亮綠線；
- 若想區分相鄰前景 token，可保留內部細線；若只看整體輪廓，可另提供 `foreground_outline_only` 選項；
- 標題：`image / predicted class / stage / foreground mass / kept tokens`；
- legend：綠框為 retained foreground，暗灰為 filtered background。

背景使用灰階加變暗，而不是純黑遮蔽，讓研究者仍能判斷被移除的內容究竟是背景、物體邊緣，或另一個語意物件。

### 5.2 診斷版 triptych

每張圖另外輸出一張三欄診斷圖，便於除錯：

1. Original：原始影像；
2. CAM：最後 stage 的連續 Grad-CAM heatmap；
3. Mask overlay：離散化後的 foreground mask。

這可以直接區分問題出在 CAM 本身，還是出在 CAM-to-mask threshold。

---

## 6. 建議的程式結構

### 6.1 新增純函式模組

新增：

`mergingViT_plot_tool/foreground_mask_visualization.py`

建議介面：

```python
def select_tokens_by_saliency_mass(cam, mass=0.60, min_tokens=1):
    """回傳 boolean mask 與 threshold metadata。"""

def stage_mask_to_pixel_mask(analyzer, stage_idx, stage_mask, image_hw):
    """依真實 source-patch mapping 產生 H_img × W_img boolean mask。"""

def render_foreground_overlay(
    image,
    pixel_mask,
    token_bounds,
    token_mask,
    background_brightness=0.30,
    draw_token_grid=True,
):
    """回傳可儲存的 RGB overlay。"""

def save_foreground_mask_diagnostic(...):
    """儲存 original / CAM / mask overlay 三欄圖。"""
```

mask selection 與 renderer 採純函式，方便單元測試；模型 forward 與 hierarchy mapping 仍由既有 `ViTAnalyzer` 提供。

### 6.2 對現有 analyzer 的最小修改

在 `Kmeans_analysis_padding_repr.py`：

1. 保留現有 `forward_for_gradcam()`；
2. 新增 `compute_foreground_mask()`，包裝 Grad-CAM、mass threshold 與 stage trace；
3. 新增 `save_foreground_mask_overlays()`，逐 stage 呼叫純函式 renderer；
4. 在 `analyze_all_stages()` 加入 opt-in 參數，不改變既有預設行為。

建議參數：

```python
save_foreground_overlays=False
foreground_source="last_stage_gradcam"
foreground_mass=0.60
foreground_stages="all"
foreground_target_class="pred"
foreground_background_brightness=0.30
foreground_draw_grid=True
```

---

## 7. 輸出格式

```text
<analysis_dir>/foreground_masks/
├── image_00000/
│   ├── stage0_overlay.png
│   ├── stage1_overlay.png
│   ├── stage2_overlay.png
│   ├── diagnostic.png
│   └── mask_metadata.json
├── image_00001/
│   └── ...
└── foreground_mask_manifest.jsonl
```

`mask_metadata.json` 至少包含：

```json
{
  "image_index": 0,
  "true_class": 3,
  "pred_class": 3,
  "target_class": 3,
  "mask_source": "last_stage_gradcam",
  "foreground_mass": 0.6,
  "cam_valid": true,
  "last_stage_shape": [4, 4],
  "selected_last_stage_positions": [1, 2, 5, 6],
  "stages": {
    "0": {"shape": [16, 16], "kept": 64, "total": 256},
    "1": {"shape": [8, 8], "kept": 16, "total": 64},
    "2": {"shape": [4, 4], "kept": 4, "total": 16}
  }
}
```

PNG 用於人工檢查；JSON／JSONL 用於日後比較不同 threshold，且避免只能從圖片反推設定。

---

## 8. 開發步驟

### Step 1：CAM 轉 boolean mask

- 實作 cumulative-mass selection；
- 保留連續 CAM 與排序結果；
- 處理 all-zero CAM；
- 先只驗證最後 stage mask。

產物：可打印／儲存的最后-stage boolean grid 與 metadata。

### Step 2：跨 stage 展開與 pixel mask

- 由 selected last-stage positions 沿既有 merge mapping 往前展開；
- 排除 padding child；
- 產生每個 stage 的 boolean mask；
- 用 stage-0 patch union 建立原圖尺寸 pixel mask。

產物：每個 stage 的 mask，以及與原圖相同大小的 boolean mask。

### Step 3：overlay renderer

- 反正規化輸入影像，沿用 `_image_to_numpy()` 的 mean/std；
- 實作背景灰階與變暗；
- 畫完整 token grid 與 foreground 邊框；
- 輸出單張 overlay 與 triptych。

產物：可人工判讀的 PNG。

### Step 4：接入既有分析流程

- 在 `analyze_all_stages()` 加 opt-in 開關；
- 每張影像只計算一次 Grad-CAM，所有 stage 共用 trace 結果；
- 寫入 per-image JSON 與總 manifest；
- 預設關閉，避免既有大量分析突然增加 backward 成本。

### Step 5：視覺驗證後才連接 K-means filtering

先人工檢查不同類型影像，再決定 `foreground_mass` 是否固定為 0.60。確認合理後，下一階段才讓 `run_kmeans_per_position()` 排除 background tokens。

---

## 9. 測試與驗收

### 9.1 單元測試

- mask shape 與 CAM shape 相同，dtype 為 boolean；
- 非零 CAM 時，選中 token 的累積 mass `>= foreground_mass`；
- 移除排序中的最後一個選中 token 後，累積 mass `< foreground_mass`；
- all-zero CAM 不產生假前景，並回報 `cam_valid=false`；
- foreground pixel 保持原 RGB（容許儲存誤差）；
- background pixel 確實灰階並降低亮度；
- pixel mask 尺寸精確等於輸入影像尺寸。

### 9.2 Merge／padding 測試

- 使用寬高不能被每層 merge factor 整除的 token grid；
- 確認 padding child 不會映射成前景 pixel；
- 每個 coarse token 的 pixel mask 應等於其 stage-0 source patches 聯集；
- 不得因 nearest resize 將邊界擴張到不存在的 patch。

### 9.3 人工驗收樣本

至少挑選：

- 單一置中物體；
- 物體靠近影像邊界；
- 背景紋理強烈的影像；
- 多物體或物體占比很小的失敗案例；
- 會觸發 merge padding 的尺寸。

MVP 驗收標準：一個指令可為指定影像輸出各 stage overlay、診斷圖與 metadata；圖中 foreground／background 與 token 邊界清楚，且 padding mapping 正確。

---

## 10. 已知限制

- Grad-CAM 是分類類別條件式訊號，常只突出最具辨識力的部位，不保證涵蓋完整物體。
- 最後 stage 網格較粗，向前展開後仍只是在細網格呈現同一個 coarse 決策，沒有增加定位資訊。
- 固定 `foreground_mass` 不一定適合所有資料集；因此必須先輸出 overlay，再決定是否用於實際 K-means 過濾。
- 若希望取得更完整且細緻的前景，後續可比較多 stage attention rollout、背景 cluster voting 或外部分割模型，但不應混入本 MVP。

---

## 11. 完成 A 後的下一步

完成並驗證 overlay 後，第二種視覺化可規劃為「filtered token crops / cluster representatives」：每個保留 token 仍以它在原圖上的真實 receptive region 顯示，而不是把過濾後的 token 當成新的固定大小圖片。這能直接回答哪些 patch 實際進入跨影像 K-means，並與本 overlay 一一對照。
