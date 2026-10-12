# 1009 Token 概念介入工具

入口：`scripts/concept_intervention.py`。

第一版支援 MergingViT 同一 stage、最後 block 後／merge 前的 token 介入。正常前向使用原模型，透過暫時 hook 修改指定位置後繼續計算。無須重新訓練分類模型。

## 概念與取樣

- 使用 Caltech-101 原有 train/validation split 和 160-in-224 letterbox。
- 只用 training images，對指定位置的 L2-normalized token fitting Euclidean KMeans。每個位置独立 codebook。
- 評估使用 validation，預設目標為真實標籤，可改成原始預測。
- 不使用 annotation center、不讀舊中心 token cache，因此不依賴漏加 bounding-box offset 的座標。這不是修正後的物體中心實驗；指定位置仍可能是背景或 padding。
- Codebook 的 cluster ID 只在同一位置／run 有意義。舊中心實驗的 cluster ID 不可直接套用。
- 限制 training 樣本時採 seeded 隨機抽樣，避免類別排序的前綴偏差。

## 執行

小規模功能驗證：

```bash
.venv/bin/python scripts/concept_intervention.py \
  --stage 2 --positions 24 25 --clusters 4 --n-init 3 \
  --max-train-samples 120 --max-eval-samples 16 --random-repeats 2 \
  --output-dir plots/concept_intervention/smoke
```

全 training fitting、airplanes validation 評估：

```bash
.venv/bin/python scripts/concept_intervention.py \
  --stage 2 --positions 24 25 --clusters 31 --n-init 10 \
  --category airplanes --random-repeats 5 \
  --output-dir plots/concept_intervention/airplanes_stage2
```

位置 24 僅為預設範例，需依 checkpoint token grid 選擇。程式檢查位置是否越界，不假設所有 checkpoint 都是 7×7。

## 同層組合

```bash
.venv/bin/python scripts/concept_intervention.py \
  --stage 2 --positions 24 25 --clusters 4 --n-init 3 \
  --max-train-samples 120 --max-eval-samples 16 --random-repeats 2 \
  --groups 24:0 25:2 24:0+25:2 \
  --output-dir plots/concept_intervention/group_smoke
```

`24:0+25:2` 僅對同時存在兩個概念的影像，同時替換兩個位置。先查看單概念試跑與 codebook，再選有支持樣本的組合。

聯合效果不能直接與全體單概念平均相加比較：各組支持影像不同。若研究 interaction，需要在聯合存在的相同 source_index 上配對 A、B 與 A+B；第一版輸出逐影像紀錄供此分析，尚未自動做交互作用檢定或跨層路徑介入。

## 替換及對照

預設 `--replacement mean`，以全部 training images 的各位置原始特徵平均替換，保留 raw activation 尺度；不把 normalized centroid 直接塞進網路。亦可使用 `--replacement zero` 作敏感度分析。

對照在同一 stage 均勻抽取相同 token 數，逐影像無放回抽樣。可與概念位置重疊；位置、前景／padding 與特徵幅度沒有匹配。因此 effect gap 是相對空間隨機介入的探索性比較，不能排除位置差異。後續需要同位置、不同概念的控制及真實 activation donor replacement。

## 輸出

- `codebooks.npz`：各位置 normalized-space centroid、各位置 raw training mean。
- `sample_effects.csv`：來源影像、群／組合、隨機對照位置、介入前後預測、目標 logit／margin／probability 下降量及 effect gap。
- `concept_summary.csv`：支持影像數與支持比例、平均下降量、平均 effect gap、預測改變率。沒有支持影像的群也保留，平均欄為空。
- `run_manifest.json`：checkpoint SHA256、模型參數、全部 train/eval IDs、類別、參數及推論限制。

定義：`logit_drop = original_target_logit - intervention_target_logit`；`effect_gap = logit_drop - random_logit_drop`。正值表示目標分數下降，負值表示上升。Margin 為目標 logit 減最大其他類別 logit。

每張影像重複多個 random controls，不代表多張獨立影像。平均 concept effect 在 repeats 相等時與影像平均相同；推論／bootstrap 應以 source_index 為單位。此 validation 曾用於 checkpoint 選擇，不能稱為 untouched test。

## 驗證

```bash
.venv/bin/python checks/verify_concept_intervention.py
```

檢查空介入、identity replacement、masked replacement 與獨立下游 forward 一致性、修改確實影響 logits、正常／失敗後 hook 清除及群組解析。

目前範圍是 token-level 單概念／同層組合；不含 head intervention、跨層 causal tracing、自動語意命名或統計顯著性宣稱。

## 可視化

```bash
.venv/bin/python scripts/visualize_concept_intervention.py --results-dir plots/concept_intervention/smoke
.venv/bin/python scripts/visualize_concept_intervention.py --results-dir plots/concept_intervention/group_smoke
```

各結果目錄的 `visualization/` 會產生概念平均與逐影像散點概覽、概念／隨機介入逐影像對照、影像 token 框與分數變化案例，以及完整 PDF。組合結果另產生同一批聯合存在影像的 A、B、A+B 對照及配對 CSV。隨機 repeats 先在各影像內平均；未做顯著性宣稱。
