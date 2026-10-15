# 遠端 10/14 模型與本機 10/15 實驗比較

已驗證兩份遠端 checkpoint SHA256 與歷史 spherical 評估完全相符，split 亦一致。新收到資料 archives 通過官方 MD5。
遠端 baseline 原 config 未保存逐筆 indices，split 證據來自綁定相同 checkpoint hash 的歷史評估 manifest。
尺度為本機對遠端 checkpoints 新執行的評估；本機 S0/S1/T1 在相同 runner、split 與 spherical protocol 下的已完成結果重用。
平移重用綁定相同 checkpoint SHA256 的 1–7 px spherical 報告；不是把 10/14 Euclidean 指標改名，也不是另一次獨立重複。

## 尺度：101 類、1,770 張 validation

未見離散倍率 0.7/0.9/1.1；各 stage/readout 以原位 training features fitting 獨立 codebook。

| 模型 | 原位 accuracy | 未見倍率 accuracy | macro accuracy | Stage 2 bbox agreement | ROI cosine |
|---|---:|---:|---:|---:|---:|
| remote_baseline | 62.09% | 44.63% | 31.77% | 57.66% | 0.7227 |
| aligned_1014 | 65.71% | 55.14% | 40.61% | 65.44% | 0.8783 |
| S0 | 63.28% | 44.41% | 32.29% | 57.12% | 0.7339 |
| S1 | 69.49% | 66.76% | 53.11% | 71.32% | 0.8633 |
| T1 | 68.36% | 54.69% | 40.86% | 68.08% | 0.8515 |

## 平移：三類 202 張、四方向 1–7 px 平均

| 模型 | 平移 accuracy | 預測一致率 | Stage 2 agreement |
|---|---:|---:|---:|
| remote_baseline | 87.80% | 89.02% | 59.83% |
| aligned_1014 | 87.78% | 93.32% | 68.85% |
| T1 | 90.77% | 97.35% | 76.61% |

## 判讀

- S1 相對 aligned_1014 的未見倍率 accuracy 差 +11.62 pp，Stage 2 bbox agreement 差 +5.88 pp。
- 先同時檢查原位 accuracy、macro accuracy、群數／最大群占比／熵與同類異圖對照，不把 agreement 單獨視為語意改善。
- 不同模型群 ID 不直接對應；ROI pooling 穩定不等於原始 token 不變。
- 遠端與本機是不同 training runs／執行環境；本機 S0/S1 是主要尺度增強配對，歷史 aligned 模型是額外比較，不是只差一個變因的嚴格消融。
- 單一 seed、validation 曾用於選模。paired_comparison.json 的影像 bootstrap 不包含 training seed 不確定性，四組比較為探索性，未做多重比較校正。

## 檔案

intake_manifest.json：來源／hash；scale_comparison.csv、translation_comparison.csv：比較；paired_comparison.json：配對差值；scale/：新跑的遠端模型 metrics、圖與 cache。
