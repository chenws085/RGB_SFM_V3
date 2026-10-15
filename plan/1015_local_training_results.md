# 本機平移與尺度實驗進度

更新：2026-10-15。資料為官方 Caltech-101；固定 seed-42 split，training 6,907／validation 1,770。三組均在 RTX 4060 Ti 從頭訓練，非 epoch-10 續訓。最新狀態以各 log/status 與 manifest 為準。

## 訓練與評估均已完成

| 模型 | 完成 epoch | best epoch | best 原位全類別 validation accuracy |
|---|---:|---:|---:|
| T1：1–7 px 平移增強 | 98，early stop | 78 | 68.36% |
| S0：固定長邊 160，無平移／尺度增強 | 83，early stop | 63 | 63.28% |
| S1：尺度增強 96／128／160／192 | 100 | 84 | 69.49% |

平移 pipeline 完成後，尺度排程依序完成 S0、S1 訓練與 S0／S1／T1 的七倍率評估。兩個 status.txt 皆為 `complete`，未留下待訓練階段。

## 平移：三類探索性比較

固定 airplanes／chandelier／brain 共 202 張，四方向 1–7 px 等權平均；所有模型使用各自的 FAISS spherical codebook。

| 模型 | 三類原位 accuracy | 平移 accuracy | prediction consistency |
|---|---:|---:|---:|
| 歷史 baseline | 91.09% | 87.80% | 89.02% |
| 歷史 aligned 8／16／32 px | 89.60% | 87.78% | 93.32% |
| 本機 T1 | 91.09% | 90.77% | 97.35% |

T1 相對歷史 baseline 平移 accuracy +2.97 pp，一致率 +8.33 pp。兩個歷史對照來自 Git 保存的已完成評估，非本機重新訓練；三類樣本量不均衡，不能替代全類別結論。

完整報告：[本機平移報告](../plots/kmeans/caltech_non_aligned_restart_1015/REPORT.md)。狀態／log：`overnight_logs/non_aligned_translation_restart_1015/`。

## 尺度：101 類 validation，比較未見倍率

未見離散倍率為 0.7／0.9／1.1，即長邊 112／144／176。完整來源圖與背景一起縮放、置中；不是只縮物件。S0／S1 使用相同訓練預算與選模規則。

| 模型 | 1.0 accuracy | 未見倍率 accuracy | 未見倍率 macro accuracy | Stage 2 bbox agreement |
|---|---:|---:|---:|---:|
| S0 | 63.28% | 44.41% | 32.29% | 57.12% |
| S1 | 69.49% | 66.76% | 53.11% | 71.32% |
| T1，平移參考 | 68.36% | 54.69% | 40.86% | 68.08% |

S1−S0：未見倍率 accuracy +22.35 pp，影像層級 paired bootstrap 95% CI 約 [+20.79, +23.88] pp；Stage 2 bbox agreement +14.20 pp，CI 約 [+12.33, +16.10] pp。每張來源先彙整三倍率，再 bootstrap；CI 不包含重訓 seed 的不確定性。

該 stage/readout 的平均 cosine 為 S0 0.7339、S1 0.8633；同圖對應 cosine 減同類異圖 cosine 的 gap 為 0.3564 → 0.3794。七倍率中的三個未見倍率皆使用 31 群，最大群占比平均 6.46% → 6.16%，群熵 3.3537 → 3.3736（自然對數）。此組指標未呈現單群坍縮；不代表所有 stage 或語意品質已獲證明。

這是單一 seed、validation 已參與選模的探索性結果。ROI pooling 穩定不等同原始 token 不變；目前也沒有人工部件 ground truth 或語言概念驗證。後續優先檢視各類／各 stage，再決定是否進行三 seeds 重複，而非立即加入新 loss。

完整報告：[本機尺度報告](../plots/kmeans/caltech_scale_1015/REPORT.md)。主差值：`plots/kmeans/caltech_scale_1015/primary_comparison.json`；狀態／log：`overnight_logs/caltech_scale_1015/`。

## 權重與資料傳輸

目前權重仍在本機 `runs/train/`，未主動傳到遠端。已另建 HDD 接收服務供遠端向本機同步權重／資料集；詳見 [傳檔服務操作與進度](1015_artifact_transfer_service.md)。不因服務已啟動就宣稱遠端資料已同步。
