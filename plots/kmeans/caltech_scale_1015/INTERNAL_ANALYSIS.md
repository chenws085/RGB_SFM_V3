# 尺度案例的中間狀態分析

本報告直接彙整既有 summary.csv，未重新 forward 或訓練模型。統計範圍為 1,770 張 validation 在未見長邊 112／144／176 的三尺度等權平均，不是三張案例圖的逐圖結論。

## 已有的內部狀態證據

特徵擷取位置是各 stage 最後 block 後、merge 前。每模型／stage／readout 的 FAISS spherical K=31 字典僅以 training-160 特徵建立，之後固定字典比較各尺度。bbox 為依正確 annotation offset 及實際 resize 比例映射的區域，以 token-cell 面積交集加權平均。不同模型、stage、readout 的群編號不能直接對應。

| Stage | S0 bbox agreement | S1 bbox agreement | T1 bbox agreement | S0 cosine | S1 cosine | T1 cosine |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 83.15% | 84.44% | 79.72% | 0.9561 | 0.9705 | 0.9643 |
| 1 | 68.51% | 78.89% | 75.05% | 0.8570 | 0.9266 | 0.9296 |
| 2 | 57.12% | 71.32% | 68.08% | 0.7339 | 0.8633 | 0.8515 |
| 3 | 51.90% | 77.91% | 62.56% | 0.6262 | 0.8705 | 0.7595 |

S1 在四層的上述穩定性指標均高於 S0，深層差異尤其明顯。但平滑的 ROI pooling 可能掩蓋 token 變化，群一致率亦須搭配群多樣性與同類異圖對照判讀；這些數字尚不能證明概念語意或因果作用。

## 依本研究方法接續逐案例分析

1. 固定現有 airplanes／chandelier／brain 案例，依原 manifest 的 source index 與 checkpoint hash 還原七尺度輸入；另固定抽樣正確→錯誤、錯誤→正確及預測不變案例，避免只挑成功圖。
2. 記錄 Stage 0–3 全部 pre-merge token activation。以實際 resize 軸比例與置中 offset 對應原圖區域，保留 patch 相位、ROI 覆蓋、背景及 merging padding；同一格索引不等於同一影像內容。
3. 保留既有 center／bbox／central_roi 字典作其原 readout 的比較。若新增逐 token 概念圖，另用 training token 建立固定取樣協定的 spherical 字典，不能直接把 ROI 字典視作已驗證的 token 字典；不在每個測試尺度重新 fitting。
4. 各案例輸出七尺度×四層概念圖、對應區域 cosine、換群／margin 與 prediction／logit 變化。跨層依真實 2×2 patch merging 建立 child→parent 邊；統計連線僅稱組合或轉移，不先稱因果路徑。
5. 對高支持候選概念做 activation intervention，量測原目標類別 logit drop、prediction 及下游概念變化。加入 identity、同 stage／token 數且盡量匹配位置／前景／norm 的控制；mean／zero／真實 donor 做敏感度比較。
6. A、B、A+B 必須在聯合存在的同一組來源影像比較；先在來源影像內彙整尺度與隨機重複，再計 CI，不能將同圖七尺度當獨立樣本。

## 2026-10-20 執行更新

完成版 S0／S1／T1 已從共享 inbox 的 weights/local/ 下載至遠端專案的 overnight_logs/scale_internal_1020/inbox/weights/local/。12 個檔案 SHA256 及三模型與既有評估的 checkpoint hash 均通過。

已完成 [三案例逐 token 圖與介入 pilot](./internal_1020/REPORT.md)：九張四層七尺度圖、252 筆 token 對應指標、63 筆介入比較及真實 merge 邊。採固定 512 張 training 影像另建 token 字典，不沿用 ROI 字典。原尺度自對應與 63 次 identity 檢查通過；非全體 validation 重測，亦尚未完成下游概念追蹤、A/B/A+B 或 donor 控制。

來源：[評估程式](../../../scripts/evaluate_caltech_scale.py)、[研究脈絡](../../../RESEARCH_CONTEXT.md)、[原報告](./REPORT.md)。
