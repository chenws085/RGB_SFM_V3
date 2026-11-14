# 2026-10-20 縮放案例中間狀態分析

## 完成版權重已取得

已從共享 E 槽 inbox 的 weights/local/ 下載 S0／S1／T1 完成版 checkpoint、config、history、manifest 到 remote140165 專案的 `overnight_logs/scale_internal_1020/inbox/weights/local/`。12 個檔案逐一符合來源 SHA256；三個 checkpoint 亦符合既有 `plots/kmeans/caltech_scale_1015/*/run_manifest.json`。

- S0 best epoch 63，SHA256 `3938ec6b20ce90e951515c22c4b4ab05ae8831f21f04850a6f75ee31f52bf516`。
- S1 best epoch 84，SHA256 `b1c092edee1c03a503212c35c999108a85892b4d8ed9155025052c406ad367e4`。
- T1 best epoch 78，SHA256 `88bba79f9f6357c58e592272657b505def602acb10a98914902e2eeab09bc889`。

來源通知：`notes/local/20261020T053812Z_completed_S0_S1_T1_weights_done.md`。來源清單：`weights/local/completed_S0_S1_T1_20261020T053812Z_manifest.json`。這些是本輪完成版，不是遠端原 epoch-10 run。

## 逐案例 pilot

程式：`scripts/analyze_caltech_scale_internal.py`；log：`overnight_logs/scale_internal_1020/run.log`；輸出：`plots/kmeans/caltech_scale_1015/internal_1020/`。狀態以輸出 status.json 為準。

固定原報告三個案例與七尺度，擷取各層 pre-merge token。以固定 512 張 training-160 影像、每圖每層 8 個 token，另建 FAISS spherical K=31 字典，與 ROI 字典分開。輸出 token 圖、逆座標對應特徵與真實 merging 邊；Stage 2 做 reference-center 概念的 raw position mean 介入，20 次 bbox 內同 token 數隨機對照。此為描述性 pilot；不宣稱群體泛化、語意或跨層因果。介入對照尚未匹配 norm／精確位置；尚未做 donor、A/B/A+B 與下游概念追蹤。

## 完成與驗證

三組 pilot 均完成。產物含 9 張案例圖、252 筆 token metrics、63 筆介入與 merge_edges.csv。63 次 identity 最大 logit 誤差 1.90735e-6；原尺度自對應 agreement／cosine 約 1，12 組 spherical centers 均為 unit norm；9 張 PNG 及報告圖片路徑通過檢查。

Stage 2 未見尺度三案例平均 token agreement：S0 43.99%、S1 47.35%、T1 77.89%；cosine 0.5845／0.7783／0.7973。此為全局 token 字典與 nearest-cell 逆座標對應，不能替代原全體 ROI 評估，也不能以 3 張案例推論全體排序。介入支持集合不同，不將 mean effect gap 當模型間因果優劣。
