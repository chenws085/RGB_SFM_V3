# 物件縮放／平移成對訓練開發

日期：2026-10-31。機器：remote140165。專案根：`/home/ref/RGB_SFM_V3`。

目標是同一物件在不同位置與大小下保持分類及對應區域表示；本輪是物件獨立 cutout，沒有縮放背景。不是舊整張來源圖 letterbox 尺度評估的延續數值。

## 實作

- `dataloader/ObjectScalePairs.py`：讀取 Caltech contour，使用既有已修正的 `obj_contour + [box_x,box_y] - 1` 座標。按輪廓 rasterize，再裁切 mask 外框。從原 crop 直接 resize premultiplied RGB 與 alpha，貼至獨立背景，避免舊物件殘影與邊界背景顏色滲漏。背景顏色按 source／epoch 決定，與類別無關，雙 view 相同。
- `models/ObjectScaleViT.py`：可選重疊局部內容加權 merging。固定父層 token 數，窗口為父層支援範圍加一 child halo，排除 padding。修正了初版窗口中心偏向第一 child 的座標問題。依舊存在固定網格及絕對位置編碼，沒有嚴格等變保證。
- `scripts/train_object_scale.py`：B 為原 merging＋成對損失，D 為內容加權 merging＋相同成對損失。兩者共享初始 backbone／head 權重、資料與超參數，各自 merger 初始化；不是既有 S0/S1/T1 checkpoint 微調。CE + 0.2 symmetric detached-target KL + 0.2 region cosine。Stage index 1、實體 stride 16，依來源 crop 相對座標對應 readout。此為整物件 region pooled consistency，尚非有部件標註的逐部件一致性。
- `scripts/evaluate_object_scale.py`：固定背景，物件長邊 48/64/80/96/112/128/144 × 原位／(7,-5) 平移。訓練長邊為 64/96/128，訓練位移每軸 -16..16；(7,-5) 不是未見位移值。80/112 為尺度內插，48/144 為尺度外插。只在仍可辨識的範圍討論近似穩定性。

## 資料、資源及執行

沿用歷史 train／validation 的 source 分組，分別每類抽取 16／4 張，共 1616／404 張，seed 42。兩組完全分離。初版選樣排除 index 5776（輪廓越界）。config 保存實際 index、輪廓 SHA256 與來源程式 SHA256，run 根下另存 2020 張原始影像 SHA256。

GPU 0 記憶體近滿，不更動其工作；本輪用 GPU 1 RTX 3060。20 epochs／組，batch 16，AdamW lr 3e-4、weight decay .05，先 B 再 D。

有效 run：`runs/train/object_scale_1031_20261031T093309Z/`。啟動器 `scripts/run_object_scale_1031.sh`。執行狀態以 process、每組 status.json 及 history.json 共同確認，不單憑 running 檔案。

保留早期 smoke `runs/train/object_scale_1031_smoke_B/`。`...093157Z` launcher 未真正持續執行；`...093216Z/B` 執行兩 epochs 後主動中斷，status 已記錄 interrupted，用於修正 merging 支援範圍；不可與有效 run 混用。

## 已通過驗證

`checks/check_object_scale_prototype.py`：baseline state_dict strict 載入與 logits 逐值相同、奇數／極小網格、padding 排除、固定背景、物件縮小、finite loss 與可訓練 merging 梯度。

`checks/check_object_scale_training.py`：含 terminal padding 的 physical-coordinate ramp 採樣還原、父層 nominal support 中心、真實 cutout 決定性、原 merging 及新 merging 分別在四張真實來源的雙 view 上 50 steps 過擬合，兩 view 訓練 accuracy 都 100%，loss 分別約 5.15→0.058、4.77→0.017。這是 pipeline correctness，不能作為泛化證據。

抽查八類 paired preview：`overnight_logs/object_scale_training_1031/mask_audit.png`。輪廓近似外框不一定精確消除椅背等孔洞，並非所有 2020 張已人工稽核，也不是部件 mask。主實驗仍需 mask 品質分層、自然背景及合成邊界控制、無一致性 loss 的 A 組、計算／參數量對照、多 seeds 與未探索 final test。

## 本機交接

研究紀錄與程式碼版本副本、SHA256 manifest 經 HTTPS 傳送至 E:\RGB_SFM_artifacts 收件匣。收件匣交付不表示本機專案已匯入；本機專案絕對路徑仍未確認。核對 SHA256 後再匯入，保留不同版本。

## 首輪結果與第二版開發

首輪 B/D 20 epochs 均完成，checkpoint、14 個條件評估及 logits 已落盤。結果見 `runs/train/object_scale_1031_20261031T093309Z/REPORT.md` 與 `pilot_curves.png`。

- B 原位長邊96 accuracy 5.45%，長邊80＋(7,-5) 6.19%，最差條件 2.97%；參考條件最大預測類占比 14.11%。
- D 原位長邊96 accuracy 2.97%，長邊80＋(7,-5) 3.47%，最差條件 1.98%；參考條件最大預測類占比 42.57%。
- 兩者準確率偏低；D 本輪未勝出，也更集中預測少數類別。不能將高 agreement 解讀為尺度不變性，也不能將單 seed 小資料短訓練視為架構定論。

根據這項證據，新增 `ResidualContentMerging`：保留原順序串接 merging，加入由 tanh gate 控制的內容加權 residual；gate 初值0，初始 logits 與原模型完全一致，gate 梯度非零。避免純 pooling 立即丟棄順序細節，此為待驗證假設。

第二版 pilot：`runs/train/object_scale_1031_hybrid_pilot/`，20 epochs、相同 1616／404 source、GPU1。訓練程式新增 `--hybrid-merging`、`--valid-per-class`，並自動保存來源程式快照。新版本於模型初始化完成後重設 torch RNG，以供後續完整匹配的 A/B/hybrid 比較；它與首輪 B/D 並非逐 dropout RNG 匹配，暫不直接做嚴格架構勝負結論。第一輪程式已按各自 config 的 SHA256 保存於各分支 `source_snapshot/`。

下一步：完成 hybrid pilot、檢查 gate 是否學習、使用相同新入口重跑充足資料／訓練預算的 A（CE only）、B（原 merging＋一致性）、hybrid；判讀原尺度與未見尺度準確率，搭配群／特徵區辨性。目標仍在進行中，尚未得到可靠尺度不敏感模型。

## 完整來源比較已排程

啟動 `scripts/run_object_scale_full.py`，佇列路徑記錄於 `overnight_logs/object_scale_training_1031/full_run.txt`。它會核對 `/proc/<pilot pid>/cmdline`，只等待正在執行的 hybrid pilot；pilot 成功完成後，在 GPU1 依序執行 A（兩 view CE、兩項一致性權重0）、B（原 merging＋一致性）、H（gated residual＋一致性），各100 epochs、batch16、seed42，使用既有 train/validation 的全部可用來源，逐項排除無效輪廓。每組完成後評估七尺度×兩位移。actual split 與數量由每組 config 記錄，不預先把歷史6907／1770當作輪廓稽核後數量。

單 seed 完整訓練仍是開發證據，validation 是探索／選模集；不是多 seeds 或未見 final test 結論。佇列狀態在 run 根的 queue_status.json，須搭配實際程序確認。GPU0工作未變更。所有 checkpoint 留遠端，不將文件 inbox 交付誤稱本機專案匯入。

## Hybrid pilot 結案與完整訓練啟動核實

Hybrid pilot 20 epochs 與14條件評估已完成。長邊80＋(7,-5) accuracy 3.71%，目前未超過首輪B的6.19%；兩輪 dropout RNG規約不同，僅描述結果。best checkpoint 的三個 raw gate 為 -0.10307、0.003176、-0.008657，顯示 gate 參數有更新，不代表機制有效。詳細值見 hybrid pilot 的 scale_evaluation.json 與 gate_diagnostics.json。

完整佇列已確認從等待轉為 A 組實際訓練，父／訓練程序均存在。全部來源經輪廓稽核後實際為6902 train／1768 validation，排除7張，實際index及理由在A/config.json；不是先前小樣本1616／404。三組各100epochs尚未完成。

新增 `scripts/report_object_scale_full.py`，用各組落盤checkpoint SHA256與共同source IDs核對，整理準確率曲線及H相對A/B的配對來源bootstrap差值。80/112的原位和平移條件先按來源平均，不將重複變換當成獨立影像。訓練完成後執行報告；尚未對未完成模型產出比較結論。

## A 組半程快照

完整 A 組已完成50/100 epochs，訓練程序150676與佇列150312經 /proc command line 核對存活。第50輪原尺度 validation accuracy 73.53%，長邊80＋(7,-5) 69.68%；第48輪分別74.83%／71.15%，驗證指標有正常波動。這是中途 epoch，不是最終或已選best的七尺度結果，不能拿來判定B/H勝負。完整 run 根的 source_images_manifest.json 已核對8670張影像SHA256，train/validation來源無重疊，各自涵蓋101類。B/H仍依序等待。
