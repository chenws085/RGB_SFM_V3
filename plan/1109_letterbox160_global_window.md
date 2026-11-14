# 沿用0921 letterbox160 global，匹配重訓window

遠端remote140165，根目錄/home/ref/RGB_SFM_V3。使用者指出資料夾已有letterbox160 global，已查實並重現，無須重訓global。

## 可沿用的global

`runs/train/mergingvit_caltech101_letterbox160/MergingViT_best.pth`，seed42、best epoch66，history共86epochs（patience20提前停止，最高預算100epochs），1770validation accuracy62.090395%、airplanes160張98.75%。SHA256 `3cf8ab1056dfb025aa63c349e4d06e3ef49e519418fceb7798f49af1be39daca`。現行loader／model嚴格載入後重現兩項accuracy完全相同；稽核與完整split見`overnight_logs/letterbox_window_1109/global_audit.json`。

## 匹配訓練設定

window從頭訓練，唯一結構變更為7/7/7/4固定窗口（後兩stage覆蓋全層），無移窗、relative bias或額外參數；原absolute position及merging保留。初始化state_dict逐張量相同已驗證。沿用舊trainer，data6907/1770、seed42、224畫布內完整自然原圖長邊160、PIL/torchvision bilinear resize與整數meanRGB padding；training水平翻轉、ColorJitter(.3/.3/.2)、RandomErasing(.25)，無平移／尺度增強。batch32、AdamW lr3e-4 wd.01、CE label_smoothing.1、cosine100epochs eta_min1e-6、patience20，原drop/drop_path .1/.3。不能混用前輪原圖224模型的6902/1768 split與不同loss/optimizer。

新run `runs/train/window_letterbox160_20261109`；入口`scripts/run_letterbox_window_comparison.sh`，日誌`overnight_logs/letterbox_window_1109/pipeline.log`。window訓練完成後自動執行兩模型160原位＋四方向1–8/16/32px評估；各模型用自身training160中心token建固定FAISS spherical K31、100iter、50restarts、seed42字典；此輪完整1770validation。修正absolute bbox中心，不重用0921錯誤座標；主看Stage0/1/2的8/16/32px，同時固定pixel比較深度、方向與非對齊平移。輸出`plots/reports/attention_translation_letterbox160_20261109`。原位global評估accuracy應與62.09%再次匹配。

新評估loader已對齊歷史training CenterLetterbox的像素（不使用前輪bicubic／浮點mean padding）；前輪224訓練→160測試的數值保留為尺度分布不同的診斷，不作此次最終選型依據。

啟動不是完成；以history、process及評估manifest.status=complete確認。此刻尚無新window最終結果。研究文件與SHA256版本送E:\RGB_SFM_artifacts inbox；本機project root未核實，交付不是本機匯入。


## 已完成

window訓練82epochs、best62（patience20），1770validation62.99%、airplanes99.38%。匹配global62.09/98.75%。所有平移與圖文彙整完成：`plots/reports/attention_translation_letterbox160_20261109/REPORT.md`。1770×41位置×4stage×2模型共580560個token配對，finite、identity與one-token幾何檢查通過。one-token群一致率global/window94.18/89.52、89.82/85.75、83.60/79.80%；window三組CI均低於0。32pxaccuracy27.32/24.00%，差−3.32pp CI[-4.48,-2.26]；分類一致率30.45/26.54%，差−3.91pp CI[-5.20,-2.61]。原位+0.90pp CI跨0。保留global主基準，本窗口版沒有解決平移敏感性。飛機局部穩定但分類漂移現象亦重現；不能據單seed比較否定所有窗口attention。
