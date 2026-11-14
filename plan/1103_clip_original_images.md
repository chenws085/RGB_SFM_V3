# 修正方向：使用保留自然背景的原圖訓練

2026-11-03，remote140165 /home/ref/RGB_SFM_V3。使用者先暫停純色背景無平移組，隨後指定「使用原圖訓練」。該暫停組維持paused，不恢復。

## 原圖的實際處理

整張原始RGB影像與自然背景完整保留，僅等比例縮放長邊至224，再置中補邊至224×224、ImageNet normalization。不使用物件cutout、輪廓mask、隨機位置／大小、flip、color jitter或其他增強。補邊是模型固定輸入尺寸所需，不替換原圖背景。single view，a/b API別名同一影像，訓練程式只跑一次forward，CE不因重複view改變。教師仍用先前凍結CLIP原始bbox cache；學生深層ROI以bbox均勻8×8採樣平均，不用分割mask。

## 新對照

CE-only與CE＋Stage3/4 CLIP cosine0.2，兩組原merging、seed42、相同初始化、6902train／1768validation、100epochs、batch16、AdamW3e-4；學生主幹全訓練，教師凍結。沿用既有已稽核共同source indices以匹配教師cache，先前排除的7張不加回。輪廓只影響歷史共同索引選擇，不參與新輸入圖片的處理。兩組都從頭訓練，不用cutout模型作初始化。

驗證也用完整原圖自然背景及同樣deterministic resize/pad，不使用平移／尺度挑戰。主要看accuracy與macro accuracy、CLIP−CE來源配對bootstrap95%CI。原分類head推論不需要CLIP教師或bbox；評估loader雖讀bbox生成輔助欄位，該欄位不送分類模型。validation仍用於選best，非untouched final test；單seed CI不含training seed變異。

## 實作與驗證

dataloader/OriginalImagePairs.py；scripts/train_object_scale.py新增--original-images，其他實驗預設行為不變。scripts/run_clip_original.py在GPU1現有深度消融結束後，依序CE／CLIP各100epochs，scripts/report_clip_original.py自動評估、生成REPORT.md與results.json。run位置記錄在overnight_logs/clip_direct/original_run.txt，狀態須搭配實際PID。

已完成runs/train/clip_original_1103_smoke/的一輪CLIP smoke；selected head及backbone gradient runtime checks通過。像素檢查證明補邊內完整來源影像與BICUBIC resize結果一致、背景未被替換；跨view／epoch輸入相同、bbox讀取沒有segmentation mask。正式兩組仍排程等待，不預填效果。

## 交接

本次研究說明與程式版本及SHA256送E: artifact inbox；本機專案根未核實，收件匣交付不表示本機專案匯入。匯入核對SHA256並保留不同版本。完整正式結果落遠端run，完成後另交付。
