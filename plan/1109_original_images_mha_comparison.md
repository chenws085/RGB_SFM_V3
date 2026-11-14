# 完整自然原圖：全局MHA／窗口MHA對照

日期2026-11-09；機器remote140165；專案/home/ref/RGB_SFM_V3。

使用者要求「跑一筆這個的MHA對照，使用完整原始圖片」。本輪用完整來源RGB與自然背景，等比例長邊224縮放並置中補邊到224畫布、ImageNet normalization，無去背、裁切、flip、隨機尺度或平移增強。single view／單次forward，CE only，無CLIP；bbox只供ROI特徵分析，不送進分類模型。歷史共同來源排除7張後6902training／1768validation，沿用相同split，不將「完整原圖」解讀為回加不同來源。

## 訓練對照

全局基準沿用已完成100epoch的runs/train/clip_original_20261103T065816Z/CE，best epoch97，validation56.56%。確認config original_images=True、clip_cache=None；原圖loader與當時source SHA256一致。窗口模型在runs/train/window_original_20261109T080440Z/window從頭訓練，seed42、100epochs、batch16、AdamW3e-4/weight_decay.05，原merging、相同參數1304261與初始權重規則。不是拿全局已訓練checkpoint切換窗口後直接評估。

window7實際窗口7/7/7/4、grid28/14/7/4，後兩stage覆蓋整層，無移窗、無額外relative position bias；為前兩stage局部化消融，非完整Swin。MHA選擇仍未定，不預填勝負。資料／loss／預算等比對與像素完整性檢查在run/input_checks.json。原先去背模型的數值不能當成本輪同分布對照。

## 評估協定

完整原圖224分類accuracy、macro accuracy及來源配對CI。自然圖每層原始bbox ROI（均勻8×8、無分割mask）建立FAISS spherical K31／100iter／50restarts／seed42字典，只用training224，Hungarian跨層群對應也只用training224。validation評估各層、跨層與跨尺度指派一致率、同類異圖差距及群佔用，不用validation擬合字典或群號映射。仍是整物件ROI，非逐token或投影共用字典。

附帶的尺度測試把完整圖片連同背景等比例縮成長邊112/160/192/224再置中補邊；不是物件獨立縮放。主跨尺度224↔160/192；相鄰層名義支援比例控制為淺層112→深層224。保持完整內容、不加移位；224圖片在224畫布上作位移相位校正可能裁切，所以本輪不做該控制，也不聲稱有效RF已匹配。

## 啟動時狀態與交接（最終結果見下方完成段）

啟動時訓練已開始，當時尚無窗口最終結果。scripts/finalize_original_window_attention.py核對live trainer PID及命令，完成後自動執行scripts/report_original_window_attention.py；最終報告預期在run/spherical/REPORT.md，pipeline_status必須搭配process/status核對，不把working檔案視為完成。原圖dataset dataloader/OriginalImageViews.py於224與原loader逐值相同，12來源×4尺度完整RGB像素／epoch決定性檢查通過。

此計畫、程式與狀態版本及SHA256交付E:\RGB_SFM_artifacts inbox，本機project root與匯入未確認；交付不是專案匯入，確認root後驗hash並保留不同版本。模型與大型cache留遠端。正式結果完成後再補交。

## 實驗已完成

窗口100epochs完成，best84；全局沿用同設定100epoch best97。正式結果：`runs/train/window_original_20261109T080440Z/spherical/REPORT.md`。

# 完整自然原圖：全局／窗口MHA純CE對照

訓練保留完整RGB影像與自然背景，只做長邊224等比例縮放、置中補邊與正規化；無去背／裁切／額外增強，single forward。global沿用已完成同設定100epoch純CE模型，window從頭訓練；seed42、batch16、6902training／1768validation、原merging與相同參數量。沿用共同來源索引，歷史排除的7張不加回。

window7實際窗口7/7/7/4，後兩stage覆蓋全層；無移窗或relative position bias，非完整Swin。分類輸入不需bbox；bbox僅供8×8整物件ROI特徵讀取，沒有分割mask。分群是原始ROI各層獨立字典＋training Hungarian對應，不是投影共用字典或逐token比較。

| 原圖224分類 | 全局MHA | 窗口MHA |
|---|---:|---:|
| accuracy | 56.56% | 55.49% |
| macro accuracy | 38.24% | 37.04% |
| best epoch | 97 | 84 |

## 同層跨尺度：整圖224→160/192

| 層／配對 | 全局MHA | 窗口MHA | 差值 |
|---|---:|---:|---:|
| 1 | 57.18% | 76.56% | +19.37pp |
| 2 | 56.65% | 65.27% | +8.63pp |
| 3 | 43.55% | 48.30% | +4.75pp |
| 4 | 42.28% | 38.38% | -3.90pp |
| 平均 | 49.92% | 57.13% | +7.21pp |

## 跨層同原圖224

| 層／配對 | 全局MHA | 窗口MHA | 差值 |
|---|---:|---:|---:|
| 1↔2 | 42.82% | 44.85% | +2.04pp |
| 1↔3 | 30.26% | 25.96% | -4.30pp |
| 1↔4 | 19.85% | 16.18% | -3.68pp |
| 2↔3 | 34.45% | 36.82% | +2.38pp |
| 2↔4 | 22.91% | 17.70% | -5.20pp |
| 3↔4 | 35.69% | 31.96% | -3.73pp |
| 平均 | 31.00% | 28.91% | -2.08pp |

## 跨層＋跨尺度：整圖224↔160/192

| 層／配對 | 全局MHA | 窗口MHA | 差值 |
|---|---:|---:|---:|
| 1↔2 | 34.74% | 42.27% | +7.52pp |
| 1↔3 | 24.45% | 24.59% | +0.14pp |
| 1↔4 | 16.61% | 14.66% | -1.95pp |
| 2↔3 | 28.59% | 32.79% | +4.20pp |
| 2↔4 | 18.58% | 14.64% | -3.95pp |
| 3↔4 | 23.50% | 22.41% | -1.09pp |
| 平均 | 24.41% | 25.23% | +0.81pp |

## 相鄰層名義支援比例匹配：淺層整圖112→深層整圖224

| 層／配對 | 全局MHA | 窗口MHA | 差值 |
|---|---:|---:|---:|
| 1↔2 | 24.60% | 40.33% | +15.72pp |
| 2↔3 | 19.51% | 29.24% | +9.73pp |
| 3↔4 | 15.21% | 14.88% | -0.34pp |
| 平均 | 19.78% | 28.15% | +8.37pp |

## 限制

縮放挑戰改變完整圖片與背景大小，與先前物件獨立縮放不同，不能把兩套accuracy當同一測試分布。名義stride比例匹配不是實測有效RF或逐token支援框匹配。未做224大圖平移相位校正，因為固定224畫布可能裁切原圖，違反完整內容要求。

字典只用training224建立並固定，validation不fit／配群，但曾用於選best，非final test。bootstrap2000次以來源為單位，先在影像內平均，不含seed變異，CI未多重比較校正；群佔用、同類異圖對照與CI見results.json。

遠端/home/ref/RGB_SFM_V3；研究文件版本及SHA256交付E: inbox不是本機專案匯入，本機root未核實。

## 完成判讀與來源配對區間

全局／窗口原圖224 accuracy為56.56%／55.49%，窗口−全局−1.07pp，95%CI[-2.83,+0.62]，未支持兩者分類的明確差異。
窗口同層跨尺度平均+7.21pp，CI[+5.89,+8.53]；一般跨層＋跨尺度平均+0.81pp，CI[-0.24,+1.88]，不能宣稱整體跨層改善。相鄰層名義支援比例匹配平均+8.37pp，CI[+6.73,+9.99]，主要來自1↔2及2↔3；3↔4未改善。同類異圖差距在支援比例匹配下增加6.94pp。這些CI僅反映固定模型／字典的來源抽樣，不含training或kmeans seeds。
因此保留全局／窗口架構選擇未定：窗口對部分尺度指派較穩定，分類與深層跨層並未全面勝出；本輪不是物件獨立縮放，也不是有效RF實測。
驗證詳見run/validation_checks.json；兩模型224分類重現各自best epoch的歷史accuracy，所有尺度來源一致且字典中心為unit norm。
