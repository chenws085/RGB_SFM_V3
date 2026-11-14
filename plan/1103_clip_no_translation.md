# 本次追加：去除平移需求，評估CLIP訓練訊號

2026-11-03，remote140165 /home/ref/RGB_SFM_V3。使用者要求若未做則追加無平移訓練。檢查既有實驗沒有同條件無平移對照，因此新增。

## 問題與共同條件

只移除平移，保留既有物件尺度64/96/128取樣、contour cutout、224畫布與合成純色背景；不自行把問題改成自然背景原圖或同時去除尺度。每對view仍有不同大小，但物件始終置中。評估亦不含平移，尺度48/64/80/96/112/128/144全置中。主要指標置中96分類accuracy；80/112平均為次要未見尺度指標。這是純色加工圖上的訓練幫助，不代表自然場景泛化。

新增兩組CE-only與CE＋深層Stage3/4 CLIP cosine0.2。共享seed42、原merging、初始化、資料索引、6902train／1768validation、100epochs、batch16、AdamW3e-4。CLIP凍結、學生主幹及兩個active投影可訓練；原prediction/region consistency損失權重皆0。不以既有含平移A作唯一控制，重新訓練無平移CE基準。

## 程式與驗證

ObjectScalePairs新增no_translation（預設False），仍取樣原位移亂數後覆寫為0，保留其他尺度亂數順序，避免改位置同時改尺度取樣。訓練入口新增--no-translation，作用於train與validation。評估入口同名flag只評估shift(0,0)。保留既有預設行為，正在進行的深度消融不改設定。

已驗證train／validation雙view的ox/oy皆為center，且與原流程的物件尺寸取樣一致。CLIP深層無平移smoke已完成1epoch，checkpoint與history位於runs/train/clip_no_translation_1103_smoke/。

新增scripts/run_clip_no_translation.py與scripts/report_clip_no_translation.py。正式run根目錄記錄在overnight_logs/clip_direct/no_translation_run.txt；先核對正在跑的depth pipeline的/proc command line與terminal status，待它完成才使用GPU1，避免同GPU兩組正式訓練重疊。佇列狀態須搭配PID，不只看waiting檔。CE→CLIP依序各100epochs，完成各自7個置中尺度評估，再自動產REPORT.md、results.json、comparison.png。報告會核對split、初始化規約、checkpoint hash，作來源層級bootstrap差值及95%CI。CI不含training seed變異。

## 尚未有正式結果

目前只有smoke與活佇列，不能寫成已改善。採探索validation依共同置中80準確率選best，非untouched final test。兩組從頭訓練、單seed；若結果不支持CLIP增益，照實報告。

## 交接

文件、程式、狀態快照及SHA256經HTTPS送E: artifact inbox；本機專案根尚未核實，交付不表示本機專案匯入。匯入核對SHA256，保留不同版本；原有深度消融繼續執行。

## 使用者要求暫停

已依使用者要求暫停此組。當時狀態為waiting_for_live_depth_pipeline，正式CE／CLIP均尚未開始；已停止佇列及等待最終交接的watcher，pipeline_status記錄paused。smoke與既有檔案保留。先前深度消融仍在訓練，未停止。此組不會在深度消融結束後自動啟動；恢復需重新依使用者指示處理。
