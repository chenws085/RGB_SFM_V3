# Letterbox160局部attention變體篩選

遠端remote140165，專案/home/ref/RGB_SFM_V3。使用者要求多測window/local attention及相關超參數。第一輪只調attention範圍、網格原點與使用stage，保持optimizer/augmentation/loss不變。沿用已核對的global與window7 checkpoint，不重訓既有基準。

## 新增五組

|名稱|Stage0/1/2/3 attention|目的|
|---|---|---|
|window4|4/4/4/4固定窗|縮小局部範圍，含深層局部化；14與7網格padding有valid-key mask|
|window14|14/14/7/4固定窗|較大窗口，僅Stage0仍分窗|
|window7_offset3|7/7/7/4，前兩stage原點偏移3token|固定窗邊界位置消融；非循環padding，無跨圖邊緣連接；非完整Swin交替移窗|
|window7_stage0|Stage0 window7，其餘global|只限制最淺層，保留後續全局整合|
|sliding7_stage01|Stage0/1重疊7×7中心鄰域，其餘global|移除互斥分窗邊界，保留相同名義鄰域寬度；邊緣遮掉無效keys|

Stage從0開始。每stage只有1block，不能以單一stage內交替移窗描述本模型。全部沿用原absolute position與merging，無relative bias、CLIP或新attention參數。原state_dict逐張量初始化一致，參數1304261；不同attention運算dropout形狀不同，不主張整段訓練亂數軌跡相同。

## 公平性與測試

同歷史global：Caltech101 seed42、6907training/1770validation、長邊160完整自然原圖置中224、歷史bilinear resize/整數meanRGB補邊；flip/jitter/erasing、無平移增強；batch32、AdamW lr3e-4 wd.01、CE smoothing.1、cosine100epoch eta_min1e-6、patience20。預算與停止規則一致，不要求每組實際停止epoch相同。

每組獨立從頭訓練後，用各自training中心token固定FAISS spherical K31/100iter/50restarts/seed42字典；四方向1–8/16/32px，1770validation與airplanes子集，修正absolute bbox中心。原位accuracy、分類一致/accuracy、各stage token cosine/群一致率、上下左右分別報。one-token主比較Stage0/1/2=8/16/32px；固定pixel結果避免深度與位移混淆。不可僅以原位accuracy挑選；需保住分類同時改善局部與分類平移。來源配對CI不含training/分群seed變異，單seed多變體探索且validation曾選模/選設定，CI未多重比較校正；不作最終泛化結論。

新增測試：offset窗／padding窗／sliding對照獨立dense masked attention參考，有限梯度與完整model backward、初始化參數一致；原window/global相容檢查通過。輸出在overnight_logs/local_attention_variants_1109/*checks.json。

入口scripts/run_local_attention_sweep.py，以GPU1依序訓練和評估；每組狀態、來源SHA256與快照、日誌、訓練與測試路徑見runs/train/local_attention_sweep_20261109T180400Z/status.json。每完成一組更新REPORT.md與comparison.json。部分失敗記錄原因並繼續其餘組；不能將queue啟動當作完成。此刻尚無新五組最終結果。

完成時自動交付研究文件/結果版本及SHA256至E:\RGB_SFM_artifacts inbox；本機專案root/access未確認，交付不是專案匯入。模型留遠端，manifest提供checkpoint hashes；匯入文件須確認root、驗hash並保留不同版本。


## 完成判讀（2026-11-10）

五組訓練／1770validation四方向41位置評估全部完成，各組finite、零位移identity、精確一token與manifest artifacts SHA256通過。單seed、多變體探索，以下CI未校正多重比較、不含訓練seed變異。

分類候選：window7_stage0原位63.56%，1–7px60.39%、32px29.96%；global為62.09/58.71/27.32%。32pxaccuracy差+2.64pp，來源95%CI[+1.55,+3.73]，consistency+2.50pp CI[+1.23,+3.79]。但one-token群一致87.05%較global89.20%低2.15pp，CI[-3.03,-1.24]；不能說同時改善局部表示。

折衷候選：window14，1–7px60.66%高於global58.71%，32px27.43%接近27.32%；one-token88.46%接近89.20%，差CI跨0。這是固定模型的描述性接近，不是證明等效或不劣。

sliding7前兩stage，one-token89.68%為點估計最高，但相對global+0.48pp CI[-0.41,+1.42]未支持明確改善，32pxaccuracy24.07%低3.25pp。偏移window7原位63.90%、1–7px61.52%最高，32px25.16%卻低於global27.32%；小窗口4的32px20.69%最差。

建議保留global研究主基準，window7_stage0作分類穩定性候選，window14作折衷對照；沒有新變體同時全面提升分類與局部平移穩定性。下一個選型驗證應聚焦上述兩候選與global的額外training seeds，不能由本輪validation篩選就宣布架構優勝；尚未啟動新訓練。


## 2026-11-10 報告做法補充

已在 `runs/train/local_attention_sweep_20261109T180400Z/REPORT.md` 的「各變體的實際做法」加入七組逐stage連接表、global/window/sliding差別、offset3分組例子、padding key mask與輸入letterbox背景的區別，以及各比較能回答的問題與混雜因素。結果數值未變。
