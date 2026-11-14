# 0921平移穩定性：全局MHA／窗口版重測

遠端remote140165，專案`/home/ref/RGB_SFM_V3`。使用既有完整自然原圖100epoch CE模型，global與window相同split／seed42／訓練預算及原merging；window7/7/7/4，無移窗與額外relative bias。checkpoint與split見`plots/reports/attention_translation_20261109/manifest.json`。

主問題是兩個模型是否同樣出現局部token穩定但整體分類不穩定、深層或上下方向較敏感，以及窗口是否改善而不損害深層表示。這是診斷attention選擇的實驗，非預設窗口勝出。

完整原圖含自然背景縮到長邊160、224畫布置中；四方向1–8／16／32px，無來源裁切。使用實際整數resize比例與absolute bbox中心，不沿用0921漏加bbox offset座標。此輪bbox中心單token不同於尺度報告整物件8×8 ROI，也不同於舊contour平均中心；保留歷史結果不覆寫。每模型各層只用6902training160中心token建FAISS spherical K31／100iter／50restarts／seed42字典，1768validation固定字典測試；另列airplanes。字典不跨模型共用群號，只比較各自平移前後一致率。

主比較Stage0/1/2各移一token（8/16/32px）；同時固定像素位移比較深度，避免深度與位移混雜。保留四方向、1–7非對齊平移，觀察window邊界效應。分開報cluster agreement、token cosine、classification accuracy／prediction consistency與final pooled feature cosine；來源配對bootstrap不含training或分群seed變異。模型在224原圖訓練，160原位accuracy需同時報告，不能把尺度分布差異當純平移效果。validation曾選best，單seed診斷不等於最終泛化結論。

入口：`.venv/bin/python scripts/evaluate_attention_translation.py plots/reports/attention_translation_20261109`；日誌`overnight_logs/translation_attention_1109/eval.log`。結果完成後見該目錄REPORT.md／summary.csv／paired.npz與manifest。

文件版本與SHA256交付本機E:\RGB_SFM_artifacts inbox；本機專案根目錄尚未確認，inbox交付不代表專案匯入。匯入時依相對路徑驗hash，保留不同版本。

## 完成結果與決策

兩模型全1768validation×41位置×4stage完成，共579904個模型token配對；零位移identity、finite與各stage8/16/32px精確一token檢查通過。四方向全類別one-token群一致率global/window為90.14/85.42%、83.14/79.45%、74.84/71.30%，三組差異CI均低於0。32px分類一致率23.90/27.59%，window+3.69pp CI[+2.40,+4.99]；accuracy13.66/14.49%，差CI跨0。飛機160張32px分類一致率66.41/76.88%，但不可代表全類別。兩者上下移較左右不穩定，局部同群不等於分類穩定。

暫保留global主基準、window對照，不全面切換；window沒有同時改善表示與分類。160原位全類別accuracy20.25/19.23%，相對224輸入有很強尺度分布差異；正式選型需同letterbox160訓練協定與多seed另作對照，本輪未啟動重訓。詳細數據、方向、CI及限制見REPORT.md；全類別與飛機分開報。
