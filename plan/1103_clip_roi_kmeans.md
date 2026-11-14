# CLIP監督中間層物件region分群

2026-11-03 remote140165。使用者同意補分析後執行，已完成。腳本scripts/analyze_clip_roi_kmeans.py，報告plots/kmeans/clip_roi_1103/REPORT.md，原始向量／中心／群指派在同目錄npz，來源checkpoint SHA256與split在manifest.json。

以6902 training原位96的物件region讀取，逐模型／層／readout獨立FAISS spherical K31，100iterations、50restarts、seed42；1768 validation比較原位96與長邊80+(7,-5)。是跨尺度與平移的聯合條件，不單獨歸因尺度。這是mask加權8x8對應採樣後的整物件region pooling，不是逐token圖或部件概念／語意命名。群號不可跨字典比較。四層index0..3，報告顯示index。

原始特徵群一致率A→CLIP：73.02→76.02、76.24→73.59、63.07→67.02、77.09→81.22%。深兩層提高3.96及4.13pp，第二層下降2.66pp，屬單seed描述性差異。投影後一致率65.67/61.54/64.82/73.13%，未勝過相應CLIP raw。投影配對cosine約.99，但同類異圖也.95-.96，不能把高cosine當語意／尺度不變性證明。各組31群均有佔用、有效群數26.14至29.31，未見單群坍縮；這不排除方向各向異性、語意不足或細節丟失。

結論：CLIP監督對深層region群指派有局部改善，但非所有層、未轉化為分類提升；尚未做逐token代表圖、部件標註、文字檢索或跨層因果驗證。不將region結果外推至原始token。全體分類結果見plan/1101_direct_clip_supervision.md。

版本副本、結果與SHA256送E: artifact inbox；本機專案根未核實，收件匣交付不代表本機專案匯入。匯入核對hash，保留不同版本。
