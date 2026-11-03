# 面試研究摘要（2026-11-03）

研究主題：階層式 Vision Transformer 的概念形成，以及表示在平移與尺度變化下的穩定性。

已完成：自訂 MergingViT、跨影像 token 分群、代表 patch 與跨層 merging 視覺化；建立 spherical K-means、平移／尺度評估及 activation intervention 工具。修正 Caltech annotation 座標錯誤，重新界定舊結果的有效範圍。

平移結果：101類、每類5張，共505張validation。1–7px平移accuracy：baseline41.56%、對齊增強48.39%、小位移增強T1 54.62%；8/16/32px：30.12%、46.65%、40.07%。顯示增強效果依位移範圍而異，小位移改善不保證大位移泛化。

物件獨立縮放：建立輪廓cutout、固定合成背景、paired views與物理座標ROI readout。完整6902 train／1768 validation、100epochs比較中，未見內插尺度80/112的平均accuracy：CE基準80.42%、一致性損失77.84%、gated residual merging加一致性78.01%。目前未勝過基準。

CLIP教師蒸餾：凍結CLIP ViT-B/32，以bbox image embedding監督四層ROI表示，學生backbone與投影共同訓練，推論不需要CLIP。未見內插尺度accuracy79.84%，相對基準差-0.58百分點，來源配對bootstrap95%CI[-1.61,+0.40]；尚無分類增益或獨立語意改善證據。

目前貢獻：建立可追蹤、可重現的表示分析與介入流程；以匹配對照、來源影像層級統計與資料座標稽核，區分特徵相似度、分類穩健性及語意可解釋性。

下一步：獨立validation逐層語意／檢索評估、constant-mean教師對照、多seed及獨立test，再驗證高支持跨層概念路徑。目前介入僅有單層與同層探索案例，尚未完成跨層因果驗證。

限制：數字來自不同協定，不可跨表直接比較；目前主要為單seed及探索validation。

依據：RESEARCH_CONTEXT.md、plan/1101_direct_clip_supervision.md、plots/kmeans/caltech_allclass_translation_1020/REPORT.md。

交接：本文件存於remote140165 /home/ref/RGB_SFM_V3；版本副本交付E: artifact inbox不代表已匯入本機專案。本機專案根待確認，匯入前核對SHA256，保留不同版本。
