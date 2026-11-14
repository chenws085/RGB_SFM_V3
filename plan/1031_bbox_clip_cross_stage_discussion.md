# bbox→CLIP 跨層語意對齊與物件縮放：討論提案

使用者提案：用 bbox 資料訓練各層中間表示到 CLIP 語意空間，改善跨層解釋性，再建立縮放不變性。此為討論，尚未授權啟動這項新訓練；不更改既有A/B/H實驗。

可行，但須區分：(1) frozen backbone 的 projection probe 只驗證可讀出語意；(2) gradient 回到 backbone/merging 的聯合訓練才可能改變原分類尺度穩定性。投影後一致不表示原始特徵不變，也不證明原分類決策使用該語意。

架構：z_s = normalize(P_s(ROIAlign(F_s, transformed_bbox)))。每stage用不同小型投影器，但輸出到同一 frozen CLIP final embedding 座標系。Teacher 用原始高解析bbox crop的CLIP image embedding；文字類別／屬性向量作額外語意錨點與評估。bbox只給位置，不自動提供部件／屬性標籤。物件bbox監督整個ROI，不能把框內每個token當作完整物件概念；淺層無法讀出的語意本身也是研究結果。

縮放時只改物件，背景固定；bbox/mask隨幾何變換。學生要看變換後的整張畫布，不能先把bbox resize回固定大小當輸入，否則把尺度問題在輸入端消去。Teacher可使用同來源canonical crop作共同target，但先稽核teacher對尺度／背景／模糊的敏感性；極小物件資訊已遺失時不硬性要求細粒度語意一致。

第一階段：凍結checkpoint與CLIP，訓練各層linear／小MLP probe，按source切分，檢查跨層與跨尺度的檢索／概念區辨；加入類別名only與同類異圖／相近概念對照，避免只學分類或投影坍縮。相同來源不同view不得當負例；同類不同實例是否為負例依instance或semantic目標區分。

第二階段：保留classification CE與原readout，加入CLIP ROI蒸餾及同內容跨尺度consistency，逐層解凍或adapter微調。先晚層再往前消融，loss權重使用探索validation決定。不要把所有層／所有token硬性對齊完整物件向量。後續有part bbox／mask時，再研究小物件淺層與大物件深層的部件對應、實際merging路徑與介入。

關鍵對照：原模型；僅projection；projection+語意蒸餾回傳；加跨尺度一致性；等容量image-only對照。原分類頭未見尺度accuracy／最差條件、投影前後region cosine／retrieval、不同概念區辨、部件及分類介入分開報。只在CLIP head改善不能聲稱原merge_ViT已改善。

先例：ViLD https://arxiv.org/abs/2104.13921 已用region embedding對齊teacher圖文embedding；MaskCLIP https://arxiv.org/abs/2112.01071 探索CLIP dense supervision；CLIP局部語意對齊並非原生保證。創新應聚焦跨尺度的跨層部件組合與merging作用，而非投影本身。

交接：遠端專案 /home/ref/RGB_SFM_V3。版本副本與SHA256清單經既有HTTPS上傳E: artifact inbox；不代表本機專案已匯入，本機專案根尚未核實，匯入須核對hash並保留差異版本。
