# CBM、跨層CLIP對齊與物件縮放：討論

2026-11-01。提案討論，尚未開始新CBM訓練，既有實驗設定不改。

CBM可將跨層CLIP語意probe延伸為真正決策瓶頸：F_s→ROI→P_s→共同CLIP空間→具名概念c_s；用少量層的概念讀取／聚合成c，最後分類只能用c。僅在原模型旁加語意head仍是probe，若保留latent bypass則為hybrid，不宣稱所有分類皆經概念。

bbox提供區域，不能提供部件／屬性的概念真值；CLIP image embedding可作region教師，概念詞向量與CLIP分數作弱監督。須獨立人工概念子集與同類異概念對照，不能只檢查CLIP自我一致。概念庫涵蓋形狀、部件、材質、排列，避免以完整類別名稱充當全部概念。CLIP cosine不是校準機率；連續概念分數可能洩漏非語意資訊，即使沒有bypass也不保證概念可解釋。

尺度研究目標：對應可見物件／部件概念穩定，空間支援隨幾何變換；小到部件不可見時提供unknown／visibility，不強迫所有層所有概念相同。跨層語意可能改變，先以Stage1/2做輔助probe與末端單一嚴格CBM，再研究有部件標註的階層式CBM；不預設每層都已是同一物件概念。

訓練可用 L=CE(y, Wc+b)+lambda_concept L_concept+lambda_scale L_corresponding_concept_consistency，保留不同概念區辨。尺度一致性不是CBM自動帶來，需要paired物件變換、loss或架構。線性head的logit差可由||W||與概念差界定；只有足夠分類margin時才支持預測不變，不能泛化成架構保證。

實驗對照：原head；frozen backbone+CLIP probes；frozen backbone+嚴格CBM新head；同一CBM加paired scale consistency並回傳訓練；等容量latent bottleneck；可選hybrid作準確率／透明度折衷。概念accuracy/AUPRC、跨尺度穩定性與區辨性、定位、原尺度與未見尺度accuracy、分類margin、unknown率分開報。概念編輯對CBM分類有作用不等於原head使用該概念，也不是現實因果；若要解釋merging需在實際child→parent路徑做局部干預與下游追蹤。

bbox訓練可用GT輔助；推論若需GT box則明列oracle protocol，否則用預測定位／mask或image-only pooling，避免答案洩漏。按source切分，同圖不同尺度不得跨train/test。

來源：Koh et al. Concept Bottleneck Models https://arxiv.org/abs/2007.04612；Oikarinen et al. Label-Free Concept Bottleneck Models https://arxiv.org/abs/2304.06129；Yang et al. Language in a Bottle https://arxiv.org/abs/2211.11158。投影／語言CBM本身已有先例，研究貢獻應聚焦merging中部件組合、跨尺度支援與經驗驗證。

交接：遠端專案 /home/ref/RGB_SFM_V3。版本副本與SHA256送E: artifact inbox，本機專案根尚未核實；收件匣交付不代表本機專案匯入。匯入時核對hash並保留差異版本。
