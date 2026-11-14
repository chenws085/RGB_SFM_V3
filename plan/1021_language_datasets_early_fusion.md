# 自然語言語意分析：資料集與 early fusion 規劃

日期：2026-10-21。狀態：規劃，尚未下載資料、訓練或啟動實驗。
後續討論與先例查證見 [語言融合層對齊相關研究](1021_language_fusion_related_work.md)，含 FIBER、Bridger、SAM-PTx 等近似方法及更新後的控制組。
本文件位於 remote140165 專案 `/home/ref/RGB_SFM_V3`；不表示已同步至使用者本機專案或 E 槽 artifact inbox。

## 研究起點

既有 `vit_analysis_vlm/run_caption_analysis.py` 與 `run_experiment_pipeline.py` 將 input patch 和同群四張代表 patch 組成圖片，再由 Qwen-VL 按 prompt 描述共同特徵與吻合程度。設定範例指定 Qwen2-VL-7B，獨立腳本另支援 Qwen2.5-VL；每批實際版本須查輸出的 model 欄位。這是事後語意描述，不是將自然語言融入 MergingViT 的訓練或 forward。Qwen 的文字不直接當作語意真值。

延續 [10/15 自動語意對齊規劃](1015_language_alignment_feasibility.md)：自動命名為主，人工審核可作補充；分開回答「群可否命名」「語言能否改善表徵／分組」「命名概念是否參與分類」。

## 資料集選擇

| 資料集 | 官方提供的內容 | 預定用途 | 限制 |
|---|---|---|---|
| Visual Genome | 約 108,077 張影像、540 萬筆 region descriptions，另有物件、屬性與關係 | 區域文字候選來源；ROI–文字配對；projection 或 fusion 訓練 | 區域描述不是區域內每個 token 的標籤；不強迫整句落到每個小 patch |
| Broden | 場景、物件、部件、材質、紋理、顏色概念；標註粒度依來源而異 | 首輪外部語意驗證，檢查群 affinity 與概念標註、Qwen 描述是否一致 | 不當作自由文字 caption corpus；保留各來源有效標註範圍，未標註不一律視為負例 |
| PACO，先考慮 PACO-LVIS | 75 類物件、200 種部件及顏色／材質／圖案等屬性，含物件與部件 masks | 細粒度部件／屬性與組合概念驗證，如黑色把手；文字→區域檢索 | 屬性不是自由敘述；注意 unknown 與非完整標註，詞彙覆蓋不等同 Caltech |

官方來源（2026-10-21 查閱）：

- [Visual Genome](https://visualgenome.org/home)、[下載入口](https://visualgenome.org/api/v0/api_home)。
- [Broden / Network Dissection](https://netdissect.csail.mit.edu/)。
- [PACO 資料與格式](https://github.com/facebookresearch/paco/blob/main/docs/PACO_DATASET.md)。

實施前核對版本、使用條款、下載可用性及所需儲存量，再建立 source-image manifest、SHA256 與 split。目前不宣稱這些資料已在任一機器可用。

## 執行順序

1. **Qwen／既有群的語意基線**：凍結既有 checkpoint 與 training codebook，用 Broden 的固定子集做外部 probe；先 Stage 2，再擴展其他 stages。固定 Qwen 型號、prompt、代表圖採樣、解碼與 crop/context 條件。群名只從 training/calibration 決定，再在保留影像驗證；容許 unknown/mixed。
2. **細粒度檢查**：PACO-LVIS 加入部件、屬性及其組合，按標註 mask 與 token footprint 的重疊定義讀出；跨邊界 token 保留混合比例或排除，不硬貼單一名稱。
3. **圖文對齊基線**：Visual Genome 的 region box 對應 ROI pooled readout，凍結 backbone 與預訓練文字 encoder，先訓練線性 projection。Qwen 可提供候選敘述，但獨立標註負責主要驗證。
4. **early/intermediate fusion 消融**：以下列設計與 projection-only 對照；不跳過前述基線，不假設融合必然較好。

按 source image 切分，不能讓同圖不同 crop 跨 split；檢查多個資料集的來源影像重疊。保留外部資料與 Caltech 的 domain shift，以及預訓練資料重疊未知等限制。既有 validation 已用於選模，不重新宣稱 untouched final test。

## Early fusion 可行路線

本文件把「原 backbone 在特徵形成期間受到文字影響」稱為融合；需明列注入 stage。Stage 2 注入屬於中途融合，不能籠統稱最早期融合。僅在輸出向量加圖文 loss 是對齊監督，並非 token 層 early fusion。

| 路線 | 接入方式 | 對本研究的適合度與代價 |
|---|---|---|
| 文字條件 FiLM／gated adapter | 用文字產生通道 scale/bias，或以殘差 adapter 調整 stage token；比較 Stage 0/1/2 | 首輪低容量候選；保留空間網格，容易與原 merging 對照，但文字已改變特徵 |
| Visual→text cross-attention | stage token 作 Q，文字 token 作 K/V；殘差 gate 控制注入 | 優先深入的融合候選；可利用詞級訊息，需明確定義推論文字與負例 |
| ViLT 類 joint self-attention | patch 與文字 tokens 投影到共同維度，加 modality/position embedding 後共同 attention | 若要求真正輸入端 early fusion，這是清楚的基線；文字 tokens 須與 2D patch 網格分開保存，不能送入空間 patch merging |
| 文字條件 merging／grouping | 由文字影響局部 merging 權重，或另建可學 grouping/prototypes | 與概念組合主題直接相關，但會改變方法與概念路徑；首輪僅考慮固定局部父子關係內加權，動態跨位置 grouping 作後續獨立模型 |

這些是依文獻機制提出的本專案延伸，尚無本機成效證據。現有 FlexiblePatchMerging 的固定空間網格不能直接等同可學語意 grouping。

參考：

- [ViLT](https://arxiv.org/abs/2102.03334)：簡化視覺 embedding，讓圖文共同進入 Transformer，可作 joint-token 路線依據。
- [METER](https://arxiv.org/abs/2111.02387)：比較 merged attention 與 co-attention 等融合設計，支持把架構選擇做成消融，並非保證適合本資料量。
- [FiLM](https://arxiv.org/abs/1709.07871)：文字條件的 feature-wise affine modulation 參考。
- [GroupViT](https://arxiv.org/abs/2202.11094)：文字監督下學習視覺 grouping 的參考；其圖文對比訓練不是直接把文字 tokens early-fuse 進視覺 backbone，不能混稱。

## 先決定推論時文字從哪裡來

**文字查詢型**：推論時提供「金屬把手」等查詢，輸出相符區域或群；適合 cross-attention／FiLM。此時群是 query-conditioned，同圖換文字可能換群。名稱與指標須保留 query 條件，不能與原 image-only 群直接視為同一概念。

**影像自主型**：目標是只有影像也能形成語意群。可優先採圖文監督 grouping／中間層對齊 loss，推論的視覺分組不依賴每張圖的描述；也可研究 fusion teacher→image-only student 蒸餾。後者須另驗證蒸餾後的語意、分類與穩定性。單純訓練時輸入正確 caption、測試時拿掉，會產生輸入分布落差；null-text／text dropout 可作控制，但不保證解決問題。

不可在分類評估時輸入含真實類別的 caption，然後把分類提升解讀為視覺表徵改善。即使文字由 Qwen 自動生成，也須明列 teacher 額外資訊，與 image-only 路線分開比較。

## 最小比較與評估

第一輪比較原視覺模型＋Qwen 事後描述、projection-only、FiLM、單一 cross-attention adapter；Stage 1/2 為先導，若要驗證嚴格 early fusion 再補 Stage 0。使用相同來源切分、ROI、文字 encoder 與可比訓練預算，記錄 trainable parameters、VRAM 與延遲。

- 語意：群內概念純度／support、命名 coverage/abstention、獨立標註下 P@5/P@10、AUPRC 與 prevalence；有可用區域標註時另報 localization IoU。
- 文字敏感性：同義改寫、錯配文字、同類 hard negatives、null-text、text-only 與 shuffled-pair 控制。類別名-only、teacher-only 作基線。
- 視覺穩定性：沿用固定平移／尺度協定；融合模型在相同 query 下比較，另測改寫 query。跨模型群 ID 須匹配，不能直接比較編號。
- 分類與機制：image-only 與 query-conditioned 結果分開；命名或 attention 圖不當作因果證據。仍需原模型 activation intervention、位置／norm 等匹配控制。
- crop-only、context-only、兩者同時輸入分開測，避免只讀背景或整圖物件。

首輪產物預定含資料 manifest、版本與 split、概念字典、Qwen 輸出及設定、群→詞排序、ROI/token 標註對應、檢索與拒答指標、融合消融報告。實際樣本量、checkpoint、文字 encoder、超參數與執行機器在實施時固定，不在本次筆記中假定已選定。
