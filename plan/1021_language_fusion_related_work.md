# 語言融合層對齊：討論紀錄與相關研究

日期：2026-10-21。位置：remote140165 `/home/ref/RGB_SFM_V3`。
狀態：文獻調查與研究設計；未下載資料／模型、未啟動訓練，未同步本機專案或共享 inbox。
承接 [資料集與 early fusion 規劃](1021_language_datasets_early_fusion.md) 與 [10/15 可行性分析](1015_language_alignment_feasibility.md)。

## 本次討論與判斷

使用者依序確認現有 Qwen 語意分析、詢問後續資料集、要求記錄規劃，再提出 early fusion，以及「能否只訓練這些融合層，對齊 CLIP 或其他文字端語意」。最後要求擴大查證先例並留下筆記。

現有流程是 Qwen-VL 對 input＋四張群代表 patch 圖片做事後描述，沒有將文字放進 MergingViT forward。本次提出的新實驗是凍結原 backbone 和預訓練文字 encoder，在中間 stage 插入可訓練 FiLM／cross-attention adapter，配合 ROI projection 與文字語意目標。可行性有文獻支持，但尚未在本模型上實證。

更新判斷：**融合層、凍結模型的多模態 adapter、局部文字對齊、文字監督 patch grouping 都已有研究，不能單憑加入這些元件宣稱新穎性。** 本次是針對性調查，不是系統性文獻回顧，也不足以宣稱精確組合沒有人做過。

## 最接近的先例

以下以原論文、會議出版頁與作者官方程式為依據（2026-10-21 查閱）。表中「可借鑑」是本研究的推論，非原文對本模型的驗證。未逐篇執行程式；凍結範圍、loss 與注入位置在實作前仍應逐一核對官方 code/config。

| 方法 | 已做過的事情 | 與我們的關係及界線 |
|---|---|---|
| **FIBER，NeurIPS 2022** | 在視覺與文字 backbone 內插 cross-attention，使用可學 gate；先圖文、再圖文框的訓練 | backbone 內 gated fusion 的直接先例；原文不是只訓練融合層、凍結自訂 MergingViT 的設定 |
| **Bridger / ETRIS，ICCV 2023** | 凍結圖文 encoder，用中間多層特徵、可訓練 Bridger 與輕量 decoder 做 referring image segmentation | 凍結 encoder＋中間特徵融合的直接先例；須分清旁路特徵融合和改寫原視覺主幹，不能僅依名稱假設改變原 merging |
| **SAM-PTx，2025 preprint／IEEE Access 2026** | 將凍結 CLIP 文字 embeddings 經 Parallel-Text adapter 注入 SAM image encoder；多數原參數凍結 | 很接近「凍結文字端＋視覺 encoder 內可訓練 adapter」；原任務是文字／點提示的監督分割，不是原視覺群的無條件命名，也不等同純對比對齊 |
| **Fusioner，2022 preprint** | 用輕量 Transformer fusion module 結合凍結視覺與語言特徵，做 open-vocabulary segmentation；亦測獨立預訓練的視覺／語言模型 | 凍結非共同預訓練 encoder 仍可學融合的先例；特徵後接 fusion 不等同視覺 backbone 輸入端 early fusion |
| **MaPLe，CVPR 2023** | 耦合視覺與文字 prompts，在多個早期層學習跨模態提示 | 深層 prompt tuning 的重要對照；learned prompts 與每張圖的自然語言 query、cross-attention 是不同機制 |
| **LSeg，ICLR 2022** | 以文字 encoder 的語意表示監督 dense visual embeddings，學像素與文字標籤對齊 | 支持「文字作目標、影像負責讀出」；對齊 loss 本身不要求 early fusion |
| **DenseCLIP，CVPR 2022** | pixel–text score maps 引導 dense prediction，並以影像 context 改善文字 prompting | 局部圖文分數與語意監督先例；不能描述為我們同樣的固定文字 encoder＋stage adapter |
| **GLIP，CVPR 2022** | 將 detection 與 phrase grounding 統一，加入語言感知 deep fusion、word–region alignment | 區域文字對齊與 deep fusion 已有成熟先例；其大規模 grounding 訓練不代表本資料量可直接重現 |
| **GroupViT，CVPR 2022；SegCLIP，ICML 2023** | 圖文監督下學視覺 grouping；SegCLIP 用可學中心聚合 patches，另有 masked reconstruction 與 superpixel KL | 與「讓合併／分組更語意化」直接相關；文字監督 grouping 不等同將文字 tokens 放入視覺主幹 |
| **CAT-Seg，CVPR 2024** | 聚合 CLIP dense image–text cosine cost volume，並調適 encoder 做 open-vocabulary segmentation | 應補入 score/cost-level 基線，避免只比較 feature fusion；不是證明本模型也會得到相同 unseen 泛化 |
| **SAIL，CVPR 2025** | 對獨立預訓練的視覺／文字模型做輕量 alignment tuning，官方流程含 pre-encoding 與對齊 | 支持除 CLIP 以外文字端的 projection 基線；不等同中間 stage fusion，官方資料／硬體規模不等於本機預算 |

## 原始來源與程式入口

1. FIBER：[原論文全文](https://arxiv.org/html/2206.07643v2)、[官方 code](https://github.com/microsoft/FIBER)。特別核對 §3：gate 初始化為 0；image–text contrastive loss 時計算關閉 fusion，MLM／ITM 時計算開啟 fusion。因此不能聲稱 FIBER 用「已看過正解 caption 的 fused feature 對齊同一 caption」作為其標準 ITC。
2. Bridger：[論文](https://arxiv.org/abs/2307.11545)、[作者 PDF](https://guanbinli.com/papers/Bridging_Vision_and_Language_Encoders_Parameter-Efficient_Tuning_for_Referring_Image_ICCV_2023_paper.pdf)、[官方 code](https://github.com/kkakkkka/ETRIS)。
3. SAM-PTx：[2025 preprint](https://arxiv.org/abs/2508.00213)、[2026 出版頁](https://ieeexplore.ieee.org/document/11413927/)。出版頁瀏覽遇 JS 驗證，以可讀的作者 preprint 核對機制；尚未核對正式版全文差異／可重現 code。
4. Fusioner：[作者論文](https://arxiv.org/abs/2210.15138)。
5. MaPLe：[論文](https://arxiv.org/abs/2210.03117)、[官方 code](https://github.com/muzairkhattak/multimodal-prompt-learning)。
6. LSeg：[論文](https://arxiv.org/abs/2201.03546)、[官方 code](https://github.com/isl-org/lang-seg)。
7. DenseCLIP：[論文](https://arxiv.org/abs/2112.01518)、[官方 code](https://github.com/raoyongming/DenseCLIP)。
8. GLIP：[論文](https://arxiv.org/abs/2112.03857)、[官方 code](https://github.com/microsoft/GLIP)。
9. GroupViT：[論文](https://arxiv.org/abs/2202.11094)、[官方 code](https://github.com/NVlabs/GroupViT)。SegCLIP：[ICML 出版頁](https://proceedings.mlr.press/v202/luo23a.html)、[官方 code](https://github.com/ArrowLuo/SegCLIP)。
10. CAT-Seg：[作者專案](https://cvlab-kaist.github.io/CAT-Seg/)、[論文](https://arxiv.org/abs/2303.11797)。
11. SAIL：[CVPR 2025 正式論文](https://openaccess.thecvf.com/content/CVPR2025/papers/Zhang_Assessing_and_Learning_Alignment_of_Unimodal_Vision_and_Language_Models_CVPR_2025_paper.pdf)、[官方 code](https://github.com/lezhang7/SAIL)。搜尋另返回標題不同的 OpenReview 稿，不把其版本細節與正式論文無條件混用。

## 我們可測的具體設計

首輪固定 CLIP text encoder，把候選概念編成共同概念庫 E；每張圖看到同一庫，包含候選正／負概念，不按 ground truth 篩選。視覺特徵 H 在 stage s 經過：

`H' = H + alpha * CrossAttention(LN(H), E, E)`

跨注意力含可學 Q/K/V 與 output projections，維度由 projection 對齊；H' 再進原後續 blocks／merging。Stage 0 才是更早期注入；Stage 1/2 應明確稱中間層 fusion。概念庫過大時須記錄計算量與候選採樣策略，避免候選選取洩漏答案。全局句向量可作概念 memory 的一個 slot，不能直接等同已具局部對齊能力的詞級 CLIP hidden states。

以 region caption 的 frozen text embedding 作目標，ROI pooled visual readout 經可訓練 P 投影／正規化後計算多正例圖文對比 loss；可加同 ROI 的凍結 CLIP image embedding 蒸餾，以及原分類 loss／原模型輸出蒸餾。權重為超參數，尚未選定；CLIP 小 crop teacher 亦可能不可靠，不能把其相似度當獨立真值。

原 backbone 權重可凍結，但 adapter 後的主幹仍需可微運算才能傳梯度，不能整段使用 no_grad 或預先快取原末層特徵取代。凍結權重不代表 activation 記憶體為零。若 alpha 初始化 0，首步 adapter 支路梯度可能為零，須檢查 gate 與後續梯度；不以 gate/attention 大小作語意或因果證據。

第一輪基線建議：

- B0：原模型、原球面 codebook＋Qwen 事後描述。
- B1：凍結 backbone，僅 ROI projection 對齊文字。
- B2：與融合模型參數量接近的 image-only residual adapter＋相同對齊 loss，隔離新增容量效果。
- B3：共同文字概念庫＋單一 gated cross-attention adapter，先 Stage 1/2，再比較 Stage 0。
- B4：若預算允許，文字分數／cost aggregation 基線；FiLM 作低容量替代。

CLIP、SigLIP 或其他文字 encoder 應以各自固定版本／維度／正規化獨立比較。若換成純文字 sentence encoder，需透過圖文配對學視覺連結，不能直接視為已有視覺共同空間。

## 必須防止的捷徑與評估界線

把同一張圖的正解 caption 送進融合層，再要求 fused feature 靠近該 caption，存在複製文字捷徑；stop-gradient 文字端不會消除這個捷徑。共同概念庫能減少正解獨享資訊，但也不保證模型使用了影像。

因此加入 image-zero/shuffle、text-only、錯配文字、shuffle-pair、null-text、概念庫增減／次序與同義改寫控制；比較共同庫與逐圖 query 的兩種設定，分別報告。若只做查詢條件的定位，文字輸入是合法任務資訊，但不能把結果解讀成 image-only 群改善。

Visual Genome 提供 ROI–文字訓練配對；Broden／PACO 保留來源影像獨立的語意評估。沿用前一份規劃的 mask/footprint 對應、未標註處理、跨資料集重疊檢查、P@k/AUPRC/support、未知拒答、crop/context 消融。額外測未見同義詞與部件屬性組合、平移／尺度穩定性及分類保存。

融合後的群必須重建 training codebook、保留新模型 lineage；可以另測相對原 codebook 的漂移，但不直接將新特徵群與原群同編號視為相同語意。原群解釋與語言訓練後新群分開報告。

## 潛在研究問題與下一步

目前較合理的研究問題是：**在受控容量、資料與文字先驗下，merging 前的語言融合，是否比輸出端對齊更能形成可定位、跨變換穩定、且參與分類的部件／屬性概念？** 這是待驗證的問題，不是已成立的貢獻。

先深入 FIBER、Bridger、SAM-PTx 的注入位置與梯度路徑，LSeg／SAIL 作 alignment 對照，SegCLIP／GroupViT 作 grouping 對照。對任何「新方法／首次」主張，還需針對 shared concept memory、language-conditioned merging 與 causal concept composition 做更完整調查。即使精確架構組合不同，套到自訂 backbone 本身也不足以證明研究新穎性。

尚待決定：主要目標為 query-conditioned 定位，或 image-only（可含固定共用概念庫）語意分組；實際 checkpoint、文字 encoder、stage、ROI 協定、樣本量、可訓練參數與運算預算。先完成可審閱設計再排實驗。
