# 聚類自動產生語意：CLIP／LLaMA 路線可行性

日期：2026-10-15。此為文獻核對與實驗設計，尚未下載／重訓語言模型。使用者希望由模型自動發現／描述聚類，而非人工看採樣圖歸納名稱；此需求取代先前以人工產生候選詞為主要流程的提案。

## 判斷

建議先用**預訓練視覺語言模型，自動建立既有群與文字的對應**，再比較**凍結 MergingViT、學習小型對齊 head**。如果研究目標是讓分組本身更語意化，再做帶文字監督的 grouping／backbone 訓練。首輪不從零訓練 CLIP 或 LLaMA。

這個選擇是根據目前 6,907 張 training、沒有 region-caption pairs、單張 16 GiB GPU 的研究判斷，不是宣稱小資料永遠不能微調。兩個不同目標要分開量測：

| 目標 | 是否必須改變原模型？ | 驗證方式 |
|---|---|---|
| 自動描述既有群包含什麼 | 不必 | 群分數與文字分數在獨立影像上的對應／檢索 |
| 使學到的群更接近語意概念 | 需改變表徵／分組或訓練目標 | 新群與舊群的獨立語意品質、分類與穩定性消融 |
| 讓群名反映原模型分類所用概念 | 命名或投影本身不足 | 原模型 activation intervention 與匹配控制 |

自動命名可減少人工解讀，但語言知識仍來自預訓練模型、語料或圖文監督；不能把外部語言先驗說成完全由無標註 K-means 自發得到英文詞義。

## 各路線適合解什麼

| 方法 | 適合度 | 判斷理由 |
|---|---|---|
| CLIP／SigLIP 從零重訓 | 首輪不採用 | 需有效圖文配對與足夠多樣性；目前 Caltech class labels 不等於部件／紋理 captions |
| 預訓練 CLIP／SigLIP + 自動群描述 | 優先 | 保留原群，建立可量化的文字對應，較易控制計算與解釋範圍 |
| 凍結原 backbone + stage projector | 第二步 | 測原特徵能否讀出語意；投影後更可讀不能直接算成原始群改善 |
| LoRA／局部微調視覺 encoder | 有條件採用 | 先有 region-caption 或可靠 teacher targets，再檢查是否遺忘／退化成類別辨識 |
| 純文字 LLaMA | 不直接適合 | 不能僅憑任意 token／centroid 數字理解視覺；必須先有視覺 encoder、connector 與對齊資料 |
| LLaVA 類視覺 encoder + connector + LLM | 適合生成候選描述 | 可以自動產生開放文字，但流暢描述不保證忠於原群或原模型決策 |
| GroupViT 類文字監督 grouping | 最貼近「分組本身顯現語意」 | 訓練時讓分組參與圖文對齊；是新的方法／模型實驗，不能套名到既有固定 patch-merging |
| SAE／稀疏字典 | 原群混合多概念時的備選 | 可以拆解混合特徵，但稀疏性不保證每個 latent 都有單一可靠語意 |

[CLIP 原論文](https://arxiv.org/abs/2103.00020)使用大規模圖文配對學共同空間；它支持重用預訓練語意空間的思路，沒有證明只用 Caltech 類別名即可得到局部部件語意。[SigLIP 原論文](https://arxiv.org/abs/2303.15343)研究 pairwise sigmoid 圖文損失，提供小 batch 對齊損失候選，但其大型訓練結果不能直接視為本機資料量下的保證。

[LLaVA 原論文](https://arxiv.org/abs/2304.08485)將 vision encoder 與 LLM 透過可訓練連接結合，並使用圖文／指令資料。對目前任務，更合理的是使用預訓練多模態模型作 caption teacher，而非把既有 centroid 任意轉成文字 token 交給純 LLaMA。7B 模型光 FP16 權重約 14 GB（參數數量×2 bytes 的估算），尚不含訓練狀態與 activations；不把全量微調列入本機首輪預算。[QLoRA](https://arxiv.org/abs/2305.14314)可降低微調記憶體，但不會替我們解決視覺對齊與可靠 caption 的資料問題。

[GroupViT](https://arxiv.org/abs/2202.11094)在圖文對比訓練中學階層 grouping，展示沒有 pixel mask 監督也能形成可作 segmentation 的分組。其 grouping、配對資料與訓練目標都與現有 MergingViT + 事後 K-means 不同；若採其方向，需作獨立消融與新 checkpoint。

## 建議的自動流程

### A：不改原模型，先問既有群能否被自動解碼

1. 固定本機 T1／S1 checkpoint、原 spherical codebook 與 source-image split；每個 stage／readout 分開。不能把中心 token、bbox pooled 與文字投影群混為同一概念。
2. 收集 training 全部有效來源的群 affinity（與 centroid 的 cosine 或 soft assignment），不只挑最漂亮的幾張代表圖。以類別均衡或匹配的負樣本補足類別／背景控制。
3. 候選文字由既有公開詞庫、caption corpus 或凍結 VLM 的 training-only captions 自動產生；記錄語料版本、模型、prompt 與篩選規則。去重同義詞，保留 unknown／mixed，測試拿掉 101 類別名後是否仍有屬性／部件訊號。
4. 凍結 CLIP／SigLIP，對每個 training crop、適量 context 與候選文字計算其共同空間相似度。這一步比較 image embedding 與 text embedding，**不是**把 MergingViT centroid 直接與 text vector 算 cosine。
5. 對每個群，使用整批影像的 affinity 向量與每個詞的 image-text score 向量做對應，搭配群外／同類負例，得到群→文字排序。可採 CLIP-Dissect 的 matching 思路；從 neuron activation 改成 cluster affinity 是本研究的延伸，不能說原方法已驗證過這個版本。
6. 由 training 中的 source bootstrap、同義改寫一致性與 permutation null 評估名稱穩定程度；在 calibration data 固定拒答門檻，再凍結名稱去測 validation／外部標註 benchmark。沒有超過 null 或名稱只依賴 context 的群，輸出 unknown／context-dependent。

[CLIP-Dissect](https://arxiv.org/abs/2204.10965)正是自動描述內部 neuron 的先例，可在不由人工選樣命名的情況下使用開放概念匹配。其[官方程式](https://github.com/Trustworthy-ML-Lab/CLIP-dissect)提供自訂 target model、probe dataset 與 concept set 的入口；需要實作本專案的 cluster-affinity adaptor。

### B：原特徵到語言空間，只訓練小 head

凍結原 backbone，逐 stage 訓練線性 head（不足再比較小 MLP），將同 ROI 的 MergingViT readout 對齊凍結視覺語言 teacher。先用 image-embedding distillation，若已有經篩選的 region-caption pairs，再加文字對齊。raw feature 與 projected feature 指標分開報；低維輸入映射到高維不會自動增加原特徵沒有的資訊。

至少比較 raw K-means 自動命名、projection-only、teacher-only，以及打亂 region／teacher 配對的控制。若 projection 只複製 teacher，沒有較好的原模型概念讀出或介入證據，就只宣稱外部語意讀出，不宣稱解開原模型機制。

### C：要讓群真的更語意化，才做重訓

在 classification CE 外加入 region-level 圖文／teacher 對齊，視需要加入 grouping tokens 或可訓練 prototypes；比較：原始 CE、CE+對齊、CE+對齊+grouping。caption 需對應該 ROI，不把整圖的 airplanes label 貼到每個 8×8 token。訓練保留尺度／平移穩定與群多樣性控制，避免將所有局部都壓成物件類別。

如果仍出現一群混合多個概念，再比較稀疏字典。[MSAE 研究](https://arxiv.org/abs/2502.20578)提供 CLIP 稀疏特徵分解的例子，但不能保證特徵一定單義；[後續 CB-SAE 研究](https://openaccess.thecvf.com/content/CVPR2026/papers/Kulkarni_Interpretable_and_Steerable_Concept_Bottleneck_Sparse_Autoencoders_CVPR_2026_paper.pdf)亦指出 interpretability、coverage 與 steerability 需要分別檢查。

## 不以人工歸納作主要流程，仍可獨立驗證

語意名稱自動產生，不要求人工先從九群 gallery 整理詞彙。但不能用同一 CLIP/VLM 生成標籤又用它打分，當成唯一正確性證據。建議使用既有的獨立概念標註資料，例如 [Broden／Network Dissection](https://github.com/CSAILVision/NetDissect)，核對顏色、材質、物件、部件等可對應的概念；需記錄外部資料 domain shift、詞彙 coverage 與預訓練資料重疊未知的限制。

| 量測 | 目的與控制 |
|---|---|
| 自動命名 coverage／abstention | 多少群有超過 null 的可靠名稱，容許未知 |
| 獨立標註下 retrieval P@k／AUPRC | 與 teacher-only、類別名-only、random/shuffle 比較，報 support |
| 同義改寫與 source bootstrap | 名稱是否對抽樣／措辭穩定 |
| crop-only／context-only／兩者 | 是否只是框外物件名稱或背景捷徑 |
| 同 ROI 平移／尺度後詞分數 | 是否穩定且仍能區分真的不同概念 |
| 原模型分類與 activation intervention | 命名的概念是否與原模型決策作用有關 |

少量人工審核可用作額外品質檢查，並非詞義生成主流程，也不必是唯一評估來源。現有 validation 已參與模型選擇，仍稱探索性；若要 final test，另固定未使用資料，不能把看過的資料改名。

## 實作順序與進度

首輪先 A，再 B；C 需先看到 A/B 的具體不足並備齊配對資料。建議先 Stage 2 bbox/中心 token，之後擴至 Stage 1；Stage 0 小 crop 資訊不足，較需分開檢查 context 依賴。先快取 pretrained embeddings，再在 CPU／小 head 上跑比較，降低與模型訓練搶 GPU 的成本。

目前完成：文獻與方法定位、對照組／評估準則、外部監督與原模型解釋的界線。尚未完成：模型選型與實測 VRAM、checkpoint 下載、cluster-affinity adaptor、自動候選詞 pipeline、projection 訓練與語意 benchmark。沒有排程或啟動新的語言 GPU 工作；先前尺度優先的執行決定維持。
