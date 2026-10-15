# 今日另外兩個方向：尺度穩定性與自然語言概念

日期：2026-10-15。此文件是待執行規劃，不代表實驗已啟動。現行 1–7 px 平移重訓與自動評估照常執行；本規劃不插入其 GPU 工作。

## 一、尺度：分類改善是否伴隨對應內容的概念穩定？

### 首輪問題與對照

先分開觀察分類、中心 token 與對應 ROI。中心 token 在縮放後涵蓋的物件比例不同，因此換群不直接判為失敗。

| 組別 | 訓練 | 用途 |
|---|---|---|
| S0 | 固定長邊 160，無平移／尺度增強 | 純尺度實驗的主要配對對照 |
| S1 | 每次等機率抽長邊 96／128／160／192 | 測單靠尺度增強的效果 |
| T1 | 本機正在訓練的 1–7 px 增強模型 | 額外參考：平移增強能否轉移至尺度；不替代 S0 |
| S2，條件式追加 | S1 加 prediction consistency | S1 的分類仍對尺度敏感時再做 |
| S3，條件式追加 | S2 加對應 ROI feature consistency | 分類穩定而 ROI 仍不穩時再做 |

本機缺少歷史 baseline checkpoint。S0 需本機重新訓練，不能拿 Git 上的平移 CSV 當尺度基準。S0／S1 同 seed 42、split、模型初始化、batch 32、AdamW／scheduler、100 epochs 上限與 patience 20；flip／jitter／erasing 均保留。訓練中不同增強會消耗不同 RNG，不聲稱 batch 級完全配對。首輪依原位長邊 160 的全類別 validation accuracy 選模，兩組用相同規則。

### 變換與對應

- 固定 224 畫布、影像置中，只縮放完整來源影像，背景一起變化。每個條件從原圖直接 resize，保持長寬比；倍率相對長邊 160 定義。
- 訓練倍率：0.6／0.8／1.0／1.2；評估全部七種倍率：0.6／0.7／0.8／0.9／1.0／1.1／1.2。其中 0.7／0.9／1.1 為未見離散倍率，屬範圍內測試。
- 首輪不混平移。長邊 192 時每側只有約 16 px 邊界，後續聯合平移需重訂共同安全範圍。
- 以修正 annotation 座標得到物件 bbox；主要 ROI 為整個物件 bbox，輔助 ROI 為 bbox 中央 50% 寬高區域。兩者隨已知 resize／置中變換映射，不稱為部件 ground truth。
- 每 stage 同時保留中心 token、bbox 與中心 ROI 的網格交疊面積加權平均；pool 後 L2 normalize。固定各 stage 網格的定義，Stage 3 需按模型實際邊界處理核對覆蓋；ROI 對有效網格覆蓋不足時記錄原因及共同有效集合。
- 保存原尺寸、實際 resized 尺寸、offset、ROI、有效覆蓋率、插值／antialias、annotation 版本。用可視化核對狹長影像與奇數尺寸 rounding。

### 分群與指標

每個模型／stage／readout 各建獨立 FAISS spherical codebook：K=31、niter=100、nredo=50、seed 42。僅用 training images 的長邊 160 features fitting；同一 codebook 固定用於七種倍率。中心-token 與 ROI codebook 不混用；不同模型的 cluster ID 不直接對齊比較。

| 面向 | 指標 |
|---|---|
| 分類，101 類全 validation | 每倍率 accuracy、macro class accuracy、相對 1.0 的 prediction consistency |
| 對應內容 | 各 stage／readout 相對 1.0 的 cosine、cluster agreement、spherical margin |
| 是否失去區辨 | 使用群數、最大群占比、群熵；與同類不同影像 ROI cosine 對照 |
| 可解讀案例 | 同圖七倍率，並排 token／ROI 範圍、prediction、cluster 與 margin |

先用 airplanes／chandelier／brain 共 202 張固定 validation 圖做流程驗證；正式首輪分類涵蓋全部 1,770 張，ROI 主表也擴至全 validation。若部分 ROI 無效，另報排除比例，所有模型用共同有效集合。多倍率先在 source image 內彙整，再做影像層級 bootstrap；此 CI 不包含重新訓練的 seed 不確定性。

### 決策與交付

預先固定主比較：S1−S0 的三個未見倍率平均 accuracy，以及 Stage 2 bbox-ROI agreement；其餘 stage／readout 是診斷。原位 accuracy 必須一併呈現，不用一致率掩蓋原圖退步。只有 agreement、同圖／異圖區辨與群多樣性一起檢查後，才判讀概念穩定性。首輪 seed 42 只做探索；有值得保留的效果再做三個配對 training seeds。

預定產物：`plots/kmeans/caltech_scale_1015/` 下的 transform manifest、逐筆 metrics、分類／token／ROI 三組曲線、案例圖與 REPORT.md。先完成 S0／S1，不直接擴張至一致性損失或改架構。

## 二、語言：群能否被命名、被查詢，再連回分類作用？

### L0：先建立可被推翻的群命名

第一輪凍結本機訓練完成的 T1 checkpoint，使用既有 spherical 群與修正中心 token；每個概念以 checkpoint／stage／取樣協定／cluster 識別。未來 ROI 群另建字典，不沿用群編號。

先做 Stage 0／1／2，每 stage 依 training support 選前三大群，共九群。每群保留最多 12 張不同 training source，包含最接近 centroid、中等 margin 與低 margin 樣本；不足即全取。另抽相鄰群作 hard negatives。此 pilot 是高支持群探索，不代表所有群。

每筆同時呈現 token crop、2 倍邊長的 context crop、原圖位置框；crop 超出畫布需記錄有效區域。小 crop 放大只供閱讀，不能當新增視覺資訊。候選詞先控制在約 20 個，涵蓋顏色／邊緣／紋理／形狀／部件；保留 unknown、mixed、不可見。分 stage 詞彙只是候選，不先假設深層必然學到部件。

CLIP 可提供 crop／context 與文字的外部相似度，VLM 可協助提出候選；它們的輸出不直接當人工真值。MergingViT centroid 不與 text embedding 直接算 cosine。若 crop-only 名稱不穩、加 context 才出現物件名，須標示上下文依賴。

### 獨立概念標註與最小評估

- 候選詞僅從 training 範例制定。第一批人工標註建議 120 張不同 source：60 張 training 作詞義校準與後續 projection pilot；60 張 validation 作探索性評估。三個既有 pilot 類各約 20 張／split，選圖固定 seed，不能按文字模型分數挑圖。
- 每張標固定 Stage 1／2 的目標 ROI，共約 240 筆；採多標籤，有／無／不確定／不可見。train／validation 不共享 source，所有變換與 crop 跟隨來源。
- 標註介面隱藏群名稱與 CLIP 分數，先給清楚定義與正反例。抽 20% 交由第二位標註者獨立判讀；若只有一位，明列限制，不把自動生成標籤代替第二人。
- 人工標註尚未完成前，只交候選字典與可審閱 gallery，不宣稱語意準確率。既有 validation 已用於選模，這批仍屬探索性評估，不能改稱 untouched final test。

首輪報告群內概念純度／support，以及 text→region 的 Precision@5、Precision@10；不足 k 筆不硬報。對有足夠正負樣本的概念報 AUPRC 與 positive prevalence，稀少概念只列 support。用隨機排序、類別名-only、CLIP-only 作比較，並測同義改寫與 hard negatives，避免只學類別名稱。

### L1／L2：有獨立語意證據後才追加

| 階段 | 方法與對照 | 能回答的問題 |
|---|---|---|
| L1 | 凍結 backbone，逐 stage 線性 probe／projection；train-only 擬合，與 shuffle-label／nearest-centroid／CLIP-only 比較 | 原特徵能否讀出已標註概念？ |
| 變換評估 | 同一 ROI 在平移／尺度後的概念分數、排名及檢索品質，保留不同概念 negatives | 穩定是否伴隨可區辨性？ |
| L2 | 同 stage／位置、相近 norm／前景的真實 activation donor；同概念／異概念／隨機 donor 對照 | 該概念方向的介入是否影響原分類輸出？ |

60 張 training source 不足以支撐高容量 projection 或稀少概念，先看標註 support 再決定擴充。若用 CLIP target 訓練 projection，必須保留人工概念評估，不能只以對 CLIP 的擬合程度自我驗證。

介入前先固定支持集合與原圖的目標類別，報 target logit、分類 margin、accuracy 及下游概念分數；random repeats 在來源圖內平均。同群、相近 norm 仍不能排除全部混淆，首輪只稱指定替換下的機制證據。

預定產物：`plots/concepts/language_1015/` 的 concept_dictionary.csv、annotation_template.csv、gallery、retrieval metrics 與 REPORT.md。字典需含詞義、同義詞、證據來源、crop/context 差異、support、unknown 與版本。

## 今天的執行順序與完成界線

1. 讓現行 T1 訓練及平移評估完成。
2. 先準備尺度 transform／ROI 座標核對，以及語言候選與空白標註表；兩者均可先做 CPU 工作。
3. GPU 空出後先做 T1 尺度 pilot，再依序訓練 S0／S1 與共用尺度評估；語言線接 T1 training 群建立九群 gallery。
4. 今日最小交付目標為尺度 S0／S1 首輪比較、語言九群候選與可標註素材。語言獨立評估取決於人工標註完成；不預先承諾 L1／L2 或三 seeds 今天完成。

原規劃本身不會啟動工作；其後使用者已指定僅排程尺度方向，詳見下列執行紀錄。

## 尺度排程實作（2026-10-15）

使用者要求在本機平移實驗全部跑完後自動啟動尺度方向。入口為 `scripts/run_caltech_scale_after_translation_1015.sh`；背景排程等待既有平移 launcher 的 flock 釋放，並確認 status.txt 為 complete。若平移失敗，尺度標為 blocked 並退出，不自行跳過失敗階段。

執行順序為 S0 訓練 → S1 訓練 → 三模型 S0／S1／最終 T1 的尺度評估。每組訓練均為 seed 42、最多 100 epochs、patience 20，依原位 validation accuracy 選模。本機 GPU 逐項執行。語言方向未排程。為讓尺度增強只作用於 training，trainer 新增 `--scale-content-sizes`；未提供時維持原訓練路徑，現行平移程序不受影響。

模型輸出為 `runs/train/mergingvit_caltech101_scale_S0_1015/` 與 `runs/train/mergingvit_caltech101_scale_S1_1015/`。狀態及分階段 log 位於 `overnight_logs/caltech_scale_1015/`。排程 PID 保存在 `launcher.pid`。預定報告為 `plots/kmeans/caltech_scale_1015/REPORT.md`。

已通過 CPU 小型完整流程驗證：七種倍率、四 stages × 三 readouts、spherical fitting、逐筆 CSV／transform manifest、曲線／案例與比較報告；同一 checkpoint 作兩組時主要配對差值與 bootstrap CI 皆為零。另驗證 training-only 尺度抽樣、validation 固定 160、直接從原圖 resize 的像素一致，以及 Stage 3 末端網格 ROI 權重。

實際模型的 Stage 3 會將 7×7 補齊後合併成 4×4，沒有丟棄右／下邊界。評估末端 cell 與畫布交疊時裁至 224；其 feature 仍含模型自身的 merging padding，不宣稱 ROI 可消除 attention／padding 上下文。bbox 使用 `box_coord` 的零起始 pixel-edge 區間；中心沿用修正輪廓點平均，兩軸採實際 rounded resize 比率。與舊中心取樣是獨立座標版本，舊 codebook 不重用。
