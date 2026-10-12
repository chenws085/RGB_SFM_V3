# 近期研究方向與交接脈絡

更新日期：2026-10-12。研究紀錄涵蓋 2026 年 9 月至 10 月 9 日；GitHub 精選產物整理至 10 月 12 日。

本文件供未來 clone 專案、換機或接續研究對話時使用。依據 `summaries/` 的逐字稿統整、`plan/` 的近期計畫與實驗報告，以及目前程式與 Git 追蹤狀態整理；不是完整歷史對話逐字稿。數值以連結的原始報告與 CSV 為準。以下「建議下一步」是本次整理的執行建議，不代表所有項目已決定或完成。

## 1. 現在要回答的研究問題

**階層式 Vision Transformer 的低階特徵如何組合成高階概念？這些概念及其組合路徑，是否實際影響分類決策？**

現有基礎是自訂 MergingViT、跨影像 token／head 分群、代表 patch、Grad-CAM 回溯及 VLM 描述。研究希望把這些工具連成可重現、可量化、可追蹤的階層解釋，而不只展示代表圖。

暫定主線為 **Hierarchical Causal Concept Graph（階層因果概念圖）**：

1. 從各 stage 的特徵群建立候選概念。
2. 依真實 patch-merging 對應建立 child → parent 組合關係。
3. 記錄支持樣本數、條件機率、PMI／條件熵與穩定性。
4. 介入內部 token，檢查概念及路徑對分類的作用。

目前「因果概念圖」是研究目標；已完成的介入工具只涵蓋單層 token 與同層組合，尚未完成跨層路徑驗證。可參考[方向轉向提案](plan/0923_research_direction_pivot.md)與[方法定位及後續規劃](plan/0928_future_plan.md)。

## 2. 研究方向如何演進

| 時期 | 主要工作與問題 | 留給現在的判斷 |
|---|---|---|
| 9 月初 | 整理階層模型、head／token K-means、Grad-CAM、VLM 與 bottom-up 解釋 | 需要量化、merge lineage 和完整實驗紀錄，人工圖片判讀不足 |
| 9 月 21 日 | Caltech-101 letterbox 與逐 stage 平移一個 token | 用來診斷位置敏感性；不同 stage 的 pixel 位移也不同，不能把差異全歸因於深度 |
| 9 月 23 日 | 提出階層因果概念圖 | 平移與 assignment confidence 降為驗證／輔助方向，主問題轉為概念組合及分類作用 |
| 9 月 28–30 日 | 校正 K-means 方法名稱、文獻定位、spectral 教學視覺化與前景 overlay 規劃 | normalized Euclidean 與 spherical 必須區分；教學 demo 不能當主模型實驗證據 |
| 10 月 5 日 | 球面法／FAISS 對照、50 restarts、20 seeds、silhouette 與代表樣本稽核 | 幾何提升幅度小；發現 annotation 座標問題，物體中心解讀需撤回並重跑 |
| 10 月 9 日 | 實作 activation intervention、同層 A／B／A+B 與隨機對照 | 已有功能及探索性案例，尚缺語意、匹配控制、大樣本與跨層證據 |
| 10 月 12 日 | 推送腳本、報告及精選 plots | 遠端可閱讀研究脈絡與主要產物；不等於所有重現依賴都已備齊 |

較早的脈絡見[正確逐字稿統整與路線圖](summaries/08_correct_transcript_synthesis_and_roadmap.md)。

## 3. 模型與分群的共同語言

模型為[自訂 MergingViT](models/MergingViT.py)：patch embedding → Transformer stages → 相鄰 patch merging → global average pooling → 分類 head。沒有 CLS token，也不是 Swin 或 PVT 的完整實作。研究中的 token 框表示網格／合併來源範圍；self-attention 會混合其他位置，框不能當作完整感受野。

原有分析主要是 **position-wise、dataset-wise clustering**：固定 stage 與位置，收集多張 training images 的 token 再建立 codebook；head 模式再固定 head。概念至少需用 `(run, stage, position, cluster)` 識別，head 模式另加 head。同一編號跨位置、stage、seed 或 run 沒有天然共同語意。

另需區分舊 Caltech「依 annotation 選取 token」的比較實驗，與 10 月 9 日「固定位置 24／25」的介入實驗。兩者的取樣協定與 codebook 不同，不能共用 cluster 編號或結論。

原主流程是 **L2-normalized features + Euclidean K-means**：樣本正規化，但 centroid 是算術平均，更新後沒有重新投影至單位球面。不能因樣本間平方距離滿足 `2 − 2 cosine` 就稱整個演算法等同 spherical K-means。真正的 spherical 方法每輪重新正規化中心；FAISS native 與 shared-init 也需分開記錄。

方法與文獻入口：[方法校正](plan/0928_future_plan.md)、[ViT 特徵分群文獻整理](plan/0928_vit_feature_kmeans_literature_survey.md)。

## 4. 已有結果及其有效範圍

### 4.1 分群品質：有小幅幾何改善，尚未證明語意改善

10 月 5 日的 20 seeds 評估固定 K=31、seeds 42–61、每次 50 次初始化，使用同一批 1,770 張 validation 影像；training 為 6,907 張。下表為 held-out cosine silhouette 的 seed 平均：

| Stage | 原 sklearn 方法 | 手寫 spherical | FAISS native spherical |
|---|---:|---:|---:|
| 0 | 0.19836 | 0.20292 | 0.20288 |
| 1 | 0.12803 | 0.13195 | 0.13348 |
| 2 | 0.09084 | 0.09905 | 0.09925 |

球面方法提升約 0.0039–0.0084；負 silhouette 比例降低約 3.93–6.58 個百分點。報告在固定資料、seed 層級的六組配對比較使用 Holm 校正，結果支持穩定的小幅改善。這不等於 cluster 具有人類語意，也不證明所有任務或 checkpoint 都更好。

增加 `n_init` 是增加同一 seed 內的最佳解搜尋；增加 seeds 才提供初始化的重複評估。同一批影像跨 seeds 重用，不能當成 35,400 張獨立影像。現有 bootstrap 不涵蓋重新抽 training data、重新訓練模型等不確定性。

球面法在平移穩定性上的排名隨 stage 改變，不能由 silhouette 較高就宣稱整體較好，也沒有充分依據全面替換原主流程。詳見[20 seeds 報告](plan/1005_kmeans_20_seeds_experiment_results.md)、[50 restarts 報告](plan/1005_kmeans_50_restarts_experiment_results.md)與[三方法完整說明](plan/1005_three_method_kmeans_silhouette_report.md)。

### 4.2 座標稽核：舊物體中心實驗必須重新取樣

Caltech-101 的 `obj_contour` 是相對 bounding box 的座標；舊流程漏加 `box_coord` offset。修正後，在 Stage 0／1／2 分別有約 91.13%／84.58%／64.01% 的影像改落到不同 token。

因此，舊 cache 上的 K-means、silhouette、ARI 與平移數值仍描述原取樣特徵，但**不能作為物體中心／物體部件分群的證據**。目前只完成稽核與代表圖對照，尚未完成修正座標後的完整重跑。

若繼續物體中心路線，必須修正座標、重新抽 training／validation／translated features、重新 fitting，再重算指標；新舊 cache 和 manifest 要有不同座標版本。輪廓點平均亦不是 mask 面積質心。詳見[座標稽核](plan/1005_caltech101_annotation_coordinate_audit.md)。

### 4.3 概念介入：機制可用，研究結論仍屬探索性

[介入工具](scripts/concept_intervention.py)在指定 stage 最後 block 後、merge 前，用暫時 hook 替換 token，接著以原網路重新計算下游輸出。支援單概念、同層聯合概念及相同 stage／token 數的隨機對照。

現有試跑用 120 張 training images 建立固定位置 24／25 的 K=4 codebook，16 張 validation images 評估，random repeats=2。預設以各位置 **raw training activation mean** 替換；另支援 zero。分群使用 normalized features，不把 unit centroid 直接插入原網路。

`logit_drop = 原目標 logit − 介入後目標 logit`；正值表示下降。`effect_gap = 概念 drop − 隨機 drop`；gap 正值不保證概念 drop 本身為正。Prediction change 也不等於 accuracy drop。

單概念支持數僅 1–8，聯合組合僅 2 張；未做顯著性宣稱。A、B、A+B 必須在聯合存在的同一批影像配對比較，不能相加各自不同支持集合的平均。Random repeats 要先在影像內彙整，推論以 source image 為單位。

已完成空介入、identity、與獨立下游 forward 比對、hook 清除等 4 項驗證；這些驗證支持程式行為，不代表概念語意或因果圖已成立。固定位置介入未用 annotation center，避開上述舊座標取樣問題，但仍可能落在背景或 padding。

現有對照只匹配 stage／token 數，沒有匹配位置、前景或 feature norm，且替換可能偏離真實 activation 分布。Validation 曾用於 checkpoint 選擇，不能稱為 untouched final test。詳見[工具規格](plan/1009_concept_intervention_tool.md)與[圖解實驗報告](plan/1009_concept_intervention_experiment_report.md)。

## 5. 建議接續工作的順序

| 順序 | 工作 | 可交付的驗證結果 |
|---|---|---|
| 1 | 補齊 fresh clone 的腳本與環境依賴；確認 checkpoint、split、前處理及座標版本 | 能從 manifest 重現既有功能驗證，明確列出外部資料需求 |
| 2 | 依研究問題選擇固定位置或修正物體中心協定，重抽相應特徵 | 新 cache／codebook／manifest，不混用舊座標與舊群編號 |
| 3 | 擴大單層介入評估，加入同位置不同概念及真實 activation donor 控制 | 類別內效果、支持數、影像層級 CI、mean／zero／donor 敏感度 |
| 4 | 用真實 merge mapping 建立 Stage 0→1→2 概念轉移與組合統計 | 支持度、條件機率、PMI／條件熵、代表樣本及 seed 對齊 |
| 5 | 對少量高支持路徑比較 child、parent 與 path 介入，追蹤下游概念變化 | 相對匹配隨機節點／邊的可重複 effect gap，檢查路徑是否與決策相關 |
| 6 | 執行方法消融 | global vs per-position、token vs Key／Value／head、K／seed／checkpoint 穩定性 |

既有提案的最低可行目標是先從 airplanes 與少量高支持路徑開始，再擴到更多類別。正式評估應預先固定抽樣、對照位置與統計方案，另保留未參與選模及探索的 final test。

跨層連線本身只代表模型結構及統計關係；在沒有匹配介入與下游追蹤證據前，不將它命名為已驗證的因果路徑。

## 6. 保留但暫不作主線的方向

- **平移／controlled-position**：用來拆解位置與內容影響。完整影像平移同時改變背景與 patch 邊界；同 patch 跨位置與完整平移要分開解讀。
- **Assignment confidence／soft assignment**：低 margin 顯示多個候選群，作為概念不確定性的輔助。
- **Foreground overlay**：[9 月 28 日規劃](plan/0928_foreground_token_mask_overlay_plan.md)先檢查 CAM-derived mask 可視化，不把 CAM 自動當成真實前景標註，也不由計畫文件推定實作已完成。
- **Spectral／PCA preprocessing**：作為分群消融。[視覺化 demo](experiments/kmeans_spectral_visualizer/README.md)使用合成資料，不能支撐 Caltech 概念品質結論。
- **Grad-CAM／VLM**：用於檢視與描述候選，不作為唯一語意或 faithfulness 證據。
- **可訓練 prototypes／counterfactual editor／failure atlas**：後續備選，不先要求重訓整個模型。

## 7. 新環境拉下專案後先看什麼

建議閱讀順序：本文件 → [專案 README](Readme.md) → [方向轉向](plan/0923_research_direction_pivot.md) → [方法定位](plan/0928_future_plan.md) → [座標稽核](plan/1005_caltech101_annotation_coordinate_audit.md) → [介入報告](plan/1009_concept_intervention_experiment_report.md)。要核對分群數字，再讀[三方法報告](plan/1005_three_method_kmeans_silhouette_report.md)與[20 seeds 報告](plan/1005_kmeans_20_seeds_experiment_results.md)。

### GitHub 已有與仍需另外取得的內容

- 已追蹤主要分析腳本、研究報告、合成視覺化 demo，以及約 19.2 MiB 的精選 plots：silhouette 圖／PDF、比較圖、介入圖表、CSV 與 manifests。範圍見[plots 說明](plots/README.md)。
- `data/`、`runs/`、模型 `.pth`、本機虛擬環境、特徵 cache、多數群代表影像及大型逐輪紀錄未納入 Git。只閱讀報告不需取得這些檔案；重跑模型實驗需要另備。
- 常用 checkpoint 為 `runs/train/mergingvit_caltech101_letterbox160/MergingViT_best.pth`。同一路徑不能保證同一權重；以各 `run_manifest.json` 的 SHA256 和模型參數確認。
- 環境基準見[requirements.txt](requirements.txt)。本機指令中的 `.venv/bin/python` 是當時環境路徑，換機後可改為自己的 Python；FAISS 對照另需檢查所需套件。

### 已確認的重現缺口（2026-10-12）

根目錄 `.gitignore` 的 `test_*.py` 規則也排除研究用腳本。目前 Git 未追蹤 `scripts/test_caltech_translation_kmeans.py`，而介入及部分分群評估會 import 它；故 fresh clone 不能直接執行所有報告指令，甚至介入驗證也會受影響。

`scripts/test_negative_silhouette_significance.py` 同樣未追蹤；報告中的連結尚未能在遠端直接打開。重現前需要取得並納入這些研究依賴，或把共享函式移到受追蹤模組。本次交接整理未修改這些腳本或忽略規則。

不能只因精選圖表已上傳，就假定所有圖片連結或完整 pipeline 都可從乾淨 clone 重現。多數 cluster gallery 刻意未上傳；缺圖時先看 plots 範圍說明，再決定取得本機產物或重跑。

## 8. 下一次研究對話可直接使用的摘要

> 我們在自訂階層式 MergingViT 上研究跨影像、位置條件式概念發現，主目標是依真實 patch merging 建立概念組合路徑，再以 activation intervention 驗證分類作用。原方法是 normalized-feature Euclidean K-means，不能稱為嚴格 spherical。20 seeds 的球面比較有小幅 cosine silhouette 改善，但沿用漏加 annotation bounding-box offset 的舊取樣特徵，尚未證明物體部件語意；修正物體中心實驗未重跑。10 月 9 日已完成固定位置 token 單概念／同層組合介入及功能驗證，只有 120 張 fitting／16 張 validation 的探索性結果，尚無跨層因果路徑。下一步先補重現依賴、固定取樣協定與匹配控制，再擴大介入及建立跨層圖。詳細來源、產物範圍與 fresh clone 缺口見本文件。
