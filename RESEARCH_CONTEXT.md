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

因此，舊 cache 上的 K-means、silhouette、ARI 與平移數值仍描述原取樣特徵，但**不能作為物體中心／物體部件分群的證據**。2026-10-12 已完成原 Euclidean 方法 seed 42、n_init=50 的修正中心靜態重測（6,907 training／1,770 validation），及 160 張 airplanes 的四方向 0／4／8／16／32 px 平移，詳見[新實驗紀錄](plan/1012_caltech_corrected_center_static_translation.md)。尚未重跑修正座標後的三方法／20 seeds 比較。

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

根目錄 `.gitignore` 的 `test_*.py` 規則也排除研究用腳本。2026-10-12 修正中心重跑時已為 `scripts/test_caltech_translation_kmeans.py` 加入明確例外，讓介入及部分分群評估依賴的共享模組可納入版本控制；舊座標流程本身保持不變。

`scripts/test_negative_silhouette_significance.py` 仍未追蹤；報告中的連結尚未能在遠端直接打開。重現該分析前需要另取得此依賴。

不能只因精選圖表已上傳，就假定所有圖片連結或完整 pipeline 都可從乾淨 clone 重現。多數 cluster gallery 刻意未上傳；缺圖時先看 plots 範圍說明，再決定取得本機產物或重跑。

## 8. 下一次研究對話可直接使用的摘要

> 我們在自訂階層式 MergingViT 上研究跨影像、位置條件式概念發現，主目標是依真實 patch merging 建立概念組合路徑，再以 activation intervention 驗證分類作用。原方法是 normalized-feature Euclidean K-means，不能稱為嚴格 spherical。20 seeds 的球面比較有小幅 cosine silhouette 改善，但沿用漏加 annotation bounding-box offset 的舊取樣特徵，尚未證明物體部件語意。10 月 12 日已完成修正物體中心的原 Euclidean seed-42 靜態重測與 airplanes 的 0／4／8／16／32 px 四方向平移，保留逐筆 patch 相位、邊界與 padding；三方法／20 seeds 尚待重跑。10 月 9 日已完成固定位置 token 單概念／同層組合介入及功能驗證，只有 120 張 fitting／16 張 validation 的探索性結果，尚無跨層因果路徑。下一步固定取樣協定與匹配控制，再擴大介入及建立跨層圖。詳細來源、產物範圍與 fresh clone 缺口見本文件。


## 9. 使用者最新方法決定（2026-10-15）

後續分群選用真正的 spherical K-means 實作，例如 FAISS `spherical=True`，而非僅將特徵預先 L2 正規化後套用 sklearn Euclidean KMeans。此決定尚未套用／重跑；10/14 平移增強實驗及換群案例仍屬原 sklearn 歐式方法，不能改稱球面法。詳見[實驗筆記](plan/1014_aligned_translation_augmentation.md)。


## 10. 後續三方向規劃（2026-10-15）

已整理[非對齊平移、尺度穩定與自然語言概念規劃](plan/1015_translation_scale_language_roadmap.md)。先做 FAISS spherical 基準與非對齊整數平移，再獨立測尺度／對應 ROI；不足時逐步加 consistency loss 或架構消融。語言線從群命名、獨立概念標註與檢索評估進到 stage projection，最後接原模型概念介入。此為規劃，尚未啟動新實驗；不要求所有固定 token 在尺度改變後硬性同群。


## 11. 1–7 px 非對齊平移實驗（2026-10-15）

依使用者要求已啟動[初步實驗](plan/1015_non_aligned_translation_experiment.md)：先對既有 baseline／aligned augmentation 模型測四方向 1–7 px，再從頭訓練全類別 1–7 px 平移增強模型；三模型統一用 FAISS spherical K-means，評估固定飛機／吊燈／腦部影像。啟動時 CUDA 無法使用，以 CPU 執行，預計數小時；實際狀態見 `overnight_logs/non_aligned_translation_1015/status.txt`，階段與結果見 `plots/kmeans/caltech_non_aligned_1015/REPORT.md`。不要把已啟動或中途報告當成已完成改善驗證。


## 12. 最新狀態：使用者要求暫停並上傳 GitHub（2026-10-15）

10/14 aligned augmentation 已完成：seed 42，best epoch 84，飛機 32 px accuracy 98.13%、prediction consistency 99.06%；此輪分群仍是 sklearn normalized-feature Euclidean。詳細比較、margin、深層非對齊與 66 張換群／不換群案例見[報告](plots/kmeans/caltech_aligned_translation_1014/REPORT.md)。

10/15 的 1–7 px 實驗已停止，不會自動繼續。既有 baseline 與 aligned augmentation 兩模型已完成相同 FAISS spherical 評估；新模型 training 完成至 epoch 10，最佳 checkpoint 亦 epoch 10，全類別原位 validation accuracy 48.25%，不是最終改善證據。Checkpoint／optimizer／scheduler 留在本機，只有 config／history／SHA256 快照公開。Trainer 尚無 resume，不能直接重跑當成續訓。接續細節、外部資料需求與未完成工作見[實施／暫停紀錄](plan/1015_non_aligned_translation_experiment.md)。

下一階段的非對齊平移、尺度與自然語言概念研究方向見[roadmap](plan/1015_translation_scale_language_roadmap.md)。未收到後續指示前，不重啟已暫停的實驗。

## 13. 本機重訓、尺度結果與傳檔服務（2026-10-15）

使用者後續已授權下載官方資料並在本機從頭重訓，再排程尺度實驗。兩條 pipeline 均已完成：T1 best epoch 78／原位 validation 68.36%；S0 best epoch 63／63.28%；S1 best epoch 84／69.49%。尺度未見倍率 0.7／0.9／1.1 的平均 accuracy 為 S0 44.41%、S1 66.76%，Stage 2 bbox agreement 為 57.12%／71.32%。單一 seed 與使用過的 validation，不等同最終泛化證據。詳見[本機完成紀錄](plan/1015_local_training_results.md)。本節的新 run 不應與第 12 節原機暫停的 epoch-10 run 混用。

已建立 HDD `D:\RGB_SFM_artifacts` 接收服務，LAN URL `https://192.168.100.233:40115`，TLS＋bearer，私密 credential 留在 Git 外。使用者指定優先讓遠端 `140.115.51.165:40008` root agent 上傳權重／資料集，未主動把本機權重送過去。遠端首批同步已查驗完成，詳見第 14 節與[操作教學與狀態](plan/1015_artifact_transfer_service.md)。

語言需求更新為自動發現／命名聚類，避免人工從採樣圖歸納作主流程。已完成 [CLIP／LLaMA／GroupViT 可行性評估](plan/1015_language_alignment_feasibility.md)，建議先預訓練視覺語言模型的自動群描述，再凍結 backbone 學小 head；真正讓 grouping 學語意則需獨立圖文監督訓練。尚未啟動語言訓練。

## 14. 遠端接收區與雙 agent 筆記（2026-10-15）

**持續記住的本機資料入口：`/mnt/d/RGB_SFM_artifacts/`（D 槽 HDD）。之後可能持續有遠端同步的新權重／資料；找檔時除了 repo 的 runs/ 與 data/，先檢查這裡。** `weights/remote140165/` 已有歷史 baseline 與 aligned_translation_1014 的 checkpoint/config/history；`datasets/remote140165/` 已有 Caltech 官方 archives，兩份權重與 archive hash 已確認。先前「尚無遠端資料入庫」狀態已更新。對照本機平移／尺度的新評估見[比較紀錄](plan/1015_remote_1014_comparison.md)。

雙 agent 文件通訊在 `/mnt/d/RGB_SFM_artifacts/notes/`；本機寫 notes/local/，遠端寫 notes/remote140165/，以 UTC 唯一檔名逐則新增，不互相覆寫。接續任務時先看最新筆記，重要結論再整理進 plan/。Git 入口：[agent_notes/README.md](agent_notes/README.md)。遠端沿用現有 HTTPS client 的 notes/ namespace，無即時通知。

## 15. 完成版權重接收與尺度案例內部分析（2026-10-20）

共享收件匣已搬至 E:\RGB_SFM_artifacts（本機 WSL /mnt/e/RGB_SFM_artifacts），D 槽是備份。遠端已接收 weights/local/ 的完成版 S0／S1／T1（best epochs 63／84／78），12 檔 SHA256 與既有評估 checkpoint hashes 通過，存於 overnight_logs/scale_internal_1020/inbox/weights/local/。完成固定三案例×七尺度的逐 token spherical 圖、真實 merging 邊及 Stage 2 raw-mean 介入 pilot；training 512 影像、每圖每層 8 token 的新全局字典，不能混作原 ROI 或位置條件字典。九張圖、252 筆指標、63 筆介入，identity 與原尺度自對應通過。少量案例排序不代表全體；尚無跨層因果路徑驗證。詳見 [執行紀錄](plan/1020_scale_internal_analysis.md) 與 [結果](plots/kmeans/caltech_scale_1015/internal_1020/REPORT.md)。

## 16. 對齊／非對齊平移統一補評（2026-10-20）

依使用者授權，在remote140165補齊baseline／aligned 8/16/32／完成版T1三模型的202張三類validation × 原位及四方向1–8/16/32 px × 四stage，共99,384筆token配對。全部重新用同一spherical FAISS規格與固定training codebooks，未新增訓練、未覆寫10/14 Euclidean結果。非對齊1–7 accuracy為87.80%／87.78%／90.77%；對齊初始patch 8/16/32平均為69.39%／87.58%／74.05%。T1在32 px僅53.34%，aligned為87.00%；小位移穩健性不等於大位移泛化。兩組各50% accuracy為78.60%／87.68%／82.41%。完整逐位移／類別／stage、影像層級bootstrap、分群集中度與限制見[獨立混合分析](plots/kmeans/caltech_mixed_translation_1020/REPORT.md)與[執行紀錄](plan/1020_mixed_translation_analysis.md)。T1下載到遠端runs/train並核對hash；本機專案尚未同步這批新產物。

## 17. 101類分層採樣平移圖文（2026-10-20）

依使用者要求新增三類評估群一致率圖，並在101類每類5張、共505張validation固定採樣上完成baseline／aligned／T1的相同平移grid評估（248460筆token配對），沿用第16節完全相同的frozen training spherical codebooks。原位accuracy44.95/49.90/56.24%，1–7px41.56/48.39/54.62%，8/16/32px30.12/46.65/40.07%；兩組各50%為35.84/47.52/47.34%。這是類別平衡採樣，不與原1770張分布加權accuracy混用；T1與aligned的混合點估計相近，不能沿用三類子集的排名差距。Stage3小位移的T1群一致率高於aligned、cosine卻較低，需逐指標解讀。完整分類／群一致率／cosine PNG、MD/TXT報告與來源／逐類CSV見[全類別圖文報告](plots/kmeans/caltech_allclass_translation_1020/REPORT.md)及[執行紀錄](plan/1020_allclass_translation_analysis.md)。沒有新訓練或本機專案產物同步。

## 18. 語意資料集與 early fusion 規劃（2026-10-21）

補充[語言融合層對齊討論與文獻](plan/1021_language_fusion_related_work.md)：FIBER、Bridger、SAM-PTx、Fusioner 等已有 backbone fusion／凍結 encoder adapter 的先例，LSeg／SAIL 提供對齊基線，GroupViT／SegCLIP 提供 grouping 對照。不能將加入融合層本身稱新穎貢獻。擬測共用概念庫＋gated stage adapter，並加入等容量 image-only adapter、文字複製捷徑與影像依賴控制；重點轉為 merging 前後的概念定位、變換穩定及分類作用。僅調查與筆記，未啟動實驗。

已記錄[資料集與融合規劃](plan/1021_language_datasets_early_fusion.md)：先用 Broden 驗證既有群與 Qwen-VL 事後描述，再用 PACO 檢查部件／屬性組合，Visual Genome 提供 ROI–文字配對以建立 projection 基線。後續比較文字條件 FiLM、stage cross-attention、ViLT 類 joint tokens 與文字條件 merging；分清推論需文字查詢的模型與 image-only 語意群，GroupViT 類文字監督 grouping 不直接稱為 early fusion。現有 Qwen 看代表 patch 圖片，未融入原 MergingViT forward。此次僅文件規劃，未下載資料、啟動訓練或同步本機專案。

## 2026-10-31 物件獨立縮放／平移訓練開發

已接通 contour cutout 的固定合成背景 paired dataset、原／內容加權 merging 的相同雙 view 訓練入口及實體座標 region readout。CPU 幾何／padding／梯度與真實四來源過擬合通過。GPU 1 上的 101 類 1616 train／404 validation、20 epochs B/D pilot 位於 `runs/train/object_scale_1031_20261031T093309Z/`；本輪從頭訓練、單 seed，尚不可宣稱尺度不變或泛化改善。狀態及後續結果見 [訓練紀錄](plan/1031_object_scale_training.md)。這是物件 cutout 實驗，與舊整張來源圖縮放分開。

10/31 首輪物件 cutout B/D pilot 已完成：長邊80＋(7,-5) 準確率 B 6.19%、D 3.47%，目前兩者皆低，純內容 pooling 未勝出。已新增保留原 merging 的零初始化 gated residual，啟動 `runs/train/object_scale_1031_hybrid_pilot/`；初始 logits parity 與 gate gradient 通過，泛化結果待驗證。首輪報告在 `runs/train/object_scale_1031_20261031T093309Z/REPORT.md`，詳見上列訓練紀錄。

## 2026-11-03 CLIP bbox直接監督完成

100epochs原模型CE＋四層ROI到frozen CLIP ViT-B/32的0.2 cosine蒸餾已完成。原尺度accuracy A79.92%／CLIP79.36%；未見80/112平均A80.42%／CLIP79.84%，差-0.58pp，配對來源bootstrap95%CI[-1.61,+0.40]，單seed探索validation、不含seed變異。未觀察到分類增益。最差14條件A52.04%／CLIP53.51%。高training cosine約0.798尚非獨立語意或可解釋性證據。詳見 plan/1101_direct_clip_supervision.md 及 runs/train/clip_supervision_1101/full/REPORT.md；未啟動新訓練。

## 2026-11-09 CE／CLIP同物件跨層補評

凍結既有A與all-layer CLIP checkpoint，兩組各層同容量train-only rank32 PCA＋ridge10映射至共同CLIP教師，1768 validation原位96 ROI評估。直接跨層cosine六組四升；去training預測均值後六組皆升，最大2↔4 +0.0311，但同物件相對同類異圖差距未全面升，尚非逐token／部件語意證據。raw CKA六組五升。單seed探索validation、來源bootstrap不含訓練seed變異。詳見[比較報告](plan/1109_clip_cross_stage_comparison.md)與`plots/reports/clip_cross_stage_20261109T053032Z/REPORT.md`。

## 2026-11-09 投影後spherical跨尺度／跨層補評

沿用FAISS spherical K31／100iter／50restarts／seed42與舊raw字典，1768 validation、原位96對80/112純縮放。CLIP raw→前次matched probe投影：同層79.87%→60.52%，跨層37.64%→32.20%；相對CE raw78.38%／35.92%亦低。跨層字典只用training作Hungarian群號對應，六配對僅2↔4提高3.87pp；不是原生逐token／部件語意。舊96→80平移兩條件群指派逐筆重現；來源配對CI不含seed／probe重fit變異。詳見[分群比較](plan/1109_clip_spherical_scale_comparison.md)。先前cosine提升未轉為整體同群率改善。

## 2026-11-09 使用者研究偏好：CLIP優先深層，主看跨層＋跨尺度

記住使用者決定：後續CLIP直接監督優先Stage3/4，主指標為跨層＋跨尺度同群率；這是工作假設，不改寫既有深度消融尚未支持普遍優越的結論。目前CE vs CE+CLIP表是去背cutout學生、四層監督模型，教師看原始bbox自然背景。training建字典／群號配對，validation評估且曾選模。同層固定網格跨尺度有相對覆蓋範圍混雜；跨層亦不自動解決。後續考慮stride/物件長邊比例匹配（例Stage3-64對Stage4-128），不可把名義patch範圍當global-attention有效RF。尚未啟動新訓練／新配對評估。詳見[偏好與協定](plan/1109_deep_clip_priority_and_scale_protocol.md)。

## 2026-11-09 三組跨層名義支援尺度匹配完成

重用既有CE／全層CLIP0.2／深層Stage3/4 CLIP0.2的100epoch checkpoint，training96固定spherical K31字典與Hungarian群對應，1768validation原始整物件ROI評估。相鄰層64→128及96→192對應stride/長邊比例，置中平均38.59/39.23/36.74%；網格位置校正後37.75/39.08/36.90%。全層相對CE平均CI跨0，深層限定未勝出；不能由深層表示改善推出僅深層監督更好。保留使用者深層優先偏好為假設。既有deep每層loss係數是all兩倍。這是名義支援尺度與近似相位控制，非實測有效RF或逐token匹配；非相鄰層極小物件有資訊退化。無新訓練，詳見[三組報告](plan/1109_clip_support_matched_three_models.md)。

## 2026-11-09 合併架構文獻定位

查核Swin v1官方BasicLayer：stage所有blocks後做PatchMerging，與本repo相同邊界及2×2 concat→LN→Linear形式；attention不同（Swin窗／移窗，本repo全局）。相關階層式全局／縮減attention架構包括PiT、HVT、PVT、MViT，不能把局部merging來源框當完整有效RF。詳細原始來源與差異見[架構相關工作](plan/1109_swin_mergingvit_hierarchical_related_work.md)。無新模型修改／實驗。

## 2026-11-09 窗口attention原版消融完成

純CE窗口7（實際7/7/7/4，前兩stage局部、後兩覆蓋全層），原merging／相同初始化／同資料與100epochs，window best epoch96。全局→窗口：跨層96↔80/112 35.92→37.20%，3↔4 38.73→44.98%；支援比例匹配平均38.59→35.60%，位置校正37.75→35.99%，未見尺度分類80.42→81.04%的來源CI跨0，無全面改善結論。詳見[窗口對照](plan/1109_window_attention_comparison.md)。另將先前投影cosine與四層共用spherical字典（無Hungarian）的結果明確分開記錄：[投影直接比較](plan/1109_projected_direct_cosine_shared_kmeans.md)。

## 2026-11-09 使用者最新研究決定與原圖確認

CLIP監督確立為跨層分類／分群及物件縮放不變性研究方向；主要基準為原始merge_ViT，匹配限於同範圍內容。後續目標是跨層改善且同層指標不下降，尚未達成。全局／窗口MHA選擇未定；全局計算量大不保證準確率高，窗口收斂優勢待驗證。確認最近100epoch global/window純CE比較皆為輪廓去背物件＋合成純色背景＋縮放平移，不是完整自然原圖；CLIP學生亦同，教師只看原圖bbox crop。未啟動新實驗。詳見[最新決定與輸入稽核](plan/1109_research_decisions_clip_scale_and_mha.md)。

## 2026-11-09 完整原圖MHA對照啟動

使用者指定改用完整原始圖片。新窗口CE從頭訓練100epochs，run runs/train/window_original_20261109T080440Z/window；全局CE沿用同設定已完成原圖100epoch checkpoint（best97、56.56%），split/model/seed/預算/loss/來源註記與原圖loader驗證通過。完整自然背景、只224等比resize/pad、無去背裁切增強，無CLIP。完成後自動跑原圖分類與training224固定spherical字典的validation分析，尺度評估縮放整圖與背景，不混作物件獨立縮放。尚無新窗口最終結果，詳見[原圖MHA對照](plan/1109_original_images_mha_comparison.md)。

## 2026-11-09 完整自然原圖MHA對照完成

window100epochs完成best84，global原圖CE100epoch best97。原圖224分類global56.56/window55.49%，差−1.07pp CI[-2.83,+0.62]；同層整圖224→160/192同群率49.92→57.13%；跨層＋跨尺度24.41→25.23%（差CI跨0）；相鄰層名義支援比例112→224 19.78→28.15%，主要前兩個配對改善，3↔4未升。保留自然背景、只resize/pad、無去背裁切增強；縮放整圖與背景，不能混作物件獨立縮放。原始bbox ROI各層字典＋training配群，非逐token或投影共用字典；MHA選擇仍未定。詳見[完整原圖MHA結果](plan/1109_original_images_mha_comparison.md)。

## 2026-11-09 0921式全局／窗口平移補評完成

既有完整原圖CE100epoch兩checkpoint，完整自然圖長邊160置中224，無裁切四方向1–8/16/32px，6902training中心token固定FAISS spherical K31字典，1768validation；修正absolute bbox中心，非0921舊錯誤座標／Euclidean數值重現。one-token(Stage0/1/2=8/16/32)群一致率global/window90.14/85.42、83.14/79.45、74.84/71.30%，window三組CI均較低；32px分類一致率23.90/27.59%，window+3.69pp CI[+2.40,+4.99]，accuracy差CI跨0。飛機32px一致率66.41/76.88%，不代表全體。兩者仍有局部穩定但分類漂移與垂直較敏感現象。160原位全類別accuracy僅20.25/19.23%，224訓練→160尺度差異限制attention因果選型。建議global保留主基準、window對照，不全面替換；同160訓練／多seed尚未啟動。詳見[平移比較](plan/1109_attention_translation_comparison.md)與plots/reports/attention_translation_20261109/REPORT.md。

## 2026-11-09 沿用歷史160 global，匹配window訓練

使用者指出既有letterbox160 global；已驗SHA256 3cf8ab1056dfb025aa63c349e4d06e3ef49e519418fceb7798f49af1be39daca，現行loader重現best66的1770validation62.09%、飛機98.75%。global無須重訓。已準備window使用同6907/1770 split、160-in224、原flip/jitter/erase、batch32、AdamW wd.01、CE smoothing.1、100epoch cosine與patience20從頭訓練；與前輪224原圖模型不同協定。原初始化逐張量一致、eval像素與歷史loader一致已檢查。run runs/train/window_letterbox160_20261109，訓練後自動補四方向1–8/16/32px spherical平移評估，尚無最終window結果。見[沿用global與匹配訓練](plan/1109_letterbox160_global_window.md)。前輪224→160診斷不作本次最終選型依據。


## 2026-11-09 相同letterbox160全局／窗口平移比較完成

window匹配歷史global重訓至82epochs/best62；1770原位accuracy global62.09/window62.99%，+0.90pp CI跨0。訓練augmentation、split6907/1770、optimizer/loss、100epoch預算與patience20匹配；原初始化逐張量一致、checkpoint原位accuracy重現。one-token群一致率Stage0/1/2(8/16/32px)94.18/89.82/83.60→89.52/85.75/79.80%，window三組CI均較低。32pxaccuracy27.32→24.00%，差−3.32pp CI[-4.48,-2.26]；分類一致率30.45→26.54%，差−3.91pp CI[-5.20,-2.61]。飛機32px分類一致率60.94/59.84%，對應Stage2局部cosine.9654/.9533，兩者仍有局部穩定但分類漂移現象；上下均比左右敏感。建議global保留主基準，此固定窗7/7/7/4版本未改善平移，不推及所有窗口機制。單seed／validation曾選模，非因果機制定位。完整[匹配設定與結果](plan/1109_letterbox160_global_window.md)，圖文在plots/reports/attention_translation_letterbox160_20261109/REPORT.md。前輪224→160分布差異診斷不可混入本輪。

## 2026-11-09 Window/local attention五組探索

使用者要求多測window/local attention變體或相關超參數。以歷史letterbox160 global與已完成window7作固定基準，新增window4、window14、window7固定網格offset3、僅Stage0 window7、Stage0/1重疊sliding7。相同6907/1770、seed42、歷史augmentation/loss/optimizer、100epoch上限與patience20；無額外參數/relative bias，初始化參數逐張量一致。每stage1block，offset3不是完整Swin交替移窗。dense masked參考、padding/offset/sliding有限梯度及完整model backward、舊版相容檢查通過。GPU1依序訓練後自動四方向1–8/16/32px spherical測試與彙整，狀態見runs/train/local_attention_sweep_20261109T180400Z/status.json。尚無新變體最終結果；單seed多變體探索，不能以原位最高分直接選型。詳見[變體規劃](plan/1109_local_attention_variant_sweep.md)。


## 2026-11-10 局部attention五組完成判讀

全部5/5完成，1770validation checks與結果hash通過。global/window7/window4/window14/offset3/stage0-only/sliding7的原位accuracy62.09/62.99/63.73/62.94/63.90/63.56/63.16%；32pxaccuracy27.32/24.00/20.69/27.43/25.16/29.96/24.07%。stage0-only是分類候選，32px+2.64pp CI[+1.55,+3.73]，但one-token群一致87.05低於global89.20%。window14小位移60.66高於global58.71%，32px與群一致差CI跨0，描述性接近不等於等效。sliding7群一致89.68點估計最高但對global差CI跨0，32px仍較差；offset3原位與小位移最高但32px下降。保留global主基準、stage0-only分類候選、window14折衷對照；無全面優勝。單seed多變體validation探索，未多重比較校正；額外seed尚未啟動。結果runs/train/local_attention_sweep_20261109T180400Z/REPORT.md與comparison.json，完整[完成判讀](plan/1109_local_attention_variant_sweep.md)。

## 2026-11-10 使用者收斂變量，明確暫緩掃lr

主比較改為global、window4/7、sliding_window4/7五欄；局部方法統一Stage0/1、深層global、保留absolute position、offset0。Absolute position/stage/offset暫緩；低通濾波只記未來方案。使用者追加「掃lr暫緩」，因此不啟動1e-4/1e-3或其他lr；本輪只固定3e-4，重用已驗global/window7/sliding7。只補Stage0/1 window4（舊window4含Stage2不合規）與sliding4；sliding4每軸[-1,0,1,2]，偶數鄰域方向偏置明確記錄。歷史sliding7與window7深層global逐值相容檢查通過。新主表runs/train/attention_size_sliding_20261110_lr3e4/REPORT.md；[固定lr規劃](plan/1110_attention_size_sliding_fixed_lr.md)。

## 2026-11-10 五方法固定lr主表完成與轉置

lr3e-4之Stage0/1 window4與sliding4已完成，五方法資料齊全；lr掃描仍暫緩。使用者要求方法作列、指標作欄，並確認最後一欄8px一致性。主表欄位為原位accuracy、1–7px_acc、8px_acc、1–7px分類一致率、8px分類一致率；報告定義一致率為與原位預測相同，不是KMeans群一致。路徑runs/train/attention_size_sliding_20261110_lr3e4/REPORT.md；其他詳細指標保留results.json與各方法完整評估，不與主表混放。

## 2026-11-14 Sliding size細掃與RF順序

使用者要求sliding window_size找甜蜜點，納入比4小、比7大；固定尺寸清單1/2/3/4/5/6/7/9/11/15，lr3e-4、Stage0/1 sliding、深層global、原位置編碼與merging不變。重用global/sliding4/7核對hash結果，其他8組未開始：目前NVML初始化失敗，torch.cuda不可用、device_count0且cuda allocation失敗。準備好入口scripts/sliding_size_sweep.py及runs/train/sliding_size_20261114/status.json（waiting_for_gpu），無背景GPU輪詢；GPU恢復後同命令可接續。各k明確定義offsets、偶數半格偏置、邊界mask與名義stride跨度。使用者指定做完size實驗才分析每個設計的RF變化；後續區分名義窗口、理論attention/merging連接與實測ERF，對照平移指標，不以k×stride直接稱有效RF。lr/position/stage/offset掃描與低通仍暫緩。詳見[細掃與RF後續規劃](plan/1114_sliding_window_size_sweet_spot.md)。

## 2026-11-14 GPU修復後啟動sliding size實驗

使用者確認GPU修復並要求開始。GPU1 RTX3060實際CUDA配置與matmul通過，來源程式及重用checkpoint hashes驗證通過；啟動scripts/run_sliding_size_1114.sh，原runs/train/sliding_size_20261114依序補1/2/3/5/6/9/11/15，重用4/7及global。lr3e-4、Stage0/1 sliding/深層global、absolute position與原merging固定。日誌overnight_logs/sliding_size_1114/queue.log及每size.log，status.json記錄實際進度。尚無新增尺寸最終結果，RF分析仍依使用者要求在實驗完成後才做。
本協作規則亦已寫入根目錄 `AGENTS.md`，要求之後的 agent 接續任務前先讀最新共享消息。遠端模型比較已完成：aligned 10/14 未見倍率 accuracy 55.14%、Stage 2 bbox agreement 65.44%；本機 S1 為 66.76%／71.32%。T1 小位移增強在尺度 accuracy 54.69%，未優於 aligned 10/14，但在 1–7 px 平移較佳。詳見[完整對照](plots/kmeans/caltech_remote_1014_comparison/REPORT.md)。
