# 1014 全類別平移增強，僅比較飛機

狀態：已完成訓練、飛機評估與產圖。增強模型最佳 epoch 84；完整結果見 [MD 報告](../plots/kmeans/caltech_aligned_translation_1014/REPORT.md)。後續 1–7 px 實驗另見 [10/15 紀錄](1015_non_aligned_translation_experiment.md)，目前已依使用者要求暫停。

- 依使用者最終設定：160-in-224；64 px 暫緩。
- 所有 training 類別增強：原位與上下左右 × 8/16/32 px，共 13 種等機率條件。每次讀取樣本抽一次；不增加每 epoch 樣本數。保留完整影像，不插值平移。平移後沿用既有 flip、color jitter、random erasing。
- 從頭重訓，不載入舊權重。沿用原模型架構、seed 42、train/validation split、batch size 32、AdamW lr=3e-4、100 epochs 上限、patience=20、以原位全類別 validation accuracy 選模。
- 對照使用既有 `runs/train/mergingvit_caltech101_letterbox160/MergingViT_best.pth`，不更動原產物。
- 只對固定 160 張 validation airplanes 做原位及四方向 8/16/32 px 評估。Stage 0/1/2/3 網格為 8/16/32/64 px；本輪所有位移對齊初始 patch，但不一定對齊深層 merging。
- 用修正 annotation offset 的輪廓點平均定位中心 token。K=31、n_init=50，分別以各模型未平移的全類別 training activations 擬合，群編號不跨模型對應。
- 沿用原本按 stage／方向的 cluster agreement、matched-token cosine、prediction consistency 圖，加上 accuracy 與兩模型四方向平均比較圖、平移範例、逐筆 CSV、manifest 和自動報告。
- Validation 曾參與選模；單一訓練 seed，只作探索性比較。

啟動：`bash scripts/run_aligned_translation_1014.sh`（已啟動時不要重複執行）。使用實體 GPU 1，既有模型不覆寫。訓練成功後自動進行評估與產圖。

輸出：`plots/kmeans/caltech_aligned_translation_1014/REPORT.md`、`augmentation_comparison.png`、兩組 `*_translation_stability.png`。


## 2026-10-15 後續分群實作決定

使用者指定：之後分群要選用真正符合 spherical K-means 的實作，例如 FAISS 的 `spherical=True`；樣本先 L2 正規化，centroid 每輪更新後也須正規化，不能把「僅預先正規化特徵＋sklearn Euclidean KMeans」當作等價球面法。

本次 10/14 平移增強評估與換群案例仍使用原本的 normalized-feature sklearn Euclidean K-means（K=31、n_init=50、seed=42）。此項為後續方法選擇，尚未修改程式或重跑；既有報告數值仍屬原方法。新實驗需記錄實作、正規化與初始化設定，並建立獨立 codebook／產物版本。
