# 1–7 px 非對齊平移初步實驗

使用者授權：先直接測平移對 accuracy／K-means 的影響，再訓練 1–7 px 增強模型，比較改善並寫 MD 報告。

## 固定設定

- 全部 101 類參與從頭訓練；架構、seed 42、split、100 epochs 上限、patience 20、batch 32、AdamW／scheduler 與原實驗一致。以原位全類別 validation accuracy 選模。
- 新增強每次讀取均勻選原位或上下左右 × 1–7 px，共 29 種。保留既有 flip／jitter／erasing。只新增小位移，不混入 8／16／32 px 增強。
- 160-in-224；完整來源圖整數平移，無平移插值／來源內容裁切。分析修正 annotation offset 後的輪廓點平均中心 token。
- 固定評估 airplanes 160、chandelier 22、brain 20 張。原位＋四方向 × 1–7 px；各模型完全相同來源。每模型 202×29×4 stages=23,432 筆 token observations。
- 依序評估既有無平移模型、既有 aligned 8／16／32 px 模型；從頭訓練 1–7 px 模型後作相同評估。
- 三模型統一使用 **FAISS spherical K-means**：normalized training features、每輪 centroid 正規化、K=31、niter=100、nredo=50、seed 42；只以未平移全類別 training features fitting，所有評估條件固定同一 codebook。每個 stage 共享跨位置 codebook。
- 指標：accuracy、prediction consistency、cluster agreement、token cosine、spherical margin（top-1 減 top-2 cosine）、群數／熵。與舊 sklearn Euclidean margin 不直接比較。
- 不對齊網格時，中心 token 像素涵蓋會變，不能聲稱是完全相同 crop 的純位置效應。這輪訓練／測試位移皆 1–7，結果屬 seen-range robustness，非未見位移泛化。
- 單一 training seed，已有 validation 參與選模；三類探索性比較，不能推論全類別或 untouched final test。

## 執行與產物

執行：`bash scripts/run_non_aligned_translation_1015.sh`。執行中請勿重複啟動。自動偵測 CUDA；不可用時以 CPU 執行，不更改模型規格或訓練預算。

2026-10-15 啟動時 `nvidia-smi` 無法初始化、PyTorch CUDA unavailable，使用 CPU。batch 32 前向＋反向初測約 0.98 秒，完整訓練預期需數小時，以實際 epoch 時間為準。

- 狀態：`overnight_logs/non_aligned_translation_1015/status.txt`
- 分階段 log：同目錄 `baseline.log`、`aligned.log`、`train.log`、`augmented.log`
- 新訓練：`runs/train/mergingvit_caltech101_non_aligned_1015/`
- 自動報告：`plots/kmeans/caltech_non_aligned_1015/REPORT.md`；每完成一個模型即更新，明列未完成項目。
- 每模型保留 cache identity、centroids、逐筆 CSV、summary、曲線及 manifest。既有 unshifted feature cache 只有在 checkpoint／source IDs／extractor／座標程式 hash 一致才重用。

## 已完成的流程驗證

- 全部 29 種 training 平移與 evaluation safe-letterbox 逐像素相同。
- 小型 FAISS 測試驗證 centroid 單位 norm、內積排名與最高 cosine 指派一致。
- 評估程式另驗證 0 px identity、source IDs 一致、中心位移量、train／validation 分離與逐筆唯一鍵。

流程驗證不等於模型改善的研究結果；實際數值以報告及 CSV 為準。


## 使用者要求暫停（2026-10-15）

已停止本輪 launcher、training 主程序與 dataloader workers，不會自動繼續執行。兩個既有模型的 1–7 px spherical 評估均完成；新模型完成到 epoch 10，停止時尚未完成下一個 epoch。

- 最佳 checkpoint：epoch 10；原位全類別 validation accuracy 48.25%、飛機原位 accuracy 98.13%。不是最終模型結果，亦不是平移後 accuracy。
- checkpoint、optimizer、scheduler 留在本機 `runs/train/mergingvit_caltech101_non_aligned_1015/MergingViT_best.pth`，SHA256 與參數見[暫停狀態](../plots/kmeans/caltech_non_aligned_1015/pause_state.json)。模型權重與特徵 cache 不納入 Git。
- 已公開訓練 config／history 快照；完整 raw training log 留本機。已完成模型的逐筆 metrics、summary、曲線、manifests 可直接閱讀。
- 待完成：新模型完整訓練、其 1–7 px 評估、最終改善比較。沒有尺度或語言對齊新實驗已執行的證據。

### 之後如何接續

本次暫停以終止程序並保留最佳 checkpoint 實施，不是將完整 CPU／RNG／dataloader 狀態凍結。現有 trainer **沒有 `--resume`**，直接重跑相同 output-dir 會拒絕覆寫。

1. 若要延續現有最佳 checkpoint，先實作並驗證明確的 resume：載入 model／optimizer／scheduler、接續 epoch、best metric 與 history。原 checkpoint 沒有完整 RNG／sampler 狀態，因此只能說從最佳 epoch 延續，不能聲稱逐步等價於未中斷訓練。
2. 若接受重新從頭訓練，使用新的 output-dir，例如：

   ```bash
   .venv/bin/python -u scripts/train_caltech_mergingvit.py      --translation-shifts 1 2 3 4 5 6 7 --content-size 160      --output-dir runs/train/mergingvit_caltech101_non_aligned_restart      --device cuda:0
   ```

   GPU 可用後才指定 CUDA，否則使用 `--device cpu`。新 checkpoint 路徑需同步更新 evaluation 的 `CHECKPOINTS['augment_1_7']`；不要混用舊 cache，另備新輸出版本後再執行。
3. 不要直接重跑一鍵 pipeline：它不是 resume 工具；既有 cache identity／已存在 training directory 會防止不相容重用與覆寫。fresh clone 取得相同外部資料／checkpoint 後可按腳本從頭建立產物。

### 環境與可重現範圍

可用 `pip install -r requirements-faiss.txt` 安裝原 requirements 與 `faiss-cpu==1.12.0`（本輪 CPU spherical clustering 版本）。`scripts/test_caltech_translation_kmeans.py` 是共享模組，雖名為 test，仍是執行依賴，已納入 Git。

資料 `data/`、原模型與增強模型 `.pth`、NPZ features／centers、虛擬環境與 overnight logs 仍需外部／本機提供。GitHub 上的精選報告與 CSV 可獨立閱讀，但不代表 checkpoint 可從 repo 自動取得。報告手工編輯部分（解讀、簡化表格、暫停狀態）重跑自動報告時可能被覆寫，重跑前需保留版本並重新合併。
