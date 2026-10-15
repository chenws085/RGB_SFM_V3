# 1–7 px 非對齊平移：本機從頭重訓

2026-10-15 使用者指定從網路取得資料並重新訓練。這是新的 seed-42 run，並非 epoch-10 checkpoint 續訓。

- 資料來源：https://data.caltech.edu/records/mzrjq-6wc02/files/caltech-101.zip
- 官方 ZIP MD5：`3138e1922a9193bfa496528edbbc45d0`，下載後已驗證。解開原始影像與 Annotations.tar 至 `data/caltech101/`。
- 已核對 training 6,907／validation 1,770 的逐筆 indices 與公開暫停 config 完全相同。
- 設定保持原協定：101 類、160-in-224、batch 32、seed 42、最多 100 epochs、patience 20、1–7 px 四方向加原位共 29 種，保留 flip／jitter／erasing。以原位 validation accuracy 選 best checkpoint。
- 使用 RTX 4060 Ti。WSL 預設 CUDA loader 不可用；專案 `.venv/wsl-cuda/libcuda.so.1` 連至 WSL 掛載的實際 NVIDIA driver library，僅對本流程設定 LD_LIBRARY_PATH。已驗證 CUDA tensor 運算。
- 修正 dataloader 與 models 的不相關 eager imports，讓獨立 Caltech 腳本不依賴本機應用程式 config.py。模型與資料前處理本身未改動。

執行：`bash scripts/run_non_aligned_translation_restart_1015.sh`。已有執行程序時不要重複啟動。

產物：

- 訓練：`runs/train/mergingvit_caltech101_non_aligned_restart_1015/`
- 狀態及 log：`overnight_logs/non_aligned_translation_restart_1015/`
- 訓練完成後自動評估：`plots/kmeans/caltech_non_aligned_restart_1015/REPORT.md`

評估沿用修正 annotation 中心與 FAISS spherical K-means：K=31、niter=100、nredo=50、seed 42；相同三類共 202 張、原位及四方向 1–7 px。比較報告會複製 Git 內已完成 baseline／aligned 模型的 CSV、圖與 manifest 作歷史對照；不代表本機重新訓練或重新評估這兩個對照模型。新模型只使用新的 cache／輸出目錄，不覆寫暫停報告。

單一 seed，validation 曾用於選模，且 training／evaluation 位移範圍相同。結果只支持本次協定下的探索性比較，不當作未見位移泛化或跨硬體逐步重現。執行是否完成請看 status.txt 與各階段 log；本文件不是完成證據。
