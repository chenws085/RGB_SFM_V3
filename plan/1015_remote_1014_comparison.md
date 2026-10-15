# 遠端 10/14 權重接收與比較

更新：2026-10-15。使用者要求拿遠端同步來的 10/14 模型與剛完成的本機平移／尺度實驗比較；不重新訓練歷史模型。

## HDD 接收目錄是後續資料來源

後續找 checkpoint、config、history 或資料集時，**除了 repo 的 runs/、data/，必須先檢查 `/mnt/d/RGB_SFM_artifacts/`**；這裡是遠端 agent 可持續追加資料的 HDD 接收區，不是一次性測試目錄。不要只因 Git 沒有權重就推定檔案不存在。

目前收到：

- `weights/remote140165/mergingvit_caltech101_letterbox160/`：best checkpoint、config、history。
- `weights/remote140165/mergingvit_caltech101_aligned_translation_1014/`：best checkpoint、config、history。
- `datasets/remote140165/caltech101_official_archives/`：101_ObjectCategories.tar.gz、Annotations.tar、show_annotation.m。
- `notes/`：雙 agent 共用筆記，[操作約定](../agent_notes/README.md)。

## 已完成查驗

| 權重 | best epoch | 原位 validation accuracy | SHA256 |
|---|---:|---:|---|
| 遠端 baseline | 66 | 62.09% | `3cf8ab1056dfb025aa63c349e4d06e3ef49e519418fceb7798f49af1be39daca` |
| 遠端 aligned 10/14 | 84 | 65.71% | `c7fe1d1125494c7cc7459cf1324419673d94977d2f141a3d88faa63372239a63` |

兩份權重皆與 Git 中歷史 1–7 px spherical 評估的 checkpoint hash 完全相符；相同 seed-42 train／validation source 集合與本機一致。舊 baseline config 沒有保存逐筆 indices，該組 split 的證據來自綁定相同 checkpoint 的歷史評估 manifest，不宣稱 config 自己就能證明。

遠端兩份資料 archive 的 MD5 分別為 `b224c7392d521a49829488ab0f1120d9` 與 `6f83eeb1f24d99cab4eb377263132c91`，符合官方值。新資料留在 HDD，不覆寫目前 data/。

## 比較協定與執行

入口：`scripts/compare_remote_1014.py --device cuda:0`，需使用現有 WSL CUDA library 設定。

- 尺度：對兩個遠端 checkpoint 新跑七倍率、全 validation 1,770 張、四 stages × center／bbox／central ROI。沿用原尺度 evaluator、實際 rounded 座標映射與 train-only、160 px fitting 的 FAISS spherical K=31／niter=100／nredo=50／seed42；codebook 各模型各 stage/readout 獨立。
- 本機 S0／S1／T1 重用已完成尺度結果前，檢查同 split、evaluator hash 與 fitting 規格。
- 平移：由相同權重 hash 驗證，可重用已完成 1–7 px spherical 評估，與本機 T1 比較。不是把原 10/14 sklearn Euclidean 分群當作 spherical 結果，也不是新的獨立重複。
- 額外比較皆屬探索性。遠端／本機不同 training run／環境，主要單變因尺度對照仍是本機 S0／S1。

狀態／產物：`plots/kmeans/caltech_remote_1014_comparison/status.json`、`intake_manifest.json`、`REPORT.md`；log 為 `overnight_logs/remote_1014_comparison/evaluation.log`。已於 2026-10-15 12:02 UTC 完成，status 為 complete。

## 完成結果

| 模型 | 原位 accuracy | 未見倍率 accuracy | Stage 2 bbox agreement |
|---|---:|---:|---:|
| 遠端 baseline | 62.09% | 44.63% | 57.66% |
| 遠端 aligned 10/14 | 65.71% | 55.14% | 65.44% |
| 本機 S0 | 63.28% | 44.41% | 57.12% |
| 本機 S1 尺度增強 | 69.49% | 66.76% | 71.32% |
| 本機 T1 小位移增強 | 68.36% | 54.69% | 68.08% |

S1 相對 aligned 10/14 的未見倍率 accuracy +11.62 pp、ROI agreement +5.88 pp。T1 的未見倍率 accuracy 比 aligned 10/14 低 0.45 pp，配對影像 CI 跨 0，這輪不能說小位移增強優於 aligned 模型的尺度泛化。另在三類 1–7 px 平移，T1 accuracy 90.77%、prediction consistency 97.35%，高於 aligned 10/14 的 87.78%／93.32%。

值得注意：aligned 10/14 的 Stage 2 bbox cosine 為 0.8783，高於 S1 的 0.8633，但 accuracy 與 hard agreement 都較低；不能以單一 cosine 排名代替分類或概念品質結論。完整表與探索性配對 bootstrap 見[比較報告](../plots/kmeans/caltech_remote_1014_comparison/REPORT.md)。
