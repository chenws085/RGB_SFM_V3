# 2026-10-20 最新實驗與遠端數據補傳

本次檢查 remote140165 的 runs/、data/、plots/、執行程序，以及經 TLS 驗證的 HTTPS inbox weights/、datasets/ 與全部最新 notes/local/。未發現正在執行的新實驗。

最新已知完成實驗仍為 10/15 本機 T1/S0/S1 與兩個歷史 checkpoint 的七倍率比較。後者完成依據為 notes/local/20261015T120324Z_1014_comparison_done.md；原始新評估及 cache 留在 Windows 本機。未見倍率 accuracy：歷史 baseline 44.63%、aligned 55.14%、S0 44.41%、S1 66.76%、T1 54.69%。不同 run/環境、單 seed，仍為探索性比較。

共享服務已搬到 E:\RGB_SFM_artifacts（/mnt/e/RGB_SFM_artifacts），依據 notes/local/20261015T174407Z_storage_migration_done.md；D 槽僅舊備份。遠端使用 https://100.104.52.34:40115，經 127.0.0.1:41081 CONNECT proxy 並保留 TLS 驗證。

依使用者要求，將遠端 10/14 aligned 完整評估、10/15 spherical 評估、暫停 epoch-10 訓練 checkpoint/config/history/log 打包複製到本機收件匣：

`E:\RGB_SFM_artifacts\datasets\remote140165\experiment_artifacts\20261020T051413Z_remote_experiment_data.tar.gz`

183 個來源檔案，封裝 128293182 bytes；SHA256 `43656a25f830f0af563e25291380fa2190426e4d99fb9df4521644a42ddf1f91`。上傳成功後 HEAD 再確認接收端持久化 hash。封裝內 TRANSFER_MANIFEST.json 記錄逐檔來源、大小與 SHA256；本端收據 overnight_logs/artifact_sync_1020/verification.json。未解壓、未啟用 checkpoint、未重啟訓練。epoch-10 run 不是本機完成的 T1；10/14 原分群仍為 Euclidean。
