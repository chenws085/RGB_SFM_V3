# 2026-10-20 對齊／非對齊平移補評與混合分析

使用者授權補齊既有三模型的缺漏交叉評估並寫獨立報告；沒有新訓練。執行位置為 remote140165 `/home/ref/RGB_SFM_V3`，不是 Windows/WSL 本機專案。完成版 T1 從目前 E 槽 inbox 的 `weights/local/mergingvit_caltech101_non_aligned_restart_1015/` 下載到遠端 `runs/train/`，SHA256 `88bba79f9f6357c58e592272657b505def602acb10a98914902e2eeab09bc889`，不是 epoch-10 暫停模型。

三模型（歷史 baseline、aligned 8/16/32、T1 1–7）統一重新推論：固定 202 張三類 validation，原位及四方向 × 1–8/16/32 px，四 stage；每模型 33,128 筆，總計 99,384 筆 token 配對。Codebooks 全部使用 spherical FAISS K31/niter100/nredo50/seed42，由未平移全類別 training fitting，不沿用歷史 Euclidean 群指派。實際使用空閒 GPU cuda:1（RTX 3060）。

三類合併 accuracy：

| 模型 | 非對齊1–7平均 | 對齊初始patch 8/16/32平均 | 兩組各50% |
|---|---:|---:|---:|
| Baseline | 87.80% | 69.39% | 78.60% |
| Aligned augmentation | 87.78% | 87.58% | 87.68% |
| T1 | 90.77% | 74.05% | 82.41% |

T1 在 8 px 為90.10%，16 px降至78.71%，32 px為53.34%；aligned在32 px為87.00%。小位移改善不能推論可取代大位移增強。T1−baseline accuracy：非對齊+2.97 pp，影像層級 paired bootstrap 95% CI [+0.72,+5.34] pp；對齊組+4.66 pp，CI [+2.76,+6.56] pp。T1−aligned在對齊組為−13.53 pp，CI [−16.17,−10.93] pp。

已驗證資料影像／annotation fingerprint、split/source IDs、0 px identity、配對唯一性／完整筆數、中心位移、centroid norm、artifact hashes。與歷史1–7 px accuracy／prediction consistency／群一致率完全相同，cosine/margin差異小於1e-7；歷史對齊飛機分類亦一致。單seed，validation參與選模，alignment與位移大小混雜，非嚴格平移不變性或全類別泛化證據。

獨立報告：[REPORT.md](../plots/kmeans/caltech_mixed_translation_1020/REPORT.md)。同目錄有完整summary、分方向表、每影像配對指標、bootstrap差值、verification及圖。各模型raw paired CSV、feature cache、codebook留遠端；compact報告與數據允許納入Git，沒有覆寫歷史產物。

重現：`MIXED_DEVICE=cuda:1 bash scripts/run_mixed_translation_1020.sh`，再執行 `.venv/bin/python scripts/analyze_mixed_translation.py`。執行log與狀態在 `overnight_logs/mixed_translation_1020/`；同步下載與shared note receipts在其inbox。這些遠端檔案尚未複製或解壓到本機專案。
