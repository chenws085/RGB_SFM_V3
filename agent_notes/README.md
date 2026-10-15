# 雙 Agent 共用筆記區

實際共享資料夾：`/mnt/d/RGB_SFM_artifacts/notes/`（Windows：`D:\RGB_SFM_artifacts\notes`）。本機 agent 直接讀寫；遠端 root agent 透過既有 HTTPS inbox 的 `notes/` namespace 上傳與下載。這裡的 README 是 Git 上的入口，討論內容以 HDD 的實際檔案為準。

## 約定

- 本機發文放 `notes/local/`，遠端放 `notes/remote140165/`。
- 檔名採 UTC 時間戳記加主題，例如 `20261015T120000Z_scale_comparison.md`。同秒多則可加隨機尾碼。
- 每則新建檔案，不覆寫另一個 agent 的檔案；更正／回覆另開檔，註明 `reply_to` 路徑。server 亦拒絕以不同內容覆寫既有路徑。
- 開頭列出 author、UTC、task、status（info／question／working／done／blocked），內文列依據、檔案位置、下一步；回覆時引用原訊息。
- 不把 token、私鑰、密碼寫入筆記。討論可保留在 HDD；重要結論另整理進 repo 的 `plan/` 與 `RESEARCH_CONTEXT.md`。
- 筆記是非同步交接，沒有即時通知；每次接續任務時先看最新筆記。共同任務先寫認領訊息，避免兩邊同時重跑同一實驗。

## 遠端使用

沿用原本的 `client.py`、`token` 與 `server.crt`，URL 為 `https://192.168.100.233:40115`：

```bash
cd /root/.config/rgb-sfm-transfer
python3 client.py --url https://192.168.100.233:40115 --token-file token --cert server.crt list
python3 client.py --url https://192.168.100.233:40115 --token-file token --cert server.crt upload /tmp/my_note.md notes/remote140165/20261015T120000Z_my_note.md
python3 client.py --url https://192.168.100.233:40115 --token-file token --cert server.crt download notes/README.md /tmp/shared_notes_readme.md
```

讀取本機 agent 訊息時，從 list 找 `notes/local/` 的檔名，逐筆 download 到未存在的本機檔案；下載 client 預設不覆寫目的檔。只需增補筆記，不用同步整個 HDD，也不必在 Git 兩邊反覆 merge。

相關：[資料傳輸操作](../plan/1015_artifact_transfer_service.md)、[10/14 遠端模型比較進度](../plan/1015_remote_1014_comparison.md)。
