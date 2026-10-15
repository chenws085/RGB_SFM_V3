# Local research handoff

## 雙 Agent 文件協作（2026-10-15 啟用）

- 兩個 agent 共用 HDD 筆記區 `/mnt/d/RGB_SFM_artifacts/notes/`，不是 Git
  中 `agent_notes/` 的副本；後者存放操作教學。
- 每次開始或接續任務時，先讀 `notes/README.md` 與兩邊最新消息。
  本機寫 `notes/local/`，遠端 root agent 寫 `notes/remote140165/`。
- 使用 UTC 時間戳記與主題建立唯一 Markdown 檔；回覆新建檔並列出
  `reply_to`，不要覆寫／刪除對方消息。開始共享任務先寫 working／認領消息，
  完成後寫 done、結果路徑與待辦。
- 遠端沿用既有 HTTPS client／憑證，透過
  `https://192.168.100.233:40115` 的 `notes/` namespace 上下載消息；
  認證資訊不放在 Git 或筆記。這是非同步文件協作，沒有即時通知。
- 具體指令與格式見 `agent_notes/README.md`。重要結果另整理進 `plan/`
  與 `RESEARCH_CONTEXT.md`，不要只留在聊天或未讀的單則消息。

## 資料位置與研究交接

- Before concluding that checkpoints or datasets are unavailable, inspect both
  this repo's `runs/` and `data/` and the persistent HDD inbox at
  `/mnt/d/RGB_SFM_artifacts/` (`D:\RGB_SFM_artifacts` on Windows).
- Remote agents may add new files at any time. Inspect `weights/`, `datasets/`
  and recent `notes/` files when resuming work; verify hashes/configuration before
  reusing an artifact. Received files are not automatically active experiment data.
- Shared agent notes live on the HDD in `notes/local/` and `notes/remote140165/`.
  Follow `agent_notes/README.md`: create uniquely named messages, do not overwrite
  another agent's note, and promote durable conclusions into `plan/` and
  `RESEARCH_CONTEXT.md`. Credentials remain outside Git and shared notes.
- Research progress and model lineage are summarized in `RESEARCH_CONTEXT.md`.
  Preserve unrelated uncommitted work when committing experiment changes.
