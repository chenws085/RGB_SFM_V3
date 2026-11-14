# Local research handoff

## Research document copies (user preference, 2026-10-21)

- Keep research plans, literature reviews, and discussion notes on both machines. Save in the remote project and upload a versioned copy with SHA256 manifest to the local E: artifact inbox via the verified HTTPS service. Include a handoff note for importing into the local project when its root/access is confirmed. Inbox delivery is not proof of local project import; report those separately. Preserve differing versions.

## Two separate RGB project directories

- Both the remote machine and the user's local Windows/WSL machine have an
  RGB project checkout. They are separate directories, not a shared filesystem.
- The current remote140165 container checkout is `/home/ref/RGB_SFM_V3`.
  The user also has a local RGB project; its exact absolute path has not been
  verified here. Do not invent that path or equate it with the remote checkout.
- Before downloading, extracting, or reporting a destination, verify the active
  machine and project root. Explicitly distinguish the remote project, local
  project, and artifact inbox. A download into this container does not put files
  into the user's local project.
- The local artifact inbox is now `E:\RGB_SFM_artifacts`
  (`/mnt/e/RGB_SFM_artifacts` in local WSL), per
  `notes/local/20261015T174407Z_storage_migration_done.md`.
  The old D: location is a migration backup, not the active inbox.
- Copying an archive into the inbox does not restore its contents into either
  project's `plots/`. When asked to restore experiment data, use the intended
  project's `plots/` directory and verify file hashes; preserve differing files.
  If the intended machine is inaccessible, state that limitation explicitly.
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

## Remote agent on cws-ws-140165

- This container is the remote140165 side. The HDD path above belongs to the
  receiving Windows/WSL machine; access it through the HTTPS inbox rather than
  assuming `/mnt/d/` exists in this container.
- Use `https://100.104.52.34:40115` (cws-lab). The LAN URL in older examples is
  not reachable from this container. Existing client credentials are under
  `/root/.config/rgb-sfm-transfer/`; never copy their contents into Git or notes.
- Tailscale runs as user `ref` in userspace mode. Device name: `cws-ws-140165`;
  observed IP on 2026-10-15: `100.105.161.28`. State and socket are under
  `/home/ref/.local/state/tailscale/`. Start idempotently with
  `runuser -u ref -- /home/ref/.local/bin/tailscale-user-start`.
- HTTPS requests must use the userspace HTTP CONNECT proxy at
  `127.0.0.1:41081`, preserving certificate verification for `100.104.52.34`.
  The original Python client does not automatically honor proxy environment
  variables; use an explicit proxied connection, as implemented in
  `overnight_logs/artifact_sync_1014/retry.py`. That script uploads artifacts;
  do not run it merely to read notes.
- On resuming a task, list the inbox and read `notes/README.md` and the latest
  `notes/local/` messages. Cache downloads locally without overwriting existing
  files. There are no push notifications.
- When authorized to coordinate a shared task, post a claim before starting
  overlapping experiments. Our messages belong in `notes/remote140165/` with
  unique UTC filenames, e.g. `20261015T120000Z_topic.md`. Include `author`,
  `utc`, `task`, and `status` (info/question/working/done/blocked), followed by
  evidence, file paths, and next steps. Replies/corrections are new files with
  `reply_to`; never overwrite another agent's note. New artifacts should include
  their paths, provenance, and SHA256. See `agent_notes/README.md` for the shared
  protocol. A request to inspect notes alone does not authorize posting messages.
- Last read: `notes/local/20261015T115708Z_received_1014_models.md`.
  The receiving agent confirmed the historical models/dataset and claimed the
  seven-scale comparison of the two historical checkpoints against S0/S1/T1.
  Its status was `working`; check newer notes/status before assuming completion
  or starting duplicate work. Original 10/14 Euclidean clustering must not be
  relabeled as spherical clustering.
- Login state is on the persistent `/home` mount, and daemon restart has been
  verified to reuse the login/IP. Container autostart is not yet configured:
  the host must invoke `/home/ref/.local/bin/tailscale-container-entrypoint`
  before the normal container command. Do not claim uninterrupted connectivity
  during restart or guaranteed login beyond key expiry/revocation.
