# HDD 權重／資料集接收服務

更新：2026-10-15。使用者指定先讓 `140.115.51.165:40008` 的 root agent 將權重與資料集同步到本機；目前不主動把本機權重傳過去。使用者確認兩機在相同區域網路，故直接提供 LAN 入口，不依賴遠端 SSH 登入。

## 現況與位址

- LAN URL：`https://192.168.100.233:40115`
- 備援 tailnet URL：`https://100.104.52.34:40115`，僅已獲授權加入同 tailnet 的機器可用，不能假設遠端已加入。
- WSL TLS backend：port 40115，由 `rgb-sfm-transfer.service` 管理，已啟用隨 WSL systemd 啟動。
- Windows LAN TCP relay：`artifact_lan_proxy.py`，監聽 `0.0.0.0:40115`，TLS 在 backend 終止。已有 Windows Python 入站規則匹配這個 Python 執行檔；未建立寬鬆的全域防火牆規則。
- 實際儲存：`D:\RGB_SFM_artifacts`，WSL 路徑 `/mnt/d/RGB_SFM_artifacts`。
- 已透過 Windows Storage Spaces 的 VirtualDisk→PhysicalDisk 查證 D 槽對應 `ST3500418AS` HDD，並非只依 WSL ROTA 推測；建置時約 282 GiB 可用。

目錄分為 `weights/`、`datasets/` 與 `notes/`。傳入內容視為 bytes：不自動載入 `.pth`、不解壓、不執行檔案，也不自動替換本機 training 使用的資料路徑。遠端資料先落在各自命名空間，確認版本後再接到實驗。

**完成的連線驗證：** WSL localhost TLS health、Windows LAN IP 到 relay/backend 的 TLS health，以及檔案上下載 SHA256 驗證。後續使用者已通知遠端完成同步，且本機已實際查驗兩個 run 的 checkpoint/config/history 與 Caltech 官方 archives 入庫；hash 與現有 manifest／官方 MD5 一致。經 proxy 的 server log 顯示來源可能是 localhost，不能把 proxy IP 當成遠端主機身份。詳見[接收與比較紀錄](1015_remote_1014_comparison.md)。

新增 `notes/` namespace 作雙 agent 文件通訊，與 weights／datasets 沿用同一 TLS/token；[共用筆記約定](../agent_notes/README.md)。此目錄與權重／資料集接收目錄之後可能持續新增內容，接續任務前先查目錄／list。

## 憑證與認證包

私有設定放在 `/home/chenws/.local/share/rgb-sfm-transfer/`（目錄 0700，token／私鑰 0600），位於 Git 之外。服務採 TLS 1.2+ 與隨機 bearer token，所有 endpoints 都要求驗證。不要把 token 或認證包貼到公開 issue／Git。

遠端 client 包：`/home/chenws/.local/share/rgb-sfm-transfer/remote-client.tar.gz`（0600），內容為：

- `client.py`：純 Python 3 標準函式庫，同步時不需安裝 PyTorch。
- `server.crt`：用於驗證 TLS 的公有憑證。
- `token`：授權讀寫 weights／datasets／notes namespaces 的私密 credential。
- `README.txt`：操作範例。

TLS 憑證 SHA256 指紋：`E3:98:EC:01:08:CC:65:3D:DA:6C:F8:BE:AE:CD:75:8F:45:E5:B6:7B:AF:45:25:2E:C4:CF:8E:E9:02:12:0B:A6`。憑證一年效期，包含 LAN IP、localhost 與 tailnet 名稱／IP。client 必須使用隨包提供的 cert，不使用 `--insecure`。

遠端已完成首批檔案同步，後續 notes/ 沿用同一份 client、cert 與 token，無須重新交付認證。SSH 40008 已有回應，但本機沒有遠端登入金鑰，也未啟動 SSH tunnel；目前使用直接服務入口。為準備 tunnel 產生的本機專用 key 尚未安裝到遠端。

## 給遠端 root agent

將私有認證包放在遠端後：

```bash
mkdir -p /root/.config/rgb-sfm-transfer
chmod 700 /root/.config/rgb-sfm-transfer
tar -xzf /path/to/remote-client.tar.gz -C /root/.config/rgb-sfm-transfer
cd /root/.config/rgb-sfm-transfer
chmod 600 token
python3 client.py --url https://192.168.100.233:40115 --token-file token --cert server.crt health
```

確認 health 成功後同步；下列 source 路徑需替換成遠端實際位置：

```bash
python3 client.py --url https://192.168.100.233:40115 --token-file token --cert server.crt sync /path/to/runs/train weights/remote140165
python3 client.py --url https://192.168.100.233:40115 --token-file token --cert server.crt sync /path/to/data datasets/remote140165
python3 client.py --url https://192.168.100.233:40115 --token-file token --cert server.crt list
```

這些指令只把遠端 source 上傳到本機，不會從本機取權重回遠端。server 保留 download API，供之後使用者授權的取檔工作使用。

`sync` 會遞迴上傳一般非隱藏檔案，略過 symlink，不刪除任何 source／destination。已存在且 SHA256 相同便略過；同一路徑內容不同會拒絕，應換新的 run／dataset version prefix。它不是 rsync 的差分區塊傳輸，也不是持續監看的 daemon；中斷後重跑可略過完整檔案，未完成的單檔重新傳送。需先停止 checkpoint writer，避免同步正在改寫的模型。

若資料集有必要的隱藏檔或 symlink，可先在遠端自行打包成單一 archive 再用 `upload` 上傳，server 不自行解壓。單檔目前限制 100 GiB，保留至少 2 GiB 空間，最多兩個並行 upload；更大資料集分 shard，不把整包載入 RAM。

## 管理與停止

```bash
systemctl status rgb-sfm-transfer.service
journalctl -u rgb-sfm-transfer.service -n 50 --no-pager
sudo systemctl restart rgb-sfm-transfer.service
sudo systemctl disable --now rgb-sfm-transfer.service
```

Windows relay log：`D:\RGB_SFM_artifacts\lan_proxy.log` 與 `lan_proxy_error.log`。登入啟動入口位於使用者 Startup 的 `RGB_SFM_Artifact_Inbox.cmd`，呼叫 repo 的 `scripts/start_artifact_lan_proxy.ps1`；該 script 每次啟動重新取得 WSL IP。已有相同 port listener 時不重複啟動。WSL 在 Windows session 中途重啟、LAN DHCP IP 改變或搬移 repo 時需重新核對 relay／憑證／URL。

LAN relay 限來源為 `140.115.51.165`、`192.168.100.0/24` 與 WSL 私網；backend 接受 localhost、指定遠端 IP 與 WSL proxy 私網，所有來源仍需 TLS 與 token。tailnet TCP forward 亦已建立，停用：Windows 執行 `tailscale serve --tcp=40115 off`。不使用 public Funnel。

撤銷遠端存取可先停 backend；更換 token 時在私有設定內產生新的隨機 token、保持 0600、重啟 backend，再重建／重新交付認證包。舊 token 立即失效；不要只改 bundle 而沒改 server token。TLS 續期／更換 cert 也必須重新交付 client 信任檔。

## 已驗證的服務行為

TLS 與 bearer 驗證、串流上傳／下載、SHA256、相同檔案重跑略過、不同內容拒絕覆寫、路徑 traversal 拒絕、錯誤 checksum 不落正式檔、無 token 的 401 均已通過。測試資料已移除，未拿本機模型做遠端傳送。

實作：[server](../scripts/artifact_transfer_server.py)、[client](../scripts/artifact_transfer_client.py)、[Windows relay](../scripts/artifact_lan_proxy.py)、[systemd unit](../scripts/rgb-sfm-transfer.service)。
