# 物件縮放／平移研究原型

2026-10-31，remote140165 `/home/ref/RGB_SFM_V3`。

已實作 `models/ObjectScaleViT.py`：獨立子類、可選局部內容加權 merging、pre-merge stage 特徵輸出、固定背景的 premultiplied-alpha 物件合成、對應區域 readout 與雙 view 損失。預設沿用既有 merging，原模型檔未修改。停用新 merging 時可 strict 載入 baseline state_dict 並逐值重現 logits。

局部 merging 在固定 stride 的重疊窗口內，用學習權重聚合，排除 padding，輸出 token 預算不變。這是可消融原型，尚未依物件大小動態調整合併深度，也沒有平移或尺度等變保證。last_weights、last_valid 僅記錄最近一次 forward 的 detached 局部權重；不是完整跨層 lineage 或來源面積追蹤。

`render_object` 接受同尺寸 foreground、真實 object mask、獨立 background；theta 是 view→source 正規化座標。不能用 bbox 充當精確分割，也不能從未移除物件的原圖背景直接合成，否則留下原物件。尚未接上經稽核的 Caltech 輪廓資料、背景重建與正式 paired DataLoader。

`corresponding_regions` 需要呼叫端提供 source→view 的共同採樣 grid 與共同有效遮罩。深層網格包含 padding，正式接線時必須依實際 patch/merge stride 與支援範圍建立座標，不可將 padded feature map 直接當均勻覆蓋整個畫布。現有測試只驗證函式與梯度，不驗證真實部件幾何對應。

CPU 驗證：`.venv/bin/python checks/check_object_scale_prototype.py` 通過 baseline parity、奇數／1×1 網格、padding 權重、固定背景、物件縮小、finite loss 與 merging 梯度。未啟動訓練，尚無新泛化結果。

下一階段：稽核 mask／背景，建置有 provenance 的 paired dataset、正確 feature 座標對應、A/B/D 共同訓練入口；先小樣本 overfit 與幾何驗證，再跑相同 seeds／預算的正式比較。B 與 D 須使用相同雙 view 和 loss，另以參數／計算量相近的平均 pooling 控制拆開架構容量效應。保留原尺度、未見尺度／位移及聯合變換測試；目前 validation 已用於探索，不當作 final test。

交接：版本副本與 SHA256 清單送至 E: artifact inbox；這不代表已匯入本機 RGB 專案。本機專案根目錄尚未核實，匯入時核對 SHA256，保留不同版本。
