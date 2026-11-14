# window_size × sliding_or_not：先固定lr

使用者2026-11-10指定主表只保留global、window4、window7、sliding_window4、sliding_window7。其後明確要求「掃lr暫緩」；本輪只使用既有lr=3e-4，其他lr沒有排程，不會自動開始。

遠端remote140165，專案/home/ref/RGB_SFM_V3。新主表與逐項狀態位於runs/train/attention_size_sliding_20261110_lr3e4/REPORT.md及status.json。主表五欄方法、每個指標各一張表，避免把多指標及多種架構因素擠在一起。

## 固定條件

局部方法一律用於Stage0/1，Stage2/3均global；這是固定共同設定，不掃stage。保留原absolute positional embedding，offset=0；absolute position、stage選擇、offset均暫緩。原merging、head、QKV/heads/MLP不變，參數量不變。Caltech101 6907/1770、seed42、letterbox160完整自然圖、歷史flip/jitter/erase、無平移增強，batch32、AdamW lr3e-4 wd.01、CE smoothing.1、100epoch cosine eta_min1e-6、patience20。

固定window4/7是互斥窗口。Sliding4/7是每個query的重疊鄰域；sliding7每軸[-3,…,3]，sliding4每軸[-1,0,1,2]，都含query本身，出網格keys遮蔽，不循環連接。偶數4不能在整數網格完全對稱置中，其方向偏置是此配置的已知限制，不額外掃offset。

可重用global、window7、sliding7_stage01的lr3e-4結果；已驗checkpoint/結果hash、訓練設定及歷史sliding7逐值相容。window7原本Stage2/3單一窗口，與深層global逐值相同，重用檢查通過。舊window4曾將Stage2局部化，不能直接填入本主表，需重訓固定Stage0/1的window4；另補sliding4。只有這兩個固定lr新模型會訓練。

主指標分表：原位accuracy、1–7px accuracy、32px accuracy與分類一致率、one-token群一致率與cosine。採相同training-only spherical字典規格及1770validation四方向0/1–8/16/32px測試。單seed多方法探索、validation曾選模，不把差異當最終泛化結論。

## 未來規劃（不執行）

- lr sweep：暫緩，候選範圍可在重新授權後討論；本程式目前只有3e-4，不包含其他lr。
- Absolute position、stage與offset消融：暫緩，不混入目前五欄。
- 低通濾波：保留為未來方案，分開檢查初始patch embedding降採樣前、merging前，再考慮兩者聯用；須記錄濾波kernel、padding、細節損失與原位分類。不在此輪加濾波或執行訓練。

舊七組表完整保留為歷史探索；window14/offset3/stage0-only不放進現在主表。文件版本及SHA256交付E:\RGB_SFM_artifacts inbox；本機專案root/access未確認，交付不是匯入，之後匯入需驗hash並保留不同版本。

## 2026-11-10 主表版型修訂

使用者指定改為2D表：第一欄五種方法，每列一種；後續欄位accuracy（原位）、1–7px_acc、8px_acc、1–7px一致性、8px一致性。使用者已確認最後一欄為8px一致性。「一致性」沿用分類prediction consistency，即與同張原位預測相同的比例，不是分群一致率；報告已明確定義。1–7px是7位移×4方向平均，8px為4方向平均。五組均已完成或驗證重用；lr維持3e-4，不恢復掃描。原其他指標仍保留完整評估及results.json，主表不混放。
