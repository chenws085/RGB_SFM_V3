# 最早代表圖／平移實驗：global attention 解讀修正

日期：2026-11-09；機器：remote140165；專案：/home/ref/RGB_SFM_V3。
範圍：討論與建議，未修改程式、舊結果或啟動訓練。

## 已核對的問題

`plan/0921_stage_translation_results.md` 與 `scripts/test_caltech_translation_kmeans.py` 把 8/16/32 pixel 的幾何 token crop 稱為 RF。對經 global self-attention 的 token，此框是空間錨定區域／名義 merge footprint，不是完整資訊來源或有效感受野。代表樣本是依 token feature 與固定 training centroid 的距離選出；顯示其 crop 不等於證明群只編碼 crop 的內容。global attention 允許全圖依賴，但不表示每個位置對 token 等量或實際有顯著影響。Window attention 經多層、shift 與 merge 後也不宜直接以單一 window 表示完整感受野。

原平移協定仍可測同一空間內容對應位置的 contextual-token 穩定性。正確比較為 f_s(T_delta x)[p+delta/a_s] 與 f_s(x)[p]；a_s 是 token stride。需驗證整數網格對齊與 crop 像素一致。跨 stage 同時改變位移量，不能只歸因於深度；補相同 pixel 位移的跨層比較，非對齊條件另外說明對應取樣。

歷史報告已註明 annotation bounding-box offset 遺漏；不可直接作物體中心證據。正式分析採 10/12 修正座標版本。原 KMeans 是 normalized-feature Euclidean，不改稱 spherical。

## 建議的圖與補實驗

1. 最小修改：每個代表樣本顯示完整影像＋token 框，旁列局部 crop；名稱改「群代表 token 的來源影像／空間對應區域」。保留 training top-k、距離及來源，避免只展示一張作語意定論。
2. 增加 token／cluster 專屬歸因：以該 token 對固定 centroid 的 affinity，或原群與競爭群的分數差為目標，計算輸入梯度／Integrated Gradients。原 sklearn 分群若用 margin，應使用與實際 assignment 一致的距離與特徵處理；cosine affinity 必須另標示為診斷分數。分類 logit 歸因回答分類依據，不能直接代替 cluster 歸因。熱圖稱敏感度／歸因，不當作真實感受野。
3. 局部依賴控制：固定錨定區域像素而改外部 context；固定外部而替換／遮擋局部；用完整平移作原對照。明定 mask、邊界與替換資料來源，重算特徵、assignment、margin、分類，匹配擾動面積／強度並採多種替換。遮擋會離開訓練分布，不能單獨作自然語意因果結論。多層 token 即使 anchor pixels 一致，也可因外部資訊而改變。
4. Attention rollout 可作輔助資訊路徑視覺化；須處理 residual、各層 attention 與真實 merging 的維度轉換。單層 attention 或 CLS heatmap 不是該 token 的完整來源，也不能獨立當成因果證明。

推薦圖說：「框線標示 token 在輸入上的空間對應區域。代表樣本依訓練集 token feature 與群中心的距離選取。因模型採用 global self-attention，token 特徵可能整合框外資訊；局部裁切不表示完整感受野或獨立局部概念。」

參考：ViT https://arxiv.org/abs/2010.11929；Attention flow/rollout https://arxiv.org/abs/2005.00928。

## 本機交接

本文件保存於遠端專案。應將本版與 SHA256 manifest 送到已確認的 E:\RGB_SFM_artifacts inbox；此處不是本機專案根目錄。本機專案絕對路徑尚未確認，收到後由本機代理核對 SHA256，再匯入其 plan/，保留不同版本。
本輪透過指定 userspace CONNECT proxy 存取 HTTPS /files 發生 read timeout，無法讀取最新共享筆記。沒有投遞共享協調訊息，沒有執行重疊實驗。Inbox 送達與本機匯入皆需分別確認。
