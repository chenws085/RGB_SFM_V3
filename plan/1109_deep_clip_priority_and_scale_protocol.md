# 深層 CLIP 優先與跨層／跨尺度評估定位

日期：2026-11-09。機器：remote140165。專案：/home/ref/RGB_SFM_V3。

## 使用者偏好與主指標

使用者明確要求記住「CLIP監督適合用在深層」，後續設計優先考慮Stage3/4直接監督；研究重點為跨層＋跨尺度同群率。這是工作假設與設計偏好，不能寫成普遍已證實：既有深度消融未證明深層監督普遍優於全層或提升分類。此回合僅回顧及記錄，未授權或啟動新的模型訓練。

## 本次兩組的實際設定

比較的是runs/train/object_scale_full_20261031T093902Z/A與runs/train/clip_supervision_1101/full。後者是四層CLIP直接監督，並非deep-only。兩組使用原merging、從頭訓練、seed42、100epochs、batch16、AdamW lr3e-4、6902training/1768validation來源。

學生兩組皆是Caltech contour cutout去背，物件等比例縮放並貼到224×224純色畫布；每個來源／epoch抽背景色，與類別無關，兩view共用背景。各view訓練長邊64/96/128、x/y平移-16..16，兩view都做CE並取平均。prediction KL與region consistency權重皆0。

CLIP組額外加0.2×四層兩view平均cosine蒸餾loss。各層pre-merge整物件mask加權8×8 ROI經各自Linear到512維；梯度回到學生backbone與heads。CLIP ViT-B/32教師凍結；教師輸入是原始自然影像的annotation bbox crop，保留框內背景，補正方形後official preprocess；不是學生去背view。教師target是canonical同來源image embedding，不是文字、部件或每token標註。

目前分群表比較兩模型raw中間表示的ROI pooling，不使用CLIP訓練head或事後probe。保留自然背景原圖的另一組不是此表來源，不能混用。

## 資料切分用途

6902 training原位96建各層FAISS spherical K31／100iter／50restarts／seed42字典；跨層training同物件指派建立Hungarian一對一群號對應。字典及群對應凍結後，1768 validation算96→80/112置中及歷史80+(7,-5)的同群率，不在validation重fit字典或群號映射。

所以跨層比較有用validation做評估，只是不用它學群或配群。validation先前也以80+(7,-5)分類accuracy選best checkpoint，因此不是untouched final test。跨層數值應稱training群號對齊後的指派一致率；跨層群號沒有天然共享語意，一對一對應也不是部件階層關係真值。

## Receptive field 疑慮與後續協定

同層跨尺度仍是有效固定深度穩定性指標，但固定網格在不同大小物件上覆蓋不同相對區域，不宜將下降全歸為語意不穩定。跨層比較也不自動解決，因層深與特徵幾何另有變化。

四stage名義stride／patch merging支援尺度為8/16/32/64px；attention為全局，這些不是完整或實測有效receptive field。當前物件96↔80/112、層間2倍stride並未做相對支援尺度匹配。8×8物件正規化ROI採樣匹配的是來源座標與pooling區域，沒有讓各層token取得相同資訊範圍。

後續跨層主指標可另外報告物件相對支援尺度匹配的配對，近似條件stride_s/L_s≈stride_t/L_t，例如Stage3的64物件對Stage4的128物件（更深層配較大物件），以及相反方向；這只是名義支援尺度控制，不是已測得ERF匹配，也不是已完成新實驗。需保留既有96→80/112協定作獨立對照，避免換協定後混報。若研究逐token／部件，需另抽對應局部token，不把整物件ROI結果當作逐token證據。同層跨尺度降為輔助診斷，分類與同類異圖控制保留以確認區辨性。

## 交接

此研究偏好與討論需版本副本及SHA256送E:\RGB_SFM_artifacts HTTPS inbox。本機專案絕對路徑／匯入尚未核實；收件匣交付不是專案匯入，匯入需驗hash並保留不同版本。
