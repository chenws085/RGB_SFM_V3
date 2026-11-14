# 使用者決定：CLIP跨層分類／分群與物件縮放不變性

日期2026-11-09。remote140165，/home/ref/RGB_SFM_V3。此為使用者研究決定與本次輸入稽核，未啟動新訓練。

## 已決定的研究主線

使用者指定：「CLIP監督方法作為使用跨層分類，面對物件縮放不變性的研究方向」。以CLIP監督建立／改善跨層分類與分群及物件尺度穩定性；原始merge_ViT為主要基準。基準匹配限於同樣範圍的內容，至少需同來源、同物件／部位、對應物件相對座標與範圍，不能只因不同層網格比例相等便稱嚴格相同有效RF。現有整物件ROI與名義stride比例／相位控制僅為近似協定，不能事後改稱嚴格局部token或實測有效RF匹配。

下一步方法目標：保留跨層＋跨尺度能力，同時希望CLIP監督後同層指標也不下降；此為後續目標，尚非已達成結論。原始特徵與投影後特徵的指標分開報告；各層獨立字典＋training配群、投影共用字典直接比較、cosine，不可混稱同一指標。後續應事先固定同層／跨層指標、對照內容與可接受差值，不能僅以某深層改善替代整體非劣證據。

延續前述深層CLIP優先的工作假設，但不將既有deep-only對照未勝出的事實改寫成已證實深層監督最佳。

## MHA選擇未定

使用者明確表示目前全局MHA／窗口MHA尚無定論，不自動指定最終架構。全局attention有較高attention計算成本和直接跨遠距位置互動能力；計算量較大不構成準確率必然較高的理論保證。窗口具較低attention成本，局部先驗可能有利優化或泛化，但「更容易收斂」仍是待評估假設，需用共同收斂門檻／learning curves／時間與多seed對照確認。現有window7消融後兩stage仍覆蓋全層，不能泛化為所有局部attention設計。

## 本次訓練輸入的確認：不是完整自然背景原圖

查核runs/train/object_scale_full_20261031T093902Z/A/config.json、runs/train/window_attention_20261109T064322Z/local/config.json，以及window的source_snapshot/dataloader/ObjectScalePairs.py：

- 全局CE與窗口CE皆為Caltech輪廓cutout去背學生輸入，物件按長邊64/96/128縮放並每軸−16..16平移，貼到224×224合成純色背景。
- 背景色依source/epoch抽樣，與類別無關，同來源兩view共用背景；不是保留原始背景。
- 共同6902training／1768validation、seed42、100epochs。這是同一加工資料協定的attention控制比較，不是自然完整原圖比較。
- window config明確original_images=False、no_translation=False、attention_mode=window、window_size=7、clip_cache=None。global歷史config未記錄original_images旗標，但background明確記為class-independent uniform color, same background across views，歷史訓練程式與dataset相符。
- 先前CE＋CLIP監督的學生也使用相同cutout。凍結CLIP教師使用原圖annotation bbox crop、保留框內背景，並非整張完整原圖；教師看到原圖bbox不表示學生使用原圖。

「原始merge_ViT模型架構作基準」與「完整原始自然背景影像作輸入」是不同選擇。此回合未自動將已完成的去背比較改稱原圖比較，也未自行啟動自然背景重訓。若後續要求原圖比較，需在同一原圖處理協定下另作全局／窗口或CE／CLIP公平對照。

## 交接

此決定與來源稽核保存遠端，版本與SHA256經HTTPS交付E:\RGB_SFM_artifacts inbox。本機專案root與匯入未核實；收件匣交付不是專案匯入，匯入需驗hash並保留不同版本。
