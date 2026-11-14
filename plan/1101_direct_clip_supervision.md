# 直接 CLIP bbox 教師監督：一次完整訓練

2026-11-01，remote140165 /home/ref/RGB_SFM_V3。使用者取消CBM方向，授權直接用CLIP訊號監督做一次。此次不建立具名概念瓶頸。

既有A/B/H完整100epochs已完成：未見內插尺度80/112的原位及平移平均accuracy為80.42%／77.84%／78.01%。原位96為79.92%／78.28%／78.73%。H相對A的配對來源差值-2.40百分點，source bootstrap95% CI[-3.48,-1.27]；只含影像不確定性，單training seed、探索validation不作final test。完整原報告 runs/train/object_scale_full_20261031T093902Z/REPORT.md。

新實驗：原MergingViT merging，從頭訓練，共同初始化規則、seed42、完整6902 train／1768 validation、100epochs、AdamW lr3e-4、batch16、GPU1。兩view原分類CE加0.2 CLIP cosine蒸餾，原prediction／region一致性權重皆0，主要與舊A CE-only比較。各層pre-merge物件region經各自Linear投影到512維，兩view、四stage的cosine loss平均；梯度回傳投影及backbone，不凍結學生。原分類head保留，推論不需要CLIP。這是ROI pooled教師蒸餾，不等於每token或部件的語意定位。

教師：OpenAI官方CLIP ViT-B/32，凍結eval+inference_mode；從原圖annotation bbox crop出物件區域，先置中補成正方形再套官方preprocess，保留長寬比和完整框，避免CLIP預設center crop截掉長物件。框內仍可能包含背景。教師用原始canonical bbox，不隨學生物件尺寸變換；學生仍看完整224畫布上的物件縮放／平移。目標為image embedding，不使用CBM文字概念表。

程式：scripts/cache_clip_bbox.py、scripts/train_object_scale.py新增 --clip-cache/--clip-weight、scripts/run_clip_supervision_1101.py。權重 weights/clip/ViT-B-32.pt，cache datasets/clip_bbox_vitb32_1101.pt；cache包含8670個source索引、teacher權重SHA256、bbox及annotation SHA256和原split。學生config保存cache SHA256，查核train/validation索引分屬教師cache的原split，validation教師不加入訓練loss。

模型及投影head由同一optimizer訓練，checkpoint分開保存model與alignment_heads，既有分類評估入口可讀model。首次batch檢查投影與patch embedding梯度存在且finite；每epoch記錄train_clip_cosine。smoke兩view101類每類2張、1epoch已成功完成，正式完整訓練已啟動，狀態與history以 runs/train/clip_supervision_1101/full/ 為準；pipeline_status.json需搭配實際PID，不能只看running字樣。100epochs結束後自動跑14條件分類評估。

目前尚無完整結果，不能宣稱CLIP蒸餾改善分類、語意或縮放不變性。後續需看未見尺度accuracy及原尺度代價、獨立validation的跨層CLIP讀取品質，並將投影head與原分類head效果分開。各層共同完整物件教師可能壓抑淺層細節，是本次需留意的假設。

官方來源 https://github.com/openai/CLIP 。安裝記錄 overnight_logs/object_scale_training_1031/clip_install.log，快取log同目錄clip_cache.log。

交接：研究紀錄及程式版本副本與SHA256清單送至E: artifact inbox；本機專案根尚未核實，收件匣交付不表示本機專案匯入。匯入核對hash，保留不同版本。CLIP大權重及訓練checkpoint留遠端。

## 2026-11-03 完成結果

Pipeline與full/status均done，100epochs完成。逐尺度評估所用best checkpoint SHA256重新核對通過；與A的source IDs、split、seed、epochs、batch、model args、prediction/region loss權重相同。

A／CLIP原位96 accuracy：79.92%／79.36%；未見80/112（原位與平移平均）：80.42%／79.84%；14條件最差accuracy：52.04%／53.51%。CLIP−A未見尺度差值-0.58百分點，配對來源1000次bootstrap95% CI[-1.61,+0.40]，不含training seed變異。沒有觀察到整體分類增益；最差條件改善不代表全體更穩定。單seed探索validation與選模有重用，不能判斷普遍優劣。

訓練CLIP cosine最後約0.798，僅為四層兩view的training平均。尚未完成獨立validation的逐層教師檢索、語意區辨、文字／部件解釋性或因果驗證，不能把高training cosine當作語意對齊成功。

結果與來源bootstrap：runs/train/clip_supervision_1101/full/REPORT.md、comparison_with_A.json、scale_evaluation.json與scale_predictions.npz。此結果沒有啟動新訓練；後續若研究跨層解釋，優先檢查validation層別對齊與constant-mean教師基準，排查CLIP各向異性及只學共同方向的捷徑，再決定是否改loss／蒸餾層。
