# 本次報告：CLIP直接監督留給深層之架構消融

日期2026-11-03，remote140165 /home/ref/RGB_SFM_V3。使用者要求改成深層CLIP訊號並提供實驗佐證。狀態：架構／smoke已完成，完整消融執行中；尚無最終勝負結論。

## 假設與改動

四stage採1起算：淺層1/2、深層3/4（CLI zero-based 0/1與2/3）。新增 --clip-stages；只有選中stage的projection head接直接CLIP loss，其他head凍結、不進optimizer。保留四個head索引以兼容舊checkpoint／分析。主幹、merging、分類head仍訓練；深層梯度仍反傳淺層，不稱淺層完全不受CLIP影響。預設全四層保留舊行為。

## 預先指定的對照

沿用已完成A CE-only及all CLIP四層0.2，新增deep 3/4層0.2、shallow 1/2層0.2、deep_half 3/4層0.1。新組同seed42、6902train／1768validation、原merging、100epochs、batch16、AdamW3e-4、相同CLIP教師cache、相同雙view資料，從頭訓練。Loss先對active layers與view平均，再乘總權重。deep0.1使每層係數與all0.2相同，排除移除淺層同時增強深層係數的混雜。

主評估：原分類head原尺度、未見80/112平均、最差14條件accuracy。deep-all、deep-shallow、deep_half-all、deep-A做來源影像配對bootstrap1000次，不把view當獨立影像。既有all/A為歷史單seed對照，核對split／seed／model args／batch／epochs及teacher cache，保留程式快照；不宣稱多seed確認。

語意證據分兩種：(1) 凍結A各層等rank32 PCA＋ridge10到CLIP教師，僅training fit，驗證同來源teacher檢索與cosine超過training teacher平均常數基準。檢驗原始各層的CLIP語意可讀取性，不讓深層較大維度單獨解釋投影容量差異。(2) 各訓練組active projection在validation的teacher cosine及同來源retrieval，不能用training cosine作唯一證據。CLIP教師檢索仍是代理，不是人工部件／屬性真值。

## 檔案與狀態

訓練scripts/train_object_scale.py；佇列scripts/run_clip_depth_ablation.py；凍結probe scripts/probe_clip_stage_semantics.py；報告scripts/report_clip_depth.py。

runs/train/clip_depth_1103_smoke/：1epoch已done；selected heads梯度finite、未選head無梯度的runtime checks通過。runs/train/clip_depth_ablation_1103/：deep→shallow→deep_half依序訓練及14條件評估，完成後自動生成REPORT.md、results.json。完整報告的結論必須依結果支持或否定假設，不能預填改善。先啟動deep，GPU1，未動GPU0其他工作。probe結果寫stage_semantic_probe.json。

仍需多seed、未探索final test及人工概念評估才能主張一般性；本次是探索性架構實驗。將「監督深度較有效」與「深層具有較明確語意」分開判讀。未直接觀察到淺層受損時，不寫成已證實機制。

## 交接

研究計畫／本次報告架構、程式及狀態快照以版本與SHA256經HTTPS送E: artifact inbox。未確認本機專案根；收件匣交付不是本機專案匯入。匯入請核對hash並保留不同版本。最終報告落遠端run，待訓練結束補交結果。

## 已完成的初步實證：凍結A之等秩probe

同rank32、同ridge10、只用training fit後，各層validation teacher cosine為0.7888／0.8011／0.8120／0.8266，常數training teacher平均向量基準0.7403；同來源retrieval top1為0.96%／0.74%／1.58%／2.83%（1768候選）。深兩層CLIP語意代理的可讀取性較強，初步支持選深層的動機，但retrieval仍低，且不是人工語意真值。這不能先行證明深層監督的分類／尺度效果，後者等待完整消融。探測與新訓練分開，沒有用probe結果改變事先設定的rank或ridge。

## 訓練圖片流程圖

已用真實source2216繪製圖片處理與深層CLIP監督兩圖，PNG／PDF／SVG位於 plots/reports/clip_depth_training_flow_1103/。詳細圖說README.md，幾何及程式SHA256見figure_manifest.json。圖示依目前程式，區分學生contour cutout與教師原始annotation bbox兩路，未將示意圖當作效果證據。

## 完整消融完成

三個新增組100epochs與14條件評估、held-out語意代理及配對統計已完成。最終報告 runs/train/clip_depth_ablation_1103/REPORT.md。
目前對照不足以支持深層監督普遍優於全層及淺層；不將推斷視為已證實。
