# 2026-10-20 全類別採樣平移與群一致率圖

使用者要求補群一致率比較圖，以及全類別適量採樣版本，輸出圖片與文字。遠端根目錄為 `/home/ref/RGB_SFM_V3`；本次沒有重訓模型，沒有將新產物恢復至Windows/WSL本機專案。

## 方法與完成範圍

- 在既有三類202張評估新增 `cluster_agreement_comparison.png`，四個stage，三模型同圖；原分類與cosine曲線保留。新增REPORT.txt。
- 全類別版本從原seed42 validation split每類無放回抽5張，採樣seed42，共101類505張。類別等權，micro=macro；不是全部1770張validation。固定source IDs供三模型共同評估。
- 原位、四方向×1–8/16/32px；160-in-224、修正annotation中心；每模型82820筆token配對，總計248460筆。
- 直接沿用上一輪三類混合評估的每模型、每stage frozen spherical codebook；字典原以6907張未平移training中心token fitting，K31/niter100/nredo50/seed42。不用新505張重fit。
- 空閒GPU cuda:1 RTX3060；三模型全評估完成。0px identity、完整筆數、唯一鍵、data fingerprint、source IDs、train/validation分離、centroid norm與hash皆通過；15張與舊三類重疊的影像在全部位移的分類／群指派一致、cosine差異<1e-5。
- 5000次seed42 paired stratified bootstrap，固定101類、各類內重抽5張來源影像；先於每張影像彙整同組位移／方向。不包含重訓seed或其他採樣seed的不確定性。

## 主要結果

| 模型 | 原位 | 1–7px平均 | 8/16/32px平均 | 十個位移等權 | 兩組各50% |
|---|---:|---:|---:|---:|---:|
| Baseline | 44.95% | 41.56% | 30.12% | 38.12% | 35.84% |
| Aligned 8/16/32 | 49.90% | 48.39% | 46.65% | 47.87% | 47.52% |
| T1 1–7 | 56.24% | 54.62% | 40.07% | 50.25% | 47.34% |

全類別平衡採樣上，T1在小位移的accuracy最高，aligned在大位移最高。兩組各50%時點估計相近，不應把三類子集的排名差距泛化至此。44.95/49.90/56.24為本505張平衡採樣原位accuracy，不能與全部1770張原始加權62.09/65.71/68.36直接比較為模型退化。T1絕對accuracy較高也包含較高原位表現，需一起檢查平移相對原位降幅。

Stage0–2相應對齊條件下，aligned增強後群一致率與cosine高於T1在1–7平均的值；T1組的表徵絕對增益更大。但Stage3同樣1–7測試時，T1群一致率74.34%高於aligned71.07%，cosine卻0.8925低於0.9125，不能用單一指標或三類子集概括所有stage。

單一training seed、validation參與選模、每類只有5張、兩組位移量不同，不能推論alignment-only因果作用、單類穩健結論或嚴格平移不變性。

## 產物與重現

全類別圖片與文字：

- `plots/kmeans/caltech_allclass_translation_1020/classification_comparison.png`
- `plots/kmeans/caltech_allclass_translation_1020/cluster_agreement_comparison.png`
- `plots/kmeans/caltech_allclass_translation_1020/token_cosine_comparison.png`
- [REPORT.md](../plots/kmeans/caltech_allclass_translation_1020/REPORT.md)與同目錄REPORT.txt
- selection_manifest.json、all_summary.csv、plot_data.csv、group_summary.csv、per_image_metrics.csv、paired_contrasts.csv、direction_summary.csv、overall_accuracy.csv、matched_training_test_comparison.csv、category_effect_counts.csv、verification.json、analysis_manifest.json。

原三類群一致率圖：`plots/kmeans/caltech_mixed_translation_1020/cluster_agreement_comparison.png`。Raw paired CSV、NPZ feature caches與codebooks留遠端，不納入Git；圖文與compact表格允許追蹤。

執行 `ALLCLASS_DEVICE=cuda:1 bash scripts/run_allclass_translation_1020.sh` 後 `.venv/bin/python scripts/analyze_allclass_translation.py`；三類圖文使用 `.venv/bin/python scripts/analyze_mixed_translation.py`。Log/status：`overnight_logs/allclass_translation_1020/`，shared claim及收據在其inbox。
