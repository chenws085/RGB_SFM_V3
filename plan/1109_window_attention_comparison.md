# 全局MHA與局部窗口MHA：實驗完成

日期2026-11-09。機器remote140165，專案/home/ref/RGB_SFM_V3。

## 問題與架構解讀

Transformer channel對應d_model，本實驗四層32/64/128/256、heads2/4/8/16，每head16維；同stage維持C，MLP內C→4C→C，2×2 merging先4C再Linear→2C。全局MHA可早期整合遠距部件，但其token已混入全圖資訊，局部父子merging框不是完整有效RF，對部件組合研究的局部解釋有限。局部MHA仍是多頭，改的是範圍。

就目前程式架構而言，屬全局attention的階層式ViT，merging形式近Swin、全圖金字塔思路可對照PVT/PiT；不是PVT實作，因為沒有PVT的SRA（壓縮K/V）。資料來源：https://arxiv.org/html/2102.12122v2 、https://arxiv.org/abs/2103.14030 。不能將未讀完整論文的所有論證等同於本repo實作。

## 實驗設定及結果定位

使用7×7不重疊窗口，無移窗／無新relative position bias，保留原版merging、位置編碼、深度、heads及參數量。grid28/14/7/4，實際窗口7/7/7/4，所以後兩stage仍覆蓋全層；這次主要局部化前兩stage，不能稱為所有stage均嚴格局部或完整Swin。

純CE、相同初始張量、6902training／1768validation、seed42、100epochs、batch16、AdamW3e-4。去背學生＋同view純色背景，原64/96/128物件縮放與每軸−16..16平移；KL／region consistency均0，無CLIP。原全局CE基準沿用既有100epoch checkpoint，新window best epoch96。實際torch參數量兩者1304261。初始參數完全相同；attention dropout張量形狀改變，不能聲稱逐位置隨機遮罩相同。

完成模型局部性、padding、梯度及原版相容性驗證，implementation_checks.json；完成checkpoint／輸出及來源驗證，validation_checks.json。相同FAISS spherical K31／100iter／50restarts／seed42，各模型training96建各層字典，training Hungarian配群，validation評估。分群是原始整物件ROI，沒有投影或共用跨層字典。

跨層96↔80/112平均35.92→37.20%，+1.28pp，來源95%CI[+0.13,+2.49]，但同類異圖差距只增加0.12pp。3↔4為38.73→44.98%。名義支援比例匹配平均38.59→35.60%，−3.00pp；網格位置校正後37.75→35.99%，−1.75pp。因此不能宣稱局部化全面改善跨層尺度一致性。原位分類79.92→81.73%，未見尺度分類80.42→81.04%，後者差值+0.62pp的CI[-0.54,+1.80]跨0。單seed探索／選模validation，非final test。

載入window checkpoint必須用config中的attention_mode/window_size，不能因state_dict keys與全局模型相容就以全局forward評估。訓練及分析程式與來源hash均保存。權重及大型特徵cache保留遠端，文件與小型結果版本副本及SHA256送E: inbox，不表示本機專案匯入，本機root尚未核實。

# 原版全局MHA vs 局部窗口MHA：純CE控制實驗

相同初始化張量、參數量1304261、原merging、絕對位置編碼、stage深度與head數；純CE無CLIP，完整6902training／1768validation、seed42、100epochs、batch16、AdamW3e-4。學生皆去背物件與合成背景、雙view縮放64/96/128與每軸−16..16平移。

窗口7×7，四stage實際窗口7/7/7/4；後兩stage7×7與4×4網格的attention仍覆蓋全層。沒有shifted windows或relative position bias，故不是完整Swin；本次主要隔離前兩stage全局／窗口attention。兩組相同初始權重及資料，attention dropout張量形狀不同，不能聲稱逐位置dropout亂數完全匹配。

各模型原始pre-merge整物件ROI 8×8 mask加權pool後分群；不是逐token分群。FAISS spherical K31、100iter、50restarts、seed42，training96建字典及Hungarian群號配對，validation只評估。global重用歷史原版字典並逐筆重現舊96／80平移群指派。

| 分類指標 | 全局MHA | 窗口MHA |
|---|---:|---:|
| 原位96 | 79.92% | 81.73% |
| 未見80/112（含原位與平移） | 80.42% | 81.04% |

## 同層跨尺度80/112

| 層／配對 | 全局MHA | 窗口MHA | 差值 |
|---|---:|---:|---:|
| 1 | 77.91% | 77.83% | -0.08pp |
| 2 | 80.51% | 74.35% | -6.17pp |
| 3 | 73.81% | 76.67% | +2.86pp |
| 4 | 81.28% | 84.93% | +3.65pp |
| 平均 | 78.38% | 78.44% | +0.06pp |

## 舊條件96→80+(7,-5)

| 層／配對 | 全局MHA | 窗口MHA | 差值 |
|---|---:|---:|---:|
| 1 | 73.02% | 72.85% | -0.17pp |
| 2 | 76.24% | 70.25% | -6.00pp |
| 3 | 63.07% | 69.12% | +6.05pp |
| 4 | 77.09% | 82.30% | +5.20pp |
| 平均 | 72.36% | 73.63% | +1.27pp |

## 跨層＋跨尺度96↔80/112

| 層／配對 | 全局MHA | 窗口MHA | 差值 |
|---|---:|---:|---:|
| 1↔2 | 49.38% | 42.80% | -6.58pp |
| 1↔3 | 33.60% | 37.08% | +3.48pp |
| 1↔4 | 23.81% | 25.38% | +1.57pp |
| 2↔3 | 41.20% | 44.58% | +3.38pp |
| 2↔4 | 28.79% | 28.38% | -0.41pp |
| 3↔4 | 38.73% | 44.98% | +6.25pp |
| 平均 | 35.92% | 37.20% | +1.28pp |

## 相鄰層名義支援比例匹配64→128／96→192

| 層／配對 | 全局MHA | 窗口MHA | 差值 |
|---|---:|---:|---:|
| 1↔2 | 46.61% | 38.63% | -7.98pp |
| 2↔3 | 36.74% | 33.06% | -3.68pp |
| 3↔4 | 32.44% | 35.10% | +2.66pp |
| 平均 | 38.59% | 35.60% | -3.00pp |

## 相鄰層支援比例＋網格位置校正

| 層／配對 | 全局MHA | 窗口MHA | 差值 |
|---|---:|---:|---:|
| 1↔2 | 46.61% | 38.63% | -7.98pp |
| 2↔3 | 35.07% | 34.47% | -0.59pp |
| 3↔4 | 31.56% | 34.87% | +3.31pp |
| 平均 | 37.75% | 35.99% | -1.75pp |

## 配對來源bootstrap：窗口−全局，平均指標

- within: +0.06pp，95% CI [-0.86, +1.11]，同類異圖差距改變 -3.34pp。
- within_shift: +1.27pp，95% CI [-0.01, +2.53]，同類異圖差距改變 -2.42pp。
- cross_historical: +1.28pp，95% CI [+0.13, +2.49]，同類異圖差距改變 +0.12pp。
- support_center: -3.00pp，95% CI [-4.41, -1.54]，同類異圖差距改變 -3.02pp。
- support_phase: -1.75pp，95% CI [-3.10, -0.36]，同類異圖差距改變 -2.00pp。

單seed、探索／選模validation；bootstrap2000次先在來源內平均，再重抽來源，不含訓練或K-means seed變異。支援比例及位置校正不是實測有效RF或逐token支援框匹配；位置校正也改變絕對位置。群佔用與同類異圖控制見results.json。

遠端專案 /home/ref/RGB_SFM_V3；E: inbox版本交付不代表本機專案匯入，本機root未確認，匯入需驗SHA256並保留不同版本。
