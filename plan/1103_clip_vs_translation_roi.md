# CLIP 與歷史平移增強T1：同條件ROI補評

6902 training建獨立spherical K31字典；1768 validation，同物件96原位對80+(7,-5)，四層mask加權region readout。100 iterations/50 restarts/seed42。使用T1完成版best epoch78，SHA256見manifest。與舊整圖平移中心token結果不可直接互換。

| 層（1起算） | T1 平移增強 | A 物件增強CE | CLIP監督 | CLIP−T1 |
|---|---:|---:|---:|---:|
| 1 | 83.37% | 73.02% | 76.02% | -7.35 pp |
| 2 | 73.47% | 76.24% | 73.59% | +0.11 pp |
| 3 | 58.99% | 63.07% | 67.02% | +8.03 pp |
| 4 | 54.07% | 77.09% | 81.22% | +27.15 pp |

以上為群指派一致率，群號不可跨模型比較。四層T1／CLIP的配對cosine減同類異圖cosine差距：
- Stage 1: 0.3625 / 0.1641
- Stage 2: 0.2795 / 0.1323
- Stage 3: 0.2134 / 0.1541
- Stage 4: 0.2843 / 0.2670

T1與新模型的訓練資料前處理／增強不同，此為共同評估條件下的checkpoint比較，不能把T1差距全歸因CLIP訊號。CLIP純效果應優先比較A。單seed探索validation、ROI pooling、單一尺度+平移條件，非逐token或部件語意證據。bootstrap僅含來源影像不確定性，數值見comparison.json。
