# 案例分析重點

每模型皆使用同樣的來源影像、四方向及 stage-aligned 位移。每張影像有 3 stages × 4 directions = 12 筆配對，並非 12 張獨立影像。
換群表示同一模型／stage 的離散 codebook 指派改變；不直接等同語意改變。

| Model | Class | Pairs | 換群／預測改變 | 不換群／預測改變 | 換群／預測不變 | 不換群／預測不變 |
|---|---|---:|---:|---:|---:|---:|
| baseline | airplanes | 1920 | 45 | 357 | 130 | 1388 |
| baseline | chandelier | 264 | 27 | 166 | 3 | 68 |
| baseline | brain | 240 | 29 | 121 | 16 | 74 |
| augmented | airplanes | 1920 | 4 | 28 | 113 | 1775 |
| augmented | chandelier | 264 | 12 | 90 | 15 | 147 |
| augmented | brain | 240 | 1 | 39 | 18 | 182 |

## 值得先看的案例

- [chandelier #3876: cluster_changed__prediction_same](augmented/chandelier/stage0/cluster_changed__prediction_same/case_01_source_3876.png)：cluster 25→21，cosine 0.99696。
- [brain #3198: cluster_same__prediction_changed](augmented/brain/stage0/cluster_same__prediction_changed/case_01_source_3198.png)：cluster 27→27，cosine 0.99808。
- [chandelier #3911: cluster_changed__prediction_changed](augmented/chandelier/stage0/cluster_changed__prediction_changed/case_01_source_3911.png)：cluster 26→25，cosine 0.99619。
- [chandelier #3921: cluster_same__prediction_same](augmented/chandelier/stage0/cluster_same__prediction_same/case_01_source_3921.png)：cluster 18→18，cosine 0.99834。

## 可支持的解讀

1. 換群不必然改變預測；不換群也不保證預測不變。中心 token 無法單獨代表整張影像的決策。
2. 高 cosine 仍可能跨越 K-means 指派邊界；案例不能只用 cosine 或群編號判斷語意穩定性。
3. 兩組模型的群中心分別 fitting，因此只比較各模型自己的前後指派，不能把不同模型相同編號當成相同概念。
4. 局部 crop 逐像素相同，仍可能因位置編碼、self-attention 與後續處理改變 activation；目前未隔離機制。
5. 另外兩類是依 validation 表現挑選的探索性案例；單一 seed，沒有全類別或 final-test 推論。

[全部案例與分 stage 計數](REPORT.md) · [Mermaid 流程](../../../../plan/1014_aligned_translation_workflow.md)