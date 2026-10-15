# 平移增強模型：飛機 stage-aligned 評估

| Stage | 位移 | Cluster agreement | 平移前後 token cosine similarity | Cluster margin | 平移後分類 accuracy |
|---|---:|---:|---:|---:|---:|
| Stage 0 | 8 px | 95.16% | 0.9958 | 0.1409 | 97.66% |
| Stage 1 | 16 px | 93.28% | 0.9891 | 0.1042 | 97.34% |
| Stage 2 | 32 px | 93.28% | 0.9845 | 0.1111 | 98.13% |

160 張 validation airplanes，四方向合併，每列 640 筆配對。Cluster margin 為平移後 normalized token 對最近與次近 Euclidean centroid 距離差的平均。取樣使用修正後 annotation 中心；使用者提供的舊表使用舊座標，非完全相同取樣協定。
