# 0928 小專案規劃：本作 K-means 與頻譜 K-means 差異可視化

## 一、專案目的

建立一個獨立、可重現的小型互動式專案，用視覺化方式比較：

1. 本研究目前採用的 **L2-normalized feature Euclidean K-means**；
2. 先建立 affinity graph、進行 graph Laplacian embedding，再執行 K-means 的 **spectral K-means**。

此專案的目的不是取代主研究 pipeline，也不是宣稱 spectral clustering 一定優於本作方法，而是讓讀者直觀看到兩者在下列方面的差異：

- 分群所使用的空間不同；
- 本作方法偏好原始特徵空間中的近似凸形群集；
- spectral 方法能利用局部鄰接關係分開非凸形流形；
- affinity graph、鄰居數與 kernel bandwidth 會顯著影響 spectral 結果；
- 本作實際的 per-position 設計與一般全域 spectral clustering 的樣本組織方式不同。

本文件先完成專案設計，後續再交由子 agent 依工作包實作。

## 二、名詞與比較邊界

### 2.1 本作 K-means

本作方法定義為：

\[
\hat{x}_i=\frac{x_i}{\|x_i\|_2},
\qquad
\min_{\{C_k,\mu_k\}}
\sum_{k=1}^{K}\sum_{i\in C_k}
\|\hat{x}_i-\mu_k\|_2^2.
\]

實作使用 `sklearn.cluster.KMeans`。輸入樣本會先 L2 normalization，但 centroid 更新後不會重新投影至單位球面，因此專案介面與說明一律稱為：

> Normalized-feature Euclidean K-means

不得標示成 spherical K-means 或 cosine K-means。

### 2.2 頻譜 K-means

本專案中的 spectral 方法定義為：

1. 由樣本特徵建立 affinity matrix $W$；
2. 計算 degree matrix $D$；
3. 建立 normalized graph Laplacian；
4. 取前 $K$ 個相關 eigenvectors 形成 spectral embedding；
5. 對 embedding 的 row vectors 執行標準 K-means。

預設採用 normalized Laplacian：

\[
L_{sym}=I-D^{-1/2}WD^{-1/2}.
\]

專案中「spectral K-means」指上述 pipeline，不得與 spherical K-means 混用。

### 2.3 公平比較原則

兩種方法必須使用：

- 完全相同的樣本；
- 相同的 $K$；
- 相同的 random seed；
- 相同的 train/test 或取樣範圍；
- 相同的顏色映射規則，但 cluster label 必須先做 Hungarian matching 才能並排比較。

主畫面不得只展示對 spectral clustering 有利的資料。至少需要同時包含凸形與非凸形案例。

## 三、預定專案位置與結構

專案預定建立於：

```text
experiments/kmeans_spectral_visualizer/
├── README.md
├── app.py
├── algorithms.py
├── datasets.py
├── metrics.py
├── plotting.py
├── feature_loader.py
├── requirements.txt
├── configs/
│   └── default.yaml
├── tests/
│   ├── test_algorithms.py
│   ├── test_datasets.py
│   └── test_metrics.py
└── outputs/
    └── .gitkeep
```

初版以 Streamlit 實作互動介面；核心演算法與繪圖不可寫死在 `app.py`，必須可由測試或命令列獨立呼叫。

如果 Streamlit 依賴與現有環境衝突，允許改成「Python CLI 產生靜態 HTML/PNG 報告」，但需在實作前於交接紀錄說明原因。

## 四、資料設計

### 4.1 合成資料

第一版必須支援下列資料：

| 資料類型 | 用途 | 預期觀察 |
|---|---|---|
| Gaussian blobs | 驗證兩種方法在凸形群集上的基本行為 | 兩者應有相近結果 |
| Unequal-density blobs | 顯示 centroid-based clustering 對密度與尺度差異的反應 | 兩者可能受不同超參數影響 |
| Two moons | 顯示非凸形流形 | spectral 通常較容易沿流形分開 |
| Concentric circles | 顯示中心距離無法直接分開的結構 | spectral 可利用 graph connectivity |
| Bridge/noise case | 顯示 graph 被少量橋接點連接的風險 | spectral 可能對鄰居與 bandwidth 敏感 |

所有資料產生器必須接受 `seed`、`n_samples`、`noise` 與必要的形狀參數。

### 4.2 特徵範數案例

加入方向相似但 norm 不同的二維或三維資料，並並排顯示：

1. raw Euclidean K-means；
2. 本作 normalized-feature Euclidean K-means；
3. spectral K-means。

此案例專門解釋 L2 normalization 排除樣本 norm 差異的效果，但不得將 normalized Euclidean K-means 標成嚴格 spherical K-means。

### 4.3 真實 MergingViT 特徵

第二階段才接入本專案既有 feature：

- 選定 checkpoint；
- 固定一個 `(stage, block, position)`；
- 從 train split 取得跨影像 feature；
- 以 PCA 或 UMAP 僅供二維顯示；
- clustering 必須在指定的原特徵空間或 spectral embedding 執行，不能直接在二維視覺化投影上分群，除非明確標成額外消融。

真實特徵載入器不得複製整套模型程式，應重用現有 `MergingViT` 與 feature extraction 邏輯，或讀取預先匯出的 `.npz`。

## 五、兩種演算法的實作規格

### 5.1 Normalized-feature Euclidean K-means

流程：

```text
X
→ 對每筆非零樣本做 L2 normalization
→ sklearn KMeans
→ labels、centroids、距離與 inertia
```

需顯示：

- 正規化前後的樣本 norm 分布；
- centroid 位置；
- centroid norm；
- decision regions；
- inertia；
- 每個 cluster 的樣本數。

### 5.2 Spectral K-means

至少支援兩種 affinity：

1. `k-nearest-neighbor` graph；
2. RBF affinity。

KNN graph 需可選擇是否 mutual，並清楚顯示 graph 是否連通。RBF affinity 定義為：

\[
W_{ij}=\exp\left(-\frac{\|x_i-x_j\|_2^2}{2\sigma^2}\right).
\]

流程：

```text
X
→ affinity matrix W
→ degree matrix D
→ normalized Laplacian / normalized affinity
→ eigen decomposition
→ 取 spectral embedding
→ row normalization（若所選演算法規格需要）
→ sklearn KMeans
→ labels
```

實作時必須在 README 明確記錄採用哪一個標準變體，例如 Ng–Jordan–Weiss，避免只寫模糊的「spectral clustering」。可以用 sklearn 結果作為參考，但核心中間產物 $W$、eigenvalues 與 embedding 必須能輸出，才能完成教學視覺化。

## 六、互動介面規格

### 6.1 Sidebar 控制項

- dataset 類型；
- `n_samples`；
- noise；
- cluster 數 $K$；
- random seed；
- K-means `n_init`；
- affinity 類型；
- KNN 的 `n_neighbors`；
- RBF 的 `sigma` 或 `gamma`；
- 是否先做 L2 normalization；
- 是否顯示 graph edges；
- 是否對 cluster labels 做 Hungarian matching。

### 6.2 主畫面

主畫面至少包含六個 panel：

1. **Input space**：原始資料與 ground truth（若合成資料有真值）；
2. **Normalized feature space**：L2 normalization 後資料；
3. **本作 K-means 結果**：labels、centroids 與 decision regions；
4. **Affinity graph**：節點、邊與連通分量；
5. **Spectral embedding**：eigenvector coordinates 與 eigenvalues；
6. **Spectral K-means 結果**：回投到原始輸入空間顯示 labels。

另提供一個差異 panel：

- 以外框或特殊符號標出兩種方法 label 不一致的樣本；
- 顯示 disagreement rate；
- 顯示兩種方法各自的 silhouette、ARI、NMI；
- 若無 ground truth，ARI/NMI 改為兩種方法之間的一致性，不得誤標成 accuracy。

### 6.3 動畫模式

非必要但建議加入逐步動畫：

```text
原始點
→ L2 normalization
→ 建立 affinity edges
→ spectral embedding 展開
→ K-means assignment
```

動畫必須可以關閉，靜態圖仍需完整表達結果。

## 七、評估指標

合成資料有 ground truth 時，報告：

- Adjusted Rand Index；
- Normalized Mutual Information；
- silhouette score；
- cluster size distribution；
- 執行時間；
- spectral graph connected components；
- eigengap。

真實 feature 沒有 cluster ground truth 時，報告：

- silhouette score；
- cluster compactness；
- cluster size entropy；
- 跨 random seed stability；
- 兩種方法的 AMI／ARI agreement；
- 代表 patch 的人工語意一致率，作為後續選配。

silhouette 必須註明在哪個空間與何種距離下計算，避免把 spectral embedding 的指標與原始 feature space 指標直接混用。

## 八、必須呈現的核心案例

### Case A：凸形群集

Gaussian blobs 上兩種方法應大致一致，用來說明 spectral 方法不是所有情況都必然更好。

### Case B：非凸形群集

Two moons 或 concentric circles 上，本作 K-means 依 centroid 切割，spectral 方法則依 graph connectivity 分群。

### Case C：圖超參數敏感性

固定資料，只調整 `n_neighbors` 或 `sigma`：

- graph 太稀疏時產生多個 disconnected components；
- graph 太密時不同流形被錯誤連接；
- 合理區間內才能呈現預期群集。

### Case D：特徵範數差異

顯示 raw Euclidean K-means 與 normalized-feature Euclidean K-means 的差異，並說明本作正規化排除的是 norm 影響，但不等於已實作 spherical centroid update。

### Case E：MergingViT 固定位置特徵

固定 `(stage, block, position)` 比較兩種方法，並搭配 centroid-nearest patches 或 cluster representatives。此案例只在合成資料版本通過驗收後實作。

## 九、測試規格

### 9.1 單元測試

- L2 normalization 後，所有非零樣本 norm 接近 1；
- 零向量不產生 NaN；
- affinity matrix 對稱且非負；
- degree 與 Laplacian shape 正確；
- normalized Laplacian 對稱；
- eigenvalues 依序排列，數值誤差範圍內非負；
- 固定 seed 時輸出可重現；
- (K>N) 時提供清楚錯誤或自動降 K，行為需在 README 固定；
- Hungarian matching 不改變 partition，只改變 label permutation；
- 指標在退化 cluster 或 disconnected graph 時有明確處理。

### 9.2 視覺回歸檢查

- 所有 panel 的色彩標籤一致；
- cluster label 經 matching 後再比較；
- centroid 與 graph edge 不遮蔽主要樣本；
- 圖例清楚區分 ground truth、本作結果與 spectral 結果；
- 高維真實特徵的二維圖明確標示「projection for visualization only」。

## 十、驗收標準

第一階段 MVP 完成條件：

1. 可用單一指令啟動；
2. 支援 blobs、two moons、circles 與 norm-variation 四種資料；
3. 可並排顯示本作 K-means 與 spectral K-means；
4. 可顯示 affinity graph、spectral embedding 與 disagreement samples；
5. 可調整 (K)、seed、KNN neighbors 與 RBF 參數；
6. 所有核心演算法有單元測試；
7. README 清楚區分 normalized Euclidean、spectral 與 spherical K-means；
8. 至少保存四組固定 seed 的範例輸出；
9. `git diff --check` 與測試皆通過；
10. 不修改現有訓練與主要 K-means pipeline 的行為。

第二階段完成條件：

1. 可載入一組 MergingViT 真實 feature；
2. 可指定 `(stage, block, position)`；
3. clustering 與二維顯示投影明確分離；
4. 可顯示每群代表 patches；
5. 可匯出 PNG、CSV 與包含設定的 JSON manifest。

## 十一、後續子 agent 工作包

待開始開發時，主 agent 將工作拆成以下可平行任務：

### Agent A：核心演算法與測試

- 實作 normalized-feature Euclidean K-means；
- 實作 affinity、Laplacian、spectral embedding 與 K-means；
- 完成核心數學單元測試；
- 不負責 UI。

### Agent B：合成資料與視覺化

- 實作資料產生器；
- 實作六個核心 panels；
- 實作 Hungarian label matching 與 disagreement 標示；
- 產生固定 seed 範例。

### Agent C：互動介面與文件

- 建立 Streamlit app；
- 串接 sidebar 控制項；
- 撰寫 README、啟動方式與方法警語；
- 實作輸出下載。

### 主 agent：整合與驗收

- 先定義共用函式介面，避免 agents 修改同一檔案；
- 整合三個工作包；
- 核對數學定義、命名與圖表敘事；
- 執行完整測試與人工視覺驗收；
- 第二階段再安排真實 MergingViT feature adapter。

子 agent 開發前，主 agent需先確認工作樹狀態，並為每個 agent 指派互不重疊的檔案範圍。共享介面應先固定為：

```python
run_normalized_kmeans(X, n_clusters, seed, n_init) -> ClusteringResult
build_affinity(X, method, **params) -> AffinityResult
run_spectral_kmeans(X, n_clusters, affinity_config, seed) -> SpectralResult
align_labels(reference, predicted) -> AlignmentResult
compute_metrics(X, labels, truth=None, space_name="input") -> dict
```

## 十二、風險與限制

1. Spectral clustering 的時間與記憶體成本通常為 (O(N^2)) affinity storage，真實特徵只能先使用子樣本或 sparse KNN graph。
2. 二維投影可能扭曲高維幾何，UI 必須明確標示 projection 僅用於顯示。
3. Synthetic moons/circles 天生有利於 spectral clustering，因此必須同時展示 blobs 與 graph failure cases。
4. 不同 spectral variants 的 Laplacian、eigenvector 選取與 row normalization 不完全相同，README 必須鎖定實作版本。
5. Cluster label 本身沒有語意順序，任何逐點差異比較前都必須 label matching。
6. Spectral clustering 產生較好幾何分群，不代表對分類模型具有較高 faithfulness；後續需另做 concept intervention 才能回答。

## 十三、預定交付物

- 可執行的互動式視覺化小專案；
- 核心演算法與測試；
- 方法比較 README；
- 合成資料固定案例圖；
- graph 與 spectral embedding 圖；
- 逐案例 metrics CSV；
- 可重現設定 JSON/YAML；
- 第二階段的 MergingViT 真實特徵展示。
