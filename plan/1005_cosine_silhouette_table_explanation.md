# 1005 K-means 評估表格：Cosine Silhouette 計算與判讀

> 2026-10-05 座標稽核更新：既有抽特徵流程使用 Caltech-101 的相對輪廓座標，漏加 bounding-box 偏移。本文件數值對應原取樣位置；不能直接作為物體質心或物體部件分群的結論。修正座標後需重新抽取特徵與 fitting。詳見 `plan/1005_caltech101_annotation_coordinate_audit.md` 與代表影像圖庫。

本文件說明下列表格的欄位、計算流程與判讀範圍。每個數字都是 **cosine silhouette 的平均值**，衡量同群樣本是否相近，以及不同群樣本是否分開；不是正確率，也不是平移同群比例。

| Stage | 現有 K-means | 手寫球面 K-means | FAISS 球面 K-means |
|---|---:|---:|---:|
| 0 | 0.1996 | 0.2024 | 0.2010 |
| 1 | 0.1320 | 0.1339 | 0.1324 |
| 2 | 0.0903 | 0.1000 | 0.1012 |

## 1. 表格標示

### Stage

Stage 0、1、2 代表 MergingViT 不同階段輸出的特徵。每張圖片依物體標註質心，選取該 stage 對應位置的 token feature。三個 stages 分別訓練分群中心、分別評估，不將不同 stages 的向量混在一起。

### 三種方法

| 方法 | 樣本處理 | 中心更新 | 表格使用的初始化 |
|---|---|---|---|
| 現有 K-means | 每筆特徵 L2 normalize | 群內算術平均，保留中心長度 | sklearn K-means++ |
| 手寫球面 K-means | 每筆特徵 L2 normalize | 群內算術平均後，再將中心 L2 normalize | 與配對 Euclidean 版本共享的 K-means++ 初始中心 |
| FAISS 球面 K-means | 每筆特徵 L2 normalize | 使用 `spherical=True`，每輪正規化中心 | FAISS 自身 random 初始化 |

表格中的 FAISS 欄是 `faiss_native`，不是 `faiss_shared_init`。共享初始化版本另列於完整評估報告；本次共享初始化的結果與手寫球面版本一致。

共同設定：

- K = 31。
- Seeds = 42、43、44。
- 每個 seed、stage、方法執行 50 次初始化。
- 每個 seed 依該方法的 **training objective** 選最佳中心，不以 held-out silhouette 或平移穩定性選模型。
- 使用既有 Caltech-101 letterbox160 checkpoint。
- 中心由 6,907 張未平移 training images 建立。
- 本表使用 1,770 張全類別、未平移的 held-out images 評估。

## 2. 準備評估特徵與分群標籤

對某個 stage，每張 held-out 圖片提供一個特徵向量 $x_i$，先做 L2 normalization：

$$
z_i = \frac{x_i}{\|x_i\|_2}.
$$

接著用各方法已訓練好的固定中心，得到每個樣本的分群標籤。評估時不更新中心，也不重新分群。

現有方法使用 Euclidean assignment；球面方法在樣本與中心均為 unit norm 時，Euclidean assignment 與最大 cosine similarity assignment 等價。

同一個 stage 中，各方法使用完全相同的 held-out features。因此樣本間的 cosine distance 矩陣相同，差異來自各方法產生的分群標籤。

這裡不使用影像類別標籤計算 silhouette；類別 AMI／ARI 是另外的評估指標。

## 3. 計算樣本間的 Cosine Distance

對兩個已正規化的特徵：

$$
d(i,j) = 1 - z_i^\top z_j.
$$

| 特徵方向關係 | Cosine similarity | Cosine distance |
|---|---:|---:|
| 相同方向 | 1 | 0 |
| 互相垂直 | 0 | 1 |
| 相反方向 | -1 | 2 |

距離越小，表示特徵方向越接近。實作上建立 $1770 \times 1770$ 的距離矩陣，並將自身距離的對角線設為 0。

**Silhouette 使用樣本與樣本之間的距離，不是樣本到 centroid 的距離。**

## 4. 每個樣本的群內平均距離 $a_i$

令 $C_i$ 為樣本 $i$ 所屬的群。計算它到同群其他樣本的平均距離：

$$
a_i = \frac{1}{|C_i|-1}
\sum_{\substack{j\in C_i\\j\ne i}} d(i,j).
$$

- 排除樣本自身，避免距離 0 拉低平均。
- $a_i$ 越小，表示樣本與自己的群越接近。
- 本表的 $C_i$ 是同一批 held-out 樣本中，被固定中心指派至同群的樣本集合；不是 training 樣本集合。

## 5. 最近其他群的平均距離 $b_i$

對每個其他群 $C$，先計算樣本 $i$ 到該群所有樣本的平均距離，再選取最小值：

$$
b_i = \min_{C\ne C_i}
\left[\frac{1}{|C|}\sum_{j\in C} d(i,j)\right].
$$

$b_i$ 對應「平均而言最接近自己的另一群」。它不是最近其他樣本的距離，也不是最近其他 centroid 的距離。

$b_i$ 越大，表示即使最接近的其他群，仍與自己有較大距離。

## 6. 每個樣本的 Silhouette $s_i$

$$
s_i = \frac{b_i-a_i}{\max(a_i,b_i)}.
$$

| 數值 | 判讀 |
|---|---|
| 接近 1 | 與自己群很接近，與其他群明顯分開 |
| 接近 0 | 自己群與最近其他群差不多近 |
| 小於 0 | 平均而言，另一群比自己群更接近 |

Silhouette 的範圍為 $[-1,1]$。只有一個樣本的群，該樣本 silhouette 按慣例設為 0。若 $a_i=b_i=0$，實作將分數設為 0；整個 partition 必須有至少兩個群，且群數小於樣本數，才有一般 silhouette 的定義。

### 數值範例

假設某個樣本到自己群的平均距離為：

$$
a_i = 0.20.
$$

它到其他三個群的平均距離分別為 0.25、0.40、0.55，因此：

$$
b_i = \min(0.25,0.40,0.55)=0.25.
$$

該樣本的 silhouette 為：

$$
s_i = \frac{0.25-0.20}{0.25}=0.20.
$$

對這個樣本而言，群內平均距離比最近其他群的平均距離小 20%。但整體平均 silhouette 是許多樣本比例值的平均，不能直接當作一個統一的距離比例。

## 7. 從逐樣本分數得到表格數字

每個 seed 會選出一組最佳中心，然後產生 1,770 個 silhouette 值。先平均該 seed 的樣本分數：

$$
S_r = \frac{1}{1770}\sum_{i=1}^{1770}s_{i,r},
\qquad r\in\{42,43,44\}.
$$

最後取三個 seeds 的平均：

$$
\bar S = \frac{S_{42}+S_{43}+S_{44}}{3}.
$$

表格顯示 $\bar S$，四捨五入至小數點後四位。

例如 Stage 2、FAISS 球面的 **0.1012**，就是三個 seeds 各自計算全部 held-out 樣本的平均 silhouette，再對三個平均值取平均。

**50 次初始化沒有直接平均進表格。**每個 seed 先從 50 次初始化中依 training objective 選一組最佳中心，只有該組中心進入上述評估。

實際計算流程：

```text
每個 stage 的同一批 held-out features
→ 每筆 feature L2 normalize
→ 建立完整 cosine distance 矩陣
→ 載入某方法、某 seed 已選定的中心
→ 用固定中心取得分群標籤
→ 對每筆樣本計算 silhouette
→ 平均 1,770 筆分數
→ 平均 seeds 42、43、44 的結果
→ 顯示四位小數
```

## 8. 如何判讀本次結果

### 同一個 Stage 內比較方法

相同 stage 使用同一批特徵與同一套 cosine distance，因此可以比較各方法形成的方向空間分群。

Stage 2 的結果為：

- 現有 K-means：0.0903。
- 手寫球面 K-means：0.1000。
- FAISS 球面 K-means：0.1012。

球面法的分數較高，表示本次 held-out 特徵在 cosine 距離下，具有較好的群內集中與群間分離。原始未四捨五入分數的配對差值：

| Stage 2 比較 | Silhouette 平均差值 | 影像 bootstrap 95% CI |
|---|---:|---|
| 手寫球面 − 現有 K-means | +0.00976 | [+0.00727, +0.01230] |
| FAISS native − 現有 K-means | +0.01094 | [+0.00857, +0.01335] |

這些差值使用原始分數計算，因此可能與表格四位小數直接相減略有不同。

CI 先平均配對 seeds 的逐影像差值，再以 held-out 影像為單位重抽樣 5,000 次。過程固定中心、距離矩陣與逐樣本 silhouette，沒有重新分群；它反映這批影像平均分數的抽樣不確定性，不涵蓋更多 seeds、重新訓練模型或改變 reference set 的不確定性。

Stage 0、1 的球面法相對現有 sklearn，其 silhouette 差值 CI 仍涵蓋 0，沒有同樣明確的改善證據。

### 跨 Stage 比較

Stage 0、1、2 的特徵空間不同。Stage 0 分數較高，表示它在自己的空間中較容易形成這種幾何分群；不能直接推論 Stage 0 比 Stage 2 更有語義辨識能力。

### 不能從表格得到的結論

- 0.1012 不是 10.12% 正確率。
- Silhouette 較高不保證平移同群比例較高。
- Silhouette 較高不保證群具有物體或部件語義。
- 本次固定 cosine 距離，是針對方向特徵空間的評估選擇；不代表方法在所有距離幾何下都較好。
- Held-out split 曾用於原模型的 checkpoint 選擇，並非完全未使用過的最終 test。

因此 silhouette 應與群使用分布、跨 seed ARI、類別 AMI／ARI、平移診斷一起判讀，而非作為唯一評分。

## 9. 與其他圖表標示的區別

- 本文簡表：只有三個 seeds 的平均 silhouette，不含標準差。
- `quality_comparison.png` 上排：平均 silhouette，誤差棒為三個 seeds 的樣本標準差，不是 bootstrap CI。
- `quality_comparison.png` 下排：樣本到其實際所屬中心方向的平均 cosine distance，越低越好；不是 silhouette。
- `coverage_seed_stability.png`：有效群數與跨 seeds 的 ARI，評估群使用與初始化穩定性。

## 10. 實作、產物與來源

實作：[`scripts/evaluate_kmeans_quality.py`](../scripts/evaluate_kmeans_quality.py)。held-out 評估使用完整距離矩陣及 sklearn `silhouette_samples(..., metric="precomputed")`，沒有抽樣近似。

Training silhouette 則使用固定 2,000 個 query 樣本，對全部 6,907 個 training 樣本計算每個 query 的精確 silhouette；**本文表格沒有使用這組 training 結果**。

數值已與 sklearn 直接使用 `metric="cosine"` 的計算核對。

相關產物：

- [完整品質評估報告](1005_kmeans_quality_evaluation_results.md)。
- [逐方法、逐 seed 的幾何指標](../plots/kmeans/quality_evaluation_1005/geometry_metrics.csv)。
- [逐影像指標](../plots/kmeans/quality_evaluation_1005/heldout_sample_metrics.csv)。
- [配對差值與信賴區間](../plots/kmeans/quality_evaluation_1005/heldout_paired_differences.csv)。
- [品質比較圖](../plots/kmeans/quality_evaluation_1005/quality_comparison.png)。

指標定義：[scikit-learn Silhouette Samples 官方文件](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_samples.html)。
