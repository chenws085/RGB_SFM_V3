# 0928 未來研究規劃：K-means 方法定位與階層概念分析

## 一、研究定位

參考 `plan/0928_related.txt` 所整理的 DINO、Deep Spectral Methods、TokenCut 與 STEGO，本研究的 K-means 分析不以產生前景遮罩或語意分割圖為主要目的，而是作為已訓練 MergingViT 的 post-hoc 解釋工具，用來觀察模型在不同 stage、空間位置及 attention head 中形成的跨影像概念。

本研究與既有方法最核心的差異是分群軸向：

> DINO、Deep Spectral Methods 與 TokenCut 多在單張影像內比較不同空間位置的 token；本研究則固定 stage 與空間位置，沿資料集維度比較不同影像在該位置產生的 token feature。

令第 $i$ 張訓練影像在 stage $s$、位置 $p$ 的 token feature 為 $x_{i,p}^{s}$，本研究對以下集合分群：

\[
X_p^s=\{x_{1,p}^s,x_{2,p}^s,\ldots,x_{M,p}^s\}.
\]

因此，每個 `(stage, position)` 都有獨立的 K-means 模型；在 head 模式下，則進一步為每個 `(stage, position, head)` 建立獨立模型。

## 二、與頂會相關工作的差異

| 比較面向 | DINO | Deep Spectral Methods | TokenCut | STEGO | 本研究 |
|---|---|---|---|---|---|
| 主要目的 | 發現物體及語意部件 | 無監督物體分割 | 顯著前景／背景分割 | 全資料集語意分割 | 解釋階層式分類模型的內部概念 |
| 模型來源 | 自監督 ViT | 自監督 ViT | 自監督 ViT | 自監督特徵與蒸餾網路 | 已訓練的監督式 MergingViT |
| 分群樣本 | 單圖內不同 patches | 單圖內不同 patches | 單圖內不同 patches | 全資料集、跨位置 patches | 全資料集中相同位置的 patches |
| 特徵來源 | 最後一層 attention Key | 圖拉普拉斯特徵向量 | Fiedler vector | 非線性 projector 輸出 | 各 stage/block 的 token 或單一 head 輸出 |
| 前處理 | L2 normalization | affinity graph 與 spectral embedding | normalized cut | 特徵蒸餾與正規化 | L2 normalization，未做圖分解 |
| 分群方式 | 小 K 的 cosine-oriented clustering | 低維 Euclidean K-means | 固定二分 | dataset-wide codebook | per-position、per-stage、可 per-head 的 K-means |
| 空間位置處理 | 單圖各位置共同分析 | 空間關係進入 affinity graph | 空間關係進入 graph cut | 不同位置共用全域 codebook | 不同位置使用獨立 codebook |
| 跨層關係 | 無 | 無 | 無 | 無 | 使用真實 patch-merging mapping 回溯 |
| 主要輸出 | 物體／部件區域 | 分割遮罩 | 二值前景遮罩 | 語意分割圖 | cluster 代表 patch、重要 token 與跨層概念路徑 |

### 2.1 相較 DINO

DINO 類方法著重單張影像內部的 token 差異，其 cluster 通常對應該影像中的背景、物體或物體部件；本研究則比較不同影像在固定位置上的表徵，cluster 表示模型在該位置反覆形成的典型特徵。

DINO 偏重最後一層 attention Key，本研究則分析多個 stage 的 block 輸出 token，也能使用 attention 中各 head 的加權 value 輸出。因此，本研究的問題不是「這張圖可切成哪些區域」，而是「階層分類模型在各層及各位置學到了哪些概念」。

### 2.2 相較 Deep Spectral Methods

Deep Spectral Methods 先建立單圖 patch affinity graph，再以圖拉普拉斯特徵向量得到低維 spectral embedding，最後進行 K-means。其設計將空間連續性與特徵相似度納入分割。

本研究不建立 affinity matrix，也不進行 eigendecomposition，而是直接在 MergingViT 的 L2-normalized 中間特徵上分群。前者著重分割邊界及區域平滑性；本研究著重跨影像概念的一致性、代表樣本及概念在模型階層中的形成方式。

### 2.3 相較 TokenCut

TokenCut 將單張影像投影到一維 Fiedler vector，並固定分成前景與背景兩群。本研究不限定 $K=2$，也不預設 cluster 必須對應前景或背景，而是在高維 token/head 特徵空間中建立多個群集。

本研究的 cluster 可能表達顏色、紋理、局部形狀、物體部件或高階分類線索，其用途是模型內部概念分析，不是顯著物體偵測。

### 2.4 相較 STEGO

STEGO 與本研究都屬於 dataset-wide clustering，但 STEGO 將不同影像、不同位置的特徵放入共享的全域 codebook，以取得跨位置一致的語意類別。其 codebook 可表示道路、汽車或天空等全域語意。

本研究則保留位置條件，為每個 stage 與位置建立：

\[
C_{s,p}=\{\mu_{s,p,1},\ldots,\mu_{s,p,K_s}\},
\]

head 模式再擴充為 $C_{s,p,h}$。這可細緻分析特定位置與 head 的行為，但同時表示不同位置的 cluster 編號不具有天然的共同語意。

STEGO 的 clustering 是無監督語意分割系統的一部分；本研究的 clustering 則是分類模型訓練完成後的解釋分析，並會配合 MergingViT 的真實 patch-merging 關係研究概念如何逐層組合。

## 三、本研究目前的特色與限制

### 3.1 特色

1. **位置條件式跨影像分群**：固定位置後比較不同影像，而非在單圖內切割不同區域。
2. **多階層分析**：同時分析 MergingViT 各 stage，而非只使用最後一層特徵。
3. **Head-level 分析**：可對每個 `(position, head)` 獨立分群，觀察 head specialization。
4. **真實合併關係**：依模型的 patch-merging mapping，將高階 token 回溯到實際低階來源。
5. **代表樣本視覺化**：以距離 centroid 最近的訓練 patch 說明每個 cluster，而不只輸出 cluster label。
6. **分類解釋整合**：結合 GradCAM 選取重要高階 token，再追蹤相關 cluster 與低階來源。

### 3.2 限制

1. 每個位置各自建立 K-means，因此 `position A, cluster 0` 與 `position B, cluster 0` 不能直接視為同一概念。
2. 不同 stage 的 cluster label 也沒有直接對應關係，必須使用 merge transition、代表特徵相似度或共同出現關係建立連結。
3. 保留絕對位置可能使 cluster 同時編碼視覺內容與位置先驗，需要 controlled-position 與平移實驗拆解兩者影響。
4. K 值為人工設定；需驗證結果是否對 K、random seed 及初始化穩定。
5. 目前 cluster 主要提供相關性解釋，不能僅由代表圖宣稱該概念對分類具有因果作用。
6. 位置獨立建模會產生大量 K-means 模型，計算成本及概念對齊難度高於全域 codebook。

## 四、距離度量的論文措辭

目前實作先對每筆特徵做 L2 normalization，再使用 scikit-learn 的 Euclidean K-means。對單位向量而言：

\[
\|\hat{x}_i-\hat{x}_j\|_2^2=2-2\cos(\hat{x}_i,\hat{x}_j),
\]

因此，兩個都位於單位球面上的**樣本向量**之間，Euclidean distance 與 cosine distance 單調對應。不過這個推導不能直接證明標準 Euclidean K-means 與 spherical K-means 的完整聚類程序一致，因為 K-means 還包含 centroid 更新與樣本至 centroid 的重新指派。

### 4.1 原始碼查核結果

目前主流程實際採用的是「L2-normalized samples + scikit-learn Euclidean KMeans」，而不是 spherical K-means：

1. `Kmeans_analysis_padding_repr.py` 先以 `normalize(X, norm='l2', axis=1)` 將每筆樣本正規化。
2. 接著直接呼叫 `KMeans(...).fit(X_norm)`；專案依賴版本為 `scikit-learn==1.7.0`。
3. 沒有任何程式在每次 M-step 後執行 `normalize(kmeans.cluster_centers_)` 並將結果寫回模型。
4. 訓練完成後的代表圖距離與 inference `predict` 都直接使用原始 `kmeans.cluster_centers_`。
5. `scripts/test_caltech_translation_kmeans.py` 第 394 行雖然另行正規化 centroid，但只用於計算報告中的 cosine 指標；cluster fitting、`predict` 與 `transform` 仍使用未正規化的 sklearn centroids。

scikit-learn 的標準 K-means 將中心更新為群內樣本的算術平均。即使每個樣本 $x_i$ 都滿足 $\|x_i\|_2=1$，其平均值

\[
\mu_k=\frac{1}{|C_k|}\sum_{i\in C_k}x_i
\]

通常仍滿足 $\|\mu_k\|_2<1$。標準 K-means 的指派距離為：

\[
\|x-\mu_k\|_2^2=1+\|\mu_k\|_2^2-2x^\top\mu_k.
\]

其中包含會因 cluster 而異的 $\|\mu_k\|_2^2$；spherical K-means 則會將中心正規化成 $\hat{\mu}_k=\mu_k/\|\mu_k\|_2$，再依 $x^\top\hat{\mu}_k$ 指派。因此，只有在所有 centroid 也保持單位長度等附加條件下，才可把兩者視為相同；目前實作不滿足此條件。

### 4.2 現有模型的實測證據

檢查專案既有 K-means cache，centroid norm 並非 1：

| Cache | Centroid norm 範圍 | 平均 norm | 單位中心數 |
|---|---:|---:|---:|
| Colorful-MNIST Stage 0 | 0.6413–0.9991 | 0.8923 | 0/392 |
| Colorful-MNIST Stage 1 | 0.5707–0.9873 | 0.8405 | 0/128 |
| Colorful-MNIST Stage 2 | 0.4798–0.8111 | 0.6880 | 0/32 |
| Colorful-MNIST Stage 3 | 0.5490–0.7087 | 0.6323 | 0/8 |

這證明目前保存的中心不在 unit hypersphere 上，不能把現有結果稱為 spherical K-means 的結果。

論文暫定使用下列描述：

> We apply Euclidean K-means in an L2-normalized feature space, where pairwise Euclidean distance is monotonically related to cosine distance.

中文建議寫法為：

> 本研究先對每個特徵向量進行 L2 正規化，再使用標準歐氏距離 K-means 進行聚類。對任意兩個已正規化的樣本向量，平方歐氏距離與餘弦相似度具有單調對應關係。須注意，本實作的群中心為群內樣本的算術平均，更新後未重新投影至單位超球面，因此屬於 normalized-feature Euclidean K-means，而非嚴格的 spherical K-means。

若後續希望使用「spherical K-means」名稱，需改成每次中心更新後都重新正規化 centroid，並與目前方法進行消融比較。

### 4.3 第一篇論文第 3.3.2 節查核

第一篇論文「餘弦等價 K-means」一節屬於**部分正確，但核心結論過度延伸**。逐項判定如下：

| 原論文敘述 | 判定 | 說明 |
|---|---|---|
| 先對 feature 進行 L2 normalization | 正確 | 與目前原始碼一致 |
| 正規化後的非零樣本位於單位超球面 | 正確 | 零向量為例外 |
| 兩個單位向量的平方歐氏距離與 cosine similarity 單調等價 | 正確 | 應明確寫成平方範數 $\|u-v\|_2^2$ |
| 實作使用 Euclidean K-means | 正確 | 使用 `sklearn.cluster.KMeans` |
| 聚類結果與 spherical K-means 在數學上一致 | 不正確 | sklearn centroid 更新後沒有重新正規化 |
| 當 $K>M$ 時自動將有效 K 降為 M | 正確 | `_effective_n_clusters()` 有此處理 |
| 正規化實作為 $f/(\|f\|_2+\epsilon)$ | 與原始碼不完全一致 | 程式使用 sklearn `normalize()`，沒有直接在分母加上指定的 $\epsilon$ |

#### 4.3.1 原式（3.21）修正

原文必須將左側清楚排版成平方歐氏距離：

\[
\|u-v\|_2^2
=\|u\|_2^2+\|v\|_2^2-2u^\top v
=2-2\cos\theta.
\]

這個式子只證明兩個單位向量的成對距離具有單調對應，不能單獨證明包含 centroid 更新的完整 K-means 演算法與 spherical K-means 相同。

#### 4.3.2 原式（3.22）修正

目前原始碼使用 sklearn L2 normalization。若要忠實描述實作，應寫成：

\[
\hat f=
\begin{cases}
\dfrac{f}{\|f\|_2}, & \|f\|_2>0,\\[6pt]
0, & \|f\|_2=0.
\end{cases}
\]

不建議一方面將分母寫成 $\|f\|_2+\epsilon$，另一方面宣稱向量被嚴格投影至 unit hypersphere，因為加入 $\epsilon$ 後，非零向量的 norm 也不會精確等於 1。若論文一定要保留 $\epsilon$ 公式，程式亦應改成相同的明確實作，並將文字改為「近似單位長度」。

#### 4.3.3 原式（3.23）修正

建議用平方距離表示標準 K-means assignment，並補齊 stage、position 與 cluster 的 centroid 索引：

\[
\ell_n^{(i,b)}(m)
=\arg\min_{k\in\{0,\ldots,K_n^{(i,b)}-1\}}
\left\|
\hat F_{m,n}^{(i,b)}-\mu_{n,k}^{(i,b)}
\right\|_2^2.
\]

centroid 應定義為正規化樣本的算術平均：

\[
\mu_{n,k}^{(i,b)}
=\frac{1}{|C_{n,k}^{(i,b)}|}
\sum_{m\in C_{n,k}^{(i,b)}}
\hat F_{m,n}^{(i,b)},
\]

並註明 $\|\mu_{n,k}^{(i,b)}\|_2$ 不一定等於 1。平方與未平方距離的最近中心相同，但平方形式與標準 K-means objective 及 sklearn inertia 的定義一致。

#### 4.3.4 第一篇論文建議替換文字

建議將章節名稱由「餘弦等價 K-means」改成「正規化特徵空間中的 Euclidean K-means」，並將方法描述替換為：

> Transformer 的 patch feature 可能具有明顯的範數差異。若直接在原始特徵空間中使用歐氏距離，聚類結果可能同時受到特徵方向與向量範數影響。為降低範數差異的影響，本研究先對每個非零特徵向量進行 L2 正規化，再於正規化後的特徵空間中執行標準 Euclidean K-means。
>
> 對任意兩個單位向量 $u,v$，其平方歐氏距離滿足 $\|u-v\|_2^2=2-2u^\top v=2-2\cos\theta$。因此，正規化樣本之間的平方歐氏距離與餘弦相似度具有單調對應關係。須注意，本研究使用的標準 K-means 將各群中心更新為群內正規化樣本的算術平均，並未在每次更新後將中心重新投影至單位超球面。因此，本方法應視為 L2-normalized feature space 中的 Euclidean K-means，而非嚴格的 spherical K-means；兩者的聚類結果不保證完全一致。

原文「Transformer 的 patch feature 向量通常具有較高的範數變異性」屬於經驗性主張。正式論文應補上相關文獻，或報告本研究資料的 feature-norm 分布與 raw-feature K-means 消融；若缺乏證據，應改成較保守的「可能具有明顯的範數差異」。

## 五、由方法差異導出的後續實驗

### 5.1 Global codebook 與 per-position codebook 消融

比較三種設定：

1. 所有位置共享一組 global codebook，接近 STEGO 的 dataset-wide 設定；
2. 每個位置使用獨立 codebook，為本研究目前設定；
3. 加入位置條件但共享部分 prototypes 的折衷設定。

比較指標包括 cluster compactness、代表圖語意一致率、跨平移穩定性、分類 faithfulness，以及 codebook 數量與計算成本。此實驗可回答保留位置條件是否真的帶來更好的模型解釋。

### 5.2 Token feature、Key 與 head output 消融

參考 DINO，分別對下列特徵建立相同 K 與相同資料切分的 K-means：

- block 輸出 token；
- attention Key；
- attention Value；
- attention 加權後、projection 前的 individual head output。

比較 cluster compactness、代表圖一致率、跨 seed stability 及概念介入效果，確認目前選用的 block/head 輸出是否比 Key feature 更適合作為概念表示。

### 5.3 Spectral preprocessing 消融

參考 Deep Spectral Methods，在局部候選範圍建立 affinity graph 或加入低維 spectral/PCA embedding，再與直接 K-means 比較。重點不是複製分割流程，而是檢查目前 cluster 是否受到高維雜訊影響。

### 5.4 K 與初始化穩定性

在固定 checkpoint、資料 split 與特徵來源下，比較多個 K 與 random seeds，使用 AMI、ARI、centroid matching 和代表圖重疊率衡量穩定性。不同 run 的 cluster label 必須先以 Hungarian matching 對齊，不能直接比較 label 編號。

### 5.5 Controlled-position 實驗

將相同局部內容放入不同 token 位置，分離「內容變化」與「絕對位置」的影響。此實驗需與完整影像平移分開報告，因為完整影像平移同時改變背景、patch boundary 與 receptive-field composition。

### 5.6 階層概念圖

利用模型真實的 patch merging，建立 child cluster 到 parent cluster 的轉移關係：

\[
w(c_i^s\rightarrow c_j^{s+1})
=P(c_j^{s+1}\mid c_i^s).
\]

同時記錄支持樣本數、PMI、條件熵及跨 seed edge stability，避免把高頻但缺乏特異性的 cluster 誤認為重要組合。

### 5.7 概念介入與因果驗證

對 cluster 對應的 token 執行 zero removal、stage mean replacement、centroid replacement 與 background replacement，觀察目標類別 logit、accuracy、comprehensiveness 與 sufficiency 的變化。

除了單一 cluster，也需比較：

- 只介入低階 child concept；
- 只介入高階 parent concept；
- 介入完整 child-to-parent concept path；
- 介入相同 token 數量的隨機 baseline。

只有在概念介入穩定地優於隨機 baseline 後，才能將 cluster 從「相關表徵」提升為「對模型決策具有作用的概念」。

## 六、近期執行優先順序

| 優先級 | 工作 | 目的 | 完成條件 |
|---|---|---|---|
| P0 | 完成 global 與 per-position codebook 消融 | 確認本研究相對 STEGO 的核心設計是否必要 | 產出概念品質、平移穩定性與 faithfulness 對照表 |
| P0 | 完成 token／Key／head feature 消融 | 確認特徵選取依據 | 相同資料、K、seed 下完成四種特徵比較 |
| P0 | 建立跨 stage cluster transition matrix | 將獨立 stage clusters 連成概念形成路徑 | 產出 Stage 0→1→2 的支持度、條件機率與 PMI |
| P1 | 執行 K 與 random-seed stability | 排除 K-means 偶然性 | 完成 Hungarian matching 後的 AMI／ARI 與 edge stability |
| P1 | 完成 controlled-position 實驗 | 拆解內容與位置效應 | 相同 patch 跨位置比較，且不混用完整影像平移結論 |
| P1 | 完成 concept intervention baseline | 驗證 cluster 對分類的作用 | 概念介入與等量隨機介入有可重複的 effect gap |
| P2 | 評估 spectral preprocessing | 檢查高維雜訊問題 | 與直接 K-means 完成 compactness、stability、faithfulness 比較 |
| P2 | 評估真正 spherical K-means | 校正距離方法名稱與效果 | 與 normalized Euclidean K-means 完成消融 |

## 七、預期論文定位

本研究不應將貢獻描述成新的無監督分割方法，也不應只宣稱「將 K-means 用在 ViT feature」。較合適的定位是：

> 一套針對階層式 Vision Transformer 的位置條件式概念發現與跨層追蹤框架。此框架在資料集尺度上分析各 stage、空間位置與 attention head 的特徵群集，並利用模型真實的 patch-merging 關係建立低階至高階的概念組合路徑，再透過內部表徵介入檢驗這些概念及路徑對分類決策的作用。

相較既有工作，本研究預計主張的差異為：

1. 從單圖內空間分群轉為固定位置的跨影像概念分析；
2. 從單層或全域 codebook 轉為 stage／position／head 條件式 codebook；
3. 從靜態代表圖轉為依真實 patch merging 建立的跨層概念組合；
4. 從相關性視覺化延伸到完整概念路徑的介入驗證。

其中前兩點是目前方法已具備的特性；後兩點是接下來需要以實驗完成並驗證的主要研究貢獻。
