# ViT 特徵分群文獻整理：跨影像語意分析、Prototype 與 Category Discovery

更新日期：2026-09-30

## 1. 調查問題與結論

本文件調查頂會研究如何對 ViT／self-supervised ViT 特徵進行分群，特別區分其使用的 K-means 種類，而不把所有「先正規化再 K-means」籠統稱為 spherical K-means。

整體結論如下：

> 跨影像語意分析、prototype 建立與 category discovery，多使用一般 Euclidean K-means、輸入正規化後的一般 Euclidean K-means、semi-supervised K-means，或 online prototype clustering。Spectral 方法主要出現在單張影像內的 patch graph segmentation／object discovery，而不是跨影像固定位置的 prototype 分析。

與本研究最接近的先例是：

1. **Teaching Matters（CVPR 2023）**：直接以 scikit-learn `KMeans` 分析不同 ViT layer 的跨影像 CLS／pooled spatial features；
2. **Deep ViT Features as Dense Visual Descriptors（ECCV 2022）**：先 L2-normalize descriptors，再呼叫標準 K-means；
3. **Leopart（CVPR 2022）**：標準化、PCA 後使用明確設定 `spherical=False` 的 FAISS K-means；訓練階段另有單位 prototype 的 cosine assignment；
4. **Generalized Category Discovery（CVPR 2022）**：使用帶標籤限制的 semi-supervised K-means；
5. **3DMiner（ICCV 2023）**：對跨影像 DINO-ViT global／patch features 使用 K-means。

## 2. 本文件的分類標準

### 2.1 一般 Euclidean K-means

輸入為 $x_i$，centroid 使用算術平均：

\[
\mu_k=\frac{1}{|C_k|}\sum_{i\in C_k}x_i,
\]

並最小化：

\[
\sum_k\sum_{i\in C_k}\|x_i-\mu_k\|_2^2.
\]

centroid 更新後不會重新正規化。

### 2.2 L2-normalized input + Euclidean K-means

先計算：

\[
\hat{x}_i=\frac{x_i}{\|x_i\|_2},
\]

再呼叫一般 Euclidean K-means。樣本位於單位球面，但 centroid 通常滿足：

\[
\|\mu_k\|_2<1.
\]

本類型**不是嚴格的 spherical K-means**。

### 2.3 Spherical K-means

樣本與 centroid 都被限制在單位球面；每次 centroid 更新後還需要：

\[
\mu_k\leftarrow\frac{\mu_k}{\|\mu_k\|_2}.
\]

assignment 等價於最大化 cosine similarity。只有論文或程式明確指出 centroid normalization、cosine K-means，或例如 FAISS `spherical=True` 時，本文件才標成 spherical K-means。

### 2.4 其他 K-means 變體

- **Semi-supervised K-means**：部分樣本的 assignment 受標籤限制；
- **Soft／local K-means**：只在局部候選中心中，以 soft assignment 更新；
- **Online prototype clustering**：prototype 隨訓練更新，未必等同離線 Lloyd K-means；
- **Spectral clustering／normalized cut**：先建立 affinity graph 並計算 Laplacian eigenvectors，不屬於一般 K-means。

## 3. 代表性頂會文獻總表

| 論文 | 會議 | 使用的 ViT 特徵與範圍 | 分群種類 | 是否為 spherical K-means | 是否為 spectral | 判定依據 |
|---|---|---|---|---:|---:|---|
| Teaching Matters | CVPR 2023 | 5,000 張 ImageNet 影像的逐層 CLS token；spatial tokens 先 average pooling | **一般 Euclidean K-means**，$K=50$ | 否 | 否 | 官方程式直接呼叫 `sklearn.cluster.KMeans(...).fit(dataset_feats)`，未在 clustering path 進行 L2 normalization |
| Deep ViT Features as Dense Visual Descriptors | ECCV 2022 | DINO-ViT dense descriptors、跨影像部件與 correspondence candidates | **L2-normalized input + 一般 Euclidean K-means** | 否 | 否 | 官方程式先除以 descriptor length，再呼叫 sklearn／FAISS K-means；未重新正規化 centroid |
| Self-Supervised Learning of Object Parts（Leopart） | CVPR 2022 | 跨資料集 spatial-token embeddings | 評估時為 **StandardScaler + PCA + 一般 FAISS Euclidean K-means**；訓練時另有 normalized prototypes | 評估 K-means：否；訓練 prototype assignment：具有 spherical/cosine 性質 | 否 | 官方評估程式明確使用 `faiss.Kmeans(..., spherical=False)`；論文訓練段則明確將 prototypes L2-normalize |
| Generalized Category Discovery | CVPR 2022 | DINO ViT-B/16 image-level CLS features | **Semi-supervised Euclidean K-means** | 未標成 spherical；論文未聲明 centroid normalization | 否 | 以 class labels 固定已標記資料 assignment；新類別以 K-means++ 初始化，再進行 centroid update／assignment |
| 3DMiner | ICCV 2023 | 跨影像 DINO-ViT global features；不同 layer/location 的 patch features | **一般 K-means** | 論文未提供足夠資訊證明為 spherical | 否 | 論文明確寫 run K-means；未宣稱 spectral 或 spherical variant |
| Expediting Large-Scale Vision Transformer | NeurIPS 2022 | 單張影像／局部視窗內的 ViT tokens | **改良 SLIC 的 local soft K-means** | 否 | 否 | 使用空間鄰近候選中心、softmax Euclidean assignment 與加權中心更新 |
| PaCa-ViT | CVPR 2023 | patch-to-cluster attention 中的 learned clusters | **端到端 learned soft clustering**，不是離線 Lloyd K-means | 否 | 否 | clusters 由模型聯合學習，用於 patch-to-cluster attention |
| TokenCut | CVPR 2022 | 單張影像最後一層 DINO patch tokens | **Normalized cut／spectral graph partition** | 不適用 | 是 | cosine affinity graph + generalized eigendecomposition；以第二小 eigenvector 切 foreground/background |
| Deep Spectral Methods | CVPR 2022 | 單張影像 self-supervised dense features | **Spectral segmentation**；跨影像語意區域另做 clustering | 不適用 | 是 | feature affinity Laplacian eigenvectors 用於 image decomposition |

### 3.1 Dataset-wise K-means 研究

此處的 dataset-wise 是指將跨樣本、跨影像或跨影片收集的特徵共同聚類，而不是只在每張影像內獨立聚類。

「嚴謹：是」代表 centroid 每輪更新後明確維持 unit norm；「等價形式」代表以 cosine distance 對 raw-mean centroid 重新除以 norm，assignment 在數學上等價；「未能確認」代表論文名稱雖寫 Spherical K-means，但缺少 centroid 更新細節。

| 論文 | 會議 | 方法 | 用途 | 模型哪部分的特徵 | 聚類單位 |
|---|---|---|---|---|---|
| [DeepCluster](https://openaccess.thecvf.com/content_ECCV_2018/papers/Mathilde_Caron_Deep_Clustering_for_ECCV_2018_paper.pdf) | ECCV 2018 | Euclidean K-means | 產生 pseudo-label，反向訓練視覺表徵 | AlexNet／VGG 最後卷積層經 pooling、PCA 的 image feature | 整個訓練集的影像 |
| [DeepCluster-v2](https://proceedings.neurips.cc/paper_files/paper/2020/file/70feb62b69f16e0238f741fab228fec2-Supplemental.pdf) | NeurIPS 2020 | Spherical K-means（**嚴謹：是**；features 與 centroids 均為 unit norm） | 產生全資料集 pseudo-label | ResNet-50 projection head 的 image representation | 整個 ImageNet 訓練集的影像 |
| [Editing in Style](https://openaccess.thecvf.com/content_CVPR_2020/supplemental/Collins_Editing_in_Style_CVPR_2020_supplemental.pdf) | CVPR 2020 | Mini-batch Spherical K-means（**嚴謹：是**；輸入與每輪 centroids 均正規化） | 發現跨影像共享的語義部件 | StyleGAN／ProgGAN 中間層的 spatial activation vectors | 多張生成影像的所有空間 patch |
| [Cross-Level Instance-Group Discrimination（CLD）](https://openaccess.thecvf.com/content/CVPR2021/supplemental/Wang_Unsupervised_Feature_Learning_CVPR_2021_supplemental.pdf) | CVPR 2021 | Spherical K-means（**嚴謹：等價形式**；使用 cosine objective，但未明示每輪原地投影） | 建立 instance groups，提供跨層對比學習訊號 | CNN encoder 的 image-level representation | 訓練資料中的影像 instance |
| [Source-Free Domain Adaptation via Distribution Estimation（SFDA-DE）](https://openaccess.thecvf.com/content/CVPR2022/papers/Ding_Source-Free_Domain_Adaptation_via_Distribution_Estimation_CVPR_2022_paper.pdf) | CVPR 2022 | Cosine／Spherical K-means（**嚴謹：等價形式**；cosine assignment，但 centroid 僅保存 raw mean） | 對 target data 產生 pseudo-label | Pretrained CNN feature extractor 的 image feature | 整個 target dataset 的影像 |
| [SLIC](https://openaccess.thecvf.com/content/CVPR2022/papers/Khorasgani_SLIC_Self-Supervised_Learning_With_Iterative_Clustering_for_Human_Action_Videos_CVPR_2022_paper.pdf) | CVPR 2022 | FINCH；Spherical K-means baseline（**嚴謹：未能確認**；實作細節不足） | 依 cluster 採樣較困難的 positive pairs | Self-supervised video encoder 的 video embedding | 整個訓練集的影片 |
| [COINCIDE](https://aclanthology.org/2024.emnlp-main.291.pdf) | EMNLP 2024 | Spherical K-means（**嚴謹：未能確認**；輸入為 unit norm，但 centroid 更新未交代） | 發現 concept-skill clusters，進行 coreset selection | TinyLLaVA 多個 Transformer 層的 visual 與 text token pooled activations | 整個視覺指令資料集的 image–instruction samples |
| [Unsupervised Hierarchical Semantic Segmentation](https://web.eecs.umich.edu/~stellayu/publication/doc/2022hsgCVPR.pdf) | CVPR 2022 | Spherical K-means（**嚴謹：未能確認**；正規化實作未公開） | 建立基礎 segmentation groups | Clustering Transformer 的 pixel-wise features | 多視角影像中的 pixel features；主要偏影像內 grouping |
| [Mining and Unifying Heterogeneous Contrastive Relations](https://openaccess.thecvf.com/content/WACV2024/papers/Duan_Mining_and_Unifying_Heterogeneous_Contrastive_Relations_for_Weakly-Supervised_Actor-Action_Segmentation_WACV_2024_paper.pdf) | WACV 2024 | Spherical K-means（**嚴謹：是**；更新式明確將 centroid 正規化） | 建立 feature centroids 與 contrastive relations | 視覺網路的 pixel-level features | 影像／影片資料中的 pixel features 與 semantic regions |

## 4. 各論文方法與 K-means 類型

### 4.1 Teaching Matters: Investigating the Role of Supervision in Vision Transformers

- **會議**：CVPR 2023
- **目的**：比較不同 supervision 下，ViT 不同 layer 是否逐漸形成 global／local semantic structure。
- **資料組織**：
  - 5,000 張 ImageNet 影像；
  - 每層 CLS token；
  - spatial tokens 則先在位置維度 average pooling；
  - 每層獨立執行 $K=50$ 分群。
- **K-means 類型**：一般 Euclidean K-means。
- **不是 spherical 的理由**：官方程式為：

```python
kmeans = KMeans(n_clusters=k, random_state=0).fit(dataset_feats)
```

在傳入 `KMeans` 前沒有看到 per-sample L2 normalization，centroid 也使用 sklearn 的普通算術平均更新。
- **不是 spectral 的理由**：沒有 affinity graph、Laplacian 或 eigendecomposition。
- **與本研究的關係**：這是最直接支持「以 K-means 分析跨影像 ViT layer features」的頂會先例，但本研究額外加入了 per-sample L2 normalization。

來源：

- [CVPR 2023 論文](https://openaccess.thecvf.com/content/CVPR2023/papers/Walmer_Teaching_Matters_Investigating_the_Role_of_Supervision_in_Vision_Transformers_CVPR_2023_paper.pdf)
- [官方程式庫](https://github.com/mwalmer-umd/vit_analysis)
- [實際 K-means 實作](https://github.com/mwalmer-umd/vit_analysis/blob/main/cka/utils.py)

### 4.2 Deep ViT Features as Dense Visual Descriptors

- **會議**：ECCV 2022
- **目的**：以 DINO-ViT dense descriptors 處理 co-segmentation、part co-segmentation 與 correspondence。
- **資料組織**：跨影像收集 patch descriptors 或 correspondence-pair descriptors。
- **K-means 類型**：L2-normalized input + 一般 Euclidean K-means。
- **實作證據**：官方 correspondence 程式先計算：

```python
length = np.sqrt((all_keys_together ** 2).sum(axis=1))[:, None]
normalized = all_keys_together / length
kmeans = KMeans(...).fit(normalized)
```

- **不是 spherical 的理由**：輸入 descriptors 雖被 L2-normalize，但 sklearn `KMeans` 的 centroid 更新後沒有重新投影到單位球面。
- **不是 spectral 的理由**：該分群步驟沒有建立 graph 或執行 eigendecomposition。
- **與本研究的關係**：這是「L2-normalized ViT descriptors + ordinary K-means」非常直接的先例，方法命名也應與本研究一樣避免寫成 spherical K-means。

來源：

- [論文專案頁](https://dino-vit-features.github.io/)
- [官方程式庫](https://github.com/ShirAmir/dino-vit-features)
- [正規化後呼叫 sklearn K-means 的程式](https://github.com/ShirAmir/dino-vit-features/blob/main/correspondences.py)

### 4.3 Self-Supervised Learning of Object Parts for Semantic Segmentation（Leopart）

- **會議**：CVPR 2022
- **目的**：讓 ViT spatial tokens 學到跨影像一致的 object parts，並做完全無監督 semantic segmentation。
- **評估階段的 K-means 類型**：
  1. 對每一維 feature 做 StandardScaler；
  2. PCA 降維；
  3. 使用 FAISS Euclidean K-means；
  4. 官方程式明確設定 `spherical=False`。
- **訓練階段的 prototype assignment**：論文明確說明 prototype $C$ 在每次 gradient step 後做 L2 normalization，使 $Z^TC$ 直接成為 cosine similarity。這具有 spherical/cosine prototype assignment 性質，但它是端到端訓練的一部分，不等同於評估階段的離線 spherical K-means。
- **後處理**：高粒度 K-means clusters 還可依局部 co-occurrence graph，以 Infomap community detection 合併。
- **與本研究的關係**：顯示頂會研究會把「離線 K-means evaluation」與「訓練時 normalized prototypes」分開描述，不能因為模型使用 cosine prototype 就把所有後續 K-means 都稱為 spherical。

來源：

- [CVPR 2022 論文](https://openaccess.thecvf.com/content/CVPR2022/papers/Ziegler_Self-Supervised_Learning_of_Object_Parts_for_Semantic_Segmentation_CVPR_2022_paper.pdf)
- [官方程式庫](https://github.com/mkuuwaujinga/leopart)
- [官方 clustering utilities](https://github.com/mkuuwaujinga/leopart/blob/main/experiments/utils.py)

### 4.4 Generalized Category Discovery

- **會議**：CVPR 2022
- **目的**：在部分影像有標籤、未標記影像同時包含已知與未知類別時，自動發現 categories。
- **特徵**：DINO-pretrained ViT-B/16 的 image-level feature，並對最後一個 transformer block 做 contrastive fine-tuning。
- **K-means 類型**：semi-supervised K-means。
- **和普通 K-means 的差異**：
  - 已知類別的初始 centroids 由標記資料建立；
  - 未知類別以 K-means++ 初始化；
  - 已標記資料在每輪 assignment 中被強制留在正確 class cluster；
  - 未標記資料仍依 centroid distance 自由分配。
- **是否為 spherical**：論文沒有聲明 centroid 每輪重新正規化，因此不能標成 spherical K-means。
- **是否為 spectral**：否。

來源：

- [CVPR 2022 論文與官方索引](https://openaccess.thecvf.com/content/CVPR2022/html/Vaze_Generalized_Category_Discovery_CVPR_2022_paper.html)

### 4.5 3DMiner: Discovering Shapes from Large-Scale Unannotated Image Datasets

- **會議**：ICCV 2023
- **目的**：從大型未標記影像資料中發現可進行 3D reconstruction 的 object collections。
- **資料組織**：
  - 以多種 augmentation 的 DINO-ViT global features 建立 image clusters；
  - 另外收集不同 ViT layer、不同 spatial location 的 patch features，再做 part segmentation／keypoint discovery。
- **K-means 類型**：論文明確為 K-means，但沒有足夠證據標成 spherical K-means。
- **是否為 spectral**：否。
- **與本研究的關係**：支持在跨影像 global／patch ViT features 上直接建立 K-means prototypes。

來源：

- [ICCV 2023 論文](https://openaccess.thecvf.com/content/ICCV2023/papers/Cheng_3DMiner_Discovering_Shapes_from_Large-Scale_Unannotated_Image_Datasets_ICCV_2023_paper.pdf)
- [官方專案頁](https://ttchengab.github.io/3dminerOfficial/)

### 4.6 Expediting Large-Scale Vision Transformer for Dense Prediction without Fine-tuning

- **會議**：NeurIPS 2022
- **目的**：在不重新 fine-tune backbone 的情況下降低高解析度 ViT token 數量。
- **K-means 類型**：改良 SLIC 的 local soft K-means。
- **方法特徵**：
  - adaptive average pooling 初始化低解析度 centers；
  - 每個 token 只與空間鄰近的 λ 個中心比較；
  - 以 Euclidean-distance softmax 形成 soft assignment；
  - 以 assignment-weighted sum 更新 centers。
- **是否為 spherical**：否。
- **是否為 spectral**：否。
- **注意**：這是模型內 token reduction，不是事後 feature interpretability analysis。

來源：

- [NeurIPS 2022 論文](https://proceedings.neurips.cc/paper_files/paper/2022/file/e6c2e85db1f1039177c4495ccd399ac4-Paper-Conference.pdf)

### 4.7 PaCa-ViT: Learning Patch-to-Cluster Attention in Vision Transformers

- **會議**：CVPR 2023
- **目的**：以少量 learned clusters 取代完整 patch-to-patch attention。
- **分群種類**：端到端 learned patch-to-cluster soft assignment。
- **是否為普通／spherical K-means**：都不是；cluster assignment 與 representation 一起學習，不是對凍結 features 執行 Lloyd iterations。
- **是否為 spectral**：否。

來源：

- [CVPR 2023 論文](https://openaccess.thecvf.com/content/CVPR2023/html/Grainger_PaCa-ViT_Learning_Patch-to-Cluster_Attention_in_Vision_Transformers_CVPR_2023_paper.html)

## 5. Spectral 方法：主要用於單張影像內的 token graph

### 5.1 TokenCut

- **會議**：CVPR 2022
- **資料組織**：單張影像最後一層 DINO patch tokens。
- **方法**：
  1. 以 token cosine similarity 建立 affinity graph；
  2. 解 normalized-cut generalized eigensystem；
  3. 使用第二小 eigenvector 形成 foreground/background partition。
- **分類**：spectral graph partition，不是 spectral embedding 後再做多群 K-means 的一般 pipeline。
- **額外細節**：作者曾比較在第二小 eigenvector 上使用 K-means、EM 或平均值切割，最後採用平均值切割。

來源：

- [CVPR 2022 論文](https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_Self-Supervised_Transformers_for_Unsupervised_Object_Discovery_Using_Normalized_Cut_CVPR_2022_paper.pdf)

### 5.2 Deep Spectral Methods

- **會議**：CVPR 2022
- **資料組織**：單張影像的 self-supervised dense features。
- **方法**：從 feature affinity matrix 建立 graph Laplacian，分析 eigenvectors 並做 image decomposition。
- **分類**：spectral segmentation。
- **與跨影像 K-means 的差別**：它利用同一影像內 patch adjacency／affinity 的流形結構；這種結構在本研究的固定位置、跨影像 samples 中不一定存在。

來源：

- [CVPR 2022 論文](https://openaccess.thecvf.com/content/CVPR2022/html/Melas-Kyriazi_Deep_Spectral_Methods_A_Surprisingly_Strong_Baseline_for_Unsupervised_Semantic_CVPR_2022_paper.html)
- [官方程式庫](https://github.com/lukemelas/deep-spectral-segmentation)

## 6. 對本研究的建議命名與定位

本研究目前流程為：

```text
固定 (stage, block, position)
→ 收集跨影像 feature vectors
→ per-sample L2 normalization
→ sklearn Euclidean K-means
→ centroid-nearest representative patches
```

最精確名稱是：

> **L2-normalized-feature Euclidean K-means**

或：

> **Normalized-feature Euclidean K-means**

不建議稱為：

- spherical K-means；
- cosine K-means；
- spectral K-means。

原因是：

1. 樣本會被投影到單位球面；
2. sklearn K-means centroid 是群內樣本的算術平均；
3. centroid 更新後不會重新投影到單位球面；
4. assignment 仍使用樣本到非單位 centroid 的 Euclidean distance。

### 建議的文獻論述

可在論文中寫成：

> Following prior analyses that apply K-means directly to cross-image ViT representations, we cluster features collected at each fixed spatial position independently. Before clustering, each feature vector is L2-normalized to reduce the influence of feature magnitude. We then apply standard Euclidean K-means; cluster centroids are arithmetic means and are not reprojected onto the unit hypersphere. Therefore, our procedure should be distinguished from spherical K-means.

中文版本：

> 參考既有研究以 K-means 分析跨影像 ViT 表徵的做法，本研究針對各固定空間位置分別蒐集特徵並執行分群。為降低 feature magnitude 的影響，每筆特徵在分群前先進行 L2 normalization，之後使用標準 Euclidean K-means。由於群中心為算術平均且更新後不會重新投影至單位球面，本方法不同於 spherical K-means。

## 7. 實務判斷原則

閱讀論文或程式時，可依下列順序判定 K-means 種類：

1. 是否在 K-means 前對每筆 feature 做 L2 normalization？
2. 使用 sklearn `KMeans`、FAISS `spherical=False`，或一般 Lloyd update？
3. centroid 更新後是否明確重新 L2-normalize？
4. assignment 使用 squared Euclidean distance、inner product，還是 cosine similarity？
5. 是否只是訓練時的 learned prototypes，而不是離線 K-means？
6. 是否建立 affinity graph 並解 Laplacian eigensystem？若是，才屬 spectral 類方法。

若論文只寫「K-means」而沒有公開 normalization、metric 或 centroid update，應記錄為：

> **K-means，具體 metric／normalization variant 未由論文確認。**

不應自行推定為 spherical K-means。

## 8. 跨影像、非固定位置的 ViT 特徵分群

### 8.1 問題定義

本節專門調查下列資料組織：

```text
影像 1：position 1, 2, ..., P
影像 2：position 1, 2, ..., P
...
影像 N：position 1, 2, ..., P
→ 將不同影像、不同 position 的 patch tokens 一起分析
```

這和本研究目前的 per-position 設定不同：

```text
目前設定：每個 position 各自跨影像分群
本節設定：所有或多個 positions 跨影像共同分群
```

調查結果顯示，這類工作確實存在，而且多數不使用 spectral clustering。常見策略是：

1. 將所有 patch descriptors 視為 bag-of-descriptors，使用 K-means；
2. 先以前景／instance mask 篩選 tokens，再跨影像分群；
3. 使用 cosine-normalized prototype 或 minibatch K-means；
4. 以跨影像 feature correspondence 訓練額外的 compact segmentation head；
5. 先在每張影像或每個 instance 內分群，再跨資料集對 region／part representations 做第二階段分群。

### 8.2 方法總表

| 論文 | 會議 | 是否混合不同 spatial positions | 跨影像方式 | 分群／分析方法 | K-means 類型 |
|---|---|---:|---|---|---|
| Deep ViT Features as Dense Visual Descriptors | ECCV 2022 | **是** | 將所有影像、所有 spatial locations 合併成 bag-of-descriptors | 全部 descriptors K-means；saliency voting；前景 descriptors 再做第二次 K-means；CRF 平滑 | **L2-normalized input + 一般 FAISS Euclidean K-means** |
| Leopart | CVPR 2022 | **是** | 收集資料集內所有影像的 spatial-token embeddings | StandardScaler、PCA、FAISS K-means；foreground attention filtering；overclustering 後用 Infomap 合併 | 評估時明確為 **Euclidean K-means (`spherical=False`)**；訓練 prototypes 為 cosine-normalized |
| STEGO | ICLR 2022 | **是** | DINO dense features 建立同圖、相似影像、隨機影像間 correspondence | contrastive correspondence distillation；學習 segmentation head；cluster probe | **Cosine minibatch K-means-like learned prototypes**；features 與 prototypes 每次 forward 都 normalize |
| 3DMiner | ICCV 2023 | **是** | 收集不同 layer、所有 spatial locations、所有影像的 DINO-ViT features | K-means 後以跨影像 voting 選常見且 salient segments | 論文明確為 K-means；是否 spherical 未充分揭露 |
| PartDistillation | CVPR 2023 | **是，但先限制在 object instance** | 每個 instance 內對不同 positions 分群；再聚合全資料集的 part queries | instance-level K-means、weighted K-means density、self-training | 論文比較 cosine 與 L2，**cosine K-means 較佳**；後段為 weighted K-means-based density |
| Scaling the Codebook Size of VQGAN to 100,000 | NeurIPS 2024 | **是** | 從 ImageNet／FFHQ 所有訓練影像抽取 CLIP ViT patch features | CUDA K-means 建立 100K codebook；後續 quantizer 使用 cosine similarity | 論文未充分揭露 K-means centroid normalization；不可直接標成 spherical |
| PDiscoNet | ICCV 2023 | **是** | 共享 part prototypes 在所有影像位置上產生 assignment maps | 端到端 part prototypes、classification 與 compactness／equivariance losses | 不是離線 K-means；屬 learned part-prototype model |

### 8.3 Deep ViT Features：最直接的 all-images/all-positions 案例

該研究明確寫道：將所有輸入影像、所有 spatial locations 的 descriptors 視為一個 bag-of-descriptors，再共同執行 K-means。

流程為：

```text
多張影像
→ 各自抽取 DINO-ViT dense key descriptors
→ 合併所有影像的所有 positions
→ 每筆 descriptor L2 normalization
→ FAISS Euclidean K-means
→ 依 DINO saliency maps 對各 cluster 投票
→ 保留 foreground clusters
→ foreground descriptors 再做 K-means，取得共同 parts
→ DenseCRF 平滑邊界
```

官方程式的關鍵操作為：

```python
all_descriptors = np.concatenate(descriptors_list, axis=2)
faiss.normalize_L2(normalized_all_descriptors)
algorithm = faiss.Kmeans(d=..., k=..., niter=300, nredo=10)
algorithm.train(normalized_all_sampled_descriptors)
```

因此它是：

> **跨影像、跨位置、L2-normalized input + ordinary FAISS Euclidean K-means。**

FAISS 此處沒有設定 spherical mode；centroid 不保證重新投影到單位球面，所以不能稱為嚴格 spherical K-means。

這篇工作處理「不同位置混合」的關鍵不是加入 position coordinates，而是：

- 使用較深層 DINO keys，降低 positional bias；
- 先以 saliency voting 找 foreground；
- 再對 foreground descriptors 做第二階段 part clustering；
- 使用 CRF 恢復空間連續性。

來源：

- [ECCV 2022 論文](https://dino-vit-features.github.io/paper.pdf)
- [官方 all-images K-means 實作](https://github.com/ShirAmir/dino-vit-features/blob/main/part_cosegmentation.py)

### 8.4 Leopart：跨資料集 spatial tokens，但先做 feature preprocessing

Leopart 的 overclustering evaluation 會收集跨影像 spatial embeddings，不固定同一 position，再依序執行：

```text
所有 spatial-token features
→ per-dimension StandardScaler
→ PCA
→ FAISS K-means (`spherical=False`)
→ Hungarian／many-to-one evaluation
```

它另外使用 DINO CLS attention map 篩選 foreground tokens，並可對高粒度 clusters 建立局部 co-occurrence graph，再用 Infomap community detection 合併成較高階 objects。

需區分兩個階段：

- **訓練階段**：learned prototypes 會 L2-normalize，以 cosine similarity assignment；
- **離線評估階段**：StandardScaler + PCA 後使用 `spherical=False` 的 Euclidean K-means。

這說明跨位置語意分析不一定要求所有 tokens 位於單位球面；也有研究選擇先做逐維標準化與 PCA，再以 ordinary K-means 評估 learned representation。

來源：

- [CVPR 2022 論文](https://openaccess.thecvf.com/content/CVPR2022/papers/Ziegler_Self-Supervised_Learning_of_Object_Parts_for_Semantic_Segmentation_CVPR_2022_paper.pdf)
- [官方 clustering utilities](https://github.com/mkuuwaujinga/leopart/blob/main/experiments/utils.py)

### 8.5 STEGO：不用一次性全域 K-means，而是學習跨影像 compact feature space

STEGO 並非單純將原始 DINO tokens 丟入一次 K-means。它先利用 DINO dense feature cosine similarity，建立三類 correspondence：

1. 同一影像內；
2. retrieval 找到的相似影像間；
3. 隨機影像間。

接著訓練一個 shallow segmentation head，使原本 DINO feature correspondence 被蒸餾成更緊密、可分群的 segmentation features。

其 cluster probe 使用 learned prototype matrix；官方程式每次 forward 都執行：

```python
normed_clusters = F.normalize(self.clusters, dim=1)
normed_features = F.normalize(x, dim=1)
inner_products = einsum(normed_features, normed_clusters)
```

並以 cosine inner product 進行 hard 或 soft assignment。因此更接近：

> **可學習、minibatch、cosine/spherical prototype clustering。**

它不是傳統離線 Lloyd spherical K-means，因為 prototypes 由 Adam 與 clustering loss 更新；但在幾何上，sample 與 prototype 都被 normalize，assignment 與 cosine similarity 一致。

這提供另一個重要方向：若 raw ViT tokens 跨位置直接分群不夠 compact，可以學習一個低維 projection／segmentation head，而不是改用 spectral clustering。

來源：

- [ICLR 2022 論文](https://openreview.net/forum?id=SaKO6z6Hl0c)
- [官方專案頁](https://mhamilton.net/stego.html)
- [官方程式庫](https://github.com/mhamilton723/STEGO)

### 8.6 3DMiner：跨影像、跨 layer/location 的 K-means 與 voting

3DMiner 先使用 augmentation-aggregated DINO-ViT global features 對影像集合分群。進入一個 image cluster 後，再收集多個 ViT layers、所有 spatial locations 的 features，執行 K-means，並以 voting 選出在多張影像中共同出現且 salient 的 segments，作為 keypoints 與後續 3D reconstruction 的依據。

它的核心概念是：

```text
先縮小到外觀／物體較一致的 image cluster
→ 再跨影像、跨位置聚合 patch features
→ K-means
→ 以跨影像出現頻率與 saliency voting 排除偶然 clusters
```

這比把整個異質資料集的所有 positions 一次混在一起更穩健。論文沒有充分揭露 centroid normalization，因此只能標為 K-means，不能推定為 spherical。

來源：

- [ICCV 2023 論文](https://openaccess.thecvf.com/content/ICCV2023/papers/Cheng_3DMiner_Discovering_Shapes_from_Large-Scale_Unannotated_Image_Datasets_ICCV_2023_paper.pdf)
- [官方專案頁](https://ttchengab.github.io/3dminerOfficial/)

### 8.7 PartDistillation：先 instance-level cosine K-means，再跨資料集整理 parts

PartDistillation 沒有直接對全資料集所有 pixels 做一次全域 K-means。它先使用 object instance mask 限制範圍，再對每個 instance 內不同 positions 的 Mask2Former／Swin features 做 K-means，產生 class-agnostic part proposals。

接著：

```text
每個 object instance 內的 pixel/token features
→ K-means 產生 part proposals
→ query-based decoder self-training
→ 按 object class 聚合整個 training set 的 candidate part queries
→ weighted K-means-based density estimation
→ 產生 class-specific part labels
```

論文消融直接比較 cosine 與 L2 distance，並報告 cosine similarity 整體較佳。此外，instance-level clustering 明顯優於直接對 dataset-level pixels 做一次 one-stage clustering。

這給本研究的重要啟示是：

> 不固定位置不等於必須把所有影像的所有 tokens 無條件混在一起；先用 object／foreground／image-cluster context 限制候選集合，可能比替換 clustering algorithm 更重要。

來源：

- [CVPR 2023 論文](https://openaccess.thecvf.com/content/CVPR2023/papers/Cho_PartDistillation_Learning_Parts_From_Instance_Segmentation_CVPR_2023_paper.pdf)
- [官方程式庫](https://github.com/facebookresearch/PartDistillation)

### 8.8 VQGAN-LC：以全資料集 CLIP patch features 建立大型 codebook

VQGAN-LC 使用 CLIP ViT-L/14，對 ImageNet 或 FFHQ training split 的影像抽取 patch-level features，先加入 4×4 average pooling，再使用 CUDA K-means 建立最多 100,000 個 cluster centers 作為固定 codebook。

這是另一個清楚的：

```text
跨影像
＋跨 spatial positions
＋大規模 K-means prototypes
```

案例。論文未充分揭露 K-means 本身是否使用 spherical centroid update，因此不可直接標為 spherical；但後續 quantizer 以 cosine similarity 選 codebook entry。

來源：

- [NeurIPS 2024 論文](https://proceedings.neurips.cc/paper_files/paper/2024/file/1716d022edeac750e57a2986a7135e13-Paper-Conference.pdf)

### 8.9 PDiscoNet：不用離線 K-means，直接學跨影像 part prototypes

PDiscoNet 使用共享的 part prototypes 在所有影像的 spatial feature maps 上產生 part assignment maps，並以 image-level classification、part compactness、distinctiveness、equivariance 與 part dropout 等 losses 聯合學習。

它不是離線 K-means，而是將「跨影像相同 part 應使用同一 prototype」直接寫進模型與訓練目標。這類方法的 semantic consistency 通常比單次 clustering 強，但需要標籤或額外訓練，也不再是單純的 post-hoc feature analysis。

來源：

- [ICCV 2023 論文](https://openaccess.thecvf.com/content/ICCV2023/html/van_der_Klis_PDiscoNet_Semantically_consistent_part_discovery_for_fine-grained_recognition_ICCV_2023_paper.html)

## 9. 對本研究「跨影像、不固定位置」版本的建議

### 9.1 最小可行 baseline

最直接對應 Deep ViT Features 的版本為：

```text
固定 stage/block
→ 收集所有訓練影像的所有 positions
→ 保留 image_id 與 position_id metadata
→ per-sample L2 normalization
→ normalized-feature Euclidean K-means
→ 將 labels 映回各影像的 token grid
→ 顯示 centroid-nearest patches
```

這個 baseline 與本研究現有 pipeline 最接近，也有直接頂會先例。

### 9.2 建議同時加入 spherical 消融

由於跨位置 features 的 norm 與方向分布可能比固定位置更複雜，建議比較：

1. L2-normalized input + ordinary Euclidean K-means；
2. spherical K-means；
3. 若成本允許，StandardScaler + PCA + Euclidean K-means（Leopart-style）。

不建議第一版直接使用 spectral clustering，因為全資料集 token graph 的樣本數會快速增長，且沒有天然跨影像 edge。

### 9.3 必須處理的混雜因素

跨位置分群新增了固定位置版本沒有的問題：

- **positional bias**：cluster 可能只代表左上、中央或邊界位置；
- **background dominance**：背景 tokens 的數量通常遠多於物體 parts；
- **image imbalance**：不同尺寸／裁切方式可能讓某些影像貢獻更多 tokens；
- **near-duplicate tokens**：同一影像中的相鄰 positions 高度相似，可能支配 centroid；
- **stage-dependent semantics**：淺層可能依位置、顏色或紋理分群，深層才較接近 parts／objects。

因此至少需報告：

1. 每個 cluster 的 position entropy；
2. 每個 cluster 涵蓋的 unique image 數；
3. 每張影像對 cluster 的最大貢獻比例；
4. foreground/background 比例；
5. centroid/resultant norm；
6. cosine silhouette 與 seed stability；
7. centroid-nearest patches 是否來自多張不同影像，而非單一影像的相鄰 patches。

### 9.4 推薦的逐步研究順序

```text
Step 1：all positions + normalized Euclidean K-means
        建立最小 baseline，對應 Deep ViT Features

Step 2：加入 spherical K-means
        檢查 cosine geometry 是否改善跨位置 semantic consistency

Step 3：加入 foreground／saliency filtering
        對應 Deep ViT Features 與 Leopart

Step 4：先按 global image feature 粗分，再做 patch clustering
        對應 3DMiner，降低完全異質資料混合

Step 5：若 raw tokens 仍不 compact，再考慮 learned projection head
        對應 STEGO，而不是先引入全域 spectral graph
```

### 9.5 對本研究最有價值的三個比較

建議優先實作：

| 實驗 | Sample organization | Clustering | 回答的問題 |
|---|---|---|---|
| A | 固定 position、跨影像 | spherical K-means | 同一空間位置是否形成穩定方向 prototypes？ |
| B | 所有 positions、跨影像 | spherical K-means | semantic parts 是否能跨位置對齊？ |
| C | foreground positions、跨影像 | spherical K-means | 排除背景與位置混雜後，semantic consistency 是否改善？ |

這三組比較比直接加入 spectral clustering 更能對應現有文獻，也能回答「本研究的 per-position 設計究竟提供了幫助，還是限制了跨位置共同語意的發現」。
