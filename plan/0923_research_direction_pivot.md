# 0923 研究方向轉向提案

## 一、保留但不作為主線的方向

先前提出的 `cluster confidence` 與 `soft assignment` 保留為輔助模組：

- 使用最近與次近 centroid 的距離差衡量 cluster assignment confidence。
- 對低信心 token 顯示 Top-2／Top-3 clusters，不強迫給出唯一解釋。
- 使用既有平移結果驗證低 margin token 是否更容易換群。

此方向能改善 KMeans 解釋的可靠性，但主要仍是在修正 hard assignment，研究貢獻較接近局部方法改良，因此不建議作為論文唯一主軸。

## 二、建議主線：Hierarchical Causal Concept Graph

### 1. 核心問題

目前的方法只能回答：

> 某一個 stage 的 token 屬於哪一個 KMeans cluster？

新的研究問題改為：

> 低階視覺概念如何在階層式 Vision Transformer 中組合成高階概念？這些概念是否真的對模型分類具有因果作用？

研究成果不再只是各 stage 的代表圖，而是一張具有「概念節點、跨層組合關係與分類因果效果」的階層式概念圖。

### 2. 概念圖定義

每個 stage 的 KMeans cluster 視為一個概念節點：

- Stage 0：邊緣、顏色、局部紋理等低階概念。
- Stage 1：局部形狀或結構。
- Stage 2：較完整的物體部件。
- Stage 3：接近整體物體或類別層級的概念。

利用模型原本的 2×2 patch merging 對應關係，統計低層 cluster 如何組合成下一層 cluster。例如：

```text
Stage 0: 金屬邊緣 + 藍色背景
                 ↓
Stage 1: 機翼局部形狀
                 ↓
Stage 2: 飛機機身／機翼結構
                 ↓
Stage 3: Airplane prediction
```

圖中的邊不是只靠視覺判斷，而由實際 token merging、共同出現頻率及條件機率建立：

\[
w(c_i^s \rightarrow c_j^{s+1})
=P(c_j^{s+1}\mid c_i^s)
\]

可再使用 pointwise mutual information 或條件熵排除只是高頻出現、但沒有特定組合關係的邊。

### 3. 因果概念介入

概念圖建立後，不能只用 correlation 宣稱某個 cluster 很重要。對中間 token feature 進行介入，再觀察模型輸出：

#### Concept removal

將屬於指定 cluster 的 token：

- 設為零；
- 替換成該 stage 的平均 token；
- 替換成背景 cluster centroid。

計算 airplane logit 或 probability 的下降：

\[
CE_{remove}(c)=f_y(x)-f_y(do(c\leftarrow baseline))
\]

#### Concept insertion／replacement

將某一 token 替換成目標 cluster centroid，觀察目標類別機率是否上升。這能形成概念層級的 counterfactual：

> 如果將「機翼概念」替換成「背景概念」，飛機預測是否下降？

#### Causal path intervention

依序介入概念圖中的子節點或父節點，檢查低階概念是否透過預期的高階概念影響輸出。這比只遮蔽輸入 patch 更能檢驗階層模型內部的概念形成過程。

### 4. 最終解釋輸出

對一張影像，輸出不再只是 cluster 代表圖，而包含：

1. 影像中出現的主要概念。
2. 概念由哪一些低階 clusters 組成。
3. 概念在 Stage 0–3 的演化路徑。
4. 移除每個概念後，目標類別分數下降多少。
5. 哪些概念是相關但非必要，哪些概念對預測具有較強的介入效果。
6. Assignment confidence；低信心節點顯示多個候選概念。

## 三、與原方法的差異

| 能力 | 原本 KMeans 代表圖 | Confidence／Soft assignment | Hierarchical Causal Concept Graph |
|---|---:|---:|---:|
| 顯示 cluster 代表圖 | ✓ | ✓ | ✓ |
| 顯示指派不確定性 |  | ✓ | ✓ |
| 連結不同 stages |  |  | ✓ |
| 描述概念如何組成 |  |  | ✓ |
| 測量概念對分類的作用 |  |  | ✓ |
| 產生概念層級 counterfactual |  |  | ✓ |
| 區分相關性與介入效果 |  |  | ✓ |

真正的主貢獻將從「更穩定的 KMeans」改成：

> 一套針對階層式 Vision Transformer 的跨層概念形成與因果驗證框架。

## 四、實驗設計

### Experiment A：跨層概念圖

建立 Stage 0→1、1→2、2→3 的 cluster transition／composition matrix。

輸出：

- 每個高階 cluster 最常見的低階 cluster 組合。
- 概念節點與跨層有向邊。
- 每條邊的條件機率、PMI、支持樣本數。
- 代表圖與實際影像中的空間位置。

### Experiment B：概念因果效果

對每個 cluster 進行 removal、mean replacement 與 background replacement。

輸出：

| Stage | Cluster | 支持樣本數 | 原始類別分數 | 介入後分數 | Causal effect | Accuracy drop |
|---|---:|---:|---:|---:|---:|---:|

### Experiment C：階層路徑驗證

比較三種介入：

1. 只移除低階子概念。
2. 只移除高階父概念。
3. 同時移除整條概念路徑。

若路徑介入效果明顯大於隨機 cluster 或相同 token 數量的隨機遮蔽，表示概念圖捕捉到與分類機制相關的結構。

### Experiment D：反事實概念替換

將 airplane 的重要概念替換成其他 cluster centroid，記錄：

- 原始與替換後類別。
- 目標類別 logit 變化。
- 最容易造成 airplane→helicopter、airplane→ferry 等轉移的概念。
- 替換後下游 stages 的概念路徑如何改變。

## 五、評估指標

### Faithfulness

- Comprehensiveness：移除重要概念後，目標分數下降量。
- Sufficiency：只保留重要概念時，目標分數保留程度。
- Random-intervention gap：概念介入與相同數量隨機 token 介入的差距。
- Causal path effect：介入整條概念路徑的輸出變化。

### Concept quality

- Cluster purity／人工語意一致率。
- Cluster compactness 與 margin。
- 同一概念跨影像的代表圖一致性。
- 跨 seed 的 concept matching 與 graph edge stability。

### Hierarchy quality

- Parent cluster 對 child-cluster combination 的可預測性。
- 跨層條件熵。
- 概念路徑的支持樣本數。
- 真實 merge edges 相對隨機 edges 的 effect gap。

## 六、研究風險與最低可行版本

「概念介入」本身已有相關研究，因此不能只做 cluster masking 就宣稱方法新穎。此研究需要把新意放在下列組合：

1. 使用階層式 ViT 的真實 patch-merging 關係建立跨層概念圖。
2. 對完整的 child→parent concept path 進行介入，而不是只排名單層概念。
3. 追蹤介入後概念在後續 stages 的重新形成或消失。
4. 將 assignment uncertainty 整合進圖節點及因果效果估計。

最低可行版本先使用 airplanes，完成 Stage 0→1→2 的概念圖，選擇 5 個高支持度路徑進行介入，並以相同 token 數量的隨機介入作為 baseline。若能觀察到穩定的 causal-effect gap，再擴充至其他 Caltech-101 類別。

## 七、備選的更大幅度方向

### A. Differentiable Hierarchical Prototype Network

將離線 KMeans 改為可訓練 prototype layers，讓分類必須透過 Stage 0–3 的 prototype activations 完成。模型在輸出類別的同時，也必須輸出概念組成路徑。這是由 post-hoc explanation 轉向 self-explaining architecture，創新幅度最大，但需要重新設計 loss、重新訓練及處理 prototype collapse，風險也最高。

### B. Concept Counterfactual Editor

學習最小幅度的 token-level concept replacement，使模型從原類別轉為指定類別，同時保持非目標區域表徵不變。輸出「需要改變哪些階層概念才能改變判斷」，適合研究模型的決策邊界，但需要嚴格確認修改後的 token 仍位於真實 feature manifold。

### C. Concept Failure Atlas

跨類別建立模型錯誤的概念路徑圖，分析哪些低階概念組合會形成錯誤高階概念。此方向偏向模型診斷系統，工程成果直觀，但方法創新度低於因果概念圖與可訓練 prototype model。

## 八、建議選擇

建議以 `Hierarchical Causal Concept Graph` 為主線，原因是：

- 能直接利用目前完成的逐 stage features、KMeans centroids 與代表圖。
- 研究問題由「cluster 是否穩定」提升為「概念如何形成並影響決策」。
- 不必一開始就重新訓練模型，能先驗證最低可行版本。
- 後續可自然延伸成 Differentiable Hierarchical Prototype Network。

平移實驗則降為方法驗證的一部分，用來測試概念節點與概念路徑的穩定性，不再作為研究主線。

## 九、相關工作定位

- VTCD 使用概念分群及 concept masking 解釋 video transformer，說明「自動概念發現＋介入排序」已有研究基礎：https://openaccess.thecvf.com/content/CVPR2024/html/Kowal_Understanding_Video_Transformers_via_Universal_Concept_Discovery_CVPR_2024_paper.html
- BLIP causal tracing 將 activation patching 用於視覺語言模型，提供內部狀態介入的相關方法：https://openaccess.thecvf.com/content/ICCV2023W/CLVL/papers/Palit_Towards_Vision-Language_Mechanistic_Interpretability_A_Causal_Tracing_Tool_for_BLIP_ICCVW_2023_paper.pdf
- TCFormer 使用 progressive token clustering 建立彈性 token，但目的主要是模型計算與辨識，而非 post-hoc 階層因果解釋：https://arxiv.org/abs/2204.08680

因此，正式宣稱創新前仍需要更完整的系統性文獻回顧；目前最值得檢驗的差異點是「依照真實 patch merging 建立概念組合圖，並介入完整跨層概念路徑」。
