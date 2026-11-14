# 自訂MergingViT、Swin與階層式ViT相關工作

日期2026-11-09；remote140165，/home/ref/RGB_SFM_V3。這裡MergingViT指本repo自訂架構，不據名稱假設對應某篇同名論文。

## Swin v1合併位置與算子

Swin官方BasicLayer.forward先跑該stage所有SwinTransformerBlock，再執行downsample(PatchMerging)。因此合併在stage邊界，與本repo models/MergingViT.py forward順序相同。兩者2×2鄰近token concat成4C，LayerNorm後Linear到2C，空間高寬各約減半。不是MHA內把同位置多heads串接的concat。這是相同形式，不是實作／權重逐項相同：本repo有奇數網格padding、linear bias和不同鄰位串接順序等細節。

主要差異：本repo使用全局MHA；經典Swin交替window MSA與shifted-window MSA，逐步建立跨窗資訊傳遞。Swin合併的特徵也已混合窗內、以及前層傳入的其他位置資訊，不能把2×2 merging來源框當完整有效RF；最後窗口也可能涵蓋整個小網格。stage開頭／結尾的標記可因圖示慣例不同，本質皆上一組blocks和下一組blocks之間。

來源：
- Swin paper https://arxiv.org/abs/2103.14030
- 官方實作 https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py （PatchMerging及BasicLayer）
- 本repo models/MergingViT.py:40,130,270；models/ObjectScaleViT.py:71。

## 相關架構（皆2021）

1. PiT — Rethinking Spatial Dimensions of Vision Transformers，ICCV2021。保留ViT式全局self-attention，stage間用depth-wise convolution pooling降空間、增通道；不是2×2 concat。與本repo全局attention＋階層式空間壓縮的研究設定相近，另有class token處理差異。https://arxiv.org/html/2103.16302v2 （3.2）。
2. HVT — Scalable Vision Transformers with Hierarchical Pooling，ICCV2021。標準MSA blocks之後階層pool token，預設1D max pooling，不可當作與2D鄰域merging完全相同；也討論去CLS後平均pool分類。https://arxiv.org/html/2103.10619v2 （3.2.1、appendix S1）。
3. PVT — Pyramid Vision Transformer: A Versatile Backbone for Dense Prediction without Convolutions，ICCV2021。stage起點用patch embedding將局部特徵flatten投影來逐級縮小網格；stage內SRA只壓縮K/V，與stage邊界降採樣不同。SRA仍跨全圖降採樣後的K/V，原文明確描述global receptive field。https://arxiv.org/html/2102.12122v2 （3.2、3.3及Introduction）。
4. MViT — Multiscale Vision Transformers，ICCV2021。attention內pool Q/K/V；Q pooling會降低輸出token數，K/V pooling本身不改變輸出token數。不同於本repo獨立於attention之外的2×2 concat merger。https://arxiv.org/html/2104.11227v1 （3.1、3.2）。

## 與目前研究的關係

依本repo結構判斷：局部merging算子接近Swin v1；全局attention結合空間金字塔的設計可與PiT／PVT對照，但不是任何完整架構的等同實作。「先跨位置混合，再做局部降採樣」已有先例；局部合併仍有空間壓縮及父子來源對應意義，不保證嚴格局部有效RF或部件語意組合。若比較attention範圍對跨層／跨尺度指派的影響，需另控制參數、深度、訓練預算、位置編碼等，不直接由Swin或PiT文獻推斷目前CLIP實驗結果。此回合僅查證與討論，未修改模型或啟動新實驗。

## 雙機交接

版本副本與SHA256交付E:\RGB_SFM_artifacts HTTPS inbox；本機project root尚未確認。收件匣交付不是專案匯入；確認root後驗hash、保留不同版本再匯入。
