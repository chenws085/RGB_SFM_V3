# Sliding window size細掃與後續receptive field分析

遠端remote140165，根目錄/home/ref/RGB_SFM_V3。使用者要求針對sliding做更細size比較，包含小於4、大於7，以尋找甜蜜點；並指定RF分析在實驗完成後才進行。

## 本輪尺寸與固定條件

尺寸k=1、2、3、4、5、6、7、9、11、15，每個query讀取k×k tokens鄰域；不是pixels，不是互斥分窗。固定lr3e-4、Stage0/1 sliding、Stage2/3 global、原Absolute position、offset參數0與原merging；lr掃描仍暫緩，不做新位置編碼、stage、offset或低通濾波消融。Caltech101 seed42、6907/1770 split、letterbox160、原增強／AdamW／CE／batch32／100epoch上限與patience20保持一致。

每軸offset = [−floor((k−1)/2), …, floor(k/2)]；例k2=[0,1]，k3=[−1,0,1]，k4=[−1,0,1,2]，k7=[−3,…,3]，k15=[−7,…,7]。奇數對稱、偶數有半格方向偏置，兩者趨勢需分開解讀；不另掃偶數的相反方向。k1是只讀自身的attention控制，但merging與深層global仍跨token交流。邊界無效keys遮蔽，無循環連接；k不自動截成stage大小，k15在14×14網格的邊角仍不是全局attention。

Stage0/1名義stride為8/16px，故名義當層跨度為8k/16k px；這不是完整有效RF。精確offsets、最大keys與名義跨度表見runs/train/sliding_size_20261114/REPORT.md。

重用已核對global、sliding4與sliding7結果，補訓其他8個尺寸。主表沿用：第一欄方法/尺寸，其後原位accuracy、1–7px_acc、8px_acc、1–7px分類一致性、8px分類一致性。完整四方向0/1–8/16/32px、training-only spherical codebooks與token指標仍保留。先看原位／平移分類共同取捨及來源配對區間，不用臨時混權重強選唯一尺寸；對候選點才另議多seed，單seed驗證不代表最終甜蜜點。

## 執行狀態

原位4與7結果已核對checkpoint、config及評估manifest hashes，填入新表。GPU檢查：nvidia-smi報Failed to initialize NVML: Unknown Error；torch2.4.1+cu124的cuda.is_available=False、device_count=0，cuda:1實際配置報No CUDA GPUs are available。雖/dev/nvidia*存在，容器目前不能使用GPU。新8組未開始訓練，不改用CPU長時間訓練，也不重設共享host driver。

入口`.venv/bin/python scripts/sliding_size_sweep.py runs/train/sliding_size_20261114`會先檢查GPU1與實際CUDA allocation，成功才進入固定size queue；現有已完成size不重跑。status.json標記waiting_for_gpu，沒有背景輪詢程序；GPU恢復後可用同命令接續。來源SHA256與快照保留，若來源改變需先核對，不默默混用版本。

## 實驗完成之後：分析每個設計的receptive field

依使用者順序，先完成size實驗，再做RF分析與結果解釋。分開三項：

1. 名義局部範圍：各stage的k×k tokens、有效key數與邊界截斷。這部分是配置說明，不稱實測ERF。
2. 理論連接範圍：從指定stage token沿attention與真實merging邊追蹤至輸入。包含前序層、merging與深層global，不只k乘stride；global連接不等於有效訊號分布均勻。
3. 實測有效RF：在相同來源、對齊的物體位置及相同target定義下，分別觀察stage token與最終分類對輸入的梯度影響圖；以相同正規化/聚合方式比較能量集中範圍、涵蓋80/90%梯度質量的面積或半徑、方向性及padding區比例。梯度是局部敏感度，不直接稱因果證據。

RF結果再與局部token cosine/群一致、1–7與8px分類及方向差異對照；不得預設RF較大或較小一定更穩定。原始圖像、畫布與物體位置保持對齊；分列原位與平移條件及邊界案例。具體ERF target與採樣協定待size實驗完成後固定，本輪不先執行RF測量。

研究文件版本及SHA256交付E:\RGB_SFM_artifacts inbox；本機專案root/access未確認，inbox交付不是專案匯入。差異版本保留。

## GPU已恢復，正式開始

使用者確認GPU修復並要求開始。GPU1 RTX3060 12GB的實際CUDA配置、matmul與finite檢查通過；來源SHA256及重用global/sliding4/7 checkpoint hashes吻合，證據gpu_recovery_checks.json。以scripts/run_sliding_size_1114.sh啟動原queue，八個新尺寸依序訓練與評估；4與7不重跑。實際進度以status.json、各size.log及process核對，不把啟動當完成。lr/stage/position/offset等固定條件未改；RF仍在全部size實驗完成後分析。結束後wrapper嘗試交付版本文件，若服務不可用需另外重試，不將上傳嘗試視為成功。
