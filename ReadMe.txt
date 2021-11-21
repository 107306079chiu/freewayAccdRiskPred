# Digi+talent 跨域數位人才加速躍升計畫 中華顧問工程司-多元運輸組 《高速公路之車流狀況與事故預警平台》: 預測模型
- project running on: http://35.74.237.16/ceci
- group members: S. Lin, T. Cheng, C. Zhu
- intern project @ CECI, instructors: Y. Lee, F. Su, Q. Hsieh

--------catalog--------
1 Introduction
2 Input/Output Form
 2.1 Raw VD之處理
 2.2 全局-Input/Output
 2.3 局部-Input/Output
3 Model Structure & Usage
 3.1 1D CNN
 3.2 SVM
-----------------------

# 1 Introduction
- 使用訓練資料"inputData_newAll.csv"、呼叫已訓練模型"models.py"
- 這個實驗以國道一號北部(基隆0-新竹110)作為實驗路段
- 訓練資料為2020/1-2021/7間的VD資料(車流量、車速)+事故資料(時間、公里數)
- 包含4500筆資料，事故、正常各半(class 0 = 正常；class 1 = 事故)
- @TO-DO天氣特徵尚未加入(因為之前僅納入事發地之天氣，需補上全局的天氣)
- 雖然下面實作兩種Input/Output的方式，但我認為之後(實際執行專題)只要專注在2.3局部-Input/Output

# 2 Input/Output Form
## 2.1 Raw VD之處理
- 原始xml範例(VD五分鐘動態資訊)
<Infos>
	<Info vdid="nfbVD-N3-S-230.310-M-OH" status="0" datacollecttime="2020/01/31 23:55:00">
		<lane vsrdir="0" vsrid="1" speed="104" laneoccupy="1">
			<cars carid="S" volume="4"></cars>
			<cars carid="T" volume="1"></cars>
			<cars carid="L" volume="0"></cars>
		</lane>
		<lane vsrdir="0" vsrid="2" speed="118" laneoccupy="1">
			<cars carid="S" volume="6"></cars>
				<cars carid="T" volume="0"></cars>
				<cars carid="L" volume="0"></cars>
		</lane>
		<lane vsrdir="0" vsrid="3" speed="98" laneoccupy="1">
			<cars carid="S" volume="6"></cars>
			<cars carid="T" volume="0"></cars>
			<cars carid="L" volume="0"></cars>
		</lane>
	</Info>
	...
</Infos>
- 紀錄每個<Info>的
 (1) datacollecttime = "2020/01/31 23:55:00"
 (2) vdid = "nfbVD-N3-S-230.310-M-OH"
 (3) lane_amount = 3
 (4) average_speed = (104+118+98)/lane_amount
 (5) average_volume = ((4+1+0)+(6+0+0)+(6+0+0))/lane_amount
- 處理VD的資料缺失 & 增加空間資訊與時間資訊
由於VD五分鐘動態資訊每次更新收回的資料會有VD數量波動或是直接漏更新，所以在近一步使用資料之前先進行以下處理：
 (1) 去除vdid裡面寫I或O(匝道)，僅留下主線的VD資料 --> 實驗內使用之vdid詳見----"國1vdid_km只留主線.csv"
 (2) VD資料分為南、北向，照公里數升幂排列(分佈不平均，110km內有大約100初個VD)
 (3) 分別對average_speed, average_volume使用內插(numpy.interp)將單個時間點之下110km內VD資料轉換成shape=(1,110) 110代表0-110km平均分佈每一公里的資料
 (4) 以上形式的資料會收集兩份VD五分鐘動態資訊(預測時間之前10分鐘、前5分鐘)作為一筆
 (5) 送入模型前經過min-max normalization
 資料實際欄位詳見----"inputData_newAll.csv" (註: 欄位命名"10s0" --> 10分鐘前第0km處的speed)

## 2.2 全局-Input/Output
- Input分為兩種，全局/局部，首先介紹以一整條高速公路作為輸入的全局輸入。
- 我們將整條高速公路作為輸入，輸出將也是整條高速公路的風險矩陣。
- 範例: 
 Input = [[90,101,88,...,80],
 [99,100,110,...,100],
 [500,430,300,...,50],
 [200,310,220,...,100]], dtype=flaot, shape=(4, 110), 4為channel數(分別為10分鐘前的speed, 5分鐘前的speed, 10分鐘前的volume, 5分鐘前的volume), 110代表0-110km
 Output = [0.1, 0.11, 0.7, ..., 0.05], dtype=flaot, shape=(1, 110), 0-110km每km的事故風險

## 2.3 局部-Input/Output
- 我們將輸入的空間範圍下縮到事故地點的前後數公里，目前以前後3km為實驗參數，未來將持續嘗試不同組合。
- 預測模型則進行分類任務，針對每個公里數給出風險數值(0-1)。
- 範例: 
 假設目標公里數為70km處，區間以前後3km為例
 Input = [[100,110,...,100],
 [101,88,...,80],
 [430,300,...,50],
 [310,220,...,100]], dtype=flaot, shape=(4, 7), 4為channel數(分別為10分鐘前的speed, 5分鐘前的speed, 10分鐘前的volume, 5分鐘前的volume), 7代表目標公里處(含)及前後3公里
 Output = 0.75, dtype=float, 為class 1(事故)的probability

# 3 Model Structure
- 根據上述提及之兩種Input/Output，我們選擇合適預測模型，並且多方嘗試不同演算方法與參數組合。
- "models.py"內已完成load下述兩種預測模型，可以直接測試與使用

## 3.1 1D CNN
- 因應2.2全局輸出為shape=(1, 110)之風險矩陣，我們使用1D Convolution Layer作為模型的主要組成，它具卷積特性得以學習空間資訊並且接受1-dimension資料。
- 建立在Python環境，並使用TensorFlow，其中包含Conv1d, ReLu, Dropout, Dense等Layer。
- 訓練參數: epochs=10, batch=32, km=110, channel=4), optimizer=Adam, lr=0.0005, loss=mse
- input_shape=(len(X_train); output=(110)
- 訓練結果: mse=0.0047。雖然訓練資料保持資料平衡，但由於是全局輸出，一個交通事故在整條高速公路上僅佔了一公里，一個風險矩陣中的0與1比例依舊懸殊，造成模型傾向給出全部極低的風險。
- 訓練過程檔案"model_1DCNN.py"
- 模型儲存於檔案夾"model_1DCNN"

## 3.2 SVM
- 因應2.3局部輸出，進行的是分類任務，在模型的選擇上相對彈性，第一個預測模型首先選擇使用適合學習高維度資料的SVM。
- 建立在Python環境，並使用Sklearn。
- 訓練參數: probability=True
- input_shape=(len(X_train), 4channel*7km); output=[class0 prob, class1 prob]
- 訓練結果: accuracy=0.74。錯誤主要集中在false positive。
- 訓練過程檔案"model_svm.py"
- 模型儲存於檔案夾"model_svm.joblib"
