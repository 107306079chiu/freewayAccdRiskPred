import pandas as pd
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
from joblib import load

def call_1dcnn():
	model = keras.models.load_model('model_1DCNN')

	return model

def call_svm():
	clf = load('model_svm.joblib')

	return clf

def test_data_global():
	num_features = 4 # 5, 10 min speed + 5, 10 min volume, weather to be added
	sequence_length = 110 # kms

	# transfer y from pure km value(colume: eventKm) into risk matrix
	x = pd.read_csv('inputData_newAll.csv')
	x = x.sample(frac=1, random_state=1)
	y = np.zeros((x.shape[0],110))
	for i in range(len(x)): 
		if x.loc[i, 'class']==1:
			km = round(x.loc[i, 'eventKm'])
			y[i, km] = 1
	# drop useless col
	x = x.drop(['class', 'eventNo', 'date', 'time', 'direction', 'eventKm'], axis=1) 
	# do sliding -> sliding_window_view
	x = x.to_numpy().reshape((len(x), sequence_length, num_features))
	# min-max normalization
	x  = (x-x.min())/(x.max()-x.min())

	from sklearn.model_selection import train_test_split
	X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

	return X_test, y_test

def test_data_section():
	from sklearn.preprocessing import MinMaxScaler
	from sklearn.model_selection import train_test_split

	num_features = 4 # 5, 10 min speed + 5, 10 min volume, weather to be added
	sequence_length = 110 # kms
	interval = 3 # goal km's +-[3]km as prediction interval
	# 1 prepare data
	df = pd.read_csv('inputData_newAll.csv')
	df = df.sample(frac=1, random_state=1)
	# filter edge kms that exceed the prediction interval
	df = df[df['eventKm']>=interval]
	df = df[df['eventKm']<sequence_length-interval]

	y = df['class'].to_numpy()
	km = df['eventKm'].to_numpy().astype(int)

	# drop useless col
	df = df.drop(['class', 'eventNo', 'date', 'time', 'direction', 'eventKm'], axis=1)

	x = df.to_numpy()
	# min-max norm for each column
	# norm before window select to get real distribution and perform norm, shall?
	x = x.reshape((len(df), -1)) # shape=(row=4348, 440) 
	scaler = MinMaxScaler()
	x = scaler.fit_transform(x)

	# back to shape=(row=4348, channels=4, kms=110), to select window only around the event km 
	x = x.reshape((len(df), num_features, sequence_length))
	x_section = np.empty((len(x), num_features, interval*2+1)) # shape=(row=4348, channels=4, window=7)
	for i in range(len(x)): 
		for j in range(num_features):
			x_section[i,]
			for k in range(interval*2+1):
				x_section[i,j,k] = x[i,j,km[i]-3+k] # sorry for the complexity (actually goal km-3 ~ +3)

	# flatten the input for svm only fit by 2-dim LoL reshape again
	x_section = x_section.reshape((len(x_section),-1))

	X_train, X_test, y_train, y_test = train_test_split(x_section, y, test_size=0.2, random_state=42)

	return X_test, y_test

# model: 1dcnn
X_test, y_test = test_data_global()
model = call_1dcnn()
'''
#result:
loss = model.evaluate(X_test, y_test, verbose=2)
print("Restored model, mse loss: ", loss) # 29/29 - 0s - loss: 0.0047
#plot the prediction:
p = model.predict(X_test)
plt.figure(figsize = (20,2))
plt.imshow([p[100].reshape(110), y_test[100].reshape(110)])
plt.show()
'''

# model: svm
X_test, y_test = test_data_section()
clf = call_svm()
'''
#result:
p = clf.predict_proba(X_test)
print(p[0])
p = clf.predict(X_test)
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, p))
# acc = 0.74
'''