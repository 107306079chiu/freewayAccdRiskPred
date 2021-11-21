'''
Model: SVM(SVC)
Input: [eventKm+-3km -> 5 & 10 mins before the predicted time as temporal data](shape=(4, 11)) 
Output: risk(0-1)
Label: class 1 -> an accident has occured; class 0 -> normal
'''

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import confusion_matrix
from joblib import dump

num_features = 4 # 5, 10 min speed + 5, 10 min volume, weather to be added
sequence_length = 110 # kms
interval = 15 # goal km's +-[3]km as prediction interval

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
		#x_section[i,]
		for k in range(interval*2+1):
			x_section[i,j,k] = x[i,j,km[i]-interval+k] # sorry for the complexity (actually goal km-3 ~ +3)

# flatten the input for svm only fit by 2-dim LoL reshape again
x_section = x_section.reshape((len(x_section),-1))

X_train, X_test, y_train, y_test = train_test_split(x_section, y, test_size=0.2, random_state=42)

# 2 modeling
# svm
#clf = svm.SVC(probability=True)
#clf.fit(X_train, y_train)

# decision tree
#from sklearn import tree
#clf = tree.DecisionTreeClassifier()
#clf = clf.fit(X_train, y_train)

# knn
#from sklearn.neighbors import KNeighborsClassifier
#clf = KNeighborsClassifier(n_neighbors=3)
#clf.fit(X_train, y_train)

innn = (interval*2+1)*4
# simple NN
import tensorflow as tf
model = tf.keras.models.Sequential([
	tf.keras.layers.Dense(128, activation='relu', input_shape=(None,innn)),
	tf.keras.layers.Dropout(0.1),
	tf.keras.layers.Dense(64, activation='relu'),
	tf.keras.layers.Dropout(0.1),
	tf.keras.layers.Dense(32, activation='softmax'),
	tf.keras.layers.Dropout(0.1),
	tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(from_logits=False), metrics=['accuracy'])
model.fit(X_train, y_train, epochs=30)
model.evaluate(X_test,  y_test, verbose=2)
              

#p = clf.predict_proba(X_test)
#print(p[0][1]) -> the prob of class 1
'''
p = clf.predict(X_test)

tn, fp, fn, tp = confusion_matrix(y_test, p).ravel()
print('acc:', (tn+tp)/(tn+fp+fn+tp))
print('spe:', tn/(tn+fn))
print('sen:', tp/(tp+f
'''


'''
p = clf.predict(X_test)
print(confusion_matrix(y_test, p))
    0   0
1[[292 138]
1 [ 85 355]]
acc = 74.4%
'''

# 3 save model
#dump(clf, 'model_svm.joblib') 