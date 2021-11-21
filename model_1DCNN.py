'''
Model: 1D CNN 
Input: [0-110km's speed and volume -> 5 & 10 mins before the predicted time as temporal data](shape=(4, 110)) 
Output: [0-110km's risk](e.g., [0,0,1,...,0])
Label: class 1 -> an accident has occured; class 0 -> normal
'''

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from numpy.lib.stride_tricks import sliding_window_view
import os
from tensorflow import keras
from tensorflow.keras import layers

# 0 params
num_features = 4 # 5, 10 min speed + 5, 10 min volume, weather to be added
sequence_length = 110 # kms
#batch_size = 32
epochs = 10

def create_model():
	model = keras.models.Sequential([

		keras.layers.Conv1D(64, 5, strides=1, input_shape=(sequence_length, num_features)),
		# padding='causal' results in dilatation, useful in temporal data where the model should not violate the temporal order
		#keras.layers.BatchNormalization(),
		keras.layers.ReLU(),
		keras.layers.Conv1D(64, 5, strides=2),
		#keras.layers.BatchNormalization(),
		keras.layers.ReLU(),
		keras.layers.Dropout(0.1),
		
		keras.layers.Conv1D(128, 5, strides=1),
		#keras.layers.BatchNormalization(),
		keras.layers.ReLU(),
		keras.layers.Conv1D(128, 5, strides=2),
		#keras.layers.BatchNormalization(),
		keras.layers.ReLU(),
		keras.layers.Dropout(0.1),

		keras.layers.Conv1D(256, 5, strides=1),
		#keras.layers.BatchNormalization(),
		keras.layers.ReLU(),
		keras.layers.Conv1D(256, 5, strides=2),
		#keras.layers.BatchNormalization(),
		keras.layers.ReLU(),
		keras.layers.Dropout(0.1),

		keras.layers.Flatten(),
		#keras.layers.Dense(512, activation='relu'),
		#keras.layers.Dropout(0.1),
		keras.layers.Dense(sequence_length),
		keras.layers.Activation('sigmoid')

		])
		

	model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0005, beta_1=0.9, beta_2=0.999), loss=tf.keras.losses.mean_squared_error)

	return model

# 1 prepare data
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

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# 2 modeling
model = create_model()
model.summary()
model.fit(X_train, y_train, epochs=epochs, shuffle=True, validation_split=0.2)
model.save('model_1DCNN')

'''
p = model.predict(X_test)
print(p[0])
print(y_test[0])
'''