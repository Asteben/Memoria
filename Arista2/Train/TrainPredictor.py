import matplotlib . pyplot as plt
import pandas as pd
import tensorflow as tf
import numpy as np

from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import Dense
from keras import (Model, Input)
from keras.layers import (LSTM, Dense, Activation, BatchNormalization, Dropout, Bidirectional, Add)

print (tf.__version__)

look_back  = 200
k = 90
Seconds = 30
emergency = 'Earthquake'

def load_data ( inputPath ):
    cols = ["Unix", "Quantity"]
    df = pd.read_csv ( inputPath , sep ="," , header = None , names = cols )
    df1 = df [["Unix", "Quantity"]]
    return df1

dataset_path = f'../Data/Gby_data/{Seconds}s/Train{emergency}.csv'

raw_dataset = load_data( dataset_path )
dataset = raw_dataset.copy()
dataset.tail()
dataset.isna().sum()
df_train = dataset.dropna()

X_train = df_train["Unix"]
Y_train = df_train["Quantity"]

ye = np.array(Y_train)
x = Y_train[:-(k)]
x = np.append(x,0)
train_series_x = np.array(x).reshape(len(x), 1)
y=[]
for i in range (0,len(ye)-(k-1)):
    for j in range (0,k):
        y.append(ye[i+j])
y = np.array(y)
from numpy import split
y = split(y, len(y)/k)
final = TimeseriesGenerator(train_series_x, y, length =look_back, batch_size = 16)

model = Sequential()
model.add(LSTM(200, input_shape=(look_back, 1)))
model.add(Dense(100, activation='relu'))
model.add(Dense(k))

model.compile(loss='mse', optimizer='adam')
model.fit(final,epochs=30, verbose=2)

model.save(f'../Models/LSTM_models_2/{emergency}/{Seconds}s/{look_back}LB-{k}K')