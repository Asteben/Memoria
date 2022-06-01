import pandas as pd
import tensorflow as tf
import numpy as np
from csv import DictReader

from keras.preprocessing.sequence import TimeseriesGenerator
from keras import (Model, Input)
from keras.models import Sequential
from keras.layers import (LSTM, Dense, Activation, BatchNormalization, Dropout, Bidirectional, Add)
from sklearn.preprocessing import MinMaxScaler
import joblib

print (tf.__version__)

Data = []
with open('../Data/TrainCorona.csv', 'r', encoding="utf-8") as csv_obj:
    csv = DictReader(x.replace('\0', '') for x in csv_obj)
    csv = sorted(csv, key = lambda row: row['created_at'])      

    rowCount = 0                
    rowCountTotal = 0           
    Seconds = 60              
    UnixGroup = 0               
    for  row in csv:
        rowCount = rowCount + 1
        rowCountTotal = rowCountTotal + 1
        Unixinrow = int(row['created_at'])          

        if rowCount == 1:
            UnixGroup = (int(Unixinrow/Seconds))*Seconds
        
        if Unixinrow - UnixGroup > Seconds :                
            Data.append({'Unix' : int(UnixGroup), 'Quantity' : (rowCount-1)})
            UnixGroup = UnixGroup + Seconds                 
            while Unixinrow - (UnixGroup) > Seconds :       
                Data.append({'Unix' :int(UnixGroup), 'Quantity' : 0})
                UnixGroup = UnixGroup + Seconds
            rowCount = 1
        print(f'RowCountTotal:{rowCountTotal}', end = '\r')
    Data.append({'Unix' : int(UnixGroup), 'Quantity' : (rowCount)}) #Ultimo grupo




dataset = pd.DataFrame(Data)
dataset = dataset[['Unix', 'Quantity']]
dataset.tail()
dataset.isna().sum()
dataset = dataset.dropna()
'''
X_train = np.array(dataset["Unix"].tolist())
Y_train =(np.array(dataset["Quantity"].tolist())).reshape(-1, 1)
'''
X_train = dataset["Unix"]
Y_train = dataset["Quantity"]

#scaler = MinMaxScaler()
#scaler.fit(Y_train)
#Y_train = scaler.transform(Y_train)
#joblib.dump(scaler,'../Models/Corona-GBy60-scaler.pkl')

train_series = np.array(Y_train).reshape((len(Y_train), 1))

look_back  = 250
k = 1

train_generator = TimeseriesGenerator(train_series, train_series, length = look_back, sampling_rate = 1, stride = 1, batch_size = 10)

n_neurons  = 32
model = Sequential()
model.add(LSTM(n_neurons, kernel_regularizer='l2', input_shape=(look_back, 1),return_sequences=True))
model.add(Dense(k))
model.compile(optimizer='adam', loss='mse')

model.fit(train_generator,epochs=200, verbose=2)
'''
inputs = Input(shape=(look_back, 1))

bd_seq = Bidirectional(LSTM(128, return_sequences=True, kernel_regularizer='l2'), merge_mode='sum')(inputs)
bd_sin = Bidirectional(LSTM(32, return_sequences=True, kernel_regularizer='l2'), merge_mode='sum') (bd_seq)

bd_1 = Bidirectional(LSTM(1, activation='linear'), merge_mode='sum')(bd_seq)
bd_2 = Bidirectional(LSTM(1, activation='tanh'), merge_mode='sum')(bd_sin)
output = Add()([bd_1, bd_2])

model = Model(inputs=inputs, outputs=output)
model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(train_generator,epochs=100, verbose=2)
'''
model.save(f'../Models/Lstm-LB250-32N-Corona-Gby{Seconds}')