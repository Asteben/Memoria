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

def load_data ( inputPath ):
    cols = ["Unix", "Quantity"]
    df = pd.read_csv ( inputPath , sep ="," , header = None , names = cols )
    df1 = df [["Unix", "Quantity"]]
    return df1


dataset_path = "../FinalSystem/Data/corona_full-GBy600s.csv"

raw_dataset = load_data( dataset_path )
dataset = raw_dataset.copy()
dataset.tail()
dataset.isna().sum()
dataset = dataset.dropna()

len_data = len(dataset)
sample_rate = 0.8
look_back  = 200
k = 200

df_train = dataset.iloc[:int(len_data*sample_rate)]
Test = dataset.iloc[int(len_data*sample_rate):]

X_train = df_train["Unix"]
Y_train = df_train["Quantity"]

df_test = pd.concat([df_train.tail(look_back),Test],ignore_index=True)

X_test = df_test["Unix"]
Y_test = df_test["Quantity"]

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
#test_generator = TimeseriesGenerator(test_series, test_series, length = look_back, sampling_rate = 1, stride = 1, batch_size = 10)
'''
n_neurons  = 64
model = Sequential()
model.add(LSTM(n_neurons, input_shape=(look_back, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
'''

model = Sequential()
model.add(LSTM(500, input_shape=(look_back, 1),  kernel_regularizer='l2',return_sequences=True ))
model.add(LSTM(250,kernel_regularizer='l2'))
model.add(Dense(100, activation='relu'))
model.add(Dense(k))

'''
inputs = Input(shape=(look_back, 1))

bd_seq = Bidirectional(LSTM(128, return_sequences=True, kernel_regularizer='l2'), merge_mode='sum')(inputs)
bd_sin = Bidirectional(LSTM(32, return_sequences=True, kernel_regularizer='l2'), merge_mode='sum') (bd_seq)

bd_1 = Bidirectional(LSTM(1), merge_mode='sum')(bd_seq)
bd_2 = Bidirectional(LSTM(1), merge_mode='sum')(bd_sin)
output = Add()([bd_1, bd_2])

model = Model(inputs=inputs, outputs=output)'''
model.compile(loss='mse', optimizer='adam')
model.fit(final,epochs=15, verbose=2)
model.save(f'Multistep-Lstm-GBy600s-LB{look_back}-{k}K')
'''
#model = tf.keras.models.load_model('../FinalSystem/Models/Lstm-LB250-32N-Corona-Gby60')
test_predictions = []
first_eval_batch = np.array(Y_train.tail(look_back).tolist())
current_batch = np.reshape(first_eval_batch,(1, look_back, 1))
for i in range(len(Test.index)):
    
    pred = model.predict(current_batch)[0][0]
    
    test_predictions.append(pred)
    current_batch = np.append(current_batch[:,1:,:],[[pred]],axis=1)
    #print(current_batch)

'''
first_eval_batch = np.array(Y_train.tail(look_back).tolist())
current_batch = np.reshape(first_eval_batch,(1, look_back, 1))
test_predictions = model.predict(current_batch)[0]

fig, ax = plt.subplots(1, 1, figsize=(15, 5))
ax.plot(X_train,Y_train, lw=2, label='train data')
ax.plot(X_test,Y_test, lw=3, c='y', label='test data')
ax.plot(X_test.iloc[look_back:look_back+k],test_predictions, lw=3, c='r',linestyle = ':', label='predictions')
ax.legend(loc="lower left")
plt.show()


