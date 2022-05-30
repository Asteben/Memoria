import matplotlib . pyplot as plt
import pandas as pd
import tensorflow as tf
import numpy as np

from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

print (tf.__version__)


def load_data ( inputPath ):
    cols = ["Unix", "Quantity"]
    df = pd.read_csv ( inputPath , sep ="," , header = None , names = cols )
    df1 = df [["Unix", "Quantity"]]
    return df1


dataset_path = "../Data/dataset/dataset_groupby_60s.csv"

raw_dataset = load_data( dataset_path )
dataset = raw_dataset.copy()
dataset.tail()
dataset.isna().sum()
dataset = dataset.dropna()

len_data = len(dataset)
sample_rate = 0.8
look_back  = 250

df_train = dataset.iloc[:int(len_data*sample_rate)]
Test = dataset.iloc[int(len_data*sample_rate):]

X_train = df_train["Unix"]
Y_train = df_train["Quantity"]

df_test = pd.concat([df_train.tail(look_back),Test],ignore_index=True)

X_test = df_test["Unix"]
Y_test = df_test["Quantity"]
'''
train_series = np.array(Y_train).reshape((len(Y_train), 1))
#test_series = np.array(Y_test).reshape((len(Y_test), 1))


train_generator = TimeseriesGenerator(train_series, train_series, length = look_back, sampling_rate = 1, stride = 1, batch_size = 10)

#test_generator = TimeseriesGenerator(test_series, test_series, length = look_back, sampling_rate = 1, stride = 1, batch_size = 10)

n_neurons  = 64
model = Sequential()
model.add(LSTM(n_neurons, input_shape=(look_back, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

model.fit(train_generator,epochs=150, verbose=2)
'''
model = tf.keras.models.load_model('../FinalSystem/Models/Lstm-LB250-32N-Corona-Gby60')
test_predictions = []
first_eval_batch = np.array(Y_train.tail(look_back).tolist())
current_batch = np.reshape(first_eval_batch,(1, look_back, 1))
for i in range(len(Test.index)):
    
    pred = model.predict(current_batch)[0][0]
    
    test_predictions.append(pred)
    current_batch = np.append(current_batch[:,1:,:],[[pred]],axis=1)
    #print(current_batch)
#test_predictions = model.predict(test_generator)


fig, ax = plt.subplots(1, 1, figsize=(15, 5))
ax.plot(X_train,Y_train, lw=2, label='train data')
ax.plot(X_test,Y_test, lw=3, c='y', label='test data')
ax.plot(X_test.iloc[look_back:],test_predictions, lw=3, c='r',linestyle = ':', label='predictions')
ax.legend(loc="lower left")
plt.show()


