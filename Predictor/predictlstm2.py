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


dataset_path = "dataset/dataset1h_5.csv"
#dataset_path = "dataset/dataset1h_60.csv"
#dataset_path = "dataset/dataset1h_180.csv"
#dataset_path = "dataset/dataset1h_300.csv"

raw_dataset = load_data( dataset_path )
dataset = raw_dataset.copy()
dataset.tail()
dataset.isna().sum()
dataset = dataset.dropna()

len_data = len(dataset)
sample_rate = 0.8

X_train = dataset["Unix"].iloc[:int(len_data*sample_rate)]
Y_train = dataset["Quantity"].iloc[:int(len_data*sample_rate)]

X_test = dataset["Unix"].iloc[int(len_data*sample_rate):]
Y_test = dataset["Quantity"].iloc[int(len_data*sample_rate):]

train_series = np.array(Y_train).reshape((len(Y_train), 1))
test_series = np.array(Y_test).reshape((len(Y_test), 1))

look_back  = 20

train_generator = TimeseriesGenerator(train_series, train_series, length = look_back, sampling_rate = 1, stride = 1, batch_size = 10)

test_generator = TimeseriesGenerator(test_series, test_series, length = look_back, sampling_rate = 1, stride = 1, batch_size = 10)

n_neurons  = 64
model = Sequential()
model.add(LSTM(n_neurons, input_shape=(look_back, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

model.fit(train_generator,epochs=25, verbose=2)

test_predictions = model.predict(test_generator)


fig, ax = plt.subplots(1, 1, figsize=(15, 5))
ax.plot(X_train,Y_train, lw=2, label='train data')
ax.plot(X_test,Y_test, lw=3, c='y', label='test data')
ax.plot(X_test.iloc[look_back:],test_predictions, lw=3, c='r',linestyle = ':', label='predictions')
ax.legend(loc="lower left")
plt.show()


