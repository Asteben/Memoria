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
    dataset = df1.copy()
    dataset.tail()
    dataset.isna().sum()
    dataset = dataset.dropna()
    return dataset

look_back  = 200
k = 60
Seconds = 60
emergency = 'Covid'


dataset_path_test = f'../Data/Gby_data/{Seconds}s/Test{emergency}.csv'

df_test = load_data(dataset_path_test)

X_test = df_test["Unix"]
Y_test = df_test["Quantity"]

model = tf.keras.models.load_model(f'../Models/LSTM_models/{emergency}/{look_back}LB-{k}K')

first_eval_batch = np.array(Y_test.head(look_back).tolist())
current_batch = np.reshape(first_eval_batch,(1, look_back, 1))
test_predictions =[]
i = 0
while (((i+1)*k)+look_back) < len(df_test.index):
    pred = model.predict(current_batch)[0]
    test_predictions.append(pred)
    next_eval_batch = np.array(Y_test[(i+1)*k:((i+1)*k)+look_back].tolist())
    current_batch = np.reshape(next_eval_batch,(1, look_back, 1))
    i = i + 1

test_predictions = np.concatenate(np.array(test_predictions))

fig, ax = plt.subplots(1, 1, figsize=(15, 5))
ax.plot(X_test,Y_test, lw=3, c='y', label='test data')
ax.plot(X_test.iloc[look_back:look_back+k*i],test_predictions, lw=3, c='r',linestyle = ':', label='predictions')
ax.legend(loc="lower left")
plt.show()


