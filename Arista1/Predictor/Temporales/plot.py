import matplotlib . pyplot as plt
import pandas as pd
import tensorflow as tf
import numpy as np

from tensorflow import keras
from tensorflow.keras import layers

print (tf.__version__)


def load_data ( inputPath ):
    cols = ["Unix", "Quantity"]
    df = pd.read_csv ( inputPath , sep ="," , header = None , names = cols )
    df1 = df [["Unix", "Quantity"]]
    return df1


dataset_path = "dataset/dataset1h_30.csv"
#dataset_path = "dataset/dataset1h_60.csv"
#dataset_path = "dataset/dataset1h_180.csv"
#dataset_path = "dataset/dataset1h_300.csv"

raw_dataset = load_data( dataset_path )
dataset = raw_dataset.copy()
dataset.tail()
dataset.isna().sum()
dataset = dataset.dropna()

len_data = len(dataset)

X_train = dataset["Unix"].iloc[:int(len_data*0.8)]
Y_train = dataset["Quantity"].iloc[:int(len_data*0.8)]

X_test = dataset["Unix"].iloc[int(len_data*0.8):]
Y_test = dataset["Quantity"].iloc[int(len_data*0.8):]

print(X_test)
print(X_test.iloc[20:])

'''
fig, ax = plt.subplots(1, 1, figsize=(15, 5))
ax.plot(X_train,Y_train, lw=2, label='train data')
ax.plot(X_test,Y_test, lw=3, c='y', label='test data')
ax.legend(loc="lower left")
plt.show()
'''