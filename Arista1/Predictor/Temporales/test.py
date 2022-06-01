import numpy as np
import pandas as pd
data = [[1,6],
        [2,7],
        [3,8],
        [4,9],
        [5,10]]
data = pd.DataFrame(data)

#print(data)
#print(data.iloc[:int(len(data)*0.8)])
#print(data.iloc[int(len(data)*0.8):])
#print(data)
#data_reshape = data.reshape(len(data),1)

#print(data_reshape)



import matplotlib . pyplot as plt
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

print (tf.__version__)


def load_data ( inputPath ):
    cols = ["Unix", "Quantity"]
    df = pd.read_csv ( inputPath , sep ="," , header = None , names = cols )
    df1 = df [["Unix", "Quantity"]]
    return df1


dataset_path = "dataset/dataset1h_30.csv"
raw_dataset = load_data( dataset_path )
dataset = raw_dataset.copy()
dataset.tail()

dataset.isna().sum()
dataset = dataset.dropna()
#print(dataset)
dataset["Unix"] = dataset["Unix"] - dataset["Unix"].iloc[0]
#print(dataset)
train_dataset = dataset.sample( frac = 0.8 , random_state =0)
test_dataset = dataset.drop( train_dataset.index )


train_stats = train_dataset.describe()
train_stats.pop ("Quantity")
train_stats = train_stats.transpose()

train_labels = train_dataset.pop("Quantity")
test_labels = test_dataset.pop("Quantity")
#print(np.array(train_labels))
#print ( dataset ["Unix"] )


def norm (x):
    return (x - train_stats ["mean"]) / train_stats ["std"]


#normed_train_data = norm( train_dataset )
#normed_test_data = norm( test_dataset )
normed_train_data_aux = np.array(norm( train_dataset ))
normed_test_data_aux = np.array(norm( test_dataset ))

normed_train_data = normed_train_data_aux.reshape(len(normed_train_data_aux),1,1)
normed_test_data = normed_test_data_aux.reshape(len(normed_test_data_aux),1,1)

len_data = len(dataset)

print("######## TRAIN_X ########")
print(dataset["Unix"].iloc[:int(len_data*0.8)])
print("######## TEST_X ########")
print(dataset["Unix"].iloc[int(len_data*0.8):])
print("######## TRAIN_Y ########")
print(dataset["Quantity"].iloc[:int(len_data*0.8)])
print("######## TEST_Y ########")
print(dataset["Quantity"].iloc[int(len_data*0.8):])
print(len(dataset["Quantity"].iloc[int(len_data*0.8):]))