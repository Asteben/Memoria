import pandas as pd
#import tensorflow as tf
import numpy as np
from csv import DictReader
from numpy import split

from keras.preprocessing.sequence import TimeseriesGenerator
n = 250
k=60
X_train = np.arange(1,9811,1)

Y_train = X_train
#Y_train = X_train[(n):]

X_train = X_train[:-(k)]
X_train = np.append(X_train,0)


#print(X_train)
#print(Y_train)
train_series_x = np.array(X_train).reshape(len(X_train), 1)
y=[]

for i in range (0,len(Y_train)-(k-1)):
    for j in range (0,k):
        y.append(Y_train[i+j])
y = np.array(y)
y = split(y, len(y)/k)
#print(y)
#train_series = split(train_series, len(train_series)/k)
#print(train_series)



'''
train_generator = TimeseriesGenerator(train_series, train_series, length = n, sampling_rate = 1, stride = 1, batch_size = 1)
train_generator2 = TimeseriesGenerator(Y_train, Y_train, length = k, sampling_rate = 1, stride = 1, batch_size = 1)

#Y_train = array(split(X_train, len(X_train)/k))
print(train_generator.__len__())
print(train_generator2.__len__())
input = [x[0] for x in train_generator]
#print(input)
output = [x[0] for x in train_generator2]
#print(output)
'''
final = TimeseriesGenerator(train_series_x, y, length = n, batch_size = 1)
print(final.__getitem__(final.__len__()-1))