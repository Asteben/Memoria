import pandas as pd
#import tensorflow as tf
import numpy as np
from csv import DictReader
from numpy import split
from numpy import array

from keras.preprocessing.sequence import TimeseriesGenerator
n = 7
k=3
X_train = np.arange(1,21,1)
Y_train = X_train[(n):]
X_train = X_train[:-k+1]
Y_train = np.append(Y_train, 0) 


print(X_train)
print(X_train[n:n+k])
