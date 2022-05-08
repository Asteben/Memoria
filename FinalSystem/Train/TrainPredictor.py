import pandas as pd
import tensorflow as tf
import numpy as np
from csv import DictReader

from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler

print (tf.__version__)


def load_data ( inputPath ):
    cols = ["Unix", "Quantity"]
    df = pd.read_csv ( inputPath , sep ="," , header = None , names = cols )
    df1 = df [["Unix", "Quantity"]]
    return df1

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

X_train = dataset["Unix"]
Y_train = dataset["Quantity"]

train_series = np.array(Y_train).reshape((len(Y_train), 1))

look_back  = 10

train_generator = TimeseriesGenerator(train_series, train_series, length = look_back, sampling_rate = 1, stride = 1, batch_size = 10)

n_neurons  = 512
model = Sequential()
model.add(LSTM(n_neurons, input_shape=(look_back, 1)))#, return_sequences=True))
#model.add(LSTM(int(n_neurons/2)))#, return_sequences=True))
#model.add(LSTM(int(n_neurons/4)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

model.fit(train_generator,epochs=300, verbose=2)

model.save(f'../Models/Lstm-Corona-Gby{Seconds}')