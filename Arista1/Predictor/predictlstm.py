import pathlib
import matplotlib . pyplot as plt
import pandas as pd
import seaborn as sns
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



#print ( dataset )

train_dataset = dataset.sample( frac = 0.8 , random_state =0)
test_dataset = dataset.drop( train_dataset.index )

# sns.pairplot ( train _ dataset [[" Unix ", "Quantity"]] , diag _ kind =" kde ")
# plt.show ()

train_stats = train_dataset.describe()
train_stats.pop ("Quantity")
train_stats = train_stats.transpose()

train_labels = train_dataset.pop("Quantity")
test_labels = test_dataset.pop("Quantity")
print(np.array(train_labels))

def norm (x):
    return (x - train_stats ["mean"]) / train_stats ["std"]


#normed_train_data = norm( train_dataset )
#normed_test_data = norm( test_dataset )
normed_train_data_aux = np.array(norm( train_dataset ))
normed_test_data_aux = np.array(norm( test_dataset ))

normed_train_data = normed_train_data_aux.reshape(len(normed_train_data_aux),1,1)
normed_test_data = normed_test_data_aux.reshape(len(normed_test_data_aux),1,1)


def build_model() :
    model = keras.Sequential([layers.LSTM ( 64 , input_shape = [1, len(train_dataset.keys())] ) , layers.Dense(64 , activation ="relu") , layers.Dense(1)])
    optimizer = tf.keras.optimizers.RMSprop(0.001)
    model.compile( loss ="mse" , optimizer = optimizer , metrics = ["mae", "mse"] )
    return model


def plot_history ( history ):
    hist = pd.DataFrame ( history.history )
    hist ["epoch"] = history.epoch
    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Mean Abs Error")
    plt.plot( hist ["epoch"] , hist ["mae"] , label ="Train Error")
    plt.plot( hist ["epoch"] , hist ["val_mae"] , label ="Val Error")
    # plt. ylim ([0 ,5000000000000000])
    plt.legend()

    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Mean Square Error")
    plt.plot( hist ["epoch"] , hist ["mse"] , label ="Train Error")
    plt.plot( hist ["epoch"] , hist ["val_mse"] , label ="Val Error")
    # plt. ylim ([0 ,2000000000000000])
    plt.legend()
    plt.show()


class PrintDot ( keras.callbacks.Callback ):
    def on_epoch_end (self , epoch , logs ):
        if(epoch % 100 == 0):
            print ("")
        print ("." , end ="")


EPOCHS = 1000
model = build_model()


early_stop = keras.callbacks.EarlyStopping( monitor ="loss", patience = 10)

history = model.fit( normed_train_data , train_labels , epochs = EPOCHS , validation_split = 0.2 , verbose = 0 , callbacks = [early_stop, PrintDot() ])

#plot_history( history )

loss , mae , mse = model.evaluate( normed_test_data , test_labels , verbose =2)

print("Mean Absolute Error : {:5.2f}".format(mae))

print("Mean Square Error : {:5.2f}".format(mse))

print("Mean Loss : {:5.2f}".format(loss))

test_predictions = model.predict( normed_test_data ).flatten()

plt.scatter( test_labels , test_predictions )
plt.xlabel("True Values Quantity")
plt.ylabel("Predictions Quantity")
plt.axis("equal")
plt.axis("square")
_= plt.plot([ -1000 , 1000] , [ -1000 , 1000])
plt.show()


error = test_predictions - test_labels
plt.hist(error , bins =25)
plt.xlabel("Prediction Error")
_ = plt.ylabel("Count")
plt.show()