import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import matplotlib . pyplot as plt
from sklearn import metrics

look_back  = 200
k = 90
Seconds = 30

def load_data ( inputPath ):
    cols = ["Unix", "Quantity"]
    df = pd.read_csv ( inputPath , sep ="," , header = None , names = cols )
    df1 = df [["Unix", "Quantity"]]
    dataset = df1.copy()
    dataset.tail()
    dataset.isna().sum()
    dataset = dataset.dropna()
    return dataset

def group_by (sec, csv):
    csv = csv.sort_values(by=['created_at'], ignore_index = True)   
    Data = []
    rowCount = 0                
    rowCountTotal = 0           
    Seconds = sec                
    UnixGroup = 0              
    for row in csv.itertuples(index=False, name=None):
        rowCount = rowCount + 1
        rowCountTotal = rowCountTotal + 1
        Unixinrow = int(row[0])          

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
    
    dataset.isna().sum()
    dataset = dataset.dropna()
    return dataset

def Predecir(test_data_gby,df,modelo):
    df_test = load_data(test_data_gby)

    GBy_Data = group_by(Seconds, df)
    GBy_qt = GBy_Data["Quantity"]
    current_batch = np.reshape(np.array(GBy_qt.head(look_back).tolist()),(1, look_back, 1))
    test_predictions =[]
    i = 0
    while (((i+1)*k)+look_back) < len(df_test.index):
        pred = modelo.predict(current_batch)[0]
        test_predictions.append(pred)
        next_eval_batch = np.array(GBy_qt[(i+1)*k:((i+1)*k)+look_back].tolist())
        current_batch = np.reshape(next_eval_batch,(1, look_back, 1))
        i = i + 1
    test_predictions = np.concatenate(np.array(test_predictions))
    X_test = df_test["Unix"]
    Y_test = df_test["Quantity"]
    x_pred = np.arange(X_test.iloc[look_back],(len(test_predictions)*Seconds)+X_test.iloc[look_back],Seconds)
    largo = len(test_predictions)
    error_mse = tf.keras.losses.mean_squared_error(Y_test.iloc[look_back:look_back+largo], test_predictions)
    print (f'Error cuadratico medio: {error_mse.numpy()}')
    fig, ax = plt.subplots(1, 1, figsize=(15, 5))
    ax.plot(X_test,Y_test, lw=3, c='y', label='test data')
    ax.plot(x_pred,test_predictions, lw=3, c='r',linestyle = ':', label='predictions')
    ax.legend(loc="lower left")
    ax.set(xlabel="Tiempo", ylabel="Número de tweets")
    plt.show()

corona_data=pd.read_csv('Data/Test_data/TestCovid.csv',names=['created_at','text','label'])
earthquake_data=pd.read_csv('Data/Test_data/TestEarthquake.csv',names=['created_at','text','label'])
hurricane_data=pd.read_csv('Data/Test_data/TestHurricane.csv',names=['created_at','text','label'])


corona_data['labelnum'] = corona_data.label.map({'covid':1,'other':0})
earthquake_data['labelnum'] = earthquake_data.label.map({'earthquake': 2})
hurricane_data['labelnum'] = hurricane_data.label.map({'hurricane':3})

other_data_test = corona_data.loc[corona_data["labelnum"]==0,:]
corona_data_test = corona_data.loc[corona_data["labelnum"]==1,:]
earthquake_data_test = earthquake_data.loc[earthquake_data["labelnum"]==2,:]
hurricane_data_test = hurricane_data.loc[hurricane_data["labelnum"]==3,:]

df_test = pd.concat([other_data_test,corona_data_test,earthquake_data_test,hurricane_data_test],ignore_index=True)
df_test['created_at'] = df_test['created_at'].astype(int)
df_test = df_test.sort_values(by='created_at', ascending=True).reset_index(drop=True)

print(len(df_test))

tmp_test = df_test["created_at"]
xtest_classify = df_test["text"]
ytest_classify = df_test["labelnum"]

NLP_path = 'Models/Classifier_models/NLP'
Classifier_path = 'Models/Classifier_models/Classifier'

cv = joblib.load(f'{NLP_path}/CountVectorizer.pkl')
clf = joblib.load(f'{Classifier_path}/NaiveBayes-CV.pkl')

model_covid = tf.keras.models.load_model(f'Models/LSTM_models_2/Covid/{Seconds}s/{look_back}LB-{k}K')
model_earthquake = tf.keras.models.load_model(f'Models/LSTM_models_2/Earthquake/{Seconds}s/{look_back}LB-{k}K')
model_hurricane = tf.keras.models.load_model(f'Models/LSTM_models_2/Hurricane/{Seconds}s/{look_back}LB-{k}K')

covid_data_classified = []
earthquake_data_classified = []
hurricane_data_classified = []

xtest_dtm=cv.transform(xtest_classify)
all_data_classified = clf.predict(xtest_dtm)

aux = 0
for i in all_data_classified:
    if int(i) == 1:
        covid_data_classified.append(df_test.iloc[aux])
        aux = aux + 1
    elif int(i) == 2:
        earthquake_data_classified.append(df_test.iloc[aux])
        aux = aux + 1
    elif int(i) == 3:
        hurricane_data_classified.append(df_test.iloc[aux])
        aux = aux + 1
    else:
        aux = aux + 1
    print(f"Clasificando:{aux}", end='\r')
print(f"Clasificando:{aux}")

df_covid = pd.DataFrame(covid_data_classified)
df_earthquake = pd.DataFrame(earthquake_data_classified)
df_hurricane = pd.DataFrame(hurricane_data_classified)

print('\n Accuracy of the classifier is',metrics.accuracy_score(ytest_classify,all_data_classified))
print('\n Confusion matrix')
print(metrics.confusion_matrix(ytest_classify,all_data_classified))
print('\n The value of Precision', metrics.precision_score(ytest_classify,all_data_classified,average='macro'))
print('\n The value of Recall', metrics.recall_score(ytest_classify,all_data_classified,average='macro'))

covid_data_pred = Predecir(f'Data/Gby_data/{Seconds}s/TestCovid.csv',df_covid,model_covid)
hurricane_data_pred = Predecir(f'Data/Gby_data/{Seconds}s/TestHurricane.csv',df_hurricane,model_hurricane)
earthquake_data_pred = Predecir(f'Data/Gby_data/{Seconds}s/TestEarthquake.csv',df_earthquake,model_earthquake)

