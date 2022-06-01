import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import matplotlib . pyplot as plt
from sklearn import metrics

look_back  = 200
k = 60
Seconds = 60

corona_data=pd.read_csv('../Data/Test_data/TestCovid.csv',names=['created_at','text','label'])
earthquake_data=pd.read_csv('../Data/Test_data/TestEarthquake.csv',names=['created_at','text','label'])
hurricane_data=pd.read_csv('../Data/Test_data/TesttHurricane.csv',names=['created_at','text','label'])

corona_GBy_data=pd.read_csv(f'Data/Gby_data/{Seconds}s/TestCovid.csv',names=['Unix','Quantity'])
earthquake_GBy_data=pd.read_csv(f'Data/Gby_data/{Seconds}s/TestEarthquake.csv',names=['Unix','Quantity'])
hurricane_GBy_data=pd.read_csv(f'Data/Gby_data/{Seconds}s/TestHurricane.csv',names=['Unix','Quantity'])


corona_data['labelnum'] = corona_data.label.map({'covid':1,'other':0})
earthquake_data['labelnum'] = earthquake_data.label.map({'earthquake': 2})
hurricane_data['labelnum'] = hurricane_data.label.map({'hurricane':3})


other_data_test = corona_data.loc[corona_data["labelnum"]==0,:]
corona_data_test = corona_data.loc[corona_data["labelnum"]==1,:]
earthquake_data_test = earthquake_data.loc[earthquake_data["labelnum"]==2,:]
hurricane_data_test = hurricane_data.loc[hurricane_data["labelnum"]==3,:]


df_test = pd.concat([other_data_test,corona_data_test,earthquake_data_test,hurricane_data_test],ignore_index=True)
df_test = df_test.sort_values(by=['created_at']).reset_index(drop=True)

print(len(df_test))

tmp_test = df_test.created_at
xtest_classify = df_test.text
ytest_classify = df_test.labelnum

NLP_path = '../Models/Classifier_models/NLP'
Classifier_path = '../Models/Classifier_models/Classifier'

cv = joblib.load(f'{NLP_path}/TfidfVectorizer.pkl')
clf = joblib.load(f'{Classifier_path}/NaiveBayes-tfidf.pkl')

all__data_classified = []


current_time_batch = look_back*Seconds
j=0
for i in range (len(df_test.index)):
    if int(tmp_test[i]) < current_time_batch :
        i=i
    else:
        covid_data_classified = []
        earthquake_data_classified = []
        hurricane_data_classified = []

        xtest_dtm=cv.transform(xtest_classify[j:i]) # atencion el i:j
        current_data_clasified = clf.predict(xtest_dtm)

        aux = j
        for l in current_data_clasified:
            if int(l) == 1:
                covid_data_classified.append(df_test.iloc[aux])
                aux = aux + 1
            elif int(l) == 2:
                earthquake_data_classified.append(df_test.iloc[aux])
                aux = aux + 1
            elif int(l) == 3:
                hurricane_data_classified.append(df_test.iloc[aux])
                aux = aux + 1
            else:
                aux = aux + 1
        
    pred = model.predict(current_batch)[0]
    test_predictions.append(pred)
    next_eval_batch = np.array(Y_test[(i+1)*k:((i+1)*k)+look_back].tolist())
    current_batch = np.reshape(next_eval_batch,(1, look_back, 1))
    i = i + 1

xtest_dtm=cv.transform(xtest)
predicted = clf.predict(xtest_dtm)


print('\n Accuracy of the classifier is',metrics.accuracy_score(ytest_classify,all__data_classified))
print('\n Confusion matrix')
print(metrics.confusion_matrix(ytest_classify,all__data_classified))
print('\n The value of Precision', metrics.precision_score(ytest_classify,all__data_classified,average='macro'))
print('\n The value of Recall', metrics.recall_score(ytest_classify,all__data_classified,average='macro'))

covid_data_classified = []
earthquake_data_classified = []
hurricane_data_classified = []

aux = 0
for i in predicted:
    if int(i) == 1:
        covid_data_pred.append(df_test.iloc[aux])
        aux = aux + 1
    elif int(i) == 2:
        earthquake_data_pred.append(df_test.iloc[aux])
        aux = aux + 1
    elif int(i) == 3:
        hurricane_data_pred.append(df_test.iloc[aux])
        aux = aux + 1
    else:
        aux = aux + 1
    print(aux, end='\r')
        
df_covid = pd.DataFrame(covid_data_pred)
df_earthquake = pd.DataFrame(earthquake_data_pred)
df_hurricane = pd.DataFrame(hurricane_data_pred)


import tensorflow as tf
import matplotlib . pyplot as plt

print (tf.__version__)



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
    
  
    
def Predecir (sec, data_clas, data_test, modelo):
    GBy_data = group_by(sec, data_clas)
    Last_Unix = GBy_data.tail(1)['Unix'].tolist()[0]
    Init_Unix = data_test.head(1)['Unix'].tolist()[0]
    rowCount = 0
    look_back  = 20
    DataT = []
    for row in data_test.itertuples(index=False, name=None):
        DataT.append({'Unix' :(Last_Unix + (int(row[0]) - Init_Unix)), 'Quantity' : int(row[1])})  
        #data_test.iloc[rowCount]['Unix'] = Last_Unix + (row[0] - Init_Unix)
        rowCount = rowCount + 1
    Test = pd.DataFrame(DataT)
    
    df_test = pd.concat([GBy_data.tail(look_back),Test],ignore_index=True)
    X_test = df_test["Unix"]
    Y_test = df_test["Quantity"]
    '''
    test_series = np.array(Y_test).reshape((len(Y_test), 1))
    test_generator = TimeseriesGenerator(test_series, test_series, length = look_back, sampling_rate = 1, stride = 1, batch_size = 10)
    test_predictions = modelo.predict(test_generator)
    '''
    test_predictions = []
    #Select last n_input values from the train data
    #data_to_scaler = (np.array(GBy_data.tail(look_back)['Quantity'].tolist())).reshape(-1, 1)
    #first_eval_batch =  scaler.transform(data_to_scaler)
    first_eval_batch = np.array(GBy_data.tail(look_back)['Quantity'].tolist())
    
    current_batch = np.reshape(first_eval_batch,(1, look_back, 1))
    for i in range(len(data_test.index)):
    
        pred = modelo.predict(current_batch)[0]
    
        test_predictions.append(pred)
        current_batch = np.append(current_batch[:,1:,:],[[pred]],axis=1)
    
    #test_predictions = scaler.inverse_transform(test_predictions)
    fig, ax = plt.subplots(1, 1, figsize=(15, 5))
    ax.plot(X_test,Y_test, lw=3, c='y', label='test data')
    ax.plot(X_test.iloc[look_back:],test_predictions, lw=3, c='r',linestyle = ':', label='predictions')
    ax.legend(loc="lower left")
    plt.show()

corona_GBy_data=pd.read_csv('Data/TestCorona-GBy60s.csv',names=['Unix','Quantity'])
#earthquake_GBy_data=pd.read_csv('Data/TestEarthquake-GBy60s.csv',names=['Unix','Quantity'])
#hurricane_GBy_data=pd.read_csv('Data/TestHurricane-GBy60s.csv',names=['Unix','Quantity'])


model_Corona = tf.keras.models.load_model('Models/OldLSTMs/Lstm-Corona-Gby60')
#scaler = joblib.load('Models/Corona-GBy60-scaler.pkl') 
predict_covid = Predecir(60, df_covid, corona_GBy_data, model_Corona)
'''
model_Earthquake = tf.keras.models.load_model('Models/Lstm-Earthquake-Gby60')
predict_earthqueake = Predecir(60, df_earthquake, earthquake_GBy_data, model_Earthquake)

model_Hurricane = tf.keras.models.load_model('Models/Lstm-Hurricane-Gby60')
predict_hurricane = Predecir(60, df_hurricane, hurricane_GBy_data, model_Hurricane)
'''
    