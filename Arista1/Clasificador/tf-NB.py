import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

corona_data=pd.read_csv('../Data/rawdataset/CoronaTweetsLimpio4.csv',names=['id', 'created_at','text','label'])
earthquake_data=pd.read_csv('../Data/rawdataset/NepalTweetsLimpio.csv',names=['id', 'created_at','text','label'])
hurricane_data=pd.read_csv('../Data/rawdataset/HurricaneTweetsLimpio.csv',names=['id', 'created_at','text','label'])



corona_data['labelnum'] = corona_data.label.map({'covid':1,'other':0})
earthquake_data['labelnum'] = earthquake_data.label.map({'earthquake': 2})
hurricane_data['labelnum'] = hurricane_data.label.map({'hurricane':3})

sample_train = 5000

other_data_train = corona_data.loc[corona_data["labelnum"]==0,:][:sample_train]
corona_data_train = corona_data.loc[corona_data["labelnum"]==1,:][:sample_train]
earthquake_data_train = earthquake_data.loc[earthquake_data["labelnum"]==2,:][:sample_train]
hurricane_data_train = hurricane_data.loc[hurricane_data["labelnum"]==3,:][:sample_train]

df_train = pd.concat([other_data_train,corona_data_train,earthquake_data_train,hurricane_data_train],ignore_index=True)
df_train = df_train.sample(frac=1).reset_index(drop=True)


other_data_test = corona_data.loc[corona_data["labelnum"]==0,:][sample_train:]
corona_data_test = corona_data.loc[corona_data["labelnum"]==1,:][sample_train:]
earthquake_data_test = earthquake_data.loc[earthquake_data["labelnum"]==2,:][sample_train:]
hurricane_data_test = hurricane_data.loc[hurricane_data["labelnum"]==3,:][sample_train:]


df_test = pd.concat([other_data_test,corona_data_test,earthquake_data_test,hurricane_data_test],ignore_index=True)
df_test = df_test.sample(frac=1).reset_index(drop=True)
print(len(df_test))


xtrain = df_train.text
xtest = df_test.text
ytrain = df_train.labelnum
ytest = df_test.labelnum






cv = TfidfVectorizer()
xtrain_dtm = cv.fit_transform(xtrain)
xtest_dtm=cv.transform(xtest)

#df=pd.DataFrame(xtrain_dtm.toarray(),columns=cv.get_feature_names())
'''
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=500)
clf.fit(xtrain_dtm, ytrain)
'''
clf = MultinomialNB().fit(xtrain_dtm,ytrain)

predicted = clf.predict(xtest_dtm)


print('\n Accuracy of the classifier is',metrics.accuracy_score(ytest,predicted))
print('\n Confusion matrix')
print(metrics.confusion_matrix(ytest,predicted))
print('\n The value of Precision', metrics.precision_score(ytest,predicted,average='macro'))
print('\n The value of Recall', metrics.recall_score(ytest,predicted,average='macro'))
'''
covid_data_pred = []
earthquake_data_pred = []
hurricane_data_pred = []

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

print('\n The tweets of covid are:')
print(df_covid.iloc[:5])

print('\n The tweets of earthquake are:')
print(df_earthquake.iloc[:5])

print('\n The tweets of hurricane are:')
print(df_hurricane.iloc[:5])
'''

