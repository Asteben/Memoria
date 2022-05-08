import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

corona_data=pd.read_csv('../Data/TrainCorona.csv',names=['created_at','text','label'])
earthquake_data=pd.read_csv('../Data/TrainEarthquake.csv',names=['created_at','text','label'])
hurricane_data=pd.read_csv('../Data/TrainHurricane.csv',names=['created_at','text','label'])



corona_data['labelnum'] = corona_data.label.map({'covid':1,'other':0})
earthquake_data['labelnum'] = earthquake_data.label.map({'earthquake': 2})
hurricane_data['labelnum'] = hurricane_data.label.map({'hurricane':3})

sample_train = 10000

other_data_train = corona_data.loc[corona_data["labelnum"]==0,:][:sample_train]
corona_data_train = corona_data.loc[corona_data["labelnum"]==1,:][:sample_train]
earthquake_data_train = earthquake_data.loc[earthquake_data["labelnum"]==2,:][:sample_train]
hurricane_data_train = hurricane_data.loc[hurricane_data["labelnum"]==3,:][:sample_train]

df_train = pd.concat([other_data_train,corona_data_train,earthquake_data_train,hurricane_data_train],ignore_index=True)
df_train = df_train.sample(frac=1).reset_index(drop=True)


xtrain = df_train.text
ytrain = df_train.labelnum


cv = CountVectorizer()
xtrain_dtm = cv.fit_transform(xtrain)

clf = MultinomialNB().fit(xtrain_dtm,ytrain)

joblib.dump(cv,'../Models/CountVc.pkl')
joblib.dump(clf, '../Models/NaiveBayes-CountVc.pkl')