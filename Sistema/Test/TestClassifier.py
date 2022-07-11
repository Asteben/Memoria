import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import joblib

corona_data=pd.read_csv('../Data/test_data/TestCovid.csv',names=['created_at','text','label'])
earthquake_data=pd.read_csv('../Data/test_data/TestEarthquake.csv',names=['created_at','text','label'])
hurricane_data=pd.read_csv('../Data/test_data/TestHurricane.csv',names=['created_at','text','label'])

corona_data['labelnum'] = corona_data.label.map({'covid':1,'other':0})
earthquake_data['labelnum'] = earthquake_data.label.map({'earthquake': 2})
hurricane_data['labelnum'] = hurricane_data.label.map({'hurricane':3})

other_data_test = corona_data.loc[corona_data["labelnum"]==0,:]
corona_data_test = corona_data.loc[corona_data["labelnum"]==1,:]
earthquake_data_test = earthquake_data.loc[earthquake_data["labelnum"]==2,:]
hurricane_data_test = hurricane_data.loc[hurricane_data["labelnum"]==3,:]

df_test = pd.concat([other_data_test,corona_data_test,earthquake_data_test,hurricane_data_test],ignore_index=True)
df_test = df_test.sample(frac=1).reset_index(drop=True)
print(len(df_test))

xtest = df_test.text
ytest = df_test.labelnum

NLP_path = '../Models/Classifier_models/NLP'
Classifier_path = '../Models/Classifier_models/Classifier'

cv = joblib.load(f'{NLP_path}/TfidfVectorizer.pkl')
xtest_dtm=cv.transform(xtest)

clf = joblib.load(f'{Classifier_path}/NaiveBayes-tfidf.pkl')
predicted = clf.predict(xtest_dtm)


print('\n Accuracy of the classifier is',metrics.accuracy_score(ytest,predicted))
print('\n Confusion matrix')
print(metrics.confusion_matrix(ytest,predicted))
print('\n The value of Precision', metrics.precision_score(ytest,predicted,average='macro'))
print('\n The value of Recall', metrics.recall_score(ytest,predicted,average='macro'))

