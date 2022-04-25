import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

dataset=pd.read_csv('../Data/rawdataset/CoronaTweetsLimpio4.csv',names=['id', 'created_at','text','label'])

print('The dimensions of the dataset',dataset.shape)

dataset['labelnum'] = dataset.label.map({'covid':1,'other':0})
sample_train = 10000

pos_df_train = dataset.loc[dataset["labelnum"]==1,:][:sample_train]
neg_df_train = dataset.loc[dataset["labelnum"]==0,:][:sample_train]
df_train = pd.concat([pos_df_train,neg_df_train],ignore_index=True)
df_train = df_train.sample(frac=1).reset_index(drop=True)

pos_df_test = dataset.loc[dataset["labelnum"]==1,:][sample_train:]
neg_df_test = dataset.loc[dataset["labelnum"]==0,:][sample_train:]
df_test = pd.concat([pos_df_test,neg_df_test],ignore_index=True)
df_test = df_test.sample(frac=1).reset_index(drop=True)
print(len(df_test))

#splitting the dataset into train and test data
#len_data = len(df)
#sample_rate = 0.75
#df_train = df.loc[:int((len_data*sample_rate)-1)]
#df_test = df.loc[int(len_data*sample_rate):]
xtrain = df_train.text
xtest = df_test.text
ytrain = df_train.labelnum
ytest = df_test.labelnum




#output the words or Tokens in the text documents
cv = CountVectorizer()
xtrain_dtm = cv.fit_transform(xtrain)
xtest_dtm=cv.transform(xtest)

df=pd.DataFrame(xtrain_dtm.toarray(),columns=cv.get_feature_names())

# Training Naive Bayes (NB) classifier on training data.
clf = MultinomialNB().fit(xtrain_dtm,ytrain)
predicted = clf.predict(xtest_dtm)

#printing accuracy, Confusion matrix, Precision and Recall
print('\n Accuracy of the classifier is',metrics.accuracy_score(ytest,predicted))
print('\n Confusion matrix')
print(metrics.confusion_matrix(ytest,predicted))
print('\n The value of Precision', metrics.precision_score(ytest,predicted))
print('\n The value of Recall', metrics.recall_score(ytest,predicted))
print('\n The tweets of covid are:')
aux=0
for i in predicted:
    if int(i) == 1:
        print(f'label:{df_test.iloc[aux].label}     text:{df_test.iloc[aux].text}')
    aux = aux + 1
    if aux == 15:
        break