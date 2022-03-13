# Ignore  the warnings
import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

# data visualisation and manipulation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
#configure
# sets matplotlib to inline and displays graphs below the corressponding cell.
#% matplotlib inline  
style.use('fivethirtyeight')
sns.set(style='whitegrid',color_codes=True)

#nltk
import nltk

#preprocessing
from nltk.corpus import stopwords  #stopwords
from nltk import word_tokenize,sent_tokenize # tokenizing
from nltk.stem import PorterStemmer,LancasterStemmer  # using the Porter Stemmer and Lancaster Stemmer and others
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer  # lammatizer from WordNet

# for part-of-speech tagging
from nltk import pos_tag

# for named entity recognition (NER)
from nltk import ne_chunk

# vectorizers for creating the document-term-matrix (DTM)
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer

# BeautifulSoup libraray
from bs4 import BeautifulSoup 

import re # regex

#model_selection
from sklearn.model_selection import train_test_split,cross_validate
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

#evaluation
from sklearn.metrics import accuracy_score,roc_auc_score 
from sklearn.metrics import classification_report
from mlxtend.plotting import plot_confusion_matrix

#preprocessing scikit
from sklearn.preprocessing import MinMaxScaler,StandardScaler,LabelEncoder
from sklearn.impute import SimpleImputer

#classifiaction.
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC,SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB,MultinomialNB
 
#stop-words
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')
stop_words=set(nltk.corpus.stopwords.words('english'))

#keras
import keras
from keras.preprocessing.text import one_hot,Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense , Flatten , Embedding , Input , CuDNNLSTM , LSTM
from keras.models import Model
from keras.preprocessing.text import text_to_word_sequence
from keras.utils import np_utils
from tensorflow.keras import optimizers


#gensim w2v
#word2vec
from gensim.models import Word2Vec

rev_frame=pd.read_csv(r'../dataset/dataset3.csv')

df=rev_frame.copy()

df.head()
print(df.loc[[159220]])

""""
df=df[["created_at","text"]]

df["review"]=df["text"]
df["rating"]=df["mark"]
#df.drop(['Text','Score'],axis=1,inplace=True)
"""
#print(df.shape)
#df.head(5)
#print(df.loc[[159220]])

#print(df['rating'].isnull().sum())
#df['review'].isnull().sum()  # no null values.

# remove duplicates/ for every duplicate we will keep only one row of that type. 
#df.drop_duplicates(subset=['rating','review'],keep='first',inplace=True) 

print(df.shape)
df.head()

"""
for review in df['review'][:5]:
    print(review+'\n'+'\n')

def mark_sentiment(rating):
  if(rating<=3):
    return 0
  else:
    return 1

df['sentiment']=df['rating'].apply(mark_sentiment)
df.drop(['rating'],axis=1,inplace=True)
df.head()
df['sentiment'].value_counts()
"""
# function to clean and pre-process the text.
def clean_reviews(review):  
    
    # 1. Removing html tags
    review_text = BeautifulSoup(review,"lxml").get_text()
    
    # 2. Retaining only alphabets.
    review_text = re.sub("[^a-zA-Z]"," ",review_text)
    
    # 3. Converting to lower case and splitting
    word_tokens= review_text.lower().split()
    
    # 4. Remove stopwords
    le=WordNetLemmatizer()
    stop_words= set(stopwords.words("english"))     
    word_tokens= [le.lemmatize(w) for w in word_tokens if not w in stop_words]
    
    cleaned_review=" ".join(word_tokens)
    return cleaned_review


pos_df=df.loc[df["mark"]==1,:][:50000]
neg_df=df.loc[df["mark"]==0,:][:50000]
pos_df.head()
neg_df.head()

#df=df.loc[:100000]

#combining
df=pd.concat([pos_df,neg_df],ignore_index=True)
print(df.shape)
df.head()

# shuffling rows
df = df.sample(frac=1).reset_index(drop=True)
print(df.shape)  # perfectly fine.
df.head()

# import gensim
# # load Google's pre-trained Word2Vec model.
# pre_w2v_model = gensim.models.KeyedVectors.load_word2vec_format(r'drive/Colab Notebooks/amazon food reviews/GoogleNews-vectors-negative300.bin', binary=True) 

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
sentences=[]
suma=0
for review in df["text"]:
  sents=tokenizer.tokenize(review.strip())
  suma+=len(sents)
  for sent in sents:
    cleaned_sent=clean_reviews(sent)
    sentences.append(cleaned_sent.split()) # can use word_tokenize also.
print(suma)
print(len(sentences))  # total no of sentences

# trying to print few sentences
for te in sentences[:5]:
  print(te,"\n")

import gensim
w2v_model=gensim.models.Word2Vec(sentences=sentences,vector_size=300,window=10,min_count=1)
w2v_model.train(sentences,epochs=10,total_examples=len(sentences))
# embedding of a particular word.
w2v_model.wv.get_vector('covid')

# total numberof extracted words.
vocab=w2v_model.wv.key_to_index
print("The total number of words are : ",len(vocab))
# words most similar to a given word.
w2v_model.wv.most_similar('covid')
# similaraity b/w two words
w2v_model.wv.similarity('corona','covid')

print("The no of words :",len(vocab))
# print(vocab)

# print(vocab)
vocab=list(vocab.keys())

word_vec_dict={}
for word in vocab:
  word_vec_dict[word]=w2v_model.wv.get_vector(word)
print("The no of key-value pairs : ",len(word_vec_dict)) # should come equal to vocab size

# # just check
# for word in vocab[:5]:
#   print(word_vec_dict[word])

# cleaning reviews.
df['clean_review']=df['text'].apply(clean_reviews)

# number of unique words = 56379.

# now since we will have to pad we need to find the maximum lenght of any document.

maxi=-1
for i,rev in enumerate(df['clean_review']):
  tokens=rev.split()
  if(len(tokens)>maxi):
    maxi=len(tokens)
print(maxi)

tok = Tokenizer()
tok.fit_on_texts(df['clean_review'])
vocab_size = len(tok.word_index) + 1
encd_rev = tok.texts_to_sequences(df['clean_review'])

max_rev_len=1565  # max lenght of a review
vocab_size = len(tok.word_index) + 1  # total no of words
embed_dim=300 # embedding dimension as choosen in word2vec constructor

# now padding to have a amximum length of 1565
pad_rev= pad_sequences(encd_rev, maxlen=max_rev_len, padding='post')
pad_rev.shape   # note that we had 100K reviews and we have padded each review to have  a lenght of 1565 words.

# now creating the embedding matrix
embed_matrix=np.zeros(shape=(vocab_size,embed_dim))
for word,i in tok.word_index.items():
  embed_vector=word_vec_dict.get(word)
  if embed_vector is not None:  # word is in the vocabulary learned by the w2v model
    embed_matrix[i]=embed_vector
  # if word is not found then embed_vector corressponding to that vector will stay zero.

# checking.
print(embed_matrix[14])

# prepare train and val sets first
Y=keras.utils.np_utils.to_categorical(df["mark"])  # one hot target as required by NN.
x_train,x_test,y_train,y_test=train_test_split(pad_rev,Y,test_size=0.20,random_state=42)

from keras.initializers import Constant
from keras.layers import ReLU
from keras.layers import Dropout
from tensorflow.keras import layers
import tensorflow as tf

model=Sequential()
model.add(Embedding(input_dim=vocab_size,output_dim=embed_dim,input_length=max_rev_len,embeddings_initializer=Constant(embed_matrix)))
# model.add(CuDNNLSTM(64,return_sequences=False)) # loss stucks at about 
model.add(Flatten())
model.add(Dense(16,activation='relu'))
model.add(Dropout(0.50))
# model.add(Dense(16,activation='relu'))
# model.add(Dropout(0.20))
model.add(Dense(2,activation='sigmoid'))  # sigmod for bin. classification.
'''
model=Sequential([
  keras.Input( shape = x_train.shape),
  layers.Conv2D(32, kernel_size = (3, 3), activation = "relu"),
  layers.MaxPooling2D( pool_size = (2, 2)),
  layers.Conv2D(64, kernel_size = (3, 3), activation = "relu"),
  layers.MaxPooling2D( pool_size = (2, 2)),
  layers.Flatten(),
  layers.Dropout(0.5),
  layers.Dense(2, activation = "softmax") ,
])
'''
model.summary()
# compile the model
#model.compile(optimizer=optimizers.RMSprop(lr=1e-3),loss='binary_crossentropy',metrics=['accuracy'])
#model.compile(optimizer=optimizers.RMSprop(lr=1e-3), loss='mse')
#model.compile( loss = "categorical_crossentropy", optimizer ="adam", metrics =["accuracy"])
model.compile(optimizer=optimizers.Adam(), loss='categorical_crossentropy',  metrics=['accuracy'])

# specify batch size and epocj=hs for training.
epochs=5
batch_size=10

# fitting the model.
# model.fit(x_train,y_train,epochs=epochs,batch_size=batch_size,validation_data=(x_test,y_test))

def limitgpu(maxmem):
	gpus = tf.config.list_physical_devices('GPU')
	if gpus:
		# Restrict TensorFlow to only allocate a fraction of GPU memory
		try:
			for gpu in gpus:
				tf.config.experimental.set_virtual_device_configuration(gpu,
						[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=maxmem)])
		except RuntimeError as e:
			# Virtual devices must be set before GPUs have been initialized
			print(e)


# 1.5GB
limitgpu(1024+512) 

model.fit(x_train, y_train, epochs = epochs, batch_size = batch_size, verbose = 2)

score = model.evaluate(x_test, y_test, batch_size = batch_size)


resultado = model.predict(x_test)

def cero0(temp):
    lista = []
    for aux in temp:
        if aux[0] == 1:
            lista.append(1)
        else:
            lista.append(0)
    return lista

def cero1(temp):
    lista = []
    for aux in temp:
        if aux[1] == 1:
            lista.append(1)
        else:
            lista.append(0)
    return lista

print(cero0(resultado))
print(cero1(resultado))
print(sum(cero0(resultado)))
print(sum(cero1(resultado)))
#print(f'porcentaje de la muestra:{(sum(cero(df["mark"]))/len(df))*100}')
print(f'score en evualate: {score}')
print(f'tasa del resultado 0:{sum(cero0(resultado))/(len(df)*0.2)}')
print(f'tasa del resultado 1:{sum(cero1(resultado))/(len(df)*0.2)}')

