import os
import glob
import errno
import json
import numpy as np
import pandas as pd
import nltk


filepath = '../data/bbc/'

labels = []
# Accessing the labels names and saving then in "labels"
files = os.listdir(filepath)
for name in files:
    if os.path.isdir(filepath+name): 
        labels.append(name)

text = []   
texts_aux = [] 
texts_labels = [] 



# Label the content of each article
for label in labels:
    path = filepath+label+'/*.txt'
    files = glob.glob(path)
    for name in files:
        try:
            with open(name, 'r',encoding='ISO-8859-1') as f:
                texts_aux.append(f.read())
                texts_aux.append(label)
        except IOError as exc:
            if exc.errno != errno.EISDIR:
                raise
        texts_labels.append(texts_aux)
        texts_aux=[]
 
# Labeled texts stored in numpy array
texts_labels_np = np.array(texts_labels)

# Labeled texts stored in panda dataframe
df = pd.DataFrame(texts_labels, columns=['text','label'])
#print(text_labels_df.head)


######
######
# Data preparation
######
######
## First step: Tokenize each text
from nltk.tokenize import RegexpTokenizer

## Load library for removing stopwords
from nltk.corpus import stopwords
##nltk.download('stopwords') --> First time has to be downloaded

# Import libraries for stemming
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize
stemmer_ps = PorterStemmer()

from nltk.stem.cistem import Cistem
stemmer_cs = Cistem()

# Import lemmatization libraries
from nltk.stem import WordNetLemmatizer 
lemmatizer = WordNetLemmatizer() 
#nltk.download('wordnet')

# Load stop words 
stop_words = stopwords.words('english')
#print(stop_words[:5])


tokenizer = RegexpTokenizer(r'\w+')
texts_clean = []
texts_aux = []
aux = []

for article in texts_labels_np:
        # Text to lower case
        text = article[0].lower()
        # Tokenize and Remove punctuation
        tokens = tokenizer.tokenize(text)
        # Remove stop words
        tokens = [word for word in tokens if word not in stop_words]
        # Stemming
        for token in tokens:
                aux.append(stemmer_cs.stem(token))
        tokens = aux
        
        texts_aux.append(tokens)
        texts_aux.append(article[1])
        texts_clean.append(texts_aux)
        texts_aux = []
        aux=[]


##
##
# Emedding the data
##
##
# Transforming labels into numbers [business, entertainment, politics, sport, tech] -- [0,1,2,3,4]
for text in texts_clean:
        if text[1]=='business':
                text[1]=0
        if text[1]=='entertainment':
                text[1]=1
        if text[1]=='politics':
                text[1]=2
        if text[1]=='sport':
                text[1]=3
        if text[1]=='tech':
                text[1]=4

text_clean_np = np.array(texts_clean)
text_clean_pd = pd.DataFrame(texts_labels, columns=['text','label'])

tokenized_texts = []
labels = []
for article in texts_clean:
        tokenized_texts.append(article[0])
        labels.append(article[1])

from sklearn.model_selection import train_test_split
import gensim

# Data file path
dataPath = '../data/'

# Train ratio
train_ratio = 0.85

x_train, x_test, t_train, t_test = train_test_split(tokenized_texts, labels, test_size=1 - train_ratio, stratify=labels)
#x_dev, x_test, t_dev, t_test = train_test_split(x_test, t_test, test_size=0.5, stratify=t_test)

##
##
# Word2Vec
##
##

from  gensim.models.doc2vec import TaggedDocument
from  gensim.models.doc2vec import Doc2Vec


import multiprocessing
cores = multiprocessing.cpu_count()

# We contruct the training and testing dataframe for the word2vec
## Training
words_train = pd.DataFrame(np.array(x_train), columns=['words'])
tags_train = pd.DataFrame(np.array(t_train), columns=['tags'])
documents_train = pd.concat([words_train, tags_train], axis=1)
# Testing
words_test = pd.DataFrame(np.array(x_test), columns=['words'])
tags_test = pd.DataFrame(np.array(t_test), columns=['tags'])
documents_test = pd.concat([words_test, tags_test], axis=1)


# Build the document vectors
def tag_docs(docs):
    tagged = docs.apply(lambda r: gensim.models.doc2vec.TaggedDocument(words=r[0], tags=[r[1]]), axis=1)
    return tagged

# Train the doc2vec model
def train_doc2vec_model(tagged_docs, window, vector_size):
    sents = tagged_docs.values
    doc2vec_model = Doc2Vec(sents, vector_size=vector_size, window=window, epochs=20, dm=0)
    return doc2vec_model

# Construct the final vector feature for the classifier
def vec_for_learning(doc2vec_model, tagged_docs):
    sents = tagged_docs.values
    # Unzipping the values
    targets, regressors = zip(*[(doc.tags[0], doc2vec_model.infer_vector(doc.words, steps=20)) for doc in sents])
    return targets, regressors

train_tagged = tag_docs(documents_train)
test_tagged = tag_docs(documents_test)


model = train_doc2vec_model(train_tagged, 15, 5)


y_train, X_train = vec_for_learning(model, train_tagged)
y_test, X_test = vec_for_learning(model, test_tagged)





##
##
# Logistic regression
##
##
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
print('Testing accuracy with LOGISTIC REGRESSION %s' % accuracy_score(y_test, y_pred))
print('Testing F1 score with LOGISTIC REGRESSION: {}'.format(f1_score(y_test, y_pred, average='weighted')))

##
##
# Random forest
##
##
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=300, max_depth=150,n_jobs=1)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print('Testing accuracy with RANDOM FOREST %s' % accuracy_score(y_test, y_pred))
print('Testing F1 score with RANDOM FOREST: {}'.format(f1_score(y_test, y_pred, average='weighted')))


