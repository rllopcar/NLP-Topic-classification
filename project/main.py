import os
import glob
import errno
import json
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
import nltk
import gensim


filepath = '../data/bbc/'

labels = []
# Accessing the labels names and saving then in "labels_names"
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
 
# Labeled texts stored in a python list
#print(len(texts_labels))



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
nltk.download('wordnet')

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

np.save('../data/features/texts_clean',text_clean_np)



