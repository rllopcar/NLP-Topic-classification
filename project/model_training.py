import numpy as np
from sklearn.model_selection import train_test_split
import gensim

# Data file path
dataPath = '../data/'

# Train ratio
train_ratio = 0.3

# Load features array 
tokenized_attributes = np.load(dataPath+'features/tokenized.npy')
# Load labels array 
labels = np.load(dataPath+'features/labels.npy')


x_train, x_test, t_train, t_test = train_test_split(tokenized_attributes, labels, test_size=1 - train_ratio, stratify=labels)
x_dev, x_test, t_dev, t_test = train_test_split(x_test, t_test, test_size=0.5, stratify=t_test)


##
##
# Word2Vec
##
##

import multiprocessing
cores = multiprocessing.cpu_count()

# Building the Vocabulary
## Splitting training set into the training and testing for word2vec
#x_train_vec, x_test_vec, t_train_, t_test_vec = train_test_split(x_train, t_train, test_size=0., stratify=t_test)

max_epochs = 100
vec_size = 20
alpha = 0.025

model = gensim.models.doc2vec.Doc2Vec(vector_size=50, min_count=2, epochs=40)
  
print(x_train[:2])

#model.build_vocab(x_train)

