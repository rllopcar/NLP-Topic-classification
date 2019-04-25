import os
import glob
import errno
import json
import numpy as np
import pandas as pd
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
 
texts_labels = np.array(texts_labels)

######
######
# Data discovery
######
######
