import numpy as np
from sklearn.model_selection import train_test_split


# Data file path
dataPath = '../data/'

# Train ratio
train_ratio = 0.3

# Load features array
tokenized_attributes = np.load(dataPath+'features/tokenized.npy')
labels = np.load(dataPath+'features/tokenized.npy')

x_train, x_test, t_train, t_test = train_test_split(tokenized_attributes, labels, test_size=1 - train_ratio, stratify=labels)
x_dev, x_test, t_dev, t_test = train_test_split(x_test, t_test, test_size=0.5, stratify=t_test)







