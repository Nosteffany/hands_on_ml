import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split


path = "/home/dmytro/Desktop/hands_on_ml/classification_p121/mnist.pickle"
with open(path, 'rb') as f:
    data = pickle.load(f)

X = data.data[:50000]
y = data.target[:50000]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=84)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=84)



