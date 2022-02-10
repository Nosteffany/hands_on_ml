# from sklearn.datasets import fetch_openml
import pickle
import sklearn as sk
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score


with open('/home/dmytro/Desktop/hands_on_ml/classification_p121/mnist.pickle', 'rb') as f:
    mnist = pickle.load(f)

# Prepare data
X = mnist['data']
y = mnist['target']

X_train = X[:60000]
X_test = X[60000:]

y_train = y[:60000]
y_test = y[60000:]

y_train = y_train.astype('int')
y_test = y_test.astype('int')

# Training a Binary Classifier
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train)
print(sgd_clf.score(X_test, y_test))
################################# Performance Measures (p 126)
# Measuring Accuracy Using Cross-Validation






