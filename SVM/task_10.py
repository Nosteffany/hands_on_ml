# from random import random
# import numpy as np
# from sklearn.datasets import fetch_california_housing
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression, Lasso, Ridge
# from sklearn.preprocessing import StandardScaler

# np.random.seed(55)
# data = fetch_california_housing()

# X = data.data
# y = data.target

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,random_state=55)

# sc = StandardScaler()
# X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test)


# def batch_gradient(X,y=None,epochs=5000, learning_rate=0.01, eps=None):
    
#     X = np.c_[np.ones([len(X), 1]), X]
#     y = np.array([y])
#     n = len(X[0])
#     m = len(X)
#     weights = np.random.random(n).reshape(n,-1)
      
#     for i in range(epochs):
#         grad = 2/m * X.T.dot(X.dot(weights) - y.T)
#         weights = weights - learning_rate * grad
    
#     return weights

# # print(batch_gradient(X_train, y_train),'\n')

# lr = LinearRegression()
# lr.fit(X_train,y_train)

# lasso = Lasso(max_iter=10000,alpha=0.001)
# lasso.fit(X_train,y_train)

# ridge = Ridge(alpha=1)
# ridge.fit(X_train,y_train)

# print(lr.score(X_test, y_test))
# print(lasso.score(X_test, y_test))
# print(ridge.score(X_test, y_test))

import numpy as np
from sklearn.datasets import fetch_california_housing


np.set_printoptions(suppress=True)

data = fetch_california_housing()
X = data.data

X_centered = X - X.mean(axis=0)
a, b, Vt = np.linalg.svd(X_centered, full_matrices=False)
c1 = Vt.T[:, 0]



print(len(a[0]))