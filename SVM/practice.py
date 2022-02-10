from unicodedata import name
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.svm import LinearSVR, SVR
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

#                                        GET THE DATA
data = fetch_california_housing()

fnames = data.feature_names

# df = pd.DataFrame(np.c_[data.data,data.target], columns=fnames)
X = pd.DataFrame(data.data, columns=fnames)
y = pd.Series(data.target)

#                                     ANALYZE THE DATA

# print(df.corrwith(target))
# pd.plotting.scatter_matrix(df)





#                                        TRAIN MODEL

X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=65, test_size=0.25)

parameters = {'svr__C':[0,0.01,0.1,1,10],
             'svr__loss':['epsilon_insensitive', 'squared_epsilon_insensitive'],
             'svr__max_iter':[1000,3000,10000]}

estimator = Pipeline([
    ('scaler', StandardScaler()),
    ('svr', LinearSVR(C=0.1,loss='squared_epsilon_insensitive',max_iter=10000))
])

# grid_estimator = GridSearchCV(estimator=estimator, param_grid=parameters)
# grid_estimator.fit(X_train, y_train)

# best = grid_estimator.best_estimator_
# print(best)
estimator.fit(X_train, y_train)

mse = mean_squared_error(y_test, estimator.predict(X_test))

# score = estimator.score(X_test, y_test)
print(np.sqrt(mse))





