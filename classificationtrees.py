import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from sklearn import tree
from sklearn.model_selection import GridSearchCV , train_test_split
import sklearn.model_selection as skl_ms

np.random.seed(1)
data = pd.read_csv('data/train.csv')

mean = 0
data.info()

n_fold = 10
cv = skl_ms.KFold(n_splits = n_fold, random_state = 2, shuffle = True)

max_depth = range(1,1000)
param = dict(max_depth=max_depth)
model = tree.DecisionTreeClassifier(criterion='entropy')
gcv = GridSearchCV(estimator=model,param_grid=param,cv=cv)
X,y = data.drop(columns=['Lead']), data['Lead']
gcv.fit(X,y)
print(gcv.best_estimator_) #returns the best estimator e.g. DecisionTreeClassifier(max_depth=17, criterion='entropy')
X_train, X_test, y_train, y_test = train_test_split(X,y)
tuned_model = gcv.best_estimator_
tuned_model.fit(X_train, y_train)
prediction  = tuned_model.predict(X_test)

print(pd.crosstab(prediction, y_test))
