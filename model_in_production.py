from typing import Dict
import numpy as np
import pandas as pd
import random
from sklearn.linear_model import LogisticRegression as LogReg
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.ensemble import AdaBoostClassifier as Boosting, ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA, QuadraticDiscriminantAnalysis as QDA
from sklearn.feature_selection import SelectFromModel
from progress.bar import Bar
from utils import gen_csv_from_pred
'''
TODO: Fulfil all the steps below 
- [x] Choose CV split size and specific strategy
- [x] Feature selection
- [x] Create parameters to optimize for all models
- [x] Optimize all models
- [x] Crown a winner
'''


class SMLClassifictionModel:

    def __init__(self, model, parameter_dictionary, cv) -> None:
        self.model = model
        self.parameter_dictionary = parameter_dictionary
        self.cv = cv
        self.fixed_param = None

    def set_fixed_param(self, parameters) -> None:
        self.fixed_param = parameters


def hypertune(curr_model, X, y) -> SMLClassifictionModel:
    model = curr_model.model
    param = curr_model.parameter_dictionary
    cv = curr_model.cv

    # GridSearch CV for tuning of hyperparameters
    best = GridSearchCV(estimator=model, param_grid=param, cv=cv)
    best.fit(X, y)
    SMLClassifictionModel.set_fixed_param(curr_model, best.best_params_)
    return curr_model


'''
Section for functions which return the parameters for every one of our models
'''


def param_log_reg() -> Dict:
    C = np.logspace(-4, 4, 50)
    penalty = ['l2']
    return dict(C=C, penalty=penalty)


def param_lda() -> Dict:
    solver = ['svd']
    tol = [1e-4, 1e-3, 1e-2, 1e-1]
    return dict(solver=solver, tol=tol)


def param_qda() -> Dict:
    tol = [1e-4, 1e-3, 1e-2, 1e-1]
    return dict(tol=tol)


def param_dtc() -> Dict:
    criterion = ["gini", "entropy", "log_loss"]
    max_depth = range(1, 10)
    min_samples_split = range(2, 10)
    min_samples_leaf = range(1, 10)
    return dict(criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)


def param_boosting() -> Dict:
    n_estimators = range(10, 100, 10)
    learning_rate = np.arange(0.1, 1, 0.1)
    return dict(n_estimators=n_estimators, learning_rate=learning_rate)


def get_score(y, pred) -> float:
    y = y.tolist()
    pred = pred.tolist()
    corr = 0
    total = len(y)
    for i in range(total):
        if y[i] == pred[i]:
            corr += 1
    return corr/total


def main() -> None:
    base_models = [LogReg(max_iter=10000), LDA(), QDA(),  DTC(), Boosting()]
    params = [param_log_reg(), param_lda(), param_qda(),
              param_dtc(), param_boosting()]
    data = pd.read_csv('data/train.csv')
    label = "Lead"
    custom_models = [None] * len(base_models)

    k = 5  # TODO: Figure out best choice here or just default, may differ strategy
    seed = random.randint(0, 10000)
    print(f"Seed for this run was {seed}")
    cv = StratifiedKFold(k, shuffle=True, random_state=seed)

    # Minor preprocessing + feature selection based on impurity-based feature importances (https://scikit-learn.org/stable/modules/feature_selection.html#tree-based-feature-selection)
    X, y = data.drop(columns=['Lead']), data[label]
    clf = ExtraTreesClassifier(n_estimators=50)
    clf = clf.fit(X, y)
    slt = SelectFromModel(clf, prefit=True)
    X = slt.fit_transform(X)

    bar = Bar("Tuning models", max=len(params))
    for i in range(len(base_models)):
        custom_models[i] = SMLClassifictionModel(base_models[i], params[i], cv)
        hypertune(custom_models[i], X, y)
        bar.next()
    print("\n")  # nonsense next line to separate stdout from progress barZn

    # Use the found hyperparameters and crown the winner
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=seed)
    best_log_reg = LogReg(max_iter=10000, **custom_models[0].fixed_param)
    best_lda = LDA(**custom_models[1].fixed_param)
    best_qda = QDA(**custom_models[2].fixed_param)
    best_dtc = DTC(**custom_models[3].fixed_param)
    best_boosting = Boosting(**custom_models[4].fixed_param)
    preds = [best_log_reg, best_lda, best_qda, best_dtc, best_boosting]
    best = (None, {}, 0.0, [])
    for i in range(len(base_models)):
        curr = preds[i]
        pred = curr.fit(X_train, y_train).predict(X_test)
        score = get_score(y_test, pred)
        if score > best[2]:
            best = (curr, curr.get_params(), score, pred)
    

    print(
        f"Best model was {type(best[0]).__name__}({best[1]}) with an accuaracy of {100*best[2]:.2f}%")
    actual_test = pd.read_csv('data/test.csv')
    pred = best[0].predict(slt.transform(actual_test))
    gen_csv_from_pred(pred, "production_model")


if __name__ == "__main__":
    main()
