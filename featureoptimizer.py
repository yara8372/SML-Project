from array import array
from select import select
import pandas as pd
import numpy as np
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import chi2, SelectKBest, RFECV


def feature_selection_using_kbest_and_chi(models: array, X: pd.DataFrame, y: pd.DataFrame):
    for model in models:
        best_model = (0.00005,0,0)
        for i in range(len(X.columns)):
            selector = SelectKBest(chi2, k=i+1)
            X_new = selector.fit_transform(X, y)
            score = score_model(y,X, X_new, model,len(X_new))
            if score > best_model[0]:
                    best_model = (score, model ,i+1)
        if best_model[1] != 0: # sanity check if the model wasn't updated
            print("--------------------------------------------")
            print(f"Model was: {type(model).__name__} with {X.columns.values[0:best_model[2]]} as features choosen with Chi-Square Test")
            print(f"Performed {score:.6f} percentage points better ")
            print("--------------------------------------------\n")
        else: 
            X_train, X_test, y_train, y_test = train_test_split(X,y)
            model.fit(X_train,y_train)
            pred = model.predict(X_test)
            base_score = get_score(y_test, pred)
            print("--------------------------------------------")
            print(f"{type(model).__name__} did not perform better than when using all features, i.e. accuracy wasn't greater than {base_score:.2f}")
            print("--------------------------------------------\n")

def feature_selection_using_ref(models: array, X: pd.DataFrame, y: pd.DataFrame):
        for model in models:
            best_model = (0.00005, 0, 0)
            for i in range(len(X.columns)):
                new_X,selector = get_best_features(model, X,y, i+1)
                score = score_model(y,X,new_X, model, i+1)
                if score > best_model[0]:
                    best_model = (score, model, i+1)
            if best_model[1] != 0: # sanity check if the model wasn't updated
                print("--------------------------------------------")
                print(f"Model was: {type(best_model[1]).__name__} with {X.columns.values[0:best_model[2]]} as features")
                print(f"Performed {best_model[0]:.6f} percentage points better ")
                print("--------------------------------------------\n")
            else:
                X_train, X_test, y_train, y_test = train_test_split(X,y)
                model.fit(X_train,y_train)
                pred = model.predict(X_test)
                base_score = get_score(y_test, pred)
                print("--------------------------------------------")
                print(f"{type(model).__name__} did not perform better than when using all features, i.e. accuracy wasn't greater than {base_score}")
                print("--------------------------------------------\n")



def get_best_features(model: sk.base.BaseEstimator, X: pd.DataFrame, y:pd.DataFrame, n:int):
    selector = RFECV(model, min_features_to_select=n, n_jobs=6,scoring="accuracy").fit(X,y)
    return selector.transform(X),selector

def get_score(y,pred):
    y = y.tolist()
    pred = pred.tolist()
    corr = 0
    total = len(y)
    for i in range(total):
        if y[i] == pred[i]:
            corr += 1
    return corr/total
        
def score_model(y,X,new_X, model,n):
    X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=42)
    new_X_train, new_X_test, y_train_new_x, y_test_new_x = train_test_split(new_X,y, random_state=42)
    Y_pred = model.fit(X_train,y_train).predict(X_test)
    Y_pred_new_x = model.fit(new_X_train,y_train_new_x).predict(new_X_test)
    before = get_score(y_test,Y_pred)
    after = get_score(y_test_new_x,Y_pred_new_x)
    return after - before


def main():
        df = pd.read_csv("data/train.csv")
        df.Lead.replace(to_replace=['Female','Male'],value=[1,0],inplace=True)
        label = "Lead"  
        y, X = df[label], df.drop([label], axis=1)
        model1 = LogisticRegression(max_iter = 10000)
        model2 = LinearDiscriminantAnalysis()
        model3 = AdaBoostClassifier()
        model4 = DecisionTreeClassifier()
        model5 = QuadraticDiscriminantAnalysis()
        models = [model1,model2,model3,model4]
        feature_selection_using_ref(models, X, y)
        feature_selection_using_kbest_and_chi(models, X, y)
    


if __name__ == "__main__":
    main()
