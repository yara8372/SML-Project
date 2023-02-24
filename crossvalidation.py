from statistics import mean, mode
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier
import pandas as pd
from progress.bar import Bar




def get_score(y,pred):
    y = y.tolist()
    pred = pred.tolist()
    corr = 0
    total = len(y)
    for i in range(total):
        if y[i] == pred[i]:
            corr += 1
    return corr/total

'''
    We do not have contiously labelled examples, as the movies are not ordered by production year,  hence we do not need to use strategies suggested [here](https://scikit-learn.org/stable/modules/cross_validation.html#a-note-on-shuffling), i.e. KFold + shuffling data
'''
def main():
        df = pd.read_csv("data/train.csv")
        df.Lead.replace(to_replace=['Female','Male'],value=[1,0],inplace=True)
        label = "Lead"  
        y, X = df[label], df.drop([label], axis=1).to_numpy()
        model1 = LogisticRegression(max_iter = 10000)
        model2 = LinearDiscriminantAnalysis()
        model3 = AdaBoostClassifier()
        model4 = DecisionTreeClassifier()
        model5 = QuadraticDiscriminantAnalysis()
        models = [model1,model2,model3,model4,model5]
        # Checking if we should use a stratified folding techique, if we have more samples of a specific class
        if len(y) - sum(y) > sum(y) or sum(y) > len(y) - sum(y):
            # This is true for our case
            print(f"Negative class samples were {len(y)-sum(y)} and positive were {sum(y)}")
            model_bar = Bar("Trying model",max=len(models))
            for model in models:
                cvs = []
                bar = Bar('Trying certain k for fold', max=10)
                for k in range(2,12):
                    # print(f"Trying {k} splits for {type(model).__name__}")
                    skl = StratifiedKFold(n_splits=k)
                    shl = StratifiedShuffleSplit(n_splits=k)
                    score_shl = 0
                    score_skl = 0
                    for train, test in shl.split(X,y):
                        # print(train, test)
                        X_train, X_test = X[train], X[test]
                        y_train, y_test = y[train], y[test]
                        model.fit(X_train, y_train)
                        pred = model.predict(X_test)
                        score_shl += get_score(y_test, pred)
                    for train, test in skl.split(X,y):
                        X_train, X_test = X[train], X[test]
                        y_train, y_test = y[train], y[test]
                        model.fit(X_train,y_train)
                        pred = model.predict(X_test)
                        score_skl += get_score(y_test, pred)
                    cvs.append((score_shl / k, score_skl / k))
                    bar.next()
                print(f"\n{type(model).__name__}")
                print("ShuffleSplit:")
                print(f"Max Acc. {max(cvs[:][0])}")
                print(f"Min Acc. {min(cvs[:][0])}")  
                print(f"Mean Acc. {mean(cvs[:][0])}")  
                print("KFold:")
                print(f"Max Acc. {max(cvs[:][1])}")
                print(f"Min Acc. {min(cvs[:][1])}")  
                print(f"Mean Acc. {mean(cvs[:][1])}")  
                
                        




        else:
            # false in our case, if treu you'd use any type of cv technique
            return 0



if __name__ == "__main__":
    main()

