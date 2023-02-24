from utils import gen_csv_from_pred
from sklearn.model_selection import train_test_split
import pandas as pd
"""
Dummy function to generate a naive classifier which only guess male
"""

def gen_naive():
    df = pd.read_csv('data/train.csv')
    X,y = df.drop(['Lead'], axis= 1), df['Lead']
    X_train, X_test, y_train, y_test = train_test_split(X,y)
    guesses = ['Male'] * len(y_test)
    correct = 0
    y_test = y_test.to_list()
    for i in range(len(guesses)):
        if guesses[i] == y_test[i]:
            correct += 1
    print(f"Naive Classifier had {correct/len(guesses):.2f} accuaracy") 
    gen_csv_from_pred(guesses,"naive")


gen_naive()