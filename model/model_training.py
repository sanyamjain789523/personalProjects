from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import pickle
import pandas as pd
import numpy as np
import pathlib
from sklearn.model_selection import train_test_split

path = pathlib.Path.cwd()
data_path = path / "data"
model_path = path / "model"


def train_model():
    df=pd.read_csv(f'{data_path}/BankNote_Authentication.csv')

    X=df.iloc[:,:-1]
    y=df.iloc[:,-1]


    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)
    print("data processed")
    classifier=RandomForestClassifier()
    classifier.fit(X_train,y_train)
    print("model trained")
    y_pred=classifier.predict(X_test)


    score=accuracy_score(y_test,y_pred)

    pickle_out = open(f"{model_path}/classifier.pkl","wb")
    pickle.dump(classifier, pickle_out)
    pickle_out.close()
    print("model saved")

if __name__ == "__main__":
    train_model()