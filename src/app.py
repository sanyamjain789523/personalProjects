import uvicorn
from fastapi import FastAPI
from src.data_model.banknote import BankNote
import numpy as np
import pickle
import pathlib
import pandas as pd
# 2. Create the app object
app = FastAPI()

path = pathlib.Path.cwd()
model_path = path / "model"
with open(f"{model_path}\classifier.pkl", 'rb') as pickle_file:
    classifier = pickle.load(pickle_file)

@app.get('/')
def index():
    return {'message': 'Hello, World'}

@app.post('/predict')
def predict_banknote(data:BankNote):
    data = data.dict()
    variance=data['variance']
    skewness=data['skewness']
    curtosis=data['curtosis']
    entropy=data['entropy']
   # print(classifier.predict([[variance,skewness,curtosis,entropy]]))
    prediction = classifier.predict([[variance,skewness,curtosis,entropy]])
    if(prediction[0]>0.5):
        prediction="Fake note"
    else:
        prediction="Its a Bank note"
    return {
        'prediction': prediction
    }

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)