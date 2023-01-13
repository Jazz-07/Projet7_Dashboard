

from joblib import load
from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel
from pydantic.utils import GetterDict
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline

data = pd.read_csv('df_projet')
loaded_model = load('modele_projet7')

# creation d'une nouvelle instance fastAPI
app = FastAPI()

# Définir un objet pour realiser les requetes
# dot notation (.)
class request_body(BaseModel):
    EXT_SOURCE_1 : float
    EXT_SOURCE_2 : float
    EXT_SOURCE_3 : float
    AMT_GOODS_PRICE : float
    INSTAL_AMT_PAYMENT_MIN : float
    PAYMENT_RATE : float
    DAYS_EMPLOYED : float
    
    

# Définition du chemin de point de terminaison 
@app.post("/predict")

# Définition de la fonction de prediction 
def predict (req : request_body) :
    data= req.dict() 
    # Nouvelles données sur lesquelles on fait la prediction
    new_data = [[
        data['EXT_SOURCE_1'],
        data['EXT_SOURCE_2'],
        data['EXT_SOURCE_3'],
        data['AMT_GOODS_PRICE'],
        data['INSTAL_AMT_PAYMENT_MIN'],
        data['PAYMENT_RATE'],
        data['DAYS_EMPLOYED']
        ]]
    
    # prédiction 
    prediction = loaded_model.predict(new_data)[0]
    probability = loaded_model.predict_proba(new_data).max()

    # Retourner la decision
    return {
        'prediction': prediction,
        'probability': probability
    }

#print(data)


