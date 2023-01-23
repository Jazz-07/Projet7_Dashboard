# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 14:29:56 2022

@author: jason
"""

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
import shap


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
    DAYS_EMPLOYED_PERC : float
    AMT_ANNUITY : float
    PREV_CNT_PAYMENT_MEAN : float
    ACTIVE_DAYS_CREDIT_MAX : float
    
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
        data['DAYS_EMPLOYED_PERC'],
        data['AMT_ANNUITY'],
        data['PREV_CNT_PAYMENT_MEAN'],
        data['ACTIVE_DAYS_CREDIT_MAX']
        ]]
    
    # prédiction 
    prediction = loaded_model.predict(new_data)[0]
    probability = loaded_model.predict_proba(new_data).max()

    # Retourner la decision
    return {
        'prediction': prediction,
        'probability': probability
    }


# interprétabilité des résultat 
@app.get("/INTERPRETABILITE")

def INTERPRETABILITE (f1:float,f2:float,f3:float,f4: float,f5:float,f6:float,f7:float):
    classifier=loaded_model['model']
    valeurs=[[f1,f2,f3,f4,f5,f6,f7]]
    #user=pd.DataFrame([valeurs])
    x_transfo= loaded_model['scaler'].fit_transform(valeurs)
    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(x_transfo)
    
    return {'shap_values':shap_values.tolist(),'x_transfo':x_transfo.tolist()}
    
    