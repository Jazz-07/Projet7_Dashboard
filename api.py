from joblib import load
from typing import Any,Optional
from fastapi import FastAPI
from pydantic import BaseModel
from pydantic.utils import GetterDict
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier

from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline

import uvicorn
import gunicorn
from fastapi import FastAPI
from Model import DataModel, Data
import jsbeautifier
import cssbeautifier

opts = jsbeautifier.default_options()
opts.indent_size = 2
opts.space_in_empty_paren = True
res = jsbeautifier.beautify('some javascript', opts)


# Create app and model objects
my_app = FastAPI()
model = DataModel()

@my_app.post('/predict')
def predict_class(req: Data):
    my_data= req.dict()
    mdata =[
        [my_data['EXT_SOURCE_1'], my_data['EXT_SOURCE_2'], my_data['EXT_SOURCE_3'], my_data['PAYMENT_RATE']]
    ]
    prediction, probability = model.predict_species(my_data['EXT_SOURCE_1'], my_data['EXT_SOURCE_2'], my_data['EXT_SOURCE_3'], my_data['PAYMENT_RATE'])
    return {
        'prediction': prediction,
        'probability': probability
    }

