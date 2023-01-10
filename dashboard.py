# Librairie 

import os, sys, time
import numpy as np
import pandas as pd
import streamlit as st
import requests
import json
import shap
from joblib import load
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline

import re

st.title('Realiser un Dashboard')
# Données sur les caracterisques d'un client
st.subheader('Echantillon des données ')
df_client=pd.read_csv('df_data')
st.write(df_client.head())

from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve,confusion_matrix

# Classification Report
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score,accuracy_score

# fastapi endpoint
url =' http://127.0.0.1:8000/predict'
endpoint = '/predict'



feat = [f for f in df_client.columns if f not in ['TARGET','SK_ID_CURR','DAYS_EMPLOYED']]
test1=df_client['SK_ID_CURR'].tolist()
number = st.sidebar.selectbox('Client ID',test1)
number=int(number)
if number in test1:
    st.write('The client number is ', number)
    test=df_client['PAYMENT_RATE'].loc[df_client['SK_ID_CURR']==number].values
    EXT_SOURCE_1=df_client['EXT_SOURCE_1'].loc[df_client['SK_ID_CURR']==number].values
    EXT_SOURCE_2=df_client['EXT_SOURCE_2'].loc[df_client['SK_ID_CURR']==number].values
    EXT_SOURCE_3=df_client['EXT_SOURCE_3'].loc[df_client['SK_ID_CURR']==number].values
    PAYMENT_RATE=df_client['PAYMENT_RATE'].loc[df_client['SK_ID_CURR']==number].values
    

    vector = np.vectorize(np.float64)
    
    EXT_SOURCE_1=float(vector( EXT_SOURCE_1))
    EXT_SOURCE_2=float(vector( EXT_SOURCE_2))
    EXT_SOURCE_3=float(vector( EXT_SOURCE_3))
    PAYMENT_RATE=float(vector( PAYMENT_RATE))
    

    valeurs={"EXT_SOURCE_1": EXT_SOURCE_1,"EXT_SOURCE_2": EXT_SOURCE_2,"EXT_SOURCE_3": EXT_SOURCE_3,"PAYMENT_RATE": PAYMENT_RATE}
    val=json.dumps(valeurs)
    

    st.write(valeurs)
    if st.button("Predict"):
        response = requests.post(url="https://test-projet7-2022.herokuapp.com/predict",data=val)
        #prediction =response.text
        st.subheader(f"The prediction from model and probability : {response.text}")
    
    if st.button('Interprétabilité des résultats'):
        shap.initjs()
        model = load('modele_joblib')
        classifier=model['HGBClassifier']
        user=pd.DataFrame([valeurs])
        st.write(user)
        x_transfo= model['RobustScaler'].fit_transform(user)
        explainer = shap.TreeExplainer(classifier)
        shap_values = explainer.shap_values(x_transfo)
        features=['EXT_SOURCE_3', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'PAYMENT_RATE']
        #st.pyplot(shap.force_plot(explainer.expected_value[0],shap_values[0],matplotlib=True))
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot(shap.summary_plot(shap_values,x_transfo,plot_type='bar',feature_names=features,max_display=len(features)))
        #shap.force_plot(explainer.expected_value, shap_values[0,:], df_client[feat].iloc[0,:])
        #st.pyplot()
        
