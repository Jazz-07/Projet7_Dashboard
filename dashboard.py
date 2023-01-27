# Librairie 

import os, sys, time
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import requests
import json
import shap
from joblib import load
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from streamlit_shap import st_shap

import re

st.title('REALISER UN TABLEAU DE BORD')
# Données sur les caracterisques d'un client
#st.subheader('Echantillon des données ')
df_client=pd.read_csv('df_projet')
#st.write(df_client.head())

from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve,confusion_matrix

# Classification Report
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score,accuracy_score

# fastapi endpoint
#url =' http://127.0.0.1:8000/predict'
#endpoint = '/predict'

#url="https://test-projet7-2022.herokuapp.com/predict"

feat = [f for f in df_client.columns if f not in ['TARGET','SK_ID_CURR','DAYS_EMPLOYED']]
test1=df_client['SK_ID_CURR'].tolist()
number = st.sidebar.selectbox('Client ID',test1)
number=int(number)
if number in test1:
    st.write('The client number is ', number)
    test=df_client['EXT_SOURCE_3'].loc[df_client['SK_ID_CURR']==number].values
    EXT_SOURCE_1=df_client['EXT_SOURCE_1'].loc[df_client['SK_ID_CURR']==number].values
    EXT_SOURCE_2=df_client['EXT_SOURCE_2'].loc[df_client['SK_ID_CURR']==number].values
    EXT_SOURCE_3=df_client['EXT_SOURCE_3'].loc[df_client['SK_ID_CURR']==number].values
    DAYS_EMPLOYED_PERC=df_client['DAYS_EMPLOYED_PERC'].loc[df_client['SK_ID_CURR']==number].values
    AMT_ANNUITY=df_client['AMT_ANNUITY'].loc[df_client['SK_ID_CURR']==number].values
    PREV_CNT_PAYMENT_MEAN=df_client['PREV_CNT_PAYMENT_MEAN'].loc[df_client['SK_ID_CURR']==number].values
    ACTIVE_DAYS_CREDIT_MAX=df_client['ACTIVE_DAYS_CREDIT_MAX'].loc[df_client['SK_ID_CURR']==number].values
    

    vector = np.vectorize(np.float64)
    
    EXT_SOURCE_1=float(vector( EXT_SOURCE_1))
    EXT_SOURCE_2=float(vector( EXT_SOURCE_2))
    EXT_SOURCE_3=float(vector( EXT_SOURCE_3))
    DAYS_EMPLOYED_PERC=float(vector( DAYS_EMPLOYED_PERC))
    AMT_ANNUITY=float(vector( AMT_ANNUITY))
    PREV_CNT_PAYMENT_MEAN=float(vector( PREV_CNT_PAYMENT_MEAN))
    ACTIVE_DAYS_CREDIT_MAX=float(vector( ACTIVE_DAYS_CREDIT_MAX))
    

    valeurs={"EXT_SOURCE_1": EXT_SOURCE_1,"EXT_SOURCE_2": EXT_SOURCE_2,"EXT_SOURCE_3": EXT_SOURCE_3,"DAYS_EMPLOYED_PERC":DAYS_EMPLOYED_PERC,"AMT_ANNUITY":AMT_ANNUITY,"PREV_CNT_PAYMENT_MEAN": PREV_CNT_PAYMENT_MEAN,"ACTIVE_DAYS_CREDIT_MAX":ACTIVE_DAYS_CREDIT_MAX}
    val=json.dumps(valeurs)
    essai={0:{"EXT_SOURCE_1": EXT_SOURCE_1,"EXT_SOURCE_2": EXT_SOURCE_2,"EXT_SOURCE_3": EXT_SOURCE_3,"DAYS_EMPLOYED_PERC":DAYS_EMPLOYED_PERC,"AMT_ANNUITY":AMT_ANNUITY,"PREV_CNT_PAYMENT_MEAN": PREV_CNT_PAYMENT_MEAN,"ACTIVE_DAYS_CREDIT_MAX":ACTIVE_DAYS_CREDIT_MAX}}
    val_df=pd.DataFrame(essai)
    val_df=val_df.transpose()
     
    features=["EXT_SOURCE_2","EXT_SOURCE_3","EXT_SOURCE_1","DAYS_EMPLOYED_PERC","AMT_ANNUITY","PREV_CNT_PAYMENT_MEAN","ACTIVE_DAYS_CREDIT_MAX"] 
    
    st.subheader('INFORMATIONS DU CLIENT')
    st.write(val_df)
    st.subheader('INFORMATIONS DESCRIPTIVES DES CLIENTS')
    st.write(df_client[features].describe())
    if st.button("DECISION"):
        response = requests.post(url =' https://test-projet7-2022.herokuapp.com/predict',data=val)
        #prediction =response.text
        #st.subheader(f"The prediction from model and probability : {response.text}")
        REP=response.json()
        KEY_1=REP['prediction']
        KEY_2=REP['probability']
        categories=['OCTROI','REFUS']
        
        if KEY_1==0:
            st.subheader('DEMANDE DE CREDIT ACCEPTE')
            proba=[KEY_2,1-KEY_2]
            fig, ax = plt.subplots()
            #plt.figure(figsize=(4,4))
            plt.style.use('dark_background')
            colors=['indigo','navy']
            explode=[0.09,0]
            ax.pie(proba,labels=categories,colors=colors,explode=explode,autopct='%1.2f%%')
            st.pyplot(fig)
        if KEY_1==1:
            st.subheader('DEMANDE DE CREDIT REFUSE')
            proba=[1-KEY_2,KEY_2]
            fig1, ax1 = plt.subplots()
            plt.style.use('dark_background')
            colors=['indigo','navy']
            explode=[0,0.09]
            ax1.pie(proba,labels=categories,colors=colors,explode=explode,autopct='%1.2f%%')
            st.pyplot(fig1)
            

    
    if st.button('IMPORTANCE DES VARIABLES'):
        x=[EXT_SOURCE_1,EXT_SOURCE_2,EXT_SOURCE_3,DAYS_EMPLOYED_PERC,AMT_ANNUITY,PREV_CNT_PAYMENT_MEAN,ACTIVE_DAYS_CREDIT_MAX]
        url=f'https://test-projet7-2022.herokuapp.com/INTERPRETABILITE?f1={x[0]}&f2={x[1]}&f3={x[2]}&f4={x[3]}&f5={x[4]}&f6={x[5]}&f7={x[6]}'
        resultat = requests.get(url )
        
        #st.subheader(resultat.json())
        #val=resultat.text
        result=resultat.json()
        clé_1=result['shap_values']
        clé_2=result['x_transfo']
        shap_values=np.array(clé_1)
        x_transfo=np.array(clé_2)
        plt.style.use('classic')
        plt.title('INTERPRETABILITE ')
        features=["EXT_SOURCE_2","EXT_SOURCE_3","EXT_SOURCE_1","DAYS_EMPLOYED_PERC","AMT_ANNUITY","PREV_CNT_PAYMENT_MEAN","ACTIVE_DAYS_CREDIT_MAX"]
        st_shap(shap.summary_plot(shap_values,x_transfo,plot_type='bar',feature_names=features,max_display=len(features)))
        #st.write(shap_values)
        #st_shap(shap.plots.waterfall(shap_values[0]))
        
