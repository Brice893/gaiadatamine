from tkinter import W
import streamlit as st
from streamlit_shap import st_shap
import shap
import pandas as pd
from pandas import MultiIndex, Int16Dtype
import numpy as np
import matplotlib
#import seaborn as sns
import requests
import json
import pickle
#import os
from starlette.responses import Response
from sklearn.preprocessing import StandardScaler
import io
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objs as go
import streamlit.components.v1 as components
#import xgboost as xgb
#from sklearn.ensemble import RandomForestClassifier
from PIL import Image
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


# Command to execute script locally: streamlit run app.py
# Command to run Docker image: docker run -d -p 8501:8501 <streamlit-app-name>:latest

st.sidebar.title("Prêt à dépenser")
#st.write ('---debug chargement image ')
########################################################
# Loading images to the website
########################################################
image = Image.open("credit.png")
st.sidebar.image(image)


# charger le modèle
pickle_lgb = open("storage/model_rf2.pkl", "rb")
model = pickle.load(pickle_lgb)

def impPlot(imp, name):
    figure = px.bar(imp,
                    x=imp.values,
                    y=imp.keys(), labels = {'x':'Importance Value', 'index':'Columns'},
                    text=np.round(imp.values, 2),
                    title=name + ' Feature Selection Plot',
                    width=1000, height=600)
    figure.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
    })
    st.plotly_chart(figure)

def main_page():

    st.sidebar.markdown("# Calcul du risque")
    st.title('Calcul du risque de remboursement de prêt')

    st.subheader("Prédictions de scoring client et positionnement dans l'ensemble des clients")

    if 'client' not in st.session_state:
        st.session_state.client = 0
    else:
        id_input = st.session_state.client

    #entities = requests.get("http://frontend.docker:8000/liste_id")
    #entities = requests.get("http://host.docker.internal:8000/liste_id") # Specify this path for Dockerization to work

    #endpoint = 'http://host.docker.internal:8000/liste_id'
    entities = requests.post('http://127.0.0.1:8000/liste_id',timeout=9500)

    #entities = requests.get("http://12:8000/liste_id")

    liste_id = entities.json()

    id_input = st.selectbox('Choisissez le client que vous souhaitez visualiser', liste_id)
    st.session_state.client = id_input

    #
    #-- récup infos pour faire une df encodée
    #endpoint = 'http://host.docker.internal:8000/liste_df'
    entities = requests.post('http://127.0.0.1:8000/liste_df',timeout=8500)
    liste_df = entities.json()

    df = pd.DataFrame(liste_df, columns =[
        'TARGET',
        'SK_ID_CURR',
        'CODE_GENDER',
         'CNT_CHILDREN',
         'DEF_30_CNT_SOCIAL_CIRCLE',
         'EXT_SOURCE_1',
         'EXT_SOURCE_2',
         'EXT_SOURCE_3',
         'NAME_EDUCATION_TYPE_High education',
         'NAME_EDUCATION_TYPE_Low education',
         'NAME_EDUCATION_TYPE_Medium education',
         'ORGANIZATION_TYPE_Government/Industry',
         'ORGANIZATION_TYPE_Services',
         'ORGANIZATION_TYPE_Trade/Business',
         'OCCUPATION_TYPE_Accountants/HR staff/Managers',
         'OCCUPATION_TYPE_Core/Sales staff',
         'OCCUPATION_TYPE_Private service staff',
         'OCCUPATION_TYPE_Tech Staff',
         'NAME_FAMILY_STATUS_Married',
         'NAME_FAMILY_STATUS_Single'])

    client_infos = df[df['SK_ID_CURR'] == id_input]

    TARGET = client_infos['TARGET'].values

    client_infos.drop(['SK_ID_CURR', 'TARGET'], axis=1, inplace= True)

    X = client_infos

    features = client_infos.columns

    client = json.dumps({"num_client": id_input})
    header = {'Content-Type': 'application/json'}
    response = requests.request("POST","http://127.0.0.1:8000/client_infos",headers=header,data=client)

    prediction = response.text


    if "1" in prediction:
        if TARGET == 1:
            st.error('Crédit Refusé (TP)')
        else:
            st.success('Crédit Accepté (FP)')
    else:
        if TARGET == 1:
            st.error('Crédit Refusé (FN)')
        else:
            st.success('Crédit Accepté (TN)')

    # On va construire la dataframe à partir de liste_df_1
    #endpoint = 'http://host.docker.internal:8000/liste_df_1'
    entities = requests.post('http://127.0.0.1:8000/liste_df_1',timeout=8500)

    liste_df_1 = entities.json()

    dataframe = pd.DataFrame(liste_df_1, columns =[
        'TARGET',
        'SK_ID_CURR',
        'CODE_GENDER',
        'AGE',
        'CNT_CHILDREN',
        'DEF_30_CNT_SOCIAL_CIRCLE',
        'NAME_EDUCATION_TYPE',
        'ORGANIZATION_TYPE',
        'OCCUPATION_TYPE',
        'NAME_FAMILY_STATUS',
        'AMT_INCOME_TOTAL',
        'INCOME_CREDIT_PERC',
        'DAYS_EMPLOYED_PERC',
        'EXT_SOURCE_1',
        'EXT_SOURCE_2',
        'EXT_SOURCE_3'])

    dataframe.drop(['AGE','AMT_INCOME_TOTAL','INCOME_CREDIT_PERC','DAYS_EMPLOYED_PERC'], axis=1, inplace= True)
    X_infos_client = dataframe[dataframe['SK_ID_CURR'] == id_input]

    # informations du client
    st.header("Informations du client")
    st.write(X_infos_client)

    # scatter plot
    st.header("OCCUPATION_TYPE / EXT_SOURCE_3 / target")
    fig = px.box(dataframe, x="OCCUPATION_TYPE", y="EXT_SOURCE_3", color="TARGET", notched=True)
    st.plotly_chart(fig)

    st.header("OCCUPATION_TYPE / EXT_SOURCE_2 / target")
    fig = px.box(dataframe, x="OCCUPATION_TYPE", y="EXT_SOURCE_2", color="TARGET", notched=True)
    st.plotly_chart(fig)

    st.header("OCCUPATION_TYPE / EXT_SOURCE_1 / target")
    fig = px.box(dataframe, x="OCCUPATION_TYPE", y="EXT_SOURCE_1", color="TARGET", notched=True)
    st.plotly_chart(fig)


    # Variables globales
    st.header('Variables globales du modèle Random Forest :')
    feat_importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=True)
    impPlot(feat_importances, 'Random Forest Classifier')

    # SHAP

    # Variables locales
    st.header('Variables locales du modèle Random Forest :')
    # compute SHAP values

    # Objet permettant de calculer les shap values
    data_for_prediction_array = X.values.reshape(1, -1)


    model.predict_proba(data_for_prediction_array)
    explainer = shap.TreeExplainer(model)
    # Calculate Shap values
    shap_values = explainer.shap_values(X)


    #st_shap(shap.plots.waterfall(shap_values[0]), height=300)
    #st_shap(shap.plots.beeswarm(shap_values), height=300)

    st_shap(shap.summary_plot(shap_values[0], X))

    #st_shap(shap.force_plot(explainer.expected_value, shap_values, X))
    st_shap(shap.force_plot(explainer.expected_value[1], shap_values[1], X))

    #st_shap(shap.summary_plot(shap_values, X))

    #st_shap(shap.force_plot(explainer.expected_value, shap_values), height=200, width=1000)

    #SHAP valeurs explicatives (tableau NumPy)
    #st_shap(shap.force_plot(explainer.expected_value[0],shap_values[0], features=Features))
    #st_shap(shap.force_plot(explainer.expected_value[0],X.iloc[0,:]), height=200, width=1000)
    #st_shap(shap.force_plot(explainer.expected_value, shap_values[0], X), height=200, width=1000)

    st.write("Transparence des informations du client ",id_input)

    # réalimenter X2 avec les variables saisies
    # Saisie des informations Client dans X2 pour prédiction nouvelle
    X2 = client_infos
    #X2 = dataframe[dataframe['SK_ID_CURR'] == id_input]

    EXT_SOURCE_1 = st.slider("EXT_SOURCE_1", 0.1, 1.0,0.1)
    X2['EXT_SOURCE_1'] = EXT_SOURCE_1

    EXT_SOURCE_2 = st.slider("EXT_SOURCE_2", 0.1, 1.0,0.1)
    X2['EXT_SOURCE_2'] = EXT_SOURCE_2

    EXT_SOURCE_3 = st.slider("EXT_SOURCE_3", 0.1, 1.0,0.1)
    X2['EXT_SOURCE_3'] = EXT_SOURCE_3

    ORGANIZATION_TYPE = st.selectbox("ORGANIZATION_TYPE",options=['Services', 'Government_Industry', 'Trade_Business'])

    X2['ORGANIZATION_TYPE_Government/Industry'] = 0
    X2['ORGANIZATION_TYPE_Services'] = 0
    X2['ORGANIZATION_TYPE_Trade/Business'] = 0

    if ORGANIZATION_TYPE ==  'Government/Industry':
               X2['ORGANIZATION_TYPE_Government/Industry'] = 1
    elif ORGANIZATION_TYPE == 'Services':
               X2['ORGANIZATION_TYPE_Services'] = 1
    elif ORGANIZATION_TYPE == 'Trade/Business':
               X2['ORGANIZATION_TYPE_Trade/Business'] = 1


    OCCUPATION_TYPE = st.selectbox("OCCUPATION_TYPE",options=['Accountants_HR_staff_Managers',
                                                              'Core_Sales_staff',
                                                              'Private_service_staff',
                                                              'Tech_Staff'])

    X2['OCCUPATION_TYPE_Accountants/HR staff/Managers'] = 0
    X2['OCCUPATION_TYPE_Core/Sales staff'] = 0
    X2['OCCUPATION_TYPE_Private service staff'] = 0
    X2['OCCUPATION_TYPE_Tech Staff'] = 0

    if OCCUPATION_TYPE == 'Accountants/HR staff/Managers':
               X2['OCCUPATION_TYPE_Accountants/HR staff/Managers'] = 1
    elif OCCUPATION_TYPE == 'Core/Sales staff':
               X2['OCCUPATION_TYPE_Core/Sales staff'] = 1
    elif OCCUPATION_TYPE == 'Private service staff':
               X2['OCCUPATION_TYPE_Private service staff'] = 1
    elif OCCUPATION_TYPE ==  'Tech Staff':
               X2['OCCUPATION_TYPE_Tech Staff'] = 1

    X3 = X2[['CODE_GENDER',
         'CNT_CHILDREN',
         'DEF_30_CNT_SOCIAL_CIRCLE',
         'EXT_SOURCE_1',
         'EXT_SOURCE_2',
         'EXT_SOURCE_3',
         'NAME_EDUCATION_TYPE_High education',
         'NAME_EDUCATION_TYPE_Low education',
         'NAME_EDUCATION_TYPE_Medium education',
         'ORGANIZATION_TYPE_Government/Industry',
         'ORGANIZATION_TYPE_Services',
         'ORGANIZATION_TYPE_Trade/Business',
         'OCCUPATION_TYPE_Accountants/HR staff/Managers',
         'OCCUPATION_TYPE_Core/Sales staff',
         'OCCUPATION_TYPE_Private service staff' ,
         'OCCUPATION_TYPE_Tech Staff',
         'NAME_FAMILY_STATUS_Married',
         'NAME_FAMILY_STATUS_Single'
         ]]
    #print(X3.values)
    #ligne_client = json.dumps({"list_client": X3})
    #ligne_client =json.dumps(X3)
    #result = X3.to_json(orient="records")
    #client_test = json.dumps({"num_client": id_input})
    #print (client_test)
    #result = json.dumps("num_client":{id_input},"m_ext_source_1":{EXT_SOURCE_1})

	#payload=json.dumps(result)
    #to_predict_dict = X3.to_dict()

    #header = {'Content-Type': 'application/json'}
    #response = requests.request("POST","http://127.0.0.1:8000/predi_form",headers=header,json=to_predict_dict)

    #roba = response.json()

    #st.write(proba)
    #proba = response.json()

    predict_probability = model.predict_proba(X3)
    #st.write('Probabilité d"appartenance aux classes : ', proba)

    if predict_probability [0][0] > predict_probability [0][1]:
       st.success('Votre crédit serait accordé')
       st.subheader('Le client {} a une probabilité de remboursement de {}%'.format
                              (id_input ,round(predict_probability[0][0]*100 , 2)))
    else:
       st.error('Votre crédit serait refusé')
       st.subheader('Le client {} a une probabilité de défaut de paiement de {}%'.format
                                  (id_input ,round(predict_probability[0][1]*100 , 2)))


my_dict = {
    "Calcul du risque": main_page }


keys = list(my_dict.keys())

selected_page = st.sidebar.selectbox("Select a page", keys)
my_dict[selected_page]()
