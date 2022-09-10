import json
import pickle
import pandas as pd
import httpx
from fastapi import Body, FastAPI, Response
from fastapi.logger import logger
from pydantic import BaseModel
#from BankCredit import BankCredit
#from BankCredit1 import BankCredit1
import os
# Command to execute script locally: uvicorn main:app --host 0.0.0.0 --port 8000 --reload
# Command to run Docker image: docker run -d -p 8000:8000 <fastapi-app-name>:latest

app = FastAPI()

#Creating a class for the attributes input to the ML model.
class Bank_Credit(BaseModel):
    CODE_GENDER: str
    CNT_CHILDREN: int
    DEF_30_CNT_SOCIAL_CIRCLE: float
    EXT_SOURCE_1: float
    EXT_SOURCE_2: float
    EXT_SOURCE_3: float
    NAME_EDUCATION_TYPE_High_education : int
    NAME_EDUCATION_TYPE_Low_education : int
    NAME_EDUCATION_TYPE_Medium_education : int
    ORGANIZATION_TYPE_Government_Industry : int
    ORGANIZATION_TYPE_Services: int
    ORGANIZATION_TYPE_Trade_Business: int
    OCCUPATION_TYPE_Accountants_HR_staff_Managers: int
    OCCUPATION_TYPE_Core_Sales_staff: int
    OCCUPATION_TYPE_Private_service_staff: int
    OCCUPATION_TYPE_Tech_Staff: int
    NAME_FAMILY_STATUS_Married: int
    NAME_FAMILY_STATUS_Single: int


class Client(BaseModel):
    num_client: int
    #OCCUPATION_TYPE_Private_service_staff: int

# Chargement des fichiers
def chargement_data(path):
        #dataframe = pd.read_csv(path)
        #liste_id = dataframe['SK_ID_CURR'].tolist()
        #liste_df = dataframe.values.tolist()
        if path == "df2":
           pickle_fi = open("./df2.pkl", "rb")
        elif path == "application_API":
           pickle_fi = open("./application_API.pkl", "rb")

        dataframe = pickle.load(pickle_fi)
        liste_id = dataframe['SK_ID_CURR'].tolist()
        liste_df = dataframe.values.tolist()
        return dataframe, liste_id, liste_df

# charger le modèle
pickle_lgb = open("./model_rf2.pkl", "rb")
model = pickle.load(pickle_lgb)

# Données encodées
path = "df2"
dataframe, liste_id, liste_df  = chargement_data(path)

X = dataframe[['CODE_GENDER',
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

# Données non encodées pour infos client
path = "application_API"
dataframe_1, liste_id_1, liste_df_1 = chargement_data(path)

@app.post('/liste_id')
async def get_liste_id():
    return liste_id

@app.post('/liste_df')
async def get_liste_id():
    return liste_df

#@app.post('/predi_form')
#def get_predi_form(data: Bank_Credit):
#    print (data.CODE_GENDER)
#    #print (data.m_ext_source_1)

@app.post('/client_infos')
async def get_client_infos(data: Client):
    print (data.num_client)
    data_df  = dataframe[dataframe['SK_ID_CURR'] == data.num_client]
    data_df = data_df[X.columns]
    prediction = model.predict(data_df)[0]

    if prediction == 1:
        #pass
        #payload = json.dumps({
        #"Prediction": 1, })
        #print(payload)
        #return payload

        #return {"prediction": 1}
        return {'prediction': str(prediction)}
        headers = {
        "Content-Type": "appplication/json",
        "Link": '<http://context:ngsi-context.jsonld>; rel="http://www.w3c.org/ns/json-ld#context"; type="lapplication/ld+json"'
        }
    else:
        #pass
        #payload = json.dumps({
        #"Prediction": 0, })
        #print(payload)
        #return payload
        return {"prediction": 0}
        headers = {
        "Content-Type": "appplication/json",
        "Link": '<http://context:ngsi-context.jsonld>; rel="http://www.w3c.org/ns/json-ld#context"; type="lapplication/ld+json"'
        }

@app.post('/liste_df_1')
async def get_liste_id_1():
  return liste_df_1

# homepage route
@app.get("/")
def read_root():
  return {'message': 'This is the homepage of the API '}
