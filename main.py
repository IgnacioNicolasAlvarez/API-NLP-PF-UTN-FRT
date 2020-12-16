from fastapi import FastAPI
import pickle
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np 
import pandas as  pd
from pydantic import BaseModel
from enum import Enum


class Clasificacion_Polaridad(Enum):
    POSITIVO = 0
    NEGATIVO = 1
    NEUTRO = 2
    

class Mensaje(BaseModel):
    texto: str



app = FastAPI()


@app.get("/predecir")
def read_root(mensaje: Mensaje):
    archivo_modelo = 'modelo_transformador_proyecto.sav'
    transformer, modelo = pickle.load(open(archivo_modelo, 'rb'))
    print(mensaje)
    x = pd.DataFrame([mensaje.texto])
    x = transformer.transform(x.iloc[0])

    return {"Polaridad": f"{Clasificacion_Polaridad(modelo.predict(x)[0]).name}"}

