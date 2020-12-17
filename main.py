from fastapi import FastAPI
import pickle
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd
from pydantic import BaseModel
import spacy

from enum import Enum


class Clasificacion_Polaridad(Enum):
    NEGATIVO = 0
    POSITIVO = 1
    NEUTRO = 2


class Mensaje(BaseModel):
    texto: str


app = FastAPI()


@app.post("/predecir")
def read_root(mensaje: Mensaje):

    nlp = spacy.load("es_core_news_sm")
    tokens_limpios = []

    doc = nlp(mensaje.texto.lower())
    for token in doc:
        if token.pos_ in ["NOUN", "VERB", "ADJ", "ADV"] and token.is_alpha:
            tokens_limpios.append(token.lemma_)
    texto = " ".join(tokens_limpios)

    archivo_modelo = "modelo_transformador_proyecto.sav"
    transformer, modelo = pickle.load(open(archivo_modelo, "rb"))

    x = pd.DataFrame([texto])
    x = transformer.transform(x.iloc[0])

    return {
        "Polaridad": f"{Clasificacion_Polaridad(modelo.predict(x)[0]).name}",
        "Mensaje L": texto,
    }
