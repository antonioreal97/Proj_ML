import os
# import sys
# sys.path.append("..")
import numpy as np
import time
from model.models import XGB
from pydantic import BaseModel
from fastapi import status, HTTPException, FastAPI
from features.preprocessing import preprocess 
from dataclasses import dataclass


class Input(BaseModel):
    """
    Classe que vai ser utilizada pela API para determinar as variáveis de entrada e seus tipos.
    """
    u_q: float
    coolant: float
    stator_winding: float
    u_d: float
    stator_tooth: float
    motor_speed: float
    i_d: float
    i_q: float
    pm: float
    stator_yoke: float
    ambient: float


@dataclass(frozen=True)
class Dir:
    """
    Classe com o caminho dos diretórios que contem artefatos que serão utilizados para fazer a previsão.
    """
    model_dir = 'model/artifacts/model.pkl'
    scaler_dir = 'model/artifacts/scaler.pkl'

class MeasureValue(BaseModel):
    """
    Classe que retorna a probabilidade da predição.
    """
    
    value: float



model = XGB(Dir.model_dir,Dir.scaler_dir)

app = FastAPI()
@app.post('/measure/predict', response_model=MeasureValue, status_code=status.HTTP_200_OK)
async def run_model(input: Input) -> float:
    """
    Recebe os dados

    Transforma em um dicionário

    Preprocessa e realiza a predição

    """
    dataframe = preprocess(input.dict())
    y_hat = model.realiza_previsao(dataframe)

    return MeasureValue(value=y_hat)

@app.get('/')
async def root():
    return 'Drinking API'
