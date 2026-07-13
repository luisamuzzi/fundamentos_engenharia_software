import os
import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import pandas as pd
from typing import Optional
import joblib

from fundamentos_engenharia_software.config import TOP_FEATURES
from fundamentos_engenharia_software.logging import setup_logging

from main import run_training

setup_logging()

logger = logging.getLogger(__name__)

app = FastAPI(
    title="API de predição de fraude",
    description="API de exemplo",
    version="1.0.0",
)

MODEL_PATH = os.path.join("artifacts/models/", "modelo_final.joblib")
FEATURE_ENGINEER_PATH = os.path.join(
    "artifacts/preprocessors/", "feature_engineer.joblib"
)
IMPUTER_SCALER_PATH = os.path.join(
    "artifacts/preprocessors/", "imputer_scaler.joblib"
)

model = None
feature_engineer = None
imputer_scaler = None


@app.on_event("startup")
def load_model():
    global model
    global feature_engineer
    global imputer_scaler

    model = joblib.load(MODEL_PATH)
    feature_engineer = joblib.load(FEATURE_ENGINEER_PATH)
    imputer_scaler = joblib.load(IMPUTER_SCALER_PATH)


class TransactionData(BaseModel):
    score_1: int
    score_2: float
    score_3: float
    score_4: int
    score_5: float
    score_6: int
    pais: str
    score_7: int
    produto: str
    categoria_produto: str
    score_8: float
    score_9: int
    score_10: int
    entrega_doc_1: int
    entrega_doc_2: Optional[str] = None
    entrega_doc_3: str
    valor_compra: float


@app.post("/predict")
def predict(data: TransactionData):
    """
    Endpoint para executar predição no modelo
    """
    input_df = pd.DataFrame([data.model_dump()])

    input_with_features = feature_engineer.transform(input_df)
    input_scaled = imputer_scaler.transform(input_with_features)

    probability = model.predict_proba(input_scaled[TOP_FEATURES])[0][1]
    prediction = model.predict(input_scaled[TOP_FEATURES])

    logger.info("Predição realizada")
    logger.info(prediction)

    return {
        "prediction": int(prediction),
        "probability_of_fraud": float(probability),
    }


@app.post("/train_model")
def train_model():
    """
    Iniciar o nosso processo de treinamento
    """
    try:
        run_training()

        return {"status": "success", "message": "Modelo treinado"}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Não foi possível treinar modelo {e}"
        )


@app.get("/")
def health_check():
    """Apenas verifica se API está rodando"""
    return {"status": "ok", "message": "Tudo funcionando!"}
