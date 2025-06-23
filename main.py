from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import os

app = FastAPI(title="Chatbot Predictivo", version="0.1")

# Cargar el modelo al iniciar la aplicación
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
model = joblib.load(MODEL_PATH)

class PredictRequest(BaseModel):
    user_id: int
    session_id: str
    # De momento usamos 3 features aleatorias para el toy model
    feature_1: float
    feature_2: float
    feature_3: float

class PredictResponse(BaseModel):
    probability: float

@app.get("/health")
def health_check():
    return {"status": "OK"}

@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    # 1. Agrupar las features en array para scikit-learn
    X_new = np.array([[payload.feature_1, payload.feature_2, payload.feature_3]])
    # 2. Pedir la probabilidad de clase positiva (índice 1)
    prob = model.predict_proba(X_new)[0, 1]
    return PredictResponse(probability=prob)
