from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Load model
model = joblib.load("fraud_model.pkl")

class Transaction(BaseModel):
    V1: float = 0
    V2: float = 0
    Amount: float = 0
    Time: float = 0

@app.get("/")
def root():
    return {"message": "Fraud Detection API", "status": "running"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
def predict(transaction: Transaction):
    data = np.array([[
        transaction.V1, transaction.V2, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        transaction.Amount, transaction.Time
    ]])
    pred = model.predict(data)
    prob = model.predict_proba(data)[0]
    return {
        "is_fraud": bool(pred[0]),
        "fraud_probability": round(float(prob[1]), 4)
    }
