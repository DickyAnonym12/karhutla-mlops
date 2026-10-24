import os
from typing import List

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="API Prediksi Risiko Karhutla")

# Lokasi model hasil training (disimpan oleh train.py)
MODEL_PATH = os.getenv("MODEL_PATH", "model/model.joblib")

# Urutan fitur HARUS sama dengan saat training.
# Sesuaikan dengan fitur yang kamu buat di features.py
FEATURES = [
    "rain_7d", "rain_14d", "rain_30d",
    "temp", "humidity", "wind",
    "hotspot_7d", "month",
]

# Ambang level (sementara). Ganti setelah analisis biaya false alarm vs miss.
THRESH_SEDANG = 0.45
THRESH_TINGGI = 0.60

model = None


@app.on_event("startup")
def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)


class Kecamatan(BaseModel):
    nama: str
    rain_7d: float
    rain_14d: float
    rain_30d: float
    temp: float
    humidity: float
    wind: float
    hotspot_7d: float
    month: int


class PredictRequest(BaseModel):
    data: List[Kecamatan]


def to_level(score: float) -> str:
    if score >= THRESH_TINGGI:
        return "TINGGI"
    if score >= THRESH_SEDANG:
        return "SEDANG"
    return "RENDAH"


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict")
def predict(req: PredictRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model belum dimuat")
    if not req.data:
        raise HTTPException(status_code=400, detail="Data kosong")

    df = pd.DataFrame([k.model_dump() for k in req.data])
    scores = model.predict_proba(df[FEATURES])[:, 1]

    hasil = [
        {"kecamatan": nama, "skor": round(float(s), 3), "level": to_level(float(s))}
        for nama, s in zip(df["nama"], scores)
    ]
    hasil.sort(key=lambda x: x["skor"], reverse=True)
    return {"horizon": "7 hari ke depan", "hasil": hasil}
