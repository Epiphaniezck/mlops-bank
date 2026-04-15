"""
API de prédiction — Bank Marketing
Prédit la probabilité qu'un client souscrive à un dépôt à terme
"""
import os
import pickle
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

app = FastAPI(
    title="Bank Marketing Prediction API",
    description="Prédit si un client va souscrire à un dépôt à terme bancaire.",
    version="1.0.0",
)

MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/model.pkl")
artifact = None


@app.on_event("startup")
def load_model():
    global artifact
    try:
        with open(MODEL_PATH, "rb") as f:
            artifact = pickle.load(f)
        print(f"Modèle chargé depuis {MODEL_PATH}")
    except FileNotFoundError:
        print(f"AVERTISSEMENT : modèle introuvable à {MODEL_PATH}")


# ── Schémas ───────────────────────────────────────────────

class PredictRequest(BaseModel):
    age:       int   = Field(..., example=42,    description="Âge du client")
    job:       str   = Field(..., example="management", description="Type d'emploi")
    marital:   str   = Field(..., example="married",    description="Statut marital")
    education: str   = Field(..., example="tertiary",   description="Niveau d'éducation")
    default:   str   = Field(..., example="no",         description="Défaut de crédit (yes/no)")
    balance:   int   = Field(..., example=2143,  description="Solde annuel moyen (€)")
    housing:   str   = Field(..., example="yes",        description="Prêt immobilier (yes/no)")
    loan:      str   = Field(..., example="no",         description="Prêt personnel (yes/no)")
    contact:   str   = Field(..., example="cellular",   description="Type de contact")
    day:       int   = Field(..., example=5,     description="Jour du dernier contact")
    month:     str   = Field(..., example="may",        description="Mois du dernier contact")
    duration:  int   = Field(..., example=261,   description="Durée du dernier appel (s)")
    campaign:  int   = Field(..., example=1,     description="Nb de contacts campagne actuelle")
    pdays:     int   = Field(..., example=-1,    description="Jours depuis dernier contact (-1 = jamais)")
    previous:  int   = Field(..., example=0,     description="Nb contacts avant cette campagne")
    poutcome:  str   = Field(..., example="unknown",    description="Résultat campagne précédente")


class PredictResponse(BaseModel):
    will_subscribe:    bool
    label:             str
    probability_yes:   float
    probability_no:    float


# ── Endpoints ─────────────────────────────────────────────

@app.get("/health", summary="Vérification de santé")
def health():
    return {
        "status": "ok",
        "model_loaded": artifact is not None,
        "model_type": type(artifact["model"]).__name__ if artifact else None,
    }


@app.post("/predict", response_model=PredictResponse, summary="Prédiction de souscription")
def predict(req: PredictRequest):
    if artifact is None:
        raise HTTPException(status_code=503, detail="Modèle non disponible")

    model    = artifact["model"]
    scaler   = artifact["scaler"]
    encoders = artifact["encoders"]
    cat_cols = artifact["cat_cols"]
    num_cols = artifact["num_cols"]
    feat_names = artifact["feature_names"]

    # Construire le DataFrame d'entrée dans le bon ordre
    row = {
        "age":      req.age,
        "job":      req.job,
        "marital":  req.marital,
        "education":req.education,
        "default":  req.default,
        "balance":  req.balance,
        "housing":  req.housing,
        "loan":     req.loan,
        "contact":  req.contact,
        "day":      req.day,
        "month":    req.month,
        "duration": req.duration,
        "campaign": req.campaign,
        "pdays":    req.pdays,
        "previous": req.previous,
        "poutcome": req.poutcome,
    }
    X = pd.DataFrame([row])

    # Encodage catégoriel
    for col in cat_cols:
        le = encoders[col]
        val = str(X[col].iloc[0])
        if val not in le.classes_:
            val = le.classes_[0]   # valeur inconnue → première classe connue
        X[col] = le.transform([val])

    # Normalisation
    X[num_cols] = scaler.transform(X[num_cols])
    X = X[feat_names]

    proba = model.predict_proba(X)[0]
    pred  = int(model.predict(X)[0])

    return PredictResponse(
        will_subscribe=bool(pred),
        label="yes" if pred else "no",
        probability_yes=round(float(proba[1]), 4),
        probability_no=round(float(proba[0]), 4),
    )
