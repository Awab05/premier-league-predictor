# predictor_backend.py
"""
Simple FastAPI backend that exposes one /predict endpoint.
Supports three metrics:
    • winner  ➜ binary win/loss classifier (rf_bin)
    • shots   ➜ expected shots (rolling_HS)
    • sot     ➜ expected shots‑on‑target (rolling_HST)
    • goals   ➜ expected goals (rolling_goals_for)

Assumptions
-----------
* A joblib dump `rf_bin.joblib` exists in the same folder (model, feature list).
* A pre‑processed `combined_data.parquet` (with shifted rolling features) exists.
  It gets loaded once at startup.

Run:
    uvicorn predictor_backend:app --reload
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from enum import Enum
import pandas as pd
import joblib
import numpy as np
from typing import Literal

# --------------------------------------------------
# Load assets ONCE when the server starts
# --------------------------------------------------
combined_data = pd.read_parquet("combined_data.parquet")

model, FEATURE_COLS = joblib.load("rf_bin.joblib")  # (RandomForestClassifier, List[str])

# most recent row per team helper
def _latest_row(team: str) -> pd.Series:
    rows = combined_data[combined_data["Team"] == team]
    if rows.empty:
        raise ValueError(f"Team '{team}' not found in dataset")
    return rows.sort_values("Date").iloc[-1]

# --------------------------------------------------
# FastAPI setup
# --------------------------------------------------
app = FastAPI(title="Premier‑League Predictor")

class Metric(str, Enum):
    winner = "winner"
    shots = "shots"
    sot = "sot"       # shots on target
    goals = "goals"

class PredictRequest(BaseModel):
    teamA: str
    teamB: str
    metric: Metric

# --------------------------------------------------
# Core prediction logic
# --------------------------------------------------

def make_feature_vector(teamA: str, teamB: str) -> pd.DataFrame:
    """Return a 1×N DataFrame in the exact order FEATURE_COLS."""
    rowA = _latest_row(teamA)
    rowB = _latest_row(teamB)
    diff = (rowA[FEATURE_COLS].values - rowB[FEATURE_COLS].values)
    return pd.DataFrame(diff.reshape(1, -1), columns=FEATURE_COLS)


@app.post("/predict")
def predict(req: PredictRequest):
    if req.teamA == req.teamB:
        raise HTTPException(status_code=400, detail="teamA and teamB must differ")

    try:
        vec = make_feature_vector(req.teamA, req.teamB)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    # winner metric – use the Random‑Forest
    if req.metric == Metric.winner:
        prob = model.predict_proba(vec)[0]
        cls_idx = list(model.classes_).index(1)  # probability of win for teamA
        probA = float(prob[cls_idx])
        return {
            "metric": "winner",
            "teamA": req.teamA,
            "teamB": req.teamB,
            "prob_teamA_win": round(probA, 3),
            "prob_teamB_win": round(1 - probA, 3),
            "predicted": req.teamA if probA >= 0.5 else req.teamB,
        }

    # shots, shots on target, goals – use shifted rolling averages directly
    rowA = _latest_row(req.teamA)
    rowB = _latest_row(req.teamB)

    if req.metric == Metric.shots:
        key = "rolling_HS"
        return {
            "metric": "shots",
            "teamA": req.teamA,
            "teamB": req.teamB,
            "expected_teamA": round(rowA[key], 1),
            "expected_teamB": round(rowB[key], 1),
        }
    elif req.metric == Metric.sot:
        key = "rolling_HST"
        return {
            "metric": "sot",
            "teamA": req.teamA,
            "teamB": req.teamB,
            "expected_teamA": round(rowA[key], 1),
            "expected_teamB": round(rowB[key], 1),
        }
    elif req.metric == Metric.goals:
        key = "rolling_goals_for"
        return {
            "metric": "goals",
            "teamA": req.teamA,
            "teamB": req.teamB,
            "expected_teamA": round(rowA[key], 2),
            "expected_teamB": round(rowB[key], 2),
        }

    # should never reach
    raise HTTPException(status_code=400, detail="Unsupported metric")
