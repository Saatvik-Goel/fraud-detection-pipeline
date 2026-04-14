import bentoml
import numpy as np
import pandas as pd
import json
import logging
import os
import sys
from typing import Dict, Any
from datetime import datetime
from prometheus_client import Counter, Histogram, Gauge
import time

sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from config import REDIS_HOST, REDIS_PORT

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Prometheus Metrics ─────────────────────────────────────────────────────
REQUEST_COUNT = Counter(
    'fraud_prediction_requests_total',
    'Total number of prediction requests',
    ['result']   # labels: fraud / not_fraud
)

PREDICTION_LATENCY = Histogram(
    'fraud_prediction_latency_seconds',
    'Time taken for a prediction',
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
)

FRAUD_PROBABILITY = Histogram(
    'fraud_probability_distribution',
    'Distribution of fraud probabilities',
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

# ── Feature columns (must match training order) ────────────────────────────
# These are the features the model was trained on
# Order matters — must be identical to training
FEATURE_COLUMNS = [
    'TransactionAmt', 'ProductCD', 'card1', 'card2', 'card3',
    'card4', 'card5', 'card6', 'addr1', 'addr2', 'P_emaildomain',
    'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9',
    'C10', 'C11', 'C12', 'C13', 'C14',
    'D1', 'D2', 'D3', 'D4', 'D10', 'D11', 'D15',
    'M1', 'M2', 'M3', 'M4', 'M6',
    'V1', 'V2', 'V3', 'V4', 'V5',
]

# ── Load model runner ──────────────────────────────────────────────────────
fraud_model_runner = bentoml.xgboost.get("fraud_detector:latest").to_runner()

svc = bentoml.Service(
    name="fraud_detection_service",
    runners=[fraud_model_runner]
)

# ── Helper: get real-time features from Redis ──────────────────────────────
def get_redis_features(card1: int) -> Dict[str, float]:
    """Fetch real-time card features from Redis."""
    try:
        import redis
        r = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            decode_responses=True
        )
        key = f"features:card:{card1}"
        features = r.hgetall(key)

        if not features:
            return {}

        return {
            k: float(v) if v not in ('null', 'None', '') else 0.0
            for k, v in features.items()
            if k not in ('last_updated', 'card1', 'window_start', 'window_end')
        }
    except Exception as e:
        logger.warning(f"Redis unavailable for card {card1}: {e}")
        return {}

# ── Helper: build feature vector ───────────────────────────────────────────
def build_feature_vector(transaction: Dict[str, Any]) -> np.ndarray:
    """
    Build the feature vector for prediction.
    Combines transaction data with real-time Redis features.
    """
    card1 = transaction.get('card1', -1)

    # Get real-time features from Redis
    redis_features = get_redis_features(card1)

    # Merge transaction data with Redis features
    # Redis features override transaction features (more up to date)
    merged = {**transaction, **redis_features}

    # Build ordered feature vector matching training columns
    feature_vector = []
    for col in FEATURE_COLUMNS:
        val = merged.get(col, 0.0)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            val = 0.0
        try:
            feature_vector.append(float(val))
        except (ValueError, TypeError):
            feature_vector.append(0.0)

    return np.array([feature_vector], dtype=np.float32)

# ── Prediction endpoint ────────────────────────────────────────────────────
@svc.api(
    input=bentoml.io.JSON(),
    output=bentoml.io.JSON()
)
async def predict(transaction: dict) -> dict:
    """
    Main prediction endpoint.

    Input:
        transaction: dict with transaction fields
        e.g. {"TransactionAmt": 100.0, "card1": 5912, ...}

    Output:
        {
            "transaction_id": ...,
            "is_fraud": true/false,
            "fraud_probability": 0.87,
            "risk_level": "HIGH",
            "prediction_time_ms": 45.2
        }
    """
    start_time = time.time()

    try:
        # Build feature vector
        features = build_feature_vector(transaction)

        # Run prediction
        fraud_proba = await fraud_model_runner.predict_proba.async_run(features)
        fraud_prob = float(fraud_proba[0][1])  # probability of class 1 (fraud)
        is_fraud = fraud_prob > 0.5

        # Determine risk level
        if fraud_prob >= 0.8:
            risk_level = "CRITICAL"
        elif fraud_prob >= 0.6:
            risk_level = "HIGH"
        elif fraud_prob >= 0.4:
            risk_level = "MEDIUM"
        elif fraud_prob >= 0.2:
            risk_level = "LOW"
        else:
            risk_level = "MINIMAL"

        latency = (time.time() - start_time) * 1000

        # Update Prometheus metrics
        REQUEST_COUNT.labels(result='fraud' if is_fraud else 'not_fraud').inc()
        PREDICTION_LATENCY.observe(time.time() - start_time)
        FRAUD_PROBABILITY.observe(fraud_prob)

        result = {
            "transaction_id":   transaction.get('TransactionID', 'unknown'),
            "card1":            transaction.get('card1', -1),
            "transaction_amt":  transaction.get('TransactionAmt', 0.0),
            "is_fraud":         is_fraud,
            "fraud_probability": round(fraud_prob, 4),
            "risk_level":       risk_level,
            "prediction_time_ms": round(latency, 2),
            "timestamp":        datetime.utcnow().isoformat(),
        }

        logger.info(
            f"Prediction: card={result['card1']} "
            f"amt={result['transaction_amt']} "
            f"fraud_prob={result['fraud_probability']} "
            f"risk={result['risk_level']} "
            f"latency={result['prediction_time_ms']}ms"
        )

        return result

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return {"error": str(e), "is_fraud": False, "fraud_probability": 0.0}


# ── Health check endpoint ──────────────────────────────────────────────────
@svc.api(
    input=bentoml.io.JSON(),
    output=bentoml.io.JSON()
)
async def health(request: dict) -> dict:
    """Health check endpoint for monitoring."""
    return {
        "status": "healthy",
        "model": "fraud_detector:latest",
        "timestamp": datetime.utcnow().isoformat()
    }