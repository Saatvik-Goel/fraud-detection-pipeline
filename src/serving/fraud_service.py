import bentoml
import numpy as np
import pandas as pd
import json
import logging
import os
import sys
from typing import Dict, Any
from datetime import datetime
# from prometheus_client import Counter, Histogram, Gauge
import time

sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from config import REDIS_HOST, REDIS_PORT

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Prometheus Metrics ─────────────────────────────────────────────────────
# REQUEST_COUNT = Counter(
#     'fraud_prediction_requests_total',
#     'Total number of prediction requests',
#     ['result']   # labels: fraud / not_fraud
# )

# PREDICTION_LATENCY = Histogram(
#     'fraud_prediction_latency_seconds',
#     'Time taken for a prediction',
#     buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
# )

# FRAUD_PROBABILITY = Histogram(
#     'fraud_probability_distribution',
#     'Distribution of fraud probabilities',
#     buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
# )

# ── Feature columns (must match training order) ────────────────────────────
# These are the features the model was trained on
# Order matters — must be identical to training
from config import PRODUCER_COLUMNS

FEATURE_COLUMNS = [
    col for col in PRODUCER_COLUMNS
    if col not in ['isFraud', 'TransactionID', 'TransactionDT']
]
# ── Load model runner ──────────────────────────────────────────────────────
fraud_model_runner = bentoml.xgboost.get("fraud_detector:latest").to_runner()

svc = bentoml.Service(
    name="fraud_detection_service",
    runners=[fraud_model_runner]
)

# ── Helper: get real-time features from Redis ──────────────────────────────

import redis

# ── Create connection pool ONCE at module level ────────────────────────────
# This is outside any function — created when service starts
_redis_pool = redis.ConnectionPool(
    host=REDIS_HOST,
    port=REDIS_PORT,
    decode_responses=True,
    max_connections=10,
    socket_connect_timeout=1,    # fail fast if Redis is down
    socket_timeout=1
)
_redis_client = redis.Redis(connection_pool=_redis_pool)

def get_redis_features(card1: int) -> dict:
    """Fetch real-time features from Redis using persistent connection."""
    try:
        key = f"features:card:{card1}"
        features = _redis_client.hgetall(key)
        if not features:
            return {}
        return {
            k: float(v) if v not in ('null', 'None', '') else 0.0
            for k, v in features.items()
            if k not in ('last_updated', 'card1', 'window_start', 'window_end')
        }
    except Exception as e:
        logger.warning(f"Redis lookup failed for card {card1}: {e}")
        return {}   # gracefully degrade — predict without Redis features

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
    start_time = time.time()

    try:
        features = build_feature_vector(transaction)
        fraud_proba = await fraud_model_runner.predict_proba.async_run(features)
        fraud_prob = float(fraud_proba[0][1])
        is_fraud = fraud_prob > 0.5

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

        # ── NO prometheus_client calls here ───────────────────────────────
        # BentoML automatically tracks request count and latency
        # Access at http://localhost:3000/metrics

        result = {
            "transaction_id":    transaction.get('TransactionID', 'unknown'),
            "card1":             transaction.get('card1', -1),
            "transaction_amt":   transaction.get('TransactionAmt', 0.0),
            "is_fraud":          is_fraud,
            "fraud_probability": round(fraud_prob, 4),
            "risk_level":        risk_level,
            "prediction_time_ms": round(latency, 2),
            "timestamp":         datetime.utcnow().isoformat(),
        }

        logger.info(
            f"card={result['card1']} "
            f"amt={result['transaction_amt']} "
            f"prob={result['fraud_probability']} "
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

# Add to bottom of fraud_service.py, after svc definition
# import asyncio
# import numpy as np

# async def warmup():
#     """Pre-warm the model runner on startup."""
#     dummy = np.zeros((1, len(FEATURE_COLUMNS)), dtype=np.float32)
#     await fraud_model_runner.predict_proba.async_run(dummy)
#     logger.info("Model runner warmed up")

# # Register warmup
# svc.on_startup(warmup)