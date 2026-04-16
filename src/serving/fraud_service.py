import bentoml
import numpy as np
import logging
import os
import sys
import time
from datetime import datetime
from typing import Dict, Any
import redis
from functools import lru_cache

sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from config import REDIS_HOST, REDIS_PORT, PRODUCER_COLUMNS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# ⚡ LOAD MODEL DIRECTLY (NO RUNNER = LOW LATENCY)
# ─────────────────────────────────────────────────────────────
model = bentoml.xgboost.get("fraud_detector:latest").load_model()
booster = model.get_booster()

# ─────────────────────────────────────────────────────────────
# ⚡ SERVICE INIT (NO RUNNERS)
# ─────────────────────────────────────────────────────────────
svc = bentoml.Service(
    name="fraud_detection_service"
)

# ─────────────────────────────────────────────────────────────
# ⚡ FEATURE CONFIG (PRECOMPUTED)
# ─────────────────────────────────────────────────────────────
FEATURE_COLUMNS = [
    col for col in PRODUCER_COLUMNS
    if col not in ['isFraud', 'TransactionID', 'TransactionDT']
]

FEATURE_INDEX = {col: i for i, col in enumerate(FEATURE_COLUMNS)}

# ─────────────────────────────────────────────────────────────
# ⚡ REDIS CONNECTION (REUSED)
# ─────────────────────────────────────────────────────────────
_redis = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    decode_responses=True,
    socket_timeout=0.5,
    socket_connect_timeout=0.5
)

# ─────────────────────────────────────────────────────────────
# ⚡ REDIS CACHE (VERY IMPORTANT)
# ─────────────────────────────────────────────────────────────
@lru_cache(maxsize=10000)
def get_cached_features(card1: int):
    try:
        data = _redis.hgetall(f"features:card:{card1}")
        if not data:
            return {}

        return {
            k: float(v) if v not in ("", "None", "null") else 0.0
            for k, v in data.items()
            if k not in ("card1", "window_start", "window_end", "last_updated")
        }
    except Exception:
        return {}

# ─────────────────────────────────────────────────────────────
# ⚡ FAST FEATURE VECTOR BUILDER
# ─────────────────────────────────────────────────────────────
def build_feature_vector(transaction: Dict[str, Any]) -> np.ndarray:
    card1 = transaction.get("card1", -1)

    # Redis features (cached)
    redis_features = get_cached_features(card1)

    # Merge (Redis overrides)
    merged = transaction.copy()
    merged.update(redis_features)

    # Pre-allocated array (FASTER than list append)
    vector = np.zeros(len(FEATURE_COLUMNS), dtype=np.float32)

    for col, idx in FEATURE_INDEX.items():
        val = merged.get(col)

        if val is None:
            continue

        try:
            vector[idx] = float(val)
        except:
            vector[idx] = 0.0

    return vector.reshape(1, -1)

# ─────────────────────────────────────────────────────────────
# ⚡ MAIN PREDICTION API (SYNC = FASTEST)
# ─────────────────────────────────────────────────────────────
@svc.api(
    input=bentoml.io.JSON(),
    output=bentoml.io.JSON()
)
def predict(transaction: dict) -> dict:
    start = time.time()

    try:
        features = build_feature_vector(transaction)

        # 🔥 DIRECT MODEL CALL (NO IPC)
        pred = booster.inplace_predict(features)
        fraud_prob = float(pred[0])

        is_fraud = fraud_prob > 0.5

        latency = (time.time() - start) * 1000

        return {
            "transaction_id": transaction.get("TransactionID", "unknown"),
            "is_fraud": is_fraud,
            "fraud_probability": round(fraud_prob, 4),
            "latency_ms": round(latency, 2),
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return {
            "error": str(e),
            "is_fraud": False,
            "fraud_probability": 0.0
        }

# ─────────────────────────────────────────────────────────────
# ⚡ HEALTH CHECK
# ─────────────────────────────────────────────────────────────
@svc.api(
    input=bentoml.io.JSON(),
    output=bentoml.io.JSON()
)
def health(_: dict) -> dict:
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat()
    }