import time
import logging
import sys
import os
from prometheus_client import (
    start_http_server, Gauge,
    Counter, Histogram
)

sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Define all metrics ─────────────────────────────────────────────────────
DRIFT_SCORE = Gauge(
    'fraud_feature_drift_score',
    'Fraction of features drifted (from Evidently)'
)
DRIFT_DETECTED = Gauge(
    'fraud_dataset_drift_detected',
    '1 if dataset drift detected, 0 otherwise'
)
N_DRIFTED_FEATURES = Gauge(
    'fraud_n_drifted_features',
    'Number of drifted features'
)
FRAUD_RATE = Gauge(
    'fraud_prediction_rate',
    'Fraction of recent predictions flagged as fraud'
)
REDIS_CARD_COUNT = Gauge(
    'fraud_redis_card_count',
    'Number of cards with features in Redis'
)

def update_redis_metrics():
    """Update Redis-based metrics."""
    try:
        import redis
        r = redis.Redis(host='localhost', port=6379, decode_responses=True)
        card_count = len(r.keys("features:card:*"))
        REDIS_CARD_COUNT.set(card_count)
    except Exception as e:
        logger.warning(f"Redis metrics update failed: {e}")

def run_metrics_server(port: int = 8001):
    """
    Start Prometheus metrics HTTP server.
    Prometheus scrapes this endpoint every 15 seconds.
    """
    start_http_server(port)
    logger.info(f"Prometheus metrics server started on port {port}")
    logger.info(f"Scrape URL: http://localhost:{port}/metrics")

    while True:
        update_redis_metrics()
        time.sleep(15)

if __name__ == "__main__":
    run_metrics_server()