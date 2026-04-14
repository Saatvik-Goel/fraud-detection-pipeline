import pandas as pd
import numpy as np
import os
import sys
import json
import logging
from datetime import datetime

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, ClassificationPreset
from evidently.metric_preset import DataDriftPreset

sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from config import DATA_RAW_PATH, DATA_PROCESSED_PATH, PRODUCER_COLUMNS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Key features to monitor for drift ─────────────────────────────────────
# Focus on the most important features from MLflow feature importance
MONITOR_FEATURES = [
    'TransactionAmt',
    'card1', 'card2',
    'C1', 'C2', 'C6', 'C11',
    'D1', 'D4', 'D10',
    'V1', 'V2', 'V3', 'V4',
]

# ── Load reference data (training distribution) ────────────────────────────
def load_reference_data(sample_size: int = 10000) -> pd.DataFrame:
    """
    Load a sample of training data as the reference distribution.
    This is what 'normal' looks like to Evidently.
    """
    logger.info("Loading reference data...")
    df = pd.read_csv(
        os.path.join(DATA_RAW_PATH, 'train_transaction.csv'),
        usecols=MONITOR_FEATURES + ['isFraud']
    )

    # Use first 80% as reference (training split)
    train_size = int(len(df) * 0.8)
    reference = df.iloc[:train_size].sample(
        n=min(sample_size, train_size),
        random_state=42
    )

    # Fill nulls
    reference = reference.fillna(reference.median(numeric_only=True))
    logger.info(f"Reference data: {len(reference)} rows")
    return reference

# ── Load current data (recent predictions / new data) ─────────────────────
def load_current_data(sample_size: int = 5000) -> pd.DataFrame:
    """
    Load recent data to compare against reference.
    In production this would be last 24h of transactions.
    For now we use the test split to simulate drift detection.
    """
    logger.info("Loading current data...")
    df = pd.read_csv(
        os.path.join(DATA_RAW_PATH, 'train_transaction.csv'),
        usecols=MONITOR_FEATURES + ['isFraud']
    )

    # Use last 20% as current data (test split)
    train_size = int(len(df) * 0.8)
    current = df.iloc[train_size:].sample(
        n=min(sample_size, len(df) - train_size),
        random_state=42
    )

    current = current.fillna(current.median(numeric_only=True))
    logger.info(f"Current data: {len(current)} rows")
    return current

# ── Run Drift Report ───────────────────────────────────────────────────────
def run_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    output_dir: str = "reports/"
) -> dict:
    """
    Run Evidently drift report and save as HTML + JSON.
    Returns drift metrics for Prometheus.
    """
    os.makedirs(output_dir, exist_ok=True)

    logger.info("Running Evidently drift report...")

    # ── Data Drift Report ──────────────────────────────────────────────────
    drift_report = Report(metrics=[DataDriftPreset()])

    drift_report.run(
        reference_data=reference,
        current_data=current
    )

    # Save HTML report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    html_path = os.path.join(output_dir, f"drift_report_{timestamp}.html")
    drift_report.save_html(html_path)
    logger.info(f"HTML report saved: {html_path}")

    # Extract metrics as dict
    report_dict = drift_report.as_dict()

    # Parse key metrics
    dataset_drift = report_dict['metrics'][0]['result']
    drift_metrics = {
        'dataset_drift_detected': dataset_drift.get('dataset_drift', False),
        'drift_share':            dataset_drift.get('drift_share', 0.0),
        'n_drifted_features':     dataset_drift.get('number_of_drifted_columns', 0),
        'n_features':             dataset_drift.get('number_of_columns', 0),
        'timestamp':              datetime.utcnow().isoformat(),
        'html_report':            html_path,
    }

    # Save JSON metrics
    json_path = os.path.join(output_dir, f"drift_metrics_{timestamp}.json")
    with open(json_path, 'w') as f:
        json.dump(drift_metrics, f, indent=2)

    logger.info(f"Drift metrics: {drift_metrics}")
    return drift_metrics

# ── Prometheus Metrics Publisher ───────────────────────────────────────────
def publish_drift_metrics(drift_metrics: dict) -> None:
    """
    Publish drift metrics to Prometheus so Grafana can visualize them.
    """
    from prometheus_client import Gauge, start_http_server

    drift_score = Gauge(
        'fraud_feature_drift_score',
        'Fraction of features that have drifted'
    )
    drift_detected = Gauge(
        'fraud_dataset_drift_detected',
        '1 if drift detected, 0 otherwise'
    )
    n_drifted = Gauge(
        'fraud_n_drifted_features',
        'Number of features with detected drift'
    )

    drift_score.set(drift_metrics['drift_share'])
    drift_detected.set(1 if drift_metrics['dataset_drift_detected'] else 0)
    n_drifted.set(drift_metrics['n_drifted_features'])

    logger.info("Drift metrics published to Prometheus on port 8001")

# ── Retraining Decision ────────────────────────────────────────────────────
def should_retrain(drift_metrics: dict, threshold: float = 0.3) -> bool:
    """
    Decide if model should be retrained based on drift score.
    Threshold of 0.3 means: retrain if >30% of features have drifted.
    """
    drift_share = drift_metrics.get('drift_share', 0.0)
    should = drift_share > threshold

    if should:
        logger.warning(
            f"⚠️  RETRAINING RECOMMENDED: "
            f"drift_share={drift_share:.2%} > threshold={threshold:.2%}"
        )
    else:
        logger.info(
            f"✅ No retraining needed: "
            f"drift_share={drift_share:.2%} <= threshold={threshold:.2%}"
        )

    return should

# ── Main ───────────────────────────────────────────────────────────────────
def main():
    print("="*50)
    print("EVIDENTLY DRIFT MONITORING")
    print("="*50)

    # Load data
    reference = load_reference_data(sample_size=10000)
    current = load_current_data(sample_size=5000)

    # Run drift report
    drift_metrics = run_drift_report(reference, current)

    # Print summary
    print(f"\n{'='*50}")
    print("DRIFT REPORT SUMMARY")
    print(f"{'='*50}")
    print(f"Drift detected:      {drift_metrics['dataset_drift_detected']}")
    print(f"Drift share:         {drift_metrics['drift_share']:.2%}")
    print(f"Drifted features:    {drift_metrics['n_drifted_features']}/{drift_metrics['n_features']}")
    print(f"HTML report saved:   {drift_metrics['html_report']}")

    # Retraining decision
    retrain = should_retrain(drift_metrics)
    print(f"\nRetrain model:       {'YES ⚠️' if retrain else 'NO ✅'}")
    print(f"{'='*50}")

    return drift_metrics

if __name__ == "__main__":
    main()