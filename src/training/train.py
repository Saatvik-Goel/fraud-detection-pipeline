import pandas as pd
import numpy as np
import mlflow
import mlflow.xgboost
import os
import sys
import logging
from datetime import datetime

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score, roc_auc_score,
    precision_score, recall_score,
    confusion_matrix, classification_report
)
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from config import (
    DATA_RAW_PATH, DATA_PROCESSED_PATH,
    MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME,
    PRODUCER_COLUMNS
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── MLflow Setup ───────────────────────────────────────────────────────────
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

# ── Load Data ──────────────────────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    logger.info("Loading transaction data...")
    df = pd.read_csv(
        os.path.join(DATA_RAW_PATH, 'train_transaction.csv'),
        usecols=PRODUCER_COLUMNS
    )
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    logger.info(f"Fraud rate: {df['isFraud'].mean():.2%}")
    return df

# ── Preprocessing ──────────────────────────────────────────────────────────
def preprocess(df: pd.DataFrame):
    """
    Clean and encode features for training.
    """
    logger.info("Preprocessing data...")

    # Drop identifier columns — not useful for model
    drop_cols = ['TransactionID', 'TransactionDT']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Encode categorical columns
    cat_cols = ['ProductCD', 'card4', 'card6', 'P_emaildomain',
                'M1', 'M2', 'M3', 'M4', 'M6']

    le = LabelEncoder()
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)
            df[col] = le.fit_transform(df[col])

    # Fill remaining nulls with median
    df = df.fillna(df.median(numeric_only=True))

    # Split features and target
    X = df.drop(columns=['isFraud'])
    y = df['isFraud']

    logger.info(f"Features shape: {X.shape}")
    logger.info(f"Class distribution: {y.value_counts().to_dict()}")

    return X, y

# ── Handle Class Imbalance ─────────────────────────────────────────────────
def apply_smote(X_train, y_train):
    """
    Apply SMOTE to balance the training set.
    Only applied to training data — never to test data.
    """
    logger.info("Applying SMOTE to handle class imbalance...")
    logger.info(f"Before SMOTE: {y_train.value_counts().to_dict()}")

    smote = SMOTE(
        sampling_strategy=0.3,   # bring minority to 30% of majority
        random_state=42,
        n_jobs=-1
    )
    X_res, y_res = smote.fit_resample(X_train, y_train)

    logger.info(f"After SMOTE: {pd.Series(y_res).value_counts().to_dict()}")
    return X_res, y_res

# ── Train Model ────────────────────────────────────────────────────────────
def train_model(X_train, y_train):
    """Train XGBoost classifier."""
    logger.info("Training XGBoost model...")

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=30,     # additional weight for fraud class
        use_label_encoder=False,
        eval_metric='auc',
        random_state=42,
        n_jobs=-1,
        tree_method='hist',      # faster training
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train)],
        verbose=100
    )

    return model

# ── Evaluate Model ─────────────────────────────────────────────────────────
def evaluate_model(model, X_test, y_test) -> dict:
    """Compute and return all evaluation metrics."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        'f1_score':       round(f1_score(y_test, y_pred), 4),
        'roc_auc':        round(roc_auc_score(y_test, y_prob), 4),
        'precision':      round(precision_score(y_test, y_pred), 4),
        'recall':         round(recall_score(y_test, y_pred), 4),
        'fraud_detected': int(y_pred[y_test == 1].sum()),
        'total_fraud':    int(y_test.sum()),
    }

    metrics['fraud_catch_rate'] = round(
        metrics['fraud_detected'] / max(metrics['total_fraud'], 1), 4
    )

    logger.info("\n" + classification_report(y_test, y_pred,
                target_names=['Not Fraud', 'Fraud']))

    return metrics

# ── Main Training Pipeline ─────────────────────────────────────────────────
def run_training():
    logger.info("="*50)
    logger.info("Starting fraud detection model training")
    logger.info("="*50)

    # Load and preprocess
    df = load_data()
    X, y = preprocess(df)

    # Train/test split — stratified to preserve fraud ratio
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    # Apply SMOTE only on training data
    X_train_res, y_train_res = apply_smote(X_train, y_train)

    # ── MLflow Run ─────────────────────────────────────────────────────────
    with mlflow.start_run(run_name=f"xgboost_{datetime.now().strftime('%Y%m%d_%H%M')}"):

        # Log parameters
        mlflow.log_params({
            'model_type':        'XGBoost',
            'n_estimators':      300,
            'max_depth':         6,
            'learning_rate':     0.05,
            'subsample':         0.8,
            'colsample_bytree':  0.8,
            'scale_pos_weight':  30,
            'smote_strategy':    0.3,
            'test_size':         0.2,
            'train_size':        len(X_train_res),
            'n_features':        X.shape[1],
        })

        # Train
        model = train_model(X_train_res, y_train_res)

        # Evaluate
        metrics = evaluate_model(model, X_test, y_test)

        # Log metrics
        mlflow.log_metrics(metrics)

        # Log feature importance
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        # Save feature importance as artifact
        os.makedirs(DATA_PROCESSED_PATH, exist_ok=True)
        fi_path = os.path.join(DATA_PROCESSED_PATH, 'feature_importance.csv')
        feature_importance.to_csv(fi_path, index=False)
        mlflow.log_artifact(fi_path)

        # Log model
        mlflow.xgboost.log_model(
            model,
            artifact_path="fraud_model",
            registered_model_name="FraudDetectionModel"
        )

        run_id = mlflow.active_run().info.run_id

        logger.info("\n" + "="*50)
        logger.info("TRAINING COMPLETE")
        logger.info("="*50)
        logger.info(f"Run ID:          {run_id}")
        logger.info(f"F1 Score:        {metrics['f1_score']}")
        logger.info(f"ROC AUC:         {metrics['roc_auc']}")
        logger.info(f"Precision:       {metrics['precision']}")
        logger.info(f"Recall:          {metrics['recall']}")
        logger.info(f"Fraud Catch Rate:{metrics['fraud_catch_rate']}")
        logger.info("="*50)
        logger.info(f"View in MLflow: http://localhost:5000")

        return model, metrics, run_id

if __name__ == "__main__":
    run_training()