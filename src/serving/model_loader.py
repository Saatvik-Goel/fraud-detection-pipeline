import mlflow
import mlflow.xgboost
import os
import sys
import logging

sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from config import MLFLOW_TRACKING_URI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_latest_model():
    """
    Load the latest registered FraudDetectionModel from MLflow.
    Returns the model and its run metadata.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()

    # Get latest version of registered model
    model_name = "FraudDetectionModel"
    latest_versions = client.get_latest_versions(model_name)

    if not latest_versions:
        raise ValueError(f"No versions found for model: {model_name}")

    latest = latest_versions[-1]
    logger.info(f"Loading model: {model_name} v{latest.version}")
    logger.info(f"Run ID: {latest.run_id}")

    # Load model from MLflow
    model_uri = f"models:/{model_name}/{latest.version}"
    model = mlflow.xgboost.load_model(model_uri)

    logger.info("Model loaded successfully")
    return model, latest

def save_model_to_bentoml(model, model_name: str = "fraud_detector"):
    """
    Save the MLflow model into BentoML model store.
    Only needs to be run once — or when model is updated.
    """
    import bentoml

    # Save to BentoML store
    bento_model = bentoml.xgboost.save_model(
        model_name,
        model,
        signatures={
            "predict": {"batchable": True, "batch_dim": 0},
            "predict_proba": {"batchable": True, "batch_dim": 0},
        },
        metadata={
            "description": "XGBoost fraud detection model",
            "mlflow_tracking_uri": MLFLOW_TRACKING_URI,
        }
    )

    logger.info(f"Model saved to BentoML: {bento_model.tag}")
    return bento_model

if __name__ == "__main__":
    model, metadata = load_latest_model()
    bento_model = save_model_to_bentoml(model)
    print(f"✅ Model saved: {bento_model.tag}")