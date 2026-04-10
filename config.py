# config.py
import os
from dotenv import load_dotenv

load_dotenv()

# Kafka
KAFKA_BROKER = os.getenv("KAFKA_BROKER", "localhost:9092")
KAFKA_TOPIC = os.getenv("KAFKA_TOPIC", "transactions")
KAFKA_GROUP_ID = os.getenv("KAFKA_GROUP_ID", "fraud-detection-group")

# Redis
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_TTL = int(os.getenv("REDIS_TTL", 3600))

# MLflow
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "fraud-detection")

# Model serving
BENTOML_PORT = int(os.getenv("BENTOML_PORT", 3000))

# Paths
DATA_RAW_PATH = os.getenv("DATA_RAW_PATH", "data/raw/")
DATA_PROCESSED_PATH = os.getenv("DATA_PROCESSED_PATH", "data/processed/")
FEAST_REPO_PATH = os.getenv("FEAST_REPO_PATH", "./fraud_feature_repo")