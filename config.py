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

# Producer columns — derived from Person B's EDA (columns with <50% missing)
PRODUCER_COLUMNS = [
    # ── Identifiers & Target ──────────────────────
    'TransactionID', 'TransactionDT', 'TransactionAmt', 'isFraud',

    # ── Product & Card Info ───────────────────────
    'ProductCD',
    'card1', 'card2', 'card3', 'card4', 'card5', 'card6',

    # ── Address ───────────────────────────────────
    'addr1', 'addr2',

    # ── Email ─────────────────────────────────────
    'P_emaildomain',

    # ── Count Features (C1-C14) ───────────────────
    'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8',
    'C9', 'C10', 'C11', 'C12', 'C13', 'C14',

    # ── Time Delta Features (D series) ───────────
    'D1', 'D2', 'D3', 'D4', 'D10', 'D11', 'D15',

    # ── Match Features (M series) ─────────────────
    'M1', 'M2', 'M3', 'M4', 'M6',

    # ── Vesta Engineered Features (V series) ──────
    'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
    'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
    'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'V29', 'V30',
    'V31', 'V32', 'V33', 'V34', 'V35', 'V36', 'V37', 'V38', 'V39', 'V40',
    'V41', 'V42', 'V43', 'V44', 'V45', 'V46', 'V47', 'V48', 'V49', 'V50',
    'V51', 'V52', 'V53', 'V54', 'V55', 'V56', 'V57', 'V58', 'V59', 'V60',
    'V61', 'V62', 'V63', 'V64', 'V65', 'V66', 'V67', 'V68', 'V69', 'V70',
    'V71', 'V72', 'V73', 'V74', 'V75', 'V76', 'V77', 'V78', 'V79', 'V80',
    'V81', 'V82', 'V83', 'V84', 'V85', 'V86', 'V87', 'V88', 'V89', 'V90',
    'V91', 'V92', 'V93', 'V94', 'V95', 'V96', 'V97', 'V98', 'V99', 'V100',
    'V101', 'V102', 'V103', 'V104', 'V105', 'V106', 'V107', 'V108', 'V109', 'V110',
    'V111', 'V112', 'V113', 'V114', 'V115', 'V116', 'V117', 'V118', 'V119', 'V120',
    'V121', 'V122', 'V123', 'V124', 'V125', 'V126', 'V127', 'V128', 'V129', 'V130',
    'V131', 'V132', 'V133', 'V134', 'V135', 'V136', 'V137',
    'V279', 'V280', 'V281', 'V282', 'V283', 'V284', 'V285', 'V286', 'V287', 'V288',
    'V289', 'V290', 'V291', 'V292', 'V293', 'V294', 'V295', 'V296', 'V297', 'V298',
    'V299', 'V300', 'V301', 'V302', 'V303', 'V304', 'V305', 'V306', 'V307', 'V308',
    'V309', 'V310', 'V311', 'V312', 'V313', 'V314', 'V315', 'V316', 'V317', 'V318',
    'V319', 'V320', 'V321',
]