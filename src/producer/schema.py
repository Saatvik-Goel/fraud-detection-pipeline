from pyspark.sql.types import (
    StructType, StructField,
    LongType, IntegerType, DoubleType, StringType
)

import pandas as pd

# Load usable columns from CSV
usable_cols_df = pd.read_csv("data/processed/usable_columns.csv")
USABLE_COLUMNS = usable_cols_df.iloc[:, 0].dropna().tolist()

# 🔧 Define a mapping for known column types
TYPE_MAPPING = {
    "TransactionID": LongType(),
    "TransactionDT": LongType(),
    "TransactionAmt": DoubleType(),
    "isFraud": IntegerType(),

    "ProductCD": StringType(),
    "card4": StringType(),
    "card6": StringType(),
    "P_emaildomain": StringType(),
    "R_emaildomain": StringType(),

    # M features (categorical)
    "M1": StringType(),
    "M2": StringType(),
    "M3": StringType(),
    "M4": StringType(),
    "M6": StringType(),
}

# 🧠 Function to infer type
def infer_type(col):
    if col in TYPE_MAPPING:
        return TYPE_MAPPING[col]
    elif col.startswith("card") or col.startswith("addr"):
        return DoubleType()
    elif col.startswith("C") or col.startswith("D") or col.startswith("V"):
        return DoubleType()
    else:
        return StringType()  # fallback

# 🏗️ Build schema dynamically
TRANSACTION_SCHEMA = StructType([
    StructField(col, infer_type(col), True)
    for col in USABLE_COLUMNS
])

# Columns for producer
PRODUCER_COLUMNS = USABLE_COLUMNS