import pandas as pd
from schema import PRODUCER_COLUMNS

def validate_data(filepath: str) -> None:
    df = pd.read_csv(filepath, nrows=1000)
    
    missing_cols = [col for col in PRODUCER_COLUMNS if col not in df.columns]
    extra_cols = [col for col in df.columns if col not in PRODUCER_COLUMNS]
    
    print(f"✅ Columns found: {len(PRODUCER_COLUMNS) - len(missing_cols)}/{len(PRODUCER_COLUMNS)}")
    
    if missing_cols:
        print(f"⚠️  Missing from dataset: {missing_cols}")
    
    print(f"📊 Dataset shape: {df.shape}")
    print(f"🎯 Fraud rate: {df['isFraud'].mean():.2%}")
    print("Schema validation complete.")

if __name__ == "__main__":
    validate_data("data/raw/train_transaction.csv")