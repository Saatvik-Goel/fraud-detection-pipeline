import json
import time
import logging
import pandas as pd
from kafka import KafkaProducer
from kafka.errors import KafkaError
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from config import KAFKA_BROKER, KAFKA_TOPIC, PRODUCER_COLUMNS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_producer() -> KafkaProducer:
    """Create and return a Kafka producer instance."""
    return KafkaProducer(
        bootstrap_servers=KAFKA_BROKER,
        value_serializer=lambda x: json.dumps(x).encode('utf-8'),
        key_serializer=lambda x: str(x).encode('utf-8'),
        acks='all',                  # wait for all replicas to confirm
        retries=3,
        max_block_ms=10000
    )

def load_data(filepath: str) -> pd.DataFrame:
    """Load and preprocess transaction data."""
    logger.info(f"Loading data from {filepath}")
    df = pd.read_csv(filepath, usecols=PRODUCER_COLUMNS)

    # Fill missing values simply — Spark will handle advanced imputation
    df['TransactionAmt'] = df['TransactionAmt'].fillna(0.0)
    df['card1'] = df['card1'].fillna(-1).astype(int)
    df['card2'] = df['card2'].fillna(-1)

    # Add event timestamp — simulate real-time by using current time
    df['event_timestamp'] = pd.Timestamp.now().isoformat()

    logger.info(f"Loaded {len(df)} transactions")
    return df

def on_success(record_metadata):
    logger.debug(
        f"Message sent → topic: {record_metadata.topic} "
        f"partition: {record_metadata.partition} "
        f"offset: {record_metadata.offset}"
    )

def on_error(excp):
    logger.error(f"Failed to send message: {excp}")

def stream_transactions(
    filepath: str,
    events_per_second: int = 100,
    max_events: int = None
):
    """
    Stream transactions from CSV to Kafka topic.
    
    Args:
        filepath: path to train_transaction.csv
        events_per_second: how fast to stream (default 100/sec)
        max_events: stop after this many events (None = stream all)
    """
    producer = create_producer()
    df = load_data(filepath)
    
    sleep_time = 1.0 / events_per_second
    total_sent = 0
    
    logger.info(f"Starting stream at {events_per_second} events/sec...")
    logger.info(f"Publishing to topic: {KAFKA_TOPIC}")

    try:
        for idx, row in df.iterrows():
            # Use card1 as the message key
            # This ensures same card always goes to same partition
            key = row['card1']
            value = row.to_dict()

            # Handle NaN values — JSON can't serialize NaN
            value = {
                k: (None if pd.isna(v) else v)
                for k, v in value.items()
            }

            future = producer.send(
                KAFKA_TOPIC,
                key=key,
                value=value
            )
            future.add_callback(on_success)
            future.add_errback(on_error)

            total_sent += 1

            # Log progress every 1000 messages
            if total_sent % 1000 == 0:
                logger.info(f"Sent {total_sent} transactions...")

            # Stop if max_events reached
            if max_events and total_sent >= max_events:
                break

            time.sleep(sleep_time)

    except KeyboardInterrupt:
        logger.info(f"Streaming stopped. Total sent: {total_sent}")
    finally:
        producer.flush()   # make sure all buffered messages are sent
        producer.close()
        logger.info(f"Producer closed. Total sent: {total_sent}")

if __name__ == "__main__":
    stream_transactions(
        filepath="data/raw/train_transaction.csv",
        events_per_second=100,
        max_events=10000          # stream first 10k rows for testing
    )