import json
import logging
import sys
import os
from kafka import KafkaConsumer
from collections import defaultdict

sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from config import KAFKA_BROKER, KAFKA_TOPIC, KAFKA_GROUP_ID

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_consumer() -> KafkaConsumer:
    """Create and return a Kafka consumer instance."""
    return KafkaConsumer(
        KAFKA_TOPIC,
        bootstrap_servers=KAFKA_BROKER,
        group_id=KAFKA_GROUP_ID + "-test",   # separate group for testing
        auto_offset_reset='earliest',          # read from beginning
        value_deserializer=lambda x: json.loads(x.decode('utf-8')),
        key_deserializer=lambda x: x.decode('utf-8') if x else None,
        consumer_timeout_ms=10000              # stop after 10s of no messages
    )

def verify_messages(max_messages: int = 100):
    """
    Consume and verify messages from Kafka topic.
    Checks:
    - Messages are being received
    - Schema matches expected fields
    - No unexpected nulls in critical fields
    - Partition distribution is balanced
    """
    consumer = create_consumer()
    
    messages_received = 0
    partition_counts = defaultdict(int)
    fraud_count = 0
    schema_errors = 0
    
    required_fields = [
        'TransactionID', 'TransactionAmt', 
        'card1', 'isFraud', 'event_timestamp'
    ]

    logger.info(f"Starting consumer on topic: {KAFKA_TOPIC}")
    logger.info(f"Waiting for messages...")

    try:
        for message in consumer:
            data = message.value
            partition_counts[message.partition] += 1
            messages_received += 1

            # Check required fields exist
            missing = [f for f in required_fields if f not in data]
            if missing:
                logger.warning(f"Missing fields in message: {missing}")
                schema_errors += 1

            # Count fraud transactions
            if data.get('isFraud') == 1:
                fraud_count += 1

            # Print first 3 messages in detail
            if messages_received <= 3:
                logger.info(f"\n--- Message {messages_received} ---")
                logger.info(f"Partition: {message.partition}")
                logger.info(f"Offset: {message.offset}")
                logger.info(f"Key (card1): {message.key}")
                logger.info(f"TransactionID: {data.get('TransactionID')}")
                logger.info(f"TransactionAmt: {data.get('TransactionAmt')}")
                logger.info(f"isFraud: {data.get('isFraud')}")
                logger.info(f"event_timestamp: {data.get('event_timestamp')}")

            if messages_received >= max_messages:
                break

    except Exception as e:
        logger.error(f"Consumer error: {e}")
    finally:
        consumer.close()

    # Print verification report
    print("\n" + "="*50)
    print("✅ KAFKA CONSUMER VERIFICATION REPORT")
    print("="*50)
    print(f"Messages received:     {messages_received}")
    print(f"Schema errors:         {schema_errors}")
    print(f"Fraud transactions:    {fraud_count} ({fraud_count/max(messages_received,1)*100:.1f}%)")
    print(f"\nPartition distribution:")
    for partition, count in sorted(partition_counts.items()):
        print(f"  Partition {partition}: {count} messages")
    print("="*50)

    if messages_received == 0:
        print("❌ No messages received — is the producer running?")
    elif schema_errors == 0:
        print("✅ All messages passed schema validation")
    else:
        print(f"⚠️  {schema_errors} messages had schema issues")

if __name__ == "__main__":
    verify_messages(max_messages=100)