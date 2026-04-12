import logging
import sys
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    from_json, col, window, avg, count,
    stddev, max as spark_max, min as spark_min,
    current_timestamp, when, lit
)
from pyspark.sql.types import (
    StructType, StructField, StringType,
    DoubleType, IntegerType, LongType
)

sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from config import KAFKA_BROKER, KAFKA_TOPIC

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Spark Session ──────────────────────────────────────────────────────────
def create_spark_session() -> SparkSession:
    return SparkSession.builder \
        .appName("FraudDetectionStreaming") \
        .config("spark.jars.packages",
                "org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.3") \
        .config("spark.sql.streaming.checkpointLocation", "checkpoints/") \
        .config("spark.sql.shuffle.partitions", "4") \
        .config("spark.driver.memory", "2g") \
        .getOrCreate()

# ── Schema ─────────────────────────────────────────────────────────────────
# Use a focused subset of columns for streaming features
# Full 220 cols would be too heavy for real-time processing
STREAM_SCHEMA = StructType([
    StructField("TransactionID", LongType(), True),
    StructField("TransactionDT", LongType(), True),
    StructField("TransactionAmt", DoubleType(), True),
    StructField("ProductCD", StringType(), True),
    StructField("card1", IntegerType(), True),
    StructField("card2", DoubleType(), True),
    StructField("card4", StringType(), True),
    StructField("card6", StringType(), True),
    StructField("addr1", DoubleType(), True),
    StructField("P_emaildomain", StringType(), True),
    StructField("C1", DoubleType(), True),
    StructField("C2", DoubleType(), True),
    StructField("C6", DoubleType(), True),
    StructField("C11", DoubleType(), True),
    StructField("D1", DoubleType(), True),
    StructField("D4", DoubleType(), True),
    StructField("D10", DoubleType(), True),
    StructField("V1", DoubleType(), True),
    StructField("V2", DoubleType(), True),
    StructField("V3", DoubleType(), True),
    StructField("isFraud", IntegerType(), True),
    StructField("event_timestamp", StringType(), True),
])

# ── Read from Kafka ────────────────────────────────────────────────────────
def read_kafka_stream(spark: SparkSession):
    return spark.readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", KAFKA_BROKER) \
        .option("subscribe", KAFKA_TOPIC) \
        .option("startingOffsets", "latest") \
        .option("failOnDataLoss", "false") \
        .load()

# ── Parse JSON messages ────────────────────────────────────────────────────
def parse_stream(raw_df):
    return raw_df \
        .select(
            from_json(
                col("value").cast("string"),
                STREAM_SCHEMA
            ).alias("data"),
            col("timestamp").alias("kafka_timestamp")
        ) \
        .select("data.*", "kafka_timestamp")

# ── Compute Features ───────────────────────────────────────────────────────
def compute_features(parsed_df):
    """
    Compute real-time features per card1 using sliding windows.
    Features:
        - amt_mean_1h:    avg transaction amount in last 1 hour
        - amt_std_1h:     std dev of transaction amount in last 1 hour
        - txn_count_1h:   number of transactions in last 1 hour
        - amt_max_1h:     max transaction amount in last 1 hour
        - amt_min_1h:     min transaction amount in last 1 hour
    """
    return parsed_df \
        .withWatermark("kafka_timestamp", "10 minutes") \
        .groupBy(
            window(col("kafka_timestamp"), "1 hour", "10 minutes"),
            col("card1")
        ) \
        .agg(
            avg("TransactionAmt").alias("amt_mean_1h"),
            stddev("TransactionAmt").alias("amt_std_1h"),
            count("*").alias("txn_count_1h"),
            spark_max("TransactionAmt").alias("amt_max_1h"),
            spark_min("TransactionAmt").alias("amt_min_1h"),
            avg("C1").alias("c1_mean_1h"),
            avg("D1").alias("d1_mean_1h"),
        ) \
        .select(
            col("card1"),
            col("window.start").alias("window_start"),
            col("window.end").alias("window_end"),
            col("amt_mean_1h"),
            col("amt_std_1h"),
            col("txn_count_1h"),
            col("amt_max_1h"),
            col("amt_min_1h"),
            col("c1_mean_1h"),
            col("d1_mean_1h"),
        )

# ── Write to Console (for testing) ────────────────────────────────────────
def write_to_console(feature_df):
    return feature_df.writeStream \
        .outputMode("update") \
        .format("console") \
        .option("truncate", "false") \
        .trigger(processingTime="30 seconds") \
        .start()

# ── Write to Redis (via foreachBatch) ─────────────────────────────────────
def write_to_redis(feature_df):
    """
    Write computed features to Redis using foreachBatch.
    Person B's redis_writer.py handles the actual Redis writes.
    """
    from src.features.redis_writer import write_features_batch

    return feature_df.writeStream \
        .outputMode("update") \
        .foreachBatch(write_features_batch) \
        .trigger(processingTime="30 seconds") \
        .option("checkpointLocation", "checkpoints/redis/") \
        .start()

# ── Main ───────────────────────────────────────────────────────────────────
def main():
    spark = create_spark_session()
    spark.sparkContext.setLogLevel("WARN")

    logger.info("Starting Spark Structured Streaming...")
    logger.info(f"Reading from Kafka topic: {KAFKA_TOPIC}")

    # Read and parse stream
    raw_df = read_kafka_stream(spark)
    parsed_df = parse_stream(raw_df)
    feature_df = compute_features(parsed_df)

    # Step 1: Write to console first to verify features are computed correctly
    console_query = write_to_redis(feature_df)

    logger.info("Streaming started. Waiting for data...")
    logger.info("Features will appear every 30 seconds.")
    logger.info("Press Ctrl+C to stop.")

    try:
        console_query.awaitTermination()
    except KeyboardInterrupt:
        logger.info("Stopping stream...")
        console_query.stop()
        spark.stop()

if __name__ == "__main__":
    main()