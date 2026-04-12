import redis
import json
import logging
import sys
import os
from typing import Dict, Any
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from config import REDIS_HOST, REDIS_PORT, REDIS_TTL

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Redis Connection ───────────────────────────────────────────────────────
def get_redis_client() -> redis.Redis:
    """Create and return a Redis client with connection pooling."""
    pool = redis.ConnectionPool(
        host=REDIS_HOST,
        port=REDIS_PORT,
        db=0,
        max_connections=20,
        decode_responses=True
    )
    return redis.Redis(connection_pool=pool)

# ── Feature Key Format ─────────────────────────────────────────────────────
def make_feature_key(card1: int) -> str:
    """
    Consistent key format for Redis.
    Always use this function — never hardcode keys.
    """
    return f"features:card:{card1}"

# ── Write Single Card Features ─────────────────────────────────────────────
def write_card_features(
    r: redis.Redis,
    card1: int,
    features: Dict[str, Any]
) -> bool:
    """
    Write features for a single card to Redis as a Hash.
    Returns True if successful.
    """
    key = make_feature_key(card1)

    # Clean features — Redis can't store None or NaN
    clean_features = {}
    for k, v in features.items():
        if v is None or (isinstance(v, float) and v != v):  # NaN check
            clean_features[k] = "null"
        else:
            clean_features[k] = str(round(float(v), 4)) if isinstance(v, float) else str(v)

    # Add metadata
    clean_features['last_updated'] = datetime.utcnow().isoformat()
    clean_features['card1'] = str(card1)

    try:
        pipe = r.pipeline()
        pipe.hset(key, mapping=clean_features)
        pipe.expire(key, REDIS_TTL)   # expire after 1 hour
        pipe.execute()
        return True
    except Exception as e:
        logger.error(f"Failed to write features for card {card1}: {e}")
        return False

# ── Batch Writer (called by Spark foreachBatch) ────────────────────────────
def write_features_batch(batch_df, batch_id: int) -> None:
    """
    Called by Spark's foreachBatch for each micro-batch.
    Writes all card features in the batch to Redis.
    """
    if batch_df.isEmpty():
        logger.info(f"Batch {batch_id}: empty, skipping")
        return

    r = get_redis_client()
    success_count = 0
    fail_count = 0

    rows = batch_df.collect()
    logger.info(f"Batch {batch_id}: writing {len(rows)} card features to Redis")

    for row in rows:
        card1 = row['card1']
        if card1 is None or card1 == -1:
            continue

        features = {
            'amt_mean_1h':   row['amt_mean_1h'],
            'amt_std_1h':    row['amt_std_1h'],
            'txn_count_1h':  row['txn_count_1h'],
            'amt_max_1h':    row['amt_max_1h'],
            'amt_min_1h':    row['amt_min_1h'],
            'c1_mean_1h':    row['c1_mean_1h'],
            'd1_mean_1h':    row['d1_mean_1h'],
            'window_start':  str(row['window_start']),
            'window_end':    str(row['window_end']),
        }

        if write_card_features(r, card1, features):
            success_count += 1
        else:
            fail_count += 1

    logger.info(
        f"Batch {batch_id} complete — "
        f"success: {success_count}, failed: {fail_count}"
    )

# ── Read Features Back (for verification) ─────────────────────────────────
def read_card_features(card1: int) -> Dict[str, Any]:
    """
    Read features for a card from Redis.
    Used by the serving layer to get real-time features.
    """
    r = get_redis_client()
    key = make_feature_key(card1)

    features = r.hgetall(key)
    if not features:
        logger.warning(f"No features found for card {card1}")
        return {}

    # Convert numeric strings back to floats
    result = {}
    for k, v in features.items():
        if v == 'null':
            result[k] = None
        else:
            try:
                result[k] = float(v)
            except ValueError:
                result[k] = v

    return result

# ── Verification Script ────────────────────────────────────────────────────
def verify_redis_features(sample_cards: list) -> None:
    """
    Verify features are being written correctly to Redis.
    Run this after the Spark stream has processed some data.
    """
    r = get_redis_client()

    print("\n" + "="*50)
    print("REDIS FEATURE VERIFICATION")
    print("="*50)

    # Check total keys
    all_keys = r.keys("features:card:*")
    print(f"Total cards with features: {len(all_keys)}")

    # Sample a few cards
    for card1 in sample_cards[:3]:
        features = read_card_features(card1)
        if features:
            print(f"\nCard {card1}:")
            for k, v in features.items():
                print(f"  {k}: {v}")
            ttl = r.ttl(make_feature_key(card1))
            print(f"  TTL: {ttl} seconds")
        else:
            print(f"\nCard {card1}: No features found yet")

    print("="*50)

if __name__ == "__main__":
    # Standalone test — write dummy features and read them back
    r = get_redis_client()

    # Test connection
    print(f"Redis ping: {r.ping()}")

    # Write test features
    test_features = {
        'amt_mean_1h': 127.5,
        'amt_std_1h': 45.2,
        'txn_count_1h': 3,
        'amt_max_1h': 200.0,
        'amt_min_1h': 50.0,
        'c1_mean_1h': 1.0,
        'd1_mean_1h': 14.0,
        'window_start': '2024-01-01 10:00:00',
        'window_end': '2024-01-01 11:00:00',
    }

    success = write_card_features(r, card1=12345, features=test_features)
    print(f"Write test: {'✅ Success' if success else '❌ Failed'}")

    # Read back
    retrieved = read_card_features(card1=12345)
    print(f"Read back: {retrieved}")

    # Verify TTL
    ttl = r.ttl(make_feature_key(12345))
    print(f"TTL: {ttl} seconds (should be ~{REDIS_TTL})")