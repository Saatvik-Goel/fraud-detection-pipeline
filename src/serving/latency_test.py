import requests
import time
import statistics

URL = "http://localhost:3000/predict"

sample_transaction = {
    "TransactionAmt": 100.0,
    "card1": 5912,
    "card2": 321.0,
    "ProductCD": "W",
    "C1": 1.0,
    "D1": 14.0,
}

latencies = []

print("Warming up...")
for _ in range(10):
    requests.post(URL, json=sample_transaction)

print("Running 100 predictions...")

for i in range(100):
    start = time.time()

    try:
        r = requests.post(URL, json=sample_transaction, timeout=5)
        r.raise_for_status()
    except Exception as e:
        print(f"Request failed: {e}")
        continue

    latency = (time.time() - start) * 1000
    latencies.append(latency)

latencies.sort()

print(f"\n{'='*40}")
print(f"LATENCY REPORT ({len(latencies)} requests)")
print(f"{'='*40}")
print(f"p50:  {statistics.median(latencies):.1f} ms")
print(f"p95:  {latencies[int(0.95*len(latencies))-1]:.1f} ms")
print(f"p99:  {latencies[int(0.99*len(latencies))-1]:.1f} ms")
print(f"min:  {min(latencies):.1f} ms")
print(f"max:  {max(latencies):.1f} ms")
print(f"{'='*40}")