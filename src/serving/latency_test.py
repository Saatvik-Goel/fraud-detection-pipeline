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
print("Running 100 predictions to measure latency...")

for i in range(100):
    start = time.time()
    r = requests.post(URL, json=sample_transaction)
    latency = (time.time() - start) * 1000
    latencies.append(latency)

print(f"\n{'='*40}")
print(f"LATENCY REPORT (100 requests)")
print(f"{'='*40}")
print(f"p50:  {statistics.median(latencies):.1f}ms")
print(f"p95:  {sorted(latencies)[94]:.1f}ms")
print(f"p99:  {sorted(latencies)[98]:.1f}ms")
print(f"min:  {min(latencies):.1f}ms")
print(f"max:  {max(latencies):.1f}ms")
print(f"{'='*40}")