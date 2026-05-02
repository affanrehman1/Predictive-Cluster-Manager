import json
import pandas as pd
import gzip
import os

def process_cluster_data(filepath, max_rows=500000):
    """Parse compressed Google Cluster JSON traces into a DataFrame."""
    extracted = []
    
    with gzip.open(filepath, 'rt') as f:
        for i, line in enumerate(f):
            if i >= max_rows: break
            try:
                row = json.loads(line)
                usage = row.get("average_usage", {})
                extracted.append({
                    "time": int(row.get("start_time", 0)),
                    "machine_id": row.get("machine_id", "unknown"),
                    "cpu_usage": float(usage.get("cpus", 0.0)),
                    "memory_usage": float(usage.get("memory", 0.0))
                })
            except (json.JSONDecodeError, ValueError):
                continue

    return pd.DataFrame(extracted).sort_values("time").reset_index(drop=True)

if __name__ == "__main__":
    try:
        df = process_cluster_data("instance_usage-000000000000.json.gz")
        df.to_csv("clean_workload_data.csv", index=False)
        print(f"Exported {len(df)} records.")
    except Exception as e:
        print(f"Pipeline error: {e}")