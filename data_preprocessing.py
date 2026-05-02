import json
import pandas as pd
import gzip

def process_cluster_data(filepath, max_rows=500000):
    """Parse compressed Google Cluster JSON traces into a multivariate DataFrame."""
    extracted = []

    with gzip.open(filepath, 'rt') as f:
        for i, line in enumerate(f):
            if i >= max_rows: break
            try:
                row = json.loads(line)
                avg = row.get("average_usage", {})
                mx = row.get("maximum_usage", {})
                extracted.append({
                    "time": int(row.get("start_time", 0)),
                    "avg_cpu": float(avg.get("cpus", 0.0)),
                    "avg_memory": float(avg.get("memory", 0.0)),
                    "max_cpu": float(mx.get("cpus", 0.0)),
                    "assigned_memory": float(row.get("assigned_memory", 0.0)),
                    "cpi": float(row.get("cycles_per_instruction", 0.0))
                })
            except (json.JSONDecodeError, ValueError):
                continue

    return pd.DataFrame(extracted).sort_values("time").reset_index(drop=True)

if __name__ == "__main__":
    try:
        df = process_cluster_data("instance_usage-000000000000.json.gz")
        df.to_csv("clean_workload_data.csv", index=False)
        print(f"Exported {len(df)} records with {len(df.columns)} columns.")
        print(f"Columns: {list(df.columns)}")
    except Exception as e:
        print(f"Pipeline error: {e}")