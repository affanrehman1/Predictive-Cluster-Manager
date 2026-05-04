import json
import pandas as pd
import gzip
import glob

ROLLING_WINDOWS = (3, 5, 10, 30)

def aggregate_workload_by_time(df):
    """Aggregate machine workload data by timestamp."""
    if df.empty:
        return df

    required = ['time', 'avg_cpu', 'avg_memory', 'max_cpu', 'assigned_memory', 'cpi']
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns required for aggregation: {missing}")

    return (
        df[required]
        .groupby('time', as_index=False)
        .agg({
            'avg_cpu': 'sum',
            'avg_memory': 'sum',
            'max_cpu': 'sum',
            'assigned_memory': 'sum',
            'cpi': 'mean',
        })
        .sort_values('time')
        .reset_index(drop=True)
    )

def add_spike_features(df):
    """Compute and append temporal context features for spike recognition."""
    df = aggregate_workload_by_time(df)

    df['cpu_delta'] = df['avg_cpu'].diff().fillna(0)
    df['mem_delta'] = df['avg_memory'].diff().fillna(0)
    df['cpu_abs_delta'] = df['cpu_delta'].abs()
    df['mem_abs_delta'] = df['mem_delta'].abs()
    df['cpu_spike_pressure'] = (df['max_cpu'] - df['avg_cpu']).clip(lower=0)

    if 'time' in df.columns:
        df['time_delta'] = df['time'].diff().fillna(0)

    for window in ROLLING_WINDOWS:
        df[f'cpu_roll_mean_{window}'] = df['avg_cpu'].rolling(window=window, min_periods=1).mean()
        df[f'cpu_roll_std_{window}'] = df['avg_cpu'].rolling(window=window, min_periods=2).std().fillna(0)
        df[f'cpu_roll_max_{window}'] = df['avg_cpu'].rolling(window=window, min_periods=1).max()
        df[f'mem_roll_mean_{window}'] = df['avg_memory'].rolling(window=window, min_periods=1).mean()
        df[f'mem_roll_std_{window}'] = df['avg_memory'].rolling(window=window, min_periods=2).std().fillna(0)
        df[f'mem_roll_max_{window}'] = df['avg_memory'].rolling(window=window, min_periods=1).max()

    cpu_ewm_fast = df['avg_cpu'].ewm(span=5, adjust=False).mean()
    cpu_ewm_slow = df['avg_cpu'].ewm(span=30, adjust=False).mean()
    mem_ewm_fast = df['avg_memory'].ewm(span=5, adjust=False).mean()
    mem_ewm_slow = df['avg_memory'].ewm(span=30, adjust=False).mean()
    df['cpu_ewm_gap'] = cpu_ewm_fast - cpu_ewm_slow
    df['mem_ewm_gap'] = mem_ewm_fast - mem_ewm_slow

    return df.replace([float('inf'), float('-inf')], 0).fillna(0)

def process_cluster_data(target_machine_id):
    """Extract and process Google Cluster JSON traces for a specified machine."""
    extracted = []
    files = sorted(glob.glob("instance_usage-*.json.gz"))
    
    print(f"Searching across {len(files)} files for Machine {target_machine_id}...")
    
    for filepath in files:
        print(f"Processing {filepath}...")
        with gzip.open(filepath, 'rt') as f:
            for line in f:
                try:
                    row = json.loads(line)
                    if row.get("machine_id") != target_machine_id:
                        continue
                        
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

    print(f"Found {len(extracted)} instance records. Aggregating by time and calculating spike-aware features...")
    return add_spike_features(pd.DataFrame(extracted))

if __name__ == "__main__":
    TARGET_MACHINE = "259065338485"
    try:
        df = process_cluster_data(target_machine_id=TARGET_MACHINE)
        df.to_csv("clean_workload_data.csv", index=False)
        print(f"\n[SUCCESS] Exported {len(df)} records for Machine {TARGET_MACHINE}.")
        print(f"Features: {list(df.columns)}")
    except Exception as e:
        print(f"Pipeline error: {e}")
