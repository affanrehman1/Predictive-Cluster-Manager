import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def create_sequences(data, look_back=10):
    """Transform univariate time-series into (N, T, F) tensors."""
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i : i + look_back, 0])
        y.append(data[i + look_back, 0])
    
    X_arr, y_arr = np.array(X), np.array(y)
    return X_arr.reshape(X_arr.shape[0], X_arr.shape[1], 1), y_arr

def prepare_pipeline(csv_path, look_back=10, train_split=0.8):
    """Execute feature scaling and temporal partitioning."""
    df = pd.read_csv(csv_path)
    cpu_data = df['cpu_usage'].values.reshape(-1, 1)

    scaler = MinMaxScaler(feature_range=(0, 1))
    cpu_scaled = scaler.fit_transform(cpu_data)

    X, y = create_sequences(cpu_scaled, look_back=look_back)

    split_idx = int(len(X) * train_split)
    return (X[:split_idx], y[:split_idx]), (X[split_idx:], y[split_idx:]), scaler

if __name__ == "__main__":
    (X_train, y_train), (X_test, y_test), _ = prepare_pipeline("clean_workload_data.csv")
    print(f"Dataset prepared: Train {X_train.shape}, Test {X_test.shape}")
