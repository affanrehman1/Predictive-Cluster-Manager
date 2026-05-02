import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# 5 input features, 2 target outputs
INPUT_FEATURES = ['avg_cpu', 'avg_memory', 'max_cpu', 'assigned_memory', 'cpi']
TARGET_FEATURES = ['avg_cpu', 'avg_memory']

def create_sequences(data, target_data, look_back=50):
    """Transform multivariate time-series into (N, T, F) input and (N, T_out) target tensors."""
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i : i + look_back])
        y.append(target_data[i + look_back])
    return np.array(X), np.array(y)

def prepare_pipeline(csv_path, look_back=50, train_split=0.8):
    """Execute multivariate feature scaling and temporal partitioning."""
    df = pd.read_csv(csv_path)

    input_scaler = MinMaxScaler(feature_range=(0, 1))
    target_scaler = MinMaxScaler(feature_range=(0, 1))

    input_scaled = input_scaler.fit_transform(df[INPUT_FEATURES].values)
    target_scaled = target_scaler.fit_transform(df[TARGET_FEATURES].values)

    X, y = create_sequences(input_scaled, target_scaled, look_back=look_back)

    split_idx = int(len(X) * train_split)
    return (
        (X[:split_idx], y[:split_idx]),
        (X[split_idx:], y[split_idx:]),
        input_scaler,
        target_scaler
    )

if __name__ == "__main__":
    (X_train, y_train), (X_test, y_test), _, _ = prepare_pipeline("clean_workload_data.csv")
    print(f"Input tensor:  {X_train.shape}  -> (samples, time_steps, features)")
    print(f"Target tensor: {y_train.shape}  -> (samples, outputs)")
    print(f"Test input:    {X_test.shape}")
    print(f"Test target:   {y_test.shape}")
