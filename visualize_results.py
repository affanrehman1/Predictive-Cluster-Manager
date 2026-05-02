import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CSV_PATH = "clean_workload_data.csv"
MODEL_PATH = "workload_lstm_model.keras"
INPUT_SCALER_PATH = "input_scaler.pkl"
TARGET_SCALER_PATH = "target_scaler.pkl"
LOOK_BACK = 50

INPUT_FEATURES = ['avg_cpu', 'avg_memory', 'max_cpu', 'assigned_memory', 'cpi']
TARGET_FEATURES = ['avg_cpu', 'avg_memory']

# ---------------------------------------------------------------------------
# Data & Model Loading
# ---------------------------------------------------------------------------
df = pd.read_csv(CSV_PATH)

with open(INPUT_SCALER_PATH, 'rb') as f:
    input_scaler = pickle.load(f)
with open(TARGET_SCALER_PATH, 'rb') as f:
    target_scaler = pickle.load(f)

input_scaled = input_scaler.transform(df[INPUT_FEATURES].values)
target_scaled = target_scaler.transform(df[TARGET_FEATURES].values)

# Extract only the TEST segment (last 20%)
split_idx = int(len(input_scaled) * 0.8)
test_input = input_scaled[split_idx:]
test_target = target_scaled[split_idx:]

X_test, y_test = [], []
for i in range(len(test_input) - LOOK_BACK):
    X_test.append(test_input[i : i + LOOK_BACK])
    y_test.append(test_target[i + LOOK_BACK])

X_test = np.array(X_test)
y_actual = np.array(y_test)

# Load the model
print("Loading model and performing inference...")
model = tf.keras.models.load_model(MODEL_PATH)

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
y_pred_scaled = model.predict(X_test)

# Inverse Transform to get real values
y_pred = target_scaler.inverse_transform(y_pred_scaled)
y_actual_real = target_scaler.inverse_transform(y_actual)

# ---------------------------------------------------------------------------
# Visualization (Dual Output)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

# Plot 1: CPU Usage
axes[0].plot(y_actual_real[:500, 0], label='Actual CPU', color='blue', alpha=0.7)
axes[0].plot(y_pred[:500, 0], label='Predicted CPU', color='red', linestyle='--', alpha=0.9)
axes[0].set_title('Multivariate Forecast: Average CPU Usage')
axes[0].set_ylabel('CPU Usage')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: Memory Usage
axes[1].plot(y_actual_real[:500, 1], label='Actual Memory', color='green', alpha=0.7)
axes[1].plot(y_pred[:500, 1], label='Predicted Memory', color='orange', linestyle='--', alpha=0.9)
axes[1].set_title('Multivariate Forecast: Average Memory Usage')
axes[1].set_xlabel('Time Steps')
axes[1].set_ylabel('Memory Usage')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Print metrics
cpu_mae = np.mean(np.abs(y_actual_real[:, 0] - y_pred[:, 0]))
mem_mae = np.mean(np.abs(y_actual_real[:, 1] - y_pred[:, 1]))
print(f"\nInference Metrics (Multivariate):")
print(f"CPU Mean Absolute Error:    {cpu_mae:.6f}")
print(f"Memory Mean Absolute Error: {mem_mae:.6f}")
