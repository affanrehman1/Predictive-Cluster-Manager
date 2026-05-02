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
SCALER_PATH = "scaler.pkl"
LOOK_BACK = 50  # Matches Phase 2 training

# ---------------------------------------------------------------------------
# Data & Model Loading
# ---------------------------------------------------------------------------
df = pd.read_csv(CSV_PATH)
cpu_data = df['cpu_usage'].values.reshape(-1, 1)

with open(SCALER_PATH, 'rb') as f:
    scaler = pickle.load(f)

cpu_scaled = scaler.transform(cpu_data)

# Extract only the TEST segment (last 20%)
split_idx = int(len(cpu_scaled) * 0.8)
test_data = cpu_scaled[split_idx:]

X_test, y_test = [], []
for i in range(len(test_data) - LOOK_BACK):
    X_test.append(test_data[i : i + LOOK_BACK, 0])
    y_test.append(test_data[i + LOOK_BACK, 0])

X_test = np.array(X_test).reshape(-1, LOOK_BACK, 1)
y_actual = np.array(y_test).reshape(-1, 1)

# Load the model
print("Loading model and performing inference...")
model = tf.keras.models.load_model(MODEL_PATH)

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
y_pred_scaled = model.predict(X_test)

# Inverse Transform to get real CPU values
y_pred = scaler.inverse_transform(y_pred_scaled)
y_actual_real = scaler.inverse_transform(y_actual)

# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
plt.figure(figsize=(15, 6))
# We plot a slice of 500 points to see details
plt.plot(y_actual_real[:500], label='Actual CPU Usage', color='blue', alpha=0.7)
plt.plot(y_pred[:500], label='Predicted CPU Usage', color='red', linestyle='--', alpha=0.9)
plt.title('Phase 2 Forecast: Actual vs Predicted (High-Capacity Model)')
plt.xlabel('Time Steps')
plt.ylabel('CPU Usage')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Print metrics
mae = np.mean(np.abs(y_actual_real - y_pred))
print(f"\nInference Metrics (Phase 2):")
print(f"Mean Absolute Error: {mae:.6f} CPU Units")
