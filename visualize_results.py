import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

# System configuration parameters.
CSV_PATH = "clean_workload_data.csv"
MODEL_PATH = "workload_lstm_model.keras"
INPUT_SCALER_PATH = "input_scaler.pkl"
TARGET_SCALER_PATH = "target_scaler.pkl"
CALIBRATION_PATH = "prediction_calibration.pkl"
SPIKE_GUARD_PATH = "spike_guard_config.pkl"
POSTPROCESS_CONFIG_PATH = "prediction_postprocess_config.pkl"
LOOK_BACK = 60
TRAIN_SPLIT = 0.8
CPU_LAST_VALUE_BLEND = 0.60
MEMORY_LAST_VALUE_BLEND = 0.20

BASE_FEATURES = ['avg_cpu', 'avg_memory', 'max_cpu', 'assigned_memory', 'cpi']
ENGINEERED_FEATURES = [
    'cpu_delta', 'mem_delta', 'cpu_abs_delta', 'mem_abs_delta',
    'cpu_spike_pressure', 'time_delta',
    'cpu_roll_mean_3', 'cpu_roll_std_3', 'cpu_roll_max_3',
    'mem_roll_mean_3', 'mem_roll_std_3', 'mem_roll_max_3',
    'cpu_roll_mean_5', 'cpu_roll_std_5', 'cpu_roll_max_5',
    'mem_roll_mean_5', 'mem_roll_std_5', 'mem_roll_max_5',
    'cpu_roll_mean_10', 'cpu_roll_std_10', 'cpu_roll_max_10',
    'mem_roll_mean_10', 'mem_roll_std_10', 'mem_roll_max_10',
    'cpu_roll_mean_30', 'cpu_roll_std_30', 'cpu_roll_max_30',
    'mem_roll_mean_30', 'mem_roll_std_30', 'mem_roll_max_30',
    'cpu_ewm_gap', 'mem_ewm_gap',
]
INPUT_FEATURES = BASE_FEATURES + ENGINEERED_FEATURES
VALUE_TARGETS = ['avg_cpu', 'avg_memory']
DELTA_TARGETS = ['cpu_delta', 'mem_delta']
TARGET_FEATURES = VALUE_TARGETS + DELTA_TARGETS

def aggregate_workload_by_time(df):
    """Aggregate machine workload data by timestamp."""
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
    """Compute temporal features prior to scaling."""
    df = aggregate_workload_by_time(df)

    df['cpu_delta'] = df['avg_cpu'].diff().fillna(0)
    df['mem_delta'] = df['avg_memory'].diff().fillna(0)
    df['cpu_abs_delta'] = df['cpu_delta'].abs()
    df['mem_abs_delta'] = df['mem_delta'].abs()
    df['cpu_spike_pressure'] = (df['max_cpu'] - df['avg_cpu']).clip(lower=0)
    df['time_delta'] = df['time'].diff().fillna(0)

    for window in (3, 5, 10, 30):
        df[f'cpu_roll_mean_{window}'] = df['avg_cpu'].rolling(window=window, min_periods=1).mean()
        df[f'cpu_roll_std_{window}'] = df['avg_cpu'].rolling(window=window, min_periods=2).std().fillna(0)
        df[f'cpu_roll_max_{window}'] = df['avg_cpu'].rolling(window=window, min_periods=1).max()
        df[f'mem_roll_mean_{window}'] = df['avg_memory'].rolling(window=window, min_periods=1).mean()
        df[f'mem_roll_std_{window}'] = df['avg_memory'].rolling(window=window, min_periods=2).std().fillna(0)
        df[f'mem_roll_max_{window}'] = df['avg_memory'].rolling(window=window, min_periods=1).max()

    cpu_ewm_fast = df['avg_cpu'].ewm(span=4, adjust=False).mean()
    cpu_ewm_slow = df['avg_cpu'].ewm(span=24, adjust=False).mean()
    mem_ewm_fast = df['avg_memory'].ewm(span=4, adjust=False).mean()
    mem_ewm_slow = df['avg_memory'].ewm(span=24, adjust=False).mean()
    df['cpu_ewm_gap'] = cpu_ewm_fast - cpu_ewm_slow
    df['mem_ewm_gap'] = mem_ewm_fast - mem_ewm_slow

    return df.replace([np.inf, -np.inf], 0).fillna(0)

def create_window_contexts(df):
    contexts = []
    for i in range(len(df) - LOOK_BACK):
        window = df.iloc[i : i + LOOK_BACK]
        contexts.append({
            'last_cpu': window['avg_cpu'].iloc[-1],
            'last_mem': window['avg_memory'].iloc[-1],
            'cpu_recent_mean': window['avg_cpu'].tail(5).mean(),
            'mem_recent_mean': window['avg_memory'].tail(5).mean(),
            'cpu_recent_max': window['avg_cpu'].max(),
            'cpu_recent_std': window['avg_cpu'].std() if len(window) > 1 else 0.0,
            'cpu_recent_abs_delta_p90': window['cpu_abs_delta'].quantile(0.90),
            'max_cpu_recent_max': window['max_cpu'].max(),
            'cpu_pressure_recent_max': window['cpu_spike_pressure'].max(),
            'mem_recent_max': window['avg_memory'].max(),
            'mem_recent_std': window['avg_memory'].std() if len(window) > 1 else 0.0,
            'mem_recent_abs_delta_p90': window['mem_abs_delta'].quantile(0.90),
        })
    return pd.DataFrame(contexts).replace([np.inf, -np.inf], 0).fillna(0).to_numpy(dtype=np.float32)

def create_sequences(input_scaled, target_scaled, raw_values, contexts):
    X_test, y_test, last_values = [], [], []
    for i in range(len(input_scaled) - LOOK_BACK):
        X_test.append(input_scaled[i : i + LOOK_BACK])
        y_test.append(target_scaled[i + LOOK_BACK])
        last_values.append(raw_values[i + LOOK_BACK - 1])
    return (
        np.asarray(X_test, dtype=np.float32),
        np.asarray(y_test, dtype=np.float32),
        np.asarray(last_values, dtype=np.float32),
        contexts,
    )

def blend_value_and_delta_predictions(target_scaler, y_pred_scaled, last_values, blend=None):
    pred_all = target_scaler.inverse_transform(y_pred_scaled)
    direct_values = pred_all[:, :2]
    delta_values = pred_all[:, 2:4]
    residual_values = last_values + delta_values
    if blend is None:
        blend = np.array([CPU_LAST_VALUE_BLEND, MEMORY_LAST_VALUE_BLEND], dtype=np.float32)
    else:
        blend = np.asarray(blend, dtype=np.float32)
    blended = (1.0 - blend) * direct_values + blend * residual_values
    negative_count = int(np.sum(blended < 0))
    if negative_count:
        print(f"Clipped {negative_count} negative CPU/memory predictions to 0 after inverse scaling.")
    return np.maximum(blended, 0.0)

def load_prediction_calibration():
    try:
        with open(CALIBRATION_PATH, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        print("No prediction calibration found. Re-run train_model.py to generate it.")
        return {
            'scale': np.ones(2, dtype=np.float32),
            'offset': np.zeros(2, dtype=np.float32),
        }

def load_spike_guard_config():
    try:
        with open(SPIKE_GUARD_PATH, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        print("No spike guard config found. Re-run train_model.py to generate it.")
        return {'cpu_jump': 4.0, 'cpu_max': 1.65, 'cpu_pressure': 0.95, 'mem_jump': 2.4, 'mem_max': 1.30}

def load_postprocess_config():
    try:
        with open(POSTPROCESS_CONFIG_PATH, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        print("No selected postprocess config found. Re-run train_model.py to generate it.")
        return {
            'blend': np.array([CPU_LAST_VALUE_BLEND, MEMORY_LAST_VALUE_BLEND], dtype=np.float32),
            'ensemble_alpha': np.ones(2, dtype=np.float32),
            'baseline_kind': 'last',
            'calibration': load_prediction_calibration(),
            'use_calibration': True,
            'use_spike_guard': False,
            'spike_guard_config': None,
        }

def apply_prediction_calibration(y_pred, calibration):
    calibrated = y_pred * calibration['scale'] + calibration['offset']
    return np.maximum(calibrated, 0.0)

def apply_postprocessing(target_scaler, y_pred_scaled, last_values, contexts, cpu_spike_threshold, mem_spike_threshold, config):
    y_pred = blend_value_and_delta_predictions(target_scaler, y_pred_scaled, last_values, config['blend'])
    if config.get('use_calibration', False):
        y_pred = apply_prediction_calibration(y_pred, config['calibration'])
    y_pred = apply_persistence_ensemble(
        y_pred,
        contexts,
        config.get('ensemble_alpha', np.ones(2, dtype=np.float32)),
        config.get('baseline_kind', 'last'),
    )
    if config.get('use_spike_guard', False):
        y_pred = apply_spike_guard(y_pred, contexts, cpu_spike_threshold, mem_spike_threshold, config['spike_guard_config'])
    return y_pred

def get_persistence_baseline(contexts, baseline_kind):
    if baseline_kind == 'mean':
        return contexts[:, [2, 3]]
    return contexts[:, [0, 1]]

def apply_persistence_ensemble(y_pred, contexts, alpha, baseline_kind):
    baseline = get_persistence_baseline(contexts, baseline_kind)
    alpha = np.asarray(alpha, dtype=np.float32)
    return np.maximum(alpha * y_pred + (1.0 - alpha) * baseline, 0.0)

def apply_spike_guard(y_pred, contexts, cpu_spike_threshold, mem_spike_threshold, config):
    guarded = y_pred.copy()
    (
        last_cpu,
        last_mem,
        cpu_recent_mean,
        mem_recent_mean,
        cpu_recent_max,
        cpu_recent_std,
        cpu_abs_delta_p90,
        max_cpu_recent_max,
        cpu_pressure_recent_max,
        mem_recent_max,
        mem_recent_std,
        mem_abs_delta_p90,
    ) = contexts.T

    cpu_jump_ceiling = last_cpu + config['cpu_jump'] * cpu_abs_delta_p90 + 2.0 * cpu_recent_std + 0.001
    cpu_threshold_floor = np.full_like(cpu_recent_max, cpu_spike_threshold * 1.15)
    cpu_ceiling = np.maximum.reduce([
        cpu_recent_max * config['cpu_max'],
        max_cpu_recent_max * config['cpu_pressure'],
        cpu_threshold_floor,
        cpu_jump_ceiling,
    ])
    cpu_ceiling = np.maximum(cpu_ceiling, cpu_spike_threshold * 0.95)

    mem_ceiling = np.maximum(
        mem_recent_max * config['mem_max'],
        last_mem + config['mem_jump'] * mem_abs_delta_p90 + 1.2 * mem_recent_std + 0.00008,
    )
    mem_ceiling = np.where(mem_recent_max > mem_spike_threshold, mem_ceiling * 1.35, mem_ceiling)

    cpu_clipped = int(np.sum(guarded[:, 0] > cpu_ceiling))
    mem_clipped = int(np.sum(guarded[:, 1] > mem_ceiling))
    if cpu_clipped or mem_clipped:
        print(f"Spike guard clipped unsupported predictions: CPU={cpu_clipped}, Memory={mem_clipped}")

    guarded[:, 0] = np.minimum(guarded[:, 0], cpu_ceiling)
    guarded[:, 1] = np.minimum(guarded[:, 1], mem_ceiling)
    return np.maximum(guarded, 0.0)

def percentage_accuracy(y_true, y_pred):
    """Calculate sMAPE-based percentage accuracy."""
    denom = np.abs(y_true) + np.abs(y_pred) + 1e-8
    smape = np.mean(2.0 * np.abs(y_true - y_pred) / denom) * 100.0
    return max(0.0, 100.0 - smape)

def print_metrics(y_actual_real, y_pred, cpu_spike_threshold):
    cpu_error = y_actual_real[:, 0] - y_pred[:, 0]
    mem_error = y_actual_real[:, 1] - y_pred[:, 1]
    cpu_mae = np.mean(np.abs(cpu_error))
    mem_mae = np.mean(np.abs(mem_error))
    cpu_rmse = np.sqrt(np.mean(np.square(cpu_error)))
    mem_rmse = np.sqrt(np.mean(np.square(mem_error)))
    cpu_accuracy = percentage_accuracy(y_actual_real[:, 0], y_pred[:, 0])
    mem_accuracy = percentage_accuracy(y_actual_real[:, 1], y_pred[:, 1])

    actual_spikes = y_actual_real[:, 0] >= cpu_spike_threshold
    predicted_spikes = y_pred[:, 0] >= cpu_spike_threshold
    true_positive = np.sum(actual_spikes & predicted_spikes)
    precision = true_positive / max(np.sum(predicted_spikes), 1)
    recall = true_positive / max(np.sum(actual_spikes), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    print("\nPrediction Accuracy by Output:")
    print(f"CPU Usage Accuracy:    {cpu_accuracy:.2f}% | MAE: {cpu_mae:.6f} | RMSE: {cpu_rmse:.6f}")
    print(f"Memory Usage Accuracy: {mem_accuracy:.2f}% | MAE: {mem_mae:.6f} | RMSE: {mem_rmse:.6f}")
    if np.any(actual_spikes):
        spike_mae = np.mean(np.abs(cpu_error[actual_spikes]))
        spike_bias = np.mean(y_pred[actual_spikes, 0] - y_actual_real[actual_spikes, 0])
        print(f"CPU spike MAE: {spike_mae:.6f} | spike bias: {spike_bias:.6f}")
    print(f"CPU spike precision: {precision:.3f} | recall: {recall:.3f} | F1: {f1:.3f}")

# Data ingestion and model instantiation.
df = add_spike_features(pd.read_csv(CSV_PATH))

with open(INPUT_SCALER_PATH, 'rb') as f:
    input_scaler = pickle.load(f)
with open(TARGET_SCALER_PATH, 'rb') as f:
    target_scaler = pickle.load(f)
postprocess_config = load_postprocess_config()

input_scaled = input_scaler.transform(df[INPUT_FEATURES].values)
target_scaled = target_scaler.transform(df[TARGET_FEATURES].values)
raw_values = df[VALUE_TARGETS].astype(np.float32).to_numpy()

split_idx = int((len(input_scaled) - LOOK_BACK) * TRAIN_SPLIT)
test_input = input_scaled[split_idx:]
test_target = target_scaled[split_idx:]
test_raw_values = raw_values[split_idx:]
test_contexts = create_window_contexts(df.iloc[split_idx:].reset_index(drop=True))
X_test, y_actual_scaled, last_values, test_contexts = create_sequences(test_input, test_target, test_raw_values, test_contexts)

print("Loading model and performing inference...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
y_pred_scaled = model.predict(X_test)

y_actual_real = target_scaler.inverse_transform(y_actual_scaled)[:, :2]

cpu_spike_threshold = df.iloc[:split_idx + LOOK_BACK]['avg_cpu'].quantile(0.90)
mem_spike_threshold = df.iloc[:split_idx + LOOK_BACK]['avg_memory'].quantile(0.90)
y_pred = apply_postprocessing(
    target_scaler,
    y_pred_scaled,
    last_values,
    test_contexts,
    cpu_spike_threshold,
    mem_spike_threshold,
    postprocess_config,
)
visible = min(500, len(y_actual_real))
visible_actual_cpu = y_actual_real[:visible, 0]
visible_pred_cpu = y_pred[:visible, 0]
visible_spikes = visible_actual_cpu >= cpu_spike_threshold

# Generate performance visualization.
fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

axes[0].plot(visible_actual_cpu, label='Actual CPU', color='blue', alpha=0.7)
axes[0].plot(visible_pred_cpu, label='Predicted CPU', color='red', linestyle='--', alpha=0.9)
if np.any(visible_spikes):
    spike_x = np.where(visible_spikes)[0]
    axes[0].scatter(spike_x, visible_actual_cpu[visible_spikes], label='Actual CPU spike', color='black', s=18, zorder=3)
axes[0].axhline(cpu_spike_threshold, label='Spike threshold', color='gray', linestyle=':', alpha=0.8)
axes[0].set_title('Multivariate Forecast: Average CPU Usage')
axes[0].set_ylabel('CPU Usage')
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim(bottom=0)

axes[1].plot(y_actual_real[:visible, 1], label='Actual Memory', color='green', alpha=0.7)
axes[1].plot(y_pred[:visible, 1], label='Predicted Memory', color='orange', linestyle='--', alpha=0.9)
axes[1].set_title('Multivariate Forecast: Average Memory Usage')
axes[1].set_xlabel('Time Steps')
axes[1].set_ylabel('Memory Usage')
axes[1].legend()
axes[1].grid(True, alpha=0.3)
axes[1].set_ylim(bottom=0)

plt.tight_layout()
plt.show()

print_metrics(y_actual_real, y_pred, cpu_spike_threshold)
