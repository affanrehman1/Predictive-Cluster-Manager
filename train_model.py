import pickle

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Conv1D, Dense, Dropout, Input, LSTM, LayerNormalization, SpatialDropout1D
from tensorflow.keras.models import Sequential

# System configuration parameters.
CSV_PATH = "clean_workload_data.csv"
LOOK_BACK = 60
TRAIN_SPLIT = 0.8
VALIDATION_SPLIT = 0.1
EPOCHS = 100
BATCH_SIZE = 64
LSTM_UNITS = 192
DROPOUT_RATE = 0.15
L2_REGULARIZATION = 1e-5

CPU_SPIKE_QUANTILE = 0.90
CPU_JUMP_QUANTILE = 0.95
MEM_SPIKE_QUANTILE = 0.90
CPU_SPIKE_WEIGHT = 2.0
CPU_JUMP_WEIGHT = 1.5
MEM_SPIKE_WEIGHT = 0.5
CPU_LAST_VALUE_BLEND = 0.60
MEMORY_LAST_VALUE_BLEND = 0.20
BLEND_CANDIDATES = (0.0, 0.2, 0.4, 0.6, 0.8)
MIN_CPU_SPIKE_RECALL = 0.68
MIN_CPU_SPIKE_F1 = 0.68

MODEL_SAVE_PATH = "workload_lstm_model.keras"
INPUT_SCALER_PATH = "input_scaler.pkl"
TARGET_SCALER_PATH = "target_scaler.pkl"
CALIBRATION_PATH = "prediction_calibration.pkl"
SPIKE_GUARD_PATH = "spike_guard_config.pkl"
POSTPROCESS_CONFIG_PATH = "prediction_postprocess_config.pkl"

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
NUM_INPUTS = len(INPUT_FEATURES)
NUM_OUTPUTS = len(TARGET_FEATURES)

tf.keras.utils.set_random_seed(42)

# Data preprocessing pipeline.
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
    """Compute and append temporal context features for spike recognition."""
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

def create_window_contexts(df, look_back):
    """Extract contextual summaries for temporal windows."""
    contexts = []
    for i in range(len(df) - look_back):
        window = df.iloc[i : i + look_back]
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

def create_sequences(data, target_data, raw_values, contexts, look_back):
    """Convert multivariate time-series into aligned tensor sequences."""
    X, y, last_values = [], [], []
    for i in range(len(data) - look_back):
        X.append(data[i : i + look_back])
        y.append(target_data[i + look_back])
        last_values.append(raw_values[i + look_back - 1])
    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y, dtype=np.float32),
        np.asarray(last_values, dtype=np.float32),
        contexts,
    )

def make_spike_sample_weights(df, look_back, train_count):
    """Compute custom sample weights to emphasize rare high-load spikes."""
    aligned = df.iloc[look_back:].reset_index(drop=True)
    train_aligned = aligned.iloc[:train_count]

    cpu_spike_threshold = train_aligned['avg_cpu'].quantile(CPU_SPIKE_QUANTILE)
    cpu_jump_threshold = train_aligned['cpu_delta'].clip(lower=0).quantile(CPU_JUMP_QUANTILE)
    mem_spike_threshold = train_aligned['avg_memory'].quantile(MEM_SPIKE_QUANTILE)

    weights = np.ones(len(aligned), dtype=np.float32)
    weights += (aligned['avg_cpu'].to_numpy() >= cpu_spike_threshold) * CPU_SPIKE_WEIGHT
    weights += (aligned['cpu_delta'].to_numpy() >= cpu_jump_threshold) * CPU_JUMP_WEIGHT
    weights += (aligned['avg_memory'].to_numpy() >= mem_spike_threshold) * MEM_SPIKE_WEIGHT
    weights = weights / weights[:train_count].mean()

    thresholds = {
        'cpu_spike_threshold': float(cpu_spike_threshold),
        'cpu_jump_threshold': float(cpu_jump_threshold),
        'mem_spike_threshold': float(mem_spike_threshold),
    }
    return weights.astype(np.float32), thresholds

def prepare_data(csv_path, look_back, train_split, validation_split):
    """Execute complete data preparation and temporal partitioning pipeline."""
    raw_df = pd.read_csv(csv_path)
    df = add_spike_features(raw_df)
    print(f"Raw rows: {len(raw_df):,} | aggregated time steps: {len(df):,}")

    raw_inputs = df[INPUT_FEATURES].astype(np.float32).to_numpy()
    raw_targets = df[TARGET_FEATURES].astype(np.float32).to_numpy()
    raw_values = df[VALUE_TARGETS].astype(np.float32).to_numpy()

    sample_count = len(df) - look_back
    train_count = int(sample_count * train_split)
    if train_count <= 0:
        raise ValueError("Not enough rows to create training sequences.")

    scaler_fit_end = train_count + look_back
    input_scaler = StandardScaler()
    target_scaler = StandardScaler()
    input_scaler.fit(raw_inputs[:scaler_fit_end])
    target_scaler.fit(raw_targets[:scaler_fit_end])

    input_scaled = input_scaler.transform(raw_inputs)
    target_scaled = target_scaler.transform(raw_targets)
    contexts = create_window_contexts(df, look_back)
    X, y, last_values, contexts = create_sequences(input_scaled, target_scaled, raw_values, contexts, look_back)
    sample_weights, thresholds = make_spike_sample_weights(df, look_back, train_count)

    X_train_all, y_train_all = X[:train_count], y[:train_count]
    X_test, y_test = X[train_count:], y[train_count:]
    train_weights_all = sample_weights[:train_count]
    test_weights = sample_weights[train_count:]
    last_train_all = last_values[:train_count]
    last_test = last_values[train_count:]
    context_train_all = contexts[:train_count]
    context_test = contexts[train_count:]

    val_count = max(1, int(len(X_train_all) * validation_split))
    train_end = len(X_train_all) - val_count

    splits = {
        'train': (X_train_all[:train_end], y_train_all[:train_end], train_weights_all[:train_end], last_train_all[:train_end], context_train_all[:train_end]),
        'val': (X_train_all[train_end:], y_train_all[train_end:], train_weights_all[train_end:], last_train_all[train_end:], context_train_all[train_end:]),
        'test': (X_test, y_test, test_weights, last_test, context_test),
    }
    return splits, input_scaler, target_scaler, thresholds

def output_specific_peak_loss(y_true, y_pred):
    """Compute custom loss optimizing for specific output targets."""
    error = y_true - y_pred
    abs_error = tf.abs(error)
    huber = tf.where(abs_error <= 1.0, 0.5 * tf.square(error), abs_error - 0.5)

    cpu_level = tf.nn.relu(y_true[:, 0])
    mem_level = tf.nn.relu(y_true[:, 1])
    cpu_change = tf.abs(y_true[:, 2])
    mem_change = tf.abs(y_true[:, 3])

    weights = tf.stack([
        1.00 + 0.70 * cpu_level + 0.30 * cpu_change,
        1.35 + 0.04 * mem_level + 0.04 * mem_change,
        0.65 + 0.35 * cpu_change,
        0.25 + 0.05 * mem_change,
    ], axis=1)
    weights = tf.clip_by_value(weights, 0.25, 4.0)
    return tf.reduce_mean(huber * weights, axis=-1)

def inverse_targets(target_scaler, y_scaled):
    return target_scaler.inverse_transform(y_scaled)

def blend_value_and_delta_predictions(target_scaler, y_pred_scaled, last_values, blend=None):
    """Merge base predictions with residual forecasts."""
    pred_all = inverse_targets(target_scaler, y_pred_scaled)
    direct_values = pred_all[:, :2]
    delta_values = pred_all[:, 2:4]
    residual_values = last_values + delta_values
    if blend is None:
        blend = np.array([CPU_LAST_VALUE_BLEND, MEMORY_LAST_VALUE_BLEND], dtype=np.float32)
    else:
        blend = np.asarray(blend, dtype=np.float32)
    blended = (1.0 - blend) * direct_values + blend * residual_values
    return np.maximum(blended, 0.0)

def fit_prediction_calibration(y_true, y_pred):
    """Estimate linear calibration parameters to correct output bias."""
    scale = np.ones(2, dtype=np.float32)
    offset = np.zeros(2, dtype=np.float32)

    for output_idx in range(2):
        pred_col = y_pred[:, output_idx]
        true_col = y_true[:, output_idx]
        design = np.column_stack([pred_col, np.ones_like(pred_col)])
        coef, _, _, _ = np.linalg.lstsq(design, true_col, rcond=None)
        scale[output_idx] = np.clip(coef[0], 0.25, 2.5)
        offset[output_idx] = coef[1]

    return {'scale': scale, 'offset': offset}

def apply_prediction_calibration(y_pred, calibration):
    calibrated = y_pred * calibration['scale'] + calibration['offset']
    return np.maximum(calibrated, 0.0)

def smape(y_true, y_pred):
    denom = np.abs(y_true) + np.abs(y_pred) + 1e-8
    return np.mean(2.0 * np.abs(y_true - y_pred) / denom, axis=0)

def validation_score(y_true, y_pred, cpu_spike_threshold):
    """Calculate validation score balancing overall accuracy and spike detection."""
    output_smape = smape(y_true, y_pred)
    actual_spikes = y_true[:, 0] >= cpu_spike_threshold
    predicted_spikes = y_pred[:, 0] >= cpu_spike_threshold
    true_positive = np.sum(actual_spikes & predicted_spikes)
    precision = true_positive / max(np.sum(predicted_spikes), 1)
    recall = true_positive / max(np.sum(actual_spikes), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    recall_penalty = max(0.0, MIN_CPU_SPIKE_RECALL - recall) * 0.35
    f1_penalty = max(0.0, MIN_CPU_SPIKE_F1 - f1) * 0.25
    return float(output_smape[0] + output_smape[1] + recall_penalty + f1_penalty)

def select_postprocessing(target_scaler, y_pred_scaled, last_values, y_true, contexts, thresholds):
    """Select optimal postprocessing configurations using validation performance."""
    best = {
        'score': float('inf'),
        'blend': np.array([CPU_LAST_VALUE_BLEND, MEMORY_LAST_VALUE_BLEND], dtype=np.float32),
        'ensemble_alpha': np.ones(2, dtype=np.float32),
        'baseline_kind': 'last',
        'calibration': {'scale': np.ones(2, dtype=np.float32), 'offset': np.zeros(2, dtype=np.float32)},
        'use_calibration': False,
        'use_spike_guard': False,
        'spike_guard_config': None,
    }

    baseline_last = get_persistence_baseline(contexts, 'last')
    baseline_mean = get_persistence_baseline(contexts, 'mean')
    for baseline_kind, baseline_pred in (('last', baseline_last), ('mean', baseline_mean)):
        baseline_score = validation_score(y_true, baseline_pred, thresholds['cpu_spike_threshold'])
        if baseline_score < best['score']:
            best.update({
                'score': baseline_score,
                'blend': np.array([0.0, 0.0], dtype=np.float32),
                'ensemble_alpha': np.array([0.0, 0.0], dtype=np.float32),
                'baseline_kind': baseline_kind,
                'calibration': {'scale': np.ones(2, dtype=np.float32), 'offset': np.zeros(2, dtype=np.float32)},
                'use_calibration': False,
                'use_spike_guard': False,
                'spike_guard_config': None,
            })

    for cpu_blend in BLEND_CANDIDATES:
        for mem_blend in BLEND_CANDIDATES:
            blend = np.array([cpu_blend, mem_blend], dtype=np.float32)
            pred = blend_value_and_delta_predictions(target_scaler, y_pred_scaled, last_values, blend)

            candidates = [(pred, False, {'scale': np.ones(2, dtype=np.float32), 'offset': np.zeros(2, dtype=np.float32)})]
            calibration = fit_prediction_calibration(y_true, pred)
            candidates.append((apply_prediction_calibration(pred, calibration), True, calibration))

            for candidate_pred, use_calibration, candidate_calibration in candidates:
                for baseline_kind in ('last', 'mean'):
                    ensembled_pred, ensemble_alpha = select_persistence_ensemble(
                        y_true,
                        candidate_pred,
                        contexts,
                        thresholds['cpu_spike_threshold'],
                        baseline_kind,
                    )
                    score = validation_score(y_true, ensembled_pred, thresholds['cpu_spike_threshold'])
                    if score < best['score']:
                        best.update({
                            'score': score,
                            'blend': blend,
                            'ensemble_alpha': ensemble_alpha,
                            'baseline_kind': baseline_kind,
                            'calibration': candidate_calibration,
                            'use_calibration': use_calibration,
                            'use_spike_guard': False,
                            'spike_guard_config': None,
                        })

    best_pred = blend_value_and_delta_predictions(target_scaler, y_pred_scaled, last_values, best['blend'])
    if best['use_calibration']:
        best_pred = apply_prediction_calibration(best_pred, best['calibration'])
    best_pred = apply_persistence_ensemble(best_pred, contexts, best['ensemble_alpha'], best['baseline_kind'])

    spike_guard_config = fit_spike_guard_config(y_true, best_pred, contexts, thresholds)
    guarded_pred = apply_spike_guard(best_pred, contexts, thresholds, spike_guard_config)
    guarded_score = validation_score(y_true, guarded_pred, thresholds['cpu_spike_threshold'])
    if guarded_score < best['score']:
        best.update({
            'score': guarded_score,
            'use_spike_guard': True,
            'spike_guard_config': spike_guard_config,
        })

    return best

def apply_postprocessing(target_scaler, y_pred_scaled, last_values, contexts, thresholds, postprocess_config):
    y_pred = blend_value_and_delta_predictions(target_scaler, y_pred_scaled, last_values, postprocess_config['blend'])
    if postprocess_config.get('use_calibration', False):
        y_pred = apply_prediction_calibration(y_pred, postprocess_config['calibration'])
    y_pred = apply_persistence_ensemble(
        y_pred,
        contexts,
        postprocess_config.get('ensemble_alpha', np.ones(2, dtype=np.float32)),
        postprocess_config.get('baseline_kind', 'last'),
    )
    if postprocess_config.get('use_spike_guard', False):
        y_pred = apply_spike_guard(y_pred, contexts, thresholds, postprocess_config['spike_guard_config'])
    return y_pred

def get_persistence_baseline(contexts, baseline_kind):
    """Extract baseline prediction values from local context."""
    if baseline_kind == 'mean':
        return contexts[:, [2, 3]]
    return contexts[:, [0, 1]]

def apply_persistence_ensemble(y_pred, contexts, alpha, baseline_kind):
    """Combine model output with persistence baseline."""
    baseline = get_persistence_baseline(contexts, baseline_kind)
    alpha = np.asarray(alpha, dtype=np.float32)
    return np.maximum(alpha * y_pred + (1.0 - alpha) * baseline, 0.0)

def select_persistence_ensemble(y_true, y_pred, contexts, cpu_spike_threshold, baseline_kind):
    """Determine optimal ensemble weights for persistence baseline integration."""
    baseline = get_persistence_baseline(contexts, baseline_kind)
    alpha_grid = np.linspace(0.0, 1.0, 11, dtype=np.float32)
    selected_alpha = np.ones(2, dtype=np.float32)
    selected = y_pred.copy()

    for output_idx in range(2):
        best_alpha = 1.0
        best_metric = float('inf')
        for alpha in alpha_grid:
            candidate = alpha * y_pred[:, output_idx] + (1.0 - alpha) * baseline[:, output_idx]
            metric = smape(y_true[:, [output_idx]], candidate[:, None])[0]
            if output_idx == 0:
                actual_spikes = y_true[:, 0] >= cpu_spike_threshold
                predicted_spikes = candidate >= cpu_spike_threshold
                true_positive = np.sum(actual_spikes & predicted_spikes)
                precision = true_positive / max(np.sum(predicted_spikes), 1)
                recall = true_positive / max(np.sum(actual_spikes), 1)
                f1 = 2 * precision * recall / max(precision + recall, 1e-8)
                metric += max(0.0, MIN_CPU_SPIKE_RECALL - recall) * 0.20
                metric += max(0.0, MIN_CPU_SPIKE_F1 - f1) * 0.15
            if metric < best_metric:
                best_metric = metric
                best_alpha = alpha

        selected_alpha[output_idx] = best_alpha
        selected[:, output_idx] = best_alpha * y_pred[:, output_idx] + (1.0 - best_alpha) * baseline[:, output_idx]

    return np.maximum(selected, 0.0), selected_alpha

def fit_spike_guard_config(y_true, y_pred, contexts, thresholds):
    """Identify optimal threshold constraints to mitigate false positives."""
    candidates = [
        {'cpu_jump': 3.0, 'cpu_max': 1.45, 'cpu_pressure': 0.75, 'mem_jump': 1.8, 'mem_max': 1.20},
        {'cpu_jump': 3.5, 'cpu_max': 1.55, 'cpu_pressure': 0.85, 'mem_jump': 2.1, 'mem_max': 1.25},
        {'cpu_jump': 4.0, 'cpu_max': 1.65, 'cpu_pressure': 0.95, 'mem_jump': 2.4, 'mem_max': 1.30},
        {'cpu_jump': 5.0, 'cpu_max': 1.85, 'cpu_pressure': 1.10, 'mem_jump': 2.8, 'mem_max': 1.40},
    ]

    best_config = candidates[0]
    best_score = float('inf')
    for config in candidates:
        guarded = apply_spike_guard(y_pred, contexts, thresholds, config)
        cpu_mae = np.mean(np.abs(y_true[:, 0] - guarded[:, 0]))
        mem_mae = np.mean(np.abs(y_true[:, 1] - guarded[:, 1]))

        actual_spikes = y_true[:, 0] >= thresholds['cpu_spike_threshold']
        predicted_spikes = guarded[:, 0] >= thresholds['cpu_spike_threshold']
        recall = np.sum(actual_spikes & predicted_spikes) / max(np.sum(actual_spikes), 1)
        recall_penalty = max(0.0, 0.75 - recall) * cpu_mae
        score = cpu_mae + 2.0 * mem_mae + recall_penalty
        if score < best_score:
            best_score = score
            best_config = config

    return best_config

def apply_spike_guard(y_pred, contexts, thresholds, config):
    """Apply contextual upper bounds to model predictions."""
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
    cpu_threshold_floor = np.full_like(cpu_recent_max, thresholds['cpu_spike_threshold'] * 1.15)
    cpu_ceiling = np.maximum.reduce([
        cpu_recent_max * config['cpu_max'],
        max_cpu_recent_max * config['cpu_pressure'],
        cpu_threshold_floor,
        cpu_jump_ceiling,
    ])
    cpu_ceiling = np.maximum(cpu_ceiling, thresholds['cpu_spike_threshold'] * 0.95)

    mem_ceiling = np.maximum(
        mem_recent_max * config['mem_max'],
        last_mem + config['mem_jump'] * mem_abs_delta_p90 + 1.2 * mem_recent_std + 0.00008,
    )
    # Allow memory expansion only when recent baseline activity is high.
    mem_ceiling = np.where(mem_recent_max > thresholds['mem_spike_threshold'], mem_ceiling * 1.35, mem_ceiling)

    guarded[:, 0] = np.minimum(guarded[:, 0], cpu_ceiling)
    guarded[:, 1] = np.minimum(guarded[:, 1], mem_ceiling)
    return np.maximum(guarded, 0.0)

def percentage_accuracy(y_true, y_pred):
    """Calculate sMAPE-based percentage accuracy."""
    denom = np.abs(y_true) + np.abs(y_pred) + 1e-8
    smape = np.mean(2.0 * np.abs(y_true - y_pred) / denom) * 100.0
    return max(0.0, 100.0 - smape)

def regression_report(y_true, y_pred, cpu_spike_threshold):
    """Output comprehensive regression performance metrics."""
    metrics = compute_metrics(y_true, y_pred, cpu_spike_threshold)

    print("\nPrediction Accuracy by Output:")
    print(f"CPU Usage Accuracy:    {metrics['cpu_accuracy']:.2f}% | MAE: {metrics['cpu_mae']:.6f} | RMSE: {metrics['cpu_rmse']:.6f}")
    print(f"Memory Usage Accuracy: {metrics['mem_accuracy']:.2f}% | MAE: {metrics['mem_mae']:.6f} | RMSE: {metrics['mem_rmse']:.6f}")
    if metrics['has_cpu_spikes']:
        print(f"CPU spike MAE: {metrics['cpu_spike_mae']:.6f} | spike bias: {metrics['cpu_spike_bias']:.6f}")
    print(f"CPU spike precision: {metrics['cpu_spike_precision']:.3f} | recall: {metrics['cpu_spike_recall']:.3f} | F1: {metrics['cpu_spike_f1']:.3f}")

def compute_metrics(y_true, y_pred, cpu_spike_threshold):
    """Calculate core performance metrics for a specific dataset split."""
    cpu_error = y_true[:, 0] - y_pred[:, 0]
    mem_error = y_true[:, 1] - y_pred[:, 1]
    cpu_mae = np.mean(np.abs(cpu_error))
    mem_mae = np.mean(np.abs(mem_error))
    cpu_rmse = np.sqrt(np.mean(np.square(cpu_error)))
    mem_rmse = np.sqrt(np.mean(np.square(mem_error)))
    cpu_accuracy = percentage_accuracy(y_true[:, 0], y_pred[:, 0])
    mem_accuracy = percentage_accuracy(y_true[:, 1], y_pred[:, 1])

    actual_spikes = y_true[:, 0] >= cpu_spike_threshold
    predicted_spikes = y_pred[:, 0] >= cpu_spike_threshold
    true_positive = np.sum(actual_spikes & predicted_spikes)
    precision = true_positive / max(np.sum(predicted_spikes), 1)
    recall = true_positive / max(np.sum(actual_spikes), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    metrics = {
        'cpu_accuracy': cpu_accuracy,
        'mem_accuracy': mem_accuracy,
        'cpu_mae': cpu_mae,
        'mem_mae': mem_mae,
        'cpu_rmse': cpu_rmse,
        'mem_rmse': mem_rmse,
        'cpu_spike_precision': precision,
        'cpu_spike_recall': recall,
        'cpu_spike_f1': f1,
        'has_cpu_spikes': bool(np.any(actual_spikes)),
        'cpu_spike_mae': np.nan,
        'cpu_spike_bias': np.nan,
    }
    if np.any(actual_spikes):
        metrics['cpu_spike_mae'] = np.mean(np.abs(cpu_error[actual_spikes]))
        metrics['cpu_spike_bias'] = np.mean(y_pred[actual_spikes, 0] - y_true[actual_spikes, 0])
    return metrics

def print_metric_row(label, metrics):
    print(
        f"{label:<18} "
        f"CPU acc={metrics['cpu_accuracy']:6.2f}% MAE={metrics['cpu_mae']:.6f} | "
        f"Mem acc={metrics['mem_accuracy']:6.2f}% MAE={metrics['mem_mae']:.6f} | "
        f"CPU spike F1={metrics['cpu_spike_f1']:.3f} recall={metrics['cpu_spike_recall']:.3f}"
    )

def evaluate_predictions(model, X, y_scaled, last_values, contexts, target_scaler, thresholds, postprocess_config):
    pred_scaled = model.predict(X, verbose=0)
    y_pred = apply_postprocessing(target_scaler, pred_scaled, last_values, contexts, thresholds, postprocess_config)
    y_true = inverse_targets(target_scaler, y_scaled)[:, :2]
    return y_true, y_pred

def print_generalization_diagnostics(model, splits, target_scaler, thresholds, postprocess_config):
    """Output generalization diagnostics and baseline comparisons."""
    split_metrics = {}

    print("\nGeneralization Diagnostics:")
    for split_name in ('train', 'val', 'test'):
        X, y_scaled, _, last_values, contexts = splits[split_name]
        y_true, y_pred = evaluate_predictions(model, X, y_scaled, last_values, contexts, target_scaler, thresholds, postprocess_config)
        metrics = compute_metrics(y_true, y_pred, thresholds['cpu_spike_threshold'])
        split_metrics[split_name] = metrics
        print_metric_row(f"Model {split_name}", metrics)

    train_cpu_gap = split_metrics['train']['cpu_accuracy'] - split_metrics['test']['cpu_accuracy']
    train_mem_gap = split_metrics['train']['mem_accuracy'] - split_metrics['test']['mem_accuracy']
    print(f"Train-test accuracy gap: CPU={train_cpu_gap:.2f} points | Memory={train_mem_gap:.2f} points")
    if train_cpu_gap > 8 or train_mem_gap > 8:
        print("Overfitting warning: training accuracy is much higher than test accuracy.")
    else:
        print("Overfitting check: train/test gap is not large for these accuracy metrics.")

    print("\nNo-learning Baseline Comparison on Test:")
    X_test, y_test_scaled, _, _, test_contexts = splits['test']
    y_test_true = inverse_targets(target_scaler, y_test_scaled)[:, :2]
    last_metrics = compute_metrics(y_test_true, get_persistence_baseline(test_contexts, 'last'), thresholds['cpu_spike_threshold'])
    mean_metrics = compute_metrics(y_test_true, get_persistence_baseline(test_contexts, 'mean'), thresholds['cpu_spike_threshold'])
    print_metric_row("Last-value", last_metrics)
    print_metric_row("Recent-mean", mean_metrics)
    print_metric_row("Model test", split_metrics['test'])

    cpu_gain = split_metrics['test']['cpu_accuracy'] - max(last_metrics['cpu_accuracy'], mean_metrics['cpu_accuracy'])
    mem_gain = split_metrics['test']['mem_accuracy'] - max(last_metrics['mem_accuracy'], mean_metrics['mem_accuracy'])
    print(f"Model gain over best baseline: CPU={cpu_gain:.2f} points | Memory={mem_gain:.2f} points")
    if cpu_gain <= 0 or mem_gain <= 0:
        print("Learning warning: the neural model is not beating the best persistence baseline for every output.")
    else:
        print("Learning check: the neural model adds value over persistence on both outputs.")

# Hardware detection.
gpus = tf.config.list_physical_devices('GPU')
print(f"Detected GPUs: {len(gpus)}")
for gpu in gpus:
    print(f"  - {gpu.name}")

# Data preparation.
splits, input_scaler, target_scaler, thresholds = prepare_data(
    CSV_PATH,
    LOOK_BACK,
    TRAIN_SPLIT,
    VALIDATION_SPLIT,
)
X_train, y_train, train_weights, last_train_values, train_contexts = splits['train']
X_val, y_val, _, last_val_values, val_contexts = splits['val']
X_test, y_test, test_weights, last_test_values, test_contexts = splits['test']

print(f"Training input:    {X_train.shape} -> (samples, time_steps, {NUM_INPUTS} features)")
print(f"Validation input:  {X_val.shape}")
print(f"Testing input:     {X_test.shape}")
print(f"CPU spike threshold: {thresholds['cpu_spike_threshold']:.6f}")
print(f"CPU jump threshold:  {thresholds['cpu_jump_threshold']:.6f}")
print(f"Reference spike weight range: {train_weights.min():.2f} - {train_weights.max():.2f}")

with open(INPUT_SCALER_PATH, 'wb') as f:
    pickle.dump(input_scaler, f)
with open(TARGET_SCALER_PATH, 'wb') as f:
    pickle.dump(target_scaler, f)

# Model architecture definition.
strategy = tf.distribute.MirroredStrategy()
print(f"Replicas in sync: {strategy.num_replicas_in_sync}")

with strategy.scope():
    model = Sequential([
        Input(shape=(LOOK_BACK, NUM_INPUTS)),
        LayerNormalization(),
        Conv1D(64, kernel_size=3, padding='causal', activation='relu', kernel_regularizer=regularizers.l2(L2_REGULARIZATION)),
        Conv1D(64, kernel_size=3, padding='causal', activation='relu', kernel_regularizer=regularizers.l2(L2_REGULARIZATION)),
        SpatialDropout1D(0.10),
        LSTM(LSTM_UNITS, return_sequences=True, kernel_regularizer=regularizers.l2(L2_REGULARIZATION)),
        Dropout(DROPOUT_RATE),
        LSTM(LSTM_UNITS // 2, kernel_regularizer=regularizers.l2(L2_REGULARIZATION)),
        Dense(128, activation='relu', kernel_regularizer=regularizers.l2(L2_REGULARIZATION)),
        Dense(64, activation='relu', kernel_regularizer=regularizers.l2(L2_REGULARIZATION)),
        Dropout(DROPOUT_RATE),
        Dense(NUM_OUTPUTS),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005, clipnorm=1.0),
        loss=output_specific_peak_loss,
        metrics=[tf.keras.metrics.MeanAbsoluteError(name='mae')],
    )

model.summary()

# Training callback configuration.
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=12,
        restore_best_weights=True,
        verbose=1,
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-5,
        verbose=1,
    ),
    ModelCheckpoint(
        MODEL_SAVE_PATH,
        monitor='val_loss',
        save_best_only=True,
        verbose=1,
    ),
]

# Model training execution.
history = model.fit(
    X_train,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE * strategy.num_replicas_in_sync,
    validation_data=(X_val, y_val),
    callbacks=callbacks,
    verbose=1,
)

pd.DataFrame(history.history).to_csv("training_history.csv", index=False)

# Final model evaluation.
test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Loss: {test_loss:.6f} | Test MAE: {test_mae:.6f}")

val_pred_scaled = model.predict(X_val)
val_true = inverse_targets(target_scaler, y_val)[:, :2]
postprocess_config = select_postprocessing(target_scaler, val_pred_scaled, last_val_values, val_true, val_contexts, thresholds)
calibration = postprocess_config['calibration']
spike_guard_config = postprocess_config['spike_guard_config']
with open(CALIBRATION_PATH, 'wb') as f:
    pickle.dump(calibration, f)
with open(SPIKE_GUARD_PATH, 'wb') as f:
    pickle.dump(spike_guard_config, f)
with open(POSTPROCESS_CONFIG_PATH, 'wb') as f:
    pickle.dump(postprocess_config, f)
print(
    "Selected postprocessing: "
    f"blend CPU={postprocess_config['blend'][0]:.2f}, Memory={postprocess_config['blend'][1]:.2f}; "
    f"ensemble alpha CPU={postprocess_config['ensemble_alpha'][0]:.2f}, Memory={postprocess_config['ensemble_alpha'][1]:.2f}; "
    f"baseline={postprocess_config['baseline_kind']}; "
    f"use_calibration={postprocess_config['use_calibration']}; "
    f"use_spike_guard={postprocess_config['use_spike_guard']}; "
    f"validation_score={postprocess_config['score']:.4f}"
)
print(
    "Selected calibration: "
    f"CPU scale={calibration['scale'][0]:.3f}, offset={calibration['offset'][0]:.6f}; "
    f"Memory scale={calibration['scale'][1]:.3f}, offset={calibration['offset'][1]:.6f}"
)
if spike_guard_config is not None:
    print(f"Spike guard config: {spike_guard_config}")

y_pred_scaled = model.predict(X_test)
y_pred = apply_postprocessing(target_scaler, y_pred_scaled, last_test_values, test_contexts, thresholds, postprocess_config)
y_true = inverse_targets(target_scaler, y_test)[:, :2]
regression_report(y_true, y_pred, thresholds['cpu_spike_threshold'])
print_generalization_diagnostics(model, splits, target_scaler, thresholds, postprocess_config)

print(f"\nModel saved to: {MODEL_SAVE_PATH}")
print(f"Input scaler saved to: {INPUT_SCALER_PATH}")
print(f"Target scaler saved to: {TARGET_SCALER_PATH}")
print(f"Prediction calibration saved to: {CALIBRATION_PATH}")
print(f"Postprocess config saved to: {POSTPROCESS_CONFIG_PATH}")
