import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import pickle

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CSV_PATH = "clean_workload_data.csv"
LOOK_BACK = 50
TRAIN_SPLIT = 0.8
EPOCHS = 50
BATCH_SIZE = 64
LSTM_UNITS = 128
DROPOUT_RATE = 0.2
MODEL_SAVE_PATH = "workload_lstm_model.keras"
INPUT_SCALER_PATH = "input_scaler.pkl"
TARGET_SCALER_PATH = "target_scaler.pkl"

INPUT_FEATURES = ['avg_cpu', 'avg_memory', 'max_cpu', 'assigned_memory', 'cpi']
TARGET_FEATURES = ['avg_cpu', 'avg_memory']
NUM_INPUTS = len(INPUT_FEATURES)
NUM_OUTPUTS = len(TARGET_FEATURES)

# ---------------------------------------------------------------------------
# Preprocessing (self-contained for Kaggle/Colab execution)
# ---------------------------------------------------------------------------
def create_sequences(data, target_data, look_back):
    """Transform multivariate time-series into (N, T, F) tensors."""
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i : i + look_back])
        y.append(target_data[i + look_back])
    return np.array(X), np.array(y)

def prepare_data(csv_path, look_back, train_split):
    """Load, scale, vectorize, and partition the multivariate dataset."""
    df = pd.read_csv(csv_path)

    input_scaler = MinMaxScaler(feature_range=(0, 1))
    target_scaler = MinMaxScaler(feature_range=(0, 1))

    input_scaled = input_scaler.fit_transform(df[INPUT_FEATURES].values)
    target_scaled = target_scaler.fit_transform(df[TARGET_FEATURES].values)

    X, y = create_sequences(input_scaled, target_scaled, look_back)

    split_idx = int(len(X) * train_split)
    return (X[:split_idx], y[:split_idx]), (X[split_idx:], y[split_idx:]), input_scaler, target_scaler

# ---------------------------------------------------------------------------
# GPU Detection
# ---------------------------------------------------------------------------
gpus = tf.config.list_physical_devices('GPU')
print(f"Detected GPUs: {len(gpus)}")
for gpu in gpus:
    print(f"  - {gpu.name}")

# ---------------------------------------------------------------------------
# Data Preparation
# ---------------------------------------------------------------------------
(X_train, y_train), (X_test, y_test), input_scaler, target_scaler = prepare_data(CSV_PATH, LOOK_BACK, TRAIN_SPLIT)
print(f"Training input:  {X_train.shape}  -> (samples, time_steps, {NUM_INPUTS} features)")
print(f"Training target: {y_train.shape}  -> (samples, {NUM_OUTPUTS} outputs)")
print(f"Testing input:   {X_test.shape}")
print(f"Testing target:  {y_test.shape}")

# Persist both scalers for inference
with open(INPUT_SCALER_PATH, 'wb') as f:
    pickle.dump(input_scaler, f)
with open(TARGET_SCALER_PATH, 'wb') as f:
    pickle.dump(target_scaler, f)

# ---------------------------------------------------------------------------
# Model Construction under MirroredStrategy
# ---------------------------------------------------------------------------
strategy = tf.distribute.MirroredStrategy()
print(f"Replicas in sync: {strategy.num_replicas_in_sync}")

with strategy.scope():
    model = Sequential([
        Input(shape=(LOOK_BACK, NUM_INPUTS)),
        LSTM(LSTM_UNITS, return_sequences=True),
        Dropout(DROPOUT_RATE),
        LSTM(LSTM_UNITS // 2),
        Dropout(DROPOUT_RATE),
        Dense(NUM_OUTPUTS)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')

model.summary()

# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        MODEL_SAVE_PATH,
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
]

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE * strategy.num_replicas_in_sync,
    validation_data=(X_test, y_test),
    callbacks=callbacks,
    verbose=1
)

# ---------------------------------------------------------------------------
# Final Evaluation
# ---------------------------------------------------------------------------
test_loss = model.evaluate(X_test, y_test, verbose=0)
print(f"Test MSE: {test_loss:.6f}")
print(f"Model saved to: {MODEL_SAVE_PATH}")
print(f"Input scaler saved to: {INPUT_SCALER_PATH}")
print(f"Target scaler saved to: {TARGET_SCALER_PATH}")
