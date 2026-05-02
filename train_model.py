import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import pickle
import os

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
SCALER_SAVE_PATH = "scaler.pkl"

# ---------------------------------------------------------------------------
# Preprocessing (self-contained for Kaggle execution)
# ---------------------------------------------------------------------------
def create_sequences(data, look_back):
    """Transform univariate time-series into (N, T, F) tensors."""
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i : i + look_back, 0])
        y.append(data[i + look_back, 0])
    X_arr, y_arr = np.array(X), np.array(y)
    return X_arr.reshape(X_arr.shape[0], X_arr.shape[1], 1), y_arr

def prepare_data(csv_path, look_back, train_split):
    """Load, scale, vectorize, and partition the dataset."""
    df = pd.read_csv(csv_path)
    cpu_data = df['cpu_usage'].values.reshape(-1, 1)

    scaler = MinMaxScaler(feature_range=(0, 1))
    cpu_scaled = scaler.fit_transform(cpu_data)

    X, y = create_sequences(cpu_scaled, look_back)

    split_idx = int(len(X) * train_split)
    return (X[:split_idx], y[:split_idx]), (X[split_idx:], y[split_idx:]), scaler

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
(X_train, y_train), (X_test, y_test), scaler = prepare_data(CSV_PATH, LOOK_BACK, TRAIN_SPLIT)
print(f"Training tensor: {X_train.shape}")
print(f"Testing tensor:  {X_test.shape}")

# Persist the scaler for inverse transform during inference
with open(SCALER_SAVE_PATH, 'wb') as f:
    pickle.dump(scaler, f)

# ---------------------------------------------------------------------------
# Model Construction under MirroredStrategy
# ---------------------------------------------------------------------------
strategy = tf.distribute.MirroredStrategy()
print(f"Replicas in sync: {strategy.num_replicas_in_sync}")

with strategy.scope():
    model = Sequential([
        Input(shape=(LOOK_BACK, 1)),
        LSTM(LSTM_UNITS, return_sequences=True),
        Dropout(DROPOUT_RATE),
        LSTM(LSTM_UNITS // 2),
        Dropout(DROPOUT_RATE),
        Dense(1)
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
print(f"Scaler saved to: {SCALER_SAVE_PATH}")
