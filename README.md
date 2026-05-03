# Predictive Cluster Manager

## Project Overview
This repository implements a deep learning pipeline for workload forecasting using the Google Cluster Trace dataset (v3). The system transforms raw cluster telemetry into 3D temporal tensors and trains an LSTM network to predict future CPU utilization. Training is designed for multi-GPU execution on Kaggle's Dual T4 environment using TensorFlow's MirroredStrategy.

## Dataset
This project utilizes the Google Cluster-Usage Traces v3 (2019), which tracks resource utilization across approximately 12,000 machines.

*   **Kaggle Source:** [Google Cluster Data - Kaggle](https://www.kaggle.com/datasets/google/cluster-usage)
*   **Official Documentation:** [Google Cluster Data GitHub](https://github.com/google/cluster-data/blob/master/ClusterData2019.md)

## Prerequisites
*   Python 3.8+
*   NumPy
*   Pandas
*   Scikit-learn
*   TensorFlow 2.x

## Pipeline Architecture

### 1. Data Ingestion (`data_preprocessing.py`)
Parses compressed GZIP JSON traces via stream processing. Extracts CPU and memory telemetry, aggregates multiple instance rows into one machine workload row per timestamp, and exports a chronologically sorted CSV.

### 2. Feature Engineering
Builds spike-aware temporal features such as short/long rolling maxima, rolling volatility, deltas, absolute deltas, EWM momentum gaps, and CPU spike pressure. Scalers are fitted on the training period only, then sliding-window temporal sequences of shape `(N, T, F)` are generated with chronological train-validation-test partitioning.

### 3. Model Training (`train_model.py`)
Builds and trains a Conv1D + stacked LSTM architecture under `tf.distribute.MirroredStrategy` for multi-GPU acceleration. The model predicts both next values and next deltas, then validation-selects the best direct-vs-delta blend. It also validation-selects how much to trust the neural forecast versus a persistence baseline, which helps prevent stable memory usage from being shifted upward. CPU and memory use output-specific robust loss weights so CPU spike learning does not push the memory baseline upward. Validation-set calibration and spike guarding are applied only when they improve validation score. Includes EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, L2 regularization, and spatial dropout. Exports the trained model, fitted scalers, prediction calibration, spike guard configuration, and selected postprocessing configuration.

## Model Architecture
```
Input(60, F) -> LayerNorm -> Causal Conv1D(64) -> Causal Conv1D(64) -> LSTM(192) -> LSTM(96) -> Dense(96) -> Dense(4)
```
*   **Optimizer:** Adam (adaptive learning rate)
*   **Loss:** output-specific peak-weighted Huber loss over CPU, memory, CPU delta, and memory delta
*   **Spike Handling:** timestamp aggregation, delta targets, validation-selected per-output blending, validation-selected persistence ensemble, nonnegative output clipping, optional validation-set bias calibration, and optional recent-window spike guarding
*   **Overfit Control:** L2 regularization, spatial dropout, EarlyStopping, ReduceLROnPlateau, and validation-selected postprocessing
*   **Metrics:** CPU usage accuracy, memory usage accuracy, MAE, RMSE, and CPU spike precision/recall/F1
*   **Regularization:** Dropout + EarlyStopping + learning-rate reduction

## Execution

### Local Preprocessing
```bash
python data_preprocessing.py
```

### Kaggle Training
1.  Create a Kaggle Notebook with **GPU T4 x2** accelerator.
2.  Upload `clean_workload_data.csv` as a dataset.
3.  Paste and execute the contents of `train_model.py`.
4.  Download `workload_lstm_model.keras` and `scaler.pkl` from the output.

## Repository Hygiene
A `.gitignore` is configured to exclude data artifacts, model checkpoints, virtual environments, and credential files from version control.
