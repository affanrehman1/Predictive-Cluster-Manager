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
Parses compressed GZIP JSON traces via stream processing. Extracts CPU and memory telemetry and exports a chronologically sorted CSV.

### 2. Feature Engineering (`lstm_preprocessing.py`)
Applies Min-Max normalization, generates sliding-window temporal sequences of shape `(N, T, F)`, and performs chronological train-test partitioning.

### 3. Model Training (`train_model.py`)
Builds and trains a stacked LSTM architecture under `tf.distribute.MirroredStrategy` for multi-GPU acceleration. Includes EarlyStopping and ModelCheckpoint callbacks. Exports the trained model as `.keras` and the fitted scaler as `.pkl`.

## Model Architecture
```
Input(10, 1) -> LSTM(64) -> Dropout(0.2) -> LSTM(32) -> Dropout(0.2) -> Dense(1)
```
*   **Optimizer:** Adam (adaptive learning rate)
*   **Loss:** Mean Squared Error
*   **Regularization:** Dropout (20%) + EarlyStopping (patience=5)

## Execution

### Local Preprocessing
```bash
python data_preprocessing.py
python lstm_preprocessing.py
```

### Kaggle Training
1.  Create a Kaggle Notebook with **GPU T4 x2** accelerator.
2.  Upload `clean_workload_data.csv` as a dataset.
3.  Paste and execute the contents of `train_model.py`.
4.  Download `workload_lstm_model.keras` and `scaler.pkl` from the output.

## Repository Hygiene
A `.gitignore` is configured to exclude data artifacts, model checkpoints, virtual environments, and credential files from version control.
