"""Shared, tested model harness: a split-respecting data loader (train-only fit discipline), a fixed
evaluation metric set, and an MLflow-compatible run record. The three model families (XGBoost,
autoencoder, LSTM) and the RF/IF baselines plug into this; no model training lives here."""
