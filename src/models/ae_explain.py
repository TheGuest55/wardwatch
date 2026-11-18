"""Per-alert explanation for the autoencoder: which features the model could not reconstruct.

For one RAW window row: scale it (the model's train-only scaler), reconstruct, and split the row's
anomaly score (the mean squared reconstruction error) into per-feature squared errors. Features are
ranked by their share of the row's total error. Each top feature carries its raw value, the raw value
the model expected (the reconstruction mapped back through the scaler) and its share. Explanations are
for the autoencoder only; the Isolation Forest has none (CLAUDE.md).
"""
import pickle

import numpy as np
import torch

from . import autoencoder as ae
from .lstm_explain import humanise


def load_artefact(path):
    """Inverse of train_autoencoder._save_ae: model (eval), scaler, threshold, feature names."""
    blob = torch.load(path, weights_only=False)
    model = ae.AutoEncoder(n_in=blob["model_args"]["n_in"], hidden=tuple(blob["model_args"]["hidden"]),
                           bottleneck=blob["model_args"]["bottleneck"])
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return {"model": model, "scaler": pickle.loads(blob["scaler_pickle"]),
            "threshold": float(blob["threshold"]), "feature_names": list(blob["feature_names"])}


def explain(model, scaler, raw_row, feats, top=5):
    """Returns {"score", "features": all features ranked by error share, "top": the top `top` as
    sentences}. score equals ae.anomaly_scores for the row; sum(error) / n_features == score."""
    raw = np.asarray(raw_row, dtype=float).reshape(1, -1)
    if raw.shape[1] != len(feats):
        raise ValueError(f"row has {raw.shape[1]} values, expected {len(feats)}")
    x = scaler.transform(raw)
    model.eval()
    with torch.no_grad():
        recon = model(torch.as_tensor(x, dtype=torch.float32)).numpy().astype(np.float64)
    err = ((recon - x.astype(np.float32).astype(np.float64)) ** 2)[0]
    total = float(err.sum())
    expected = scaler.inverse_transform(recon)[0]
    rows = [{"feature": f, "error": float(e), "share": float(e / total) if total > 0 else 0.0,
             "raw_value": float(r), "expected_value": float(x_)}
            for f, e, r, x_ in zip(feats, err, raw[0], expected)]
    rows.sort(key=lambda d: (-d["share"], d["feature"]))
    sentences = [f"{humanise(d['feature'])} was {d['raw_value']:.4g} but the model expected "
                 f"{d['expected_value']:.4g}: {100 * d['share']:.1f}% of the reconstruction error"
                 for d in rows[:top]]
    return {"score": float(total / len(feats)), "score_check": float(ae.anomaly_scores(model, x)[0]),
            "total_error": total, "features": rows, "top": sentences}
