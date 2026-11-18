"""Curves and confusion matrices DERIVED from one run's saved predictions. These are not logged metrics
and every page that shows them says so. Headline numbers never come from here."""
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

from .loaders import Loaded

MAX_POINTS = 400


def _thin(df):
    if len(df) <= MAX_POINTS:
        return df
    return df.iloc[np.linspace(0, len(df) - 1, MAX_POINTS).astype(int)].reset_index(drop=True)


def _check(pred):
    if pred["y"].nunique() < 2:
        return "predictions hold only one class, so the curve is undefined"
    return None


def pr(pred):
    why = _check(pred)
    if why:
        return Loaded(None, why)
    p, r, _ = precision_recall_curve(pred["y"], pred["score"])
    return Loaded(_thin(pd.DataFrame({"recall": r, "precision": p})))


def roc(pred):
    why = _check(pred)
    if why:
        return Loaded(None, why)
    fpr, tpr, _ = roc_curve(pred["y"], pred["score"])
    return Loaded(_thin(pd.DataFrame({"fpr": fpr, "tpr": tpr})))


def confusion_binary(pred):
    """Counts at the run's own threshold: rows = actual, columns = predicted."""
    m = pd.crosstab(pred["y"], pred["pred"]).reindex(index=[0, 1], columns=[0, 1], fill_value=0)
    m.index = ["benign", "attack"]
    m.columns = ["predicted benign", "predicted attack"]
    return m


def confusion_multiclass(pred):
    """argmax over the run's p_<class> columns vs the true class (benign or attack_class)."""
    pc = [c for c in pred.columns if c.startswith("p_") and pred[c].notna().any()]
    if not pc:
        return Loaded(None, "these predictions carry no per-class probabilities (p_<class> columns)")
    labels = [c[2:] for c in pc]
    guess = pd.Series(np.array(labels)[pred[pc].to_numpy(dtype=float).argmax(axis=1)], index=pred.index)
    true = pred["attack_class"].where(pred["y"] == 1, "benign").fillna("attack")
    m = pd.crosstab(true, guess).reindex(index=sorted(set(true) | set(labels)), columns=labels, fill_value=0)
    return Loaded(m)
