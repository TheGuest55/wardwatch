"""LSTM sequence classifier for WardWatch (Phase 3, Pair C).

Multiclass training (benign + attack classes, softmax), binary scoring: score = 1 - P(benign). One
example = a host's last k windows (see sequence.py); the label is the label of the last window.

Fit discipline: early stopping and the decision threshold use VALIDATION only; the scaler is the
train-only scaler of the shared harness; test is touched only by the caller for the final evaluation.
Determinism: set_determinism() seeds Python/NumPy/torch, turns on deterministic algorithms and fixes
the torch thread count; DataLoader uses num_workers=0 and a seeded torch.Generator for the shuffle,
and the training noise has its own seeded generator, so the same seed + config + thread count
reproduces the same metrics bit for bit.
"""
import copy
import pickle
import random
import time

import numpy as np
import torch
from torch import nn

DEFAULT_CFG = {
    "hidden": 64, "layers": 1, "dropout": 0.25, "lr": 1e-3, "weight_decay": 1e-4,
    "clip": 1.0, "noise_sigma": 0.05, "batch_size": 256, "epochs": 30, "patience": 5,
    "seed": 0, "threads": 1, "use_presence": True,
}


def make_cfg(**overrides):
    cfg = dict(DEFAULT_CFG)
    unknown = set(overrides) - set(cfg) - {"n_feat"}
    if unknown:
        raise ValueError(f"unknown cfg keys: {sorted(unknown)}")
    cfg.update(overrides)
    return cfg


class LSTMClassifier(nn.Module):
    """LSTM over the k-window sequence + dropout + a linear head on the last hidden state."""

    def __init__(self, n_in, hidden, layers=1, dropout=0.25, n_classes=9):
        super().__init__()
        self.lstm = nn.LSTM(n_in, hidden, num_layers=layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, n_classes)

    def forward(self, x):                      # x [B, k, n_in] -> logits [B, n_classes]
        out, _ = self.lstm(x)
        return self.head(self.drop(out[:, -1, :]))


def set_determinism(seed, threads):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(int(threads))


def class_weights(y_train, n_classes):
    """w_benign = 1, w_c = sqrt(n_benign / n_c), from the training targets actually used (in
    Experiment B: the training set without the withheld class). A class with no examples gets 1."""
    y = np.asarray(y_train)
    n = np.bincount(y[y >= 0], minlength=n_classes).astype(float)
    w = np.ones(n_classes, dtype=np.float64)
    for c in range(1, n_classes):
        if n[c] > 0 and n[0] > 0:
            w[c] = np.sqrt(n[0] / n[c])
    return w


def present_mask(x, n_feat, use_presence):
    """[B, k] bool: which windows are real (not gap fill / padding). With the is_present channel it is
    that channel; without it, fill windows are exact all-zero rows, so non-zero rows are present."""
    if use_presence:
        return x[:, :, n_feat] > 0.5
    return (x[:, :, :n_feat] != 0).any(dim=-1)


def add_noise(x, sigma, n_feat, use_presence, generator):
    """Gaussian noise on the n_feat FEATURE channels of PRESENT windows only; never on is_present and
    never on filled/padded windows. Training only."""
    if sigma <= 0:
        return x
    noise = torch.randn(x.shape[0], x.shape[1], n_feat, generator=generator) * sigma
    m = present_mask(x, n_feat, use_presence).unsqueeze(-1).to(x.dtype)
    out = x.clone()
    out[:, :, :n_feat] = x[:, :, :n_feat] + noise * m
    return out


def predict_proba(model, X, batch_size=2048):
    """Softmax class probabilities [n, C] (float32), eval mode, no grad."""
    model.eval()
    X = torch.as_tensor(X)
    outs = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            outs.append(torch.softmax(model(X[i:i + batch_size]), dim=1))
    if not outs:
        return np.zeros((0, model.head.out_features), dtype=np.float32)
    return torch.cat(outs).numpy()


def binary_score(probs):
    """Attack score = 1 - P(benign); class 0 is benign."""
    return 1.0 - np.asarray(probs)[:, 0]


def _pr_auc(y_bin, score):
    from sklearn.metrics import average_precision_score
    y_bin = np.asarray(y_bin)
    if y_bin.sum() == 0:
        return float("nan")
    return float(average_precision_score(y_bin, score))


def train(model, data, cfg):
    """Train with Adam + weighted cross-entropy. `data` = {X_train, y_train, X_val, y_val} where y_* are
    class indices (benign = 0) and X_* are float32 [n, k, d] (d = n_feat + 1 with is_present, or n_feat
    without; cfg['n_feat'] and cfg['use_presence'] say which).

    Early stopping on validation binary PR-AUC (patience cfg['patience'], at most cfg['epochs']); the
    best epoch's weights are restored. Returns {history, best_epoch, best_val_pr_auc, epochs_run,
    epoch_seconds, class_weights}; history holds per-epoch train/val loss and train/val PR-AUC."""
    n_feat = int(cfg["n_feat"])
    use_presence = bool(cfg["use_presence"])
    Xtr = torch.as_tensor(data["X_train"])
    ytr = torch.as_tensor(np.asarray(data["y_train"]), dtype=torch.long)
    Xva = torch.as_tensor(data["X_val"])
    yva = np.asarray(data["y_val"])
    n_classes = model.head.out_features

    w = class_weights(data["y_train"], n_classes)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32))
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    shuffle_g = torch.Generator().manual_seed(int(cfg["seed"]))
    noise_g = torch.Generator().manual_seed(int(cfg["seed"]) + 1)
    ds = torch.utils.data.TensorDataset(Xtr, ytr)
    loader = torch.utils.data.DataLoader(ds, batch_size=int(cfg["batch_size"]), shuffle=True,
                                         generator=shuffle_g, num_workers=0)
    ytr_bin = (np.asarray(data["y_train"]) > 0).astype(int)
    yva_bin = (yva > 0).astype(int)

    history, best_state, best_ap, best_epoch, bad = [], None, -np.inf, 0, 0
    epoch_seconds = []
    for epoch in range(1, int(cfg["epochs"]) + 1):
        t0 = time.perf_counter()
        model.train()
        tot, cnt = 0.0, 0
        for xb, yb in loader:
            xb = add_noise(xb, cfg["noise_sigma"], n_feat, use_presence, noise_g)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg["clip"])
            opt.step()
            tot += float(loss.item()) * len(xb)
            cnt += len(xb)
        p_tr = predict_proba(model, Xtr)
        p_va = predict_proba(model, Xva)
        val_loss = float(loss_fn(torch.log(torch.as_tensor(p_va).clamp_min(1e-12)),
                                 torch.as_tensor(yva, dtype=torch.long)).item()) if len(yva) else float("nan")
        ap_va = _pr_auc(yva_bin, binary_score(p_va))
        rec = {"epoch": epoch, "train_loss": tot / max(cnt, 1), "val_loss": val_loss,
               "train_pr_auc": _pr_auc(ytr_bin, binary_score(p_tr)), "val_pr_auc": ap_va}
        history.append(rec)
        epoch_seconds.append(time.perf_counter() - t0)
        score = -np.inf if np.isnan(ap_va) else ap_va
        if score > best_ap:
            best_ap, best_epoch, bad = score, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= int(cfg["patience"]):
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return {"history": history, "best_epoch": best_epoch,
            "best_val_pr_auc": None if best_epoch == 0 else best_ap,
            "epochs_run": len(history), "epoch_seconds": epoch_seconds,
            "class_weights": [float(x) for x in w]}


def pick_threshold(y_val, s_val):
    """Threshold maximising VALIDATION binary F1 over the unique scores (predict attack if
    score >= t). Ties go to the highest threshold. Raises if validation has no positives."""
    y = np.asarray(y_val).astype(int)
    s = np.asarray(s_val, dtype=float)
    P = int(y.sum())
    if P == 0:
        raise ValueError("validation has no attack rows: cannot pick a threshold")
    order = np.argsort(-s, kind="mergesort")
    ys, sc = y[order], s[order]
    tp = np.cumsum(ys == 1)
    fp = np.cumsum(ys == 0)
    last = np.append(np.flatnonzero(np.diff(sc) != 0), len(sc) - 1)   # end of each tied block
    tp, fp, thr = tp[last], fp[last], sc[last]
    f1 = 2 * tp / (2 * tp + fp + (P - tp))
    i = int(np.argmax(f1))                  # first max in descending-threshold order = highest t
    return float(thr[i])


def macro_f1(y_true_class, probs):
    """Multiclass macro-F1: argmax over the class probabilities (benign + 8 attack classes), computed
    over the labels present in y_true or the predictions, zero_division = 0. TEST ONLY, and for
    classifiers in Experiment A only (a withheld class cannot be named). Shared definition: the
    XGBoost pair should call this same helper so the numbers are comparable."""
    from sklearn.metrics import f1_score
    pred = np.argmax(np.asarray(probs), axis=1)
    return float(f1_score(np.asarray(y_true_class), pred, average="macro", zero_division=0))


def save_artefact(path, model, scaler, k, feature_names, class_order, threshold, cfg):
    """One file with everything the live pipeline needs: weights, the train-only scaler (pickled),
    k, feature names, class order, threshold, and the training config."""
    import os
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    margs = {"n_in": model.lstm.input_size, "hidden": model.lstm.hidden_size,
             "layers": model.lstm.num_layers, "dropout": model.drop.p,
             "n_classes": model.head.out_features}
    torch.save({"state_dict": model.state_dict(), "model_args": margs,
                "scaler_pickle": pickle.dumps(scaler), "k": int(k),
                "feature_names": list(feature_names), "class_order": list(class_order),
                "threshold": float(threshold), "cfg": dict(cfg)}, path)
    return str(path)


def load_artefact(path):
    """Inverse of save_artefact; returns a dict with `model` (eval mode), `scaler`, `k`,
    `feature_names`, `class_order`, `threshold`, `cfg`. Own-written files only (pickle inside)."""
    blob = torch.load(path, weights_only=False)
    model = LSTMClassifier(**blob["model_args"])
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return {"model": model, "scaler": pickle.loads(blob["scaler_pickle"]), "k": blob["k"],
            "feature_names": blob["feature_names"], "class_order": blob["class_order"],
            "threshold": blob["threshold"], "cfg": blob["cfg"]}
