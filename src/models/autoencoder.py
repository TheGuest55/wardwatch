"""Undercomplete autoencoder for WardWatch (Phase 3, anomaly-detection role).

Unsupervised detector: trained on BENIGN windows only (Dataset.normal_only), it learns to
reconstruct normal host-window feature vectors. At scoring time the per-row reconstruction error
(mean squared error between input and output) is the anomaly score -- a window the model rebuilds
badly is unlike anything it saw in training, so a high error flags a likely attack. The decision
threshold on that score is chosen on VALIDATION (which does contain attacks); training never sees a
label. This is the autoencoder half of the anomaly role (Isolation Forest is the baseline); the two
feed the "anomalous / normal" axis of the triage verdict table (./CLAUDE.md).

Style mirrors src/models/lstm.py: a DEFAULT_CFG dict, make_cfg(**overrides) with unknown-key
rejection, and set_determinism() seeding Python/NumPy/torch so the same seed + config reproduces the
same result. PyTorch (torch is already in the analysis image; there is no TensorFlow).
"""
import copy
import random
import time

import numpy as np
import torch
from torch import nn

# Defaults chosen on validation later (Experiment A does a small grid); these are the starting point.
DEFAULT_CFG = {
    "hidden": (32, 16),     # encoder layer widths before the bottleneck (decoder mirrors them)
    "bottleneck": 8,        # latent width -- the "squeeze" that forces structure, not memorisation
    "lr": 1e-3,
    "weight_decay": 1e-5,
    "batch_size": 256,
    "epochs": 50,
    "patience": 6,          # early stopping on validation reconstruction error
    "dropout": 0.0,
    "seed": 0,
    "threads": 1,
}


def make_cfg(**overrides):
    """A config dict from DEFAULT_CFG + overrides. Rejects unknown keys (typo guard), like lstm.py.
    `n_feat` is allowed as a passthrough (the caller records the input width there)."""
    cfg = dict(DEFAULT_CFG)
    unknown = set(overrides) - set(cfg) - {"n_feat"}
    if unknown:
        raise ValueError(f"unknown cfg keys: {sorted(unknown)}")
    cfg.update(overrides)
    return cfg


def set_determinism(seed, threads):
    """Seed Python/NumPy/torch and fix the thread count so a seed + config reproduces bit for bit
    (same discipline as lstm.set_determinism)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(int(threads))


class AutoEncoder(nn.Module):
    """Symmetric undercomplete autoencoder: n_in -> hidden... -> bottleneck -> ...hidden -> n_in.

    ReLU between hidden layers; the output layer is linear (the inputs are standardised features, so
    reconstructions should be able to take any real value). Optional dropout in the encoder.
    """

    def __init__(self, n_in, hidden=(32, 16), bottleneck=8, dropout=0.0):
        super().__init__()
        self.n_in = int(n_in)
        self.hidden = tuple(int(h) for h in hidden)
        self.bottleneck = int(bottleneck)

        enc, prev = [], self.n_in
        for h in self.hidden:
            enc.append(nn.Linear(prev, h))
            enc.append(nn.ReLU())
            if dropout > 0:
                enc.append(nn.Dropout(dropout))
            prev = h
        enc.append(nn.Linear(prev, self.bottleneck))     # linear bottleneck
        self.encoder = nn.Sequential(*enc)

        dec, prev = [], self.bottleneck
        for h in reversed(self.hidden):
            dec.append(nn.Linear(prev, h))
            dec.append(nn.ReLU())
            prev = h
        dec.append(nn.Linear(prev, self.n_in))           # linear output
        self.decoder = nn.Sequential(*dec)

    def forward(self, x):                                 # x [B, n_in] -> reconstruction [B, n_in]
        return self.decoder(self.encoder(x))


def reconstruction_error(model, X, batch_size=4096):
    """Per-row anomaly score = mean squared error between each input row and its reconstruction.

    Returns a float64 numpy vector of length len(X) (one score per window). Eval mode, no grad, done
    in batches so a large matrix does not blow up memory. Higher = more anomalous.
    """
    model.eval()
    X = torch.as_tensor(np.asarray(X), dtype=torch.float32)
    out = np.empty(len(X), dtype=np.float64)
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = X[i:i + batch_size]
            recon = model(xb)
            err = torch.mean((recon - xb) ** 2, dim=1)    # MSE per row
            out[i:i + len(xb)] = err.numpy()
    return out


def _mean_recon_error(model, X, batch_size=4096):
    """Scalar mean reconstruction error over all rows (the validation early-stopping signal)."""
    if len(X) == 0:
        return float("nan")
    return float(reconstruction_error(model, X, batch_size).mean())


def anomaly_scores(model, X, batch_size=4096):
    """The anomaly score per window = reconstruction error. Alias for intent: downstream code asks
    for "anomaly scores", the model produces them as reconstruction error. Higher = more anomalous."""
    return reconstruction_error(model, X, batch_size)


def pick_threshold(y_val, scores_val):
    """Error threshold maximising VALIDATION binary F1 (predict attack if score >= t). Ties go to the
    highest threshold. Raises if validation has no attack rows. Same definition and tie-handling as
    lstm.pick_threshold, so autoencoder and LSTM thresholds are chosen on identical terms.

    The autoencoder is unsupervised in TRAINING, but the operating threshold is a deployment choice:
    we set it where benign and attack reconstruction errors separate best on validation (which does
    contain attacks) — never on test."""
    y = np.asarray(y_val).astype(int)
    s = np.asarray(scores_val, dtype=float)
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


def predict(scores, threshold):
    """Binary prediction: 1 (attack) where score >= threshold, else 0."""
    return (np.asarray(scores, dtype=float) >= float(threshold)).astype(int)


def train(model, X_train_normal, X_val_normal, cfg):
    """Train the autoencoder to reconstruct BENIGN traffic (unsupervised: no labels).

    `X_train_normal` and `X_val_normal` are benign-only, train-only-scaled matrices (from
    Dataset.normal_only on the respective split). MSE loss, Adam. Early stopping watches the mean
    reconstruction error on the benign VALIDATION rows (patience cfg['patience'], at most
    cfg['epochs']); the best epoch's weights are restored. A seeded torch.Generator drives the
    shuffle so the run is reproducible (lstm.py discipline). Returns {history, best_epoch,
    best_val_error, epochs_run, epoch_seconds}.

    Validation uses benign rows only on purpose: the training objective is "reconstruct normal well",
    so the stopping signal is "reconstruct unseen normal well" -- attacks never enter training or
    model selection, only the later threshold step (which does see val attacks).
    """
    Xtr = torch.as_tensor(np.asarray(X_train_normal), dtype=torch.float32)
    Xva = np.asarray(X_val_normal)
    loss_fn = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    shuffle_g = torch.Generator().manual_seed(int(cfg["seed"]))
    ds = torch.utils.data.TensorDataset(Xtr)
    loader = torch.utils.data.DataLoader(ds, batch_size=int(cfg["batch_size"]), shuffle=True,
                                         generator=shuffle_g, num_workers=0)

    history, best_state, best_err, best_epoch, bad = [], None, np.inf, 0, 0
    epoch_seconds = []
    for epoch in range(1, int(cfg["epochs"]) + 1):
        t0 = time.perf_counter()
        model.train()
        tot, cnt = 0.0, 0
        for (xb,) in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), xb)          # reconstruct the input
            loss.backward()
            opt.step()
            tot += float(loss.item()) * len(xb)
            cnt += len(xb)
        train_err = tot / max(cnt, 1)
        val_err = _mean_recon_error(model, Xva)
        history.append({"epoch": epoch, "train_mse": train_err, "val_mse": val_err})
        epoch_seconds.append(time.perf_counter() - t0)
        if val_err < best_err - 1e-9:
            best_err, best_epoch, bad = val_err, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= int(cfg["patience"]):
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return {"history": history, "best_epoch": best_epoch,
            "best_val_error": None if best_epoch == 0 else float(best_err),
            "epochs_run": len(history), "epoch_seconds": epoch_seconds}
