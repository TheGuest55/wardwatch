"""TEMPORARY Step 3 verify: threshold + first real F1. Delete after the check.

Trains on benign-only (short), scores the FULL validation split (benign + attacks), picks the
F1-maximising threshold on validation, and reports binary metrics via the shared metric set. This is
the first evidence the autoencoder actually separates attacks from normal.
"""
import numpy as np

from src.models.data import load_dataset, Dataset
from src.models import metrics as mmetrics
from src.models.autoencoder import (AutoEncoder, make_cfg, set_determinism, train,
                                     anomaly_scores, pick_threshold, predict)

df, fl = load_dataset()
ds = Dataset(df, fl)

Xtr_normal = ds.normal_only("train")      # benign only, for training
Xva_normal = ds.normal_only("val")        # benign only, for early-stopping signal
Xva_all, yva = ds.X("val"), ds.y("val")   # FULL val (benign + attacks), for the threshold

set_determinism(0, 1)
cfg = make_cfg(epochs=10, patience=4, seed=0)
model = AutoEncoder(n_in=Xtr_normal.shape[1])
train(model, Xtr_normal, Xva_normal, cfg)

# Score the full validation split and separate benign vs attack error distributions.
sva = anomaly_scores(model, Xva_all)
benign_err = sva[yva == 0]
attack_err = sva[yva == 1]
print(f"val benign error : mean {benign_err.mean():.4f}  median {np.median(benign_err):.4f}")
print(f"val attack error : mean {attack_err.mean():.4f}  median {np.median(attack_err):.4f}")
print(f"separation (attack mean > benign mean): {attack_err.mean() > benign_err.mean()}")

thr = pick_threshold(yva, sva)
pred = predict(sva, thr)
m = mmetrics.binary_metrics(yva, y_score=sva, y_pred=pred)
print(f"\nchosen threshold: {thr:.4f}")
print(f"VAL  F1 {m['f1']:.4f} | precision {m['precision']:.4f} | recall {m['recall']:.4f} "
      f"| PR-AUC {m.get('pr_auc', float('nan')):.4f} | ROC-AUC {m.get('roc_auc', float('nan')):.4f}")
print(f"val rows {m['n']} | attack rows {m['n_positive']}")
