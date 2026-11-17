"""Shared real-data fixture: the deployed pair with a freshly refit multiclass model in a temp model dir.

The multiclass refit is the same one save_multiclass saves (and checks against the recorded macro-F1), so the tests
do not depend on models/expA/xgboost_multiclass_seed0.json having been written yet."""
import json
import shutil
import tempfile
from pathlib import Path

from src.models import save_multiclass as smc
from src.models import calibrate_explain as ce

_CACHE = {}


def fitted():
    if "fit" not in _CACHE:
        _CACHE["fit"] = smc.fit()
    return _CACHE["fit"]


def pair():
    if "pair" not in _CACHE:
        from src.realtime.pipeline import Pair
        mc, classes, ds, metrics = fitted()
        d = Path(tempfile.mkdtemp(prefix="pair_"))
        mc.save_model(str(d / smc.OUT.name))
        (d / smc.CLASSES.name).write_text(json.dumps({"classes": classes, "features": ds.feature_names}))
        shutil.copy(ce.MODEL_DIR / "isolation_forest_seed0.joblib", d / "isolation_forest_seed0.joblib")
        _CACHE["dir"] = d
        _CACHE["pair"] = Pair(model_dir=d)
    return _CACHE["pair"]
