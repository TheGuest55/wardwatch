# Autoencoder Anomaly Detector — WardWatch

**Author:** Tang Weng Shang, Clyde (2401044)
**Role owned:** the anomaly-detection model family — the **autoencoder** (deployed detector) and its **Isolation Forest** baseline.
**Files:** `src/models/autoencoder.py` (model, training, scoring), `src/models/train_autoencoder.py` (experiment runners + CLI), `tests/pipeline/test_autoencoder.py` (15 unit tests).

This document explains what was built, *why* each choice was made, what the results mean, and what each decision means for the WardWatch project as a whole.

---

## 1. What this component is, and why the project needs it

WardWatch compares **three ML model families** on detecting attacks in a simulated hospital-ward IoMT network: a supervised classifier (XGBoost), a sequence model (LSTM), and an **unsupervised anomaly detector** — this component.

An autoencoder is a neural network that learns to copy its input to its output through a narrow middle layer (the *bottleneck*). Trained **only on benign traffic**, it becomes good at rebuilding normal windows and bad at rebuilding anything unfamiliar. The size of its rebuilding mistake — the **reconstruction error** — is the anomaly score: low for normal, high for attacks.

**Why the project needs an unsupervised detector specifically.** XGBoost and the LSTM are *supervised* — they learn from labelled attacks, so they are strong on attack types they have seen and can miss novel ones. The autoencoder never sees a single attack label in training, so it can flag attack types that were absent from the training data. This is the project's central research question **RQ2** (how well does detection hold up on *unseen* attacks and a *different* network). Without this model, that question cannot be answered.

**What it means for the project:** the autoencoder is not a bonus model. In the forensic-triage layer it is the **"anomalous / normal" axis** of the six-verdict decision table (see `CLAUDE.md`). The "possible novel attack" and "detectors disagree → review" verdicts only exist because an unsupervised detector sits next to the supervised classifier. This component is a load-bearing half of the detection system.

---

## 2. Framework choice: PyTorch, not TensorFlow/Keras

**Decision:** build the autoencoder in PyTorch.

**Reasoning:** the original proposal mentioned Keras, but the actual codebase standardised on PyTorch — `torch==2.4.1+cpu` is in the analysis image (`services/analysis/requirements.lock`) and `src/models/lstm.py` is pure PyTorch. There is **no TensorFlow** in the image.

**Why this choice:**
- **No new dependency.** Using PyTorch means the ~4 GB analysis image and its lock file are untouched — teammates do not have to rebuild anything, and there is no merge risk from a dependency change.
- **Consistency.** The autoencoder reads as a sibling of the LSTM: same determinism approach, same early-stopping discipline, same artefact save/load pattern. A marker reading the code sees one coherent neural-net style, not two.

**What it means for the project:** the whole "three model families" story stays reproducible from a single container image, and the user manual needs no extra setup step for this model.

---

## 3. The model (`src/models/autoencoder.py`)

### 3.1 Architecture — an undercomplete autoencoder

Symmetric shape: `input (51) → 32 → 16 → bottleneck → 16 → 32 → 51`, ReLU between hidden layers, linear output.

**Why undercomplete (a narrow bottleneck):** forcing the data through a small middle layer stops the network from simply memorising its input. It must learn the *structure* of normal traffic — the typical relationships between the 51 features — so it generalises to unseen-but-normal windows instead of overfitting the training rows.

**Why a linear output layer:** the model's inputs are standardised features (zero-centred, can be negative), so the reconstruction must be able to take any real value. A linear output can; a bounded activation (e.g. sigmoid) could not.

**What it means for the project:** the bottleneck is the mechanism that makes the detector generalise — directly supporting the RQ2 goal of catching attacks the model has never trained on.

### 3.2 Anomaly score — per-row reconstruction error (MSE)

`reconstruction_error()` returns, for each window, the mean squared error between the input row and its reconstruction. Higher = more anomalous. Batched so a large matrix does not exhaust memory.

**Why MSE per row:** it is the natural, interpretable distance between "what the window was" and "what the model expected normal to look like." It also decomposes per feature (used later for explainability).

**What it means for the project:** every downstream consumer (threshold, triage) sees one clean anomaly number per host-window — the same granularity (one row per host per 5-second window) every other WardWatch model uses, so the comparison across models is apples-to-apples.

### 3.3 Determinism (`set_determinism`) and the config guard (`make_cfg`)

Seeds Python/NumPy/torch and fixes the thread count; `make_cfg` rejects unknown keys.

**Why:** `CLAUDE.md` requires every reported number to trace to an exact, reproducible run. The same seed + config must reproduce the same result bit for bit. The typo-guard stops a silently-ignored misspelled hyper-parameter (e.g. `bottlneck`) from producing a wrong-but-plausible number.

**What it means for the project:** the autoencoder's results are reproducible evidence, not one-off observations — a requirement for the report's "every number traces to a logged run" rule and for the user manual's reproducibility guarantee.

---

## 4. Training (`train()`), and the leakage discipline

The model trains with Adam + MSE loss on **benign windows only** (`Dataset.normal_only("train")`), with **early stopping on validation reconstruction error** (benign validation rows), restoring the best epoch's weights. A seeded shuffle keeps it reproducible.

**Key disciplines, and why:**
- **Train on benign only.** The whole premise: learn "normal," then flag deviations. No attack label touches training.
- **Early-stop on *benign* validation error.** The training objective is "reconstruct normal well," so the stopping signal is "reconstruct unseen normal well." Attacks never enter model selection — only the later threshold step uses them.
- **Scaler fit on train only** (handled by the shared `Dataset`). Fitting on validation/test would leak test-set statistics into training.
- **Identifiers are never inputs** (enforced by the harness). IPs, MACs, ports, client IDs, raw topics and timestamps are metadata only — a model that learned "the attacker's IP" would cheat rather than detect behaviour.

**What it means for the project:** these are the exact anti-leakage rules the whole project is built on (`CLAUDE.md`: *"suspiciously good results are bugs until proven otherwise"*). Following them means the autoencoder's numbers are trustworthy and directly comparable to the trees and the LSTM.

---

## 5. Threshold and prediction — turning a score into a decision

`pick_threshold(y_val, scores)` chooses the reconstruction-error cutoff that **maximises binary F1 on the validation set** (which does contain attacks); `predict()` flags a window as attack when its score ≥ threshold.

**Why validation, not test:** the threshold is a deployment setting — in the real world you tune your alarm sensitivity on data you have already seen. Tuning it on test would be peeking at the exam answers. Test is scored exactly once, with the frozen threshold.

**Why this matches the LSTM exactly:** `pick_threshold` uses the same definition and tie-handling (ties go to the highest threshold) as `lstm.pick_threshold`. Identical rules mean the autoencoder and LSTM thresholds are chosen on identical terms.

**What it means for the project:** the cross-model comparison is fair — differences in F1 reflect the *models*, not different threshold-picking rules.

---

## 6. The Isolation Forest baseline

**Decision:** pair the autoencoder with an Isolation Forest (`sklearn`, already in the image), exposed through the same CLI via `--model`.

**What Isolation Forest is:** a different unsupervised idea — it isolates points with random splits. Rare/odd points (attacks) are isolated in fewer splits, so they score as more anomalous. (We negate sklearn's `score_samples` so that, like the autoencoder, higher = more anomalous.)

**Why have a baseline at all:** the project's fixed selection rule picks the **better of the two** anomaly detectors to deploy. A result of "F1 0.79" is meaningless without a reference point; the baseline provides it. It also shows the autoencoder earns its place rather than being assumed.

**What it means for the project:** this completes the entire "anomaly detector" role (chosen model + baseline) as one self-contained, individually-attributable contribution, and gives the selection rule the comparison it needs.

---

## 7. The experiments (`train_autoencoder.py`)

All experiments mirror `src/models/train_tree.py` so the autoencoder plugs into the shared harness: same CLI flags (`--model`, `--experiment`, `--for-report`, `--no-mlflow`), same 5-seeds-→-mean±std protocol, same `tracking.build_run_record(role="anomaly")` run records, same output paths under `data/processed/v2/runs/`.

### Experiment A — time-based split (`--experiment A`)

Train on benign-only; select the architecture on validation (small grid, seed 0, by val F1); evaluate 5 seeds on the **untouched test split**.

**Why a time-based split (not random):** consecutive 5-second windows are near-identical. A random shuffle would put neighbouring windows in both train and test, letting the model recognise a session it has effectively already seen — classic data snooping. Splitting by time tests genuine generalisation to later traffic.

### Experiment B — leave-one-attack-out (`--experiment B`)

For each of the 8 attack classes: train benign-only (so the withheld class is absent from training anyway), withhold that class from the validation rows used to pick the threshold, and test on benign + the withheld class only.

**Why:** this is the direct test of RQ2 — can the detector catch an attack type it has *never* seen? It is the experiment where an unsupervised model is expected to beat supervised ones.

### Experiment C — cross-dataset (`--experiment C`)

Train on one domain (our testbed), test on the public CICIoMT2024 dataset and vice-versa, using only the features both datasets share (the 3 testbed-only vital-sign features are dropped).

**Status: deferred, not faked.** The CICIoMT2024 features (`data/processed/cic/`) are not in the shared v2 bundle, so Experiment C **exits with a clear "not present — deferred" message and runs nothing**, exactly as `CLAUDE.md` requires (*never fake a result*). The code is complete and will run once the CIC data is extracted (`python -m src.features.extract_cic`).

**What it means for the project:** A, B and C together are the project's full evaluation protocol. Running A and B on both detectors, and deferring C honestly, means the autoencoder half of the comparison is ready for the report with no fabricated numbers.

---

## 8. Results and what they mean

### 8.1 Experiment A (test split, 5 seeds)

| Detector | F1 | Precision | Recall | FPR | PR-AUC | ROC-AUC | Latency p95 |
|---|---|---|---|---|---|---|---|
| **Autoencoder** | **0.793 ± 0.013** | 0.826 | 0.782 | 0.69% | **0.842** | 0.966 | **0.15 ms** |
| Isolation Forest | 0.727 ± 0.006 | 0.858 | 0.636 | 0.44% | 0.773 | 0.967 | 2.17 ms |

**What it means:**
- **The autoencoder wins** on the headline F1 (0.793 vs 0.727) and PR-AUC. It is the detector the selection rule deploys.
- **Why it wins — recall.** Isolation Forest is slightly more precise but catches far fewer attacks (recall 0.64 vs 0.78). For a hospital IDS, missing attacks is the worse failure, so the autoencoder's trade-off is the right one.
- **ROC-AUC ≈ 0.97 for both** — both *rank* attacks well; the autoencoder is simply better at the operating threshold.
- **Latency 0.15 ms** sits far under the project's < 50 ms-per-window real-time budget — a ~300× margin, so the real-time requirement (**RQ3**) is comfortably met.
- **Tight ± over 5 seeds** → the result is stable, not a lucky seed.

### 8.2 Experiment B — leave-one-attack-out (test, 5 seeds)

Mean binary F1 over the 8 withheld classes:

| Detector | Mean F1 | Mean recall |
|---|---|---|
| **Autoencoder** | 0.413 ± 0.025 | **0.808** |
| Isolation Forest | 0.410 ± 0.017 | 0.644 |

Per-class recall (fraction of a never-seen attack's windows caught):

| Withheld class | Autoencoder | Isolation Forest | Note |
|---|---|---|---|
| recon_scan | 1.00 | 1.00 | loud — both ace it |
| syn_flood | 1.00 | 1.00 | loud |
| mqtt_flood | 1.00 | 1.00 | loud |
| bruteforce | 0.99 | 1.00 | loud |
| **arp_mitm** | **0.99** | 0.42 | autoencoder far better |
| **pump_cmd** | **0.70** | 0.03 | autoencoder hugely better; IF effectively blind |
| **spoof_vitals** | **0.72** | 0.48 | autoencoder better |
| eavesdrop | 0.05 | 0.22 | weak for both (honest limitation) |

**What it means:**
- **The headline F1s look tied, but recall tells the real story:** the autoencoder catches **80.8%** of unseen-attack windows vs the baseline's 64.4%. It generalises better to attacks it has never seen — the core RQ2 claim, supported by evidence.
- **The autoencoder decisively wins on the stolen-credential attacks** (`pump_cmd` 0.70 vs 0.03, `spoof_vitals` 0.72 vs 0.48, `arp_mitm` 0.99 vs 0.42). These are exactly the attacks the proposal predicted would be "the hardest," and the autoencoder is the model that actually catches them unseen.
- **`eavesdrop` is the honest weak spot for both detectors.** Passive wildcard eavesdropping (the attacker just *listens* with stolen credentials) barely changes traffic volume or timing, so there is almost nothing anomalous to reconstruct badly. This is reported, not hidden — it is a real limitation and a candidate for the triage layer or a future feature.

### 8.3 The leakage sanity check (Experiment A, seed 0)

The top features by reconstruction error are all genuine traffic-behaviour features — `tcp_synack`, `unique_contacted_ports`, `tcp_syn_rate`, `nf_new_flows`, `tcp_rst`, packet counts — with **no identifier** (no IP, MAC, port, client ID, timestamp).

**What it means:** the model detects *attack behaviour*, not *who the attacker is*. This is the direct evidence that satisfies `CLAUDE.md`'s "suspiciously good results are bugs until proven otherwise" rule, and it doubles as the autoencoder's plain-language explanation for the triage layer (which features drove the alert).

---

## 9. The tests (`tests/pipeline/test_autoencoder.py`) — 15, all passing

Mirrors `tests/pipeline/test_lstm.py` (unittest, tiny synthetic fixtures). Groups:

- **Model** — forward output shape equals input shape; the bottleneck is the narrowest layer.
- **Reconstruction error** — one score per row; non-negative and finite; batched scoring equals a single pass; the `anomaly_scores` alias equals `reconstruction_error`.
- **Training** — loss falls; same seed is **bit-for-bit** reproducible; different seeds differ; **the detection premise** (after training on normal, attack rows reconstruct worse than benign rows).
- **Threshold** — maximises validation F1; ties go to the highest; raises if validation has no positives; `predict` applies the cutoff correctly.
- **Config** — `make_cfg` rejects an unknown (typo) key; allows the `n_feat` passthrough.

**Why these tests:** they lock the component's contract so a later change that breaks shape, determinism, the threshold rule, or the core detection premise fails loudly in CI rather than silently corrupting a reported number.

**What it means for the project:** the autoencoder is covered by the shared test suite like every other model, so the team can refactor around it safely, and the tests themselves are part of the graded "well-tested, working" implementation.

---

## 10. How to reproduce

All runs use the analysis Docker image, offline, data mounted read-only (the project's standard isolation). From the repo root:

```bash
# build the analysis image once
docker build --pull --build-arg "GIT_COMMIT=$(git rev-parse HEAD)" -t wardwatch-analysis:0.2.0 services/analysis

# Experiment A
docker run --rm --network none -v "${PWD}:/work" -v "${PWD}/data/raw:/work/data/raw:ro" -w /work wardwatch-analysis:0.2.0 python -m src.models.train_autoencoder --model autoencoder --experiment A --no-mlflow
docker run --rm --network none -v "${PWD}:/work" -v "${PWD}/data/raw:/work/data/raw:ro" -w /work wardwatch-analysis:0.2.0 python -m src.models.train_autoencoder --model isolation_forest --experiment A --no-mlflow

# Experiment B (leave-one-attack-out)
docker run --rm --network none -v "${PWD}:/work" -v "${PWD}/data/raw:/work/data/raw:ro" -w /work wardwatch-analysis:0.2.0 python -m src.models.train_autoencoder --model autoencoder --experiment B --no-mlflow
docker run --rm --network none -v "${PWD}:/work" -v "${PWD}/data/raw:/work/data/raw:ro" -w /work wardwatch-analysis:0.2.0 python -m src.models.train_autoencoder --model isolation_forest --experiment B --no-mlflow

# Experiment C (defers cleanly until data/processed/cic/ exists)
docker run --rm --network none -v "${PWD}:/work" -v "${PWD}/data/raw:/work/data/raw:ro" -w /work wardwatch-analysis:0.2.0 python -m src.models.train_autoencoder --model autoencoder --experiment C --no-mlflow

# unit tests
docker run --rm --network none -v "${PWD}:/work" -v "${PWD}/data/raw:/work/data/raw:ro" -w /work wardwatch-analysis:0.2.0 python -m unittest tests.pipeline.test_autoencoder -v
```

Run records are written under `data/processed/v2/runs/expA|expB/` and seed-0 models under `data/processed/v2/models/expA/` (both gitignored — regenerated from code).

> Report-grade runs add `--for-report`, which refuses to write a record unless full provenance (git commit, dirty flag, image digest, seeds, dataset/split hashes) is supplied via the `WARDWATCH_*` env vars — see the top-level `README.md`.

---

## 11. Open threads (honest, for `docs/REPORT_NOTES.md`)

1. **Experiment C** needs the CICIoMT2024 extracted features (`data/processed/cic/`). The code is ready; the data is not in the v2 bundle. Obtain/extract it to complete the cross-dataset evaluation.
2. **`eavesdrop`** is the weakest class for both anomaly detectors — a known, explainable limitation (passive listening barely perturbs traffic). Worth a sentence in the report and a candidate for a dedicated feature or the triage layer.
3. A pre-existing **`test_sequence` labelling assertion** (`sequence.py`, the LSTM's loader) fails on the current v2 dataset — unrelated to the autoencoder, but flagged here as it surfaced while running the full suite; it belongs to the labelling/LSTM owner.
