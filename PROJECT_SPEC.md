# PROJECT_SPEC.md — Blood Glucose Forecasting on OhioT1DM

You are implementing a time-series deep learning project. Read this whole file before writing code. Follow the interfaces and the constraints exactly; where something is genuinely ambiguous, ask rather than guess.

---

## 1. Goal

Forecast blood glucose 30 and 60 minutes ahead for individual Type 1 diabetes patients, using a recent window of continuous glucose monitor (CGM) readings plus insulin and carbohydrate history, from the **OhioT1DM** dataset.

The project must be benchmarkable against the published OhioT1DM Blood Glucose Level Prediction (BGLP) Challenge literature, where the standard metric is **RMSE in mg/dL** at prediction horizons of 30 and 60 minutes.

Deliverable: a clean, documented, reproducible repository — not a tutorial, not a notebook dump. It will be read by reviewers as a portfolio piece and defended in an interview, so correctness and clarity of the evaluation protocol matter more than model complexity.

**Priority order when trading off:** correct, leak-free evaluation > clear code and documentation > model sophistication.

---

## 2. Data

### 2.1 Access and handling rules (hard constraints)

- The dataset is under a Data Use Agreement. **Never commit raw data.** `data/` is gitignored from the first commit.
- Never upload, transmit, or paste raw patient records anywhere. Aggregate statistics only.
- Code must not hardcode absolute paths to the data; the data root comes from a config field or CLI arg.

### 2.2 Structure

Two cohorts of 6 patients each:

- 2018 release: patient IDs `559, 563, 570, 575, 588, 591` (Medtronic 530G pump, Basis Peak band)
- 2020 release: patient IDs `540, 544, 552, 567, 584, 596` (Medtronic 630G pump, Empatica Embrace band)

Each patient has `<id>-ws-training.xml` and `<id>-ws-testing.xml`. ~8 weeks per patient; the testing file covers roughly the final 10 days and is chronologically **after** the training file. Timestamps are shifted into the future for de-identification.

XML layout — one child element per field, every record is an `<event>` carrying its data in attributes:

```xml
<patient id="559" weight="99" insulin_type="Novalog">
  <glucose_level>
    <event ts="07-12-2021 01:17:00" value="101"/>          <!-- DD-MM-YYYY HH:MM:SS -->
  </glucose_level>
  <finger_stick> <event ts="..." value="..."/> </finger_stick>
  <basal>        <event ts="..." value="0.9"/> </basal>     <!-- U/hr, in effect from ts onward -->
  <temp_basal>   <event ts_begin="..." ts_end="..." value="0"/> </temp_basal>
  <bolus>        <event ts_begin="..." ts_end="..." type="normal" dose="4.2" bwz_carb_input="45"/> </bolus>
  <meal>         <event ts="..." type="Lunch" carbs="45"/> </meal>
  <exercise/>, <sleep/>, <work/>, <stressors/>, <hypo_event/>, <illness/>
  <basis_heart_rate/>, <basis_gsr/>, <basis_skin_temperature/>, ...   <!-- 2018 only -->
  <acceleration/>, ...                                                <!-- 2020 only -->
</patient>
```

Field semantics that drive preprocessing:

| Field | Meaning | Handling |
|---|---|---|
| `glucose_level` | CGM, nominally every 5 min, mg/dL, sensor range ~40–400 | Target + main input. Timestamps jitter by a few seconds; real gaps from sensor changes/dropouts. |
| `bolus` | Insulin dose delivered from `ts_begin` to `ts_end`; `type` ∈ {normal, square, dual variants} | `normal` is instantaneous (`ts_end == ts_begin`); extended types spread the dose uniformly across the covered 5-min bins. |
| `basal` | Background rate in U/hr, a step function valid until the next event | Forward-fill onto the grid; convert to U per 5-min bin (`rate / 12`). |
| `temp_basal` | Temporary override over `[ts_begin, ts_end)`; `value=0` means pump suspended | Overrides the standing basal rate inside its interval. |
| `meal` | Self-reported carbs in grams; `type` includes `HypoCorrection` | Sparse and sometimes mis-timed. `bolus.bwz_carb_input` is an alternative/complementary carb source — treat as a configurable option, not silently merged. |
| `finger_stick` | Meter calibrations | Not used in v1. |
| wearables / life events | Heart rate, GSR, sleep, exercise, stress | **Not used in v1** — sensors differ across cohorts and self-reports are sparse. Do not add them without being asked. |

### 2.3 Existing code

`ohio_parse.py` (provided, place at `src/data/parse.py`) already loads a file into `(meta, {field_tag: DataFrame})` with timestamps parsed and numeric columns converted. **Reuse it; do not rewrite the parser.** You may add functions, but keep `load_patient()`'s signature and behaviour.

---

## 3. Preprocessing specification

Produce, per patient and per split, a regular 5-minute grid DataFrame indexed by timestamp.

1. **Grid.** From the first to last CGM timestamp of that file, at exactly 5-minute steps. Snap each CGM reading to its nearest grid point (jitter is seconds-level). If two readings collide on one bin, keep the first and log the count.
2. **Columns.**
   - `glucose` — mg/dL, NaN where no reading exists
   - `glucose_observed` — boolean mask, True only where a real CGM reading landed in that bin (before any imputation)
   - `bolus` — units delivered in that bin (extended boluses spread uniformly)
   - `basal` — units delivered in that bin from basal/temp_basal (`rate/12`), suspensions = 0
   - `carbs` — grams reported in that bin
   - optional `tod_sin`, `tod_cos` — time-of-day encoding, config-gated
3. **Gap policy — this is the part most often done wrong, get it right and document it.**
   - **Training/validation data:** linearly interpolate `glucose` across gaps of at most `max_interp_gap_min` (default 30 min). Longer gaps stay NaN.
   - **Test data:** never use a value after time *t* to fill a value at or before *t*. Only causal filling (forward-fill / causal interpolation within the same limit) is allowed. Bidirectional interpolation on test inputs leaks future information and inflates results.
   - Insulin and carbs are event streams: absence means zero, so fill with 0, never interpolate.
4. **Config-driven**, with defaults in `configs/`. All thresholds named and documented.

---

## 4. Windowing / dataset construction

- Input window: `window_len` steps (default 12 = 60 min; also support 24 = 120 min).
- Horizon `PH`: 6 steps (30 min) or 12 steps (60 min).
- Sample at index *t*: inputs = grid rows `[t - window_len + 1 .. t]`, target = `glucose` at `t + PH`.
- **Rejection rules** (apply identically in train and test unless stated):
  - Reject if the target bin is not an actually observed CGM reading (`glucose_observed == False`). Never score against an imputed target.
  - Reject if any input glucose value remains NaN after the allowed interpolation.
  - Reject if the fraction of imputed values in the input window exceeds `max_imputed_frac` (default 0.25).
- Emit a `torch.utils.data.Dataset` yielding `(x: FloatTensor[window_len, n_features], y: FloatTensor[1], meta)` where `meta` carries patient id and target timestamp for per-patient reporting.
- **Scaling:** fit on training data only; persist the scaler alongside the checkpoint and reuse it at eval. Glucose is scaled/unscaled explicitly so that all reported metrics are in mg/dL. Do not compute statistics over validation or test data.
- Write a unit test asserting, for random samples, that `target_timestamp - last_input_timestamp == PH * 5 min` and that no input timestamp is ≥ the target timestamp.

---

## 5. Splits

- **Train/validation:** split each patient's training file **chronologically** — last 20% for validation. Never shuffle windows before splitting: windows overlap, so a random split leaks nearly identical samples across the boundary.
- **Test:** the provided `-ws-testing.xml` files, untouched until final evaluation. No hyperparameter is selected using test data. If you need to iterate, iterate on validation.
- **Default training regime:** one *population* model trained on all patients' training data, evaluated per patient. Per-patient fine-tuning is an optional ablation, run only after the population pipeline works end to end.

---

## 6. Models

All models share one interface: input `(B, window_len, n_features)` → output `(B, 1)` predicted glucose (in scaled space; the eval code unscales).

1. **Baselines (build these first, before any neural net).**
   - `Persistence`: predict the last observed glucose value. This is a strong 30-min baseline and must appear in every results table.
   - `LinearExtrapolation`: fit a slope over the last k readings (k configurable) and extrapolate.
   - `RidgeRegression` on the flattened window (scikit-learn or a closed-form numpy implementation).
2. **LSTM (primary model).** 1–2 layers, hidden size 64–128, dropout between layers, take the final hidden state → linear head → 1 output. MSE loss, Adam at 1e-3, gradient clipping, early stopping on validation RMSE with patience ~10, best-checkpoint restore.
3. **Transformer (optional, only after the LSTM is complete and evaluated).** Encoder-only over the input window: linear input projection to `d_model` (64), sinusoidal positional encoding, 2 layers, 4 heads, then pool (last token or mean) → linear head. Keep it small; this dataset is tiny and a large model will simply overfit.

Train one model per horizon (separate runs for PH=6 and PH=12) rather than multi-output, so results map directly onto the literature's reporting.

Every run must be seeded and the seed recorded in the results. Training must fit comfortably on a single GPU or CPU in minutes, not hours.

---

## 7. Evaluation

- Metrics: **RMSE** and **MAE** in mg/dL, computed per patient on that patient's test file, then reported as a per-patient table plus mean ± std across patients.
- Every results table includes the persistence baseline alongside the models, for both horizons.
- Additional analyses (in this order, as time allows):
  - Predicted-vs-actual plots over a representative test window, and residual distributions.
  - Error broken down by glycemic range (hypo <70, in-range 70–180, hyper >180) — clinically the interesting failure mode.
  - Clarke or Parkes error grid analysis if a clean implementation is available.
  - A lag check: compare the model's predictions against the persistence baseline to show whether the model is genuinely anticipating changes or just reproducing the last value shifted forward.
- **Do not invent or recall published benchmark numbers.** Produce the results table with a clearly marked empty column for published BGLP Challenge results; the numbers and citations will be filled in by the project owner from the papers.

---

## 8. Repository structure

```
glucose-forecasting/
├── README.md
├── requirements.txt
├── .gitignore                # data/ , checkpoints/ , __pycache__/
├── configs/
│   ├── base.yaml             # paths, grid, gap policy, window, seed
│   ├── lstm_ph30.yaml
│   ├── lstm_ph60.yaml
│   └── transformer_ph30.yaml
├── src/
│   ├── data/
│   │   ├── parse.py          # provided parser — reuse
│   │   ├── preprocess.py     # XML -> 5-min grid with gap policy
│   │   ├── dataset.py        # windowing, rejection rules, scaling, Dataset
│   │   └── explore.py        # aggregate stats / data-quality report
│   ├── models/
│   │   ├── baselines.py
│   │   ├── lstm.py
│   │   └── transformer.py
│   ├── train.py
│   ├── evaluate.py
│   └── utils.py              # seeding, config loading, logging, metrics
├── tests/                    # pytest: parsing, grid alignment, no-leakage, rejection rules
├── results/                  # metrics CSVs + figures (never raw data)
└── notebooks/                # optional, exploratory only; no pipeline logic lives here
```

CLI:

```bash
python -m src.data.preprocess --config configs/base.yaml     # caches processed grids
python -m src.train    --config configs/lstm_ph30.yaml
python -m src.evaluate --config configs/lstm_ph30.yaml --checkpoint checkpoints/lstm_ph30.pt
```

---

## 9. Code standards

- Python 3.10+, PyTorch, pandas, numpy, pyyaml, matplotlib, pytest, scikit-learn. No other dependencies without asking.
- Type hints on public functions; concise docstrings stating units and time conventions (mg/dL, U, grams, 5-min bins).
- Pure functions for data transforms; no hidden global state; no side effects at import time.
- Comments explain *why* (especially every leakage-prevention decision), not *what*.
- Deterministic: seed python/numpy/torch, log the seed and resolved config with every run.
- Results written as CSV to `results/`, figures as PNG. Never print a metric that isn't also persisted.

---

## 10. Build order (respect this sequence)

1. `preprocess.py` + a data-quality report, verified on one patient before scaling up.
2. `dataset.py` + the leakage unit tests.
3. Baselines end to end, with full per-patient results tables for both horizons. **Stop here and report the numbers before moving on.**
4. LSTM, glucose-only features. Report.
5. LSTM with insulin + carbs features. Report the ablation against glucose-only.
6. Evaluation extras (error by glycemic range, plots, lag analysis).
7. Transformer, if time remains.
8. README and results write-up: problem, data, preprocessing decisions with justification, protocol, results tables, honest discussion of limitations (single-cohort data, self-reported meals, no wearable signals, population vs personalized models).

---

## 11. Rules you must not break

- No future information in any test-time input. Ever.
- No random shuffling before chronological splits.
- No scaler or statistic fit on validation or test data.
- No hyperparameter tuning against the test set.
- No scoring against imputed target values.
- No committed raw data.
- No fabricated citations or benchmark numbers.
- No silent scope expansion (extra features, extra models, extra dependencies) — ask first.

If you hit a genuine design ambiguity — how to merge `meal.carbs` with `bolus.bwz_carb_input`, how to weight patients with much sparser data — stop and ask, with your recommended option and its rationale, instead of picking one silently.
