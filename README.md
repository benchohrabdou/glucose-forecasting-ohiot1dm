# Blood glucose forecasting on OhioT1DM

Forecast a Type 1 diabetes patient's continuous-glucose-monitor (CGM) reading **30 and 60 minutes ahead** from the last hour of CGM data plus insulin and carbohydrate history, using the OhioT1DM dataset. The evaluation follows the OhioT1DM Blood Glucose Level Prediction (BGLP) Challenge convention (RMSE in mg/dL per patient) so results can be set beside the published literature. The point of the repository is a **leak-free, reproducible evaluation protocol**; model sophistication is secondary.

**Headline.** A small LSTM trained on glucose alone reaches 19.03 ± 2.11 mg/dL RMSE at 30 min and 32.57 ± 3.65 at 60 min across the 12 patients, beating both persistence (23.40 / 38.44) and ridge regression (20.23 / 34.22). Adding insulin and carbohydrate history helps a little more. The main weakness is that the model **shrinks its forecasts toward the middle of the range, so it is no better than persistence in hypoglycemia**, the clinically important case (see [Main limitation](#43-main-limitation-forecasts-shrink-toward-the-mean)).

---

## 1. Data and handling rules

OhioT1DM has two cohorts of 6 patients each (2018 release: 559, 563, 570, 575, 588, 591; 2020 release: 540, 544, 552, 567, 584, 596). Each patient has an ~8-week training file and a test file covering roughly the final 10 days, chronologically after the training file. The dataset is distributed under a **Data Use Agreement**:

- Raw data is never committed (`data/` is gitignored) and never leaves the local machine.
- Everything published here is aggregate: counts, error metrics and aggregate figures. **No raw patient trace is published**, including per-patient prediction traces.
- The data root is a config field (`paths.data_root`); no absolute path is hardcoded.

To reproduce, obtain the dataset yourself and place the XML files under `data/raw/<year>/{train,test}/`.

## 2. Method

### 2.1 Preprocessing decisions

Each patient file becomes a regular 5-minute grid with columns `glucose`, `glucose_observed`, `bolus`, `basal`, `carbs`.

| Decision | Choice | Why |
|---|---|---|
| Time alignment | Every stream snapped to the **nearest** 5-min bin; on a collision the earlier CGM reading is kept | A bolus, a meal and a CGM reading from the same moment share a bin. The whole dataset had 1 collision. |
| Missing glucose | **Causal forward-fill of at most 6 bins (30 min)**, identical for train, validation and test; longer gaps stay missing | Linear interpolation uses readings *after* a gap, so bins near a window's end would encode future information, and the model would train on a fill pattern it never sees at test time. This tightens the original plan, which allowed interpolation for training data. |
| Imputed values are flagged | `glucose_observed` (1 = real reading, 0 = forward-filled) is always a model input | The model can tell measured values from carried-forward ones. |
| Scored targets | A window is only used if its **target is a real CGM reading** | Never score against an imputed value. |
| Insulin and carbs | Event streams: absence means 0, never interpolated. Extended boluses are spread uniformly over their duration; basal is rate/12 per bin with temp basals overriding it (0 = pump suspended) | Matches how the pump delivers. |
| Unknown basal at the start of a test file | Seeded from the same patient's **last training basal rate** | Causal, because training strictly precedes testing. A few test files have no basal event for hours or days (patient 588: 3 days). |
| Carbohydrates | `meal.carbs` only, all meal types | `bolus.bwz_carb_input` exists only in the 2018 release, so it cannot be used consistently across cohorts. |
| Events outside the CGM span | Dropped and counted in `results/data_quality.csv` | They cannot be placed on the grid. |
| Wearables, sleep, exercise, stress | **Not used** | Sensors differ between cohorts and self-reports are sparse. |

### 2.2 Windows and splits

- **Input:** the last 12 steps (60 min). **Target:** glucose 6 steps (30 min) or 12 steps (60 min) after the last input. One model per horizon.
- **A window is rejected** if its target is not a real reading, if any input glucose is still missing after the forward-fill, or if more than 25% of its input bins are forward-filled. The same rules apply in train, validation and test.
- **Split:** each patient's training file is cut **chronologically**, last 20% for validation. Windows are built *inside* each segment, so no window straddles the cut: the 17 (30 min) or 23 (60 min) windows per patient that would have straddled it are dropped. Glucose is re-filled inside each segment so no validation reading fills a training bin. Nothing is shuffled before the split.
- **Scaling:** statistics fit on training data only and saved with each checkpoint.
- **Windows kept** (30 min): 103,309 train, 25,729 validation, 30,302 test. At 60 min: 102,261 / 25,379 / 29,961.

### 2.3 Evaluation protocol

- One **population** model per horizon, trained on all 12 patients' training data and evaluated per patient on that patient's test file.
- **No model or hyperparameter choice was made using test data.** Every hyperparameter that is chosen at all (ridge penalty, extrapolation length, early stopping) is chosen on validation, and the LSTM configuration was not tuned. The test files were scored after those choices were fixed.
- **Same windows for every model.** Each run's test windows are hashed and checked against the baselines' hash before scoring; a mismatch raises an error.
- **Metrics:** RMSE and MAE in mg/dL, computed per patient, then reported as mean ± std across patients (sample std, ddof = 1). Persistence is in every table.
- **Coverage:** 95.5% (30,302 of 31,743) of real test CGM readings are scored at 30 min and 94.4% at 60 min; the rest lack a full clean input window. Per patient: `results/test_coverage.csv`.
- **Seeds:** the LSTMs are trained with seeds 0-4 and reported as mean ± std across seeds and across patients. Per-patient results are averaged over seeds.

### 2.4 Models

| Model | Notes |
|---|---|
| Persistence | Last observed glucose value. |
| Linear extrapolation | Slope over the last k readings (k = 6, the largest of {2, 3, 4, 6}, chosen on validation), extrapolated forward. |
| Ridge | On the flattened window; penalty chosen on validation. |
| LSTM | 2 layers, hidden size 64, dropout 0.2, final hidden state → linear head. MSE loss, Adam (1e-3), gradient clipping, early stopping on validation RMSE (patience 10) with best-weights restore. Trained twice: on glucose only, and on glucose + insulin + carbs. |

A Transformer (optional in the plan) was **not implemented**.

## 3. Reproducing

```bash
pip install -r requirements.txt
python -m src.data.preprocess --config configs/base.yaml      # caches 5-min grids
python -m src.data.explore    --config configs/base.yaml      # data-quality report
python -m src.models.baselines --config configs/base.yaml     # baselines, both horizons
python -m src.run_seeds --configs configs/lstm_ph30.yaml configs/lstm_ph60.yaml \
        configs/lstm_ins_ph30.yaml configs/lstm_ins_ph60.yaml --seeds 0 1 2 3 4
python -m src.evaluate --config configs/base.yaml             # rebuild comparison tables
python -m src.analysis --config configs/base.yaml             # range / lag / Clarke / figures
python -m pytest                                              # 99 tests
```

Single runs: `python -m src.train --config configs/lstm_ph30.yaml --seed 0`, then `python -m src.evaluate --config configs/lstm_ph30.yaml --checkpoint checkpoints/lstm_ph30_seed0.pt`.

The tests cover parsing, grid alignment, the gap policy, the leakage rules (target exactly one horizon after the last input, no input at or after the target, no window across the split, scaler fit on training only), rejection rules, the same-windows check, and the aggregation and analysis code, with hand-computed expected values. The leakage tests were checked by deliberately breaking the code (interpolation, backward fill, fill before the split) and confirming they fail.

---

## 4. Results

All numbers are RMSE / MAE in **mg/dL**, on the test files. "±" is the standard deviation across patients of each patient's (seed-averaged) error. **The published-benchmark column is intentionally empty**; the numbers and citations are to be filled in by the project owner from the papers. Comparisons with published work should be made carefully: preprocessing and window-rejection rules differ between papers, and this repository's rules have not been checked against the challenge's.

### 4.1 Headline results

**2020 cohort** (540, 544, 552, 567, 584, 596), the cohort BGLP papers report on:

| Model | 30 min RMSE | 30 min MAE | 60 min RMSE | 60 min MAE | Published BGLP RMSE, 30 / 60 min |
|---|---|---|---|---|---|
| Persistence | 24.22 ± 3.26 | 17.57 ± 2.46 | 40.34 ± 5.57 | 29.79 ± 4.34 |  |
| Linear extrapolation (k=6) | 27.26 ± 3.50 | 18.87 ± 2.62 | 56.63 ± 8.19 | 40.05 ± 6.13 |  |
| Ridge (α=10) | 20.20 ± 2.51 | 14.74 ± 1.81 | 35.15 ± 4.70 | 26.64 ± 3.47 |  |
| LSTM, glucose only | 19.26 ± 2.50 | 13.80 ± 1.77 | 33.83 ± 4.57 | 25.32 ± 3.38 |  |
| LSTM, glucose + insulin/carbs | 18.64 ± 2.55 | 13.38 ± 1.79 | 32.40 ± 4.47 | 24.21 ± 3.25 |  |

**All 12 patients:**

| Model | 30 min RMSE | 30 min MAE | 60 min RMSE | 60 min MAE | Published BGLP RMSE, 30 / 60 min |
|---|---|---|---|---|---|
| Persistence | 23.40 ± 2.91 | 16.96 ± 2.07 | 38.44 ± 4.79 | 28.53 ± 3.51 |  |
| Linear extrapolation (k=6) | 27.33 ± 4.05 | 18.51 ± 2.48 | 55.32 ± 7.92 | 38.62 ± 5.51 |  |
| Ridge (α=10) | 20.23 ± 2.35 | 14.48 ± 1.54 | 34.22 ± 3.73 | 25.76 ± 2.88 |  |
| LSTM, glucose only | 19.03 ± 2.11 | 13.47 ± 1.49 | 32.57 ± 3.65 | 24.21 ± 2.79 |  |
| LSTM, glucose + insulin/carbs | 18.56 ± 2.18 | 13.13 ± 1.47 | 31.70 ± 3.47 | 23.50 ± 2.61 |  |

Across the five seeds, the cross-patient mean RMSE of each LSTM varies by only 0.04-0.13 mg/dL, far below the gaps between models. Per-patient tables are in `results/comparison_per_patient.csv`; all summaries in `results/comparison_summary.csv`. The glucose-only LSTM beats ridge and persistence for every one of the 12 patients at both horizons.

### 4.2 Does insulin and carbohydrate history help?

The same LSTM trained with insulin and carbs added as inputs, compared with the glucose-only LSTM. Δ RMSE is insulin/carbs minus glucose-only, so **negative means it helps**; "improved" counts patients whose seed-averaged RMSE is lower.

| Horizon | Cohort | Patients improved (seed-averaged) | Mean Δ RMSE (mg/dL) | Patients improved, seed by seed |
|---|---|---|---|---|
| 30 min | All 12 | **11 / 12** | -0.47 | 12, 8, 10, 12, 11 |
| 30 min | 2020 (6) | **6 / 6** | -0.62 | 6, 5, 6, 6, 6 |
| 30 min | 2018 (6) | **5 / 6** | -0.33 | 6, 3, 4, 6, 5 |
| 60 min | All 12 | **9 / 12** | -0.87 | 9, 10, 9, 9, 10 |
| 60 min | 2020 (6) | **6 / 6** | -1.44 | 6, 6, 6, 6, 6 |
| 60 min | 2018 (6) | **3 / 6** | -0.31 | 3, 4, 3, 3, 4 |

- At 30 min, 11 of 12 patients improve; at 60 min, 9 of 12. Every 2020 patient improves at both horizons.
- The average gain is modest (about 2.5% of RMSE) but far outside the seed-to-seed variation.
- Single-seed counts are unstable (at 30 min, from 8 to 12 of 12 depending on the seed), which is why the seed-averaged count is the one to quote.
- As a rough guide to strength, an exact sign test gives two-sided p ≈ 0.006 for 11 of 12 and ≈ 0.15 for 9 of 12. The 60-min all-patient result is therefore not decisive by itself; it rests mostly on the 2020 cohort.
- The patients that do not improve are all from the 2018 cohort (563 at both horizons; also 575 and 591 at 60 min).
- Patient 567's test file contains no meal records, yet it improves at both horizons, so at least part of the benefit comes from insulin. The two are not separated in this experiment.

For context, **logging density** (events per day, from `results/data_quality.csv`; per-patient values next to each Δ RMSE in `results/ablation_per_patient.csv`), reported descriptively with no inference drawn:

| Cohort | Meals/day (train file) | Meals/day (test file) | Boluses/day (train file) | Boluses/day (test file) |
|---|---|---|---|---|
| 2018 (median of 6) | 4.2 | 3.5 | 5.0 | 4.6 |
| 2020 (median of 6) | 2.0 | 2.4 | 6.2 | 5.3 |

### 4.3 Main limitation: forecasts shrink toward the mean

The headline numbers hide a systematic weakness. Split by the **actual** glucose range, the LSTMs improve on persistence in the in-range and high ranges but **not in hypoglycemia**:

| Horizon | Actual range | Scored readings | Persistence | Ridge | LSTM glucose | LSTM + insulin/carbs |
|---|---|---|---|---|---|---|
| 30 min | hypo (<70) | 844 | 17.3 | 17.0 | 18.4 | 17.9 |
| 30 min | in range (70–180) | 18,802 | 21.5 | 18.0 | 16.9 | 16.4 |
| 30 min | hyper (>180) | 10,656 | 27.2 | 24.1 | 22.6 | 22.2 |
| 60 min | hypo (<70) | 841 | 38.3 | 40.1 | 41.0 | 38.9 |
| 60 min | in range (70–180) | 18,583 | 35.0 | 29.1 | 27.6 | 26.5 |
| 60 min | hyper (>180) | 10,537 | 44.4 | 41.7 | 39.6 | 39.1 |

RMSE in mg/dL, all patients pooled (so patients with more readings weigh more; this differs from the patient-averaged tables above). LSTMs are seed-averaged. Scored-reading counts are shown because the hypoglycemic range is small.

The errors are also **one-sided**. Mean error (predicted − actual), mg/dL:

| Horizon | Actual range | Scored readings | Persistence | LSTM glucose | LSTM + insulin/carbs |
|---|---|---|---|---|---|
| 30 min | hypo (<70) | 844 | +9.4 | +15.3 | +14.6 |
| 30 min | in range (70–180) | 18,802 | +2.5 | +2.8 | +3.0 |
| 30 min | hyper (>180) | 10,656 | -5.4 | -6.7 | -7.2 |
| 60 min | hypo (<70) | 841 | +25.4 | +37.4 | +35.2 |
| 60 min | in range (70–180) | 18,583 | +6.7 | +8.7 | +7.4 |
| 60 min | hyper (>180) | 10,537 | -14.4 | -20.5 | -20.2 |

In hypoglycemia the forecasts are too high (by about 15 mg/dL at 30 min and 35 at 60 min); in hyperglycemia they are too low. This is the pattern expected when a model is trained to minimise mean squared error: it predicts the conditional *average*, which pulls extreme values toward the middle, and low readings are rare in the training data (about 3% of readings). **This explanation fits the evidence but was not tested** (no experiment here changes the loss). Persistence also has a positive bias in hypoglycemia, which is what one would expect if glucose is usually still falling when a low is reached; this was not checked either.

![RMSE by glycemic range](results/figures/error_by_range.png)

**Lag analysis.** Does the model anticipate changes, or repeat the last value with a delay? The figure below compares each forecast for time *T* with the actual glucose at *T − lag*; a repeat-last-value forecast matches best at a lag equal to the horizon, a perfect forecast at zero.

![Forecast vs actual glucose at different lags](results/figures/lag_curves.png)

| Horizon | Model | Best-matching lag | Slope of predicted vs actual change | Direction right on large moves |
|---|---|---|---|---|
| 30 min | Persistence | 30 min | 0 (by construction) | n/a |
| 30 min | Ridge | 25 min | 0.25 | 76% |
| 30 min | LSTM glucose | 20 min | 0.34 | 79% |
| 30 min | LSTM + insulin/carbs | 20 min | 0.38 | 80% |
| 60 min | Persistence | 60 min | 0 (by construction) | n/a |
| 60 min | Ridge | 50 min | 0.22 | 69% |
| 60 min | LSTM glucose | 45 min | 0.29 | 73% |
| 60 min | LSTM + insulin/carbs | 45 min | 0.34 | 75% |

Best-matching lag is on the 5-minute grid; the slope compares the predicted change (forecast minus last input) with the actual change, so 0 means the last value is repeated and 1 means changes are tracked fully; "large moves" are actual changes of at least 10 mg/dL (persistence scores 0 there by construction). The LSTMs **do anticipate**: they match the actual trace about 10 min (30-min horizon) and 15 min (60-min horizon) earlier than persistence would, and get the direction of large moves right 73-80% of the time (79-80% at 30 min, 73-75% at 60 min). But they predict only about **a third** of each change, and the insulin/carbs inputs help slightly with this. This is the same shrinkage seen in the range analysis, seen from another angle.

A qualitative look at one 8-hour stretch of one test file (a raw CGM trace, so **not published here** under the Data Use Agreement) suggests the same thing: at 30 min the LSTM forecasts follow the movement of the actual trace with a modest delay at turning points, and persistence looks like a copy of the actual trace shifted by the horizon; at 60 min the LSTM forecasts move earlier than persistence but under-react to the size of large rises and falls. This is an illustration from a single stretch, not evidence.

**Clarke error grid.** The share of scored readings in each Clarke zone (higher A + B is safer; zone D is "failure to detect", which includes missed hypoglycemia):

| Horizon | Model | A | B | C | D | E | A + B |
|---|---|---|---|---|---|---|---|
| 30 min | Persistence | 83.4 | 15.6 | 0.0 | 1.0 | 0.0 | 99.0 |
| 30 min | Ridge | 88.0 | 10.7 | 0.0 | 1.3 | 0.0 | 98.7 |
| 30 min | LSTM glucose | 89.2 | 9.3 | 0.0 | 1.4 | 0.0 | 98.5 |
| 30 min | LSTM + insulin/carbs | 89.9 | 8.8 | 0.0 | 1.3 | 0.0 | 98.7 |
| 60 min | Persistence | 64.7 | 32.2 | 0.6 | 2.4 | 0.1 | 96.9 |
| 60 min | Ridge | 67.3 | 29.0 | 0.3 | 3.4 | 0.0 | 96.3 |
| 60 min | LSTM glucose | 70.1 | 26.1 | 0.2 | 3.5 | 0.0 | 96.3 |
| 60 min | LSTM + insulin/carbs | 71.4 | 25.0 | 0.2 | 3.5 | 0.0 | 96.4 |

Zone assignment was cross-checked against an independent implementation (the `clarke_error_grid` package): the two agree on every one of 300,000 random continuous points, and differ only for points lying exactly on a boundary line; on our test windows this changes zone percentages by at most 0.16 percentage points, and only for persistence. It was not checked against Clarke's original paper. The models put more readings in zone A than persistence, but also slightly more in zone D, consistent with the missed-hypoglycemia finding above; persistence has marginally higher A + B at both horizons.

![Prediction error distributions](results/figures/residual_distributions.png)

**Reading counts matter.** Hypoglycemic readings are about 3% of scored readings. In 5 of 12 patients' test files there are fewer than 30 of them (as few as 3), so patient-level hypoglycemia errors are very noisy and the pooled figure is dominated by a few patients (540, 567, 575, 591). Per-patient tables with counts: `results/error_by_range_per_patient.csv`.

## 5. Limitations

- **Small sample.** 12 patients, about 10 test days each; hypoglycemic events are rare, especially in the test files.
- **One configuration.** The LSTM was not tuned (hidden size 64 throughout), and a validation-only search was deliberately not run. Results may understate what the architecture can do.
- **Population model only.** No per-patient fine-tuning or personalisation was evaluated.
- **Self-reported meals.** Carbohydrates are sparse and sometimes mis-timed; one test file (patient 567) has none. Insulin and carbs are not separated in the ablation.
- **No wearable or life-event signals** (heart rate, sleep, exercise, stress).
- **Two cohorts with different devices** (pump and band models) pooled into one model; the 2018 cohort gains less from insulin/carbs and the reason was not investigated.
- **Seeds vs patients.** The across-seed spread measures training noise only; the paired patient counts speak to patient-to-patient consistency, and with 12 patients the statistical power is limited.
- **Comparability with published results** is unverified (see Results), and the benchmark column is empty by design.
- **Not a clinical evaluation.** The Clarke analysis compares forecasts with CGM readings; it is not a medical study.

## 6. Future work

- **Address the hypoglycemia weakness directly.** Try a **weighted loss** (heavier weight, or an asymmetric penalty, for low glucose, where over-prediction is dangerous) or **oversampling of hypoglycemic windows**, and check whether the by-range bias and the lag/slope numbers improve without losing overall RMSE. This would also test the mean-shrinkage explanation above.
- Quantile or distributional outputs to give calibrated low-glucose risk.
- Per-patient fine-tuning of the population model.
- A small validation-only hyperparameter search (e.g. hidden size 64 vs 128) and a 120-minute input window.
- The Transformer encoder from the original plan.
- Separate insulin from carbohydrate contributions in the ablation.
- Compare against published BGLP results once the numbers and citations are filled in.

## 7. Repository layout

```
configs/      base.yaml, lstm_ph{30,60}.yaml, lstm_ins_ph{30,60}.yaml
src/data/     parse.py, preprocess.py, dataset.py, explore.py
src/models/   baselines.py, lstm.py
src/          train.py, evaluate.py, run_seeds.py, analysis.py, plots.py, utils.py
tests/        pytest suite (parsing, grid, leakage, rejection, aggregation, analysis)
results/      metrics CSVs and aggregate figures (never raw data)
```

## 8. Citations

Dataset and benchmark citations are to be added by the project owner.
