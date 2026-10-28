# Fraud Detection v2

A reproducible, leakage-aware refactor of the original credit-card fraud notebook. The primary analysis uses a chronological train/validation/holdout protocol, fold-local preprocessing and resampling, train-only model selection, validation-selected thresholds, and uncertainty intervals. Accuracy is reported only as context because the data is highly imbalanced.

## Project status

The complete five-fold CV, one chronological holdout evaluation, three-seed protocol ablation, validation explainability, calibration, review-cost sensitivity, and bounded tuning are saved under `results/`. CV selected XGBoost with class weights; it achieved holdout PR-AUC 0.7952 (95% stratified bootstrap CI 0.7058–0.8756). FAST metrics are development diagnostics and are not final results.

<!--RESULTS-START-->
| Model / strategy | CV mean PR-AUC | Holdout PR-AUC |
|---|---:|---:|
| xgb / weights | 0.8717 | 0.7952 |
| catboost / weights | 0.8676 | 0.7840 |
| rf / smote | 0.8667 | 0.8124 |
| xgb / smote | 0.8606 | pending |
| rf / weights | 0.8580 | pending |
| catboost / smote | 0.8559 | pending |
| knn / none | 0.8239 | pending |
| xgb / undersample | 0.7351 | pending |
| lr / smote | 0.7202 | pending |
| rf / undersample | 0.7194 | pending |
| lr / weights | 0.7107 | pending |
| knn / undersample | 0.7070 | pending |
| catboost / undersample | 0.6602 | pending |
| lr / undersample | 0.6169 | pending |

<!--RESULTS-END-->

## Data

Place the ULB dataset at `data/creditcard.csv`. It is ignored by Git. `scripts/audit_data.py` validates the schema, removes exact duplicates, creates deterministic amount and cyclical-time features, and writes the split audit to `results/data_audit.json`.

The elapsed `Time` column is measured in seconds since the first transaction, not wall-clock time. The primary split is chronological. The legacy random split is retained for a bounded protocol ablation that excludes the primary holdout.

## Setup

Create and activate a Python virtual environment, then install the declared packages:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

If pip reports a hash mismatch, check that the requirements file or pip configuration is not enforcing hashes from a stale lock or cache. `requirements.txt` in this project contains package names only and no expected hashes.

## Reproduction

Run the split audit and tests:

```powershell
$env:PYTHONPATH = "src"
python scripts/audit_data.py
python -m pytest -q tests
```

Run EDA and the bounded development smoke test:

```powershell
python scripts/run_eda.py
python scripts/run_cv.py --fast --models lr --strategies weights
```

Run full CV in resumable model/strategy chunks. Each completed pair is saved and skipped on later runs:

```powershell
python scripts/run_cv.py --models lr rf xgb catboost --strategies weights
python scripts/run_cv.py --models lr rf xgb catboost --strategies smote
python scripts/run_cv.py --models lr rf xgb catboost --strategies undersample
python scripts/run_cv.py --models knn --strategies none undersample
```

After CV has finished for all requested pairs:

```powershell
python scripts/run_final.py
python scripts/run_ablation.py
python scripts/run_extensions.py
python scripts/run_tuning.py
python scripts/export_report_assets.py
python scripts/check_tex.py
```

The final evaluator reads the full CV outputs, chooses thresholds on validation, and accesses the chronological holdout once. The ablation uses only the development partitions so it does not revisit the primary holdout.

## Notebook and report

`notebooks/main.py` is the Jupytext percent-format source for `main.ipynb`. The notebook reads cached outputs and keeps expensive model fitting in scripts. The report is an IEEEtran journal, two-column Overleaf project in `report/`; regenerate its figures, tables, macros, and `overleaf_upload.zip` with `scripts/export_report_assets.py`.

### Which main file should I use?

- `main.ipynb` is the interactive walkthrough. It displays cached audits and results; it is not the implementation of the model pipeline.
- `notebooks/main.py` is the editable Jupytext source that corresponds to `main.ipynb`.
- `report/main.tex` is the paper source, not executable analysis code.
- For the actual data and model implementation, use `src/fraud/` and the commands in `scripts/` below. `legacy/main_v1.ipynb` is preserved historical code and is not the current workflow.

### Figure provenance

The report's figure PDFs are programmatically plotted artifacts, not AI-generated artwork. F1--F6 are drawn by `scripts/run_eda.py` from the input data (F6 is a code-drawn protocol schematic); F7--F14 are plotted by the experiment scripts or `scripts/restyle_report_figures.py` from saved CV, validation, ablation, calibration, cost, and holdout result files. Restyling uses cached outputs and does not fit predictive models. The report's `figures/` PDFs match their source PDFs in `figures/` byte-for-byte.

## Model and method rationale

The comparison includes a dummy prior, logistic regression, random forest, distance-weighted KNN, XGBoost, and CatBoost. It tests class weights, SMOTE, and random undersampling inside training folds. KNN uses the no-resampling and undersampling strategies because SMOTE makes its CPU prediction cost excessive.

Average precision is primary, supported by ROC-AUC, MCC, recall at a validation-selected precision constraint, precision at a review budget, thresholded metrics, and stratified bootstrap intervals. Results are written to `results/`; figures and report numbers are generated from those files.

## Limitations

The PCA features are anonymized and cannot support business interpretation. The dataset covers a short period, has few fraud examples, and has no card or merchant identifiers. It does not establish deployment performance, long-term drift handling, or measured review economics. Review-cost curves are illustrative sensitivity analyses.

## v1 faults and v2 fixes

| v1 issue | v2 correction |
|---|---|
| Scaler fit before splitting | Fit the scaler inside each training-fold pipeline |
| Threshold tuned on the test set | Select thresholds on validation data |
| Model chosen using test metrics | Select with train-only cross-validation |
| Duplicated training paths and typed metrics | Centralized scripts with saved results as the source of truth |
| Single random split on time-ordered data | Chronological primary split and fold-level uncertainty |
| Resampling outside cross-validation | Resample only inside training folds |
| Raw elapsed time as a predictor | Cyclical hour features and no raw `Time` feature |
| Redundant or oversized significance tests | Mann-Whitney U, Cliff's delta, and multiplicity correction |
| KNN with a tiny neighborhood | Distance-weighted neighborhood model with a larger `k` |

## Structure

- `src/fraud/`: configuration, data, features, and model pipelines
- `scripts/`: audit, EDA, CV, final evaluation, ablation, extensions, report export
- `tests/`: split and leakage checks
- `results/`: audits, rankings, CV and final metrics
- `figures/`: generated figures in PNG and PDF
- `report/`: self-contained Overleaf project
- `legacy/main_v1.ipynb`: preserved original notebook
