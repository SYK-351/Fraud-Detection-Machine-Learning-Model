# PROGRESS
## Status (edit in place)
- Current: S3.5 | Next action: run full resumable cross-validation; TEST_EVALUATED: no
- Tokens used (user-reported at last gate): n/a (local Luna; no Radox gateway) | Last commit: 09def8b Add gitignore for dataset and catboost files
## Log (append only, newest last)
### 2026-10-27 S0 setup in progress
- What: Read project brief and report digest; requested and received S0.1 decisions. Files: PROJECT_BRIEF.md, REPORT_DIGEST.md.
- Result: n/a
- Checks: branch refactor/v2 created; existing .gitignore and notebook edits preserved; legacy notebook staged as rename.
- Decisions/deviations: PDF extraction unavailable (pdftotext and Python PDF libraries absent); use supplied digest. User specified local Luna, IEEEtran two-column, author S.Yaswanth Krishna Chaitanya, and sensitivity range.

### 2026-10-27 S0.6 CHECKED
- What: Python 3.11.9 on Windows; created .venv. Files: .venv/ (ignored).
- Result: n/a
- Checks: dependency availability checked; catboost, shap, jupytext, pytest are missing from base Python.
- Decisions/deviations: dependency installation is expected to exceed five minutes; user-run install command required by brief.



### 2026-10-27 S0 setup finalized with dependency deviation
- What: Created branch, hooks, scaffold, settings, and plan. Files: CLAUDE.md, PLAN.md, requirements.txt, .gitignore, REPORT_DIGEST.md.
- Result: n/a
- Checks: Python 3.11 venv created; user hash error did not reproduce against the plain requirements manifest.
- Decisions/deviations: PDF text extraction unavailable; complete dependency install stalled while downloading CatBoost. Core packages installed; full-model environment remains incomplete.

### 2026-10-27 S1 DONE
- What: Added schema validation, deduplication, deterministic features, chronological and legacy splits. Files: src/fraud/config.py, data.py, features.py, scripts/audit_data.py.
- Result: 283,726 rows and 473 frauds; chronological partitions contain 342, 57, and 74 frauds.
- Checks: Seven tests pass; audit confirms each partition exceeds the minimum positive count.
- Decisions/deviations: Legacy split is kept isolated; primary chronological holdout remains unevaluated.

### 2026-10-27 S2 DONE
- What: Generated imbalance, hourly, amount ECDF, feature-screen, density, and protocol figures; added corrected train-only screens. Files: scripts/run_eda.py, figures/f01-f06, results/eda_feature_ranking.csv, results/eda_findings.md.
- Result: Train-only sample has 170,235 rows and 342 frauds; top absolute marginal effect features are V14, V12, V4, and V3.
- Checks: EDA script completed; outputs are PNG and PDF at 300 dpi.
- Decisions/deviations: F1 is labeled descriptive on deduplicated data; inferential feature statistics use train only.

### 2026-10-27 S3 implementation and FAST run
- What: Added model pipelines, fold-local resampling, resumable CV, config snapshot, pre-run hypotheses, and leakage tests. Files: src/fraud/models.py, evaluation.py, plots.py, scripts/run_cv.py, results/config_snapshot.json.
- Result: FAST weighted LR mean PR-AUC 0.944 in results/cv_fast/lr_weights.csv; this is development-only.
- Checks: Seven tests pass; FAST five-fold run completed and estimated full pair runtime.
- Decisions/deviations: Full CV across all model/strategy pairs remains pending; FAST artifacts are isolated from the full-results directory.

### 2026-10-27 S4 implementation ready; holdout pending
- What: Added validation threshold selection, stratified bootstrap metrics, paired comparison, baselines, and F8-F10 generation. Files: scripts/run_final.py, src/fraud/evaluation.py.
- Result: no holdout metrics; test set remains unevaluated.
- Checks: source compiles; final evaluator requires full results/cv_summary.csv and full fold outputs.
- Decisions/deviations: Preserve the one-touch holdout rule; do not run this stage using FAST results.

### 2026-10-27 S5 implementation ready; ablation pending
- What: Added legacy-like versus corrected protocol runner and F12 output. Files: scripts/run_ablation.py.
- Result: no ablation metrics yet.
- Checks: source compiles; runner uses only the development partitions.
- Decisions/deviations: The primary chronological holdout is excluded to preserve its single evaluation. A0 reproduces the legacy operations only within development data.

### 2026-10-27 S6-S7 analysis runners ready; results pending
- What: Added validation explainability/error profiles, train OOF calibration, illustrative review-cost sensitivity, and bounded tuning runners. Files: scripts/run_extensions.py, scripts/run_tuning.py.
- Result: no Tier 2 metrics yet; these runners require completed CV and final model selection.
- Checks: source compiles; explainability gracefully records a permutation-importance fallback when SHAP is unavailable.
- Decisions/deviations: Cost curve is illustrative and uses validation predictions; no actual review cost was supplied.

### 2026-10-27 S8 DONE
- What: Created Jupytext source and explanatory notebook with cached-result sections. Files: notebooks/main.py, main.ipynb.
- Result: notebook outputs read generated audit and EDA files; final metrics remain pending.
- Checks: Jupytext conversion and `python -m nbconvert --execute --to notebook --inplace main.ipynb` succeeded.
- Decisions/deviations: Heavy fitting stays in scripts; notebook uses RUN_HEAVY=False.

### 2026-10-27 S9 report scaffold ready; final evidence pending
- What: Added IEEEtran journal source, BibTeX references, generated numbers/tables, citation audit, Overleaf ZIP, and static checks. Files: report/, docs/CITATION_AUDIT.md, scripts/export_report_assets.py, scripts/check_tex.py.
- Result: source checks pass; generated metric cells correctly remain pending final evaluation.
- Checks: `scripts/check_tex.py` passes; no local LaTeX compiler is available, as expected.
- Decisions/deviations: The supplied digest does not expose the old bibliography; only its numbered placeholders could be audited. PDF extraction was unavailable in S0.

### 2026-10-27 S10 README and interim QA
- What: Replaced the placeholder README with reproduction commands, limitations, project structure, and v1-to-v2 fixes. Files: README.md, scripts/update_readme_results.py.
- Result: results table is generated from saved CV/final metrics; current display correctly says pending.
- Checks: Seven tests pass; notebook execution and LaTeX source checks pass.
- Decisions/deviations: Final QA remains open until full CV, single holdout evaluation, ablation, and report numbers are generated.
