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
