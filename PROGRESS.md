# PROGRESS
## Status (edit in place)
- Current: Complete (S0.4 deferred; extraction limitation documented) | TEST_EVALUATED: yes (one evaluation)
- Tokens used (user-reported at last gate): n/a (local Luna; no Radox gateway) | Last commit: b5df9e3 docs: complete results report and assets
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

### 2026-10-27 Dependency retry status
- What: Checked the supplied pip hash failure against the current environment. Files: requirements.txt unchanged; no hash requirements or pip hash policy configured.
- Result: hash mismatch could not be reproduced; pytest, core data stack, notebook tools, and imbalanced-learn installed.
- Checks: Seven tests, notebook execution, EDA, FAST CV, compileall, and LaTeX source checks pass.
- Decisions/deviations: Full requirements install stalled on the CatBoost wheel; no CatBoost or full CV run, and the holdout remains untouched.

### 2026-10-27 S3 DONE
- What: Completed all configured five-fold CV pairs and generated F7. Files: results/cv/, results/cv_results.csv, results/cv_summary.csv, figures/f07_cv_pr_auc.*.
- Result: XGBoost/weights led CV at PR-AUC 0.8717; CatBoost/weights was 0.8676.
- Checks: All 14 requested model/strategy pairs have five saved folds; full summary generated.
- Decisions/deviations: The preferred top-three set uses the one-standard-error rule and faster fit time.

### 2026-10-27 S4 DONE; TEST_EVALUATED: yes
- What: Fixed the duplicate fit-time column handling, then evaluated the locked top-three selection once. Files: scripts/run_final.py, results/final_metrics.json, results/test_probabilities.npz, figures/f08-f10.*.
- Result: Test has 56,746 rows and 74 frauds; XGBoost/weights PR-AUC 0.7952 (95% CI 0.7058–0.8756), recall 0.7432, precision 0.9016.
- Checks: Dummy PR-AUC 0.0013; all-legit accuracy 0.9987; paired top-two PR-AUC difference CI [-0.0037, 0.0278].
- Decisions/deviations: The paired interval includes zero, so there is no evidence of a top-two PR-AUC difference. Holdout has been accessed exactly once.

### 2026-10-27 S5 DONE
- What: Completed A0-A3 for RF and XGBoost across three seeds and generated F12. Files: scripts/run_ablation.py, results/ablation.csv, figures/f12_protocol_ablation.*.
- Result: Mean F1 for chronological A3 was 0.823 (RF) and 0.602 (XGBoost); random A2 means were 0.857 and 0.841.
- Checks: All 24 seed/model/stage rows saved; run is resumable and excluded the primary chronological holdout.
- Decisions/deviations: The old digest's RF F1 reference is not directly comparable: this A0 uses development-only random partitions, de-duplicated data, seed-specific splits, and a threshold selected on its evaluation partition rather than a fixed 0.5 threshold.

### 2026-10-27 S6 DONE; S7.1-S7.2 DONE
- What: Computed tree SHAP and validation permutation importance, error profiles, out-of-fold calibration, and review-cost sensitivity. Files: results/permutation_importance.csv, shap_importance.csv, validation_error_profile.csv, calibration_curve.csv, cost_sensitivity.csv, figures/f11/f13/f14.*.
- Result: Selected XGBoost/weights validation errors: 18 false negatives and 39 true positives; SHAP was computed on a 1,000-row validation sample.
- Checks: `scripts/run_extensions.py` completed successfully; primary test holdout was not read.
- Decisions/deviations: Cost results are illustrative across review costs 0–50 per alert; anonymized PCA features are not given business interpretations.

### 2026-10-27 S7 DONE
- What: Ran three-setting randomized searches for the top two CV models, with five-fold training CV and a validation-only recheck. Files: scripts/run_tuning.py, results/tuning_results.json.
- Result: XGBoost tuned CV PR-AUC 0.8661 / validation PR-AUC 0.7721; CatBoost tuned CV PR-AUC 0.8652 / validation PR-AUC 0.7556.
- Checks: 15 CV fits per model completed; no test predictions or labels were read.
- Decisions/deviations: Tuning uses three settings per model to keep CPU runtime bounded; the untouched original test evaluation remains the only holdout result.

### 2026-10-27 S8 DONE
- What: Updated the explanatory notebook to show protocol ablation and tuning outputs, then rebuilt and executed it. Files: notebooks/main.py, main.ipynb.
- Result: Notebook reads saved CV, final, ablation, and extension outputs; heavy fits remain outside notebook execution.
- Checks: Jupytext conversion and `nbconvert --execute --inplace` completed successfully.
- Decisions/deviations: Notebook signature storage was redirected to a writable temp data directory; execution only reads cached model results.

### 2026-10-27 S9 DONE
- What: Expanded the IEEEtran report with generated result tables and figures and rebuilt the Overleaf archive. Files: report/main.tex, report/generated/, report/figures/, report/overleaf_upload.zip.
- Result: The report now includes cross-validation, the single holdout, ablation, explainability, calibration, cost sensitivity, and tuning results.
- Checks: `scripts/check_tex.py` passed; the desktop LaTeX compiler compiled the saved source successfully.
- Decisions/deviations: The old report bibliography could not be recovered from the supplied digest; this limitation and the handling of its two visible placeholders are documented in `docs/CITATION_AUDIT.md`.

### 2026-10-27 S10 DONE
- What: Refreshed the results-driven README and completed final QA. Files: README.md, PLAN.md, PROGRESS.md.
- Result: Full CV, one holdout evaluation, ablation, extensions, and tuning outputs are documented; no data or model binaries over 5 MB were staged.
- Checks: Seven tests passed; notebook execution, LaTeX source checks, built-in LaTeX compilation, and `git diff --check` passed.
- Decisions/deviations: Nothing was pushed. S0.4 remains deferred because PDF extraction tools were unavailable; the supplied digest limitation is recorded in the citation audit.
### 2026-10-28 R0.1 DONE
- What: Verified commit-msg and pre-push hook scripts exist and are executable in Git Bash. Files: .git/hooks/commit-msg, .git/hooks/pre-push.
- Result: Both hook scripts executable; their contents block attribution lines and push.
- Checks: Git Bash `test -x` passed for both hooks.
- Decisions/deviations: None.

### 2026-10-28 R0.2 DONE
- What: Checked branch tracking and remotes. Files: none.
- Result: `refactor/v2` has no upstream; `origin` exists and was not changed.
- Checks: `git branch -vv`; `git remote -v`.
- Decisions/deviations: Preserved the existing remote as instructed.

### 2026-10-28 R0.3 DONE
- What: Removed tracked CatBoost training artifacts from the index. Files: catboost_info/ (cached removals).
- Result: `.gitignore` already lists `catboost_info/`; working artifacts remain on disk and are now ignored.
- Checks: `git rm -r --cached catboost_info`; `pytest -q` passed (7 tests).
- Decisions/deviations: No `.gitignore` edit was needed because the required pattern already existed.
