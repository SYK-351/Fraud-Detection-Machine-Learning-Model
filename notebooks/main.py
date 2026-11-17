# %% [markdown]
# # Fraud Detection: A Reproducible Study
#
# This notebook explains the analysis and loads generated outputs. Expensive fitting is kept in scripts; cached results make this notebook quick to run.

# %% [markdown]
# ## Contents
#
# - Data and chronological evaluation
# - Why imbalance changes the metric choice
# - Leakage-safe models and resampling
# - EDA findings and decisions
# - Cross-validation and final evaluation
# - Explainability, cost sensitivity, and v1 fixes

# %% [markdown]
# ## Goal
#
# Establish the study inputs and find the saved audit.
#
# ## Concept
#
# The data contains anonymized principal-component features, transaction time, amount, and a binary fraud label. Duplicate rows are removed before splits. Time is elapsed seconds from the first transaction in the file, not wall-clock time. Feature construction is deterministic and learns no statistics from validation or test data.

# %%
from pathlib import Path
import json
import pandas as pd
from IPython.display import display, Markdown, Image

ROOT = Path.cwd()
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
DATA_PATH = ROOT / "data" / "creditcard.csv"
RUN_HEAVY = False

# %% [markdown]
# ## What we see
#
# The audit file is the source for row and fraud counts. If it is missing, run the audit script.
#
# ## Decision
#
# Use contiguous chronological splits as the primary protocol.

# %%
AUDIT_PATH = RESULTS / "data_audit.json"
if AUDIT_PATH.exists():
    audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
    display(pd.json_normalize(audit["splits"], sep="_"))
else:
    display(Markdown("Run `python scripts/audit_data.py` to create the split audit."))

# %% [markdown]
# ## Goal
#
# Explain why accuracy is not the headline measure.
#
# ## Concept
#
# When fraud is rare, predicting every case as legitimate can produce high accuracy. Precision-recall curves focus on finding positive cases and the false alerts created along the way. We report average precision as the primary ranking measure, with ROC-AUC and Matthews correlation as complementary views. Alert precision at a fixed review budget connects ranking to an operational constraint.

# %%
fig = FIGURES / "f01_class_imbalance.png"
if fig.exists(): display(Image(filename=str(fig)))

# %% [markdown]
# ## What we see
#
# The imbalance figure and EDA findings are generated from the audited dataset and train split.
#
# ## Decision
#
# Keep accuracy only as a contextual baseline; do not use it to select models.

# %%
findings = RESULTS / "eda_findings.md"
if findings.exists(): display(Markdown(findings.read_text(encoding="utf-8")))

# %% [markdown]
# ## Goal
#
# Compare models and imbalance treatments without leaking across folds.
#
# ## Concept
#
# Logistic regression is convex and interpretable through coefficients, but it has a linear boundary and needs scaling. Random forests capture interactions without scaling, though their vote averages can be coarse. Distance-weighted nearest neighbors provide a local reference but suffer in high dimensions and cost more at prediction time. XGBoost regularizes gradient-boosted trees and can weight classes; CatBoost uses ordered boosting, while its categorical feature machinery is irrelevant for these numeric inputs. SMOTE and undersampling operate only inside training folds because they alter the class distribution.

# %%
config_path = RESULTS / "config_snapshot.json"
if config_path.exists(): display(Markdown("```json\n" + config_path.read_text(encoding="utf-8") + "```"))

# %% [markdown]
# ## What we see
#
# Cross-validation outputs are the evidence for model selection. FAST outputs are development-only and must not be substituted for full CV.
#
# ## Decision
#
# Prefer the strongest mean PR-AUC, using the one-standard-error rule to favor a simpler or faster option when appropriate.

# %%
cv_path = RESULTS / "cv_summary.csv"
if cv_path.exists():
    display(pd.read_csv(cv_path).sort_values("mean", ascending=False))
else:
    display(Markdown("Full CV has not been run. Use `python scripts/run_cv.py --models ... --strategies ...` in chunks."))

# %% [markdown]
# ## Goal
#
# Measure how protocol choices change the development-set results.
#
# ## Concept
#
# A random split, globally fitted scaling, and test-selected thresholds can each change reported performance. The ablation compares these steps on development data only and reserves the chronological holdout for the single final evaluation. The difference between stages is measured; it is not attributed to leakage without evidence.

# %%
ablation_path = RESULTS / "ablation.csv"
if ablation_path.exists():
    ablation = pd.read_csv(ablation_path)
    display(ablation.groupby(["model", "stage"])[["f1", "recall", "pr_auc"]].agg(["mean", "std"]).round(3))
fig = FIGURES / "f12_protocol_ablation.png"
if fig.exists(): display(Image(filename=str(fig)))

# %% [markdown]
# ## What we see
#
# The table and F12 summarize three random seeds for both tree models. A0 uses a development evaluation partition and does not access the primary chronological holdout.
#
# ## Decision
#
# Use the measured spread across seeds and protocols; do not generalize the random-split result to future transactions.

# %% [markdown]
# ## Goal
#
# Keep threshold choice separate from final evaluation.
#
# ## Concept
#
# Ranking measures do not choose an operating threshold. The threshold is selected on validation data, where false alerts and missed fraud can be inspected before the final test is opened. Stratified bootstrap intervals show how uncertain the test estimates are with a small number of fraud examples. If a paired interval includes zero, the study has no evidence of a difference for that comparison.

# %%
metrics_path = RESULTS / "final_metrics.json"
if metrics_path.exists():
    final_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    display(pd.DataFrame(final_metrics.get("models", [])))
else:
    display(Markdown("Final test evaluation is pending the full CV model selection."))

# %% [markdown]
# ## What we see
#
# Final metrics and confidence intervals are read from the results file; this notebook does not type or recompute headline values.
#
# ## Decision
#
# Use the validation-selected operating point and report uncertainty alongside ranking metrics.

# %%
for image_name in ["f08_test_pr_curves.png", "f09_confusion_matrices.png", "f10_validation_thresholds.png"]:
    image_path = FIGURES / image_name
    if image_path.exists(): display(Image(filename=str(image_path)))

# %% [markdown]
# ## Goal
#
# Describe model behavior and the limits of operational interpretation.
#
# ## Concept
#
# Permutation importance measures how validation performance changes when a feature is shuffled. SHAP decomposes a model score into feature contributions. Neither explains why a person committed fraud, and the PCA components have no direct merchant or customer meaning. Review-cost curves are illustrative sensitivity analyses because no actual review cost or recovery process was supplied.

# %%
for image_name in ["f11_reliability.png", "f13_cost_sensitivity.png", "f14_explainability_errors.png"]:
    image_path = FIGURES / image_name
    if image_path.exists(): display(Image(filename=str(image_path)))

# %%
tuning_path = RESULTS / "tuning_results.json"
if tuning_path.exists():
    tuning = json.loads(tuning_path.read_text(encoding="utf-8"))
    display(pd.DataFrame([{k: item[k] for k in ("model", "strategy", "best_cv_pr_auc", "validation_pr_auc", "validation_roc_auc", "best_params")} for item in tuning]))

# %% [markdown]
# ## Mistakes in v1 and how v2 fixes them
#
# | v1 issue | v2 correction |
# |---|---|
# | Scaling before splitting | Fit scaling inside each training-fold pipeline |
# | Threshold chosen on test | Choose threshold on validation |
# | Model selected by test score | Select by train-only cross-validation |
# | Duplicated training paths and hand-entered metrics | One scripts-to-results path |
# | One random split and a small positive test count | Chronological holdout, cross-validation, and bootstrap uncertainty |
# | Random split ignores order | Preserve order for the primary protocol |
# | Resampling outside folds | Resample only in the training pipeline |
# | Raw elapsed time | Cyclical hour features; remove raw Time from predictors |
# | Shapiro, t-test, ANOVA, and redundant Spearman screens | Use rank-based effect sizes and corrected descriptive screens |
# | KNN with too few neighbors | Use distance-weighted KNN with a larger neighborhood |
#
# ## Closing note
#
# This dataset spans about two days and has anonymized PCA features. It cannot demonstrate deployment performance, concept-drift handling, or business causes of fraud.

# %%
