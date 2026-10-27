# Hypotheses and outcomes

These hypotheses were recorded before full cross-validation. CV values are fold means from `cv_summary.csv`; test values are from the single permitted evaluation in `final_metrics.json`.

| Model/strategy | Hypothesis | Outcome and evidence |
|---|---|---|
| Dummy prior | PR-AUC equals prevalence and anchors no-skill performance. | CONFIRMED: test dummy PR-AUC 0.0013, matching test prevalence (74/56,746). |
| Logistic Regression + class weights | Competitive ROC-AUC, weaker PR-AUC than the strongest tree model. | INCONCLUSIVE for ROC-AUC; PR-AUC is weaker (CV 0.7107 vs XGBoost/weights 0.8717). |
| Random Forest + class weights | Strong precision at moderate recall, with coarse probabilities. | INCONCLUSIVE: RF/weights was not selected for the holdout, so its operating point was not evaluated there. RF/SMOTE test precision was 0.9016 at recall 0.7432. |
| KNN, k=25 + distance weights | Weakest due to 30-dimensional distance geometry and sparse minority neighbors. | REFUTED as “weakest”: KNN/no-resampling CV PR-AUC was 0.8239, above both LR strategies and all undersampling variants, though below the strongest trees. |
| XGBoost + class weights | Highest or near-highest PR-AUC among tested pairs. | CONFIRMED: highest CV mean PR-AUC (0.8717); holdout PR-AUC 0.7952. |
| CatBoost + class weights | On par with XGBoost and stable across folds. | INCONCLUSIVE: CV mean 0.8676 with SD 0.0497; the paired top-two holdout PR-AUC difference CI includes zero, so there is no evidence of a difference. |
| SMOTE strategies | Ranking and threshold behavior may shift; resampling occurs inside each training fold. | PARTLY CONFIRMED: fold-local SMOTE scored well for RF (CV PR-AUC 0.8667) but the strategy-level mean (0.826) was similar to weights (0.827). Threshold effects are model-specific. |
| Undersampling strategies | Ranking may remain useful while posterior probabilities shift. | REFUTED as a competitive ranking hypothesis here: strategy-level mean CV PR-AUC was 0.688, below weights (0.827) and SMOTE (0.826). Calibration/prior shift is addressed separately in S7. |

## Protocol ablation

Across the three development seeds, RF mean F1 was 0.871 for A0, 0.861 for A1, 0.857 for A2, and 0.823 for A3. XGBoost mean F1 was 0.863, 0.751, 0.841, and 0.602, respectively. These are protocol-specific partitions and thresholds; the differences cannot be attributed to scaler leakage alone. The A0 result is not directly comparable with the digest's fixed-threshold reference because A0 selects a threshold on its own evaluation partition.
