# REPORT_DIGEST: the old `report.pdf` (v1), condensed

**Purpose:** the agent reads this file instead of `report.pdf`. Do not open the PDF again
(except the single optional read in Stage S0.4, see PROJECT_BRIEF.md).
**Coverage:** this digest was written from the pages available to the author: the end of the
dataset/EDA section, Section 4 (Methodology) and Sections 5.1-5.6.1 (Results). **Not covered:** abstract,
introduction, related work, start of Section 3, Section 5.6.2 onward (Spearman, Mann-Whitney,
t-test/ANOVA results), conclusion, reference list.

All numbers below were produced by the **flawed v1 protocol** (see PROJECT_BRIEF.md Section 3).
They are kept only as (a) a sanity target for the legacy reproduction in Stage S5 and
(b) a list of claims to re-examine. **Never copy them into the new report as findings.**

## 1. v1 setup as described in the report
- Data: credit-card fraud dataset (`Time`, `V1..V28`, `Amount`, `Class`), ~284k rows, extreme imbalance (Fig. 1).
- Pipeline: clean -> standardise -> **SMOTE** -> models. Statistical analysis: Mutual Information,
  Spearman, Mann-Whitney U (plus t-test/ANOVA in the notebook).
- Models: Logistic Regression, Random Forest, KNN, XGBoost, CatBoost.
- Single random stratified 80/20 split. Test set = 56,746 rows, of which **95 are fraud**.
- Metrics: accuracy, F1, ROC-AUC, average precision, F1-vs-threshold (XGBoost only).

## 2. v1 results (test set, threshold 0.5 unless stated)
| Model | TN | FP | FN | TP | Precision | Recall | F1 | ROC-AUC | AP |
|---|---|---|---|---|---|---|---|---|---|
| Logistic Regression | 55169 | 1482 | 12 | 83 | 0.053 | 0.874 | 0.100 | 0.962 | 0.677 |
| Random Forest | 56644 | 7 | 23 | 72 | 0.911 | 0.758 | 0.827 | 0.949 | 0.807 |
| KNN | 56563 | 88 | 17 | 78 | 0.470 | 0.821 | 0.597 | 0.910 | 0.522 |
| XGBoost | 56545 | 106 | 17 | 78 | 0.424 | 0.821 | 0.559 | 0.976 | 0.796 |
| CatBoost | 56593 | 58 | 19 | 76 | 0.567 | 0.800 | 0.663 | 0.958 | 0.799 |

(Precision, recall and F1 were recomputed from the confusion matrices; the report states F1 values
0.100 / 0.827 / 0.597 / 0.559 / 0.663 and accuracy about 0.973 / 0.999 / 0.998 / 0.997 / 0.998.)
- XGBoost "best threshold" on F1 curve: about **0.98**, chosen on the **test** set.
- MI top features: V17, V14, V12, V10, V11; lowest: V13, V15, V22.
- Amount ratio plot (fraud / non-fraud, with +1): mean about 1.16 reported; std and quartiles near 1; max about 0.09.

## 3. Claims in the old report to re-examine (do not repeat without evidence)
1. "Fraudsters avoid large amounts to reduce detection": causal story from one `max` statistic over ~473 frauds.
2. "Random Forest is the most balanced and robust model": one split, differences of a few TPs on 95 frauds.
3. "Logistic Regression struggled with imbalance": its F1 of 0.10 is explained by SMOTE-shifted probabilities
   at threshold 0.5 (1482 FP); its ROC-AUC (0.962) is higher than RF's (0.949).
4. "Ensemble methods manage imbalance better": confounded by SMOTE and threshold 0.5; untested.
5. Threshold 0.98 "optimal for deployment": selected on test data.
6. "KNN has difficulty with high-dimensional patterns": plausible, but the stepwise ROC/PR curves (Figs. 9-10)
   are mainly an artefact of k=5 (only six distinct probability values).
7. Confusion-matrix table is transposed versus sklearn's convention (rows should be actual class); the
   specificity definition text says "TP" where TN is meant; the accuracy definition says "correct predictions of fraud".
8. Section 5.6 "Test Analysis" heading actually covers feature-relevance statistics, not hypothesis-test results on models.
9. Figure 8 values were typed by hand into the notebook (cell 61), not generated from the evaluation run.

## 4. What is worth keeping from v1
- The model set (LR, RF, KNN, XGBoost, CatBoost) and the idea of ROC + PR + confusion matrix + threshold analysis.
- Mutual information and Mann-Whitney U as descriptive feature screens (now with effect sizes and multiple-testing correction).
- Log-scale amount comparison idea (now as an ECDF).
- Formula block for accuracy / precision / recall / specificity / F1 (fix the typos in item 7 above).

## 5. Old reference list
Not visible in the pages provided. The report cites at least [1] (confusion-matrix definitions) and [3]
(accuracy definition). Stage S0.4 / S9 must list them in `CITATION_AUDIT.md` and mark each KEEP / REPLACE / DROP.
