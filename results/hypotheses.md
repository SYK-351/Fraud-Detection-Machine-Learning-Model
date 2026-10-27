# Pre-run hypotheses (written before any CV evaluation)

| Model/strategy | Hypothesis |
|---|---|
| Dummy prior | PR-AUC equals training prevalence and anchors no-skill performance. |
| Logistic Regression + class weights | Competitive ROC-AUC, weaker PR-AUC than the strongest tree model. |
| Random Forest + class weights | Strong precision at moderate recall, with coarse probabilities. |
| KNN, k=25 + distance weights | Weakest due to 30-dimensional distance geometry and sparse minority neighbors. |
| XGBoost + class weights | Highest or near-highest PR-AUC among tested pairs. |
| CatBoost + class weights | On par with XGBoost and stable across folds. |
| SMOTE strategies | Ranking and threshold behavior may shift; resampling occurs inside each training fold. |
| Undersampling strategies | Ranking may remain useful while posterior probabilities shift. |

Outcomes remain pending until CV runs finish; do not infer them from the test set.
