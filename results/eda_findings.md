# EDA findings and decisions

F1 — Finding: the class distribution is highly imbalanced. Decision: report PR-AUC and alert precision/recall; accuracy is contextual only.

F2 — Finding: transaction volume and fraud rate change over the train window. Decision: preserve chronological order and include cyclical hour features.

F3 — Finding: amount distributions overlap. Decision: retain log1p(Amount) as one input among the anonymized V features.

F4 — Finding: marginal effect sizes and mutual information rank features differently. Decision: treat these screens as descriptive and verify with model-based measures.

F5 — Finding: some components separate classes marginally, but their densities overlap. Decision: compare multivariate models without claiming causal or business meaning.

Inference notes: Shapiro-Wilk is uninformative at this sample size; t-test and ANOVA are redundant for two groups; Spearman with a binary target duplicates the rank-based group comparison.
