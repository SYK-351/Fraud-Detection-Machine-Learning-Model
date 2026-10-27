"""Optional bounded hyperparameter search on train only; never reads the test split."""
from __future__ import annotations
from pathlib import Path
import json
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import average_precision_score, roc_auc_score
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fraud.config import DATA_PATH, SEED
from fraud.data import FEATURE_COLUMNS, chronological_split, load_transactions
from fraud.models import make_pipeline
from fraud.evaluation import best_threshold, classification_metrics


def main():
    summary=pd.read_csv("results/cv_summary.csv").sort_values("mean",ascending=False).head(2)
    split=chronological_split(load_transactions(DATA_PATH)); X=split.train[FEATURE_COLUMNS]; y=split.train.Class
    results=[]
    for row in summary.itertuples(index=False):
        print(f"Tuning {row.model}/{row.strategy}: 3 randomized settings, 5-fold train CV...",flush=True)
        pipe=make_pipeline(row.model,row.strategy)
        if row.model=="rf": params={"model__max_features":["sqrt",0.5,None],"model__min_samples_leaf":[1,2,4,8]}
        elif row.model=="lr": params={"model__C":[0.1,0.3,1,3,10]}
        elif row.model=="xgb": params={"model__max_depth":[3,4,6],"model__learning_rate":[0.03,0.05,0.1],"model__min_child_weight":[1,5,10]}
        elif row.model=="catboost": params={"model__depth":[4,6,8],"model__l2_leaf_reg":[3,5,9]}
        else: continue
        search=RandomizedSearchCV(pipe,params,n_iter=3,scoring="average_precision",cv=StratifiedKFold(5,shuffle=True,random_state=SEED),n_jobs=1,random_state=SEED,refit=True,verbose=2)
        search.fit(X,y)
        Xv=split.validation[FEATURE_COLUMNS]; yv=split.validation.Class.to_numpy()
        pv=search.best_estimator_.predict_proba(Xv)[:,1]
        threshold,strict=best_threshold(yv,pv)
        results.append({"model":row.model,"strategy":row.strategy,"best_cv_pr_auc":float(search.best_score_),"best_params":search.best_params_,
            "validation_pr_auc":float(average_precision_score(yv,pv)),"validation_roc_auc":float(roc_auc_score(yv,pv)),
            "validation_threshold_f1":threshold,"validation_metrics_at_threshold":classification_metrics(yv,pv,threshold)})
        print(f"Finished {row.model}/{row.strategy}; validation PR-AUC={results[-1]['validation_pr_auc']:.4f}.",flush=True)
    Path("results/tuning_results.json").write_text(json.dumps(results,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(results,indent=2))

if __name__=="__main__": main()
