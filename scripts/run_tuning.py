"""Optional bounded hyperparameter search on train only; never reads the test split."""
from __future__ import annotations
from pathlib import Path
import json
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import make_scorer, average_precision_score
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fraud.config import DATA_PATH, SEED
from fraud.data import FEATURE_COLUMNS, chronological_split, load_transactions
from fraud.models import make_pipeline


def main():
    summary=pd.read_csv("results/cv_summary.csv").sort_values("mean",ascending=False).head(2)
    split=chronological_split(load_transactions(DATA_PATH)); X=split.train[FEATURE_COLUMNS]; y=split.train.Class
    results=[]
    for row in summary.itertuples(index=False):
        pipe=make_pipeline(row.model,row.strategy)
        if row.model=="rf": params={"model__max_features":["sqrt",0.5,None],"model__min_samples_leaf":[1,2,4,8]}
        elif row.model=="lr": params={"model__C":[0.1,0.3,1,3,10]}
        elif row.model=="xgb": params={"model__max_depth":[3,4,6],"model__learning_rate":[0.03,0.05,0.1],"model__min_child_weight":[1,5,10]}
        elif row.model=="catboost": params={"model__depth":[4,6,8],"model__l2_leaf_reg":[3,5,9]}
        else: continue
        search=RandomizedSearchCV(pipe,params,n_iter=min(15,12),scoring="average_precision",cv=StratifiedKFold(5,shuffle=True,random_state=SEED),n_jobs=1,random_state=SEED,refit=True,verbose=1)
        search.fit(X,y)
        results.append({"model":row.model,"strategy":row.strategy,"best_cv_pr_auc":float(search.best_score_),"best_params":search.best_params_})
    Path("results/tuning_results.json").write_text(json.dumps(results,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(results,indent=2))

if __name__=="__main__": main()
