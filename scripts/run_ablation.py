"""Protocol ablation; excludes the primary chronological holdout from all ablation runs."""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, average_precision_score, recall_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fraud.config import DATA_PATH
from fraud.data import FEATURE_COLUMNS, DataSplits, chronological_split, load_transactions
from fraud.models import estimator
from fraud.evaluation import best_threshold


def make_estimator(name, seed):
    model=estimator(name,"weights")
    if hasattr(model,"set_params"):
        params=model.get_params(deep=False)
        if "random_state" in params: model.set_params(random_state=seed)
    return model


def pipeline(name,seed,scale=True):
    steps=[]
    if scale: steps.append(("scale",StandardScaler()))
    steps.append(("smote",SMOTE(random_state=seed,k_neighbors=5)))
    steps.append(("model",make_estimator(name,seed)))
    return Pipeline(steps)


def record(rows,seed,name,stage,y,p,threshold):
    rows.append({"seed":seed,"model":name,"stage":stage,"threshold":float(threshold),
        "f1":float(f1_score(y,p>=threshold)),"recall":float(recall_score(y,p>=threshold)),
        "pr_auc":float(average_precision_score(y,p))})


def main():
    frame=load_transactions(DATA_PATH)
    primary=chronological_split(frame)
    dev=pd.concat([primary.train,primary.validation]).sort_values("Time")
    X=dev[FEATURE_COLUMNS].to_numpy(); y=dev.Class.to_numpy(); rows=[]
    for seed in (42,123,2026):
        # A0: legacy-like random split, globally fit scaler, SMOTE, then tune on its evaluation partition.
        tr,hold=train_test_split(np.arange(len(y)),test_size=.2,random_state=seed,stratify=y)
        scaler=StandardScaler().fit(X); Xscaled=scaler.transform(X)
        sampler=SMOTE(random_state=seed); Xr,yr=sampler.fit_resample(Xscaled[tr],y[tr])
        for name in ("rf","xgb"):
            legacy=make_estimator(name,seed).fit(Xr,yr); p=legacy.predict_proba(Xscaled[hold])[:,1]
            threshold,_=best_threshold(y[hold],p); record(rows,seed,name,"A0_legacy_like",y[hold],p,threshold)
            corrected=pipeline(name,seed).fit(X[tr],y[tr]); p1=corrected.predict_proba(X[hold])[:,1]
            record(rows,seed,name,"A1_fold_local_scale",y[hold],p1,.5)

            # A2: random train/validation/holdout, threshold set on validation only.
            trainval,test=train_test_split(np.arange(len(y)),test_size=.2,random_state=seed,stratify=y)
            train_idx,val_idx=train_test_split(trainval,test_size=.25,random_state=seed,stratify=y[trainval])
            model=pipeline(name,seed).fit(X[train_idx],y[train_idx]); pv=model.predict_proba(X[val_idx])[:,1]
            threshold,_=best_threshold(y[val_idx],pv); pt=model.predict_proba(X[test])[:,1]
            record(rows,seed,name,"A2_validation_threshold",y[test],pt,threshold)

            # A3: same steps as A2, using contiguous chronological partitions inside development data.
            ordered=dev.reset_index(drop=True); split=chronological_split(ordered)
            train_x=split.train[FEATURE_COLUMNS].to_numpy(); train_y=split.train.Class.to_numpy()
            val_x=split.validation[FEATURE_COLUMNS].to_numpy(); val_y=split.validation.Class.to_numpy()
            test_x=split.test[FEATURE_COLUMNS].to_numpy(); test_y=split.test.Class.to_numpy()
            model=pipeline(name,seed).fit(train_x,train_y); pv=model.predict_proba(val_x)[:,1]
            threshold,_=best_threshold(val_y,pv); pt=model.predict_proba(test_x)[:,1]
            record(rows,seed,name,"A3_chronological",test_y,pt,threshold)
    result=pd.DataFrame(rows); result.to_csv("results/ablation.csv",index=False)
    result=result.rename(columns={"stage":"protocol"})
    fig,ax=plt.subplots(figsize=(9,5))
    for model,group in result.groupby("model"):
        stats=group.groupby("protocol").f1.agg(["mean","std"])
        ax.errorbar(stats["mean"],stats.index,xerr=stats["std"],fmt="o",capsize=3,label=model)
    ax.set_xlabel("F1 (protocol-specific evaluation partition)"); ax.set_title("Protocol ablation with seed spread; primary test holdout excluded")
    ax.legend(); fig.tight_layout(); fig.savefig("figures/f12_protocol_ablation.png",dpi=300); fig.savefig("figures/f12_protocol_ablation.pdf"); plt.close(fig)
    print("Saved results/ablation.csv; primary chronological test partition was excluded.")

if __name__=="__main__": main()
