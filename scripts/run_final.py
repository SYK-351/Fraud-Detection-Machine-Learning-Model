"""Select from saved CV results, tune thresholds on validation, test once, bootstrap CIs."""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, roc_auc_score, precision_recall_curve,
    precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix)
from sklearn.dummy import DummyClassifier
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fraud.config import DATA_PATH, N_BOOTSTRAP, SEED, ALERT_BUDGET
from fraud.data import FEATURE_COLUMNS, chronological_split, load_transactions
from fraud.models import make_pipeline
from fraud.evaluation import best_threshold, classification_metrics as metrics, stratified_bootstrap_ci as bootstrap_ci


def main():
    result=Path("results"); cv=pd.read_csv(result/"cv_summary.csv")
    top_mean=float(cv["mean"].max())
    top_row=cv.sort_values("mean",ascending=False).iloc[0]
    one_se=float(top_row["std"] / np.sqrt(5)) if pd.notna(top_row["std"]) else 0.0
    candidates=cv[cv["mean"] >= top_mean-one_se].copy()
    # Within one standard error, favor the faster measured pipeline.
    cv_folds=pd.read_csv(result/"cv_results.csv")
    speed=cv_folds.groupby(["model","strategy"],as_index=False).fit_seconds.mean().rename(columns={"fit_seconds":"mean_fit_seconds"})
    candidates=candidates.merge(speed,on=["model","strategy"],how="left").sort_values(["mean_fit_seconds","mean"],ascending=[True,False])
    preferred=candidates.iloc[0]
    remainder=cv[~((cv.model==preferred.model)&(cv.strategy==preferred.strategy))].sort_values("mean",ascending=False)
    cv=pd.concat([candidates.iloc[[0]],remainder]).head(3)
    frame=load_transactions(DATA_PATH); splits=chronological_split(frame)
    Xtr=splits.train[FEATURE_COLUMNS].to_numpy(); ytr=splits.train.Class.to_numpy()
    Xv=splits.validation[FEATURE_COLUMNS].to_numpy(); yv=splits.validation.Class.to_numpy()
    # Test labels and features are first accessed below, after all model and threshold choices are fixed.
    selected=[]
    for row in cv.itertuples(index=False):
        if row.model=="dummy":
            model=DummyClassifier(strategy="prior").fit(Xtr,ytr)
        else:
            model=make_pipeline(row.model,row.strategy)
            if row.model == "xgb" and row.strategy == "weights":
                model.set_params(model__scale_pos_weight=(len(ytr)-int(ytr.sum()))/int(ytr.sum()))
            model.fit(Xtr,ytr)
        pv=model.predict_proba(Xv)[:,1]; threshold, strict=best_threshold(yv,pv)
        selected.append((row.model,row.strategy,model,threshold,strict,pv))
    Xtest=splits.test[FEATURE_COLUMNS].to_numpy(); ytest=splits.test.Class.to_numpy()
    output={"test_rows":int(len(ytest)),"test_frauds":int(ytest.sum()),"test_evaluation_count":1,"models":[]}
    probability_data={"y_test":ytest}
    for name,strategy,model,threshold,strict,pv in selected:
        p=model.predict_proba(Xtest)[:,1]
        m=metrics(ytest,p,threshold); m["threshold_f1_validation"]=threshold; m["threshold_precision_90_validation"]=strict
        m["recall_at_validation_precision_90_threshold"]=metrics(ytest,p,strict)["recall"] if strict is not None else None
        m["ci95_stratified_bootstrap"]=bootstrap_ci(ytest,p,threshold)
        m["model"]=name; m["strategy"]=strategy
        output["models"].append(m); probability_data[f"{name}_{strategy}"]=p
    if len(selected)>1:
        a=probability_data[f"{selected[0][0]}_{selected[0][1]}"]
        b=probability_data[f"{selected[1][0]}_{selected[1][1]}"]
        rng=np.random.default_rng(SEED); pos=np.flatnonzero(ytest==1); neg=np.flatnonzero(ytest==0); diffs=[]
        for _ in range(N_BOOTSTRAP):
            idx=np.r_[rng.choice(pos,len(pos),replace=True),rng.choice(neg,len(neg),replace=True)]
            diffs.append(average_precision_score(ytest[idx],a[idx])-average_precision_score(ytest[idx],b[idx]))
        output["paired_pr_auc_difference_top1_minus_top2"]={"estimate":float(average_precision_score(ytest,a)-average_precision_score(ytest,b)),"ci95":[float(np.quantile(diffs,.025)),float(np.quantile(diffs,.975))]}
    # Baselines are evaluated against the same sole test access.
    dummy=DummyClassifier(strategy="prior").fit(Xtr,ytr); pdummy=dummy.predict_proba(Xtest)[:,1]
    output["dummy_baseline"]={"pr_auc":float(average_precision_score(ytest,pdummy)),"roc_auc":float(roc_auc_score(ytest,pdummy))}
    output["all_legit_accuracy"]=float((ytest==0).mean())
    (result/"final_metrics.json").write_text(json.dumps(output,indent=2)+"\n",encoding="utf-8")
    np.savez_compressed(result/"test_probabilities.npz",**probability_data)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve
    Path("figures").mkdir(exist_ok=True)
    fig,ax=plt.subplots(figsize=(7,5)); ax.axhline(float(ytest.mean()),color="gray",linestyle="--",label="Test prevalence")
    for name,strategy,model,threshold,strict,pv in selected:
        p=probability_data[f"{name}_{strategy}"]; prec,rec,_=precision_recall_curve(ytest,p)
        ax.plot(rec,prec,label=f"{name} / {strategy}")
        ax.scatter([recall_score(ytest,p>=threshold)],[precision_score(ytest,p>=threshold,zero_division=0)],s=32)
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title("Final holdout precision-recall curves"); ax.legend()
    fig.tight_layout(); fig.savefig("figures/f08_test_pr_curves.png",dpi=300); fig.savefig("figures/f08_test_pr_curves.pdf"); plt.close(fig)
    fig,axes=plt.subplots(1,len(selected),figsize=(4*len(selected),4)); axes=np.atleast_1d(axes)
    for ax,(name,strategy,model,threshold,strict,pv) in zip(axes,selected):
        p=probability_data[f"{name}_{strategy}"]; cm=confusion_matrix(ytest,p>=threshold,labels=[0,1])
        ax.imshow(cm,cmap="Blues"); ax.set_title(f"{name} / {strategy}"); ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        for (i,j),value in np.ndenumerate(cm): ax.text(j,i,str(int(value)),ha="center",va="center")
    fig.suptitle("Holdout confusion matrices at validation-selected thresholds"); fig.tight_layout()
    fig.savefig("figures/f09_confusion_matrices.png",dpi=300); fig.savefig("figures/f09_confusion_matrices.pdf"); plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,5)); grid=np.linspace(0,1,201)
    for name,strategy,model,threshold,strict,pv in selected:
        precision,recall,thresholds=precision_recall_curve(yv,pv)
        f1=2*precision*recall/np.maximum(precision+recall,1e-12)
        ax.plot(thresholds,f1[:-1],label=f"{name} / {strategy}")
        ax.axvline(threshold,color="gray",alpha=.18)
    ax.set_xlabel("Validation threshold"); ax.set_ylabel("F1"); ax.set_title("Operating thresholds are selected on validation data"); ax.legend()
    fig.tight_layout(); fig.savefig("figures/f10_validation_thresholds.png",dpi=300); fig.savefig("figures/f10_validation_thresholds.pdf"); plt.close(fig)
    print(json.dumps(output,indent=2))

if __name__=="__main__": main()
