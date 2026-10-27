"""Tier-two validation-only explainability, calibration, and illustrative cost analyses."""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.calibration import calibration_curve
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fraud.config import DATA_PATH, SEED
from fraud.data import FEATURE_COLUMNS, chronological_split, load_transactions
from fraud.models import make_pipeline

RESULTS=Path("results"); FIGURES=Path("figures"); FIGURES.mkdir(exist_ok=True)


def main():
    frame=load_transactions(DATA_PATH); splits=chronological_split(frame)
    train,val=splits.train,splits.validation
    Xtr=train[FEATURE_COLUMNS].to_numpy(); ytr=train.Class.to_numpy()
    Xv=val[FEATURE_COLUMNS].to_numpy(); yv=val.Class.to_numpy()
    final_path=RESULTS/"final_metrics.json"
    if not final_path.exists(): raise SystemExit("Run scripts/run_final.py after full CV; no final model selection is available yet.")
    final=json.loads(final_path.read_text(encoding="utf-8")); chosen=final["models"][0]
    model=make_pipeline(chosen["model"],chosen["strategy"]).fit(Xtr,ytr)
    pv=model.predict_proba(Xv)[:,1]; threshold=chosen["threshold_f1_validation"]
    pred=pv>=threshold
    perm=permutation_importance(model,Xv,yv,scoring="average_precision",n_repeats=5,random_state=SEED,n_jobs=1)
    importance=pd.DataFrame({"feature":FEATURE_COLUMNS,"importance_mean":perm.importances_mean,"importance_std":perm.importances_std}).sort_values("importance_mean",ascending=False)
    importance.to_csv(RESULTS/"permutation_importance.csv",index=False)
    try:
        import shap
        sample=pd.DataFrame(Xv[:min(1000,len(Xv))],columns=FEATURE_COLUMNS)
        if chosen["model"] in {"rf","xgb","catboost"}:
            explainer=shap.TreeExplainer(model.named_steps["model"])
            values=explainer.shap_values(sample)
            if isinstance(values,list): values=values[-1]
            vals=np.asarray(values)
            if vals.ndim==3: vals=vals[:,:,1]
            shap_mean=np.abs(vals).mean(axis=0)
            shap_state="tree SHAP computed on a validation sample"
        else:
            shap_mean=importance.set_index("feature").reindex(FEATURE_COLUMNS).importance_mean.to_numpy()
            shap_state="tree model was not selected; permutation ranking shown as a descriptive proxy"
    except ImportError:
        shap_mean=importance.set_index("feature").reindex(FEATURE_COLUMNS).importance_mean.to_numpy()
        shap_state="SHAP package unavailable; permutation ranking used"
    imp=pd.DataFrame({"feature":FEATURE_COLUMNS,"mean_absolute_shap_or_proxy":shap_mean}).sort_values("mean_absolute_shap_or_proxy",ascending=False)
    imp.to_csv(RESULTS/"shap_importance.csv",index=False)
    fn=val.loc[(yv==1)&(~pred),["Amount","Time",*FEATURE_COLUMNS]].copy(); tp=val.loc[(yv==1)&pred,["Amount","Time",*FEATURE_COLUMNS]].copy()
    profiles=pd.DataFrame({"group":["false_negative","true_positive"],"count":[len(fn),len(tp)],"amount_median":[float(fn.Amount.median()) if len(fn) else None,float(tp.Amount.median()) if len(tp) else None],"hour_median":[float(((fn.Time//3600)%24).median()) if len(fn) else None,float(((tp.Time//3600)%24).median()) if len(tp) else None]})
    profiles.to_csv(RESULTS/"validation_error_profile.csv",index=False)
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    top=imp.head(10).sort_values("mean_absolute_shap_or_proxy")
    axes[0].barh(top.feature,top.mean_absolute_shap_or_proxy,color="#4477AA"); axes[0].set_xlabel("Mean absolute SHAP value or proxy")
    axes[1].bar(["False negative","True positive"],[len(fn),len(tp)],color=["#CC6677","#4477AA"]); axes[1].set_ylabel("Validation fraud cases")
    axes[1].set_title("At the threshold chosen on validation")
    fig.suptitle("Model scores identify patterns, not meanings for anonymized PCA features")
    fig.tight_layout(); fig.savefig(FIGURES/"f14_explainability_errors.png",dpi=300); fig.savefig(FIGURES/"f14_explainability_errors.pdf"); plt.close(fig)
    (RESULTS/"explainability_note.md").write_text("Feature importance describes model behavior, not causality. V1–V28 are anonymized PCA components and have no direct business interpretation.\n\nSHAP state: "+shap_state+"\n",encoding="utf-8")

    # Calibration uses only saved out-of-fold training predictions; cost sensitivity uses validation predictions.
    cvdir=RESULTS/"cv"; rel_rows=[]
    fig,ax=plt.subplots(figsize=(6,5))
    for key,color in [("lr_weights","#4477AA"),("lr_smote","#CC6677")]:
        path=cvdir/f"{key}_oof.npz"
        if path.exists():
            obj=np.load(path); frac,mean=calibration_curve(obj["y"],obj["probability"],n_bins=10,strategy="quantile")
            ax.plot(mean,frac,marker="o",label=key,color=color)
            rel_rows.extend({"model_strategy":key,"mean_predicted":float(x),"observed_fraction":float(y)} for x,y in zip(mean,frac))
    ax.plot([0,1],[0,1],"--",color="gray"); ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Observed fraud fraction"); ax.legend()
    ax.set_title("Out-of-fold calibration compares class weights and SMOTE")
    fig.tight_layout(); fig.savefig(FIGURES/"f11_reliability.png",dpi=300); fig.savefig(FIGURES/"f11_reliability.pdf"); plt.close(fig)
    pd.DataFrame(rel_rows).to_csv(RESULTS/"calibration_curve.csv",index=False)
    fraud_amount=val.Amount.to_numpy(); costs=np.linspace(0,50,26); net=[]
    for cost in costs:
        flagged=pv>=.5; saved=float(fraud_amount[(yv==1)&flagged].sum()); expense=float(flagged.sum()*cost)
        net.append({"review_cost_per_alert":float(cost),"alerts":int(flagged.sum()),"fraud_amount_flagged":saved,"illustrative_net_saving":saved-expense})
    cost_df=pd.DataFrame(net); cost_df.to_csv(RESULTS/"cost_sensitivity.csv",index=False)
    fig,ax=plt.subplots(figsize=(7,5)); ax.plot(cost_df.review_cost_per_alert,cost_df.illustrative_net_saving,color="#4477AA")
    ax.axhline(0,color="gray",linestyle="--"); ax.set_xlabel("Illustrative review cost per alert (currency units)"); ax.set_ylabel("Fraud amount flagged minus review cost")
    ax.set_title("Illustrative net savings vary with manual review cost")
    fig.tight_layout(); fig.savefig(FIGURES/"f13_cost_sensitivity.png",dpi=300); fig.savefig(FIGURES/"f13_cost_sensitivity.pdf"); plt.close(fig)
    print(json.dumps({"selected_model":chosen["model"],"validation_cases_fn":len(fn),"validation_cases_tp":len(tp),"shap":shap_state},indent=2))

if __name__=="__main__": main()
