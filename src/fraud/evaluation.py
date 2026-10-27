"""Shared thresholding and uncertainty calculations."""
from __future__ import annotations
import numpy as np
from sklearn.metrics import (average_precision_score, roc_auc_score, precision_recall_curve,
    precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix)
from .config import N_BOOTSTRAP, SEED, ALERT_BUDGET


def best_threshold(y,p):
    precision,recall,thresholds=precision_recall_curve(y,p)
    f1=2*precision*recall/np.maximum(precision+recall,1e-12)
    idx=int(np.nanargmax(f1[:-1])); eligible=np.flatnonzero(precision[:-1]>=.90)
    strict=float(thresholds[eligible[np.argmax(recall[eligible])]]) if len(eligible) else None
    return float(thresholds[idx]),strict


def classification_metrics(y,p,threshold):
    pred=(np.asarray(p)>=threshold).astype(int); y=np.asarray(y); k=min(ALERT_BUDGET,len(y)); top=np.argsort(p)[-k:]
    return {"pr_auc":float(average_precision_score(y,p)),"roc_auc":float(roc_auc_score(y,p)),
        "precision":float(precision_score(y,pred,zero_division=0)),"recall":float(recall_score(y,pred,zero_division=0)),
        "f1":float(f1_score(y,pred,zero_division=0)),"mcc":float(matthews_corrcoef(y,pred)),
        "precision_at_k":float(y[top].mean()),"k":int(k),"confusion_matrix":confusion_matrix(y,pred,labels=[0,1]).tolist()}


def stratified_bootstrap_ci(y,p,threshold,n=N_BOOTSTRAP,seed=SEED):
    rng=np.random.default_rng(seed); y=np.asarray(y); p=np.asarray(p); positives=np.flatnonzero(y==1); negatives=np.flatnonzero(y==0)
    values={key:[] for key in ("pr_auc","roc_auc","precision","recall","f1","mcc","precision_at_k")}
    for _ in range(n):
        idx=np.r_[rng.choice(positives,len(positives),replace=True),rng.choice(negatives,len(negatives),replace=True)]
        metrics=classification_metrics(y[idx],p[idx],threshold)
        for key in values: values[key].append(metrics[key])
    return {key:[float(np.quantile(v,.025)),float(np.quantile(v,.975))] for key,v in values.items()}
