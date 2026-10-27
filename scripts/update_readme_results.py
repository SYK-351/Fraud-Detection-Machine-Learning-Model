"""Refresh the README metrics table from saved result files only."""
import json
from pathlib import Path
import pandas as pd

root=Path(__file__).resolve().parents[1]; readme=root/"README.md"; text=readme.read_text(encoding="utf-8-sig")
start="<!--RESULTS-START-->"; end="<!--RESULTS-END-->"
cv_path=root/"results/cv_summary.csv"; final_path=root/"results/final_metrics.json"
rows=[]
if cv_path.exists():
    cv=pd.read_csv(cv_path)
    for item in cv.sort_values("mean",ascending=False).itertuples(index=False):
        rows.append(f"| {item.model} / {item.strategy} | {item.mean:.4f} | pending |")
if final_path.exists():
    final=json.loads(final_path.read_text(encoding="utf-8"))
    finals={f"{m['model']} / {m['strategy']}":m["pr_auc"] for m in final.get("models",[])}
    if rows:
        updated=[]
        for row in rows:
            key=row.split("|")[1].strip()
            if key in finals: row=row.rsplit("|",2)[0]+f"| {finals[key]:.4f} |"
            updated.append(row)
        rows=updated
    else:
        rows=[f"| {m['model']} / {m['strategy']} | pending | {m['pr_auc']:.4f} |" for m in final.get("models",[])]
if rows:
    block="\n| Model / strategy | CV mean PR-AUC | Holdout PR-AUC |\n|---|---:|---:|\n"+"\n".join(rows)
else: block="\nResults are pending full cross-validation and the single final holdout evaluation."
text=text[:text.index(start)+len(start)]+block+"\n\n"+text[text.index(end):]
readme.write_text(text,encoding="utf-8")
print("README results section refreshed from saved results files.")
