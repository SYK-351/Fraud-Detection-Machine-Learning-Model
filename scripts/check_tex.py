"""Static consistency checks for an Overleaf-ready LaTeX project."""
from pathlib import Path
import re

root=Path(__file__).resolve().parents[1]/"report"; tex=(root/"main.tex").read_text(encoding="utf-8")
errors=[]
for env in set(re.findall(r"\\begin\{([^}]+)\}",tex)):
    if len(re.findall(rf"\\begin\{{{re.escape(env)}\}}",tex))!=len(re.findall(rf"\\end\{{{re.escape(env)}\}}",tex)): errors.append(f"unbalanced {env} environment")
for image in re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}",tex):
    if not (root/image).exists(): errors.append(f"missing figure: {image}")
bib=(root/"references.bib").read_text(encoding="utf-8")
keys=set(re.findall(r"@\w+\s*\{\s*([^,]+)",bib))
for group in re.findall(r"\\cite\{([^}]+)\}",tex):
    for key in group.split(","):
        if key.strip() not in keys: errors.append(f"missing citation key: {key.strip()}")
labels=set(re.findall(r"\\label\{([^}]+)\}",tex))
for ref in re.findall(r"\\ref\{([^}]+)\}",tex):
    if ref not in labels: errors.append(f"missing label: {ref}")
if re.search(r"TODO",tex,re.I): errors.append("TODO marker remains")
if errors:
    print("\\n".join(errors)); raise SystemExit(1)
print("LaTeX source checks passed")
