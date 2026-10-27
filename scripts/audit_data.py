from __future__ import annotations

import json
from pathlib import Path

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fraud.config import DATA_PATH
from fraud.data import audit_splits, chronological_split, load_transactions


def main() -> None:
    frame = load_transactions(DATA_PATH)
    splits = chronological_split(frame)
    audit = audit_splits(frame, splits)
    audit["rows_before_deduplication"] = int(__import__("pandas").read_csv(DATA_PATH, usecols=["Class"]).shape[0])
    Path("results").mkdir(exist_ok=True)
    Path("results/data_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
