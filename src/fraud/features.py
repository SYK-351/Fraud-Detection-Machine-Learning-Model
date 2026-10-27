"""Deterministic model features derived directly from source columns."""
import numpy as np
import pandas as pd

FEATURE_COLUMNS = [*[f"V{i}" for i in range(1, 29)], "log1p_amount", "hour_sin", "hour_cos"]


def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add amount and cyclical hour features without fitting dataset statistics."""
    result=frame.copy()
    hour=(result["Time"]//3600)%24
    result["log1p_amount"]=np.log1p(result["Amount"])
    result["hour_sin"]=np.sin(2*np.pi*hour/24)
    result["hour_cos"]=np.cos(2*np.pi*hour/24)
    return result
