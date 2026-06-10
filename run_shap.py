"""Run Phase 3 + SHAP only. Usage: python run_shap.py"""
import json
import subprocess
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_style("whitegrid")

PROJECT = Path(__file__).parent
NOTEBOOK = PROJECT / "setup.ipynb"

subprocess.run(
    [sys.executable, "-m", "pip", "install", "lightgbm", "xgboost", "catboost", "scikit-learn", "shap", "-q"],
    check=False,
)

import lightgbm as lgb
import shap
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.metrics import (
    average_precision_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

# Exact cells to run (by index) — avoids wrong matches / skipped !pip cells
CELL_INDICES = [4, 10, 14, 15, 17, 19]  # setup, impute, clean, FE, Phase 3, SHAP

nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
namespace = {
    "__name__": "__main__",
    "np": np,
    "pd": pd,
    "plt": plt,
    "sns": sns,
    "sys": sys,
    "subprocess": subprocess,
    "lgb": lgb,
    "xgb": xgb,
    "shap": shap,
    "CatBoostClassifier": CatBoostClassifier,
    "LabelEncoder": LabelEncoder,
    "StratifiedKFold": StratifiedKFold,
    "average_precision_score": average_precision_score,
    "auc": auc,
    "f1_score": f1_score,
    "confusion_matrix": confusion_matrix,
    "precision_recall_curve": precision_recall_curve,
    "DecisionTreeClassifier": DecisionTreeClassifier,
    "LogisticRegression": LogisticRegression,
}

for i in CELL_INDICES:
    cell = nb["cells"][i]
    if cell["cell_type"] != "code":
        continue
    source = "".join(cell["source"])
    print(f"\n>>> Running cell {i}...")
    exec(compile(source, f"cell_{i}", "exec"), namespace)
    plt.close("all")

png_path = PROJECT / "shap_importance.png"
if png_path.exists():
    print(f"\nSUCCESS: {png_path.resolve()} ({png_path.stat().st_size:,} bytes)")
else:
    print("\nFAILED: shap_importance.png not created.")
    sys.exit(1)
