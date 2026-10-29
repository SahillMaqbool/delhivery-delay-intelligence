"""Package the final Part C classifier: fit on the training rows, save with joblib, prove the reload works."""
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.shipment_prep import FEATURES, clean_shipments  # noqa: E402

os.makedirs(ROOT / "models", exist_ok=True)
os.makedirs(ROOT / "outputs", exist_ok=True)

cfg = pd.read_csv(ROOT / "outputs" / "part_c_final_config.csv").iloc[0]
max_depth = None if str(cfg["max_depth"]) == "unlimited" else int(cfg["max_depth"])

# same stratified split as Part C
df = pd.read_csv(ROOT / "data" / "modelling_table.csv")
X, y = df[FEATURES], df["is_delayed"]
X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=42, stratify=y_tmp)

pipe = Pipeline([
    ("selector", SelectKBest(f_classif, k=int(cfg["selector_k"]))),
    ("model", RandomForestClassifier(n_estimators=int(cfg["n_estimators"]), max_depth=max_depth,
                                     min_samples_leaf=int(cfg["min_samples_leaf"]), random_state=42)),
]).fit(X_train, y_train)

# the three medians come from the full historical table (same call as Part A)
_, medians = clean_shipments(pd.read_csv(ROOT / "data" / "raw_shipments.csv"))

joblib.dump(pipe, ROOT / "models" / "delay_model.pkl")
pd.DataFrame({"threshold": [float(cfg["threshold"])]}).to_csv(ROOT / "models" / "delay_threshold.csv", index=False)
pd.DataFrame({"feature": FEATURES}).to_csv(ROOT / "models" / "feature_order.csv", index=False)
pd.DataFrame({"column": list(medians), "median": [round(v, 4) for v in medians.values()]}
             ).to_csv(ROOT / "models" / "impute_values.csv", index=False)

# reload check
reloaded = joblib.load(ROOT / "models" / "delay_model.pkl")
diff = np.abs(reloaded.predict_proba(X_test)[:, 1] - pipe.predict_proba(X_test)[:, 1]).max()
pd.DataFrame({"metric": ["test_rows_scored", "reload_max_abs_probability_diff", "threshold"],
              "value": pd.Series([len(X_test), round(float(diff), 4), float(cfg["threshold"])], dtype=object)}
             ).to_csv(ROOT / "outputs" / "part_d_reload_check.csv", index=False)
print("saved model; reload diff =", diff)
