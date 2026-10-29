"""Part C - delay classifier, feature signal, tuning and alert threshold."""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.shipment_prep import FEATURES, NOISE_COLUMNS  # noqa: E402

OUT = ROOT / "outputs"
os.makedirs(OUT, exist_ok=True)

df = pd.read_csv(ROOT / "data" / "modelling_table.csv")
X, y = df[FEATURES], df["is_delayed"]

# ---- 1. stratified split ----
X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=42, stratify=y_tmp)
print("sizes", len(X_train), len(X_val), len(X_test), "| delayed", y_train.sum(), y_val.sum(), y_test.sum())


def cm_metrics(y_true, y_pred):
    """Everything is recomputed from the four confusion-matrix cells, so the CSV is always consistent."""
    tn, fp, fn, tp = (int(v) for v in confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel())
    acc = (tn + tp) / (tn + fp + fn + tp)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"TN": tn, "FP": fp, "FN": fn, "TP": tp, "accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


# ---- 2. blind baseline + three classifiers (0.5 rule) ----
model_rows = [{"model": "blind_all_on_time", **cm_metrics(y_val, np.zeros(len(y_val), dtype=int))}]

logreg = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=1000))]).fit(X_train, y_train)
model_rows.append({"model": "logistic_regression", **cm_metrics(y_val, logreg.predict(X_val))})

best_d, best_f1 = None, -1.0
for d in [2, 3, 4, 5, 6, 8, 10]:                           # ascending -> ties keep the smaller depth
    t = DecisionTreeClassifier(criterion="gini", max_depth=d, random_state=42).fit(X_train, y_train)
    m = cm_metrics(y_val, t.predict(X_val))
    if m["f1"] > best_f1:
        best_d, best_f1, best_row = d, m["f1"], m
model_rows.append({"model": f"decision_tree_depth_{best_d}", **best_row})

rf_default = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
model_rows.append({"model": "random_forest_default", **cm_metrics(y_val, rf_default.predict(X_val))})
pd.DataFrame(model_rows).round(4).to_csv(OUT / "part_c_models.csv", index=False)

# ---- 3. which columns carry signal? ----
rf_all = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
skb_all = SelectKBest(f_classif, k="all").fit(X_train, y_train)
scores = pd.DataFrame({"feature": FEATURES, "f_score": skb_all.scores_, "rf_importance": rf_all.feature_importances_})
scores["f_rank"] = scores["f_score"].rank(ascending=False, method="first").astype(int)
scores["is_noise"] = scores["feature"].isin(NOISE_COLUMNS)
scores = scores[["feature", "f_score", "f_rank", "rf_importance", "is_noise"]]
scores.round(4).to_csv(OUT / "part_c_feature_scores.csv", index=False)

selectk_rows = []
for k in [5, 7, 10, 12, 17]:
    p = Pipeline([("selector", SelectKBest(f_classif, k=k)),
                  ("model", RandomForestClassifier(n_estimators=100, random_state=42))]).fit(X_train, y_train)
    selectk_rows.append({"k": k, "val_f1": cm_metrics(y_val, p.predict(X_val))["f1"]})
pd.DataFrame(selectk_rows).round(4).to_csv(OUT / "part_c_selectk.csv", index=False)

# ---- 4. tuning on the 7 selected columns (selector fitted on TRAIN only) ----
K = 7
selector = SelectKBest(f_classif, k=K).fit(X_train, y_train)
Xtr7, Xv7 = selector.transform(X_train), selector.transform(X_val)
grid = {"n_estimators": [50, 100, 150], "max_depth": [4, 6, 8], "min_samples_leaf": [1, 5, 10]}


def clean_params(p):
    return {k: int(v) for k, v in p.items()}


gs = GridSearchCV(RandomForestClassifier(random_state=42), grid, cv=5).fit(Xtr7, y_train)
rs = RandomizedSearchCV(RandomForestClassifier(random_state=42), grid, n_iter=9, cv=5, random_state=42).fit(Xtr7, y_train)
rf7 = RandomForestClassifier(n_estimators=100, random_state=42).fit(Xtr7, y_train)

tuning = pd.DataFrame([
    {"search": "grid_search", "candidates": 27, "cv_folds": 5, "total_fits": 27 * 5,
     "best_params": json.dumps(clean_params(gs.best_params_)), "best_cv_accuracy": gs.best_score_,
     "val_f1": cm_metrics(y_val, gs.best_estimator_.predict(Xv7))["f1"]},
    {"search": "randomized_search", "candidates": 9, "cv_folds": 5, "total_fits": 9 * 5,
     "best_params": json.dumps(clean_params(rs.best_params_)), "best_cv_accuracy": rs.best_score_,
     "val_f1": cm_metrics(y_val, rs.best_estimator_.predict(Xv7))["f1"]},
    {"search": "default_forest_same_7_features", "candidates": 1, "cv_folds": 0, "total_fits": 1,
     "best_params": json.dumps({"n_estimators": 100, "max_depth": "unlimited", "min_samples_leaf": 1}),
     "best_cv_accuracy": np.nan, "val_f1": cm_metrics(y_val, rf7.predict(Xv7))["f1"]},
]).round(4)
tuning.to_csv(OUT / "part_c_tuning.csv", index=False)          # blank cell for the default row's CV accuracy

# ---- 5. final configuration: highest val_f1 from the table; ties -> default, grid, randomized ----
priority = ["default_forest_same_7_features", "grid_search", "randomized_search"]
t = tuning.set_index("search")
chosen_from = max(priority, key=lambda s: (t.loc[s, "val_f1"], -priority.index(s)))
if chosen_from == "default_forest_same_7_features":
    params = {"n_estimators": 100, "max_depth": None, "min_samples_leaf": 1}
else:
    params = json.loads(t.loc[chosen_from, "best_params"])
final_model = Pipeline([("selector", SelectKBest(f_classif, k=K)),
                        ("model", RandomForestClassifier(random_state=42, **params))]).fit(X_train, y_train)

# ---- 6. alert threshold from VALIDATION probabilities ----
THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]   # exact decimals, not np.arange
p_val = final_model.predict_proba(X_val)[:, 1]
th_rows = [{"threshold": th, **cm_metrics(y_val, (p_val >= th).astype(int))} for th in THRESHOLDS]
th_df = pd.DataFrame(th_rows)
th_df.round(4).to_csv(OUT / "part_c_thresholds.csv", index=False)
ok = th_df[th_df["recall"] >= 0.80]
threshold = float(ok["threshold"].max()) if len(ok) else 0.1

# ---- 7. ONE test evaluation at the chosen threshold ----
p_test = final_model.predict_proba(X_test)[:, 1]
pd.DataFrame([{"threshold": threshold, **cm_metrics(y_test, (p_test >= threshold).astype(int))}]
             ).round(4).to_csv(OUT / "part_c_final_test.csv", index=False)

pd.DataFrame([{"selector_k": K, "n_estimators": params["n_estimators"],
               "max_depth": "unlimited" if params["max_depth"] is None else params["max_depth"],
               "min_samples_leaf": params["min_samples_leaf"], "threshold": threshold,
               "chosen_from": chosen_from}]).to_csv(OUT / "part_c_final_config.csv", index=False)

print("chosen_from:", chosen_from, "| params:", params, "| threshold:", threshold)
