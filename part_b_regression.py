"""Part B - delivery-hours regression."""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.shipment_prep import FEATURES  # noqa: E402

OUT = ROOT / "outputs"
os.makedirs(OUT, exist_ok=True)

df = pd.read_csv(ROOT / "data" / "modelling_table.csv")
X, y = df[FEATURES], df["delivery_hours"]

# ---- 1. split 840 / 180 / 180 ----
X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=42)

# ---- 2. scaling without leakage ----
scaler = StandardScaler().fit(X_train)                     # training rows ONLY
Xs_train, Xs_test = scaler.transform(X_train), scaler.transform(X_test)
pd.DataFrame({"metric": ["train_rows", "val_rows", "test_rows",
                         "train_scaled_max_abs_mean", "test_scaled_max_abs_mean"],
              "value": pd.Series([len(X_train), len(X_val), len(X_test),
                                  round(float(np.abs(Xs_train.mean(axis=0)).max()), 4),
                                  round(float(np.abs(Xs_test.mean(axis=0)).max()), 4)], dtype=object)}
             ).to_csv(OUT / "part_b_checks.csv", index=False)


def scores(model_name, split, y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {"model": model_name, "split": split, "MAE": mean_absolute_error(y_true, y_pred),
            "MSE": mse, "RMSE": float(np.sqrt(mse)), "R2": r2_score(y_true, y_pred)}


rows = []
# ---- 3. baseline ----
rows.append(scores("mean_baseline", "val", y_val, np.full(len(y_val), y_train.mean())))

# ---- 4. tree-depth diagnostic ----
depth_rows, best_depth, best_rmse = [], None, np.inf
for d in [1, 2, 3, 4, 6, 8, 12, None]:                     # ascending, so ties keep the smaller depth
    t = DecisionTreeRegressor(max_depth=d, random_state=42).fit(X_train, y_train)
    tr = np.sqrt(mean_squared_error(y_train, t.predict(X_train)))
    va = np.sqrt(mean_squared_error(y_val, t.predict(X_val)))
    depth_rows.append({"max_depth": "unlimited" if d is None else d, "train_rmse": tr, "val_rmse": va})
    if va < best_rmse:
        best_depth, best_rmse = d, va
pd.DataFrame(depth_rows).round(4).to_csv(OUT / "part_b_tree_depth.csv", index=False)

# ---- 5. four candidates ----
candidates = {
    "linear_regression": Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())]),
    "decision_tree": DecisionTreeRegressor(max_depth=best_depth, random_state=42),
    "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "gradient_boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
}
val_rmse = {}
for name, model in candidates.items():
    model.fit(X_train, y_train)
    r = scores(name, "val", y_val, model.predict(X_val))
    rows.append(r)
    val_rmse[name] = r["RMSE"]

# ---- 6. learning-rate experiment ----
lr_rows = []
for lr in [0.01, 0.1, 1.0]:
    g = GradientBoostingRegressor(n_estimators=100, learning_rate=lr, random_state=42).fit(X_train, y_train)
    p = g.predict(X_val)
    lr_rows.append({"learning_rate": lr, "val_rmse": np.sqrt(mean_squared_error(y_val, p)),
                    "val_r2": r2_score(y_val, p)})
pd.DataFrame(lr_rows).round(4).to_csv(OUT / "part_b_gb_learning_rate.csv", index=False)

# ---- 7. cross-validation of the two best + final test ----
ranked = sorted(val_rmse, key=val_rmse.get)
top2 = ranked[:2]
X_trval = pd.concat([X_train, X_val])
y_trval = pd.concat([y_train, y_val])
kf = KFold(n_splits=5, shuffle=True, random_state=42)      # ONE object shared by both models
cv_rows = []
for name in top2:
    s = cross_val_score(candidates[name], X_trval, y_trval, cv=kf)   # default scoring = R2
    cv_rows.append({"model": name, "mean_r2": s.mean(), "std_r2": s.std()})
pd.DataFrame(cv_rows).round(4).to_csv(OUT / "part_b_cv.csv", index=False)

final_name = ranked[0]                                      # lowest validation RMSE
rows.append(scores(final_name, "test", y_test, candidates[final_name].predict(X_test)))  # test used ONCE
pd.DataFrame(rows).round(4).to_csv(OUT / "part_b_metrics.csv", index=False)

print("validation ranking:", ranked, "| final model:", final_name, "| best tree depth:", best_depth)
