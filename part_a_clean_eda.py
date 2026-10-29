"""Part A - clean the raw export, build features, write the modelling table and EDA evidence."""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.shipment_prep import FEATURES, build_features, clean_shipments  # noqa: E402

os.makedirs(ROOT / "outputs", exist_ok=True)

raw = pd.read_csv(ROOT / "data" / "raw_shipments.csv")
hubs = pd.read_csv(ROOT / "data" / "hubs.csv")

dedup = raw.drop_duplicates()                       # only used for the "before" counts
clean, medians = clean_shipments(raw)
feat = build_features(clean, hubs)
feat["is_delayed"] = (feat["delivery_hours"] > feat["promised_hours"]).astype(int)

feat.to_csv(ROOT / "data" / "modelling_table.csv", index=False)

# ---------- checks file ----------
q1, q3 = feat["delivery_hours"].quantile(0.25), feat["delivery_hours"].quantile(0.75)
iqr = q3 - q1
lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr            # bounds from UNROUNDED quartiles
n_outliers = int(((feat["delivery_hours"] < lower) | (feat["delivery_hours"] > upper)).sum())

checks = [
    ("raw_rows", len(raw)),
    ("duplicate_rows_removed", len(raw) - len(dedup)),
    ("rows_after_dedup", len(dedup)),
    ("missing_distance_km_before", dedup["distance_km"].isna().sum()),
    ("missing_parcel_weight_before", dedup["parcel_weight_kg"].isna().sum()),   # sentinel rows not included
    ("sentinel_weight_rows", (dedup["parcel_weight_kg"] > 100).sum()),
    ("missing_declared_value_before", dedup["declared_value"].isna().sum()),
    ("missing_payment_mode_before", dedup["payment_mode"].isna().sum()),
    ("service_type_spellings_before", dedup["service_type"].nunique()),
    ("service_type_spellings_after", clean["service_type"].nunique()),
    ("declared_value_mean_inr", feat["declared_value_inr"].mean()),             # median-filled column
    ("declared_value_median_inr", feat["declared_value_inr"].median()),
    ("log_declared_value_mean", feat["log_declared_value"].mean()),
    ("log_declared_value_median", feat["log_declared_value"].median()),
    ("median_distance_km", medians["distance_km"]),
    ("median_parcel_weight_kg", medians["parcel_weight_kg"]),
    ("median_declared_value_inr", medians["declared_value_inr"]),
    ("rows_after_hub_merge", len(feat)),
    ("nulls_in_features_and_target", feat[FEATURES + ["delivery_hours", "is_delayed"]].isna().sum().sum()),
    ("delayed_count", feat["is_delayed"].sum()),
    ("delay_rate", feat["is_delayed"].mean()),
    ("hours_q1", q1), ("hours_q3", q3),
    ("hours_lower_bound", lower), ("hours_upper_bound", upper),
    ("hours_outlier_count", n_outliers),
]


def fmt(v):
    return int(v) if isinstance(v, (int, np.integer)) else round(float(v), 4)


pd.DataFrame({"metric": [m for m, _ in checks],
              "value": pd.Series([fmt(v) for _, v in checks], dtype=object)}
             ).to_csv(ROOT / "outputs" / "part_a_checks.csv", index=False)

# ---------- delay evidence table ----------
feat["distance_band"] = pd.cut(feat["distance_km"], bins=[0, 200, 400, 600, 2000],
                               labels=["0-200", "201-400", "401-600", "601+"], right=True)


def group_table(col, dimension):
    g = (feat.groupby(col, observed=True)
         .agg(shipments=("is_delayed", "size"), delayed=("is_delayed", "sum"),
              delay_rate=("is_delayed", "mean"), mean_hours=("delivery_hours", "mean"))
         .reset_index().rename(columns={col: "group"}))
    g.insert(0, "dimension", dimension)
    g["group"] = g["group"].astype(str)
    return g


pd.concat([group_table("service_type", "service_type"),
           group_table("hub_tier", "hub_tier"),
           group_table("distance_band", "distance_band")], ignore_index=True
          ).round(4).to_csv(ROOT / "outputs" / "part_a_delay_by_group.csv", index=False)

# ---------- correlations (numeric columns of the modelling table) ----------
numeric = feat.drop(columns=["distance_band"]).select_dtypes(include="number")
numeric.corr().round(4).to_csv(ROOT / "outputs" / "part_a_correlations.csv")   # keep row labels

print("modelling table:", feat.shape, "| delayed:", int(feat['is_delayed'].sum()))
