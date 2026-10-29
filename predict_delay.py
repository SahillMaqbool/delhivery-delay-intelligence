"""Score new parcels: delay probability -> flag -> action (uses the SAME cleaning/feature code as training)."""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.shipment_prep import build_features, clean_shipments  # noqa: E402

new_raw = pd.read_csv(ROOT / "data" / "new_shipments.csv")
hubs = pd.read_csv(ROOT / "data" / "hubs.csv")
segments = pd.read_csv(ROOT / "outputs" / "hub_segments.csv")[["hub_id", "segment"]]

model = joblib.load(ROOT / "models" / "delay_model.pkl")
threshold = float(pd.read_csv(ROOT / "models" / "delay_threshold.csv")["threshold"].iloc[0])
feature_order = pd.read_csv(ROOT / "models" / "feature_order.csv")["feature"].tolist()
imp = pd.read_csv(ROOT / "models" / "impute_values.csv")
medians = dict(zip(imp["column"], imp["median"]))

clean, _ = clean_shipments(new_raw, medians)           # stored medians, never recomputed on new data
feats = build_features(clean, hubs)
proba = model.predict_proba(feats[feature_order])[:, 1]

out = feats[["shipment_id", "hub_id"]].copy()
out["delay_probability"] = np.round(proba, 4)
out["flagged"] = (out["delay_probability"] >= threshold).astype(int)    # compare the value written to the file
out = out.merge(segments, on="hub_id", how="left")
out["action"] = np.select(
    [(out["flagged"] == 1) & (out["segment"] == "Escalate"), out["flagged"] == 1],
    ["Escalate to hub manager", "Proactive customer call"],
    default="No action",
)
out = out[["shipment_id", "hub_id", "segment", "delay_probability", "flagged", "action"]]
out.to_csv(ROOT / "outputs" / "new_shipment_actions.csv", index=False)
print(out["flagged"].sum(), "of", len(out), "parcels flagged;", (out["action"] == "Escalate to hub manager").sum(), "escalations")
