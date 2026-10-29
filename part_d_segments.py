"""Part D - hub profile, K-Means segments, separation evidence and tier check."""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

OUT = ROOT / "outputs"
os.makedirs(OUT, exist_ok=True)

df = pd.read_csv(ROOT / "data" / "modelling_table.csv")
hubs = pd.read_csv(ROOT / "data" / "hubs.csv")

# ---- 1. hub profile (groupby sorts hub_id ascending: HUB01 ... HUB24) ----
profile = (df.groupby("hub_id")
           .agg(shipments=("shipment_id", "count"), mean_delivery_hours=("delivery_hours", "mean"),
                delay_rate=("is_delayed", "mean"), mean_distance_km=("distance_km", "mean"),
                mean_n_stops=("n_stops", "mean"))
           .reset_index().sort_values("hub_id").reset_index(drop=True))
profile.round(4).to_csv(OUT / "hub_profile.csv", index=False)

# ---- 2. scale + K-Means ----
CLUSTER_COLS = ["mean_delivery_hours", "delay_rate", "mean_distance_km", "mean_n_stops"]
Xs = StandardScaler().fit_transform(profile[CLUSTER_COLS])


def fit_labels(k):
    return KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(Xs)


def across_over_within(X, labels):
    """Mean distance of pairs in different clusters / mean distance of pairs in the same cluster."""
    D = pairwise_distances(X)
    iu = np.triu_indices(len(X), k=1)                       # 276 unordered pairs for 24 hubs
    d = D[iu]
    same = (labels[:, None] == labels[None, :])[iu]
    return d[~same].mean() / d[same].mean()


# ---- 3. separation evidence for K = 2..5 ----
sep = []
for k in [2, 3, 4, 5]:
    lab = fit_labels(k)
    sizes = sorted(np.bincount(lab), reverse=True)
    sep.append({"K": k, "across_over_within": across_over_within(Xs, lab),
                "cluster_sizes": "-".join(str(int(s)) for s in sizes)})
pd.DataFrame(sep).round(4).to_csv(OUT / "part_d_separation.csv", index=False)

# ---- 4. name the K = 3 clusters by mean delay rate ----
profile["cluster"] = fit_labels(3)
order = profile.groupby("cluster")["delay_rate"].mean().sort_values().index.tolist()
names = dict(zip(order, ["Standard", "Watch", "Escalate"]))
profile["segment"] = profile["cluster"].map(names)

segments = profile[["hub_id", "segment"]].merge(hubs[["hub_id", "hub_city", "hub_tier"]], on="hub_id", how="left")
segments.to_csv(OUT / "hub_segments.csv", index=False)

seg_profile = (profile.groupby("segment")
               .agg(hubs=("hub_id", "count"), mean_delivery_hours=("mean_delivery_hours", "mean"),
                    delay_rate=("delay_rate", "mean"), mean_distance_km=("mean_distance_km", "mean"),
                    mean_n_stops=("mean_n_stops", "mean"))
               .reindex(["Standard", "Watch", "Escalate"]).reset_index())
seg_profile.round(4).to_csv(OUT / "segment_profile.csv", index=False)

tier_implies = {1: "Standard", 2: "Watch", 3: "Escalate"}
mismatch = segments[segments["segment"] != segments["hub_tier"].map(tier_implies)]
mismatch.to_csv(OUT / "tier_vs_segment_mismatch.csv", index=False)

print(pd.DataFrame(sep).round(4).to_string(index=False))
