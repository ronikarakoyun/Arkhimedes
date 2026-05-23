import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings
warnings.filterwarnings("ignore")

import polars as pl
import pandas as pd
import numpy as np
import joblib
import os
import sys

from model_core import apply_preprocessing, FEATURES_FOR_CLUSTERING
from macro_engine import load_macro_features
from backtest_engine import _assign_clusters
from config import DB_PATH, FEATURES_PATH as FEAT_PATH, ARTIFACTS_PATH

print("=" * 70)
print("  Minerva v3 — KUYAS Tüm Zamanlar G1 Bulucu")
print("=" * 70)

try:
    artifacts = joblib.load(ARTIFACTS_PATH)
except FileNotFoundError:
    print(f"Hata: {ARTIFACTS_PATH} bulunamadı.")
    sys.exit(1)

df_feat_pl = pl.read_parquet(FEAT_PATH).with_columns(pl.col("Date").cast(pl.Date))

model_bundle = {
    "cluster_method": artifacts.get("clustering_method", "gmm"),
    "clusterer":      artifacts.get("clusterer", artifacts.get("kmeans")),
}
scaler      = artifacts["scaler"]
weight_vec  = artifacts.get("feature_weights", np.ones(len(FEATURES_FOR_CLUSTERING)))
q_low       = artifacts["q_low"]
q_high      = artifacts["q_high"]
cluster_map_art = artifacts["cluster_map"]

g1_cluster_id = None
for k, v in cluster_map_art.items():
    if "G1" in str(v):
        g1_cluster_id = k
        break

df_feat_pd = df_feat_pl.to_pandas()
df_feat_pd["Date"] = pd.to_datetime(df_feat_pd["Date"])

search_dates = sorted(df_feat_pd["Date"].unique(), reverse=True)

print(f"🔎 KUYAS için tüm zamanlarda ({len(search_dates)} işlem günü) G1 taraması başlatılıyor...")

found_dates = []
for day_date in search_dates:
    day_dt = pd.to_datetime(day_date)
    
    feats = (
        df_feat_pd[(df_feat_pd["Date"] <= day_dt) & (df_feat_pd["Ticker"] == "KUYAS")]
        .sort_values("Date")
        .groupby("Ticker").last()
        .reset_index()
    )
    feats = feats.dropna(subset=FEATURES_FOR_CLUSTERING)
    
    if len(feats) == 0:
        continue

    pp = apply_preprocessing(feats[FEATURES_FOR_CLUSTERING], FEATURES_FOR_CLUSTERING, q_low, q_high)
    valid = pp.notna().all(axis=1)
    if not valid.any():
        continue
        
    feats = feats[valid].copy().reset_index(drop=True)
    pp    = pp[valid].reset_index(drop=True)

    X = scaler.transform(pp[FEATURES_FOR_CLUSTERING]) * weight_vec
    labels, _ = _assign_clusters(model_bundle, X)
    
    if labels[0] == g1_cluster_id:
        found_dates.append(str(day_date)[:10])
        # We'll just collect them all, but print the most recent ones.
        if len(found_dates) == 5:
            break

if found_dates:
    print(f"\n🎉 KUYAS hissesinin G1 kümesinde bulunduğu en yakın tarihler:")
    for d in found_dates:
        print(f"   - {d}")
else:
    print(f"\n⚠️ KUYAS hissesi tarihi boyunca hiçbir gün G1 kümesine girmemiş.")

