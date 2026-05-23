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
from config import DB_PATH, FEATURES_PATH as FEAT_PATH, ARTIFACTS_PATH
from backtest_engine import _assign_clusters

artifacts = joblib.load(ARTIFACTS_PATH)
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

df_feat_pd = df_feat_pl.to_pandas()
df_feat_pd["Date"] = pd.to_datetime(df_feat_pd["Date"])

start_date = pd.to_datetime("2025-01-01")
valid_dates = df_feat_pd[df_feat_pd["Date"] >= start_date]["Date"].unique()
search_dates = sorted(valid_dates) # Sort chronologically (oldest first)

found_entries = []

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
    
    cluster_id = labels[0]
    karakter = cluster_map_art.get(cluster_id, "Bilinmiyor")
    
    found_entries.append(f"| {str(day_date)[:10]} | {karakter} |")

md_content = "# KUYAS Hissesi 2025 Yılı Küme Geçmişi\n\n"
md_content += "Bu belge, KUYAS hissesinin 2025 yılı başından itibaren Minerva v3 sistemi tarafından her gün hangi kümeye atandığını kronolojik olarak listeler.\n\n"
md_content += "| Tarih | Atandığı Küme (Karakter) |\n"
md_content += "|---|---|\n"
md_content += "\n".join(found_entries)
md_content += "\n\n---\n*Minerva v3 Otomatik Tarama Raporu*\n"

output_file = "KUYAS_2025_Kume_Gecmisi.md"
with open(output_file, "w", encoding="utf-8") as f:
    f.write(md_content)

print(f"✅ Rapor başarıyla '{output_file}' dosyasına kaydedildi.")
