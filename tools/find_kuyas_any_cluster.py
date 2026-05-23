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
print("  Minerva v3 — KUYAS Özel Tüm Kümeler Bulucu (2025 Başına Kadar)")
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

df_feat_pd = df_feat_pl.to_pandas()
df_feat_pd["Date"] = pd.to_datetime(df_feat_pd["Date"])

# Filter dates from start of 2025 onwards, sort newest first
start_date = pd.to_datetime("2025-01-01")
valid_dates = df_feat_pd[df_feat_pd["Date"] >= start_date]["Date"].unique()
search_dates = sorted(valid_dates, reverse=True)

print(f"🔎 2025 başından günümüze KUYAS için tarama başlatılıyor... ({len(search_dates)} işlem günü)")

# Geriye doğru arayacağımız için, "ilk girdiği gün" kronolojik olarak "geriye doğru taramada bulduğumuz EN SON GÜN" dür.
# O yüzden tüm 2025'i tarayıp listeyi bulalım.
found_entries = []

for i, day_date in enumerate(search_dates):
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
    
    found_entries.append({
        "Date": str(day_date)[:10],
        "Cluster": cluster_id,
        "Karakter": karakter
    })

if found_entries:
    # found_entries geriye doğru (en yeniden en eskiye) sıralı.
    # 2025 içindeki İLK gün, listenin en son elemanıdır.
    first_detected = found_entries[-1]
    last_detected = found_entries[0]
    
    print("\n🎉 KUYAS hissesinin 2025 yılında sisteme TESPİT EDİLDİĞİ TARİHLER:")
    print(f"   📉 İLK Kez Görüldüğü Tarih: {first_detected['Date']} -> Küme: {first_detected['Karakter']}")
    print(f"   📈 SON Görüldüğü Tarih (En Güncel): {last_detected['Date']} -> Küme: {last_detected['Karakter']}")
    
    # Kümelerin dağılımını göster
    df_found = pd.DataFrame(found_entries)
    print("\n📊 2025 Yılında Bulunduğu Kümelerin Dağılımı:")
    print(df_found["Karakter"].value_counts().to_string())
else:
    print(f"\n⚠️ 2025 başından bugüne KUYAS hissesi sistemde (hiçbir kümede) tespit edilememiştir.")

