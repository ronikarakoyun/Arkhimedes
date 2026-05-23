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
from fundamental_engine import load_fundamentals, snapshot_at
from backtest_engine import _assign_clusters
from config import DB_PATH, FEATURES_PATH as FEAT_PATH, ARTIFACTS_PATH, SECTOR_MAP_PATH

MAX_DAYS_TO_SEARCH = 2000
ENKAZ_K    = 2.5
ENKAZ_DROP = 0.55

print("=" * 70)
print("  Minerva v3 — Geriye Dönük Son G1 Bulucu")
print("=" * 70)
print("\n📂 Model ve veriler yükleniyor...")

try:
    artifacts   = joblib.load(ARTIFACTS_PATH)
except FileNotFoundError:
    print(f"Hata: {ARTIFACTS_PATH} bulunamadı.")
    sys.exit(1)

df_feat_pl  = pl.read_parquet(FEAT_PATH).with_columns(pl.col("Date").cast(pl.Date))
df_mkt_pl   = pl.read_parquet(DB_PATH).with_columns(pl.col("Date").cast(pl.Date))
macro_df    = load_macro_features()
macro_df["Date"] = pd.to_datetime(macro_df["Date"])
fund_df     = load_fundamentals()

sector_map: dict = {}
if os.path.exists(SECTOR_MAP_PATH):
    sm = pd.read_csv(SECTOR_MAP_PATH)
    if "Ticker" in sm.columns and "Sector" in sm.columns:
        sector_map = dict(zip(sm["Ticker"], sm["Sector"]))

model_bundle = {
    "cluster_method": artifacts.get("clustering_method", "gmm"),
    "clusterer":      artifacts.get("clusterer", artifacts.get("kmeans")),
}
cf          = artifacts.get("combined_filter")
scaler      = artifacts["scaler"]
weight_vec  = artifacts.get("feature_weights", np.ones(len(FEATURES_FOR_CLUSTERING)))
q_low       = artifacts["q_low"]
q_high      = artifacts["q_high"]
cluster_info = artifacts.get("cluster_info", {})
cluster_map_art = artifacts["cluster_map"]

# Support both run_daily and run_21_days style winrates
if cluster_info:
    win_rates_art = {c: info["win_rate"] for c, info in cluster_info.items()}
else:
    win_rates_art = {} # Default if missing

m_cols = [c for c in macro_df.columns if c.startswith("m_")]

# All dates sorted newest first
all_dates = sorted(df_feat_pl["Date"].unique().to_list(), reverse=True)
search_dates = all_dates[:MAX_DAYS_TO_SEARCH]

print(f"   Arama aralığı: {search_dates[0]} tarihinden geriye {len(search_dates)} gün")

df_mkt_pd = df_mkt_pl.to_pandas()
df_mkt_pd["Date"] = pd.to_datetime(df_mkt_pd["Date"])

df_feat_pd = df_feat_pl.to_pandas()
df_feat_pd["Date"] = pd.to_datetime(df_feat_pd["Date"])

# We only need fundamentals if we evaluate RankScore for the found candidates, but since we are just searching backward day by day,
# doing a huge bulk snapshot for 120 days might be slow. We'll do it on-the-fly or a smaller bulk.
print("   Fundamental veriler hazırlandı (on-the-fly çekilecek)...")

def _enkaz_filter(df_mkt: pd.DataFrame, as_of_date) -> list:
    sub = df_mkt[df_mkt["Date"] <= as_of_date]
    grp = (
        sub.groupby("Ticker")
        .agg(
            cur=("Pclose", "last"),
            hi252=("Phigh", lambda x: x.tail(252).max()),
            lo252=("Plow",  lambda x: x.tail(252).min()),
        )
        .reset_index()
    )
    grp = grp[grp["Ticker"] != "XU100"]
    survivor_mask = ~(
        (grp["hi252"] > grp["lo252"] * ENKAZ_K) &
        (grp["cur"] < grp["hi252"] * ENKAZ_DROP)
    )
    return grp[survivor_mask]["Ticker"].tolist()

def _xu100_context(df_mkt: pd.DataFrame, as_of_date) -> dict:
    sub = (df_mkt[(df_mkt["Ticker"] == "XU100") & (df_mkt["Date"] <= as_of_date)]
           .sort_values("Date").tail(2))
    if len(sub) == 0:
        return {"close": None, "daily_ret": None}
    close = float(sub.iloc[-1]["Pclose"])
    daily = (float(sub.iloc[-1]["Pclose"]) / float(sub.iloc[-2]["Pclose"]) - 1) * 100 if len(sub) >= 2 else None
    return {"close": close, "daily_ret": daily}

# Find G1 target cluster ID
g1_cluster_id = None
for k, v in cluster_map_art.items():
    if "G1" in str(v):
        g1_cluster_id = k
        break

if g1_cluster_id is None:
    print("❌ Model cluster haritasında 'G1' karakteri bulunamadı!")
    print(cluster_map_art)
    sys.exit(1)

print(f"🎯 Hedef Küme ID: {g1_cluster_id} (Karakter: {cluster_map_art[g1_cluster_id]})")
print("\n🔎 Tarama başlıyor...")

found_g1 = False

for i, day_date in enumerate(search_dates, 1):
    day_dt = pd.to_datetime(day_date)
    date_str = str(day_date)[:10]
    
    # Enkaz filtresi
    survivors = _enkaz_filter(df_mkt_pd, day_dt)
    if not survivors:
        continue

    # Son bilinen feature satırı
    feats = (
        df_feat_pd[df_feat_pd["Date"] <= day_dt]
        .sort_values("Date")
        .groupby("Ticker").last()
        .reset_index()
    )
    feats = feats[feats["Ticker"].isin(survivors)].dropna(subset=FEATURES_FOR_CLUSTERING)
    if len(feats) == 0:
        continue

    # Preprocessing + GMM
    pp = apply_preprocessing(feats[FEATURES_FOR_CLUSTERING], FEATURES_FOR_CLUSTERING, q_low, q_high)
    valid = pp.notna().all(axis=1)
    feats = feats[valid].copy().reset_index(drop=True)
    pp    = pp[valid].reset_index(drop=True)

    X = scaler.transform(pp[FEATURES_FOR_CLUSTERING]) * weight_vec
    labels, uyum = _assign_clusters(model_bundle, X)
    feats["Cluster"]  = labels
    feats["Karakter"] = feats["Cluster"].map(cluster_map_art)
    
    # G1 var mı kontrol et
    g1_mask = feats["Cluster"] == g1_cluster_id
    g1_candidates = feats[g1_mask]
    
    print(f"  [{i:>3}/{len(search_dates)}] {date_str}: {len(feats)} hisse tarandı -> {len(g1_candidates)} G1 bulundu.")
    
    if not g1_candidates.empty:
        found_g1 = True
        
        # Sektör ekle
        g1_candidates["Sector"] = g1_candidates["Ticker"].map(sector_map).fillna("Bilinmiyor")
        
        # RankScore hesaplamak için Fundamental on-the-fly snapshot
        g1_tickers = g1_candidates["Ticker"].tolist()
        bulk_queries = pd.DataFrame([(t, day_dt) for t in g1_tickers], columns=["Ticker", "Date"])
        bulk_snap = snapshot_at(fund_df, bulk_queries).set_index(["Ticker", "Date"])
        f_cols = [c for c in bulk_snap.columns if c.startswith("f_") or (c.endswith("_z") and not c.startswith("m_"))]
        
        # RankScore hesapla
        if cf is not None and cf.is_useful():
            macro_row = macro_df[macro_df["Date"] <= day_dt].sort_values("Date").tail(1)
            for col in m_cols:
                g1_candidates[col] = macro_row.iloc[0][col] if not macro_row.empty else np.nan

            for col in f_cols:
                if col in bulk_snap.columns:
                    vals = []
                    for tk in g1_candidates["Ticker"]:
                        try:
                            vals.append(bulk_snap.at[(tk, day_dt), col])
                        except KeyError:
                            vals.append(np.nan)
                    g1_candidates[col] = vals
                else:
                    g1_candidates[col] = np.nan

            feat_cols = cf.feature_cols
            for c in feat_cols:
                if c not in g1_candidates.columns:
                    g1_candidates[c] = np.nan
            try:
                g1_candidates["RankScore"] = cf.predict_rank_score(g1_candidates[feat_cols])
            except Exception as e:
                g1_candidates["RankScore"] = np.nan
        else:
            g1_candidates["RankScore"] = np.nan
            
        # Sonuçları yazdır
        xu = _xu100_context(df_mkt_pd, day_dt)
        xu_str = f"XU100: {xu['close']:,.0f} ({xu['daily_ret']:+.2f}%)" if xu["close"] else "XU100: Veri Yok"
        
        print("\n" + "="*50)
        print(f"🎉 İLK G1 BULUNDU: {date_str}")
        print(f"📊 {xu_str}")
        print("="*50)
        
        # Sort by RankScore if available
        sort_col = "RankScore" if g1_candidates["RankScore"].notna().any() else "Ticker"
        g1_candidates = g1_candidates.sort_values(sort_col, ascending=False).reset_index(drop=True)
        
        for idx, row in g1_candidates.iterrows():
            rs = row.get("RankScore", np.nan)
            rs_str = f"| RankScore: {rs:+.3f}" if not pd.isna(rs) else ""
            print(f" 🔹 {row['Ticker']:<6} | {row['Sector']:<15} {rs_str}")
            
        break

if not found_g1:
    print(f"\n⚠️ Son {MAX_DAYS_TO_SEARCH} gün içerisinde G1 kümesine giren herhangi bir hisse bulunamadı.")
