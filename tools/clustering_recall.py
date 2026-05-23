import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""
Clustering precision/recall — K-Means vs HDBSCAN.
Soru: HDBSCAN yüksek-kaliteli (yüksek win-rate) küme buluyor, AMA ralli yapan
hisselerin ne kadarını yakalıyor (recall)?

Her epoch'ta, tam setup evrenindeki B/C geçişleri üzerinden:
  precision (küme win-rate) = küme geçişlerinin ralli oranı
  recall = kümedeki ralliler / TÜM ralliler (gürültü dahil)
"""
import warnings
warnings.filterwarnings("ignore")
import polars as pl
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import hdbscan as hdbscan_lib

import backtest_engine as be
from model_core import (apply_preprocessing, FEATURES_FOR_CLUSTERING, FEATURE_WEIGHTS,
                        RALLY_WINDOW_DAYS, RALLY_GAIN_THRESHOLD, SECOND_LEG_COOLDOWN,
                        CLUSTER_COOLDOWN_DAYS,
                        _isolate_representative_setups, _remove_second_leg_rallies,
                        _detect_cluster_transitions)

# ── Veri ──
df_feat = pl.read_parquet(be.FEAT_PATH).with_columns(pl.col("Date").cast(pl.Datetime("ms")))
df_outcomes = (
    df_feat.lazy().sort(["Ticker", "Date"]).group_by("Ticker", maintain_order=True).agg([
        pl.all(),
        pl.col("Pclose").reverse()
          .rolling_max(window_size=RALLY_WINDOW_DAYS, min_periods=1)
          .reverse().alias("future_max_price")
    ]).explode(pl.all().exclude("Ticker"))
).with_columns([
    ((pl.col("future_max_price") / pl.col("Pclose")) - 1).alias("future_max_gain")
]).collect()
df_feat_pd = df_outcomes.to_pandas()
df_feat_pd['Date'] = pd.to_datetime(df_feat_pd['Date'])
wv = np.array([FEATURE_WEIGHTS[f] for f in FEATURES_FOR_CLUSTERING])


def cluster_stats(train_clean, labels, total_rally):
    """labels atanmış train_clean → küme bazlı (n, precision, recall)."""
    tc = train_clean.copy()
    tc['Cluster'] = labels
    trans = _detect_cluster_transitions(tc, CLUSTER_COOLDOWN_DAYS)
    rally = trans['future_max_gain'] >= RALLY_GAIN_THRESHOLD
    rows = []
    for c in sorted(trans['Cluster'].unique()):
        m = trans['Cluster'] == c
        n = int(m.sum())
        r = int((m & rally).sum())
        rows.append({'cluster': int(c), 'n': n, 'rally': r,
                     'precision': r / n if n else 0.0,
                     'recall': r / total_rally if total_rally else 0.0})
    return pd.DataFrame(rows), int(rally.sum())


print(f"{'Epoch':>6} {'Yöntem':>8} {'toplam_geçiş':>13} {'toplam_ralli':>13} "
      f"{'yüksek-WR küme (≥65%)':>24} {'precision':>10} {'recall':>8}")
print("=" * 92)

for i, retrain_date in enumerate(be.RETRAIN_DATES):
    cutoff = retrain_date - pd.Timedelta(days=RALLY_WINDOW_DAYS)
    train_df = df_feat_pd[df_feat_pd['Date'] <= cutoff].copy()
    train_df = train_df[(train_df['mom_120'] < 0.70) & train_df['mom_120'].notna()
                        & train_df['future_max_gain'].notna()]
    success_pd = train_df[train_df['future_max_gain'] >= RALLY_GAIN_THRESHOLD].copy()
    success_pl = _isolate_representative_setups(pl.from_pandas(success_pd))
    success_pl = _remove_second_leg_rallies(success_pl, SECOND_LEG_COOLDOWN)
    pd_success = success_pl.drop_nulls(subset=FEATURES_FOR_CLUSTERING).to_pandas()
    if len(pd_success) < 500:
        continue

    q_low  = pd_success[FEATURES_FOR_CLUSTERING].quantile(0.01).to_dict()
    q_high = pd_success[FEATURES_FOR_CLUSTERING].quantile(0.99).to_dict()
    spp = apply_preprocessing(pd_success, FEATURES_FOR_CLUSTERING, q_low, q_high)
    scaler = RobustScaler()
    X = scaler.fit_transform(spp[FEATURES_FOR_CLUSTERING]) * wv

    # Tam evren
    tclean = train_df.dropna(subset=FEATURES_FOR_CLUSTERING).copy()
    tpp = apply_preprocessing(tclean, FEATURES_FOR_CLUSTERING, q_low, q_high)
    vmask = tpp[FEATURES_FOR_CLUSTERING].notna().all(axis=1)
    tclean = tclean[vmask].copy()
    Xall = scaler.transform(tpp[vmask][FEATURES_FOR_CLUSTERING]) * wv

    # K-Means
    bk, bs = 3, -1
    for k in range(3, 6):
        km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X[:5000] if len(X) > 5000 else X)
        s = silhouette_score(X[:5000] if len(X) > 5000 else X, km.labels_)
        if s > bs: bs, bk = s, k
    km = KMeans(n_clusters=bk, random_state=42, n_init=10).fit(X)
    km_labels = km.predict(Xall)

    # HDBSCAN (60,5 — kaliteli ayar)
    hdb = hdbscan_lib.HDBSCAN(min_cluster_size=60, min_samples=5, prediction_data=True).fit(X)
    hdb_labels, _ = hdbscan_lib.approximate_predict(hdb, Xall)

    for tag, labels in [("K-Means", km_labels), ("HDBSCAN", hdb_labels)]:
        # önce toplam ralli (gürültü dahil)
        df_stats, total_rally = cluster_stats(tclean, labels, 1)
        total_trans = int(df_stats['n'].sum())
        df_stats, total_rally = cluster_stats(tclean, labels, total_rally)
        # yüksek-WR kümeler (≥%65), gürültü (-1) hariç
        hi = df_stats[(df_stats['precision'] >= 0.65) & (df_stats['cluster'] >= 0)]
        hi_prec = (hi['rally'].sum() / hi['n'].sum()) if hi['n'].sum() else 0.0
        hi_rec = hi['recall'].sum()
        nclu = (df_stats['cluster'] >= 0).sum()
        label = f"{len(hi)}/{nclu} küme"
        print(f"{i+1:>6} {tag:>8} {total_trans:>13,} {total_rally:>13,} "
              f"{label:>24} {hi_prec:>9.1%} {hi_rec:>7.1%}")
    print("-" * 92)
