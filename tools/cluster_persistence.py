import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""
Küme win-rate persistence kontrolü — overfitting testi.
Her epoch için her kümenin EĞİTİM-dönemi WR'si vs TEST-dönemi (ileri yıl) WR'si.
Soru: eğitimde yüksek-WR olan küme, test'te de yüksek kalıyor mu?
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

retrain = list(be.RETRAIN_DATES)
bounds = retrain + [be.BACKTEST_END + pd.Timedelta(days=1)]


def wr_by_cluster(universe_df, labels):
    """labels atanmış universe → {cluster: (n, win_rate)}."""
    u = universe_df.copy()
    u['Cluster'] = labels
    u = u[u['Cluster'] != -1]
    trans = _detect_cluster_transitions(u, CLUSTER_COOLDOWN_DAYS)
    out = {}
    rally = trans['future_max_gain'] >= RALLY_GAIN_THRESHOLD
    for c in sorted(trans['Cluster'].unique()):
        m = trans['Cluster'] == c
        n = int(m.sum())
        out[int(c)] = (n, float((m & rally).sum()) / n if n else 0.0)
    return out


def assign(method, clusterer, scaler, q_low, q_high, df):
    dfc = df.dropna(subset=FEATURES_FOR_CLUSTERING).copy()
    pp = apply_preprocessing(dfc, FEATURES_FOR_CLUSTERING, q_low, q_high)
    v = pp[FEATURES_FOR_CLUSTERING].notna().all(axis=1)
    dfc = dfc[v].copy()
    X = scaler.transform(pp[v][FEATURES_FOR_CLUSTERING]) * wv
    if method == "hdbscan":
        lab, _ = hdbscan_lib.approximate_predict(clusterer, X)
    else:
        lab = clusterer.predict(X)
    return dfc, lab


for i, rd in enumerate(retrain):
    cutoff = rd - pd.Timedelta(days=RALLY_WINDOW_DAYS)
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

    # test penceresi: epoch'un ileri yılı
    test_df = df_feat_pd[(df_feat_pd['Date'] >= rd) & (df_feat_pd['Date'] < bounds[i+1])
                         & df_feat_pd['future_max_gain'].notna()].copy()
    test_df = test_df[(test_df['mom_120'] < 0.70) & test_df['mom_120'].notna()]

    for method, clusterer in [
        ("kmeans", None), ("hdbscan", None)
    ]:
        if method == "hdbscan":
            cl = hdbscan_lib.HDBSCAN(min_cluster_size=60, min_samples=5,
                                     prediction_data=True).fit(X)
        else:
            bk, bs = 3, -1
            Xs = X[:5000] if len(X) > 5000 else X
            for k in range(3, 6):
                km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(Xs)
                s = silhouette_score(Xs, km.labels_)
                if s > bs: bs, bk = s, k
            cl = KMeans(n_clusters=bk, random_state=42, n_init=10).fit(X)

        tr_df, tr_lab = assign(method, cl, scaler, q_low, q_high, train_df)
        te_df, te_lab = assign(method, cl, scaler, q_low, q_high, test_df)
        tr_wr = wr_by_cluster(tr_df, tr_lab)
        te_wr = wr_by_cluster(te_df, te_lab)

        print(f"\nEpoch {i+1} ({rd.date()})  —  {method.upper()}")
        print(f"  {'küme':>5} {'eğitim-WR':>11} {'eğitim-n':>9} {'test-WR':>9} {'test-n':>8}  not")
        for c in sorted(set(tr_wr) | set(te_wr)):
            tn, tw = tr_wr.get(c, (0, 0.0))
            en, ew = te_wr.get(c, (0, 0.0))
            note = ""
            if tw >= 0.65:
                note = "YÜKSEK-WR →" + ("  KORUDU" if ew >= 0.60 else "  GERİLEDİ ⚠️")
            print(f"  {c:>5} {tw:>10.1%} {tn:>9,} {ew:>8.1%} {en:>8,}  {note}")
