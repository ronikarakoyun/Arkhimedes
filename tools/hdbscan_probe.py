import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""
HDBSCAN parametre probu — son epoch'ta noise oranı + küme kalitesi.
Tam backtest yapmadan hızlı tarama: hangi (min_cluster_size, min_samples)
makul kapsam (düşük noise) + iyi küme ayrışması veriyor.
"""
import warnings
warnings.filterwarnings("ignore")
import polars as pl
import pandas as pd
import numpy as np

import backtest_engine as be
from model_core import (apply_preprocessing, FEATURES_FOR_CLUSTERING, FEATURE_WEIGHTS,
                        RALLY_WINDOW_DAYS, RALLY_GAIN_THRESHOLD, SECOND_LEG_COOLDOWN,
                        _isolate_representative_setups, _remove_second_leg_rallies)
from sklearn.preprocessing import RobustScaler
import hdbscan as hdbscan_lib

# ── Veri (son epoch'a kadar) ──
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

retrain_date = list(be.RETRAIN_DATES)[-1]   # son epoch — en çok veri
cutoff = retrain_date - pd.Timedelta(days=RALLY_WINDOW_DAYS)
train_df = df_feat_pd[df_feat_pd['Date'] <= cutoff].copy()
train_df = train_df[(train_df['mom_120'] < 0.70) & train_df['mom_120'].notna()
                    & train_df['future_max_gain'].notna()]
success_pd = train_df[train_df['future_max_gain'] >= RALLY_GAIN_THRESHOLD].copy()
# train_at_date ile aynı havuz: temsilci izolasyon + ikinci-ayak temizliği
success_pl = _isolate_representative_setups(pl.from_pandas(success_pd))
success_pl = _remove_second_leg_rallies(success_pl, SECOND_LEG_COOLDOWN)
success_pd = success_pl.drop_nulls(subset=FEATURES_FOR_CLUSTERING).to_pandas()

q_low  = success_pd[FEATURES_FOR_CLUSTERING].quantile(0.01).to_dict()
q_high = success_pd[FEATURES_FOR_CLUSTERING].quantile(0.99).to_dict()
spp = apply_preprocessing(success_pd, FEATURES_FOR_CLUSTERING, q_low, q_high)
X = RobustScaler().fit_transform(spp[FEATURES_FOR_CLUSTERING])
X = X * np.array([FEATURE_WEIGHTS[f] for f in FEATURES_FOR_CLUSTERING])
print(f"Son epoch ({retrain_date.date()}) başarı havuzu: {len(X)} nokta\n")

print(f"{'mcs':>5} {'ms':>4} {'n_küme':>7} {'noise%':>8} {'küme win-rate aralığı':>30}")
for mcs in [15, 30, 60, 100]:
    for ms in [1, 5]:
        c = hdbscan_lib.HDBSCAN(min_cluster_size=mcs, min_samples=ms,
                                prediction_data=True).fit(X)
        labels = c.labels_
        uniq = sorted(set(labels) - {-1})
        noise = (labels == -1).mean() * 100
        # küme başarı oranları (başarı havuzundaki üyelik — sadece kalite göstergesi)
        # not: bu havuzun tamamı 'başarı', win-rate için geçiş analizi gerekir;
        # burada sadece küme sayısı + boyut dağılımı + noise raporlanır
        sizes = [int((labels == u).sum()) for u in uniq]
        srange = f"{min(sizes)}–{max(sizes)} ({len(uniq)} küme)" if sizes else "—"
        print(f"{mcs:>5} {ms:>4} {len(uniq):>7} {noise:>7.1f}% {srange:>30}")
