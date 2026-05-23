"""
generation_value.py — B/C üretim katmanı değerli mi?
=====================================================
B/C geçiş sinyallerinin ralli oranı (signal_eval: genel %44.7) — rastgele bir
setup'ın ralli oranından (koşulsuz taban) yüksek mi?

Koşulsuz taban: her epoch test penceresindeki TÜM uygun setup'ların ralli oranı.
B/C üretim katmanı seçiciyse: B/C oranı >> koşulsuz oran.
"""
import warnings
warnings.filterwarnings("ignore")
import polars as pl
import pandas as pd

from backtest_engine import RETRAIN_DATES, BACKTEST_END, FEAT_PATH
from common import compute_forward_relative_gain
from model_core import (RALLY_WINDOW_DAYS, RALLY_GAIN_THRESHOLD,
                        FEATURES_FOR_CLUSTERING)

# signal_eval.py çıktısından — B/C sinyallerinin epoch bazlı ralli oranı.
# Güncelleme: GMM clustering + XU100-göreli hedef (RALLY_GAIN_THRESHOLD=0.50)
# signal_eval.py ile taban ralli oranı; generation_value.py ep = i+1 (0-index).
BC_RATE = {2: 0.798, 3: 0.473, 4: 0.257, 5: 0.423, 6: 0.384, 7: 0.379, 8: 0.364}
BC_OVERALL = 0.439

df_feat = pl.read_parquet(FEAT_PATH).with_columns(pl.col("Date").cast(pl.Datetime("ms")))
df = compute_forward_relative_gain(df_feat, RALLY_WINDOW_DAYS).to_pandas()
df['Date'] = pd.to_datetime(df['Date'])

safe_cutoff = df['Date'].max() - pd.Timedelta(days=RALLY_WINDOW_DAYS)
bounds = list(RETRAIN_DATES) + [BACKTEST_END + pd.Timedelta(days=1)]

print(f"{'Epoch':>6} {'pencere':>24} {'koşulsuz':>10} {'B/C sinyal':>11} {'lift':>7}  not")
print("=" * 76)

tot_uncond_n = tot_uncond_r = 0
for i, rd in enumerate(RETRAIN_DATES):
    ep = i + 1
    if ep not in BC_RATE:
        continue
    win_lo, win_hi = rd, bounds[i + 1]
    uni = df[(df['Date'] >= win_lo) & (df['Date'] < win_hi) & (df['Date'] <= safe_cutoff)]
    uni = uni.dropna(subset=FEATURES_FOR_CLUSTERING + ['future_max_gain'])
    if len(uni) == 0:
        continue
    uncond = (uni['future_max_gain'] >= RALLY_GAIN_THRESHOLD).mean()
    tot_uncond_n += len(uni)
    tot_uncond_r += int((uni['future_max_gain'] >= RALLY_GAIN_THRESHOLD).sum())
    bc = BC_RATE[ep]
    lift = bc / uncond if uncond > 0 else float('nan')
    note = "B/C seçici ✓" if lift > 1.15 else ("~rastgele" if lift < 1.05 else "")
    print(f"{ep:>6} {str(win_lo.date())+'→'+str(win_hi.date()):>24} "
          f"{uncond:>9.1%} {bc:>10.1%} {lift:>6.2f}x  {note}")

uncond_overall = tot_uncond_r / tot_uncond_n
print("=" * 76)
print(f"{'GENEL':>6} {'':>24} {uncond_overall:>9.1%} {BC_OVERALL:>10.1%} "
      f"{BC_OVERALL/uncond_overall:>6.2f}x")
print()
print(f"Koşulsuz taban = rastgele bir uygun setup'ın {RALLY_WINDOW_DAYS}g içinde "
      f"≥%{RALLY_GAIN_THRESHOLD*100:.0f} yapma oranı.")
print(f"B/C sinyal = clustering'in ürettiği geçiş sinyallerinin ralli oranı.")
print(f"lift > 1 → B/C üretim katmanı rastgeleden seçici.")
