"""
Twin Divergence — Post-Entry Trajectory Monitörü
================================================
Pozisyona girildikten sonra hissenin gerçek trajectory'sini,
success_twins ve failure_twins medyan trajectory'leriyle 3-kanal DTW
ile karşılaştırır. Failure'a yakın eğilim → erken çıkış sinyali.

Aynı 3-kanal mantığı twin_matcher.get_base_curve ile özdeş:
  Kanal 1 — z-score(Pclose)
  Kanal 2 — z-score(log1p(Vlot))
  Kanal 3 — z-score(range / Pclose)

Tek fark: pencere PRE-setup değil, POST-entry.
"""
import polars as pl
import pandas as pd
import numpy as np
from dtaidistance import dtw_ndim

from twin_matcher import z_score_normalize, DTW_WINDOW, MAX_DISTANCE_NDIM


def get_post_curve(df_market, ticker, start_date, n_days):
    """
    start_date'TEN SONRAKİ n_days günlük 3-kanal eğriyi döndürür.
    twin_matcher.get_base_curve'ün post-versiyonu.

    Returns: np.ndarray shape (n_days, 3) dtype np.double, veya None
             (yeterli veri yoksa).
    """
    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date)

    subset = df_market.filter(
        (pl.col("Ticker") == ticker) &
        (pl.col("Date") > start_date)
    ).head(n_days)

    if len(subset) < max(n_days * 0.9, 5):
        return None

    prices  = subset["Pclose"].to_numpy().astype(float)
    volumes = np.log1p(subset["Vlot"].to_numpy().astype(float))
    ranges  = ((subset["Phigh"] - subset["Plow"]) / subset["Pclose"]).to_numpy().astype(float)

    return np.column_stack([
        z_score_normalize(prices),
        z_score_normalize(volumes),
        z_score_normalize(ranges),
    ]).astype(np.double)


def compute_twin_median_post_curves(twins_df, df_market, window=60):
    """
    twins_df içindeki her ikiz için post-setup trajectory hesapla
    (start_date = ikizin setup tarihi). Sonra Success ve Failure grupları için
    per-channel, per-day medyan al.

    twins_df: kolonları {'Ticker', 'Date', 'Outcome'} olan DataFrame
              ('Outcome' ∈ {'Success', 'Failure'})

    Returns:
        success_median: np.ndarray (window, 3) — eğer n_success >= 1
        failure_median: np.ndarray (window, 3) — eğer n_failure >= 1
        n_success:      kullanılabilir başarı ikiz sayısı
        n_failure:      kullanılabilir hüsran ikiz sayısı

        Yetersiz veri → ilgili median None.
    """
    success_curves = []
    failure_curves = []

    for _, twin in twins_df.iterrows():
        curve = get_post_curve(df_market, twin['Ticker'], twin['Date'], window)
        if curve is None or len(curve) < window:
            continue
        if twin['Outcome'] == 'Success':
            success_curves.append(curve)
        elif twin['Outcome'] == 'Failure':
            failure_curves.append(curve)

    def _median_stack(curves):
        if not curves:
            return None
        arr = np.stack(curves, axis=0)  # (n_twins, window, 3)
        return np.median(arr, axis=0).astype(np.double)  # (window, 3)

    success_median = _median_stack(success_curves)
    failure_median = _median_stack(failure_curves)

    return success_median, failure_median, len(success_curves), len(failure_curves)


def twin_divergence_score(candidate_curve, success_median, failure_median):
    """
    candidate_curve'ün success_median ve failure_median'a 3-kanal DTW mesafesi.

    Score formülü:
        score = (d_failure - d_success) / (d_failure + d_success + 1e-9)
        ∈ (-1, +1)
        > 0  → success'e yakın (HOLD)
        = 0  → tam median (sınır)
        < 0  → failure'a yakın (EXIT sinyali)

    Returns: (score, dist_to_success, dist_to_failure)
             Curve veya median None → (None, None, None)
    """
    if candidate_curve is None or success_median is None or failure_median is None:
        return None, None, None

    # DTW eşit uzunluk gerektirmez ama z-score zaten yapıldı.
    # Aynı twin_matcher parametreleri.
    d_s = dtw_ndim.distance_fast(
        candidate_curve, success_median,
        window=DTW_WINDOW, max_dist=MAX_DISTANCE_NDIM,
    )
    d_f = dtw_ndim.distance_fast(
        candidate_curve, failure_median,
        window=DTW_WINDOW, max_dist=MAX_DISTANCE_NDIM,
    )

    # max_dist aşıldıysa dtaidistance inf döndürür → büyük bir sayıya yuvarla
    if not np.isfinite(d_s):
        d_s = MAX_DISTANCE_NDIM
    if not np.isfinite(d_f):
        d_f = MAX_DISTANCE_NDIM

    score = (d_f - d_s) / (d_f + d_s + 1e-9)
    return score, d_s, d_f


def compute_gain_curves(twins_df, df_market, window=60):
    """
    Raporlama için: her ikizin post-setup KÜMÜLATİF GAIN trajectory'sini
    hesapla, Success/Failure grupları için per-day medyan al.

    Trajectory'nin yorumlanabilir bir özetidir (DTW skoru ile birlikte
    rapora konabilir).

    Returns:
        success_gain_median: np.ndarray (window,) — her gün için medyan gain
        failure_gain_median: np.ndarray (window,) — her gün için medyan gain
        n_success, n_failure
    """
    def _gain_curve(ticker, start_date, n_days):
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        sub = df_market.filter(
            (pl.col("Ticker") == ticker) &
            (pl.col("Date") > start_date)
        ).head(n_days)
        if len(sub) < n_days:
            return None
        prices = sub["Pclose"].to_numpy().astype(float)
        return prices / prices[0] - 1.0

    s_gains, f_gains = [], []
    for _, twin in twins_df.iterrows():
        g = _gain_curve(twin['Ticker'], twin['Date'], window)
        if g is None or len(g) < window:
            continue
        if twin['Outcome'] == 'Success':
            s_gains.append(g)
        elif twin['Outcome'] == 'Failure':
            f_gains.append(g)

    s_med = np.median(np.stack(s_gains, axis=0), axis=0) if s_gains else None
    f_med = np.median(np.stack(f_gains, axis=0), axis=0) if f_gains else None
    return s_med, f_med, len(s_gains), len(f_gains)
