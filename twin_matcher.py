import polars as pl
import pandas as pd
import numpy as np
from dtaidistance import dtw, dtw_ndim
import os
import warnings

warnings.filterwarnings('ignore')

from config import (
    DB_PATH, SUCCESS_POOL_PATH, FAILURE_POOL_PATH,
    TWIN_OUTPUT_DIR as OUTPUT_DIR,
)

# DTW parametreleri
DTW_WINDOW        = 20    # Sakoe-Chiba band — maksimum zaman kayması (adım)
MAX_DISTANCE_NDIM = 60.0  # 3 kanallı eşik (~2.9x tek kanal, kalibre edildi)


def z_score_normalize(series):
    mean = np.mean(series)
    std = np.std(series)
    if std == 0:
        return np.zeros(len(series))
    return (series - mean) / std


def get_base_curve(df_market, ticker, end_date, window=120):
    """
    120 günlük 3 kanallı DTW eğrisini döndürür — shape: (T, 3), dtype: np.double

    Kanal 1 — Fiyat  : Z-score(Pclose)
    Kanal 2 — Hacim  : Z-score(log1p(Vlot))
    Kanal 3 — Aralık : Z-score((Phigh - Plow) / Pclose)
    """
    if isinstance(end_date, str):
        end_date = pd.to_datetime(end_date)

    subset = df_market.filter(
        (pl.col("Ticker") == ticker) &
        (pl.col("Date") <= end_date)
    ).tail(window)

    if len(subset) < window * 0.9:
        return None

    prices  = subset["Pclose"].to_numpy().astype(float)
    volumes = np.log1p(subset["Vlot"].to_numpy().astype(float))
    ranges  = ((subset["Phigh"] - subset["Plow"]) / subset["Pclose"]).to_numpy().astype(float)

    return np.column_stack([
        z_score_normalize(prices),
        z_score_normalize(volumes),
        z_score_normalize(ranges),
    ]).astype(np.double)


def precompute_pool_curves(df_market, df_success, df_failure, window=120):
    """
    Bilgi tabanındaki tüm havuz kayıtları için Z-Score eğrilerini önceden hesaplar.
    Döndürülen liste: [{'Ticker', 'Date', 'Outcome', 'Curve'}, ...]
    Bu fonksiyon bir kez çağrılıp sonucu twin_matcher'a geçirilmeli.
    """
    entries = []
    total = len(df_success) + len(df_failure)
    print(f"   ⏳ {total} havuz kaydı için eğriler ön hesaplanıyor...")

    for pool_df, pool_name in [(df_success, 'Success'), (df_failure, 'Failure')]:
        for _, row in pool_df.iterrows():
            curve = get_base_curve(df_market, row['Ticker'], row['Date'], window)
            if curve is not None:
                entries.append({
                    'Ticker': row['Ticker'],
                    'Date': row['Date'],
                    'Outcome': pool_name,
                    'Curve': curve,
                })

    print(f"   ✅ {len(entries)} geçerli eğri hazırlandı.")
    return entries


def twin_matcher(target_ticker, target_date=None, df_market=None, pool_cache=None):
    """
    Hedef hisse için 3 kanallı DTW ikizlerini bulur.

    Parametreler
    ------------
    target_ticker : str
    target_date   : date/str, None ise son tarih kullanılır
    df_market     : Polars DataFrame — verilmezse diskten okunur (yavaş)
    pool_cache    : precompute_pool_curves() çıktısı — verilmezse diskten okunur ve örneklenir

    Kanallar: Fiyat (Z) | Hacim log (Z) | Aralık/Pclose (Z)
    """
    print(f"🧬 3-Kanal DTW Twin Matcher: {target_ticker}  "
          f"(window={DTW_WINDOW}, max_dist={MAX_DISTANCE_NDIM})")

    if df_market is None:
        if not os.path.exists(DB_PATH):
            print("❌ market_db.parquet bulunamadı.")
            return None
        df_market = pl.read_parquet(DB_PATH)

    if target_date is None:
        target_date = df_market.filter(pl.col("Ticker") == target_ticker)["Date"].max()

    print(f"🎯 Hedef: {target_ticker} (Tarih: {target_date.date() if hasattr(target_date, 'date') else target_date})")

    target_curve = get_base_curve(df_market, target_ticker, target_date)
    if target_curve is None:
        print(f"❌ {target_ticker} için yeterli tarihsel veri yok.")
        return None

    # Havuz tarama
    if pool_cache is not None:
        all_matches = _scan_cache(target_ticker, target_date, target_curve, pool_cache)
    else:
        all_matches = _scan_from_disk(target_ticker, target_date, target_curve, df_market)

    if not all_matches:
        print("⚠️ Yakın geometrik ikiz bulunamadı.")
        return None

    df_matches = pd.DataFrame(all_matches).sort_values('DTW_Distance', ascending=True)

    TOP_K = 20
    top_matches = df_matches.head(TOP_K)

    success_count = len(top_matches[top_matches['Outcome'] == 'Success'])
    failure_count = len(top_matches[top_matches['Outcome'] == 'Failure'])
    win_rate = (success_count / len(top_matches)) * 100

    print(f"✅ Başarılı İkiz: {success_count}  |  💀 Başarısız: {failure_count}  |  🚀 Win-Rate: %{win_rate:.1f}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"{target_ticker}_DTW_Ikiz_Raporu.csv")
    df_matches.to_csv(out_path, index=False)

    return {'WinRate': win_rate, 'TopMatches': top_matches}


def _scan_cache(target_ticker, target_date, target_curve, pool_cache):
    """Ön hesaplanmış eğri cache'i üzerinden 3 kanallı DTW taraması yapar."""
    matches = []
    target_date_pd = pd.to_datetime(target_date)

    for entry in pool_cache:
        if (entry['Ticker'] == target_ticker and
                abs((pd.to_datetime(entry['Date']) - target_date_pd).days) < 60):
            continue

        distance = dtw_ndim.distance_fast(
            target_curve, entry['Curve'],
            window=DTW_WINDOW,
            max_dist=MAX_DISTANCE_NDIM,
        )
        if distance < MAX_DISTANCE_NDIM:
            matches.append({
                'Ticker':       entry['Ticker'],
                'Date':         str(entry['Date'])[:10],
                'DTW_Distance': round(distance, 4),
                'Outcome':      entry['Outcome'],
            })
    return matches


def _scan_from_disk(target_ticker, target_date, target_curve, df_market):
    """Cache yoksa diskten okuyup tarar (yedek yol)."""
    if not os.path.exists(SUCCESS_POOL_PATH) or not os.path.exists(FAILURE_POOL_PATH):
        print("❌ Bilgi tabanı parquet dosyaları eksik.")
        return []

    df_success = pl.read_parquet(SUCCESS_POOL_PATH).to_pandas()
    df_failure = pl.read_parquet(FAILURE_POOL_PATH).to_pandas()
    target_date_pd = pd.to_datetime(target_date)
    matches = []

    for pool_df, pool_name in [(df_success, 'Success'), (df_failure, 'Failure')]:
        print(f"   -> {pool_name} havuzu taranıyor ({len(pool_df)} kayıt)...")
        for _, row in pool_df.iterrows():
            if (row['Ticker'] == target_ticker and
                    abs((pd.to_datetime(row['Date']) - target_date_pd).days) < 60):
                continue
            curve = get_base_curve(df_market, row['Ticker'], row['Date'])
            if curve is None:
                continue
            distance = dtw_ndim.distance_fast(
                target_curve, curve,
                window=DTW_WINDOW,
                max_dist=MAX_DISTANCE_NDIM,
            )
            if distance < MAX_DISTANCE_NDIM:
                matches.append({
                    'Ticker':       row['Ticker'],
                    'Date':         str(row['Date'])[:10],
                    'DTW_Distance': round(distance, 4),
                    'Outcome':      pool_name,
                })
    return matches


if __name__ == "__main__":
    twin_matcher("BOSSA")
    twin_matcher("HUNER")
