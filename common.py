"""
Arkhimedes — Paylaşılan Yardımcı Fonksiyonlar
==============================================
Birden fazla dosyada tekrarlanan mantıkların tek yerden import edilen versiyonu.

İçerik:
- compute_atr20:         ATR(20) hesabı (Wilder TR bazlı, basit rolling mean)
- make_market_pivots:    Date × Ticker pivot tabloları (Pclose/Popen/ATR vs.)
- safe_pivot_lookup:     try/except KeyError yerine güvenli erişim
"""
import numpy as np
import pandas as pd
import polars as pl
from config import ATR_PERIOD, DB_PATH


def compute_forward_relative_gain(df_pl, window, market_path=None):
    """
    İleri-dönük XU100-GÖRELİ kazanç — 'ralli' tanımının tek kaynağı.

    rel = Pclose / XU100_kapanış. future_max_gain = önümüzdeki `window` gün içinde
    rel'in ulaştığı en yüksek değer / mevcut rel − 1. Yani: "en iyi noktada hisse
    XU100'ün yüzde kaç önündeydi". Mutlak nominal kazanç DEĞİL — enflasyon-nötr.

    Args:
        df_pl: polars DataFrame — Ticker, Date, Pclose içermeli
        window: ileri pencere (gün)
    Returns:
        df_pl + 'future_max_gain' (XU100-göreli; kolon adı geriye uyumluluk için aynı)
    """
    if market_path is None:
        market_path = DB_PATH
    xu = (pl.read_parquet(market_path)
            .filter(pl.col("Ticker") == "XU100")
            .select([pl.col("Date").cast(pl.Datetime("ms")),
                     pl.col("Pclose").alias("_xu100")])
            .unique(subset=["Date"], keep="first")
            .sort("Date"))

    df_pl = df_pl.with_columns(pl.col("Date").cast(pl.Datetime("ms")))
    df_pl = df_pl.join(xu, on="Date", how="left")
    df_pl = df_pl.with_columns((pl.col("Pclose") / pl.col("_xu100")).alias("_rel"))

    df_pl = (
        df_pl.lazy()
        .sort(["Ticker", "Date"])
        .group_by("Ticker", maintain_order=True)
        .agg([
            pl.all(),
            pl.col("_rel").reverse()
              .rolling_max(window_size=window, min_periods=1)
              .reverse().alias("_future_max_rel")
        ])
        .explode(pl.all().exclude("Ticker"))
        .collect()
    )
    return df_pl.with_columns(
        ((pl.col("_future_max_rel") / pl.col("_rel")) - 1).alias("future_max_gain")
    ).drop(["_xu100", "_rel", "_future_max_rel"])


def compute_atr20(df_market_pd, period=ATR_PERIOD, min_periods=5):
    """
    ATR(period) ekle (Wilder True Range bazlı, basit rolling mean).
    Beklenen sütunlar: Ticker, Date, Phigh, Plow, Pclose.

    Returns: aynı DataFrame + 'ATR20' (ya da period'a göre kolon ismi 'ATR{period}')
    """
    df = df_market_pd.sort_values(["Ticker", "Date"]).copy()
    df['PrevClose'] = df.groupby('Ticker')['Pclose'].shift(1)
    df['TR'] = np.maximum.reduce([
        (df['Phigh'] - df['Plow']).values,
        (df['Phigh'] - df['PrevClose']).abs().values,
        (df['Plow']  - df['PrevClose']).abs().values,
    ])
    col_name = f'ATR{period}'
    df[col_name] = df.groupby('Ticker')['TR'].transform(
        lambda s: s.rolling(period, min_periods=min_periods).mean()
    )
    return df.drop(columns=['PrevClose', 'TR'])


def compute_adv20(df_market_pd, period=20, min_periods=5):
    """
    ADV20 ekle — 20-günlük ortalama günlük TL işlem hacmi.
    Günlük TL hacim = Vlot × Pvwap (VWAP yoksa Pclose).
    Beklenen sütunlar: Ticker, Date, Vlot, (Pvwap | Pclose).

    Returns: aynı DataFrame + 'ADV20'
    """
    df = df_market_pd.sort_values(["Ticker", "Date"]).copy()
    price = df['Pvwap'] if 'Pvwap' in df.columns else df['Pclose']
    df['_tl_vol'] = df['Vlot'].astype(float) * price.astype(float)
    df['ADV20'] = df.groupby('Ticker')['_tl_vol'].transform(
        lambda s: s.rolling(period, min_periods=min_periods).mean()
    )
    return df.drop(columns=['_tl_vol'])


def make_market_pivots(df_market_pd, trading_days=None, fields=('Pclose', 'Popen', 'ATR20')):
    """
    Date × Ticker pivot tabloları üret. Çoklu pivot ihtiyacında yeniden kullanım.

    Args:
        df_market_pd: pandas DataFrame, Date + Ticker + ilgili field'ları içermeli
        trading_days: opsiyonel filtre (sadece bu günler dahil edilir)
        fields: pivot'a çevrilecek sütun isimleri

    Returns: dict[field] = DataFrame[Date × Ticker]
    """
    if trading_days is not None:
        df_market_pd = df_market_pd[df_market_pd['Date'].isin(trading_days)]
    pivots = {}
    for f in fields:
        if f not in df_market_pd.columns:
            continue
        pivots[f] = df_market_pd.pivot_table(index='Date', columns='Ticker', values=f)
    return pivots


def safe_pivot_lookup(pivot, date, ticker=None, default=np.nan):
    """
    Pivot tablosundan güvenli okuma. KeyError'u yutar, default döndürür.
    ticker=None ise tüm satırı (Series), aksi halde tek hücreyi döndürür.
    """
    try:
        row = pivot.loc[date]
        if ticker is None:
            return row
        v = row.get(ticker, default)
        return default if pd.isna(v) else v
    except KeyError:
        if ticker is None:
            return pd.Series(dtype=float)
        return default
