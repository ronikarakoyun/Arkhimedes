import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""
FINAL_P_THRESHOLD Kalibrasyon Scripti
======================================
Epoch eğitim + sinyal precompute'u BİR KEZ çalıştırır,
ardından simulate_trading'i 7 farklı threshold için tekrar çağırır.

Thresholds: None (filtre kapalı), 0.30, 0.35, 0.40, 0.45, 0.50, 0.55
"""
import os
import sys
import pickle
import pandas as pd
import polars as pl
import numpy as np

# Proje klasörü
sys.path.insert(0, os.path.dirname(__file__))

from backtest_engine import (
    train_at_date, precompute_signals_for_epoch, simulate_trading,
    compute_metrics, build_benchmark_nav, precompute_pool_curves,
    compute_atr20,
    BACKTEST_START, BACKTEST_END, TRADING_START, RETRAIN_DATES,
    DB_PATH, FEAT_PATH, RALLY_WINDOW_DAYS, INITIAL_CAPITAL,
    TWIN_DIV_ENABLED, TWIN_DIV_EXIT_THRESHOLD, TWIN_DIV_WINDOW,
)
from macro_engine import load_macro_features
from fundamental_engine import load_fundamentals

CACHE_PATH = "calibration_cache.pkl"
THRESHOLDS = [0.0, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55]
THRESHOLD_LABELS = ["KAPALI", "0.30", "0.35", "0.40", "0.45", "0.50", "0.55"]


def run_epoch_loop():
    """Epoch eğitim + sinyal + twin pool precompute. Cache'e kaydeder."""
    print("=" * 70)
    print("📥 Veri yükleniyor...")
    df_market = pl.read_parquet(DB_PATH).with_columns(pl.col("Date").cast(pl.Datetime("ms")))
    df_market_pd = df_market.to_pandas()
    df_market_pd['Date'] = pd.to_datetime(df_market_pd['Date'])

    df_feat = pl.read_parquet(FEAT_PATH).with_columns(pl.col("Date").cast(pl.Datetime("ms")))

    print("⏳ future_max_gain hesaplanıyor...")
    df_outcomes = (
        df_feat.lazy().sort(["Ticker", "Date"]).group_by("Ticker").agg([
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

    trading_days = sorted(df_market_pd[df_market_pd['Ticker'] == 'XU100']['Date'].unique())
    trading_days = [d for d in trading_days if BACKTEST_START <= d <= BACKTEST_END]
    print(f"   {len(trading_days):,} trading günü")

    print("\n📊 CombinedFilter verileri yükleniyor...")
    cf_macro_df = load_macro_features()
    cf_macro_df['Date'] = pd.to_datetime(cf_macro_df['Date'])
    cf_fund_df = load_fundamentals()
    print(f"   ✅ macro {cf_macro_df.shape}, fund {cf_fund_df.shape}")

    df_market_pl_for_twin = (
        pl.read_parquet(DB_PATH).with_columns(pl.col("Date").cast(pl.Date))
        if TWIN_DIV_ENABLED else None
    )

    print(f"\n🔄 Walk-Forward Retrain ({len(RETRAIN_DATES)} epoch)...")
    all_signals = []
    prev_clusters = {}
    epoch_pool_caches = []
    epoch_boundaries = list(RETRAIN_DATES) + [BACKTEST_END + pd.Timedelta(days=1)]

    for i, retrain_date in enumerate(RETRAIN_DATES):
        epoch_start = retrain_date
        epoch_end = epoch_boundaries[i + 1]
        print(f"\n   Epoch {i+1}/{len(RETRAIN_DATES)}: {retrain_date.date()}")

        model_bundle = train_at_date(df_feat_pd, retrain_date,
                                     macro_df=cf_macro_df, fund_df=cf_fund_df)
        if model_bundle is None:
            epoch_pool_caches.append(None)
            continue

        signals_df, prev_clusters = precompute_signals_for_epoch(
            model_bundle, df_feat_pd, epoch_start, epoch_end, prev_clusters, epoch_idx=i,
            macro_df=cf_macro_df, fund_df=cf_fund_df,
        )
        print(f"   {len(signals_df):,} sinyal")
        if len(signals_df) > 0:
            all_signals.append(signals_df)

        if TWIN_DIV_ENABLED:
            success_pool = model_bundle['twin_pool_success']
            failure_pool = model_bundle['twin_pool_failure']
            print(f"   🎯 Twin pool: {len(success_pool):,}s + {len(failure_pool):,}f → eğriler...")
            pool_cache = precompute_pool_curves(df_market_pl_for_twin, success_pool, failure_pool, window=120)
            epoch_pool_caches.append(pool_cache)
        else:
            epoch_pool_caches.append(None)

    all_signals_df = pd.concat(all_signals, ignore_index=True) if all_signals else pd.DataFrame()
    print(f"\n📊 Toplam: {len(all_signals_df):,} sinyal")

    # ATR precompute (simulate_trading iç hesaplamasından önce bir kez yap)
    df_market_pd = compute_atr20(df_market_pd)

    cache = {
        'all_signals_df': all_signals_df,
        'df_market_pd': df_market_pd,
        'trading_days': trading_days,
        'epoch_pool_caches': epoch_pool_caches,
        'df_market_pl_for_twin': df_market_pl_for_twin,
    }
    with open(CACHE_PATH, 'wb') as f:
        pickle.dump(cache, f)
    print(f"💾 Cache kaydedildi: {CACHE_PATH}")
    return cache


def run_calibration(cache):
    all_signals_df = cache['all_signals_df']
    df_market_pd   = cache['df_market_pd']
    trading_days   = cache['trading_days']
    epoch_pool_caches = cache['epoch_pool_caches']
    df_market_pl   = cache['df_market_pl_for_twin']

    bench_df = build_benchmark_nav(df_market_pd, trading_days)

    rows = []
    for thr, label in zip(THRESHOLDS, THRESHOLD_LABELS):
        print(f"\n▶ Threshold={label} çalışıyor...")
        trades_df, daily_nav_df = simulate_trading(
            all_signals_df, df_market_pd.copy(), trading_days,
            df_market_pl=df_market_pl,
            epoch_pool_caches=epoch_pool_caches,
            p_threshold=thr,
        )
        m = compute_metrics(daily_nav_df, trades_df, bench_df)
        veto_count = (
            int((all_signals_df['P_rally'].dropna() < thr).sum())
            if thr > 0.0 and 'P_rally' in all_signals_df.columns
            else 0
        )
        rows.append({
            'Threshold': label,
            'CAGR%': round(m['CAGR'] * 100, 1),
            'Sharpe': round(m['Sharpe'], 2),
            'MaxDD%': round(m['MaxDD'] * 100, 1),
            'Calmar': round(m['Calmar'], 2),
            'WinRate%': round(m['WinRate'] * 100, 1),
            'Trades': m['TotalTrades'],
            'Vetoed': veto_count,
        })
        print(f"   CAGR={m['CAGR']*100:.1f}%  Sharpe={m['Sharpe']:.2f}  "
              f"MaxDD={m['MaxDD']*100:.1f}%  Trades={m['TotalTrades']}  Veto={veto_count}")

    print("\n" + "=" * 80)
    print("📊 KALİBRASYON SONUÇLARI")
    print("=" * 80)
    result_df = pd.DataFrame(rows)
    print(result_df.to_string(index=False))

    # Markdown rapor
    os.makedirs("Gunluk_Raporlar", exist_ok=True)
    md = "# CombinedFilter FINAL_P_THRESHOLD Kalibrasyon\n\n"
    # Manuel markdown tablo (tabulate gerekmez)
    cols = list(result_df.columns)
    md += "| " + " | ".join(cols) + " |\n"
    md += "| " + " | ".join(["---"] * len(cols)) + " |\n"
    for _, row in result_df.iterrows():
        md += "| " + " | ".join(str(v) for v in row) + " |\n"
    md += "\n\n**Seçim kriteri:** MaxDD kötüleşmeden CAGR veya Sharpe en yüksek.\n"
    out = "Gunluk_Raporlar/threshold_calibration.md"
    with open(out, 'w') as f:
        f.write(md)
    print(f"\n📄 Rapor: {out}")
    return result_df


if __name__ == "__main__":
    if os.path.exists(CACHE_PATH):
        print(f"💾 Cache bulundu: {CACHE_PATH}, epoch loop atlanıyor...")
        with open(CACHE_PATH, 'rb') as f:
            cache = pickle.load(f)
    else:
        cache = run_epoch_loop()

    run_calibration(cache)
