import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import polars as pl
import numpy as np
import os
import warnings

warnings.filterwarnings('ignore')

# ─── Configuration ────────────────────────────────────────────────────────────
from config import DB_PATH, FEATURES_PATH, SECTOR_MAP_PATH
BENCHMARK_TICKER = "XU100"

def calculate_features():
    print("🚀 Feature Engine Starting...")

    if not os.path.exists(DB_PATH):
        print(f"❌ Error: {DB_PATH} not found. Run data_engine.py first.")
        return

    # 5. Load Sector Map
    if os.path.exists(SECTOR_MAP_PATH):
        sector_map_df = pl.read_csv(SECTOR_MAP_PATH)
    else:
        print(f"⚠️  {SECTOR_MAP_PATH} bulunamadı, sektör feature'ları atlanıyor.")
        sector_map_df = None

    # 1. Load Market Data
    df = pl.read_parquet(DB_PATH)
    
    # 2. Extract Benchmark (XU100) for Relative Metrics
    print(f"📊 Calculating Benchmark ({BENCHMARK_TICKER}) Volatility...")
    # Polars Date handling - ensure it's Date type
    df = df.with_columns(pl.col("Date").cast(pl.Date))
    
    benchmark_df  = df.filter(pl.col("Ticker") == BENCHMARK_TICKER).sort("Date")
    usdtry_df     = df.filter(pl.col("Ticker") == "USDTRY").sort("Date")
    xbank_df      = df.filter(pl.col("Ticker") == "XBANK").sort("Date")
    xusin_df      = df.filter(pl.col("Ticker") == "XUSIN").sort("Date")
    
    # Calculate Benchmark metrics: volatility + regime features
    benchmark_df = benchmark_df.with_columns([
        pl.col("Pclose").pct_change().alias("returns")
    ]).with_columns([
        pl.col("returns").rolling_std(window_size=120).alias("bm_vol_120"),
        # XU100 rejim featureları
        (pl.col("Pclose") / pl.col("Pclose").shift(60) - 1).alias("xu100_mom_60"),
        pl.col("Pclose").rolling_mean(window_size=200).alias("_xu100_ma200"),
        pl.col("Pclose").rolling_max(window_size=252).alias("_xu100_peak_252"),
    ]).with_columns([
        (pl.col("Pclose") >= pl.col("_xu100_ma200")).cast(pl.Float64).alias("xu100_above_ma200"),
        ((pl.col("Pclose") / pl.col("_xu100_peak_252")) - 1).alias("xu100_drawdown"),
        (pl.col("Pclose") / pl.col("Pclose").shift(120) - 1).alias("xu100_mom_120"),
    ]).select(["Date", "bm_vol_120", "xu100_mom_60", "xu100_mom_120",
               "xu100_above_ma200", "xu100_drawdown"])

    # Sektör endeks featureları
    xbank_df = xbank_df.with_columns([
        (pl.col("Pclose") / pl.col("Pclose").shift(60) - 1).alias("xbank_mom_60"),
    ]).select(["Date", "xbank_mom_60"])

    xusin_df = xusin_df.with_columns([
        (pl.col("Pclose") / pl.col("Pclose").shift(60) - 1).alias("xusin_mom_60"),
    ]).select(["Date", "xusin_mom_60"])

    # USD/TRY rejim featureları
    usdtry_df = usdtry_df.with_columns([
        (pl.col("Pclose") / pl.col("Pclose").shift(30) - 1).alias("usd_mom_30"),
        (pl.col("Pclose") / pl.col("Pclose").shift(60) - 1).alias("usd_mom_60"),
        pl.col("Pclose").pct_change().rolling_std(window_size=30).alias("usd_vol_30"),
    ]).select(["Date", "usd_mom_30", "usd_mom_60", "usd_vol_30"])

    # 3. Main Feature Calculation Loop (Vectorized per Ticker)
    print("🧠 Calculating Strategic (120d) and Tactical (30d) Metrics...")
    
    # We use a group_by + agg pattern for efficient vectorized rolling
    df_feat = df.lazy().sort(["Ticker", "Date"]).group_by("Ticker").agg([
        pl.col("Date"),
        pl.col("Pclose"),
        pl.col("Pvwap"),
        pl.col("Vlot"),
        pl.col("Phigh"),
        pl.col("Plow"),

        # --- Strategic Metrics (120d) ---
        (pl.col("Pclose") / pl.col("Pclose").shift(120) - 1).alias("mom_120"),
        (pl.col("Pclose").rolling_std(120) / pl.col("Pclose").rolling_mean(120)).alias("cv_120"),
        pl.col("Pclose").pct_change().rolling_std(120).alias("vol_120"),

        # --- Medium-term Metrics (60d) ---
        (pl.col("Pclose") / pl.col("Pclose").shift(60) - 1).alias("mom_60"),
        (pl.col("Pclose").rolling_std(60) / pl.col("Pclose").rolling_mean(60)).alias("cv_60"),
        pl.col("Pclose").pct_change().rolling_std(60).alias("vol_60"),

        # --- Long-term Metrics (252d) ---
        (pl.col("Pclose") / pl.col("Pclose").shift(252) - 1).alias("mom_252"),
        (pl.col("Pclose").rolling_std(252) / pl.col("Pclose").rolling_mean(252)).alias("cv_252"),

        # VWAP Dist: (Pclose - Pvwap) / Pvwap
        ((pl.col("Pclose") - pl.col("Pvwap")) / pl.col("Pvwap")).rolling_mean(20).alias("vwap_dist_avg"),

        # --- Tactical Metrics (30d) ---
        (pl.col("Pclose") / pl.col("Pclose").shift(30) - 1).alias("mom_30"),
        pl.col("Pclose").pct_change().rolling_std(30).alias("vol_30"),

        # --- Advanced Volume (V-ROC) ---
        # (EMA(Volume, 10) / EMA(Volume, 60)) - 1
        (pl.col("Vlot").ewm_mean(span=10) / pl.col("Vlot").ewm_mean(span=60) - 1).alias("v_roc"),

        # ─── PAKET 2: TRADER TAHTA FEATURE'LARI ──────────────────────────────
        # 52W high/low mesafesi (negatif = tepeden uzak, pozitif = dipten uzak)
        (pl.col("Pclose") / pl.col("Phigh").rolling_max(252) - 1).alias("dist_52w_high"),
        (pl.col("Pclose") / pl.col("Plow").rolling_min(252) - 1).alias("dist_52w_low"),

        # Son 20g range içindeki pozisyon (0=dipte, 1=tepede)
        ((pl.col("Pclose") - pl.col("Plow").rolling_min(20)) /
         (pl.col("Phigh").rolling_max(20) - pl.col("Plow").rolling_min(20) + 1e-9)
        ).alias("price_pos_20d"),

        # VWAP ±2σ bandı pozisyonu (kaç sigma uzakta)
        ((pl.col("Pclose") - pl.col("Pvwap").rolling_mean(20)) /
         (pl.col("Pvwap").rolling_std(20) * 2.0 + 1e-9)
        ).alias("vwap_band_pos"),

        # Klasik pivot point mesafesi: (H+L+C).shift(1)/3
        ((pl.col("Pclose") -
          (pl.col("Phigh").shift(1) + pl.col("Plow").shift(1) + pl.col("Pclose").shift(1)) / 3.0
         ) /
         ((pl.col("Phigh").shift(1) + pl.col("Plow").shift(1) + pl.col("Pclose").shift(1)) / 3.0 + 1e-9)
        ).alias("pivot_dist"),

        # Hacim ortalamasına göre patlama (>1 = hacim yükselişte)
        (pl.col("Vlot") / pl.col("Vlot").rolling_mean(60)).alias("vol_mean_revert"),

    ]).explode(pl.all().exclude("Ticker")).drop_nulls()

    # 4. Join with Benchmark and Macro data
    print("📈 Merging Benchmark and Macro data...")
    df_feat = df_feat.join(benchmark_df.lazy(), on="Date", how="left")
    df_feat = df_feat.join(usdtry_df.lazy(),    on="Date", how="left")
    df_feat = df_feat.join(xbank_df.lazy(),     on="Date", how="left")
    df_feat = df_feat.join(xusin_df.lazy(),     on="Date", how="left")

    df_feat = df_feat.with_columns([
        (pl.col("vol_120") / pl.col("bm_vol_120")).alias("rel_vol_120"),
        (pl.col("cv_120")  / pl.col("cv_252")).alias("cv_compression"),
        # Hissenin piyasaya göreli gücü (alpha)
        (pl.col("mom_60")  - pl.col("xu100_mom_60")).alias("rel_strength_60"),
        (pl.col("mom_120") - pl.col("xu100_mom_120")).alias("rel_strength_120"),
        # Trend kırılması: kısa vade uzun vadeyi geçiyor mu?
        (pl.col("mom_30")  - pl.col("mom_120")).alias("mom_divergence"),
    ])

    # Materialize the LazyFrame
    final_df = df_feat.collect()

    # 7. Sector-based relative strength
    if sector_map_df is not None:
        print("🏭 Sektör bazlı göreli güç hesaplanıyor...")
        final_df = final_df.join(sector_map_df, on="Ticker", how="left")

        # Her tarih için sektör medyanı hesapla (sektör "endeksi")
        sector_medians = (
            final_df
            .group_by(["Date", "Sector"])
            .agg([
                pl.col("mom_60").median().alias("sector_mom_60"),
                pl.col("mom_120").median().alias("sector_mom_120"),
                pl.col("vol_120").median().alias("sector_vol_120"),
            ])
        )
        final_df = final_df.join(sector_medians, on=["Date", "Sector"], how="left")

        # Sektör göreli güç
        final_df = final_df.with_columns([
            (pl.col("mom_60") - pl.col("sector_mom_60")).alias("sector_rel_60"),
            (pl.col("mom_120") - pl.col("sector_mom_120")).alias("sector_rel_120"),
        ])
        print(f"   ✅ Sektör feature'ları eklendi. Sektör sayısı: {final_df['Sector'].n_unique()}")
    else:
        final_df = final_df.with_columns([
            pl.lit(None).cast(pl.Float64).alias("sector_rel_60"),
            pl.lit(None).cast(pl.Float64).alias("sector_rel_120"),
        ])

    # 6. Save Feature Store
    final_df.write_parquet(FEATURES_PATH)
    print(f"✅ Feature Store Created: {FEATURES_PATH}")
    print(f"   Total Rows: {len(final_df):,}")

    # Print sample of latest date for a ticker
    latest = final_df.filter(pl.col("Date") == final_df["Date"].max()).head(1)
    if len(latest) > 0:
        print(f"   Sample Results (Latest Date: {latest['Date'][0]}):")
        print(latest.select(["Ticker", "cv_120", "mom_120", "v_roc", "rel_vol_120", "sector_rel_60"]))

if __name__ == "__main__":
    calculate_features()
