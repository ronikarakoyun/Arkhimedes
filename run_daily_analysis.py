"""
run_daily_analysis.py — Arkhimedes Günlük Tarama Motoru
=========================================================
Mevcut GMM + LambdaRank modeliyle tek günlük tarama yapar.
run_full_analysis.py'nin hafif, hızlı versiyonu — twin matching YOK.

Çıktı: Gunluk_Raporlar/Gunluk_Ozet/
  ├── Tum_Adaylar_Dashboard_YYYYMMDD.txt   (metin özet)
  └── Tum_Adaylar_YYYYMMDD.xlsx            (Excel, küme bazlı)

Çalıştırma:
  venv/bin/python3 run_daily_analysis.py              # bugün
  venv/bin/python3 run_daily_analysis.py --date YYYY-MM-DD  # belirli gün
"""
import polars as pl
import pandas as pd
import numpy as np
import joblib
import os
import warnings
from datetime import datetime
import argparse

from model_core import apply_preprocessing, FEATURES_FOR_CLUSTERING
from backtest_engine import _assign_clusters
from macro_engine import load_macro_features
from fundamental_engine import load_fundamentals, snapshot_at
from combined_filter import TECH_FEATURES

warnings.filterwarnings('ignore')

from config import (
    DB_PATH, FEATURES_PATH, ARTIFACTS_PATH,
    SUCCESS_POOL_PATH, FAILURE_POOL_PATH,
    REPORT_DAILY_DIR as REPORT_DIR,
)


def run_screener_and_dashboard(target_date_str=None, return_df=False):
    print("🚀 Arkhimedes — Günlük Tarama Motoru Başlıyor...")
    os.makedirs(REPORT_DIR, exist_ok=True)

    # ── Model artifacts ──────────────────────────────────────────────────────
    if not os.path.exists(ARTIFACTS_PATH):
        print(f"❌ Model artifacts bulunamadı ({ARTIFACTS_PATH}).")
        print("   Önce şunu çalıştırın: venv/bin/python3 build_knowledge_base.py")
        return

    artifacts = joblib.load(ARTIFACTS_PATH)

    # GMM/KMeans/HDBSCAN uyumlu model bundle
    model_bundle = {
        'cluster_method': artifacts.get('clustering_method', artifacts.get('cluster_method', 'gmm')),
        'clusterer':      artifacts.get('clusterer', artifacts.get('kmeans')),
    }
    scaler        = artifacts['scaler']
    best_k        = artifacts['best_k']
    q_low         = artifacts['q_low']
    q_high        = artifacts['q_high']
    feature_weights = artifacts.get('feature_weights', np.ones(len(FEATURES_FOR_CLUSTERING)))
    cluster_map   = artifacts['cluster_map']
    cluster_info  = artifacts['cluster_info']
    win_rates     = {c: info['win_rate'] for c, info in cluster_info.items()}
    cf            = artifacts.get('combined_filter')

    method_name = model_bundle['cluster_method'].upper()
    print(f"✅ Model yüklendi ({method_name}, k={best_k}, {len(cluster_map)} küme)")
    if cf is not None and cf.is_useful():
        print(f"   🧠 LambdaRankFilter aktif (n_train={cf.n_train:,})")
    else:
        print(f"   ⚠️ LambdaRankFilter yok — WinRate ile sıralama")

    # ── Piyasa verisi ────────────────────────────────────────────────────────
    df_market_full = pl.read_parquet(DB_PATH)

    if target_date_str:
        try:
            target_date = pd.to_datetime(target_date_str).date()
            df_market = df_market_full.with_columns(
                pl.col("Date").cast(pl.Date)
            ).filter(pl.col("Date") <= target_date)
            latest_date = df_market["Date"].max()
            print(f"⏰ Geriye Dönük Tarama (Tarih: {latest_date})")
        except Exception as e:
            print(f"❌ Tarih formatı hatası: {e}")
            return
    else:
        df_market = df_market_full
        latest_date = df_market["Date"].max()

    date_str = (latest_date.strftime('%Y-%m-%d')
                if not isinstance(latest_date, str) else str(latest_date)[:10])
    date_fname = date_str.replace('-', '')
    print(f"📅 Tarama Tarihi: {date_str}")

    # ── Operasyon Enkazı filtresi ────────────────────────────────────────────
    print("🧹 'Operasyon Enkazı' Filtresi Uygulanıyor...")
    df_sieve = (
        df_market.lazy()
        .sort(["Ticker", "Date"])
        .group_by("Ticker").agg([
            pl.col("Date").last().alias("Date"),
            pl.col("Pclose").last().alias("current_price"),
            pl.col("Phigh").tail(252).max().alias("peak_252d"),
            pl.col("Plow").tail(252).min().alias("base_252d"),
        ])
    ).collect()

    survivors = df_sieve.filter(
        ~((pl.col("peak_252d") > pl.col("base_252d") * 2.5) &
          (pl.col("current_price") < pl.col("peak_252d") * 0.55))
    )
    surviving_tickers = survivors["Ticker"].to_list()
    print(f"🛡️  {len(surviving_tickers)} hisse taramaya uygun bulundu.")

    if not surviving_tickers:
        print("Aday bulunamadı.")
        return

    # ── Feature yükleme (look-ahead bias düzeltmesi) ─────────────────────────
    df_feat = pl.read_parquet(FEATURES_PATH).with_columns(pl.col("Date").cast(pl.Date))
    df_feat = df_feat.filter(pl.col("Date") <= latest_date)

    latest_features = (
        df_feat
        .filter(pl.col("Ticker").is_in(surviving_tickers))
        .to_pandas()
        .sort_values(["Ticker", "Date"])
        .groupby("Ticker").last()
        .reset_index()
        .dropna(subset=FEATURES_FOR_CLUSTERING)
    )

    # ── Sınıflandırma (GMM/KMeans/HDBSCAN uyumlu) ───────────────────────────
    print("🤖 Tüm adaylar sınıflandırılıyor...")
    pp = apply_preprocessing(latest_features, FEATURES_FOR_CLUSTERING, q_low, q_high)
    valid = pp[FEATURES_FOR_CLUSTERING].notna().all(axis=1)
    latest_features = latest_features[valid].copy().reset_index(drop=True)
    pp = pp[valid].reset_index(drop=True)

    X = scaler.transform(pp[FEATURES_FOR_CLUSTERING]) * feature_weights
    cluster_labels, uyum_scores = _assign_clusters(model_bundle, X)

    latest_features['Cluster']    = cluster_labels
    latest_features['Uyum_Skoru'] = uyum_scores
    latest_features['Karakter']   = latest_features['Cluster'].map(cluster_map)
    latest_features['WinRate']    = latest_features['Cluster'].map(win_rates).fillna(0.0)
    latest_features['Score']      = (
        0.7 * latest_features['WinRate'] +
        0.3 * latest_features['Uyum_Skoru']
    )

    print("\n--- Dinamik Küme Haritası ---")
    for k_id, name in cluster_map.items():
        cnt = (latest_features['Cluster'] == k_id).sum()
        wr  = win_rates.get(k_id, 0.0)
        print(f"  Küme {k_id} → {name}  (WR=%{wr*100:.1f}, piyasada {cnt} aday)")

    # ── LambdaRank RankScore (opsiyonel) ─────────────────────────────────────
    latest_features['RankScore'] = np.nan
    if cf is not None and cf.is_useful():
        try:
            macro_df = load_macro_features()
            macro_df['Date'] = pd.to_datetime(macro_df['Date'])
            fund_df = load_fundamentals()

            td = pd.to_datetime(latest_date)
            macro_row = macro_df[macro_df['Date'] <= td].sort_values('Date').tail(1)
            m_cols = [c for c in macro_df.columns if c.startswith('m_')]

            df_work = latest_features.copy()
            if not macro_row.empty:
                for col in m_cols:
                    df_work[col] = macro_row.iloc[0][col]

            queries = df_work[['Ticker']].copy()
            queries['Date'] = td
            snap = snapshot_at(fund_df, queries).drop_duplicates(['Ticker', 'Date'])
            df_work = df_work.merge(snap, on='Ticker', how='left', suffixes=('', '_snap'))
            if 'Date_snap' in df_work.columns:
                df_work = df_work.drop(columns=['Date_snap'])

            feat_cols = cf.feature_cols
            for c in feat_cols:
                if c not in df_work.columns:
                    df_work[c] = np.nan

            latest_features['RankScore'] = cf.predict_rank_score(df_work[feat_cols])
            latest_features['Score']     = latest_features['RankScore']
            print(f"\n   🔍 LambdaRankFilter: RankScore hesaplandı ({latest_features['RankScore'].notna().sum()} hisse)")
        except Exception as e:
            print(f"   ⚠️ RankScore hesaplanamadı: {e}")

    # ── Rapor Üretimi ─────────────────────────────────────────────────────────
    sort_col = 'RankScore' if latest_features['RankScore'].notna().any() else 'WinRate'
    final_df = latest_features.sort_values(sort_col, ascending=False)

    dashboard_lines = [
        f"=== BIST GÜNLÜK KANTİTATİF TARAMA RAPORU ({date_str}) ===\n",
        f"Model: {method_name} + LambdaRank  |  Sıralama: {sort_col}",
        f"Taranan: {len(surviving_tickers)} hisse  |  Sınıflandırılan: {len(final_df)} hisse\n",
    ]

    for k_id, name in sorted(cluster_map.items()):
        group_df = final_df[final_df['Cluster'] == k_id]
        if group_df.empty:
            continue
        wr = win_rates.get(k_id, 0.0)
        dashboard_lines.append(f"\n{'='*45}")
        dashboard_lines.append(f"📁 KÜME {k_id} — {name}  (WR=%{wr*100:.1f}, {len(group_df)} aday)")
        dashboard_lines.append(f"{'='*45}")
        for _, r in group_df.head(20).iterrows():
            rs_str = f"RS:{r['RankScore']:+.2f}" if not pd.isna(r['RankScore']) else f"WR:%{r['WinRate']*100:.1f}"
            dashboard_lines.append(
                f"👉 {r['Ticker']:<8} | {rs_str:<12} | "
                f"Uyum:{r['Uyum_Skoru']:.3f} | "
                f"CV:{r.get('cv_120', float('nan')):>5.3f} | "
                f"mom120:{r.get('mom_120', float('nan')):>+5.2f} | "
                f"vRoc:{r.get('v_roc', float('nan')):>+5.2f}"
            )

    txt_path = os.path.join(REPORT_DIR, f"Tum_Adaylar_Dashboard_{date_fname}.txt")
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(dashboard_lines))

    xlsx_path = os.path.join(REPORT_DIR, f"Tum_Adaylar_{date_fname}.xlsx")
    export_cols = ['Ticker', 'Cluster', 'Karakter', 'WinRate', 'Uyum_Skoru', 'RankScore',
                   'mom_120', 'mom_60', 'cv_120', 'cv_compression', 'v_roc',
                   'rel_strength_60', 'sector_rel_60', 'usd_mom_30']
    export_cols = [c for c in export_cols if c in final_df.columns]

    with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
        final_df[export_cols].to_excel(writer, sheet_name='Tum_Adaylar', index=False)
        for k_id, name in sorted(cluster_map.items()):
            group_df = final_df[final_df['Cluster'] == k_id]
            if group_df.empty:
                continue
            sheet = f"K{k_id}_{name}"[:31].replace('/', '-')
            group_df[export_cols].to_excel(writer, sheet_name=sheet, index=False)

    print(f"\n✅ Tüm Raporlar '{REPORT_DIR}' klasörüne kaydedildi.")
    print(f"   📄 {txt_path}")
    print(f"   📊 {xlsx_path}")
    print(f"   Toplam {len(final_df)} aday — sıralama: {sort_col}")

    if return_df:
        return final_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arkhimedes Günlük Tarama Motoru')
    parser.add_argument('--date', type=str, help='Tarama tarihi (YYYY-MM-DD)', default=None)
    args = parser.parse_args()
    run_screener_and_dashboard(target_date_str=args.date)
