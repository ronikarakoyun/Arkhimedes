"""
BIST 4-Adımlı Tam Analiz Pipeline'ı
====================================
Adım 1: Tarama — küme bazlı ralli adayı tespit
Adım 2: Twin Matcher — her aday için 3-kanal DTW ikizleri
Adım 3: Öncü-Ardıl Analizi — başarılı ikizlerin tetiklenme zamanlaması
Adım 4: Failed-Twin Analizi — aynı kurulumda ralli yapmamış ikizler ve nedenleri

Çıktılar:
- Markdown rapor (.md) — okunabilir, scannable
- Excel (.xlsx) — küme bazlı sheet'ler
- PNG grafikler — her aday için ikiz scenario
"""

import polars as pl
import pandas as pd
import numpy as np
import joblib
import os
import warnings
import matplotlib
matplotlib.use('Agg')   # thread-safe file yazımı için (B11 paralel DTW)
import matplotlib.pyplot as plt
from datetime import datetime
from pandas.tseries.offsets import BDay   # B2: trading-day → takvim çevirisi
import argparse
from joblib import Parallel, delayed as jdelayed   # B11: paralel DTW

from twin_matcher import (
    get_base_curve, precompute_pool_curves,
    DTW_WINDOW, MAX_DISTANCE_NDIM
)
from dtaidistance import dtw_ndim
from model_core import apply_preprocessing, FEATURES_FOR_CLUSTERING
from twin_divergence import compute_gain_curves

warnings.filterwarnings('ignore')

# ─── Path & Parametreler ────────────────────────────────────────────────────
from config import (
    DB_PATH, FEATURES_PATH, ARTIFACTS_PATH,
    SUCCESS_POOL_PATH, FAILURE_POOL_PATH, SETUPS_PATH,
    REPORT_FULL_DIR as REPORT_DIR,
    FULL_ANALYSIS_TOP_N        as TOP_N,
    FULL_ANALYSIS_TWIN_TOP_K   as TWIN_TOP_K,
    FULL_ANALYSIS_RALLY_TRIGGER as RALLY_TRIGGER,
    FULL_ANALYSIS_TIMING_WINDOW as TIMING_WINDOW,
    TWIN_DIV_WINDOW, TWIN_DIV_MIN_TWINS,
    FINAL_P_THRESHOLD,
)
from combined_filter import TECH_FEATURES
from backtest_engine import _assign_clusters as _be_assign_clusters
from macro_engine import load_macro_features
from fundamental_engine import load_fundamentals, snapshot_at


# ─── ADIM 1: Tarama + Sınıflandırma ─────────────────────────────────────────

def screen_candidates(target_date=None):
    """Operasyon enkazı filtresi + cluster atama + top N seçim."""
    print("🔍 ADIM 1: Tarama ve sınıflandırma...")

    artifacts = joblib.load(ARTIFACTS_PATH)
    df_market_full = pl.read_parquet(DB_PATH)

    if target_date:
        target_date = pd.to_datetime(target_date).date()
        df_market = df_market_full.with_columns(pl.col("Date").cast(pl.Date)).filter(
            pl.col("Date") <= target_date
        )
    else:
        df_market = df_market_full

    latest_date = df_market["Date"].max()
    date_str = latest_date.strftime('%Y-%m-%d') if not isinstance(latest_date, str) else str(latest_date)[:10]
    print(f"   📅 Tarama tarihi: {date_str}")

    # Operasyon enkazı (peak'ten %55+ düşmüş)
    df_sieve = (
        df_market.lazy().sort(["Ticker", "Date"]).group_by("Ticker").agg([
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
    print(f"   🛡️  Enkaz filtresinden geçen: {len(surviving_tickers)} hisse")

    # Feature'ları yükle
    df_feat = pl.read_parquet(FEATURES_PATH).with_columns(pl.col("Date").cast(pl.Date))
    df_feat = df_feat.filter(pl.col("Date") <= latest_date)

    latest_features = (
        df_feat.filter(pl.col("Ticker").is_in(surviving_tickers))
        .sort(["Ticker", "Date"]).group_by("Ticker").last()
        .to_pandas().dropna(subset=FEATURES_FOR_CLUSTERING)
    )

    # Sınıflandırma
    pp = apply_preprocessing(
        latest_features[FEATURES_FOR_CLUSTERING],
        FEATURES_FOR_CLUSTERING,
        artifacts['q_low'], artifacts['q_high']
    )
    valid = pp.notna().all(axis=1)
    latest_features = latest_features[valid].copy()
    pp = pp[valid]

    X = artifacts['scaler'].transform(pp[FEATURES_FOR_CLUSTERING]) * artifacts['feature_weights']

    # GMM / KMeans / HDBSCAN uyumlu küme atama
    model_bundle = {
        'cluster_method': artifacts.get('clustering_method', 'kmeans'),
        'clusterer':      artifacts.get('clusterer', artifacts.get('kmeans')),
    }
    cluster_labels, uyum_scores = _be_assign_clusters(model_bundle, X)
    latest_features['Cluster']    = cluster_labels
    latest_features['Karakter']   = latest_features['Cluster'].map(artifacts['cluster_map'])
    win_rates = {c: info['win_rate'] for c, info in artifacts['cluster_info'].items()}
    latest_features['Cluster_WinRate'] = latest_features['Cluster'].map(win_rates).fillna(0.0)
    latest_features['Uyum_Skoru'] = uyum_scores

    # TOP N seçimi — Score = 0.7×WinRate + 0.3×Uyum (LambdaRankFilter yokken kullanılır)
    latest_features['Score'] = (
        0.7 * latest_features['Cluster_WinRate'] +
        0.3 * latest_features['Uyum_Skoru']
    )
    top_candidates = latest_features.sort_values('Score', ascending=False).head(TOP_N)

    print(f"   🎯 Top {len(top_candidates)} aday seçildi")
    print(f"   Dağılım: {dict(top_candidates['Karakter'].value_counts())}")

    # LambdaRankFilter — RankScore (macro + fundamental + technical)
    # Önce artifacts'tan yükle; yoksa sıfırdan eğit.
    combined_filter = _load_or_build_combined_filter(artifacts)
    if combined_filter is not None and combined_filter.is_useful():
        top_candidates = _annotate_with_p_rally(top_candidates, combined_filter, latest_date)
        # RankScore ile yeniden sırala (büyük = daha iyi)
        if 'P_rally' in top_candidates.columns:
            top_candidates = top_candidates.sort_values('P_rally', ascending=False)
            # Score sütununu da güncelle — LambdaRank ana sıralama otoritesi
            top_candidates['Score'] = top_candidates['P_rally'].fillna(top_candidates['Score'])
            ndcg_str = f"{combined_filter.cv_ndcg:.4f}" if combined_filter.cv_ndcg else "N/A"
            print(f"   🔍 LambdaRankFilter: RankScore ile sıralama yapıldı"
                  f"  (n={combined_filter.n_train:,}, cv_ndcg={ndcg_str})")
    else:
        top_candidates['P_rally'] = np.nan
        print("   ⚠️ LambdaRankFilter devre dışı (veri yok veya eğitim başarısız)")

    return top_candidates, df_market, latest_date, artifacts


def _load_or_build_combined_filter(artifacts: dict):
    """
    LambdaRankFilter'ı artifacts'tan yükle.
    artifacts['combined_filter'] varsa ve is_useful() ise direkt kullan.
    Yoksa uyarı ver ve None döndür (sistem WinRate-tabanlı Score ile devam eder).
    """
    cf = artifacts.get('combined_filter')
    if cf is not None and cf.is_useful():
        ndcg_str = f"{cf.cv_ndcg:.4f}" if cf.cv_ndcg else "N/A"
        print(f"\n🧠 LambdaRankFilter artifacts'tan yüklendi:")
        print(f"   ✅ KEEP  cv_ndcg={ndcg_str}  n_train={cf.n_train:,}  feat={len(cf.feature_cols)}")
        return cf
    elif cf is not None:
        print(f"\n🧠 LambdaRankFilter yüklendi ama is_useful()=False → devre dışı")
        return cf  # is_useful() kontrolü dışarıda yapılıyor
    else:
        print(f"\n⚠️ artifacts['combined_filter'] yok — build_knowledge_base.py çalıştırın")
        return None


def _annotate_with_p_rally(top_candidates, cf, latest_date):
    """top_candidates'e P_rally kolonu ekle (macro + fund snapshot + tech zaten var)."""
    try:
        macro = load_macro_features()
        macro['Date'] = pd.to_datetime(macro['Date'])
        fund = load_fundamentals()

        td = pd.to_datetime(latest_date)
        macro_row = macro[macro['Date'] == td]
        if macro_row.empty:
            macro_row = macro.sort_values('Date').iloc[[-1]]
        m_cols = [c for c in macro.columns if c.startswith('m_')]

        # Fundamental snapshot for each candidate
        queries = top_candidates[['Ticker']].copy()
        queries['Date'] = td
        snap = snapshot_at(fund, queries).drop_duplicates(['Ticker', 'Date'])

        df = top_candidates.reset_index(drop=True).copy()
        for col in m_cols:
            df[col] = macro_row.iloc[0][col]
        df = df.merge(snap, on='Ticker', how='left', suffixes=('', '_snap'))
        if 'Date_snap' in df.columns:
            df = df.drop(columns=['Date_snap'])

        feat_cols = cf.feature_cols
        missing = [c for c in feat_cols if c not in df.columns]
        for c in missing:
            df[c] = np.nan
        df['P_rally'] = cf.predict_rank_score(df[feat_cols])
        top_candidates = top_candidates.copy()
        top_candidates['P_rally'] = df['P_rally'].values
        return top_candidates
    except Exception as e:
        print(f"   ⚠️ P_rally hesabı başarısız: {e}")
        top_candidates = top_candidates.copy()
        top_candidates['P_rally'] = np.nan
        return top_candidates


# ─── ADIM 2: Twin Matcher ───────────────────────────────────────────────────

def find_twins(candidate_ticker, target_date, target_curve, pool_cache):
    """Tek bir aday için en yakın TWIN_TOP_K ikizi döndürür."""
    matches = []
    target_date_pd = pd.to_datetime(target_date)

    for entry in pool_cache:
        if (entry['Ticker'] == candidate_ticker and
                abs((pd.to_datetime(entry['Date']) - target_date_pd).days) < 60):
            continue

        distance = dtw_ndim.distance_fast(
            target_curve, entry['Curve'],
            window=DTW_WINDOW, max_dist=MAX_DISTANCE_NDIM,
        )
        if distance < MAX_DISTANCE_NDIM:
            matches.append({
                'Ticker':       entry['Ticker'],
                'Date':         pd.to_datetime(entry['Date']),
                'DTW_Distance': distance,
                'Outcome':      entry['Outcome'],
            })

    if not matches:
        return None
    return pd.DataFrame(matches).sort_values('DTW_Distance').head(TWIN_TOP_K).reset_index(drop=True)


# ─── ADIM 3: Öncü-Ardıl Analizi ─────────────────────────────────────────────

def days_to_rally_trigger(df_market, ticker, setup_date, threshold=RALLY_TRIGGER):
    """Setup tarihinden itibaren ilk %30+ kazanca kaç gün sonra ulaştı?"""
    setup_row = df_market.filter(
        (pl.col("Ticker") == ticker) & (pl.col("Date") <= setup_date)
    ).tail(1)
    if len(setup_row) == 0:
        return None
    setup_close = setup_row["Pclose"].to_numpy()[0]

    after = df_market.filter(
        (pl.col("Ticker") == ticker) & (pl.col("Date") > setup_date)
    ).head(TIMING_WINDOW).sort("Date")
    if len(after) == 0:
        return None

    prices = after["Pclose"].to_numpy()
    gains = prices / setup_close - 1
    triggered = np.where(gains >= threshold)[0]
    if len(triggered) == 0:
        return None
    return int(triggered[0] + 1)


def analyze_leader_timing(twins_df, df_market):
    """
    Başarılı ikizler için tetiklenme zamanlaması:
    - Her başarılı ikiz için setup_date → ilk %30+ kazancın olduğu güne kaç gün geçmiş?
    - Median + IQR + son 90 günde tetiklenmiş ikizleri ayrıca raporla
    """
    timings = []
    recent_leaders = []  # son 30-90 günde tetiklenmiş ikizler

    success_twins = twins_df[twins_df['Outcome'] == 'Success']
    if len(success_twins) == 0:
        return None

    today = df_market["Date"].max()
    today_pd = pd.to_datetime(today)

    for _, t in success_twins.iterrows():
        days = days_to_rally_trigger(df_market, t['Ticker'], t['Date'])
        if days is None:
            continue
        timings.append({
            'Ticker': t['Ticker'],
            'SetupDate': t['Date'],
            'DaysToTrigger': days,
            'DTW': t['DTW_Distance'],
        })

        # Tetiklenme tarihi son 30-90 gün arasında mı?
        # days = trading-day indeksi → BDay ile takvim karşıtına çeviriyoruz
        trigger_date = t['Date'] + BDay(days)
        days_ago = (today_pd - trigger_date).days
        if 30 <= days_ago <= 90:
            recent_leaders.append({
                'Ticker': t['Ticker'],
                'SetupDate': t['Date'],
                'TriggerDate': trigger_date,
                'DaysAgo': days_ago,
            })

    if not timings:
        return None

    # recent_leaders'ı en yakın tarihli önce (DaysAgo küçük = daha taze)
    recent_leaders.sort(key=lambda x: x['DaysAgo'])

    timing_arr = np.array([t['DaysToTrigger'] for t in timings])
    return {
        'n_leaders':       len(timings),
        'median_days':     int(np.median(timing_arr)),
        'p25_days':        int(np.percentile(timing_arr, 25)),
        'p75_days':        int(np.percentile(timing_arr, 75)),
        'min_days':        int(timing_arr.min()),
        'max_days':        int(timing_arr.max()),
        'all_timings':     timings,
        'recent_leaders':  recent_leaders,
    }


# ─── ADIM 4: Failed-Twin Analizi ────────────────────────────────────────────

def analyze_failed_twins(twins_df, success_pool_full, failure_pool_full):
    """
    Başarısız ikizleri başarılılardan ayıran feature'ları tespit et.
    Returns: failed twins listesi + ayırıcı feature'lar
    """
    success_twins = twins_df[twins_df['Outcome'] == 'Success']
    failure_twins = twins_df[twins_df['Outcome'] == 'Failure']

    if len(failure_twins) == 0:
        return None

    # B5: Pool'ları (Ticker, Date str) multi-index ile indeksle → O(1) lookup
    # Eski versiyon her ikiz için linear scan yapıyordu: O(twins × pool_size)
    def _build_index(pool):
        p = pool.copy()
        p['_date_str'] = pd.to_datetime(p['Date']).dt.strftime('%Y-%m-%d')
        return p.set_index(['Ticker', '_date_str'])

    def fetch_features(df_twins, pool_idx):
        rows = []
        for _, t in df_twins.iterrows():
            key = (t['Ticker'], str(t['Date'])[:10])
            try:
                row = pool_idx.loc[key]
                rows.append(row.iloc[0] if isinstance(row, pd.DataFrame) else row)
            except KeyError:
                continue
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    s_idx = _build_index(success_pool_full)
    f_idx = _build_index(failure_pool_full)
    s_feats = fetch_features(success_twins, s_idx)
    f_feats = fetch_features(failure_twins, f_idx)

    if len(f_feats) == 0:
        return None

    # Ayırıcı feature'ları bul (s_med vs f_med farkı yüksek olanlar)
    # B9: normalize_diff = Cohen's d (pooled std) — asimetrik s_std yerine
    key_features = [
        'mom_120', 'mom_60', 'mom_30',
        'rel_strength_60', 'rel_strength_120',
        'sector_rel_60', 'sector_rel_120',
        'xu100_mom_60', 'xu100_drawdown', 'xu100_above_ma200',
        'usd_mom_30', 'usd_mom_60',
        'v_roc', 'cv_120', 'cv_compression',
    ]

    differentiators = []
    for f in key_features:
        if f not in s_feats.columns or f not in f_feats.columns:
            continue
        s_med = s_feats[f].median()
        f_med = f_feats[f].median()
        s_std = s_feats[f].std()
        f_std = f_feats[f].std()
        if pd.isna(s_med) or pd.isna(f_med):
            continue
        # B9: Cohen's d — pooled std daha dengeli (s_std tek başına asimetrik)
        pooled_std = np.sqrt((s_std ** 2 + f_std ** 2) / 2) if (
            not pd.isna(s_std) and not pd.isna(f_std)
        ) else s_std
        if pd.isna(pooled_std) or pooled_std < 1e-9:
            continue
        cohen_d = abs(s_med - f_med) / pooled_std
        differentiators.append({
            'feature':         f,
            'success_median':  s_med,
            'failure_median':  f_med,
            'diff':            s_med - f_med,
            'normalized_diff': cohen_d,   # Cohen's d
        })

    differentiators = sorted(differentiators, key=lambda x: x['normalized_diff'], reverse=True)

    # Başarısız ikizlerin detaylı listesi (en yakın 5'i)
    failed_list = []
    for _, t in failure_twins.head(5).iterrows():
        failed_list.append({
            'Ticker': t['Ticker'],
            'SetupDate': str(t['Date'])[:10],
            'DTW': round(t['DTW_Distance'], 2),
        })

    return {
        'n_failed':         len(failure_twins),
        'n_success':        len(success_twins),
        'top_differentiators': differentiators[:5],
        'failed_list':      failed_list,
    }


# ─── Görsel: Aday + İkizleri ────────────────────────────────────────────────

def plot_candidate_scenario(ticker, latest_date, twins_df, df_market, leader_info, out_path, twin_div=None):
    fig, ax = plt.subplots(figsize=(13, 7))

    # Mevcut aday
    cur = df_market.filter(
        (pl.col("Ticker") == ticker) & (pl.col("Date") <= latest_date)
    ).tail(120)["Pclose"].to_numpy()
    if len(cur) > 0:
        cur_norm = cur / cur[0]
        ax.plot(range(-len(cur), 0), cur_norm, color='black', linewidth=3.5, label=f'{ticker} (BUGÜN)', zorder=10)

    # İkizler
    success_count = 0
    for _, m in twins_df.iterrows():
        h_ticker = m['Ticker']
        h_date = m['Date']
        h_outcome = m['Outcome']

        h_data = df_market.filter(
            (pl.col("Ticker") == h_ticker) & (pl.col("Date") <= h_date)
        ).tail(120)["Pclose"].to_numpy()
        if len(h_data) == 0:
            continue
        h_norm = h_data / h_data[0]
        ax.plot(range(-len(h_norm), 0), h_norm, color='gray', alpha=0.25, linewidth=0.8)

        future = df_market.filter(
            (pl.col("Ticker") == h_ticker) & (pl.col("Date") > h_date)
        ).head(252)["Pclose"].to_numpy()
        if len(future) > 0:
            future_norm = future / h_data[0]
            c = 'green' if h_outcome == 'Success' else 'red'
            alpha = 0.55 if h_outcome == 'Success' else 0.25
            lbl = None
            if h_outcome == 'Success' and success_count < 3:
                lbl = f"İkiz başarı: {h_ticker}"
                success_count += 1
            ax.plot(range(0, len(future_norm)), future_norm, color=c, alpha=alpha,
                    linewidth=1.0, label=lbl)

    # Twin Divergence: Success/Failure medyan gain curve'leri (post-entry)
    if twin_div is not None:
        s_gain = twin_div.get('success_gain_median')
        f_gain = twin_div.get('failure_gain_median')
        if s_gain is not None and f_gain is not None:
            # gain → normalize fiyat (1.0 = giriş)
            s_norm = 1.0 + s_gain
            f_norm = 1.0 + f_gain
            xs = np.arange(1, len(s_norm) + 1)
            ax.plot(xs, s_norm, color='darkgreen', linewidth=2.5, linestyle='--',
                    label=f'Success medyan (n={twin_div.get("n_success_used",0)})', zorder=8)
            ax.plot(xs, f_norm, color='darkred', linewidth=2.5, linestyle='--',
                    label=f'Failure medyan (n={twin_div.get("n_failure_used",0)})', zorder=8)
            # Erken çıkış sınırı = ortalama, altı tehlike bölgesi
            mid_norm = (s_norm + f_norm) / 2
            ax.fill_between(xs, 0.5, mid_norm, color='red', alpha=0.05,
                            label='Tehlike bölgesi (failure tarafı)')

    ax.axvline(0, color='blue', linestyle='--', linewidth=1.5, alpha=0.6, label='Şimdi')

    # Tetiklenme zamanlaması yatay çizgisi
    if leader_info and leader_info.get('median_days'):
        med = leader_info['median_days']
        ax.axvline(med, color='orange', linestyle=':', linewidth=2,
                   label=f"Tetiklenme medyan: +{med}g")
        ax.axvspan(leader_info['p25_days'], leader_info['p75_days'], alpha=0.1, color='orange')

    ax.set_title(f"{ticker} — 3-Kanal DTW İkiz Senaryosu", fontsize=13, fontweight='bold')
    ax.set_xlabel("Gün (0 = şimdi)")
    ax.set_ylabel("Normalize fiyat (başlangıç=1)")
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-130, 260)
    plt.tight_layout()
    plt.savefig(out_path, dpi=100)
    plt.close()


# ─── Rapor Üretimi ──────────────────────────────────────────────────────────

def generate_markdown_report(report_data, latest_date, artifacts, out_path):
    """Profesyonel, scannable Markdown raporu üretir."""
    lines = []
    date_str = latest_date.strftime('%Y-%m-%d') if not isinstance(latest_date, str) else str(latest_date)[:10]

    lines.append(f"# BIST Tam Analiz Raporu — {date_str}")
    lines.append(f"\n*4-Adımlı Pipeline: Tarama → İkiz Matchi → Öncü-Ardıl → Failed Twin*\n")

    # Model özeti
    lines.append("## 🧠 Model Özeti\n")
    _cm = artifacts.get('clustering_method', artifacts.get('cluster_method', 'gmm'))
    lines.append(f"- **Sınıflandırma:** {_cm.upper()}, {len(artifacts['cluster_info'])} küme\n")
    lines.append(f"- **Eğitim:** {sum(c['original_size'] for c in artifacts['cluster_info'].values()):,} bağımsız ralli")
    lines.append(f"- **Analiz edilen aday sayısı:** {len(report_data)}\n")

    lines.append("\n### Küme Özeti\n")
    lines.append("| # | Karakter | Win-Rate | Eğitim | Geçiş | Piyasada |")
    lines.append("|---|---|---|---|---|---|")
    # Mevcut adaylardaki cluster dağılımını say
    cluster_counts_now = {}
    for r in report_data:
        cid = r.get('Cluster', -1)
        cluster_counts_now[cid] = cluster_counts_now.get(cid, 0) + 1
    for c, info in artifacts['cluster_info'].items():
        cnt_now = cluster_counts_now.get(c, 0)
        alert = " ⚠️ YOK" if cnt_now == 0 else f" {cnt_now} aday"
        lines.append(f"| {c} | {info['name']} | **%{info['win_rate']*100:.1f}** | {info['original_size']:,} | {info['transition_size']:,} |{alert} |")

    # K2 yoksa açıkça uyar
    best_c = max(artifacts['cluster_info'].items(), key=lambda x: x[1]['win_rate'])
    if cluster_counts_now.get(best_c[0], 0) == 0:
        lines.append(f"\n> ⚠️ **Piyasa Uyarısı:** En yüksek win-rate'li küme (K{best_c[0]}, %{best_c[1]['win_rate']*100:.1f}) "
                     f"mevcut taramada **hiç aday üretmiyor**. Tüm sinyaller daha düşük win-rate'li kümelerde "
                     f"(%40-42). Bu bir model hatası değil — piyasa o karakteristiği göstermiyor. "
                     f"Sinyal kalitesi düşük, pozisyon büyüklüklerini buna göre ayarlayın.\n")

    # Yönetici özeti — Twin success'e göre sıralı
    lines.append("\n## 📊 Yönetici Özeti — Top 20 Sinyal (Twin İkiz Başarısı)\n")
    lines.append("*Sıralama: İkiz eşleşme başarı oranı. "
                 "Twin% = bu hisseye en çok benzeyen 20 tarihsel örneğin kaçı ralli yaptı.*\n")
    lines.append("| # | Hisse | Küme | Twin% | İkiz ✓/✗ | Win-Rate | Uyum | RankScore | Med Gün |")
    lines.append("|---|---|---|---|---|---|---|---|---|")

    ranked = sorted(
        report_data,
        key=lambda r: (r['n_twin_success'] / max(r['n_twins'], 1), r.get('Uyum_Skoru', 0)),
        reverse=True,
    )

    for i, r in enumerate(ranked[:20], 1):
        leader = r.get('leader_info')
        med_days = f"+{leader['median_days']}g" if leader else "—"
        twin_str = f"{r['n_twin_success']}/{r['n_twin_failure']}"
        twin_pct = r['n_twin_success'] / max(r['n_twins'], 1) * 100
        if twin_pct >= 65:
            tag = "🟢"
        elif twin_pct >= 50:
            tag = "🟡"
        else:
            tag = "⚪"
        p_rally = r.get('P_rally')
        if p_rally is None or (isinstance(p_rally, float) and np.isnan(p_rally)):
            p_str = "—"
        else:
            # LambdaRank: ham sıralama skoru (negatif olabilir, yüksek = daha iyi)
            p_str = f"{p_rally:+.2f}"
        lines.append(
            f"| {i} | {tag} **{r['Ticker']}** | {r['Karakter']} | **%{twin_pct:.0f}** | "
            f"{twin_str} | %{r['Cluster_WinRate']*100:.1f} | {r['Uyum_Skoru']:.3f} | {p_str} | {med_days} |"
        )

    lines.append(f"\n🟢 = Twin% ≥ 65 (güçlü ikiz geçmişi)  |  🟡 = Twin% 50-65 (orta)  |  ⚪ = Twin% < 50 (zayıf)\n")

    # Küme bazlı gruplama — cluster_id ile grupla, Karakter değil.
    # K0 ve K2 ikisi de "Dipte" etiketine sahip olabilir; Karakter key'i kullanmak
    # onları aynı bölüme düşürür ve items[0]['Cluster_WinRate'] yanlış kümeye ait olur.
    by_cluster = {}
    for r in report_data:
        key = (r['Cluster'], r['Karakter'])
        by_cluster.setdefault(key, []).append(r)

    for (cluster_id, cluster_name), items in sorted(by_cluster.items()):
        lines.append(f"\n---\n\n## 📁 K{cluster_id} — {cluster_name}\n")
        lines.append(f"**Kümede {len(items)} aday** | Tarihsel win-rate: **%{items[0]['Cluster_WinRate']*100:.1f}**\n")

        # Bu kümenin özellikleri (her küme için açıklama)
        info = artifacts['cluster_info'][cluster_id]
        lines.append(f"\n### Bu kümenin profili (neden bu hisseler buraya düştü):\n")
        sig_items = []
        for f, lbl in info.get('signature', {}).items():
            if lbl != "Orta":
                sig_items.append(f"`{f}={lbl}`")
        if sig_items:
            lines.append("- Belirleyici özellikler: " + ", ".join(sig_items))
        lines.append(f"- Geçmişte bu profilde {info['transition_size']:,} bağımsız rejim girişi olmuş")
        lines.append(f"- Bunların **{info.get('transition_success', '?')}**'si 252 gün içinde XU100'ü ≥%50 geçmiş (Win-rate: %{info['win_rate']*100:.1f})\n")

        # Her aday için detay
        for r in items:
            ticker = r['Ticker']
            leader = r.get('leader_info')
            failed = r.get('failed_info')

            lines.append(f"\n### 🎯 {ticker}\n")

            # Sinyal özeti
            lines.append(f"\n**📊 Sinyal Özeti**\n")
            lines.append(f"| | |")
            lines.append(f"|---|---|")
            lines.append(f"| Küme Win-Rate | %{r['Cluster_WinRate']*100:.1f} |")
            lines.append(f"| Uyum Skoru (1/(1+d)) | {r['Uyum_Skoru']:.4f} (mesafe {r.get('MinDist', 0):.3f}) |")
            lines.append(f"| Yakın İkiz | {r['n_twins']} (Başarı: {r['n_twin_success']}, Hüsran: {r['n_twin_failure']}) |")
            lines.append(f"| İkiz Başarı Oranı | %{(r['n_twin_success']/max(r['n_twins'],1))*100:.1f} |")

            # Mevcut profili
            lines.append(f"\n**📐 Klasik Profil**\n")
            lines.append(f"- Momentum 120g: `{r['mom_120']:+.2f}`  |  60g: `{r['mom_60']:+.2f}`  |  30g: `{r['mom_30']:+.2f}`")
            lines.append(f"- CV (sıkışıklık): `{r['cv_120']:.3f}`  |  Sıkışma: `{r['cv_compression']:.2f}`")
            lines.append(f"- Hacim ROC: `{r['v_roc']:+.2f}`  |  Göreli hacim: `{r['rel_vol_120']:.2f}`")
            lines.append(f"- Göreli güç vs piyasa: 60g `{r['rel_strength_60']:+.2f}`, 120g `{r['rel_strength_120']:+.2f}`")
            lines.append(f"- Sektörüne göre: 60g `{r['sector_rel_60']:+.2f}`, 120g `{r['sector_rel_120']:+.2f}`")

            # Öncü-Ardıl
            if leader:
                lines.append(f"\n**🕐 Tetiklenme Zamanlaması** (n={leader['n_leaders']} öncü ikiz)\n")
                lines.append(f"- Medyan: **+{leader['median_days']} gün** (IQR: {leader['p25_days']}-{leader['p75_days']}g)")
                lines.append(f"- Aralık: {leader['min_days']}-{leader['max_days']} gün")

                if leader.get('recent_leaders'):
                    lines.append(f"\n**🔥 Son 30-90 günde tetiklenmiş öncü ikizler ({len(leader['recent_leaders'])} adet):**")
                    for L in leader['recent_leaders'][:5]:
                        lines.append(f"  - {L['Ticker']} (setup: {L['SetupDate'].date() if hasattr(L['SetupDate'], 'date') else str(L['SetupDate'])[:10]}, tetik: {L['DaysAgo']}g önce)")

                lines.append(f"\n**Tahmin:** {ticker} mevcut kuruluma benzer geçmiş ikizleri ortalama +{leader['median_days']} gün içinde ralliyi başlatmış. Bu pencerede izlemeye değer.")
            else:
                lines.append(f"\n**🕐 Tetiklenme Zamanlaması:** Yeterli başarılı ikiz yok, zamanlama tahmin edilemiyor.")

            # Twin Divergence İzleme — post-entry trajectory referansı
            tdv = r.get('twin_div') or {}
            s_gain = tdv.get('success_gain_median')
            f_gain = tdv.get('failure_gain_median')
            n_s_used = tdv.get('n_success_used', 0)
            n_f_used = tdv.get('n_failure_used', 0)

            if (s_gain is not None and f_gain is not None
                and n_s_used >= TWIN_DIV_MIN_TWINS and n_f_used >= TWIN_DIV_MIN_TWINS):
                lines.append(f"\n**🎯 Twin Divergence İzleme Profili** (n_success={n_s_used}, n_failure={n_f_used})\n")
                lines.append(f"Bu hisseye girilirse, **60 gün boyunca** trajektori başarılı/başarısız ikiz medyanlarıyla kıyaslanır. "
                             f"Hisse 'Erken çıkış sınırı'nın altına düşerse failure pattern'ine kayıyor demektir.\n")
                lines.append(f"| Gün | Success medyan gain | Failure medyan gain | Erken çıkış sınırı |")
                lines.append(f"|---|---|---|---|")
                for d in [10, 20, 30, 45, 60]:
                    if d - 1 < len(s_gain) and d - 1 < len(f_gain):
                        sg = s_gain[d - 1] * 100
                        fg = f_gain[d - 1] * 100
                        mid = (sg + fg) / 2
                        lines.append(f"| {d} | {sg:+.1f}% | {fg:+.1f}% | < {mid:+.1f}% |")
            else:
                lines.append(f"\n**🎯 Twin Divergence İzleme:** Yeterli ikiz trajectory yok "
                             f"(n_success={n_s_used}, n_failure={n_f_used}, gerekli ≥{TWIN_DIV_MIN_TWINS}).")

            # Failed-twin analizi
            if failed and failed['n_failed'] > 0:
                lines.append(f"\n**⚠️ Risk Analizi** ({failed['n_failed']} başarısız ikiz)\n")
                lines.append(f"\nAynı görünüme sahip ama ralli yapmamış ikizler:\n")
                for F in failed['failed_list'][:3]:
                    lines.append(f"- `{F['Ticker']}` (setup: {F['SetupDate']}, DTW: {F['DTW']})")

                if failed['top_differentiators']:
                    lines.append(f"\n**Başarılı vs başarısız ikizler arası en büyük farklar:**")
                    lines.append(f"\n| Feature | Başarılı medyan | Başarısız medyan | Fark |")
                    lines.append(f"|---|---|---|---|")
                    for d in failed['top_differentiators'][:3]:
                        lines.append(
                            f"| `{d['feature']}` | {d['success_median']:+.3f} | "
                            f"{d['failure_median']:+.3f} | **{d['diff']:+.3f}** |"
                        )
                    top = failed['top_differentiators'][0]
                    # diff > 0 → başarılılar daha yüksek → başarısızlar DÜŞÜK
                    # diff < 0 → başarılılar daha düşük → başarısızlar YÜKSEK
                    direction = "düşük" if top['diff'] > 0 else "yüksek"
                    cand_val = r.get(top['feature'], None)
                    # B12: cand_val NaN olabilir → karşılaştırma öncesi guard
                    cand_is_valid = cand_val is not None and not (isinstance(cand_val, float) and pd.isna(cand_val))
                    cand_str = f"{cand_val:+.3f}" if cand_is_valid else "N/A"
                    if cand_is_valid:
                        proximity = ('başarısızlara yakın ⚠️'
                                     if abs(cand_val - top['failure_median']) < abs(cand_val - top['success_median'])
                                     else 'başarılılara yakın ✅')
                    else:
                        proximity = 'veri yok'
                    lines.append(
                        f"\n👉 *Başarısız ikizler tipik olarak daha **{direction} `{top['feature']}`** "
                        f"(medyan: {top['failure_median']:+.3f}) ile gelmiş. "
                        f"{ticker}'nin değeri: `{cand_str}` ({proximity}).*"
                    )

            lines.append("\n---")

    # Footer — B14: eğitim sayısı artifacts'tan alınır, hardcode değil
    n_training = sum(c['original_size'] for c in artifacts['cluster_info'].values())
    lines.append(f"\n\n## Metodoloji\n")
    _method_name = artifacts.get('clustering_method', artifacts.get('cluster_method', 'gmm')).upper()
    lines.append(f"- **Tarama:** {_method_name} kümeleme (eğitim: {n_training:,} bağımsız ralli, B+C geçiş tespiti)")
    lines.append(f"- **İkiz:** 3-kanal DTW (fiyat + log-hacim + range), Sakoe-Chiba window={DTW_WINDOW}, max_dist={MAX_DISTANCE_NDIM}")
    lines.append(f"- **Tetiklenme:** Setup tarihinden ilk %{int(RALLY_TRIGGER*100)}+ kazanca kadar geçen gün")
    lines.append(f"- **Failed-twin:** Aynı 120g profile sahip ama 252g içinde %20'den az getiri yapanlar")
    lines.append(f"\n*Bu rapor finansal tavsiye değildir.*")

    with open(out_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))


def generate_excel_report(report_data, out_path):
    """Küme bazlı Excel raporu."""
    with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
        # Tüm adaylar
        all_rows = []
        for r in report_data:
            leader = r.get('leader_info') or {}
            failed = r.get('failed_info') or {}
            all_rows.append({
                'Ticker':           r['Ticker'],
                'Karakter':         r['Karakter'],
                'Cluster':          r['Cluster'],
                'Cluster_WinRate':  r['Cluster_WinRate'],
                'Uyum_Skoru':       r['Uyum_Skoru'],
                'MinDist':          r.get('MinDist', None),
                # İkiz analizi
                'n_twins':          r['n_twins'],
                'n_twin_success':   r['n_twin_success'],
                'n_twin_failure':   r['n_twin_failure'],
                'twin_success_pct': r['n_twin_success'] / max(r['n_twins'], 1) * 100,
                'median_days':      leader.get('median_days'),
                'p25_days':         leader.get('p25_days'),
                'p75_days':         leader.get('p75_days'),
                'recent_leaders':   len(leader.get('recent_leaders', [])),
                'failed_twins':     failed.get('n_failed'),
                # Klasik feature'lar
                'mom_120':          r['mom_120'],
                'mom_60':           r['mom_60'],
                'cv_120':           r['cv_120'],
                'cv_compression':   r['cv_compression'],
                'v_roc':            r['v_roc'],
                'rel_strength_60':  r['rel_strength_60'],
                'sector_rel_60':    r['sector_rel_60'],
                'usd_mom_60':       r['usd_mom_60'],
            })
        df_all = pd.DataFrame(all_rows)
        # Twin başarısı birincil sıralama kriteri
        df_all = df_all.sort_values(['twin_success_pct', 'Uyum_Skoru'], ascending=False).reset_index(drop=True)
        df_all.to_excel(writer, sheet_name='Tum_Adaylar', index=False)

        # Küme bazlı sheet'ler — cluster_id ile ayırt et (K0/K2 aynı Karakter'e sahip olabilir)
        for cluster_id in sorted(df_all['Cluster'].unique()):
            subset = df_all[df_all['Cluster'] == cluster_id]
            if len(subset) == 0:
                continue
            cluster_name = subset['Karakter'].iloc[0]
            sheet = f"K{cluster_id}_{cluster_name}"[:31].replace('/', '-')
            subset.to_excel(writer, sheet_name=sheet, index=False)


# ─── Main ───────────────────────────────────────────────────────────────────

def run_full_pipeline(target_date=None):
    print("=" * 70)
    print("🚀 BIST 4-ADIMLI TAM ANALİZ PIPELINE'I")
    print("=" * 70)

    os.makedirs(REPORT_DIR, exist_ok=True)
    os.makedirs(os.path.join(REPORT_DIR, "scenarios"), exist_ok=True)

    # ADIM 1
    top_candidates, df_market, latest_date, artifacts = screen_candidates(target_date)

    # Havuzları yükle (ADIM 4 için lazım)
    df_success_pool = pl.read_parquet(SUCCESS_POOL_PATH).to_pandas()
    df_failure_pool = pl.read_parquet(FAILURE_POOL_PATH).to_pandas()

    # Pool cache (ADIM 2 için)
    print("\n🔄 ADIM 2 hazırlık: İkiz havuzu ön hesabı...")
    pool_cache = precompute_pool_curves(df_market, df_success_pool, df_failure_pool, window=120)

    # ADIM 2-4: Her aday için — B11: paralel DTW (joblib threads, Polars GIL-safe)
    # Analiz (DTW + timing + failed) paralel; matplotlib plot sıralı (Agg safe).
    print(f"\n🔬 ADIM 2-4: {len(top_candidates)} aday üzerinde derin analiz (paralel)...")

    def _analyze_one(cand_dict):
        """DTW + timing + failed + twin-divergence; plot YOK (ana thread'de yapılır)."""
        ticker = cand_dict['Ticker']
        target_curve = get_base_curve(df_market, ticker, latest_date)
        if target_curve is None:
            return None
        twins_df = find_twins(ticker, latest_date, target_curve, pool_cache)
        if twins_df is None:
            return None
        n_success = int((twins_df['Outcome'] == 'Success').sum())
        n_failure = int((twins_df['Outcome'] == 'Failure').sum())
        leader_info  = analyze_leader_timing(twins_df, df_market)
        failed_info  = analyze_failed_twins(twins_df, df_success_pool, df_failure_pool)

        # Twin Divergence: post-entry gain trajectory medyanları (raporlama için)
        s_gain, f_gain, n_s_gain, n_f_gain = compute_gain_curves(
            twins_df, df_market, window=TWIN_DIV_WINDOW
        )

        row = dict(cand_dict)
        row['n_twins']        = len(twins_df)
        row['n_twin_success'] = n_success
        row['n_twin_failure'] = n_failure
        row['leader_info']    = leader_info
        row['failed_info']    = failed_info
        row['twins_df']       = twins_df
        row['twin_div'] = {
            'success_gain_median': s_gain,
            'failure_gain_median': f_gain,
            'n_success_used':      n_s_gain,
            'n_failure_used':      n_f_gain,
        }
        return row

    cand_list = [cand.to_dict() for _, cand in top_candidates.iterrows()]
    # prefer="threads": dtaidistance C uzantısı GIL'i serbest bırakır → gerçek paralellik
    raw_results = Parallel(n_jobs=-1, prefer="threads")(
        jdelayed(_analyze_one)(c) for c in cand_list
    )

    # Sırayı koru, None'ları filtrele, ilerleme logla
    report_data = []
    for i, (row, cand_dict) in enumerate(zip(raw_results, cand_list), 1):
        ticker = cand_dict['Ticker']
        if row is None:
            print(f"   [{i:>2}/{len(cand_list)}] {ticker}... ⚠️  eğri/ikiz yok")
            continue
        leader_str = f"med+{row['leader_info']['median_days']}g" if row['leader_info'] else "yok"
        print(f"   [{i:>2}/{len(cand_list)}] {ticker}... ikiz={row['n_twins']}({row['n_twin_success']}✓/{row['n_twin_failure']}✗) {leader_str}")

        # Görsel — Agg backend thread-safe; ana thread'de sıralı yazıyoruz
        out_png = os.path.join(REPORT_DIR, "scenarios", f"{ticker}_scenario.png")
        try:
            plot_candidate_scenario(ticker, latest_date, row['twins_df'], df_market,
                                    row['leader_info'], out_png, twin_div=row.get('twin_div'))
        except Exception as e:
            print(f"      PNG hata: {e}")

        report_data.append(row)

    # Raporları üret
    print(f"\n📝 Rapor üretiliyor...")
    date_str = latest_date.strftime('%Y%m%d') if not isinstance(latest_date, str) else str(latest_date)[:10].replace('-', '')

    md_path = os.path.join(REPORT_DIR, f"Tam_Analiz_{date_str}.md")
    xlsx_path = os.path.join(REPORT_DIR, f"Tam_Analiz_{date_str}.xlsx")

    generate_markdown_report(report_data, latest_date, artifacts, md_path)
    generate_excel_report(report_data, xlsx_path)

    print(f"\n✅ TAMAMLANDI")
    print(f"   📄 Markdown : {md_path}")
    print(f"   📊 Excel    : {xlsx_path}")
    print(f"   🖼️  Grafikler: {os.path.join(REPORT_DIR, 'scenarios')}/ ({len(report_data)} PNG)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='BIST 4-Adımlı Tam Analiz')
    parser.add_argument('--date', type=str, default=None, help='Tarama tarihi (YYYY-MM-DD)')
    args = parser.parse_args()
    run_full_pipeline(target_date=args.date)
