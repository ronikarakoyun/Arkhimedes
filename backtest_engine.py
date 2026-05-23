"""
BIST 10-Yıllık Walk-Forward Backtest
=====================================
Strateji:
- Sinyal : B+C küme geçişleri (30 gün cooldown)
- Pozisyon: Max 5 paralel, her biri %20 ağırlık
- Exit   : Dinamik stop = min(peak - 4×ATR, peak × 0.80)
           → Oynak hisselerde ATR baskın (geniş stop, silkeleme'ye dayanır)
           → Sakin hisselerde floor devrede (en az %20 nefes garanti)
- Komisyon: %0.2 alış + %0.2 satış
- Slippage: Ertesi gün Popen'da işlem

Walk-Forward:
- Her yıl modeli o tarihteki bilgiyle yeniden eğit (look-ahead bias yok)
- 8 retrain epoch: 2018→2025
"""

import polars as pl
import pandas as pd
import numpy as np
import os
import warnings
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from datetime import datetime

from model_core import (
    apply_preprocessing,
    _remove_second_leg_rallies,
    _detect_cluster_transitions,
    _isolate_representative_setups,
    FEATURES_FOR_CLUSTERING,
    FEATURE_WEIGHTS,
    RALLY_WINDOW_DAYS,
    RALLY_GAIN_THRESHOLD,
    SECOND_LEG_COOLDOWN,
    FAILURE_GAIN_CEILING,
    CLUSTER_COOLDOWN_DAYS,
)

warnings.filterwarnings('ignore')

# ─── Backtest Parametreleri (config.py'den) ─────────────────────────────────
from config import (
    INITIAL_CAPITAL, MAX_POSITIONS, POSITION_WEIGHT, COMMISSION_RATE,
    ATR_MULTIPLIER, TRAILING_FLOOR_PCT, ATR_PERIOD,
    MAX_ADV_PCT, SLIPPAGE_BASE, SLIPPAGE_IMPACT,
    DB_PATH,
    FEATURES_PATH as FEAT_PATH,
    REPORT_BACKTEST_DIR as REPORT_DIR,
    TWIN_DIV_ENABLED, TWIN_DIV_WINDOW, TWIN_DIV_MIN_CHECK_DAYS,
    TWIN_DIV_CHECK_INTERVAL, TWIN_DIV_EXIT_THRESHOLD, TWIN_DIV_MIN_TWINS,
)
from common import compute_atr20, compute_adv20, compute_forward_relative_gain
from twin_matcher import get_base_curve, precompute_pool_curves, DTW_WINDOW, MAX_DISTANCE_NDIM
from twin_divergence import (
    get_post_curve, compute_twin_median_post_curves, twin_divergence_score,
)
from dtaidistance import dtw_ndim
from combined_filter import CombinedFilter, TECH_FEATURES
from macro_engine import load_macro_features
from fundamental_engine import load_fundamentals, snapshot_at
from config import (FINAL_P_THRESHOLD,
                    CLUSTER_METHOD as _CFG_CLUSTER_METHOD,
                    HDBSCAN_MIN_CLUSTER_SIZE, HDBSCAN_MIN_SAMPLES,
                    COMBINED_CV_WINDOW_YEARS,
                    ENV_FILTER_NDCG_THRESHOLD)
from combined_filter import _compute_relevance_labels

# Clustering yöntemi — env değişkeni config'i override eder (head-to-head için)
CLUSTER_METHOD = os.environ.get("CLUSTER_METHOD", _CFG_CLUSTER_METHOD).lower()
# HDBSCAN paramları da env ile override edilebilir (parametre taraması için)
HDBSCAN_MIN_CLUSTER_SIZE = int(os.environ.get("HDB_MCS", HDBSCAN_MIN_CLUSTER_SIZE))
HDBSCAN_MIN_SAMPLES = int(os.environ.get("HDB_MS", HDBSCAN_MIN_SAMPLES))
# Ranking skoru — "p_rally" (default: LambdaRank RankScore, K=10 lift=1.25x > WinRate 1.09x)
# | "winrate" (eski davranış, override için SCORE_MODE=winrate env değişkeni)
# T3.1'de binary P_rally kaybetti; LambdaRank ile yeniden devreye alındı.
SCORE_MODE = os.environ.get("SCORE_MODE", "p_rally").lower()
try:
    import hdbscan as hdbscan_lib
    HDBSCAN_AVAILABLE = True
except ImportError:
    hdbscan_lib = None
    HDBSCAN_AVAILABLE = False

BACKTEST_START     = pd.to_datetime("2016-05-23")
TRADING_START      = pd.to_datetime("2018-05-23")
BACKTEST_END       = pd.to_datetime("2026-05-23")

# Retrain takvimi
RETRAIN_DATES = [pd.to_datetime(f"{y}-05-23") for y in range(2018, 2026)]


# ─── Model Eğitimi (epoch bazlı) ────────────────────────────────────────────

def _train_combined_filter(train_df, macro_df, fund_df, verbose=True):
    """
    LambdaRank eğitimi — tüm train_df'i kullanır (binary pool yok).
    Her (Ticker, YearMonth) için tek gözlem; YearMonth içi relative rank → alaka 0-4.
    """
    # 1. future_max_gain'i olan satırlar (look-ahead güvenli, training_cutoff öncesi)
    labs = train_df[train_df['future_max_gain'].notna()].copy()
    labs['Date'] = pd.to_datetime(labs['Date'])
    labs['YearMonth'] = labs['Date'].dt.to_period('M').astype(str)

    # 2. Deduplikasyon: (Ticker, YearMonth) başına rastgele 1 gözlem
    # keep='last' ay sonu sapması yaratır; max(cv_compression) look-ahead bias içerir.
    # Rastgele örnekleme model'e homojen piyasa rejimi öğretir.
    labs = (labs.groupby(['Ticker', 'YearMonth'], group_keys=False)
               .sample(n=1, random_state=42)
               .reset_index(drop=True))

    if len(labs) < 200:
        if verbose:
            print(f"   ⚠️ LambdaRankFilter: yetersiz veri ({len(labs)} satır)")
        return None

    # 3. Feature join: makro + fundamental + teknik
    m_cols = [c for c in macro_df.columns if c.startswith('m_')]
    labs = labs.merge(macro_df, on='Date', how='left')

    snap = snapshot_at(fund_df, labs[['Ticker', 'Date']])
    labs = labs.merge(snap.drop_duplicates(['Ticker', 'Date']),
                      on=['Ticker', 'Date'], how='left')
    f_cols = [c for c in labs.columns
              if c.endswith('_z') or (c.startswith('f_') and c not in TECH_FEATURES)]

    # Teknik feature'lar labs'da zaten mevcut (train_df = features_db verileri)
    tech_keep = [c for c in TECH_FEATURES if c in labs.columns]

    # EVDS kesintisine karşı dayanıklılık: dropna yerine ffill + medyan fallback
    labs = labs.sort_values('Date')
    for col in m_cols:
        labs[col] = labs[col].ffill()
        if labs[col].isna().any():
            labs[col] = labs[col].fillna(labs[col].median())
    if len(labs) < 200:
        if verbose:
            print(f"   ⚠️ LambdaRankFilter: makro join sonrası yetersiz veri ({len(labs)} satır)")
        return None

    # 4. Alaka seviyesi etiketleri (0-4, per YearMonth)
    feat_cols = [c for c in m_cols + f_cols + tech_keep if c in labs.columns]
    feat_cols = list(dict.fromkeys(feat_cols))  # deduplikasyon, sıra koru
    X = labs[feat_cols].copy()
    relevance = _compute_relevance_labels(labs, 'future_max_gain', 'YearMonth')
    group_series = labs['YearMonth']

    # 5. LambdaRankFilter eğit
    try:
        cf = CombinedFilter(cv_window_years=COMBINED_CV_WINDOW_YEARS)
        cf.fit(X, relevance, labs['Date'], group_series)
        if verbose:
            print(f"   {cf.summary()}")
        return cf
    except Exception as e:
        if verbose:
            print(f"   ⚠️ LambdaRankFilter eğitim hatası: {e}")
        return None


def _assign_clusters(model_bundle, X):
    """X (scaled+weighted) → (labels, uyum_skoru). cluster_method'a göre dispatch.
    HDBSCAN: uyum = membership strength [0,1].
    GMM: uyum = formasyon olasılığı = max(predict_proba) [0,1].
    K-Means: uyum = 1/(1+min_centroid_distance)."""
    method = model_bundle.get('cluster_method')
    if method == 'hdbscan':
        labels, strengths = hdbscan_lib.approximate_predict(model_bundle['clusterer'], X)
        return np.asarray(labels), np.asarray(strengths, dtype=float)
    if method == 'gmm':
        proba = model_bundle['clusterer'].predict_proba(X)
        return proba.argmax(axis=1), proba.max(axis=1)
    km = model_bundle['clusterer']
    labels = km.predict(X)
    distances = km.transform(X)
    uyum = 1.0 / (1.0 + np.min(distances, axis=1))
    return labels, uyum


def train_at_date(df_feat_pd, retrain_date, verbose=True, macro_df=None, fund_df=None):
    """
    retrain_date - 252 gün cutoff ile clustering fit (K-Means ya da HDBSCAN).
    Cluster win-rate'leri de hesapla (training set üzerinde geçiş tabanlı).

    Returns: dict — cluster_method, clusterer, scaler, q_low, q_high, weight_vec,
             win_rates, best_k, twin pools, combined_filter
    """
    training_cutoff = retrain_date - pd.Timedelta(days=RALLY_WINDOW_DAYS)

    # Training setups: tarih ≤ cutoff (labels tam)
    train_df = df_feat_pd[df_feat_pd['Date'] <= training_cutoff].copy()
    train_df = train_df[train_df['mom_120'] < 0.70]
    train_df = train_df[train_df['mom_120'].notna()]
    train_df = train_df[train_df['future_max_gain'].notna()]

    if verbose:
        print(f"   Training cutoff: {training_cutoff.date()}, {len(train_df):,} setup")

    # Başarı havuzu: future_max_gain >= 1.0
    success_pd = train_df[train_df['future_max_gain'] >= RALLY_GAIN_THRESHOLD].copy()

    # Ay bazlı temsilci + ikinci-ayak temizliği (polars'a geri çevir)
    success_pl = pl.from_pandas(success_pd)
    success_pl = _isolate_representative_setups(success_pl)
    success_pl = _remove_second_leg_rallies(success_pl, SECOND_LEG_COOLDOWN)
    pd_success = success_pl.drop_nulls(subset=FEATURES_FOR_CLUSTERING).to_pandas()

    if len(pd_success) < 500:
        if verbose:
            print(f"   ⚠️  Yetersiz başarı havuzu ({len(pd_success)} < 500), bu epoch atlanır")
        return None

    # Preprocessing
    q_low  = pd_success[FEATURES_FOR_CLUSTERING].quantile(0.01).to_dict()
    q_high = pd_success[FEATURES_FOR_CLUSTERING].quantile(0.99).to_dict()
    success_pp = apply_preprocessing(pd_success, FEATURES_FOR_CLUSTERING, q_low, q_high)

    scaler = RobustScaler()
    X = scaler.fit_transform(success_pp[FEATURES_FOR_CLUSTERING])
    weight_vec = np.array([FEATURE_WEIGHTS[f] for f in FEATURES_FOR_CLUSTERING])
    X = X * weight_vec

    # ── Clustering: K-Means / HDBSCAN / GMM (config.CLUSTER_METHOD) ──
    cluster_method = CLUSTER_METHOD
    if cluster_method == "hdbscan" and not HDBSCAN_AVAILABLE:
        print("   ⚠️  hdbscan kütüphanesi yok — K-Means'e düşülüyor")
        cluster_method = "kmeans"

    if cluster_method == "hdbscan":
        clusterer = hdbscan_lib.HDBSCAN(
            min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
            min_samples=HDBSCAN_MIN_SAMPLES,
            prediction_data=True,
        ).fit(X)
        valid_labels = sorted(set(clusterer.labels_) - {-1})
        if len(valid_labels) < 2:
            if verbose:
                print(f"   ⚠️  HDBSCAN <2 küme üretti, bu epoch atlanır")
            return None
        best_k = len(valid_labels)
        best_score = float('nan')
    elif cluster_method == "gmm":
        from sklearn.mixture import GaussianMixture
        sample_size = min(5000, len(X))
        rng = np.random.default_rng(42)
        X_samp = X[rng.choice(X.shape[0], sample_size, replace=False)]
        best_k, best_bic = 3, float('inf')
        for k in range(3, 7):
            gm = GaussianMixture(n_components=k, covariance_type='full',
                                 max_iter=200, random_state=42).fit(X_samp)
            bic = gm.bic(X_samp)
            if bic < best_bic:
                best_bic, best_k = bic, k
        clusterer = GaussianMixture(n_components=best_k, covariance_type='full',
                                    max_iter=200, random_state=42).fit(X)
        best_score = float('nan')
    else:
        sample_size = min(5000, len(X))
        rng = np.random.default_rng(42)
        idx = rng.choice(X.shape[0], sample_size, replace=False)
        X_samp = X[idx]
        best_k, best_score = 3, -1.0
        for k in range(3, 6):
            km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_samp)
            s = silhouette_score(X_samp, km.labels_)
            if s > best_score:
                best_score, best_k = s, k
        clusterer = KMeans(n_clusters=best_k, random_state=42, n_init=10).fit(X)

    model_bundle_tmp = {'cluster_method': cluster_method, 'clusterer': clusterer}

    # pd_success'e Cluster label ekle (build_knowledge_base.py cluster_info için)
    success_pp = apply_preprocessing(pd_success, FEATURES_FOR_CLUSTERING, q_low, q_high)
    valid_s = success_pp[FEATURES_FOR_CLUSTERING].notna().all(axis=1)
    X_success_all = scaler.transform(success_pp[valid_s][FEATURES_FOR_CLUSTERING]) * weight_vec
    s_labels, _ = _assign_clusters(model_bundle_tmp, X_success_all)
    pd_success = pd_success[valid_s].copy()
    pd_success['Cluster'] = s_labels

    # Win-rate hesabı: training setteki tüm setup'ları cluster'la, B+C uygula
    train_clean = train_df.dropna(subset=FEATURES_FOR_CLUSTERING).copy()
    train_pp = apply_preprocessing(train_clean, FEATURES_FOR_CLUSTERING, q_low, q_high)
    valid_mask = train_pp[FEATURES_FOR_CLUSTERING].notna().all(axis=1)
    train_clean = train_clean[valid_mask].copy()
    train_pp = train_pp[valid_mask]
    X_train_all = scaler.transform(train_pp[FEATURES_FOR_CLUSTERING]) * weight_vec
    tc_labels, _ = _assign_clusters(model_bundle_tmp, X_train_all)
    train_clean['Cluster'] = tc_labels
    # HDBSCAN gürültüsü: noise satırları geçiş üretmez
    train_clean = train_clean[train_clean['Cluster'] != -1].copy()

    transitions = _detect_cluster_transitions(train_clean, CLUSTER_COOLDOWN_DAYS)
    win_rates = {}
    for c in sorted(transitions['Cluster'].unique()):
        tc = transitions[transitions['Cluster'] == c]
        n = len(tc)
        s = int((tc['future_max_gain'] >= RALLY_GAIN_THRESHOLD).sum())
        win_rates[int(c)] = s / n if n > 0 else 0.0

    if verbose:
        tag = {"hdbscan": "H", "gmm": "G"}.get(cluster_method, "K")
        head = (f"k={best_k} silhouette={best_score:.3f}" if cluster_method == "kmeans"
                else f"{cluster_method} n_küme={best_k}")
        wr_str = " | ".join([f"{tag}{c}:%{wr*100:.0f}({(transitions['Cluster']==c).sum()})"
                             for c, wr in win_rates.items()])
        print(f"   {head}  {wr_str}")

    # Twin Divergence için pool — bu epoch'a kadar olan success ve failure havuzları
    # (look-ahead bias yok: tüm setupları training_cutoff'a kadar)
    failure_pd = train_df[train_df['future_max_gain'] < FAILURE_GAIN_CEILING].copy()
    # Ay bazlı temsilci
    if len(failure_pd) > 0:
        failure_pl = pl.from_pandas(failure_pd)
        failure_pl = _isolate_representative_setups(failure_pl)
        failure_pd = failure_pl.to_pandas()
    # success_pd zaten yukarıda temizlenmiş halde — pd_success'i kullan
    twin_pool_success = pd_success[['Ticker', 'Date']].copy()
    twin_pool_failure = failure_pd[['Ticker', 'Date']].copy() if len(failure_pd) > 0 else pd.DataFrame(columns=['Ticker','Date'])

    # CombinedFilter (macro+fundamental+technical → RankScore)
    combined_filter = None
    if macro_df is not None and fund_df is not None:
        combined_filter = _train_combined_filter(
            train_df, macro_df, fund_df, verbose=verbose
        )

    return {
        'cluster_method': cluster_method,
        'clusterer':  clusterer,
        'scaler':     scaler,
        'q_low':      q_low,
        'q_high':     q_high,
        'weight_vec': weight_vec,
        'win_rates':  win_rates,
        'best_k':     best_k,
        'twin_pool_success': twin_pool_success,
        'twin_pool_failure': twin_pool_failure,
        'combined_filter':   combined_filter,
        # build_knowledge_base.py için ek veri
        'transitions':           transitions,      # geçiş DataFrame (cluster_info hesabı için)
        'success_with_clusters': pd_success,       # başarı havuzu + Cluster sütunu (eklendi)
    }


# ─── Sinyal Üretimi (epoch bazlı) ────────────────────────────────────────────

def precompute_signals_for_epoch(model_bundle, df_feat_pd, start_date, end_date, prev_clusters,
                                  epoch_idx=0, macro_df=None, fund_df=None):
    """
    Epoch [start_date, end_date) içindeki B+C geçişlerini çıkar.
    prev_clusters: dict[ticker] = (last_cluster, last_kept_date_per_cluster)
                   epoch sınırlarında durum aktarımı için.
    epoch_idx: bu epoch'un sırası — simulate_trading'de pool_cache lookup için.

    Returns: signals_df, updated_prev_clusters
    """
    # Bu epoch'taki tüm setup'lar (önceki epoch'tan biraz geri al — cooldown için)
    lookback = pd.Timedelta(days=CLUSTER_COOLDOWN_DAYS + 5)
    epoch_df = df_feat_pd[
        (df_feat_pd['Date'] >= start_date - lookback) &
        (df_feat_pd['Date'] < end_date)
    ].dropna(subset=FEATURES_FOR_CLUSTERING).copy()

    if len(epoch_df) == 0:
        return pd.DataFrame(), prev_clusters

    # Cluster predict
    pp = apply_preprocessing(epoch_df, FEATURES_FOR_CLUSTERING,
                              model_bundle['q_low'], model_bundle['q_high'])
    valid = pp[FEATURES_FOR_CLUSTERING].notna().all(axis=1)
    epoch_df = epoch_df[valid].copy()
    pp = pp[valid]

    X = model_bundle['scaler'].transform(pp[FEATURES_FOR_CLUSTERING]) * model_bundle['weight_vec']
    labels, uyum = _assign_clusters(model_bundle, X)
    epoch_df['Cluster'] = labels
    epoch_df['UyumSkoru'] = uyum
    # HDBSCAN gürültüsü: noise satırları geçiş üretmez (build_knowledge_base ile tutarlı)
    epoch_df = epoch_df[epoch_df['Cluster'] != -1].copy()
    if len(epoch_df) == 0:
        return pd.DataFrame(), prev_clusters

    # B+C tespiti — manuel (prev_clusters durumunu hesaba katarak)
    epoch_df = epoch_df.sort_values(['Ticker', 'Date']).reset_index(drop=True)
    signals = []

    for ticker, group in epoch_df.groupby('Ticker', sort=False):
        group = group.sort_values('Date')
        state = prev_clusters.get(ticker, {'last_cluster': None, 'last_kept': {}})
        prev_cluster = state['last_cluster']
        last_kept_per_cluster = dict(state['last_kept'])  # copy

        for _, row in group.iterrows():
            cur_cluster = row['Cluster']
            cur_date = row['Date']
            if cur_date < start_date:
                # cooldown bölgesi — sadece state güncelle
                prev_cluster = cur_cluster
                continue

            is_transition = (prev_cluster is None) or (cur_cluster != prev_cluster)
            if is_transition:
                last_kept = last_kept_per_cluster.get(cur_cluster)
                if last_kept is None or (cur_date - last_kept).days > CLUSTER_COOLDOWN_DAYS:
                    signals.append({
                        'Date':       cur_date,
                        'Ticker':     ticker,
                        'Cluster':    int(cur_cluster),
                        'WinRate':    model_bundle['win_rates'].get(int(cur_cluster), 0.0),
                        'UyumSkoru':  float(row['UyumSkoru']),
                        'EpochIdx':   epoch_idx,
                    })
                    last_kept_per_cluster[cur_cluster] = cur_date
            prev_cluster = cur_cluster

        prev_clusters[ticker] = {'last_cluster': prev_cluster, 'last_kept': last_kept_per_cluster}

    signals_df = pd.DataFrame(signals)
    if len(signals_df) > 0:
        signals_df['Score'] = signals_df['WinRate'] * 0.7 + signals_df['UyumSkoru'] * 0.3

        # CombinedFilter ile P_rally hesabı
        cf = model_bundle.get('combined_filter')
        if cf is not None and cf.is_useful() and macro_df is not None and fund_df is not None:
            sig_dates = pd.to_datetime(signals_df['Date'])
            sig_df = signals_df[['Ticker', 'Date']].copy()
            sig_df['Date'] = sig_dates

            # Macro join
            m_cols = [c for c in macro_df.columns if c.startswith('m_')]
            tmp = sig_df.merge(macro_df, on='Date', how='left')

            # Fundamental snapshot
            snap = snapshot_at(fund_df, sig_df)
            tmp = tmp.merge(snap.drop_duplicates(['Ticker', 'Date']),
                            on=['Ticker', 'Date'], how='left')

            # Technical features (epoch_df'ten al)
            tech_keep = [c for c in TECH_FEATURES if c in epoch_df.columns]
            tech_join = epoch_df[['Ticker', 'Date'] + tech_keep].drop_duplicates(['Ticker', 'Date'])
            tech_join['Date'] = pd.to_datetime(tech_join['Date'])
            tmp = tmp.merge(tech_join, on=['Ticker', 'Date'], how='left')

            # P_rally prediction
            feat_cols = cf.feature_cols
            missing = [c for c in feat_cols if c not in tmp.columns]
            for c in missing:
                tmp[c] = np.nan
            X_pred = tmp[feat_cols]
            try:
                signals_df['P_rally'] = cf.predict_rank_score(X_pred)
            except Exception as e:
                print(f"   ⚠️ RankScore hesaplanamadı: {e}")
                signals_df['P_rally'] = np.nan
        else:
            signals_df['P_rally'] = np.nan

        # LambdaRank RankScore → birincil sıralama otoritesi.
        # LambdaRank (K=10: 1.25x lift) WinRate-tabanlı skoru (1.09x) geçti.
        # P_rally=NaN ise (LambdaRankFilter eğitilemeyen epoch) WinRate'e düşülür.
        # SCORE_MODE=winrate env değişkeniyle eski davranışa geçilebilir.
        if SCORE_MODE == "p_rally":
            signals_df['Score'] = signals_df['P_rally'].fillna(signals_df['Score'])

    return signals_df, prev_clusters


# ─── Trading Simülasyonu ────────────────────────────────────────────────────

def _compute_stop_level(peak_close, atr20):
    """
    Dinamik stop: ATR-based + floor
      stop_atr   = peak - K × ATR
      stop_floor = peak × (1 - FLOOR_PCT)
      effective  = min(stop_atr, stop_floor)   # daha aşağı olan → daha çok nefes
    Yani: ATR sıkıştığında floor devreye girer (en az %20 nefes garanti).
    """
    floor_stop = peak_close * (1.0 - TRAILING_FLOOR_PCT)
    if atr20 is None or pd.isna(atr20) or atr20 <= 0:
        return floor_stop
    atr_stop = peak_close - ATR_MULTIPLIER * atr20
    return min(atr_stop, floor_stop)


def _find_top_twins(ticker, target_date, target_curve, pool_cache, top_k=20):
    """run_full_analysis.find_twins'in backtest-içi minimal versiyonu."""
    matches = []
    target_date_pd = pd.Timestamp(target_date)
    for entry in pool_cache:
        if entry['Ticker'] == ticker and abs((pd.Timestamp(entry['Date']) - target_date_pd).days) < 60:
            continue
        distance = dtw_ndim.distance_fast(
            target_curve, entry['Curve'],
            window=DTW_WINDOW, max_dist=MAX_DISTANCE_NDIM,
        )
        if distance < MAX_DISTANCE_NDIM:
            matches.append({
                'Ticker':       entry['Ticker'],
                'Date':         pd.Timestamp(entry['Date']),
                'DTW_Distance': distance,
                'Outcome':      entry['Outcome'],
            })
    if not matches:
        return None
    return pd.DataFrame(matches).sort_values('DTW_Distance').head(top_k).reset_index(drop=True)


def simulate_trading(signals_df, df_market_pd, trading_days, df_market_pl=None, epoch_pool_caches=None, p_threshold=None):
    """
    Günlük loop. df_market_pd: pandas DataFrame [Date, Ticker, Popen, Pclose, Phigh, Plow]

    Twin Divergence (opsiyonel): df_market_pl + epoch_pool_caches verilirse,
    pozisyon açıldığında o pozisyon için twin matching yapılır ve günlük
    checkpoint'te score < TWIN_DIV_EXIT_THRESHOLD ise erken çıkış uygulanır.

    Returns: trades_df, daily_nav_df
    """
    # Hızlı lookup için pivot table: Date × Ticker → fiyat
    print("   📐 ATR(20) + ADV(20) hesaplanıyor ve pivot tabloları hazırlanıyor...")
    df_market_pd = compute_atr20(df_market_pd)
    df_market_pd = compute_adv20(df_market_pd)
    df_market_pd = df_market_pd[df_market_pd['Date'].isin(trading_days)].copy()
    pivot_close = df_market_pd.pivot_table(index='Date', columns='Ticker', values='Pclose')
    pivot_open  = df_market_pd.pivot_table(index='Date', columns='Ticker', values='Popen')
    pivot_atr   = df_market_pd.pivot_table(index='Date', columns='Ticker', values='ATR20')
    pivot_adv   = df_market_pd.pivot_table(index='Date', columns='Ticker', values='ADV20')

    twin_div_active = (
        TWIN_DIV_ENABLED and df_market_pl is not None
        and epoch_pool_caches is not None and len(epoch_pool_caches) > 0
    )
    if twin_div_active:
        print(f"   🎯 Twin Divergence Exit aktif (threshold={TWIN_DIV_EXIT_THRESHOLD}, window={TWIN_DIV_WINDOW}g)")

    cash = INITIAL_CAPITAL
    positions = {}      # ticker → {'entry_date', 'entry_price', 'qty', 'peak_close', 'cluster', 'score', + twin_div fields}
    trades = []
    daily_nav = []
    n_twin_div_exits = 0
    n_filter_vetoes = 0
    n_capacity_capped = 0

    # Sinyalleri tarihe göre indexle
    signals_by_date = {d: g for d, g in signals_df.groupby('Date')} if len(signals_df) > 0 else {}

    for i, today in enumerate(trading_days):
        if today < TRADING_START:
            # warmup: NAV = cash, ama trading yok
            daily_nav.append({'Date': today, 'NAV': cash, 'cash': cash, 'positions': 0})
            continue

        # Hızlı erişim
        try:
            close_today = pivot_close.loc[today]
            atr_today   = pivot_atr.loc[today]
            adv_today   = pivot_adv.loc[today] if today in pivot_adv.index else pd.Series(dtype=float)
        except KeyError:
            # Bu gün için hiç fiyat yoksa
            nav = cash + sum(p['qty'] * p['peak_close'] for p in positions.values())
            daily_nav.append({'Date': today, 'NAV': nav, 'cash': cash, 'positions': len(positions)})
            continue

        # 1. EXIT KONTROLÜ
        exits = []
        exit_reasons = {}   # ticker → 'Stop' veya 'TwinDiv'

        for ticker, pos in positions.items():
            cur_close = close_today.get(ticker, np.nan)
            if pd.isna(cur_close):
                continue
            # Peak güncelle
            if cur_close > pos['peak_close']:
                pos['peak_close'] = cur_close

            # 1a. Twin Divergence checkpoint (opsiyonel, ATR stop'tan ÖNCE)
            if twin_div_active and pos.get('success_median') is not None:
                days_held = (today - pd.Timestamp(pos['entry_date'])).days
                last_check = pos.get('last_div_check_day', 0)
                if (TWIN_DIV_MIN_CHECK_DAYS <= days_held < TWIN_DIV_WINDOW
                    and days_held - last_check >= TWIN_DIV_CHECK_INTERVAL):
                    cand_curve = get_post_curve(
                        df_market_pl, ticker, pd.Timestamp(pos['entry_date']), days_held
                    )
                    if cand_curve is not None:
                        s_slice = pos['success_median'][:days_held]
                        f_slice = pos['failure_median'][:days_held]
                        score, _, _ = twin_divergence_score(cand_curve, s_slice, f_slice)
                        pos['last_div_check_day'] = days_held
                        pos['last_div_score'] = score
                        if score is not None and score < TWIN_DIV_EXIT_THRESHOLD:
                            exits.append(ticker)
                            exit_reasons[ticker] = 'TwinDiv'
                            continue   # ATR stop kontrolünü atla

            # 1b. Dinamik stop seviyesi (ATR + floor)
            atr = atr_today.get(ticker, np.nan)
            stop_level = _compute_stop_level(pos['peak_close'], atr)
            if cur_close < stop_level:
                exits.append(ticker)
                exit_reasons[ticker] = 'Stop'

        # Exit'leri ertesi gün açılışta gerçekleştir
        next_day = trading_days[i+1] if i+1 < len(trading_days) else None
        if next_day is not None:
            try:
                open_next = pivot_open.loc[next_day]
            except KeyError:
                open_next = pd.Series(dtype=float)
        else:
            open_next = pd.Series(dtype=float)

        for ticker in exits:
            pos = positions[ticker]
            reason = exit_reasons.get(ticker, 'Stop')
            exit_price = open_next.get(ticker, close_today.get(ticker, pos['peak_close']))
            if pd.isna(exit_price) or exit_price <= 0:
                exit_price = pos['peak_close']
            # Slippage — satışta fiyat aşağı kayar (işlem_TL / ADV ile orantılı)
            adv = adv_today.get(ticker, np.nan)
            slip = SLIPPAGE_BASE
            if not pd.isna(adv) and adv > 0:
                slip += SLIPPAGE_IMPACT * min((pos['qty'] * exit_price) / adv, 1.0)
            eff_exit = exit_price * (1.0 - slip)
            gross = pos['qty'] * eff_exit
            commission = gross * COMMISSION_RATE
            net_proceeds = gross - commission
            cash += net_proceeds

            entry_cost = pos['qty'] * pos['entry_price'] * (1 + COMMISSION_RATE)
            gain_pct = (net_proceeds / entry_cost) - 1
            peak_gain = (pos['peak_close'] / pos['entry_price']) - 1
            holding = (today - pos['entry_date']).days

            trades.append({
                'Ticker':       ticker,
                'EntryDate':    pos['entry_date'],
                'ExitDate':     next_day if next_day else today,
                'EntryPrice':   pos['entry_price'],
                'ExitPrice':    exit_price,
                'PeakClose':    pos['peak_close'],
                'GainPct':      gain_pct,
                'PeakGainPct':  peak_gain,
                'HoldingDays':  holding,
                'Cluster':      pos['cluster'],
                'WinRate':      pos['win_rate'],
                'Score':        pos['score'],
                'ExitReason':   reason,
                'TwinDivScore': pos.get('last_div_score'),
                'P_rally':      pos.get('p_rally'),
            })
            if reason == 'TwinDiv':
                n_twin_div_exits += 1
            del positions[ticker]

        # 2. YENİ SİNYALLER
        if today in signals_by_date and len(positions) < MAX_POSITIONS:
            todays_signals = signals_by_date[today].sort_values(
                ['Score', 'Ticker'], ascending=[False, True], kind='stable')

            for _, sig in todays_signals.iterrows():
                if len(positions) >= MAX_POSITIONS:
                    break
                ticker = sig['Ticker']
                if ticker in positions:
                    continue  # zaten açık

                # CombinedFilter veto kontrolü
                p_rally = sig.get('P_rally', np.nan)
                _thr = p_threshold if p_threshold is not None else FINAL_P_THRESHOLD
                if _thr > 0 and not pd.isna(p_rally) and p_rally < _thr:
                    n_filter_vetoes += 1
                    continue

                # Ertesi gün açılış fiyatı
                entry_price = open_next.get(ticker, np.nan)
                if pd.isna(entry_price) or entry_price <= 0:
                    continue

                # Pozisyon büyüklüğü: NAV'in %20'si — likidite tavanıyla sınırlı
                cur_nav = cash + sum(
                    p['qty'] * close_today.get(t, p['peak_close'])
                    for t, p in positions.items()
                )
                target_alloc = cur_nav * POSITION_WEIGHT
                # Likidite tavanı: pozisyon ≤ ADV20 × MAX_ADV_PCT
                adv = adv_today.get(ticker, np.nan)
                has_adv = (not pd.isna(adv)) and adv > 0
                if has_adv:
                    cap = adv * MAX_ADV_PCT
                    if target_alloc > cap:
                        target_alloc = cap
                        n_capacity_capped += 1
                # Slippage — alışta fiyat yukarı kayar
                slip = SLIPPAGE_BASE
                if has_adv:
                    slip += SLIPPAGE_IMPACT * min(target_alloc / adv, 1.0)
                eff_entry = entry_price * (1.0 + slip)
                qty = (target_alloc / (1 + COMMISSION_RATE)) / eff_entry
                cost = qty * eff_entry * (1 + COMMISSION_RATE)

                if cost > cash:
                    # Yeterli nakit yok, daha küçük al
                    qty = (cash / (1 + COMMISSION_RATE)) / eff_entry
                    cost = qty * eff_entry * (1 + COMMISSION_RATE)
                    if qty <= 0:
                        continue

                cash -= cost
                positions[ticker] = {
                    'entry_date':  next_day if next_day else today,
                    'entry_price': eff_entry,
                    'qty':         qty,
                    'peak_close':  eff_entry,
                    'cluster':     sig['Cluster'],
                    'win_rate':    sig['WinRate'],
                    'score':       sig['Score'],
                    'success_median':    None,
                    'failure_median':    None,
                    'last_div_check_day': 0,
                    'last_div_score':    None,
                    'p_rally':           float(p_rally) if not pd.isna(p_rally) else None,
                }

                # Twin Divergence pre-compute: bu pozisyon için success/failure medyan
                # trajectory'lerini sinyalin epoch'undaki pool ile hesapla
                if twin_div_active:
                    epoch_idx = int(sig.get('EpochIdx', 0))
                    pool_cache = epoch_pool_caches[epoch_idx] if epoch_idx < len(epoch_pool_caches) else None
                    if pool_cache:
                        target_curve = get_base_curve(df_market_pl, ticker, today, window=120)
                        if target_curve is not None:
                            twins = _find_top_twins(ticker, today, target_curve, pool_cache, top_k=20)
                            if twins is not None and len(twins) >= TWIN_DIV_MIN_TWINS:
                                s_med, f_med, ns, nf = compute_twin_median_post_curves(
                                    twins, df_market_pl, window=TWIN_DIV_WINDOW
                                )
                                # Her gruptan en az MIN_TWINS varsa aktive et
                                if (s_med is not None and f_med is not None
                                    and ns >= TWIN_DIV_MIN_TWINS and nf >= TWIN_DIV_MIN_TWINS):
                                    positions[ticker]['success_median'] = s_med
                                    positions[ticker]['failure_median'] = f_med

        # 3. NAV güncellemesi
        nav = cash + sum(
            p['qty'] * (close_today.get(t, p['peak_close']) if not pd.isna(close_today.get(t, np.nan)) else p['peak_close'])
            for t, p in positions.items()
        )
        daily_nav.append({'Date': today, 'NAV': nav, 'cash': cash, 'positions': len(positions)})

    # Backtest sonu: kalan pozisyonları kapat (son kapanış fiyatından)
    last_day = trading_days[-1]
    last_close = pivot_close.loc[last_day]
    for ticker, pos in list(positions.items()):
        exit_price = last_close.get(ticker, pos['peak_close'])
        if pd.isna(exit_price) or exit_price <= 0:
            exit_price = pos['peak_close']
        gross = pos['qty'] * exit_price
        commission = gross * COMMISSION_RATE
        cash += gross - commission

        entry_cost = pos['qty'] * pos['entry_price'] * (1 + COMMISSION_RATE)
        gain_pct = ((gross - commission) / entry_cost) - 1
        peak_gain = (pos['peak_close'] / pos['entry_price']) - 1
        holding = (last_day - pos['entry_date']).days

        trades.append({
            'Ticker':       ticker,
            'EntryDate':    pos['entry_date'],
            'ExitDate':     last_day,
            'EntryPrice':   pos['entry_price'],
            'ExitPrice':    exit_price,
            'PeakClose':    pos['peak_close'],
            'GainPct':      gain_pct,
            'PeakGainPct':  peak_gain,
            'HoldingDays':  holding,
            'Cluster':      pos['cluster'],
            'WinRate':      pos['win_rate'],
            'Score':        pos['score'],
            'ExitReason':   'OpenAtEnd',
            'TwinDivScore': pos.get('last_div_score'),
            'P_rally':      pos.get('p_rally'),
        })
        del positions[ticker]

    if twin_div_active:
        print(f"   🎯 Twin Divergence ile erken çıkan trade sayısı: {n_twin_div_exits}")
    if n_filter_vetoes > 0:
        print(f"   🚫 CombinedFilter ile veto edilen sinyal sayısı: {n_filter_vetoes}")
    if n_capacity_capped > 0:
        print(f"   💧 Likidite tavanıyla küçültülen pozisyon sayısı: {n_capacity_capped}")

    trades_df = pd.DataFrame(trades)
    daily_nav_df = pd.DataFrame(daily_nav)
    return trades_df, daily_nav_df


# ─── Metrikler ───────────────────────────────────────────────────────────────

def _deflated_cagr(nav, deflator):
    """nav: Date-indexli Series. deflator: Date-indexli Series (CPI ya da USDTRY).
    Deflate edilmiş seri üzerinden CAGR. deflator yoksa None."""
    if deflator is None or len(deflator) == 0:
        return None
    d = deflator.reindex(nav.index, method='ffill')
    real = (nav / d).dropna()
    if len(real) < 2 or real.iloc[0] <= 0:
        return None
    yrs = (real.index[-1] - real.index[0]).days / 365.25
    if yrs <= 0:
        return None
    return (real.iloc[-1] / real.iloc[0]) ** (1/yrs) - 1


def compute_metrics(daily_nav_df, trades_df, benchmark_df, label="",
                    cpi_series=None, usd_series=None):
    """CAGR, Sharpe, MaxDD, Calmar, WinRate, ProfitFactor + Reel/USD CAGR."""
    nav = daily_nav_df.set_index('Date')['NAV']
    bench = benchmark_df.set_index('Date')['NAV']

    # Sadece trading döneminden
    nav = nav[nav.index >= TRADING_START]
    bench = bench[bench.index >= TRADING_START]

    years = (nav.index[-1] - nav.index[0]).days / 365.25
    cagr = (nav.iloc[-1] / nav.iloc[0]) ** (1/years) - 1
    bench_cagr = (bench.iloc[-1] / bench.iloc[0]) ** (1/years) - 1

    daily_ret = nav.pct_change().dropna()
    sharpe = daily_ret.mean() / daily_ret.std() * np.sqrt(252) if daily_ret.std() > 0 else 0

    running_max = nav.cummax()
    drawdown = (nav - running_max) / running_max
    max_dd = drawdown.min()

    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    # Trade istatistikleri
    if len(trades_df) > 0:
        wins = trades_df[trades_df['GainPct'] > 0]
        losses = trades_df[trades_df['GainPct'] <= 0]
        win_rate = len(wins) / len(trades_df)
        avg_win = wins['GainPct'].mean() if len(wins) > 0 else 0
        avg_loss = losses['GainPct'].mean() if len(losses) > 0 else 0
        profit_factor = wins['GainPct'].sum() / abs(losses['GainPct'].sum()) if len(losses) > 0 and losses['GainPct'].sum() != 0 else float('inf')
        avg_hold = trades_df['HoldingDays'].mean()
    else:
        win_rate = avg_win = avg_loss = profit_factor = avg_hold = 0

    # Reel (TÜFE) ve USD bazlı CAGR — nominal TL getirinin enflasyondan arınmış hali
    real_cagr  = _deflated_cagr(nav, cpi_series)
    usd_cagr   = _deflated_cagr(nav, usd_series)
    b_real     = _deflated_cagr(bench, cpi_series)
    b_usd      = _deflated_cagr(bench, usd_series)

    return {
        'Initial':       nav.iloc[0],
        'Final':         nav.iloc[-1],
        'CAGR':          cagr,
        'BenchCAGR':     bench_cagr,
        'Alpha':         cagr - bench_cagr,
        'RealCAGR':      real_cagr,
        'USDCagr':       usd_cagr,
        'BenchRealCAGR': b_real,
        'BenchUSDCagr':  b_usd,
        'Sharpe':        sharpe,
        'MaxDD':         max_dd,
        'Calmar':        calmar,
        'WinRate':       win_rate,
        'AvgWin':        avg_win,
        'AvgLoss':       avg_loss,
        'ProfitFactor':  profit_factor,
        'AvgHoldDays':   avg_hold,
        'TotalTrades':   len(trades_df),
        'Years':         years,
    }


# ─── Benchmark ───────────────────────────────────────────────────────────────

def build_benchmark_nav(df_market_pd, trading_days):
    """XU100 buy-and-hold NAV. İlk gün INITIAL_CAPITAL ile alır, tutmaya devam eder."""
    xu100 = df_market_pd[df_market_pd['Ticker'] == 'XU100'][['Date', 'Pclose']].copy()
    xu100 = xu100.sort_values('Date').drop_duplicates('Date')

    # Trading dönemindeki ilk fiyat
    xu100_trade = xu100[xu100['Date'] >= TRADING_START]
    if len(xu100_trade) == 0:
        # XU100 yoksa equally weighted "tüm hisse" benchmark
        print("   ⚠️  XU100 verisi yok, sentetik benchmark kullanılacak")
        return None

    initial_price = xu100_trade.iloc[0]['Pclose']
    shares = INITIAL_CAPITAL / initial_price

    bench_rows = []
    last_price = initial_price
    xu100_dict = dict(zip(xu100['Date'], xu100['Pclose']))
    for d in trading_days:
        if d < TRADING_START:
            bench_rows.append({'Date': d, 'NAV': INITIAL_CAPITAL})
            continue
        p = xu100_dict.get(d, last_price)
        last_price = p
        bench_rows.append({'Date': d, 'NAV': shares * p})

    return pd.DataFrame(bench_rows)


# ─── Rapor + Görseller ───────────────────────────────────────────────────────

def plot_equity_curve(nav_df, bench_df, out_path):
    fig, ax = plt.subplots(figsize=(13, 6))
    nav_t = nav_df[nav_df['Date'] >= TRADING_START]
    bench_t = bench_df[bench_df['Date'] >= TRADING_START]
    ax.semilogy(nav_t['Date'], nav_t['NAV'], color='#2E7D32', linewidth=2, label='Strateji')
    ax.semilogy(bench_t['Date'], bench_t['NAV'], color='#666', linewidth=1.5, label='XU100 B&H', alpha=0.7)
    ax.axhline(INITIAL_CAPITAL, color='gray', linestyle=':', alpha=0.5)
    ax.set_title("Equity Curve — Strateji vs XU100 (Log)", fontsize=13, fontweight='bold')
    ax.set_xlabel("Tarih")
    ax.set_ylabel("Portföy Değeri (TL, log)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def plot_drawdown(nav_df, out_path):
    fig, ax = plt.subplots(figsize=(13, 5))
    nav_t = nav_df[nav_df['Date'] >= TRADING_START].set_index('Date')['NAV']
    rm = nav_t.cummax()
    dd = (nav_t - rm) / rm * 100
    ax.fill_between(dd.index, dd.values, 0, color='#C62828', alpha=0.4)
    ax.plot(dd.index, dd.values, color='#C62828', linewidth=1)
    ax.set_title(f"Drawdown — Max: %{dd.min():.1f}", fontsize=13, fontweight='bold')
    ax.set_ylabel("Drawdown (%)")
    ax.set_xlabel("Tarih")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def plot_monthly_returns(nav_df, out_path):
    nav_t = nav_df[nav_df['Date'] >= TRADING_START].set_index('Date')['NAV']
    monthly = nav_t.resample('ME').last().pct_change().dropna() * 100
    df = pd.DataFrame({'Year': monthly.index.year, 'Month': monthly.index.month, 'Return': monthly.values})
    pivot = df.pivot_table(index='Year', columns='Month', values='Return')

    fig, ax = plt.subplots(figsize=(11, 5))
    vmax = max(abs(pivot.min().min()), abs(pivot.max().max()))
    im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto', vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(12))
    ax.set_xticklabels(['Oca','Şub','Mar','Nis','May','Haz','Tem','Ağu','Eyl','Eki','Kas','Ara'])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i,j]
            if not pd.isna(v):
                ax.text(j, i, f"%{v:.1f}", ha='center', va='center', fontsize=8,
                        color='black' if abs(v) < vmax*0.6 else 'white')
    plt.colorbar(im, label='Aylık Getiri (%)')
    ax.set_title("Aylık Getiri Heatmap", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def generate_report(metrics, trades_df, daily_nav_df, bench_df, out_md, out_xlsx):
    """Markdown raporu + Excel."""
    lines = []
    lines.append(f"# BIST Backtest Raporu — {datetime.now().strftime('%Y-%m-%d')}\n")
    lines.append(f"*10 yıllık walk-forward backtest, B+C sinyal, 5 pozisyon, %20 trailing stop*\n")

    lines.append("\n## Strateji Parametreleri\n")
    lines.append("| Parametre | Değer |")
    lines.append("|---|---|")
    lines.append(f"| Periyot | {TRADING_START.date()} → {BACKTEST_END.date()} ({metrics['Years']:.1f} yıl) |")
    lines.append(f"| Sinyal | Tüm kümelerin B+C geçişleri (cooldown {CLUSTER_COOLDOWN_DAYS}g) |")
    lines.append(f"| Max pozisyon | {MAX_POSITIONS} hisse |")
    lines.append(f"| Pozisyon ağırlığı | %{POSITION_WEIGHT*100:.0f} |")
    lines.append(f"| Stop | Dinamik: min(peak − {ATR_MULTIPLIER:.0f}×ATR({ATR_PERIOD}), peak × {1-TRAILING_FLOOR_PCT:.2f}) |")
    lines.append(f"| Komisyon | %{COMMISSION_RATE*100:.2f} her yön |")
    lines.append(f"| Başlangıç sermayesi | {INITIAL_CAPITAL:,.0f} TL |")

    lines.append("\n## 📈 Toplam Performans\n")
    lines.append("| Metrik | Strateji | XU100 B&H |")
    lines.append("|---|---|---|")
    lines.append(f"| **Başlangıç** | {metrics['Initial']:,.0f} TL | {INITIAL_CAPITAL:,.0f} TL |")
    lines.append(f"| **Final NAV** | **{metrics['Final']:,.0f} TL** | {bench_df[bench_df['Date']<=BACKTEST_END]['NAV'].iloc[-1]:,.0f} TL |")
    lines.append(f"| **CAGR (nominal TL)** | **%{metrics['CAGR']*100:.1f}** | %{metrics['BenchCAGR']*100:.1f} |")
    if metrics.get('RealCAGR') is not None:
        br = f"%{metrics['BenchRealCAGR']*100:.1f}" if metrics.get('BenchRealCAGR') is not None else "—"
        lines.append(f"| **CAGR (reel, TÜFE)** | **%{metrics['RealCAGR']*100:.1f}** | {br} |")
    if metrics.get('USDCagr') is not None:
        bu = f"%{metrics['BenchUSDCagr']*100:.1f}" if metrics.get('BenchUSDCagr') is not None else "—"
        lines.append(f"| **CAGR (USD bazlı)** | **%{metrics['USDCagr']*100:.1f}** | {bu} |")
    lines.append(f"| **Alpha (nominal)** | **{metrics['Alpha']*100:+.1f}** pp | — |")
    lines.append(f"| **Sharpe** | {metrics['Sharpe']:.2f} | — |")
    lines.append(f"| **Max DD** | %{metrics['MaxDD']*100:.1f} | — |")
    lines.append(f"| **Calmar** | {metrics['Calmar']:.2f} | — |")

    lines.append("\n## 📊 Trade İstatistikleri\n")
    lines.append("| Metrik | Değer |")
    lines.append("|---|---|")
    lines.append(f"| Toplam trade | {metrics['TotalTrades']} |")
    lines.append(f"| Win rate | %{metrics['WinRate']*100:.1f} |")
    lines.append(f"| Avg win | %{metrics['AvgWin']*100:.1f} |")
    lines.append(f"| Avg loss | %{metrics['AvgLoss']*100:.1f} |")
    lines.append(f"| Profit factor | {metrics['ProfitFactor']:.2f} |")
    lines.append(f"| Ort. holding | {metrics['AvgHoldDays']:.0f} gün |")

    # Yıllık getiri
    nav_t = daily_nav_df[daily_nav_df['Date'] >= TRADING_START].set_index('Date')['NAV']
    yearly = nav_t.resample('YE').last().pct_change().dropna() * 100
    bench_t = bench_df[bench_df['Date'] >= TRADING_START].set_index('Date')['NAV']
    bench_yearly = bench_t.resample('YE').last().pct_change().dropna() * 100

    lines.append("\n## 📅 Yıllık Getiri\n")
    lines.append("| Yıl | Strateji | XU100 | Alpha |")
    lines.append("|---|---|---|---|")
    for date, ret in yearly.items():
        y = date.year
        bench_ret = bench_yearly.get(date, 0)
        lines.append(f"| {y} | %{ret:+.1f} | %{bench_ret:+.1f} | %{ret-bench_ret:+.1f} |")

    # En iyi/kötü tradeler
    if len(trades_df) > 0:
        lines.append("\n## 🏆 En İyi 10 Trade\n")
        lines.append("| # | Hisse | Entry | Exit | Gain | Peak | Hold | K |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for i, t in enumerate(trades_df.nlargest(10, 'GainPct').itertuples(), 1):
            lines.append(
                f"| {i} | **{t.Ticker}** | {pd.to_datetime(t.EntryDate).date()} | "
                f"{pd.to_datetime(t.ExitDate).date()} | **%{t.GainPct*100:+.1f}** | "
                f"%{t.PeakGainPct*100:+.1f} | {t.HoldingDays}g | {t.Cluster} |"
            )

        lines.append("\n## 💀 En Kötü 10 Trade\n")
        lines.append("| # | Hisse | Entry | Exit | Gain | Peak | Hold | K |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for i, t in enumerate(trades_df.nsmallest(10, 'GainPct').itertuples(), 1):
            lines.append(
                f"| {i} | {t.Ticker} | {pd.to_datetime(t.EntryDate).date()} | "
                f"{pd.to_datetime(t.ExitDate).date()} | **%{t.GainPct*100:+.1f}** | "
                f"%{t.PeakGainPct*100:+.1f} | {t.HoldingDays}g | {t.Cluster} |"
            )

        lines.append("\n## 🧮 Küme Bazlı Trade Dağılımı\n")
        lines.append("| Küme | Trade | Win | Avg Gain | Avg Peak |")
        lines.append("|---|---|---|---|---|")
        for c in sorted(trades_df['Cluster'].unique()):
            sub = trades_df[trades_df['Cluster'] == c]
            wins = (sub['GainPct'] > 0).sum()
            lines.append(
                f"| K{c} | {len(sub)} | %{wins/len(sub)*100:.1f} | "
                f"%{sub['GainPct'].mean()*100:+.1f} | %{sub['PeakGainPct'].mean()*100:+.1f} |"
            )

    lines.append("\n---\n*Bu rapor finansal tavsiye değildir. Slippage, vergi, repo etkileri modellenmemiştir.*")

    with open(out_md, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))

    # Excel
    with pd.ExcelWriter(out_xlsx, engine='openpyxl') as writer:
        if len(trades_df) > 0:
            trades_df.sort_values('EntryDate').to_excel(writer, sheet_name='Trades', index=False)
        daily_nav_df.to_excel(writer, sheet_name='Daily_NAV', index=False)
        # Monthly returns matrix
        nav_t = daily_nav_df[daily_nav_df['Date'] >= TRADING_START].set_index('Date')['NAV']
        monthly = nav_t.resample('ME').last().pct_change().dropna() * 100
        mdf = pd.DataFrame({'Date': monthly.index, 'MonthlyReturn%': monthly.values})
        mdf.to_excel(writer, sheet_name='Monthly_Returns', index=False)


# ─── Ana Akış ────────────────────────────────────────────────────────────────

def run_backtest():
    print("=" * 70)
    print("🚀 BIST 10-Yıllık Walk-Forward Backtest")
    print(f"📅 Periyot: {BACKTEST_START.date()} → {BACKTEST_END.date()}")
    print(f"💼 Trading: {TRADING_START.date()} → {BACKTEST_END.date()}")
    print("=" * 70)

    os.makedirs(REPORT_DIR, exist_ok=True)

    # Veri yükle
    print("\n📥 Veri yükleniyor...")
    df_market = pl.read_parquet(DB_PATH).with_columns(pl.col("Date").cast(pl.Datetime("ms")))
    df_market_pd = df_market.to_pandas()
    df_market_pd['Date'] = pd.to_datetime(df_market_pd['Date'])

    df_feat = pl.read_parquet(FEAT_PATH).with_columns(pl.col("Date").cast(pl.Datetime("ms")))

    # future_max_gain: XU100-göreli ileri kazanç (common helper — tek kaynak)
    print("⏳ future_max_gain (XU100-göreli) hesaplanıyor...")
    df_outcomes = compute_forward_relative_gain(df_feat, RALLY_WINDOW_DAYS)

    df_feat_pd = df_outcomes.to_pandas()
    df_feat_pd['Date'] = pd.to_datetime(df_feat_pd['Date'])

    # Trading günleri (XU100'den)
    trading_days = sorted(df_market_pd[df_market_pd['Ticker'] == 'XU100']['Date'].unique())
    trading_days = [d for d in trading_days if BACKTEST_START <= d <= BACKTEST_END]
    print(f"   {len(trading_days):,} trading günü")

    # ── Walk-Forward Retrain + Signal Pre-compute ──
    print(f"\n🔄 Walk-Forward Retrain ({len(RETRAIN_DATES)} epoch)...")
    all_signals = []
    prev_clusters = {}
    epoch_pool_caches = []  # Twin Divergence için: her epoch'un pool_cache'i

    # df_market polars (twin matching için get_base_curve, get_post_curve)
    df_market_pl_for_twin = (
        pl.read_parquet(DB_PATH).with_columns(pl.col("Date").cast(pl.Date))
        if TWIN_DIV_ENABLED else None
    )

    # CombinedFilter veri kaynakları (bir kez yükle, her epoch'ta kullan)
    print("\n📊 CombinedFilter veri kaynakları yükleniyor...")
    try:
        cf_macro_df = load_macro_features()
        cf_macro_df['Date'] = pd.to_datetime(cf_macro_df['Date'])
        cf_fund_df = load_fundamentals()
        print(f"   ✅ macro {cf_macro_df.shape}, fund {cf_fund_df.shape}")
    except Exception as e:
        print(f"   ⚠️ Yüklenemedi: {e}. CombinedFilter devre dışı kalacak.")
        cf_macro_df, cf_fund_df = None, None

    epoch_boundaries = list(RETRAIN_DATES) + [BACKTEST_END + pd.Timedelta(days=1)]
    for i, retrain_date in enumerate(RETRAIN_DATES):
        epoch_start = retrain_date
        epoch_end = epoch_boundaries[i+1]
        print(f"\n   Epoch {i+1}/{len(RETRAIN_DATES)}: {retrain_date.date()} (sinyal: {epoch_start.date()} → {epoch_end.date()})")

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

        # Twin Divergence için bu epoch'un havuzu — pool_cache pre-compute
        if TWIN_DIV_ENABLED:
            success_pool = model_bundle['twin_pool_success']
            failure_pool = model_bundle['twin_pool_failure']
            print(f"   🎯 Twin pool: {len(success_pool):,} success + {len(failure_pool):,} failure → eğri hesabı...")
            pool_cache = precompute_pool_curves(df_market_pl_for_twin, success_pool, failure_pool, window=120)
            epoch_pool_caches.append(pool_cache)
        else:
            epoch_pool_caches.append(None)

    all_signals_df = pd.concat(all_signals, ignore_index=True) if all_signals else pd.DataFrame()
    print(f"\n📊 Toplam: {len(all_signals_df):,} B+C geçiş sinyali")

    # ── Trading Simülasyonu ──
    print("\n💼 Trading simülasyonu başlıyor...")
    trades_df, daily_nav_df = simulate_trading(
        all_signals_df, df_market_pd, trading_days,
        df_market_pl=df_market_pl_for_twin,
        epoch_pool_caches=epoch_pool_caches,
    )
    print(f"   {len(trades_df)} trade gerçekleşti")

    # Benchmark
    bench_df = build_benchmark_nav(df_market_pd, trading_days)
    if bench_df is None:
        # Strateji NAV'inden sentetik
        bench_df = daily_nav_df[['Date']].copy()
        bench_df['NAV'] = INITIAL_CAPITAL

    # Reel/USD getiri deflatörleri — CPI (macro) ve USDTRY (market_db)
    cpi_series = None
    if cf_macro_df is not None and 'cpi_level' in cf_macro_df.columns:
        _c = cf_macro_df[['Date', 'cpi_level']].dropna()
        cpi_series = _c.set_index('Date')['cpi_level']
    usd_series = None
    _usd = df_market_pd[df_market_pd['Ticker'] == 'USDTRY'][['Date', 'Pclose']].dropna()
    if len(_usd) > 0:
        usd_series = _usd.set_index('Date')['Pclose'].sort_index()

    # Metrikler
    print("\n📈 Metrikler hesaplanıyor...")
    metrics = compute_metrics(daily_nav_df, trades_df, bench_df,
                              cpi_series=cpi_series, usd_series=usd_series)

    print("\n" + "="*60)
    print("📈 SONUÇLAR")
    print("="*60)
    print(f"  Initial   : {metrics['Initial']:>15,.0f} TL")
    print(f"  Final     : {metrics['Final']:>15,.0f} TL")
    print(f"  CAGR      : {metrics['CAGR']*100:>14.1f}%  (nominal TL)")
    if metrics.get('RealCAGR') is not None:
        print(f"  CAGR Reel : {metrics['RealCAGR']*100:>14.1f}%  (TÜFE'den arındırılmış)")
    if metrics.get('USDCagr') is not None:
        print(f"  CAGR USD  : {metrics['USDCagr']*100:>14.1f}%  (USD bazlı)")
    print(f"  Sharpe    : {metrics['Sharpe']:>15.2f}")
    print(f"  Max DD    : {metrics['MaxDD']*100:>14.1f}%")
    print(f"  Calmar    : {metrics['Calmar']:>15.2f}")
    print(f"  Win Rate  : {metrics['WinRate']*100:>14.1f}%")
    print(f"  Total Tr. : {metrics['TotalTrades']:>15}")
    print(f"  Avg Hold  : {metrics['AvgHoldDays']:>13.0f}g")
    print()
    print(f"  XU100 B&H : CAGR %{metrics['BenchCAGR']*100:.1f} (nominal)", end="")
    if metrics.get('BenchUSDCagr') is not None:
        print(f"  | USD %{metrics['BenchUSDCagr']*100:.1f}")
    else:
        print()
    print(f"  Alpha     : %{metrics['Alpha']*100:+.1f}")

    # Görseller + rapor
    print("\n📝 Rapor üretiliyor...")
    plot_equity_curve(daily_nav_df, bench_df, os.path.join(REPORT_DIR, "equity_curve.png"))
    plot_drawdown(daily_nav_df, os.path.join(REPORT_DIR, "drawdown_chart.png"))
    try:
        plot_monthly_returns(daily_nav_df, os.path.join(REPORT_DIR, "monthly_returns_heatmap.png"))
    except Exception as e:
        print(f"   ⚠️  Monthly heatmap hatası: {e}")

    generate_report(
        metrics, trades_df, daily_nav_df, bench_df,
        os.path.join(REPORT_DIR, "Backtest_Sonuc.md"),
        os.path.join(REPORT_DIR, "Backtest_Sonuc.xlsx"),
    )

    # Sinyalleri de kaydet (debug için)
    all_signals_df.to_parquet(os.path.join(REPORT_DIR, "all_signals.parquet"))

    print(f"\n✅ TAMAMLANDI")
    print(f"   📄 {REPORT_DIR}/Backtest_Sonuc.md")
    print(f"   📊 {REPORT_DIR}/Backtest_Sonuc.xlsx")
    print(f"   🖼️  equity_curve.png, drawdown_chart.png, monthly_returns_heatmap.png")


if __name__ == "__main__":
    run_backtest()
