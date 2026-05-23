import polars as pl
import numpy as np
import pandas as pd
import joblib
from common import compute_forward_relative_gain
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import os
import warnings

warnings.filterwarnings('ignore')

try:
    import hdbscan as hdbscan_lib
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False
    print("⚠️  hdbscan paketi yüklü değil; sadece K-Means kullanılacak. "
          "Yüklemek için: pip install hdbscan")

from config import (
    FEATURES_PATH, SUCCESS_POOL_PATH, FAILURE_POOL_PATH,
    TRANSITIONS_PATH, SETUPS_PATH, ARTIFACTS_PATH,
)

FEATURES_FOR_CLUSTERING = [
    "cv_120", "cv_60", "cv_compression",        # fiyat sıkışıklığı (3 zaman dilimi)
    "vol_120", "rel_vol_120",                    # volatilite + göreli vol
    "mom_120", "mom_60", "mom_30",               # momentum (3 zaman dilimi)
    "v_roc",                                     # hacim trendi
    "vwap_dist_avg",                             # VWAP mesafesi
    "xu100_mom_60", "xu100_above_ma200", "xu100_drawdown",  # piyasa rejimi
    "usd_mom_30", "usd_mom_60",                             # döviz baskısı
    "rel_strength_60", "rel_strength_120",                  # hissenin piyasaya göreli gücü
    "xbank_mom_60", "xusin_mom_60",                         # sektör bağlamı (XBANK, XUSIN)
    "sector_rel_60", "sector_rel_120",                      # hissenin sektörüne göreli gücü
]

FEATURE_WEIGHTS = {
    "cv_120":            1.0,
    "cv_60":             0.8,
    "cv_compression":    1.2,
    "vol_120":           1.0,
    "rel_vol_120":       1.2,
    "mom_120":           2.5,
    "mom_60":            1.5,
    "mom_30":            1.0,
    "v_roc":             0.4,
    "vwap_dist_avg":     0.8,
    "xu100_mom_60":      1.5,
    "xu100_above_ma200": 2.0,
    "xu100_drawdown":    1.5,
    "usd_mom_30":        1.2,
    "usd_mom_60":        1.0,
    "rel_strength_60":   2.0,   # hisse piyasayı geçiyor mu? — güçlü sinyal
    "rel_strength_120":  1.5,
    "xbank_mom_60":      1.0,
    "xusin_mom_60":      1.0,
    "sector_rel_60":     1.8,   # hisse sektörünü geçiyor mu?
    "sector_rel_120":    1.3,
}

RALLY_WINDOW_DAYS = 252        # Rallinin gerçekleşmesi için max süre (1 yıl)
# Hedef artık XU100-GÖRELİ (common.compute_forward_relative_gain). future_max_gain
# = pencere içinde XU100'e göre ulaşılan en yüksek üstün performans (daima ≥0).
RALLY_GAIN_THRESHOLD = 0.50    # Ralli: en iyi noktada XU100'ü ≥%50 geçti
SECOND_LEG_COOLDOWN = 60       # 1. ralliden sonra kaç gün içindeki 2. ralli sayılmaz
FAILURE_GAIN_CEILING = 0.10    # Başarısızlık: en iyi noktada bile XU100'ü <%10 geçti
CLUSTER_COOLDOWN_DAYS = 30     # Aynı (hisse, grup) çifti için minimum gün arası
HDBSCAN_MIN_CLUSTER_SIZE = 60  # Küme oluşturmak için minimum nokta sayısı
HDBSCAN_MIN_SAMPLES      = 5   # Yoğunluk eşiği — azalt → daha az noise, daha fazla küme


# ─── Veri Hazırlama ─────────────────────────────────────────────────────────

def apply_preprocessing(df, features, q_low, q_high):
    """Winsorization + log dönüşümü. q_low/q_high dışarıdan verilir."""
    df_copy = df.copy()
    for f in features:
        df_copy[f] = df_copy[f].clip(lower=q_low[f], upper=q_high[f])
        if f in ['cv_120', 'cv_60', 'vol_120', 'vol_60', 'rel_vol_120']:
            df_copy[f] = np.log1p(df_copy[f])
    return df_copy


def _remove_second_leg_rallies(success_df, cooldown_days=SECOND_LEG_COOLDOWN):
    """Bir hissenin ilk rallisinin ardından cooldown_days içindeki 2. ralliyi eler."""
    success_pd = success_df.to_pandas().sort_values(["Ticker", "Date"]).reset_index(drop=True)
    keep_indices = []
    for _, group in success_pd.groupby("Ticker", sort=False):
        group = group.sort_values("Date")
        last_kept_date = None
        for idx, row in group.iterrows():
            current_date = pd.to_datetime(row["Date"])
            if last_kept_date is None or (current_date - last_kept_date).days > cooldown_days:
                keep_indices.append(idx)
                last_kept_date = current_date
    removed = len(success_pd) - len(keep_indices)
    print(f"   ℹ️  {removed:,} ikinci-ayak ralli kaydı elendi.")
    return pl.from_pandas(success_pd.loc[keep_indices].reset_index(drop=True))


def _detect_cluster_transitions(setups_pd, cooldown_days=CLUSTER_COOLDOWN_DAYS):
    """
    Her hisse için sadece şu satırları tutar:
    1. Bir gruba YENİ giriş (önceki satırın grubu farklı, ya da ilk satır)
    2. AYNI (hisse, grup) için son tutulan tarih > cooldown_days önce

    B (geçiş anı) + C (cooldown) birleşimi: rejim girişlerini yakalar,
    A→B→A→B oscillation'ı filtreler.
    """
    setups_pd = setups_pd.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    keep_indices = []

    for _, group in setups_pd.groupby("Ticker", sort=False):
        group = group.sort_values("Date")
        prev_cluster = None
        last_kept_per_cluster = {}  # {cluster_id: last_kept_date}

        for idx, row in group.iterrows():
            cur_cluster = row["Cluster"]
            cur_date = pd.to_datetime(row["Date"])

            is_transition = (prev_cluster is None) or (cur_cluster != prev_cluster)

            if is_transition:
                last_kept = last_kept_per_cluster.get(cur_cluster)
                if last_kept is None or (cur_date - last_kept).days > cooldown_days:
                    keep_indices.append(idx)
                    last_kept_per_cluster[cur_cluster] = cur_date

            prev_cluster = cur_cluster

    return setups_pd.loc[keep_indices].reset_index(drop=True)


def _isolate_representative_setups(pool_df):
    """Aynı hissenin aynı ayındaki en sessiz (en düşük cv) setup'ını tutar."""
    return (
        pool_df.with_columns([
            pl.col("Date").dt.year().alias("year"),
            pl.col("Date").dt.month().alias("month")
        ])
        .sort(["Ticker", "year", "month", "cv_120", "Date"])
        .group_by(["Ticker", "year", "month"], maintain_order=True)
        .first()
        .drop(["year", "month"])
    )


# ─── Otomatik İsimlendirme ──────────────────────────────────────────────────

def _signature_from_centroid(centroid_dict, feature_quartiles):
    """Centroid'in her feature için Düşük/Orta/Yüksek etiketini döndürür."""
    sig = {}
    for f, val in centroid_dict.items():
        q25, _, q75 = feature_quartiles[f]
        if val < q25:
            sig[f] = "Düşük"
        elif val > q75:
            sig[f] = "Yüksek"
        else:
            sig[f] = "Orta"
    return sig


def _name_from_signature(sig, win_rate):
    """İmzadan açıklayıcı isim üretir, win-rate'e göre kalite etiketi ekler."""
    parts = []

    if sig['cv_120'] == "Düşük":
        parts.append("Sıkışık")
    elif sig['cv_120'] == "Yüksek":
        parts.append("Gevşek")

    if sig['vol_120'] == "Yüksek":
        parts.append("Yüksek Vol")
    elif sig['vol_120'] == "Düşük":
        parts.append("Sakin")

    if sig['mom_120'] == "Düşük":
        parts.append("Dipte")
    elif sig['mom_120'] == "Yüksek":
        parts.append("Trendde")

    if sig['v_roc'] == "Yüksek":
        parts.append("Hacim Patlaması")
    elif sig['v_roc'] == "Düşük":
        parts.append("Hacim Düşüyor")

    if not parts:
        parts = ["Karma Profil"]

    base = " + ".join(parts)
    tag = " 🌟" if win_rate >= 0.65 else " ⚠️" if win_rate <= 0.35 else ""
    return base + tag


# ─── Küme İstatistikleri ────────────────────────────────────────────────────

def _compute_cluster_stats(transitions_df, cluster_labels, pd_success_labeled):
    """
    Her küme (label) için eğitim üye sayısı + geçiş win-rate istatistiklerini
    döndürür. Gürültü (-1) etiketi atlanır.

    Returns:
        dict  →  { cluster_id: { orig_count, trans_count, trans_success,
                                  trans_failure, win_rate } }
    """
    stats = {}
    for c in sorted(set(cluster_labels) - {-1}):
        orig_count = int((pd_success_labeled['Cluster'] == c).sum())
        trans_c    = transitions_df[transitions_df['Cluster'] == c]
        tc         = len(trans_c)
        ts         = int((trans_c['future_max_gain'] >= RALLY_GAIN_THRESHOLD).sum())
        tf         = int((trans_c['future_max_gain'] < FAILURE_GAIN_CEILING).sum())
        stats[c]   = dict(
            orig_count    = orig_count,
            trans_count   = tc,
            trans_success = ts,
            trans_failure = tf,
            win_rate      = ts / tc if tc > 0 else 0.0,
        )
    return stats


# ─── Ana Akış (DEPRECATED) ──────────────────────────────────────────────────
# ⚠️  BU FONKSİYON ESKİDİR. KULLANIMAYIN.
# Güncel bilgi tabanı eğitimi için: venv/bin/python3 build_knowledge_base.py
# (GMM + LambdaRank, CombinedFilter — tam kapasite eğitim)
# Bu fonksiyon eski K-Means tabanlı artifacts üretir ve modern sistemi BOZAR.

def build_knowledge_base():
    print("🧠 Ralli-Odaklı Bilgi Tabanı kuruluyor...")

    if not os.path.exists(FEATURES_PATH):
        print(f"❌ Hata: {FEATURES_PATH} yok. Önce feature_engine.py çalıştırın.")
        return

    df_feat = pl.read_parquet(FEATURES_PATH).sort(["Ticker", "Date"])

    # 1. Gelecek getiriyi hesapla (tüm setups için)
    print(f"⏳ {RALLY_WINDOW_DAYS} günlük maks. getiri hesaplanıyor...")
    # Hedef: XU100-göreli ileri kazanç (common.compute_forward_relative_gain)
    df_outcomes = compute_forward_relative_gain(df_feat, RALLY_WINDOW_DAYS)

    # Lookahead bias: son 252 günün setup'larında future_max_gain eksik pencereli
    # olduğundan bu tarihler analizden çıkarılır (eğitim ve win-rate için)
    max_date = df_outcomes["Date"].max()
    cutoff_date = max_date - pl.duration(days=RALLY_WINDOW_DAYS)

    # Tüm geçerli setup'lar (nötr dahil), son 252 gün hariç
    setups = df_outcomes.filter(
        (pl.col("mom_120") < 0.70) &
        (pl.col("mom_120").is_not_null()) &
        (pl.col("Date") <= cutoff_date)
    )
    print(f"   📅 Lookahead bias filtresi: {max_date} - {RALLY_WINDOW_DAYS}g = {cutoff_date} cutoff"
          f" → {len(setups):,} setup")

    # 2. Başarı / Başarısızlık havuzları
    print("⚖️  Başarı ve başarısızlık havuzları ayrılıyor...")
    success_raw = setups.filter(pl.col("future_max_gain") >= RALLY_GAIN_THRESHOLD)
    failure_raw = setups.filter(pl.col("future_max_gain") < FAILURE_GAIN_CEILING)

    # 3. Ay bazlı temsilci seç + ikinci ralli temizliği (sadece başarı)
    success_clean = _isolate_representative_setups(success_raw)
    failure_clean = _isolate_representative_setups(failure_raw)

    print(f"🔍 İkinci ralli temizleniyor (cooldown: {SECOND_LEG_COOLDOWN} gün)...")
    success_clean = _remove_second_leg_rallies(success_clean, SECOND_LEG_COOLDOWN)

    print(f"🌟 {len(success_clean):,} bağımsız ralli, {len(failure_clean):,} başarısız setup.")

    # ── ADIM 1: K-Means SADECE ralli yapan hisselerle eğitilir ──────────────
    # Gruplar "rallinin nasıl göründüğünü" öğrenir, başarısızlık onları kirletmez.

    pd_success = success_clean.drop_nulls(subset=FEATURES_FOR_CLUSTERING).to_pandas()

    # Clipping sınırları ve feature kuartilleri sadece başarı havuzundan
    q_low  = pd_success[FEATURES_FOR_CLUSTERING].quantile(0.01).to_dict()
    q_high = pd_success[FEATURES_FOR_CLUSTERING].quantile(0.99).to_dict()

    feature_quartiles = {}
    for f in FEATURES_FOR_CLUSTERING:
        feature_quartiles[f] = (
            pd_success[f].quantile(0.25),
            pd_success[f].quantile(0.50),
            pd_success[f].quantile(0.75),
        )

    success_pp = apply_preprocessing(pd_success, FEATURES_FOR_CLUSTERING, q_low, q_high)
    scaler = RobustScaler()
    X_success = scaler.fit_transform(success_pp[FEATURES_FOR_CLUSTERING])

    # Feature ağırlıkları: nüans analizine göre mom_120 baskın, v_roc yanıltıcı
    weight_vec = np.array([FEATURE_WEIGHTS[f] for f in FEATURES_FOR_CLUSTERING])
    X_success = X_success * weight_vec

    # Optimal k (silhouette, 3..10)
    print("🤖 Optimal küme sayısı aranıyor (silhouette, sadece ralli verisi)...")
    sample_size = min(8000, len(X_success))
    rng = np.random.default_rng(42)
    idx = rng.choice(X_success.shape[0], sample_size, replace=False)
    X_sample = X_success[idx]

    best_k, best_score = 4, -1.0
    for k in range(3, 11):
        km = KMeans(n_clusters=k, random_state=42, n_init='auto').fit(X_sample)
        score = silhouette_score(X_sample, km.labels_)
        print(f"   k={k:>2}  silhouette={score:.4f}")
        if score > best_score:
            best_score, best_k = score, k

    print(f"\n✅ Optimal k = {best_k} (skor: {best_score:.4f})")
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init='auto').fit(X_success)
    pd_success['Cluster'] = kmeans.labels_

    # ── ADIM 1b: HDBSCAN — yoğunluk tabanlı alternatif ──────────────────────
    hdb_clusterer      = None
    hdb_labels_success = None
    hdb_unique         = []
    n_hdb_clusters     = 0

    if HDBSCAN_AVAILABLE:
        print(f"\n🧪 HDBSCAN deneniyor "
              f"(min_cluster_size={HDBSCAN_MIN_CLUSTER_SIZE}, "
              f"min_samples={HDBSCAN_MIN_SAMPLES})...")
        hdb_clusterer = hdbscan_lib.HDBSCAN(
            min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
            min_samples=HDBSCAN_MIN_SAMPLES,
            prediction_data=True,
        )
        hdb_clusterer.fit(X_success)
        hdb_labels_success = hdb_clusterer.labels_
        hdb_unique         = sorted(set(hdb_labels_success) - {-1})
        n_hdb_clusters     = len(hdb_unique)
        n_noise            = int((hdb_labels_success == -1).sum())
        print(f"   ℹ️  {n_hdb_clusters} küme, {n_noise:,} gürültü noktası "
              f"({n_noise / len(hdb_labels_success) * 100:.1f}% noise)")

    # ── ADIM 2: Her iki model TÜM setup'lara uygulanır ──────────────────────
    print(f"\n📡 Eğitilmiş gruplar tüm tarihsel setup'lara uygulanıyor ({len(setups):,} kayıt)...")
    setups_pd = setups.drop_nulls(subset=FEATURES_FOR_CLUSTERING).to_pandas()
    setups_pp = apply_preprocessing(setups_pd, FEATURES_FOR_CLUSTERING, q_low, q_high)
    # Preprocessing sonrası kalan NaN'ları temizle (log1p negatif değer üretebilir)
    valid_mask = setups_pp[FEATURES_FOR_CLUSTERING].notna().all(axis=1)
    setups_pd  = setups_pd[valid_mask].copy()
    setups_pp  = setups_pp[valid_mask]
    print(f"   ({valid_mask.sum():,} geçerli kayıt, {(~valid_mask).sum():,} NaN atlandı)")
    X_all = scaler.transform(setups_pp[FEATURES_FOR_CLUSTERING]) * weight_vec

    setups_pd['Cluster_KM'] = kmeans.predict(X_all)

    if hdb_clusterer is not None:
        hdb_all_labels, _ = hdbscan_lib.approximate_predict(hdb_clusterer, X_all)
        setups_pd['Cluster_HDB'] = hdb_all_labels

    # ── ADIM 3: Her model için B+C geçiş tespiti ─────────────────────────────
    print(f"\n🔄 K-Means rejim geçişleri (cooldown: {CLUSTER_COOLDOWN_DAYS} gün)...")
    setups_km = setups_pd.copy()
    setups_km['Cluster'] = setups_km['Cluster_KM']
    transitions_km = _detect_cluster_transitions(setups_km, CLUSTER_COOLDOWN_DAYS)
    print(f"   ℹ️  {len(setups_km):,} satır → {len(transitions_km):,} K-Means geçişi")

    transitions_hdb = None
    hdb_stats       = {}
    hdb_wavg        = 0.0

    if hdb_clusterer is not None and n_hdb_clusters > 0:
        print(f"🔄 HDBSCAN rejim geçişleri (gürültü hariç)...")
        setups_hdb = setups_pd[setups_pd['Cluster_HDB'] != -1].copy()
        setups_hdb['Cluster'] = setups_hdb['Cluster_HDB']
        transitions_hdb = _detect_cluster_transitions(setups_hdb, CLUSTER_COOLDOWN_DAYS)
        print(f"   ℹ️  {len(setups_hdb):,} gürültüsüz satır → "
              f"{len(transitions_hdb):,} HDBSCAN geçişi")

    # ── ADIM 4: Win-rate karşılaştırması → kazananı belirle ──────────────────
    pd_success_km       = pd_success.copy()
    pd_success_km['Cluster'] = kmeans.labels_
    km_stats = _compute_cluster_stats(transitions_km, range(best_k), pd_success_km)
    km_wavg  = (
        sum(s['trans_success'] for s in km_stats.values()) /
        max(sum(s['trans_count'] for s in km_stats.values()), 1)
    )

    # KMeans centroid: scaled space → inverse_transform → preprocessed space
    km_centroids_orig = scaler.inverse_transform(kmeans.cluster_centers_)
    km_centroids_df   = pd.DataFrame(km_centroids_orig, columns=FEATURES_FOR_CLUSTERING)

    use_hdbscan = False
    if hdb_clusterer is not None and transitions_hdb is not None and n_hdb_clusters > 0:
        pd_success_hdb = pd_success.copy()
        pd_success_hdb['Cluster'] = hdb_labels_success
        hdb_stats = _compute_cluster_stats(transitions_hdb, hdb_unique, pd_success_hdb)
        hdb_wavg  = (
            sum(s['trans_success'] for s in hdb_stats.values()) /
            max(sum(s['trans_count'] for s in hdb_stats.values()), 1)
        )
        # HDBSCAN centroid: küme üyelerinin orijinal (ön-işlem öncesi) medyanı
        hdb_centroids_df = pd.DataFrame(
            {c: pd_success_hdb[pd_success_hdb['Cluster'] == c][FEATURES_FOR_CLUSTERING].median()
             for c in hdb_unique}
        ).T
        hdb_centroids_df.index = hdb_unique

        print(f"\n📊 K-Means ağırlıklı win-rate: {km_wavg*100:.1f}%  |  "
              f"HDBSCAN ağırlıklı win-rate: {hdb_wavg*100:.1f}%")
        if hdb_wavg > km_wavg:
            print("✅ HDBSCAN daha iyi → HDBSCAN kullanılıyor.")
            use_hdbscan = True
        else:
            print("✅ K-Means daha iyi → K-Means kullanılıyor.")

    # ── ADIM 5: Kazanan modelin tablosu + cluster_info ───────────────────────
    if use_hdbscan:
        active_labels    = hdb_unique
        active_stats     = hdb_stats
        active_centroids = hdb_centroids_df
        transitions_pd   = transitions_hdb
        setups_pd['Cluster'] = setups_pd['Cluster_HDB']
    else:
        active_labels    = list(range(best_k))
        active_stats     = km_stats
        active_centroids = km_centroids_df
        transitions_pd   = transitions_km
        setups_pd['Cluster'] = setups_pd['Cluster_KM']

    cluster_info = {}
    cluster_map  = {}

    method_tag = "HDBSCAN" if use_hdbscan else "K-Means"
    w = 93
    print(f"\n{'='*w}")
    print(f"  Kazanan Model: {method_tag}")
    print(f"{'='*w}")
    print(f"{'#':<4} {'İsim':<42} {'Orig':>6} {'Geçiş':>7} {'Başarı':>7} {'Hüsran':>7} {'Win%':>6}")
    print("=" * w)

    for c in active_labels:
        s = active_stats[c]
        if use_hdbscan:
            centroid_dict = active_centroids.loc[c].to_dict()
        else:
            centroid_dict = active_centroids.iloc[c].to_dict()
        sig  = _signature_from_centroid(centroid_dict, feature_quartiles)
        name = _name_from_signature(sig, s['win_rate'])

        cluster_info[c] = {
            'name':               name,
            'win_rate':           s['win_rate'],
            'original_size':      s['orig_count'],
            'transition_size':    s['trans_count'],
            'transition_success': s['trans_success'],
            'transition_failure': s['trans_failure'],
            'signature':          sig,
            'centroid':           centroid_dict,
        }
        cluster_map[c] = name

        print(f"{c:<4} {name[:42]:<42} {s['orig_count']:>6} {s['trans_count']:>7} "
              f"{s['trans_success']:>7} {s['trans_failure']:>7} {s['win_rate']*100:>5.1f}%")

    print("=" * w)
    print(f"Orig=Eğitimde kullanılan ralliciler   "
          f"Geçiş=Bağımsız rejim girişleri (B+C)   "
          f"🌟≥%65  ⚠️≤%35   [{method_tag}]")

    # ── Kaydet ──────────────────────────────────────────────────────────────

    artifacts = {
        # Clustering
        'clustering_method':   'hdbscan' if use_hdbscan else 'kmeans',
        'kmeans':              kmeans,
        'hdbscan_clusterer':   hdb_clusterer if use_hdbscan else None,
        # Preprocessing
        'scaler':              scaler,
        'best_k':              n_hdb_clusters if use_hdbscan else best_k,
        'q_low':               q_low,
        'q_high':              q_high,
        'features':            FEATURES_FOR_CLUSTERING,
        'feature_weights':     weight_vec,
        # Küme yorumlama
        'cluster_map':         cluster_map,
        'cluster_info':        cluster_info,
        'feature_quartiles':   feature_quartiles,
    }
    joblib.dump(artifacts, ARTIFACTS_PATH)
    print(f"\n💾 Model artifacts kaydedildi: {ARTIFACTS_PATH}")

    # Havuzları kaydet (twin_matcher kullanır)
    pl.from_pandas(pd_success).write_parquet(SUCCESS_POOL_PATH)
    failure_clean.write_parquet(FAILURE_POOL_PATH)
    pl.from_pandas(transitions_pd).write_parquet(TRANSITIONS_PATH)
    pl.from_pandas(setups_pd).write_parquet(SETUPS_PATH)
    print("💾 Başarı, başarısızlık havuzları, geçiş ve tüm setup tabloları kaydedildi.")


if __name__ == "__main__":
    raise SystemExit(
        "\n❌ model_core.py doğrudan çalıştırılamaz!\n"
        "   Bilgi tabanı eğitimi için:\n"
        "   venv/bin/python3 build_knowledge_base.py\n"
    )
