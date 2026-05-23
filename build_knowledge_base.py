"""
build_knowledge_base.py — Arkhimedes LambdaRank Bilgi Tabanı Eğitici
====================================================================
GMM clustering + LambdaRankFilter (CombinedFilter) ile tam kapasite eğitim.
Tüm tarihsel veri kullanılır (bugüne kadar, look-ahead bias korunur).

Çıktı:
  model_artifacts.pkl  — run_full_analysis.py'nin beklediği format
  knowledge_success_pool.parquet
  knowledge_failure_pool.parquet
  knowledge_transitions.parquet
  knowledge_setups.parquet

Çalıştırma:
  venv/bin/python3 build_knowledge_base.py
"""
import warnings
warnings.filterwarnings("ignore")

import polars as pl
import pandas as pd
import numpy as np
import joblib
import os

from backtest_engine import (
    train_at_date, FEAT_PATH,
    _detect_cluster_transitions, _assign_clusters,
    _isolate_representative_setups, _remove_second_leg_rallies,
)
from model_core import (
    FEATURES_FOR_CLUSTERING,
    apply_preprocessing,
    RALLY_GAIN_THRESHOLD,
    RALLY_WINDOW_DAYS,
    SECOND_LEG_COOLDOWN,
    FAILURE_GAIN_CEILING,
    CLUSTER_COOLDOWN_DAYS,
    FEATURE_WEIGHTS,
)
from common import compute_forward_relative_gain
from macro_engine import load_macro_features
from fundamental_engine import load_fundamentals
from config import (
    ARTIFACTS_PATH,
    SUCCESS_POOL_PATH,
    FAILURE_POOL_PATH,
    TRANSITIONS_PATH,
    SETUPS_PATH,
    TRAINING_LABEL_WINDOW,
)

print("=" * 70)
print("🧠 Arkhimedes — GMM + LambdaRank Bilgi Tabanı Kuruluyor")
print("=" * 70)

# ─── 1. Veri Yükleme ─────────────────────────────────────────────────────────
print("\n📂 Veri yükleniyor...")
df_feat = pl.read_parquet(FEAT_PATH).with_columns(
    pl.col("Date").cast(pl.Datetime("ms"))
)
print(f"   Feature DB: {df_feat['Date'].min()} → {df_feat['Date'].max()}")
print(f"   Kayıt: {len(df_feat):,} | Hisse: {df_feat['Ticker'].n_unique()}")

# ─── 2. İleri Kazanç Hesabı (XU100-göreli) ───────────────────────────────────
print(f"\n⏳ {RALLY_WINDOW_DAYS} günlük XU100-göreli ileri kazanç hesaplanıyor...")
df_outcomes = compute_forward_relative_gain(df_feat, RALLY_WINDOW_DAYS)
df_pd = df_outcomes.to_pandas()
df_pd["Date"] = pd.to_datetime(df_pd["Date"])

# ─── 3. Eğitim Tarihi: İkili Cutoff Mimarisi ────────────────────────────────
# train_at_date içinde iki cutoff:
#   - label_cutoff      = retrain_date - RALLY_WINDOW_DAYS   (GMM/success/win_rates)
#   - lambdarank_cutoff = retrain_date - TRAINING_LABEL_WINDOW (LambdaRank)
# Production build'de retrain_date'i latest+253 ileri kurarak HER İKİ cutoff'un
# da latest_date'i aşmasını sağlıyoruz → tüm veri eğitime girer.
latest_date = df_pd["Date"].max()
retrain_date = latest_date + pd.Timedelta(days=RALLY_WINDOW_DAYS + 1)
print(f"\n🎯 Eğitim konfigürasyonu:")
print(f"   Son veri tarihi          : {latest_date.date()}")
print(f"   Retrain tarihi           : {retrain_date.date()}")
print(f"   GMM/win_rates cutoff     : {(retrain_date - pd.Timedelta(days=RALLY_WINDOW_DAYS)).date()}  (tam etiketli)")
print(f"   LambdaRank cutoff        : {(retrain_date - pd.Timedelta(days=TRAINING_LABEL_WINDOW)).date()}  (son {RALLY_WINDOW_DAYS - TRAINING_LABEL_WINDOW} gün dahil, kısmi etiket OK)")

# ─── 4. Makro + Fundamental Yükle ────────────────────────────────────────────
print("\n📊 Makro ve fundamental veriler yükleniyor...")
macro_df = load_macro_features()
macro_df["Date"] = pd.to_datetime(macro_df["Date"])
fund_df = load_fundamentals()
print(f"   Makro: {len(macro_df):,} tarih | Fund: {fund_df['Ticker'].nunique():,} hisse")

# ─── 5. Ana Eğitim (GMM + LambdaRank) ───────────────────────────────────────
print("\n🤖 GMM + LambdaRankFilter eğitimi başlıyor...")
bundle = train_at_date(
    df_pd, retrain_date,
    verbose=True,
    macro_df=macro_df,
    fund_df=fund_df,
)
if bundle is None:
    raise RuntimeError("❌ train_at_date başarısız! Yeterli veri yok.")

cluster_method = bundle["cluster_method"]
clusterer      = bundle["clusterer"]
scaler         = bundle["scaler"]
q_low          = bundle["q_low"]
q_high         = bundle["q_high"]
weight_vec     = bundle["weight_vec"]
win_rates      = bundle["win_rates"]
best_k         = bundle["best_k"]
combined_filter = bundle["combined_filter"]
transitions    = bundle["transitions"]
pd_success     = bundle["success_with_clusters"]

print(f"\n✅ Eğitim tamamlandı:")
print(f"   Clustering: {cluster_method.upper()}, {best_k} küme")
print(f"   LambdaRankFilter: {'✅ KEEP' if combined_filter and combined_filter.is_useful() else '❌ None/DROP'}")

# ─── 6. Cluster Info + Cluster Map ───────────────────────────────────────────
print("\n📊 Küme istatistikleri hesaplanıyor...")

cluster_info = {}
cluster_map  = {}
method_tag   = cluster_method[0].upper()  # G for GMM, K for KMeans, H for HDBSCAN

# Hata 3 düzeltmesi — Win-rate sıralı kararlı etiketleme:
# GMM iç ID'leri rastgele atanır → "G2" bir eğitimde en iyi, başkasında en kötü olabilir.
# Çözüm: win-rate'e göre sırala, G0 = en iyi, G1 = ikinci, ... → semantik kararlı.
sorted_by_wr = sorted(win_rates.items(), key=lambda x: x[1], reverse=True)

for rank, (orig_c, wr) in enumerate(sorted_by_wr):
    tc = transitions[transitions["Cluster"] == orig_c]
    n_trans   = len(tc)
    n_success = int((tc["future_max_gain"] >= RALLY_GAIN_THRESHOLD).sum())
    n_fail    = int((tc["future_max_gain"] < FAILURE_GAIN_CEILING).sum())
    orig_count = int((pd_success["Cluster"] == orig_c).sum()) if "Cluster" in pd_success.columns else 0

    alert = " 🌟" if wr >= 0.65 else (" ⚠️" if wr <= 0.35 else "")
    # Mevcut isim formatı KORUNUR (win-rate% + alert emoji) — sadece sıra rank'tan gelir.
    name  = f"{method_tag}{rank} — %{wr*100:.0f} WinRate{alert}"

    cluster_info[orig_c] = {
        "name":               name,
        "win_rate":           wr,
        "original_size":      orig_count,
        "transition_size":    n_trans,
        "transition_success": n_success,
        "transition_failure": n_fail,
        "signature":          {},   # GMM'de centroid-tabanlı etiketleme yok
    }
    cluster_map[orig_c] = name

    print(f"   {name:<45}  eğitim={orig_count:>5,}  geçiş={n_trans:>5,}")

# ─── 7. Havuzları Kaydet (twin_matcher + diğerleri için) ─────────────────────
print("\n💾 Havuzlar kaydediliyor...")

# Başarı havuzu (twin_matcher için)
pl.from_pandas(pd_success).write_parquet(SUCCESS_POOL_PATH)
print(f"   ✅ {SUCCESS_POOL_PATH}: {len(pd_success):,} başarılı setup")

# Başarısızlık havuzu
failure_pd = bundle["twin_pool_failure"]
if len(failure_pd) > 0:
    pl.from_pandas(failure_pd).write_parquet(FAILURE_POOL_PATH)
    print(f"   ✅ {FAILURE_POOL_PATH}: {len(failure_pd):,} başarısız setup")

# Geçiş tablosu
pl.from_pandas(transitions).write_parquet(TRANSITIONS_PATH)
print(f"   ✅ {TRANSITIONS_PATH}: {len(transitions):,} geçiş")

# Setups tablosu (tüm setup'lar — yalnızca başarı havuzu yeterli burada)
pl.from_pandas(pd_success).write_parquet(SETUPS_PATH)
print(f"   ✅ {SETUPS_PATH}: {len(pd_success):,} setup")

# ─── 8. Artifacts Kaydet ─────────────────────────────────────────────────────
artifacts = {
    # ── Clustering ──
    "clustering_method":  cluster_method,       # run_full_analysis.py: artifacts['clustering_method']
    "cluster_method":     cluster_method,
    "clusterer":          clusterer,            # GMM / KMeans nesnesi
    "kmeans":             clusterer,            # backward-compat alias
    "best_k":             best_k,

    # ── Preprocessing ──
    "scaler":             scaler,
    "q_low":              q_low,
    "q_high":             q_high,
    "feature_weights":    weight_vec,           # run_full_analysis.py: artifacts['feature_weights']
    "weight_vec":         weight_vec,
    "features":           FEATURES_FOR_CLUSTERING,

    # ── Küme Yorumlama ──
    "win_rates":          win_rates,
    "cluster_map":        cluster_map,          # run_full_analysis.py: artifacts['cluster_map']
    "cluster_info":       cluster_info,         # run_full_analysis.py: artifacts['cluster_info']
    "feature_quartiles":  {},                   # cluster naming için (basitleştirildi)

    # ── LambdaRankFilter ──
    "combined_filter":    combined_filter,

    # ── Twin Havuzları ──
    "twin_pool_success":  bundle["twin_pool_success"],
    "twin_pool_failure":  bundle["twin_pool_failure"],
}

joblib.dump(artifacts, ARTIFACTS_PATH)
print(f"\n💾 Model artifacts kaydedildi: {ARTIFACTS_PATH}")

# ─── 9. Özet ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("📋 ÖZET")
print("=" * 70)
print(f"  Clustering yöntemi : {cluster_method.upper()}")
print(f"  Küme sayısı        : {best_k}")
print(f"  Başarı havuzu      : {len(pd_success):,} setup")
print(f"  Geçiş sayısı       : {len(transitions):,}")
if combined_filter:
    print(f"  LambdaRankFilter   : ✅ KEEP")
    print(f"  n_train            : {combined_filter.n_train:,}")
    print(f"  cv_ndcg            : {combined_filter.cv_ndcg:.4f}" if combined_filter.cv_ndcg else "  cv_ndcg            : N/A")
    print(f"  features           : {len(combined_filter.feature_cols)}")
else:
    print(f"  LambdaRankFilter   : ❌ None")

print("\n✅ Arkhimedes hazır. Günün raporunu üretmek için:")
print("   venv/bin/python3 run_full_analysis.py")
