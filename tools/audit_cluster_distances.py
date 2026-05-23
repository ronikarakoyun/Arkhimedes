import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""
Cluster mesafe analizi — denetim raporu için sayısal ölçümler:
- success_clean üyeleri vs setups_pd üyelerinin centroide ortalama mesafesi
- setups_pd üyelerinin distance dağılımı (Q25/Q50/Q75)
- Üst %25, %50, %75 uyum kesimlerinde win-rate nasıl değişiyor
"""
import polars as pl
import pandas as pd
import numpy as np
import joblib

from model_core import (
    FEATURES_FOR_CLUSTERING, FEATURE_WEIGHTS,
    apply_preprocessing,
    RALLY_GAIN_THRESHOLD, FAILURE_GAIN_CEILING,
)

artifacts = joblib.load("model_artifacts.pkl")
kmeans     = artifacts['kmeans']
scaler     = artifacts['scaler']
q_low      = artifacts['q_low']
q_high     = artifacts['q_high']
weight_vec = artifacts['feature_weights']
ci         = artifacts['cluster_info']
best_k     = artifacts['best_k']

success_clean = pl.read_parquet("knowledge_success_pool.parquet").to_pandas()
setups_all    = pl.read_parquet("knowledge_setups.parquet").to_pandas()

print(f"success_clean: {len(success_clean):,}")
print(f"setups_all   : {len(setups_all):,}")
print(f"oran         : %{len(success_clean)/len(setups_all)*100:.2f}")

# Feature engineering
def transform(df):
    df = df.dropna(subset=FEATURES_FOR_CLUSTERING).copy()
    pp = apply_preprocessing(df, FEATURES_FOR_CLUSTERING, q_low, q_high)
    valid = pp[FEATURES_FOR_CLUSTERING].notna().all(axis=1)
    df = df[valid].copy()
    pp = pp[valid]
    X = scaler.transform(pp[FEATURES_FOR_CLUSTERING]) * weight_vec
    return df, X

success_clean, X_success = transform(success_clean)
setups_all, X_all = transform(setups_all)

# Centroidlere mesafe
print("\n📏 Centroid mesafeleri hesaplanıyor...")
dist_success = kmeans.transform(X_success)  # (N, k) — her cluster'a mesafe
dist_all     = kmeans.transform(X_all)

# Her noktanın kendi cluster'ına mesafesi (min)
success_clean['MinDist'] = dist_success.min(axis=1)
success_clean['Cluster'] = dist_success.argmin(axis=1)
setups_all['MinDist']    = dist_all.min(axis=1)
setups_all['Cluster']    = dist_all.argmin(axis=1)

print(f"\n{'Cluster':<10} {'İsim':<25} {'success_n':>10} {'setups_n':>10} "
      f"{'succ_med':>10} {'setup_med':>10} {'oran':>6}")
print("-" * 90)

for c in range(best_k):
    s_n  = (success_clean['Cluster'] == c).sum()
    a_n  = (setups_all['Cluster'] == c).sum()
    s_med = success_clean[success_clean['Cluster'] == c]['MinDist'].median()
    a_med = setups_all[setups_all['Cluster'] == c]['MinDist'].median()
    name  = ci[c]['name'][:24]
    print(f"K{c:<8} {name:<25} {s_n:>10,} {a_n:>10,} {s_med:>10.3f} {a_med:>10.3f} {s_n/a_n*100:>5.1f}%")

# Uyum threshold deneyi
print("\n\n🎯 Uyum (mesafe) filtresine göre win-rate analizi:")
print("Üst %X = en yakın X% setup. Win-rate'ler transitions üzerinden değil, RAW setup üzerinden hesaplanıyor.\n")
print(f"{'Cluster':<8} {'Filtre':<10} {'n':>8} {'success':>8} {'failure':>8} {'win%':>6} {'fail%':>6}")
print("-" * 70)

# transitions üzerindeki win-rate referans
print("\nReferans (transitions — mevcut sistem):")
for c in range(best_k):
    info = ci[c]
    wr = info['win_rate']
    print(f"K{c}: transitions win-rate = %{wr*100:.1f} (n={info['transition_size']})")

print("\nRaw setup win-rate'leri (lookahead bias temizlenmiş, transitions değil):")
for c in range(best_k):
    cluster_setups = setups_all[setups_all['Cluster'] == c].copy()
    # Win-rate hesap için future_max_gain notna
    cluster_setups = cluster_setups[cluster_setups['future_max_gain'].notna()]
    if len(cluster_setups) == 0:
        continue

    for label, frac in [('Tümü', 1.0), ('Üst %50', 0.5), ('Üst %25', 0.25), ('Üst %10', 0.10)]:
        if frac < 1.0:
            threshold = cluster_setups['MinDist'].quantile(frac)
            filtered = cluster_setups[cluster_setups['MinDist'] <= threshold]
        else:
            filtered = cluster_setups
        if len(filtered) == 0:
            continue
        n = len(filtered)
        success = (filtered['future_max_gain'] >= RALLY_GAIN_THRESHOLD).sum()
        failure = (filtered['future_max_gain'] < FAILURE_GAIN_CEILING).sum()
        win_pct  = success/n*100
        fail_pct = failure/n*100
        print(f"K{c}{'':4} {label:<10} {n:>8,} {success:>8,} {failure:>8,} %{win_pct:>4.1f} %{fail_pct:>4.1f}")
    print()

# Imbalance gerçeği
print(f"\n📊 İMBALANCE GERÇEĞİ:")
print(f"  success_clean / setups_all = {len(success_clean)/len(setups_all)*100:.2f}%")
print(f"  Her cluster için:")
for c in range(best_k):
    s_n = (success_clean['Cluster'] == c).sum()
    a_n = (setups_all['Cluster'] == c).sum()
    print(f"    K{c}: success_clean'de {s_n}, setups_all'da {a_n} → density %{s_n/a_n*100:.2f}")
