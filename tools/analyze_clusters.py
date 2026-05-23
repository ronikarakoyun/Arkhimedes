import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""
Küme Karşılaştırma Analizi
===========================
Her küme için iki grubu karşılaştırır:

  Orijinal  — K-Means'i eğitmek için kullanılan rallici hisseler (success_clean)
  Genişlemiş — Aynı model tüm veriye uygulandığında bu kümeye düşen tüm hisseler

Feature dağılımları arasındaki fark, kümenin "gerçek nüfusta ne yakaladığını"
vs "ne için eğitildiğini" gösterir.

    venv/bin/python model_core.py   # önce
    venv/bin/python analyze_clusters.py
"""

import polars as pl
import pandas as pd
import numpy as np
import joblib
import os

from config import ARTIFACTS_PATH, SUCCESS_POOL_PATH, SETUPS_PATH
from model_core import RALLY_GAIN_THRESHOLD, FAILURE_GAIN_CEILING

FEATURES = ["cv_120", "vol_120", "mom_120", "v_roc"]


def _bar(ratio, width=20):
    filled = int(round(ratio * width))
    return '█' * filled + '░' * (width - filled)


def run_analysis():
    for path in [ARTIFACTS_PATH, SUCCESS_POOL_PATH, SETUPS_PATH]:
        if not os.path.exists(path):
            print(f"❌ Eksik dosya: {path}  →  önce model_core.py çalıştırın.")
            return

    artifacts    = joblib.load(ARTIFACTS_PATH)
    cluster_map  = artifacts['cluster_map']
    cluster_info = artifacts['cluster_info']
    best_k       = artifacts['best_k']

    success_clean = pl.read_parquet(SUCCESS_POOL_PATH).to_pandas()   # orijinal ralliciler
    setups        = pl.read_parquet(SETUPS_PATH).to_pandas()          # tüm geçerli hisse-tarih

    print("=" * 80)
    print("  KÜME GENİŞLEME ANALİZİ")
    print("  Orijinal (ralliciler) vs Genişlemiş (tüm veri) karşılaştırması")
    print("=" * 80)
    print(f"  Orijinal eğitim seti : {len(success_clean):,} kayıt")
    print(f"  Tüm setup verisi     : {len(setups):,} kayıt")
    print()

    for c in range(best_k):
        name = cluster_map[c]
        info = cluster_info[c]

        orig = success_clean[success_clean['Cluster'] == c]
        ext  = setups[setups['Cluster'] == c]

        # Genişlemiş içinde TP/FP dağılımı
        ext_tp   = ext[ext['future_max_gain'] >= RALLY_GAIN_THRESHOLD]
        ext_fp   = ext[ext['future_max_gain'] < FAILURE_GAIN_CEILING]
        ext_neut = ext[
            (ext['future_max_gain'] >= FAILURE_GAIN_CEILING) &
            (ext['future_max_gain'] < RALLY_GAIN_THRESHOLD)
        ]
        total = len(ext)
        prec  = len(ext_tp) / (len(ext_tp) + len(ext_fp)) if (len(ext_tp) + len(ext_fp)) > 0 else 0

        print("=" * 76)
        print(f"  KÜME {c}  |  {name}")
        print(f"  Orijinal: {len(orig):,} rallici   →   Genişlemiş: {total:,} kayıt")
        print("=" * 76)

        # TP/FP barları
        print(f"  Ralli yaptı  {_bar(len(ext_tp)/total if total else 0)}  {len(ext_tp):>6} ({len(ext_tp)/total*100:4.1f}%)")
        print(f"  Başarısız    {_bar(len(ext_fp)/total if total else 0)}  {len(ext_fp):>6} ({len(ext_fp)/total*100:4.1f}%)")
        print(f"  Nötr         {_bar(len(ext_neut)/total if total else 0)}  {len(ext_neut):>6} ({len(ext_neut)/total*100:4.1f}%)")
        print(f"  Precision (Ralli / Ralli+Başarısız): {prec*100:.1f}%")
        print()

        # Feature karşılaştırması
        print(f"  {'Feature':<12} {'Orijinal Med':>13} {'Genişlemiş Med':>15} {'Δ':>10}  Yön")
        print(f"  {'-'*12} {'-'*13} {'-'*15} {'-'*10}  {'-'*22}")

        for f in FEATURES:
            if f not in orig.columns or f not in ext.columns:
                continue
            orig_med = orig[f].median()
            ext_med  = ext[f].median()
            delta    = ext_med - orig_med
            pct      = abs(delta) / (abs(orig_med) + 1e-9) * 100

            if abs(pct) < 5:
                yön = "benzer"
            elif f == 'cv_120':
                yön = "genel popülasyon daha sıkışık" if delta < 0 else "genel popülasyon daha gevşek"
            elif f == 'vol_120':
                yön = "genel popülasyon daha sakin" if delta < 0 else "genel popülasyon daha volatil"
            elif f == 'mom_120':
                yön = "genel popülasyon daha trendsiz" if delta < 0 else "genel popülasyon daha momentumlu"
            elif f == 'v_roc':
                yön = "genel popülasyon hacim düşüyor" if delta < 0 else "genel popülasyon hacim artıyor"
            else:
                yön = ""

            print(f"  {f:<12} {orig_med:>13.4f} {ext_med:>15.4f} {delta:>+10.4f}  {yön} ({pct:.0f}%)")

        print()

    # Özet tablo
    print("=" * 76)
    print("  GENEL ÖZET")
    print("=" * 76)
    print(f"  {'Küme':<30} {'Orig':>6} {'Tüm':>8} {'Ralli%':>8} {'Başarısız%':>11} {'Prec%':>7}")
    print(f"  {'-'*30} {'-'*6} {'-'*8} {'-'*8} {'-'*11} {'-'*7}")

    for c in range(best_k):
        name  = cluster_map[c][:29]
        orig  = success_clean[success_clean['Cluster'] == c]
        ext   = setups[setups['Cluster'] == c]
        ext_tp = ext[ext['future_max_gain'] >= RALLY_GAIN_THRESHOLD]
        ext_fp = ext[ext['future_max_gain'] < FAILURE_GAIN_CEILING]
        total  = len(ext)
        prec   = len(ext_tp) / (len(ext_tp) + len(ext_fp)) * 100 if (len(ext_tp) + len(ext_fp)) > 0 else 0
        print(f"  {name:<30} {len(orig):>6} {total:>8} {len(ext_tp)/total*100:>7.1f}% {len(ext_fp)/total*100:>10.1f}% {prec:>6.1f}%")

    print("=" * 76)


if __name__ == "__main__":
    run_analysis()
