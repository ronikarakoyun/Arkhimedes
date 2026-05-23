"""
Arkhimedes — Merkezi Konfigürasyon
====================================
Tüm dosyaların import ettiği tek path ve sabit kaynağı.
Bu dosya dışında HİÇBİR dosya path string'i tanımlamamalı.

Kullanım:
    from config import DB_PATH, INITIAL_CAPITAL  # vb.
"""
from pathlib import Path

# ─── Veri Dosyaları ─────────────────────────────────────────────────────────
DB_PATH           = "market_db.parquet"
FEATURES_PATH     = "features_db.parquet"
ARTIFACTS_PATH    = "model_artifacts.pkl"
SECTOR_MAP_PATH   = "sector_map.csv"

# Knowledge base (model_core.build_knowledge_base çıktıları)
SUCCESS_POOL_PATH = "knowledge_success_pool.parquet"
FAILURE_POOL_PATH = "knowledge_failure_pool.parquet"
SETUPS_PATH       = "knowledge_setups.parquet"
TRANSITIONS_PATH  = "knowledge_transitions.parquet"

# ─── Raporlama Dizinleri ────────────────────────────────────────────────────
REPORT_BACKTEST_DIR = "Gunluk_Raporlar/Backtest"
REPORT_FULL_DIR     = "Gunluk_Raporlar/Tam_Analiz"
REPORT_DAILY_DIR    = "Gunluk_Raporlar/Gunluk_Ozet"
TWIN_OUTPUT_DIR     = "BIST_Rally_Analiz_Paketi/Twin_Match_Results"

# ─── Backtest Sabitleri ─────────────────────────────────────────────────────
# Model parametreleri (RALLY_*, FEATURE_*, CLUSTER_*) model_core.py'da kalır
# çünkü onlar modelin **mantığı**. Buradakiler **trading/portföy** parametreleri.

INITIAL_CAPITAL     = 1_000_000.0
MAX_POSITIONS       = 5
POSITION_WEIGHT     = 0.20
COMMISSION_RATE     = 0.002       # %0.2 her yönde

# Dinamik stop parametreleri (backtest_engine'den taşındı)
ATR_MULTIPLIER      = 4.0
TRAILING_FLOOR_PCT  = 0.20
ATR_PERIOD          = 20

# ─── Likidite / Kapasite Modeli ─────────────────────────────────────────────
# Pozisyon büyüklüğü, hissenin 20-günlük ortalama TL hacminin (ADV) bir oranıyla
# sınırlanır; aşan kısım alınamaz. Slippage = taban + piyasa-etkisi.
MAX_ADV_PCT       = 0.10     # pozisyon ≤ ADV20 × %10
SLIPPAGE_BASE     = 0.001    # taban kayma (bid-ask), %0.1
SLIPPAGE_IMPACT   = 0.10     # piyasa etkisi: slip += IMPACT × (işlem_TL / ADV20)

# ─── Clustering Yöntemi ─────────────────────────────────────────────────────
# backtest_engine.train_at_date hangi clustering ile çalışsın.
# "kmeans" | "hdbscan". CLUSTER_METHOD env değişkeniyle override edilebilir.
CLUSTER_METHOD           = "gmm"     # Step 1: GMM > K-Means (32K vs 22K sinyal, Score@K=20: 1.11x vs 1.08x)
HDBSCAN_MIN_CLUSTER_SIZE = 60
HDBSCAN_MIN_SAMPLES      = 5

# ─── Full Analysis Sabitleri ────────────────────────────────────────────────
FULL_ANALYSIS_TOP_N        = 50
FULL_ANALYSIS_TWIN_TOP_K   = 20
FULL_ANALYSIS_RALLY_TRIGGER = 0.30     # ralli başladı sayılması için gain eşiği
FULL_ANALYSIS_TIMING_WINDOW = 252      # tetiklenme arama penceresi

# ─── Twin Divergence (Post-Entry Monitör) ───────────────────────────────────
TWIN_DIV_ENABLED         = True   # Backtest'te checkpoint exit aktif mi
TWIN_DIV_WINDOW          = 60     # Post-entry trajectory uzunluğu (gün)
TWIN_DIV_MIN_CHECK_DAYS  = 10     # İlk N gün checkpoint atlanır (gürültü)
TWIN_DIV_CHECK_INTERVAL  = 3      # Kaç günde bir DTW hesabı
TWIN_DIV_EXIT_THRESHOLD  = 0.0    # score < threshold → çık (orta: median)
TWIN_DIV_MIN_TWINS       = 5      # Her gruptan min ikiz (yetersizse kontrol devre dışı)

# ─── Makro + Fundamental Filtre (Lojistik) ──────────────────────────────────
# Veri yolları
MACRO_EXCEL_PATH         = "/Users/unalronikarakoyun/Desktop/deneme/EVDS_Verileri_2016_2026.xlsx"
FUND_PARQUET_PATH        = "/Users/unalronikarakoyun/Desktop/deneme/BIST_Tarihsel_Temel_Analiz.parquet"

# Look-ahead lag: Fundamental tarafta lag artık process_bist_raw.available_period()
# içinde KAP konsolide bildirim tarihleriyle uygulanıyor (Q1→11 May, Q2→19 Ağu,
# Q3→9 Kas, Q4→11 Mar). fundamental_engine ek lag uygulamaz. Makro için macro_engine.LAG_MAP.

# CombinedFilter (macro + fundamental + technical via LightGBM)
# LambdaRank (LGBMRanker) — binary sınıflandırma yerine geçti
ENV_FILTER_NDCG_THRESHOLD = 0.0      # 0 = her zaman KEEP; >0 = NDCG eşiği (devre dışı)
NDCG_EVAL_AT              = [5, 10, 20]  # NDCG@K değerlendirme noktaları
# Eski: ENV_FILTER_AUC_THRESHOLD = 0.65  # (binary classifier çağında kullanılıyordu)
FINAL_P_THRESHOLD        = 0.0       # RankScore veto DEVRE DIŞI: LambdaRank skoru negatif
                                     # olabilir; _thr > 0 kontrolüyle veto aktif hale alınır.
                                     # 0.0 seçildi → veto yok (simulate_trading'de _thr>0 koşulu var).
COMBINED_CV_WINDOW_YEARS = 3         # Rolling walk-forward train penceresi

# ─── GMM Sabit Küme Sayısı (Hata 1) ─────────────────────────────────────────
# Her eğitimde her zaman 5 küme (G0-G4) — BIC araması kapatıldı.
# journey_tracker'ın retrain'ler arası tutarlılığı için zorunlu.
GMM_N_COMPONENTS      = 5

# ─── LambdaRank Eğitim Cutoff'u (Hata 2 — ikili cutoff mimarisi) ───────────
# GMM/success_pd/win_rates için RALLY_WINDOW_DAYS=252 (tam etiket) KORUNUR.
# Sadece LambdaRank için kısaltılmış cutoff: retrain_date - TRAINING_LABEL_WINDOW.
# Bu sayede son 1 yıllık piyasa rejimi LambdaRank'a girer.
TRAINING_LABEL_WINDOW = 90

# ─── Seyir Defteri (Journey Tracker) ────────────────────────────────────────
TRACKING_DB_PATH       = "tracking_db.json"
TRACKING_REPORT_DIR    = "Gunluk_Raporlar/Tracking"
TRACKING_TOP_N         = 30    # Takip edilecek üst sıra sayısı
TRACKING_GHOST_TIMEOUT = 45    # GHOST modunda bu kadar gün sonra ARCHIVED_STALE
# NOT: 5 çok büyüktü — fundamental veri 2016-17'den başladığı için hiç fold
# üretilmiyordu (cv_auc=N/A → tüm epoch'lar DROP). 3 yıllık pencereyle epoch
# 6'dan itibaren geçerli fold alınabiliyor.
