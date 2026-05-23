# Arkhimedes — Claude için Proje Bağlamı

Bu dosyayı oku, tüm kodu okumana gerek yok.

## Projenin Özü

BIST (Borsa İstanbul) hisse senedi ralli tarama sistemi. Teknik + makro + temel veriyi birleştirerek XU100'ü geçecek hisseleri önceden tespit etmeye çalışır. **Olasılık değil, sıralama** üretir.

---

## Kritik Mimari Kurallar

1. **Tek path kaynağı: `config.py`** — Başka hiçbir dosya path string'i tanımlamamalı.
2. **Model eğitimi: `build_knowledge_base.py`** — `model_core.py` içindeki eski `build_knowledge_base()` fonksiyonu DEPRECATED, çalıştırma.
3. **Artifacts formatı:** `model_artifacts.pkl` şu anahtarları içermeli:
   - `clusterer` (GMM nesnesi), `scaler`, `q_low`, `q_high`
   - `feature_weights` (np.array, 21 boyut)
   - `cluster_map` {int: str}, `cluster_info` {int: dict}
   - `win_rates` {int: float}
   - `combined_filter` (CombinedFilter/LGBMRanker nesnesi)
   - `clustering_method` (str: "gmm"|"kmeans"|"hdbscan")
   - Backward-compat: `kmeans` = `clusterer` alias'ı
4. **Küme atama her zaman `_assign_clusters(model_bundle, X)`** — doğrudan `clusterer.predict()` çağırma; GMM/HDBSCAN/KMeans için farklı mantık var.
5. **`future_max_gain` = XU100'e göreli** (mutlak değil). `common.compute_forward_relative_gain()` tek kaynak.
6. **Veto mantığı:** `FINAL_P_THRESHOLD = 0.0` → veto KAPALI. LambdaRank skoru negatif olabilir; `_thr > 0` koşulu olmadan veto etme.

---

## Dosya Rolleri (önem sırasıyla)

### Çalıştırılan Scriptler
| Script | Ne Yapar | Süre |
|--------|----------|------|
| `build_knowledge_base.py` | GMM + LambdaRank eğitimi, artifacts üretir | ~15dk |
| `run_full_analysis.py` | 4-adımlı derin tarama (twin matching dahil) | ~10dk |
| `run_21_days_analysis.py` | Son 21 günün geriye dönük taraması | ~8dk |
| `run_daily_analysis.py` | Hızlı günlük tarama (twin yok) | ~1dk |
| `backtest_engine.py` | 10 yıllık walk-forward backtest (`run_backtest()`) | ~1saat |

### Core Modüller (import edilir)
| Modül | İçerik |
|-------|--------|
| `config.py` | Tüm path ve sabitler |
| `model_core.py` | `FEATURES_FOR_CLUSTERING` (21 feature), `FEATURE_WEIGHTS`, eşikler, `apply_preprocessing()` |
| `backtest_engine.py` | `train_at_date()`, `_assign_clusters()`, `precompute_signals_for_epoch()`, `simulate_trading()` |
| `combined_filter.py` | `CombinedFilter` (LGBMRanker), `_compute_relevance_labels()`, `TECH_FEATURES` (18 feature) |
| `common.py` | `compute_forward_relative_gain()`, `compute_atr20()`, `compute_adv20()` |
| `macro_engine.py` | `load_macro_features()` → EVDS Excel'den m_* kolonları |
| `fundamental_engine.py` | `load_fundamentals()`, `snapshot_at()` → BIST parquet'ten f_* kolonları |
| `twin_matcher.py` | `get_base_curve()`, `precompute_pool_curves()`, `twin_matcher()` |
| `twin_divergence.py` | `compute_gain_curves()`, `get_post_curve()`, `twin_divergence_score()` |
| `signal_eval.py` | Walk-forward sinyal kalitesi ölçümü (standalone script) |

---

## Veri Modeli

### Veri Dosyaları
```
market_db.parquet     → Ticker, Date, Popen, Phigh, Plow, Pclose, Vlot, Pvwap
                         + XU100 ve USDTRY satırları dahil
features_db.parquet   → Ticker, Date + 22 teknik feature + sektörel feature'lar
model_artifacts.pkl   → Eğitilmiş model (dict, ~1.5MB)
knowledge_success_pool.parquet   → Başarılı ralli setup'ları (Ticker, Date, feature'lar)
knowledge_failure_pool.parquet   → Başarısız setup'lar
knowledge_transitions.parquet    → B+C geçiş tablosu
knowledge_setups.parquet         → Tüm setup'lar (başarı + diğer)
```

### Harici Veri (config.py'de tanımlı path)
```
MACRO_EXCEL_PATH  → /Desktop/deneme/EVDS_Verileri_2016_2026.xlsx
FUND_PARQUET_PATH → /Desktop/deneme/BIST_Tarihsel_Temel_Analiz.parquet
```

---

## Önemli Sabitler (config.py)

```python
RALLY_GAIN_THRESHOLD  = 0.50   # Başarı: XU100'ü ≥%50 geçmek (252 gün içinde)
FAILURE_GAIN_CEILING  = 0.10   # Başarısızlık: <%10 üstünlük
RALLY_WINDOW_DAYS     = 252    # Gelecek kazanç penceresi
CLUSTER_COOLDOWN_DAYS = 30     # Aynı (hisse, küme) için min aralık
SECOND_LEG_COOLDOWN   = 60     # İkinci ralli temizlik penceresi

CLUSTER_METHOD = "gmm"         # Env: CLUSTER_METHOD override
SCORE_MODE     = "p_rally"     # Env: SCORE_MODE=winrate override

INITIAL_CAPITAL     = 1_000_000
MAX_POSITIONS       = 5
POSITION_WEIGHT     = 0.20
ATR_MULTIPLIER      = 4.0
TRAILING_FLOOR_PCT  = 0.20

ENV_FILTER_NDCG_THRESHOLD = 0.0   # 0 = LambdaRank her zaman aktif
FINAL_P_THRESHOLD         = 0.0   # 0 = veto yok
```

---

## Feature Grupları

### Clustering Features (21 adet, model_core.py)
```python
FEATURES_FOR_CLUSTERING = [
    "cv_120", "cv_60", "cv_compression",     # Volatilite/sıkışıklık
    "vol_120", "rel_vol_120",                 # Mutlak + göreceli vol
    "mom_120", "mom_60", "mom_30",            # Momentum (3 ufuk)
    "v_roc", "vwap_dist_avg",                 # Hacim + VWAP
    "xu100_mom_60", "xu100_above_ma200", "xu100_drawdown",  # Piyasa rejimi
    "usd_mom_30", "usd_mom_60",               # Kur baskısı
    "rel_strength_60", "rel_strength_120",    # XU100'e göre güç
    "xbank_mom_60", "xusin_mom_60",           # Sektör bağlamı
    "sector_rel_60", "sector_rel_120",        # Sektöre göre güç
]
```

### LambdaRank Features (combined_filter.py TECH_FEATURES)
18 teknik feature (FEATURES_FOR_CLUSTERING'in alt kümesi, sector_rel dahil)
+ m_* makro kolonlar (load_macro_features'tan)
+ f_* ve _z fundamental kolonlar (snapshot_at'tan)

---

## Bilinen Sorunlar ve Sınırlar

1. **`model_core.py::build_knowledge_base()`** — ESKİ KOD, çalıştırılmaz (SystemExit fırlatır). Uyarı etiketi eklendi. Doğru komut: `build_knowledge_base.py`.

2. **`run_daily_analysis.py`** — Twin matching yok, sadece kümeleme + LambdaRank sıralaması. Hızlıdır ama derin analiz için `run_full_analysis.py` kullan.

3. **`run_21_days_analysis.py`** — Global scope'ta veri yükleme yapıyor (modül import edilemez, sadece `python3 run_21_days_analysis.py` ile çalışır).

4. **`BACKTEST_END`** — Veri setindeki son tarihle eşleşmeli. Backtest dönemini genişletmek için `backtest_engine.py`'deki bu sabit ve `RETRAIN_DATES` güncellenebilir.

5. **Makro/Temel veri eksikliği** — `load_macro_features()` veya `load_fundamentals()` başarısız olursa LambdaRankFilter `None` döner. Sistem WinRate-tabanlı skora düşer (sessiz fallback, uyarı yazdırır).

---

## Sık Yapılan Değişiklikler

### Yeni Feature Ekleme
1. `tools/feature_engine.py` → hesaplama kodu
2. `features_db.parquet` → yeniden üret
3. `model_core.py::FEATURES_FOR_CLUSTERING` → listeye ekle
4. `model_core.py::FEATURE_WEIGHTS` → ağırlık ata
5. Eğer LambdaRank'a da gidecekse: `combined_filter.py::TECH_FEATURES` → ekle
6. `build_knowledge_base.py` çalıştır → artifacts yenile

### Backtest Dönemini Uzatma
```python
# backtest_engine.py
BACKTEST_END = pd.to_datetime("YYYY-MM-DD")  # yeni son tarih
RETRAIN_DATES = [pd.to_datetime(f"{y}-05-23") for y in range(2018, YENI_YIL)]
```

### Clustering Yöntemini Değiştirme
```python
# config.py
CLUSTER_METHOD = "gmm"  # → "kmeans" veya "hdbscan"
```
Sonra `build_knowledge_base.py` yeniden çalıştır.

### Twin Divergence Kapatma
```python
# config.py
TWIN_DIV_ENABLED = False
```

---

## Artifacts Uyumluluk Tablosu

Kod tabanındaki farklı scriptler artifacts'ı şöyle okur:

| Anahtar | Kim Kullanır |
|---------|-------------|
| `clustering_method` | run_full_analysis.py (Line 119), run_21_days_analysis.py |
| `cluster_method` | build_knowledge_base.py (alias) |
| `clusterer` | run_full_analysis.py, run_21_days_analysis.py |
| `kmeans` | run_daily_analysis.py (backward compat, alias) |
| `scaler` | hepsi |
| `feature_weights` | run_full_analysis.py, run_daily_analysis.py, run_21_days_analysis.py |
| `weight_vec` | build_knowledge_base.py (alias) |
| `cluster_map` | hepsi |
| `cluster_info` | hepsi |
| `win_rates` | backtest_engine.py |
| `combined_filter` | run_full_analysis.py, run_daily_analysis.py, run_21_days_analysis.py |
| `twin_pool_success/failure` | build_knowledge_base.py kayıt, parquet'ten okunur |

---

## Test Komutları

```bash
# Core modüller import OK mı?
venv/bin/python3 -c "
from backtest_engine import train_at_date, _assign_clusters
from combined_filter import CombinedFilter, LGB_RANK_PARAMS
from model_core import FEATURES_FOR_CLUSTERING
from macro_engine import load_macro_features
from fundamental_engine import load_fundamentals
print('✅ Tüm core modüller OK')
"

# Artifacts format geçerli mi?
venv/bin/python3 -c "
import joblib, numpy as np
a = joblib.load('model_artifacts.pkl')
assert 'combined_filter' in a, 'combined_filter eksik'
assert 'clusterer' in a or 'kmeans' in a, 'clusterer/kmeans eksik'
assert 'feature_weights' in a, 'feature_weights eksik'
cf = a.get('combined_filter')
print(f'✅ Artifacts OK | clusterer={type(a.get(\"clusterer\", a.get(\"kmeans\"))).__name__}')
print(f'   combined_filter={cf.is_useful() if cf else None} | feat={len(cf.feature_cols) if cf else 0}')
"

# Hızlı günlük tarama
venv/bin/python3 run_daily_analysis.py

# Derin analiz
venv/bin/python3 run_full_analysis.py

# Bilgi tabanı yeniden eğit
venv/bin/python3 build_knowledge_base.py
```
