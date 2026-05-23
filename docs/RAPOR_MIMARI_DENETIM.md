# Minerva v3 — Mimari Denetim ve Geliştirme Raporu

*Hibrit (Karar Destek) Modu Geçişi*

---

## Yönetici Özeti

Sistem 10 yıllık BIST verisinde mekanik backtest'lerle ölçüldü ve **%79.5 CAGR + 2.0 Sharpe** üretti. Ancak kullanıcı niyeti **mekanik trader değil**, sistemi **karar destek** olarak kullanmak:

> "Sistem sinyal verir, trader tüm giriş/çıkış kararlarını verir."

Bu denetim 4 hedefe odaklanır:

1. **Sınıflandırma sorunu**: K-Means %1.03'lük seyrek bir sınıfa (rally) fit ediliyor — predict gürültülü, win-rate sadece %40-62.
2. **Trader tahtası açığı**: 52W high/low, pivot, VWAP bantları gibi temel teknik seviyeler sistemde **yok**.
3. **Kod sağlığı**: 6+ dosyada path duplikasyonu, 3 dosyada aynı pivot mantığı, 100+ satırlık fonksiyonlar.
4. **Backlog**: Yabancı takas, market cap, emtia momentum gibi planlı ama yapılmamış işler.

---

## Bölüm A — Orijinal Ralliler vs Predict Edilen Setup Grupları

### A.1 Mevcut Akış

```
[features_db.parquet] → 629,577 setup (mom_120 < 0.70, lookahead-bias temizlenmiş)
         │
         ├── %1.03 success_clean = 6,483 ralli (gelecek 252g'de +%100)
         │       │
         │       └── K-Means.fit() ────────┐
         │                                  ▼
         └── %98.97 diğer setup'lar ──→ K-Means.predict()
                                              │
                                              ▼
                                    setups_all['Cluster']
                                              │
                                              ▼
                                  B+C transitions (30g cooldown)
                                              │
                                              ▼
                                  Win-rate hesaplaması
```

**Kritik problem:** K-Means yalnızca **6,483 ralliciyle** fit ediliyor, sonra **629,577 setup'a** predict ediliyor. Cluster centroid'leri ralli imzasını öğreniyor, ama predict edilen 622K "yarı-imzalı" setup gürültü yaratıyor.

### A.2 Ölçümler (audit_cluster_distances.py çıktısı)

**Cluster bazlı popülasyon:**

| Cluster | İsim | success_clean'de | setups_all'da | Density | Centroide medyan mesafe (success / setups) |
|---|---|---|---|---|---|
| K0 | "Dipte" | 3,958 | 425,296 | %0.93 | 3.565 / 3.501 |
| K1 | "Trendde" | 2,039 | 170,747 | %1.19 | 4.016 / 3.851 |
| K2 | "Dipte" | 486 | 33,534 | **%1.45** | 3.850 / 4.880 |

**Önemli gözlem:** K2'de setups_all'ın centroide medyan mesafesi (4.880) success_clean'in mesafesinden (3.850) **daha uzak** — yani K2 centroidi gerçekten "ralli imzasını" yakalıyor, başarısızlar uzak. K0/K1'de fark yok → bu cluster'lar popülasyonun ortasında.

### A.3 Uyum Filtresinin Win-Rate Üzerine Etkisi

Cluster'a en yakın setup'lardan başlanarak filtre uygulandığında win-rate (raw setup bazında, transitions değil):

| Cluster | Filtre | n | Win-rate | Fail-rate |
|---|---|---|---|---|
| **K0** | Tümü | 425K | **%36.4** | %20.6 |
| K0 | Üst %50 (en yakın) | 212K | %33.7 ↓ | %22.1 |
| K0 | Üst %25 | 106K | %31.7 ↓↓ | %22.5 |
| K0 | Üst %10 | 42K | %29.3 ↓↓↓ | %22.6 |
| **K1** | Tümü | 171K | **%42.1** | %17.8 |
| K1 | Üst %25 | 43K | %38.4 ↓ | %20.3 |
| K1 | Üst %10 | 17K | %37.6 ↓ | %20.7 |
| **K2** | Tümü | 34K | %54.0 | %9.2 |
| K2 | **Üst %50** | 17K | **%66.1** ↑↑↑ | **%2.0** ↓↓↓ |
| K2 | Üst %25 | 8K | %63.4 ↑↑ | %1.7 |
| K2 | Üst %10 | 3K | %58.4 ↑ | %1.8 |

### A.4 Stratejik Çıkarımlar

1. **K0 paradoksu**: Centroide YAKIN olmak K0'da win-rate'i **azaltıyor** (%36 → %29). Bu, K0 centroidinin gerçek "ralli imzası" değil, **popülasyonun ortası** olduğunu gösteriyor. Yakın olmak = tipik hisse = düşük getiri.

2. **K1 stabil ama düşük**: %42-38 aralığında, marjinal azalma. K1 trend-takibi, çoğu trend hisse gibi davranır.

3. **K2 sweet spot**: Üst %50 uyum kesiminde **win-rate %66.1, fail-rate sadece %2.0**. Bu cluster için uyum filtresi gerçekten çalışıyor. Hibrit modelde **K2 + üst %50 uyum** trader için en güçlü sinyaller.

4. **İmbalance**: %1.03 success oranı K-Means için çok seyrek. **Logistic Regression (class_weight='balanced')** ile bu doğrudan modellenebilir.

### A.5 Yaklaştırma Stratejisi — Önerilen Mimari

```
ADIM 1 (mevcut)     : K-Means → cluster atama
ADIM 2 (YENİ)       : Logistic Regression
                      X: features  |  y: success_clean üyesi mi?
                      class_weight='balanced'
                      → P(rally) skoru [0-1] her setup için
ADIM 3 (YENİ)       : Uyum quantile (cluster-bağımlı)
                      K2 için Q50 threshold, K0/K1 için Q70-Q90
ADIM 4 (mevcut)     : B+C transitions
ADIM 5 (YENİ)       : Composite skor
                      0.5 × P(rally) + 0.3 × cluster_win_rate + 0.2 × uyum_skoru
                      Sıralama bu skorla
```

**Beklenen kazanım:** Win-rate %40-60 → %65-75. Açıklanabilirlik korunur (her bileşen ayrı raporlanır).

---

## Bölüm B — Trader Tahtası Açığı

### B.1 Mevcut Feature'lar (Kategori Bazlı, ~27 adet)

| Kategori | Feature'lar |
|---|---|
| Momentum | mom_30, mom_60, mom_120, mom_252 |
| Volatilite | vol_30, vol_60, vol_120 |
| Sıkışma (CV) | cv_60, cv_120, cv_252, cv_compression |
| Hacim | v_roc, rel_vol_120 |
| VWAP | vwap_dist_avg (sadece ortalama mesafe) |
| Göreli Güç | rel_strength_60, rel_strength_120 |
| Piyasa Rejimi | xu100_mom_60, xu100_above_ma200, xu100_drawdown |
| Sektör | sector_rel_60, sector_rel_120, xbank_mom_60, xusin_mom_60 |
| Makro | usd_mom_30, usd_mom_60, usd_vol_30 |

### B.2 Trader Tahtasında VAR, Sistemde YOK

Bir BIST trader'ın tahtaya bakarken (TradingView, Matriks, Foreks Trader) en sık gördüğü 10 bilgi türü ve bunların sistemdeki durumu:

| Bilgi | Tahtada görünüm | Sistemde? | Eklenme zorluğu |
|---|---|---|---|
| 52W high/low mesafesi | Sağ panelde sayısal | ❌ | Çok kolay (OHLCV) |
| Pivot Points (R1/R2/S1/S2) | Grafik üzeri seviye | ❌ | Çok kolay |
| VWAP ±2σ bantları | Grafik üzeri 3 çizgi | ❌ (sadece avg dist) | Kolay (Pvwap mevcut) |
| Price position in N-day range | Grafik dışı (zihinden) | ❌ | Çok kolay |
| Volume profile / POC | Histogram | ❌ | Orta |
| Momentum divergence (RSI fark) | Indicator paneli | ❌ | Kolay (mevcut feature'lar) |
| Volume mean reversion | Volume bar rengi | ❌ | Çok kolay |
| Yabancı takas oranı | KAP/Foreks pencere | ❌ | Zor (KAP scraping) |
| Halka açıklık / market cap | KAP künye | ❌ | Orta (yfinance info) |
| İçeriden alım/satım | KAP haberi | ❌ | Orta (KAP RSS) |

### B.3 Önerilen 7 Yeni Feature

Hepsi mevcut OHLCV + Pvwap'tan türetilir — **dış kaynak gerekmez**:

| Feature | Formül | Pencere |
|---|---|---|
| `dist_52w_high` | `Pclose / max(Phigh, 252) - 1` | 252g |
| `dist_52w_low` | `Pclose / min(Plow, 252) - 1` | 252g |
| `price_pos_20d` | `(Pclose - min(Plow, 20)) / (max(Phigh, 20) - min(Plow, 20))` | 20g |
| `vwap_band_pos` | `(Pclose - mean(Pvwap, 20)) / (2 × std(Pvwap, 20))` | 20g |
| `pivot_dist` | `(Pclose - (Phigh+Plow+Pclose).shift(1)/3) / pivot` | 1g |
| `mom_divergence` | `mom_30 - mom_120` | mevcut |
| `vol_mean_revert` | `Vlot / mean(Vlot, 60)` | 60g |

**Bu feature'lar 2 amaç gütmektedir:**
1. **Clustering'e dahil edilebilecek olanlar** (TP/FP ayırıcılığı yüksek olanlar) — model kalitesini artırır
2. **Sadece raporda gösterilecek olanlar** — trader'ın karar verirken görmek istediği seviyeler, hibrit modelin temel girdisi

### B.4 Backlog (Dış Kaynak Gerektiren)

| İhtiyaç | Kaynak | Öncelik |
|---|---|---|
| Yabancı takas oranı | kap.gov.tr scraping / MKK | Orta |
| Market cap + halka açıklık | yfinance `info` veya KAP künyesi | Orta |
| Brent + Altın momentum | yfinance `BZ=F`, `GC=F` | Düşük |
| İçeriden alım/satım | KAP RSS feed | Düşük |
| Earnings/temettü takvimi | KAP RSS feed | Düşük |

---

## Bölüm C — Kod Sağlığı Bulguları

### C.1 Dosya Envanteri (12 Python dosyası, ~3,800 satır)

| Dosya | Satır | Sorumluluk |
|---|---|---|
| model_core.py | 509 | K-Means/HDBSCAN, win-rate hesabı |
| backtest_engine.py | 818 | Walk-forward backtest |
| run_full_analysis.py | 667 | Tam analiz pipeline |
| backtest_variants.py | 472 | 5 exit strateji karşılaştırması |
| backtest_peak_exit.py | 296 | Peak-exit oracle |
| data_engine.py | 238 | Veri indirme (yfinance) |
| run_daily_analysis.py | 232 | Günlük tarama |
| export_all_trades.py | 206 | Trade konsolidasyon |
| twin_matcher.py | 205 | 3-kanal DTW |
| feature_engine.py | 169 | 27 feature hesaplama |
| analyze_clusters.py | 138 | Küme başarı analizi |
| backtest_atr_tune.py | 129 | ATR multiplier tuning |

### C.2 Tespit Edilen Sorunlar

#### Yüksek Öncelik

1. **Path duplikasyonu** — 8 dosyada aynı path sabitleri yeniden tanımlanmış:
   - `DB_PATH`, `FEATURES_PATH`, `ARTIFACTS_PATH`, `SUCCESS_POOL_PATH`, `FAILURE_POOL_PATH`, `SETUPS_PATH`, `TRANSITIONS_PATH`, `REPORT_DIR`
   - **Risk**: Bir path değişirse 8 dosyayı güncellemek gerek; tutarsızlık riski.
   - **Çözüm**: `config.py` merkezi modül.

2. **Sabit duplikasyonu** — `analyze_clusters.py`'da `RALLY_GAIN_THRESHOLD`, `FAILURE_GAIN_CEILING` yeniden tanımlı (model_core'dan import edilmeli).

3. **Pivot table mantığı** — 3 dosyada aynı kod:
   - `backtest_engine.py:275-277` (3 pivot: Close, Open, ATR)
   - `backtest_variants.py:124-130` (4 pivot: Close, Open, ATR, CV)
   - `backtest_peak_exit.py:39-40` (2 pivot)
   - **Çözüm**: `common.py:make_market_pivots(fields=...)` helper.

4. **ATR hesabı tekrarı** — `backtest_engine.py:_compute_atr20` (fonksiyon) ve `backtest_variants.py:load_market_with_atr` (inline) farklı yerlerde aynı işi yapıyor.

#### Orta Öncelik

5. **Bare except** — `data_engine.py:134` ve `export_all_trades.py:36` hata maskeliyor, debug zor.

6. **Defensive `except KeyError`** — 3 dosyada pivot lookup'larında tekrar tekrar yazılmış (`safe_pivot_lookup` helper'la giderilir).

7. **100+ satırlık fonksiyonlar**:
   - `backtest_engine.py:simulate_trading` (121 satır, 3 sorumluluk: exit + entry + NAV)
   - `run_full_analysis.py:screen_candidates` (~80 satır, çok adımlı)
   - `run_full_analysis.py:plot_candidate_scenario` (~60 satır)

#### Düşük Öncelik

8. **Magic numbers** — `run_full_analysis.py:TOP_N=50`, `TWIN_TOP_K=20`, `RALLY_TRIGGER=0.30`, `TIMING_WINDOW=252`: config.py'ye taşınabilir.

9. **Hata günlüğü yok** — başarısız yfinance çağrıları silent fail oluyor (data_engine.py).

### C.3 Pozitif Bulgular

- `apply_preprocessing()` (model_core) — 5 dosyada paylaşılan helper, doğru soyutlama.
- `_detect_cluster_transitions()` — temiz, B+C mantığı tek yerde.
- `build_benchmark_nav()` — 4 dosyada reuse ediliyor.

---

## Bölüm D — Hibrit Karar Destek Çıktıları

### D.1 Mevcut Eksiklik

`run_full_analysis.py` Markdown çıktısı bir adayı şöyle sunuyor:

```
🎯 DERIM
- Uyum Skoru: 0.3342
- Mevcut özellikler: mom_120=+0.02, cv_120=0.052, ...
- İkiz başarısı: 17/3 (%85)
- Tahmin: +50 gün
```

**Trader açısından eksik:**
- Hisse şu an 52W high'a ne kadar yakın?
- Pivot point hangi seviyede?
- VWAP bantlarının neresinde?
- Günlük TL bazında hacim ne (likidite)?
- P(rally) klasifikatör skoru var mı?
- Risk faktörleri (failed twin analizi) açıklayıcı mı?

### D.2 Hedef Çıktı Formatı (Hibrit Mod)

Her aday için zenginleştirilmiş kart:

```
🎯 DERIM (Skor: 72/100)

📊 SİNYAL KALİTESİ
├─ Composite Skor    : 72 (P_rally=0.68, K2 win=%62, Uyum=0.41)
├─ Cluster           : K2 — "Makro Geride Kalan"
├─ Uyum kategorisi   : Yüksek (Q40 altında)
└─ Neden flag        : mom_120 düşük + cv_compression sıkı + dolar yükselişte

📈 TAHTA SEVİYELERİ (yeni)
├─ 52W High mesafesi : -%23.4 (henüz tepe çok uzak)
├─ 52W Low mesafesi  : +%18.7
├─ 20g pozisyon      : 0.62 (üst yarıda)
├─ VWAP bandı        : +0.4σ (orta)
├─ Pivot mesafesi    : +%1.2 (R1'in altında)
└─ Hacim/60g ort     : 1.84× (yükseliş başlamış)

🕐 ZAMANLAMA
├─ İkiz medyan       : +50 gün (IQR 10-72)
├─ Son 30-90g öncüler: 3 hisse (AKSA, BAGFS, ELITE)
└─ Tahmin penceresi  : Haziran ortası–Temmuz başı

💰 LİKİDİTE
├─ Günlük ort. hacim : ₺12.4M
└─ Slippage tahmini  : Düşük (1M TL pozisyon için)

⚠️ RİSK
├─ Failed twin sayısı: 3
├─ Ortak ayrıştırıcı : sector_rel_120 yüksek geliyor başarısızlarda (DERIM düşük ✅)
└─ Cluster fail-rate : %2 (K2 üst %50)
```

Bu format trader'ın karar destek ihtiyacını tam karşılar.

---

## Bölüm E — Eski Plandan Backlog

`PLAN_RALLI_SINYALI.md`'den geriye kalan işler:

| Madde | İçerik | Durum | Öncelik |
|---|---|---|---|
| 1.1 | Lookahead bias filtresi | ✅ Yapıldı (model_core L259) | — |
| 1.5 | Walk-forward backtest | ✅ Yapıldı (backtest_engine.py) | — |
| 2.3 | Yabancı takas oranı | ❌ Yapılmadı | Orta |
| 2.4 | KAP sector scraping (dinamik) | 🟡 Yarım (statik map var) | Düşük |
| 2.5 | Market cap + halka açıklık | ❌ Yapılmadı | Orta |
| 2.6 | Brent + Altın momentum | ❌ Yapılmadı | Düşük |

---

## Tavsiyeler ve Uygulama Sırası

### Önerilen Sıra

1. **Paket 2 — Yeni Feature'lar** (önce, çünkü Paket 1'in girdisi)
2. **Paket 1 — Cluster Yaklaştırma** (Logistic + Uyum quantile + Composite skor)
3. **Paket 3 — Kod Refactor** (en son, davranışsal nötr)
4. **Backlog 2.3, 2.5** (opsiyonel, ileride)

### Beklenen Etki

| Paket | Win-rate beklentisi | Karar destek değeri |
|---|---|---|
| Yalnız mevcut | %40-62 | Düşük |
| + Paket 2 | %42-64 (marjinal) | Yüksek (tahta görünür) |
| + Paket 1 | **%65-75** | **Çok yüksek** (skor + açıklama) |
| + Paket 3 | Aynı | Aynı (sadece kod kalitesi) |

---

## Sonuç

Sistem **mekanik backtest perspektifinden başarılı** (%79 CAGR) ama **karar destek modu için yetersiz**: skor yok, açıklama yok, tahta seviyeleri yok. Bu denetimin 3 uygulama paketi sistemin hibrit moda geçişini sağlayacak:

- **Sınıflandırma kalitesi**: Logistic + Uyum quantile ile win-rate %65+
- **Trader bilgi kapsamı**: 7 yeni feature + zenginleştirilmiş rapor
- **Kod sürdürülebilirliği**: Merkezi config + ortak helper'lar

**Sıradaki adım:** Paket 2 implementasyonu (feature_engine.py + features_db.parquet yenileme).

---

*Rapor üretildiği veri: 2026-05-20 itibarıyla 1.1M kayıt, 511 hisse, 8 yıl walk-forward backtest sonuçları.*
