# Arkhimedes — Sistem Detayı
## BIST Ralli Tarama ve Karar Destek Sistemi

---

## 1. Sistemin Amacı

Arkhimedes, Borsa İstanbul'da (BIST) tarihsel olarak güçlü ralli yapmış hisselerin örüntülerini öğrenir ve **bugün o örüntüye en çok benzeyen** hisseleri tespit eder.

**Temel Fikir:**
- Geçmişteki tüm büyük ralli başlangıçları incelenir → ortak teknik "şekiller" kümeler halinde sınıflandırılır
- Bugünkü her hisse o şekillere karşılaştırılır → en yakın kümeye atanır
- Kümelerin geçmişteki başarı oranı (win-rate) ve mevcut makro/temel verilerle birleştirilir → sıralanmış aday listesi üretilir

**Sistem olasılık değil, sıralama üretir.** "Bu hisse kesinlikle ralli yapar" demez; "bu hisse geçmişteki başarılı örneklerle en çok benziyor" der.

---

## 2. Mimariye Genel Bakış

```
┌─────────────────────────────────────────────────────────────────┐
│                     VERİ KATMANI                                │
│  market_db.parquet    features_db.parquet    sector_map.csv     │
│  (fiyat/hacim OHLCV)  (teknik feature'lar)   (hisse→sektör)    │
│                                                                  │
│  EVDS_Verileri.xlsx        BIST_Tarihsel_Temel.parquet          │
│  (makro: faiz, kur, CPI)   (temel: PD/DD, F/K, FAVÖK...)       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    MODEL KATMANI                                 │
│                                                                  │
│  build_knowledge_base.py                                        │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  1. Ralli havuzu (future_max_gain ≥ %50 XU100 üstü)      │   │
│  │  2. GMM kümeleme (BIC ile 3-6 küme seçimi)               │   │
│  │  3. LambdaRankFilter (makro+temel+teknik → sıralama)     │   │
│  │  4. model_artifacts.pkl kaydet                           │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   ÇALIŞMA ZAMANI                                 │
│                                                                  │
│  run_full_analysis.py     → Derin analiz (twin matching ile)    │
│  run_21_days_analysis.py  → Son 21 gün geriye dönük tarama      │
│  run_daily_analysis.py    → Hızlı günlük tarama (twin yok)      │
│  backtest_engine.py       → 10 yıllık walk-forward backtest     │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Günlük Kullanım

### Adım 1: Model Yeni mi? (Haftada bir / model güncellemesi)

```bash
cd /Users/unalronikarakoyun/Desktop/Veri
venv/bin/python3 build_knowledge_base.py
```

Bu komut:
- `features_db.parquet`'ten tüm tarihsel veriyi okur
- XU100'e göre ileri kazanç hesaplar (look-ahead bias yok)
- GMM ile 3-6 küme belirler (BIC kriterine göre)
- LambdaRank modelini eğitir (3 yıllık rolling CV)
- `model_artifacts.pkl` + 4 parquet dosyası kaydeder
- Süre: ~10-20 dakika

### Adım 2: Günlük Hızlı Tarama

```bash
venv/bin/python3 run_daily_analysis.py
# veya belirli bir gün için:
venv/bin/python3 run_daily_analysis.py --date 2026-05-15
```

Çıktı: `Gunluk_Raporlar/Gunluk_Ozet/`
- `Tum_Adaylar_Dashboard_YYYYMMDD.txt` — küme bazlı metin özet
- `Tum_Adaylar_YYYYMMDD.xlsx` — Excel, küme bazlı sheet'ler
- Süre: ~1-2 dakika

### Adım 3: Derin Analiz (önemli günlerde)

```bash
venv/bin/python3 run_full_analysis.py
# veya belirli gün:
venv/bin/python3 run_full_analysis.py --date 2026-05-15
```

Çıktı: `Gunluk_Raporlar/Tam_Analiz/`
- `Tam_Analiz_YYYYMMDD.md` — her aday için twin analizi + risk değerlendirmesi
- `Tam_Analiz_YYYYMMDD.xlsx` — Excel tablosu
- `scenarios/*.png` — her aday için 120g fiyat geçmişi + ikiz senaryoları
- Süre: ~5-15 dakika (paralel DTW)

### Adım 4: Dönem Taraması (isteğe bağlı)

```bash
venv/bin/python3 run_21_days_analysis.py
```

Çıktı: `Gunluk_Raporlar/Son_21_Gun_Analizi/`
- Her gün için ayrı Markdown + Excel
- `Son_1_Ay_Konsolide.md` — dönem özeti (en sık radar'a giren, sektör konsantrasyonu)
- `YapayZeka_Yorumlari.md` — otomatik stratejik yorum
- Süre: ~5-10 dakika

---

## 4. Veri Akışı — Adım Adım

### 4a. `build_knowledge_base.py` İçinde Ne Olur?

```
features_db.parquet
        │
        ▼
compute_forward_relative_gain()   ← common.py
  · Her (Ticker, Date) için sonraki 252 gün içinde
    XU100'e göre en yüksek üstün performansı hesaplar
  · future_max_gain = max(hisse/XU100_rel) / bugün_rel - 1
  · Look-ahead bias: son 252 günün labeli eksik → eğitim dışı
        │
        ▼
Başarı havuzu: future_max_gain ≥ 0.50 (XU100'ü ≥%50 geçenler)
Başarısızlık havuzu: future_max_gain < 0.10
        │
        ▼
GMM Kümeleme (model_core + backtest_engine)
  · Sadece başarı havuzuyla eğitilir (başarısızlar kümeleri kirletmez)
  · BIC kriterine göre 3-6 küme arasından en iyisi seçilir
  · Her hisse 0-1 arası üyelik olasılığıyla kümesine girer
        │
        ▼
Win-Rate Hesabı
  · Tüm tarihsel veri her kümeye atanır
  · B+C geçiş kuralı: aynı hisse için aynı kümeye 30 günde bir giriş sayılır
  · Her kümenin geçişteki başarı oranı = win_rate
        │
        ▼
LambdaRankFilter Eğitimi (combined_filter.py)
  · Tüm etiketli veri kullanılır (binary pool değil — hepsi dahil)
  · Her (Ticker, YearMonth) için tek satır
  · YearMonth içinde future_max_gain'e göre 0-4 alaka etiketi atanır
  · LGBMRanker: "aynı ay içinde en iyi kazananı en üste koy"
  · 3 yıllık rolling walk-forward CV ile NDCG@10 hesaplanır
        │
        ▼
model_artifacts.pkl
  · clusterer (GMM nesnesi)
  · scaler (RobustScaler)
  · q_low, q_high (winsorization sınırları)
  · feature_weights (21 feature'ın ağırlık vektörü)
  · win_rates {küme_id: oran}
  · cluster_map {küme_id: isim}
  · cluster_info {küme_id: {win_rate, size, ...}}
  · combined_filter (LGBMRanker nesnesi)
```

### 4b. `run_full_analysis.py` — 4-Adımlı Pipeline

**ADIM 1: Tarama**
```
market_db.parquet → Operasyon enkazı filtresi
  · Şart: peak_252d > base_252d × 2.5 VE fiyat < peak × %55 → ENKAZ
  · Bu hisseler analiz dışı

features_db.parquet → Her hissenin son feature vektörü
  → GMM ile küme ataması
  → LambdaRank ile RankScore
  → Top 50 aday seçilir
```

**ADIM 2: Twin Matching**
```
Başarı + başarısızlık havuzu → 3 kanallı DTW eğrileri (120 gün)
  Kanal 1: Z-score(Pclose)
  Kanal 2: Z-score(log hacim)
  Kanal 3: Z-score((Phigh-Plow)/Pclose)

Her aday için:
  · Havuzdaki tüm kayıtlarla DTW mesafesi hesaplanır (paralel)
  · En yakın 20 "ikiz" seçilir
  · Her ikizin başarı/başarısızlık etiketi → twin_success_pct
```

**ADIM 3: Tetiklenme Zamanlaması**
```
Başarılı ikizler için:
  · Setup tarihinden itibaren ilk %30+ kazancın kaç gün sonra geldiği
  → Median gün + IQR
  Son 30-90 günde tetiklenmiş ikizler (öncü sinyal olabilir)
```

**ADIM 4: Failed-Twin Analizi**
```
Başarısız ikizler için:
  · Başarılı vs başarısız ikizlerin feature medyanları karşılaştırılır
  · Cohen's d ile en ayırıcı feature'lar sıralanır
  · Adayın değeri başarılılara mı başarısızlara mı yakın? → risk skoru
```

---

## 5. Kümeleme Sistemi (GMM)

### Neden GMM?
- K-Means: Her hisse **ya** kümede **ya** değil (sert sınırlar)
- GMM (Gaussian Mixture Model): Her hisse **her kümeye belirli bir olasılıkla** üye
- Sonuç: Uyum skoru olarak `max(predict_proba)` — ne kadar o kümenin "saf örneği"

### Özellik Vektörü (21 Feature)
```
Volatilite Grubu:
  cv_120, cv_60       → 120/60 günlük coefficient of variation (sıkışıklık)
  cv_compression      → kısa vadeli sıkışma = cv_60/cv_120 (kırılmaya hazır mı?)
  vol_120, rel_vol_120 → mutlak ve göreceli volatilite

Momentum Grubu:
  mom_120, mom_60, mom_30  → 120/60/30 günlük fiyat değişimi (3 zaman ufku)

Hacim Grubu:
  v_roc        → hacim büyüme trendi (volüm momentum)
  vwap_dist_avg → VWAP'a uzaklık (ortalama maliyet bölgesi)

Piyasa Bağlamı:
  xu100_mom_60       → XU100'ün 60g momentumu (genel piyasa durumu)
  xu100_above_ma200  → XU100 200g MA üzerinde mi? (yüksek ağırlık: 2.0)
  xu100_drawdown     → XU100 zirveden ne kadar çöküş?
  usd_mom_30, usd_mom_60 → USD/TL momentumu (kur baskısı)

Göreli Güç:
  rel_strength_60/120  → hissenin XU100'e göre gücü (en yüksek ağırlık: 2.5'e kadar)
  sector_rel_60/120    → hissenin kendi sektörüne göre gücü
  xbank_mom_60, xusin_mom_60 → sektör bağlamı
```

### Küme Win-Rate Mantığı
- Training set'teki her hisse her kümeye atanır
- B+C kuralı: küme DEĞİŞTİREN geçişler ve 30 günlük cooldown uygulanır
- Her geçiş için `future_max_gain` bakılır → `>= 0.50` başarı
- Win-rate = başarılı_geçiş / toplam_geçiş

---

## 6. LambdaRankFilter

### Neden Binary Sınıflandırma Değil?
Eski sistem: `>=%50` kazanan → label=1, `<%10` kazanan → label=0
- %49 kazanan ≡ %400 kazanan (her ikisi de 1) → **bilgi kaybı**
- Gri bölge (%10-50) eğitime hiç girmiyor → **bilgi kaybı**
- Model olasılık üretmeye çalışıyor → **kalibrasyon problemi**

LambdaRank çözümü:
- Aynı YearMonth içindeki tüm hisseler kıyaslanır
- `future_max_gain`'e göre 0-4 arası alaka etiketi:
  - 0: en düşük %10'luk dilim (kötü)
  - 4: en yüksek %10'luk dilim (mükemmel)
- "Aynı aydaki en iyi kazananı listeye en üste koy" → NDCG@K
- Çıktı: ham sıralama skoru (negatif olabilir, yüksek = daha iyi)
- Olasılık değil → kalibrasyon problemi yok

### Feature Seti (3 grup)
```
Makro (m_* prefix): ~12 feature
  faiz_politika, enflasyon, usd_kur, vb.

Temel (f_* prefix + _z suffix): ~14 feature
  PD/DD, F/K, Brüt Kâr Marjı, FAVÖK, ROE, değişim oranları
  (sektörel z-score: sektör ortalamasından sapma)

Teknik (TECH_FEATURES): 18 feature
  mom_*, cv_*, rel_strength_*, sector_rel_*, xu100_*, usd_*
```

---

## 7. Twin Matching (DTW)

### Neden DTW?
Klasik Euclid: zaman kaydırmaya duyarlı (2 gün erken zirvede farklı pattern sayılır)
DTW (Dynamic Time Warping): zaman eksenini esnetilebilir → gerçek şekil benzerliği

### 3 Kanal Mantığı
```
Kanal 1 (Fiyat):  Z-score(Pclose[-120:])    → genel trend şekli
Kanal 2 (Hacim):  Z-score(log1p(Vlot[-120:]))  → hacim karakteri
Kanal 3 (Aralık): Z-score((Phigh-Plow)/Pclose) → volatilite profili
```

Üç kanalı aynı anda eşleştirme → sadece fiyata bakmaktan çok daha güçlü

### Parametreler
- `DTW_WINDOW = 20` → Sakoe-Chiba band: zaman ekseni en fazla 20 adım kayabilir
- `MAX_DISTANCE_NDIM = 60.0` → 3 kanallı mesafe eşiği (tek kanal ~2.9'a karşılık)
- `TOP_K = 20` → en yakın 20 ikiz seçilir

---

## 8. Twin Divergence (Post-Entry İzleme)

Bir hisseye girildiğinde backtest motorunda (ve run_full_analysis raporunda) aktive olur.

### Mantık
1. Giriş anında o hissenin başarılı ve başarısız ikizlerinin **ortalama giriş-sonrası fiyat trajektorysi** hesaplanır
2. 60 gün boyunca her 3 günde bir gerçek fiyat trajektorysiyle karşılaştırılır
3. Gerçek hareket başarısızlık medyanına yaklaşırsa → **erken çıkış sinyali**

```
Backtest'te:
  min_check_days = 10  → ilk 10 gün atlanır (gürültü fazla)
  check_interval = 3   → 3 günde bir kontrol
  exit_threshold = 0.0 → medyan score < 0 → çık (failure tarafına kaydı)
  min_twins = 5        → her gruptan en az 5 ikiz yoksa devre dışı
```

---

## 9. Walk-Forward Backtest

`backtest_engine.py` içindeki `run_backtest()` fonksiyonu.

### Strateji Parametreleri
```
Sermaye:      1,000,000 TL
Max Pozisyon: 5 hisse (aynı anda)
Pozisyon:     NAV'in %20'si (her biri)
Stop:         Dinamik = min(peak - 4×ATR(20), peak × 0.80)
              ATR baskınsa geniş stop (volatil hisselerde silkeleme az)
              Floor baskınsa en az %20 nefes garantisi
Komisyon:     %0.2 alış + %0.2 satış
Slippage:     %0.1 taban + piyasa etkisi (işlem TL / ADV20 × 0.10)
Giriş:        Sinyal günü ertesi sabah açılışta
```

### Walk-Forward Takvimi
```
Eğitim epochları: 2018-05-23, 2019-05-23, ..., 2025-05-23 (8 epoch)
Her epoch: O tarihe kadar olan veriyle GMM + LambdaRank eğitimi
           Bir sonraki tarihe kadar sinyal üretimi
           (Look-ahead bias: hiç ilerideki veri kullanılmaz)
Trading dönemi: 2018-05-23 → 2026-05-23
```

### Benchmark
XU100 buy-and-hold (başlangıç sermayesiyle ilk gün alınır, tutulur)

---

## 10. Dosya Yapısı

```
Veri/
│
├── ── ÇALIŞTIRMA SIRASI ──────────────────────────────────────────
│
├── build_knowledge_base.py    ① Model eğitimi (haftada bir)
├── run_full_analysis.py       ② Derin tarama (günlük/isteğe bağlı)
├── run_21_days_analysis.py    ③ 21 günlük geriye dönük tarama
├── run_daily_analysis.py      ④ Hızlı günlük tarama
│
├── ── CORE MODÜLLER (import edilir, doğrudan çalıştırılmaz) ─────
│
├── backtest_engine.py         Walk-forward backtest motoru
├── combined_filter.py         LambdaRankFilter (LGBMRanker)
├── common.py                  Paylaşılan yardımcılar (ATR, ADV, forward gain)
├── config.py                  Tüm sabitler ve path'ler (tek kaynak!)
├── fundamental_engine.py      Temel veri yükleyici + snapshot_at
├── generation_value.py        Nesil değer analizi
├── macro_engine.py            Makro veri yükleyici (EVDS Excel)
├── model_core.py              Feature tanımları, preprocessing, kümeleme util'ları
├── signal_eval.py             Walk-forward sinyal kalitesi değerlendirmesi
├── twin_divergence.py         Post-entry trajektori karşılaştırması
├── twin_matcher.py            3-kanallı DTW ikiz eşleştirici
│
├── ── VERİ DOSYALARI ────────────────────────────────────────────
│
├── market_db.parquet          OHLCV + XU100 + USDTRY (tüm tarih)
├── features_db.parquet        Teknik feature'lar (hesaplanmış)
├── model_artifacts.pkl        Eğitilmiş model (GMM + LambdaRank)
├── knowledge_success_pool.parquet   Başarılı ralli örnekleri
├── knowledge_failure_pool.parquet   Başarısız örnekler
├── knowledge_transitions.parquet    B+C geçiş tablosu
├── knowledge_setups.parquet         Tüm setup'lar
├── sector_map.csv             Ticker → Sektör haritası
├── bist_endeks_uyelikleri.txt Endeks üyelik listesi (sektör haritası için)
│
├── ── ARAÇLAR (ihtiyaç halinde, root'tan çalıştırılır) ──────────
│
├── tools/
│   ├── data_engine.py         market_db.parquet üretici (ham veri → parquet)
│   ├── feature_engine.py      features_db.parquet üretici
│   ├── analyze_clusters.py    Küme karakteristiklerini analiz et
│   ├── signal_eval.py         Walk-forward sinyal kalitesi
│   ├── export_all_trades.py   Backtest trade'lerini dışa aktar
│   └── find_*.py              Belirli küme/hisse arama araçları
│
├── ── ÇIKTILAR ──────────────────────────────────────────────────
│
├── Gunluk_Raporlar/
│   ├── Tam_Analiz/            run_full_analysis.py çıktıları
│   ├── Gunluk_Ozet/           run_daily_analysis.py çıktıları
│   ├── Backtest/              backtest_engine.py çıktıları
│   └── Son_21_Gun_Analizi/    run_21_days_analysis.py çıktıları
│
└── docs/                      Mimari kararlar + plan notları
```

---

## 11. Konfigürasyon (config.py)

Tüm path'ler ve sabitler buradan gelir. **Başka hiçbir dosya path string'i tanımlamamalı.**

```python
# Kritik Eşikler
RALLY_GAIN_THRESHOLD = 0.50    # Başarı: XU100'ü ≥%50 geçme
FAILURE_GAIN_CEILING = 0.10    # Başarısızlık: <%10 üstünlük
RALLY_WINDOW_DAYS    = 252     # Rallinin gerçekleşmesi için max süre (1 yıl)
CLUSTER_COOLDOWN_DAYS = 30     # Aynı (hisse, küme) için minimum aralık

# Portföy
INITIAL_CAPITAL  = 1_000_000
MAX_POSITIONS    = 5
POSITION_WEIGHT  = 0.20        # Her pozisyon NAV'in %20'si
ATR_MULTIPLIER   = 4.0         # Stop: peak - 4 × ATR(20)
TRAILING_FLOOR_PCT = 0.20      # Stop floor: peak'ten max %20 düşüş

# LambdaRank
ENV_FILTER_NDCG_THRESHOLD = 0.0  # 0 = her zaman aktif
COMBINED_CV_WINDOW_YEARS  = 3    # Rolling CV penceresi

# Clustering
CLUSTER_METHOD = "gmm"           # "gmm" | "kmeans" | "hdbscan"
```

---

## 12. Ortam Değişkenleriyle Override

```bash
# Farklı clustering yöntemiyle çalıştır (backtest karşılaştırması için)
CLUSTER_METHOD=kmeans venv/bin/python3 backtest_engine.py

# LambdaRank yerine WinRate sıralaması
SCORE_MODE=winrate venv/bin/python3 run_full_analysis.py

# HDBSCAN parametre taraması
HDB_MCS=80 HDB_MS=10 venv/bin/python3 backtest_engine.py
```

---

## 13. Sinyal Kalitesi Metrikleri

`signal_eval.py` — walk-forward backtest sinyallerini değerlendirir:

| Metrik | Açıklama | İdeal Değer |
|--------|----------|-------------|
| Precision@K | Top-K sinyalin ne kadarı ralli yaptı | > %60 |
| Score@K lift | Rastgele seçime göre kaç kat daha iyi | > 1.15x |
| NDCG@10 | Sıralama kalitesi (P_rally ile) | Yüksek |

**Mevcut sistem performansı (LambdaRank sonrası):**
- Score@K=20 Precision lift: ~1.25x (LambdaRank) vs 1.09x (WinRate)
- Walk-forward NDCG@10: sistem kendi sıralama iyiliğini ölçüyor

---

## 14. Sık Sorulan Sorular

**S: Ne zaman `build_knowledge_base.py` çalıştırmalıyım?**
C: Yeni piyasa verisi yüklendiğinde (haftalık ya da aylık). Model çok sık yeniden eğitilmek zorunda değil — GMM + LambdaRank stabil yapılar.

**S: `run_daily_analysis.py` ile `run_full_analysis.py` farkı ne?**
C: Daily: hızlı, twin matching yok, sıralama üretir. Full: derin analiz, her aday için ikiz senaryoları, tetiklenme zamanlaması, risk analizi, PNG grafikler. Full ~10 kat yavaş ama çok daha fazla bilgi.

**S: Küme sayısı neden değişiyor?**
C: GMM her eğitimde BIC kriterine göre 3-6 arasında optimal küme sayısını seçer. Piyasa koşulları değişince örüntü sayısı da değişebilir.

**S: RankScore negatif olabilir mi?**
C: Evet. LambdaRank ham sıralama puanı üretir (olasılık değil). Negatif skor "bu hisse bu ay kötü sıralanıyor" demek. Büyük = daha iyi, küçük (negatif de olabilir) = daha kötü.

**S: Backtest gerçekçi mi?**
C: Slippage (ADV tabanlı piyasa etkisi), komisyon (%0.2 her yön) ve ertesi gün açılıştan giriş modellendi. Vergi, repo, psikolojik faktörler modellenmedi. Backtest referans değil; metodoloji doğrulama aracı.

---

## 15. Önemli Uyarılar

1. **Look-ahead bias yok:** Her epoch modeli sadece o tarihe kadar olan veriyle eğitir. `build_knowledge_base.py` son 252 günü etiketleme sorunları nedeniyle eğitim dışı bırakır.

2. **Bu sistem finansal tavsiye değildir.** Sinyal üretilmesi alım emri vermek anlamına gelmez. Her sinyali temel analiz ve makro bağlamla değerlendirin.

3. **`model_core.py` doğrudan çalıştırılamaz.** İçindeki `build_knowledge_base()` eski format üretir. Model güncellemek için: `build_knowledge_base.py`.

4. **Veri güncelliği önemlidir.** `features_db.parquet` ve `market_db.parquet` eski kalırsa sinyaller de eski kalır. `tools/data_engine.py` ve `tools/feature_engine.py` ile veriyi güncelleyin.
