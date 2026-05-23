# Ralli Sinyali Güçlendirme Planı

## Amaç

Tek bir sorunun cevabını iyileştirmek: **"Bu hisse önümüzdeki 252 gün içinde %100+ ralli yapacak mı?"**

Çıkış sinyali, risk yönetimi, stop-loss vb. **kapsam dışı.** Sadece giriş sinyalinin precision'ı ve recall'unu artırmaya odaklı.

Şu anki referans precision: **Küme 1 (Gevşek+Trendde) %68.9 — en iyi.** Hedef bu rakamı yukarı çekmek + diğer kümeleri ondan ayırarak temizlemek.

---

## Bölüm 1 — İç Düzeltmeler (yeni veri gerektirmez)

Bunlar mevcut veriyle, kod değişiklikleriyle çözülebilir. Önce bunları bitirmek lazım — yeni veri eklemeden önce sistemi sağlamlaştır.

### 1.1 Lookahead bias düzeltmesi
**Sorun:** `future_max_gain` verinin son 252 gününde eksik pencereli hesaplanıyor. Yakın tarihli setup'lar ralli yapma şansı tam ölçülemeden FP sayılıyor.

**Çözüm:**
- `model_core.py`'de `setups`'u oluştururken `Date <= max_date - 252 gün` filtresi ekle.
- Etkilenen yer: `setups = df_outcomes.filter(...)` satırı (L182-185).
- Sonuç: ~50-100K satır düşecek, precision yapay yükselişten arınacak.

### 1.2 Hesaplanan featureları clustering'e dahil et
**Sorun:** `feature_engine.py` 9 feature üretiyor, clustering sadece 4 kullanıyor. Boşa hesap.

**Çözüm:**
- `FEATURES_FOR_CLUSTERING` listesini genişlet: `rel_vol_120, mom_30, vwap_dist_avg, vol_30` ekle.
- Her yeni feature için `FEATURE_WEIGHTS` defaultu 1.0 ile başla.
- Yeni nüans analizi çalıştır → hangileri TP/FP'yi ayırt ediyor görelim.
- Ayırt etmeyen featureları ağırlık 0.3'e indir veya çıkar.
- Etkilenen dosya: `model_core.py` L19-21.

### 1.3 Çoklu zaman dilimi
**Sorun:** Tek pencere (120 gün). Kısa vadeli (20-30 gün) ve uzun vadeli (252 gün) sinyaller görmüyor.

**Çözüm:**
- `feature_engine.py`'ye ekle: `mom_60, cv_60, vol_60` (orta vade — zaten 30 ve 120 var).
- 252 günlük momentum: `mom_252` (mevcut shift fonksiyonu ile).
- "Sıkışma tutarlılığı": `cv_120 / cv_252` (kısa vade uzun vadeden daha mı sıkı?).
- Bu featureların TP/FP ayırıcılığını ölç → en güçlü 2-3 tanesini clustering'e ekle.

### 1.4 HDBSCAN denemesi
**Sorun:** K-Means tüm noktalara grup atamak zorunda. "Hiçbir gruba ait olmayan" gürültü ayırt edemiyor. k=3 silhouette'i 0.34 — bu üst sınır gibi.

**Çözüm:**
- `model_core.py`'de K-Means yan yana HDBSCAN dene (`hdbscan` paketi).
- HDBSCAN noise (-1) etiketini doğal "tanımsız grup" olarak ele al.
- Her iki modelin precision'ını karşılaştır. Daha iyiyse geçiş yap.
- Trade-off: HDBSCAN deterministik değil, predict(yeni nokta) için ek iş gerekiyor (approximate_predict).

### 1.5 Walk-forward backtest
**Sorun:** Şu anki precision aynı verinin hem eğitim hem testinden. Out-of-sample değil.

**Çözüm:**
- Yeni dosya: `backtest_walkforward.py`.
- Yapı:
  1. Veriyi yıllara böl (örn. 2015-2020 eğit, 2021 test; 2015-2021 eğit, 2022 test; ...).
  2. Her test yılında precision/recall hesapla.
  3. Yıllar arası tutarlılığı raporla.
- Bu, "model overfit mi?" sorusunun en net cevabı.
- Önce 1.1 (lookahead düzeltmesi) bitmeden anlamlı değil.

---

## Bölüm 2 — Dışarıdan Veri Eklemeleri (etkiye göre sıralı)

Yeni veri kaynakları + bunları feature'a çevirme işi. Sırası önemli: kolaydan zora, en yüksek etkiden en düşüğe.

### 2.1 XU100 rejimi (öncelik 1 — kolay, yüksek etki)
**Sorun:** Aynı setup boğa piyasasında ralli yapar, ayı piyasasında yapmaz. Şu an model bunu bilmiyor.

**Çözüm:**
- `market_db.parquet`'ta XU100 zaten var. Sadece kullanmıyoruz.
- `feature_engine.py`'ye ekle:
  - `xu100_mom_60` — endeksin 60 günlük momentumu
  - `xu100_above_ma200` — endeks 200 günlük ortalamasının üstünde mi? (binary)
  - `xu100_drawdown` — endeksin zirveden ne kadar düştüğü
- Bunları her hissenin satırına merge et (Date üzerinden).
- Clustering'e doğrudan eklemek yerine ilk olarak: ayrı bir filtre olarak dene — `xu100_above_ma200 == True` koşulunda precision'a bak.

### 2.2 USD/TRY (öncelik 2 — kolay, BIST'e direkt etki)
**Sorun:** TL bazlı hisselerin önemli kısmı USD ile ters çalışır (ihracatçılar tersine).

**Çözüm:**
- `data_engine.py`'ye `USDTRY=X` ticker'ı ekle (yfinance üzerinden).
- `feature_engine.py`'ye:
  - `usd_mom_30` — USD/TRY 30 günlük momentum
  - `usd_vol_30` — USD volatilitesi
- TP vs FP nüans analizi: hangi kümede USD önemli?

### 2.3 Yabancı takas oranı (öncelik 3 — orta zorluk, yüksek etki)
**Sorun:** BIST'te yabancı yatırımcı hareketleri en güçlü tek sinyallerden biri. Çıkış = düşüş, giriş = yükseliş.

**Veri kaynağı:** Takasbank günlük takas verileri (https://www.takasbank.com.tr) veya MKK API.
**Erişim:** Resmi API yok, web scraping veya 3. parti veri sağlayıcısı (mynet, foreks vb.).

**Çözüm:**
- `data_engine.py`'ye yeni bir fonksiyon: `_fetch_foreign_ratio()`. 
- Yeni parquet: `foreign_ratio_db.parquet` (Ticker, Date, foreign_ratio).
- `feature_engine.py`'ye:
  - `foreign_ratio` — günlük yabancı oran
  - `foreign_ratio_chg_30` — 30 günlük değişim
- Bu en zorlu adım — veri kaynağı belirsiz. **Önce kaynak araştırması.**

### 2.4 Sektör sınıflandırması + sektör endeksleri (öncelik 4 — orta zorluk, ortayüksek etki)
**Sorun:** Bankacılık, GYO, sanayi farklı dinamiklerde. Model hepsini aynı kümeye atıyor.

**Çözüm:**
- KAP'tan sektör listesi (statik, bir kez çek): https://www.kap.org.tr
- Yeni dosya: `sector_map.csv` (Ticker, Sector, SubSector).
- yfinance üzerinden sektör endeksleri çek: XBANK, XGIDA, XSGRT, XKMYA, XUTUM, vs.
- `feature_engine.py`'ye:
  - `sector_rel_strength` — hissenin son 60 gündeki getirisi - sektör endeksi getirisi
  - `sector_mom_60` — sektör endeksi momentumu
- Sektör bilgisi clustering'e kategorik feature olarak değil, **göreli güç** üzerinden girer.

### 2.5 Halka açıklık oranı + piyasa değeri (öncelik 5 — kolay, orta etki)
**Sorun:** Penny stock ve blue chip aynı kümede oluyor. Davranışları çok farklı.

**Veri kaynağı:** KAP statik veriler. Tedavüldeki hisse sayısı + halka açıklık oranı.

**Çözüm:**
- `sector_map.csv`'ye sütun ekle: `free_float_ratio`, `shares_outstanding`.
- `feature_engine.py`'ye:
  - `market_cap = Pclose * shares_outstanding`
  - `free_float_value = market_cap * free_float_ratio`
  - `log_market_cap = log10(market_cap)` (geniş ölçek için)
- Filtre olarak kullan: piyasa değeri çok küçük olanları ayrı kümele veya el. Alternatif: log_market_cap'i feature olarak ekle.

### 2.6 Brent + altın (öncelik 6 — kolay, düşük-orta etki)
**Sorun:** Makro emtia hareketleri BIST'i etkiler (özellikle enerji ve madencilik hisseleri).

**Çözüm:**
- yfinance: `BZ=F` (Brent), `GC=F` (altın).
- `feature_engine.py`'ye:
  - `brent_mom_30`
  - `gold_mom_30`
- Sektörel etkisi (2.4) çalıştıktan sonra eklemek mantıklı — tek başına zayıf sinyal.

---

## Bölüm 3 — Twin Matcher İyileştirmesi (paralel iş)

**Sorun:** Sadece kapanış fiyatı kullanılıyor. Hacim ve fiyat aralığı (range) bilgisi yok. DTW pencere parametresi default.

**Çözüm (`twin_matcher.py`):**
- `get_base_curve` fonksiyonunu **çok kanallı** yap:
  - Kanal 1: Z-Score fiyat (mevcut)
  - Kanal 2: Z-Score hacim (normalize edilmiş)
  - Kanal 3: Z-Score range (Phigh-Plow)/Pclose
- Multi-dimensional DTW: `dtaidistance.dtw_ndim.distance_fast`
- DTW window parametresi ekle (örn. window=20) — daha katı eşleştirme.
- `max_distance` threshold'unu boyutluluğa göre yeniden ayarla.

Bu iş Bölüm 1 ve 2'den bağımsız — paralel yapılabilir.

---

## Uygulama Sırası ve Süre Tahmini

| Adım | Bağımlılık | Tahmini efor |
|---|---|---|
| 1.1 Lookahead düzeltmesi | yok | 1 saat |
| 1.2 Hesaplanan featureları ekle | 1.1 | 2 saat |
| 1.3 Çoklu zaman dilimi | 1.2 | 2 saat |
| 1.5 Walk-forward backtest | 1.1 | 4 saat |
| 2.1 XU100 rejimi | 1.5 | 2 saat |
| 2.2 USD/TRY | 2.1 | 2 saat |
| 2.4 Sektör + endeksler | 2.1 | 1 gün (KAP scraping dahil) |
| 2.5 Piyasa değeri | 2.4 | 3 saat |
| 1.4 HDBSCAN | 1.3 | 4 saat |
| 3 Twin matcher iyileştirme | yok (paralel) | 4 saat |
| 2.3 Yabancı takas | 2.4 | 1-2 gün (kaynak araştırması) |
| 2.6 Brent + altın | 2.4 | 1 saat |

**Toplam: ~5-7 iş günü** (sıralı yapılırsa).

---

## Başarı Kriteri

Her adımdan sonra şu metrikleri ölç ve kaydet:

1. **En iyi küme precision'ı** (şu an %68.9 — Gevşek+Trendde)
2. **Genel precision** (şu an %64.8)
3. **TP sayısı** (recall'ın proxy'si — şu an 11,421)
4. **Silhouette skoru** (şu an 0.34)
5. **Walk-forward yıllar arası precision tutarlılığı** (1.5 yapıldıktan sonra)

Her adım bu rakamların en az 2'sini iyileştirmeli, hiçbirini kötüleştirmemeli. Aksi halde geri al.

---

## Sınırlar

- Çıkış sinyali, stop-loss, position sizing → **bu plan kapsamı dışı.**
- Survivorship bias → veri seti yeniden yapılandırma gerektirir, ayrı bir proje.
- Temel veriler (F/K, PD/DD) → BIST için kalitesi düşük, eklemenin maliyeti yüksek, dahil değil.
- Real-time veri → günlük kapanış yeterli, intraday eklenmiyor.
