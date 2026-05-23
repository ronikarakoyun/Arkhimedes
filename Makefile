# Arkhimedes — Veri Toplama Komutları
# Kullanım: make <target>

PY := venv/bin/python3
FETCHERS := tools/fetchers

.PHONY: help data-daily data-weekly data-quarterly data-all clean-raw

help:
	@echo "Arkhimedes Veri Toplama"
	@echo ""
	@echo "Günlük komutlar:"
	@echo "  make data-daily       → BIST market + KAP incremental update"
	@echo ""
	@echo "Haftalık komutlar:"
	@echo "  make data-weekly      → KAP incremental + market update"
	@echo ""
	@echo "Çeyreklik komutlar (yeni KAP bilanço bildirimleri):"
	@echo "  make data-quarterly   → BIST fundamentals + EVDS makro yenile"
	@echo ""
	@echo "Tam tarama (yılda 1 kez, saatler sürer):"
	@echo "  make data-all         → Sıfırdan KAP arşivi + fundamentals + market"
	@echo ""
	@echo "Bireysel fetcher'lar:"
	@echo "  make fetch-kap        → KAP duyuru arşivi (incremental)"
	@echo "  make fetch-evds       → EVDS makro veriler"
	@echo "  make fetch-fund       → BIST temel veri (TTM + oran hesapları)"
	@echo "  make fetch-market     → BIST günlük fiyat verisi"
	@echo ""
	@echo "Yardımcı:"
	@echo "  make clean-raw        → raw_data/ klasörünü temizle (DİKKAT)"

# ─── Günlük operasyon ────────────────────────────────────────────────────────
data-daily: fetch-market fetch-kap-inc
	@echo "✅ Günlük veri güncellemesi tamamlandı"

# ─── Haftalık operasyon ──────────────────────────────────────────────────────
data-weekly: fetch-market fetch-kap-inc
	@echo "✅ Haftalık veri güncellemesi tamamlandı"

# ─── Çeyreklik operasyon ─────────────────────────────────────────────────────
data-quarterly: fetch-market fetch-kap-inc fetch-fund fetch-evds
	@echo "✅ Çeyreklik veri yenilemesi tamamlandı"

# ─── Yıllık tam tarama ───────────────────────────────────────────────────────
data-all: fetch-market fetch-kap-full fetch-fund fetch-evds
	@echo "✅ Tüm veriler sıfırdan çekildi"

# ─── Bireysel fetcher'lar ────────────────────────────────────────────────────
fetch-market:
	$(PY) $(FETCHERS)/fetch_bist_market.py

fetch-kap-inc:
	$(PY) $(FETCHERS)/fetch_kap_archive.py --incremental

fetch-kap-full:
	$(PY) $(FETCHERS)/fetch_kap_archive.py --full

fetch-fund:
	$(PY) $(FETCHERS)/fetch_bist_fundamentals.py --full

fetch-evds:
	$(PY) $(FETCHERS)/fetch_evds_macro.py

# ─── Test (küçük veri ile) ───────────────────────────────────────────────────
test-kap:
	$(PY) $(FETCHERS)/fetch_kap_archive.py --ticker KUYAS --limit 10

# ─── Temizlik ────────────────────────────────────────────────────────────────
clean-raw:
	@echo "⚠️  raw_data/ klasörü silinecek (5 saniye içinde Ctrl+C ile iptal)"
	@sleep 5
	rm -rf raw_data/
	@echo "✅ raw_data/ silindi"
