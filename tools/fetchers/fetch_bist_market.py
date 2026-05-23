"""
fetch_bist_market.py — BIST Günlük Fiyat Verisi Çekici
=======================================================
Yahoo Finance üzerinden tüm BIST hisseleri için OHLCV verisi çeker.
XU100, USDTRY, XBANK, XUSIN endeksleri/sembolleri de dahil edilir.

Çıktı: config.DB_PATH (varsayılan: market_db.parquet)

Çalıştırma:
  # Tam tarama (10 yıllık geçmiş, ~10 dakika):
  venv/bin/python3 tools/fetchers/fetch_bist_market.py

  # Incremental update (sadece son N gün):
  venv/bin/python3 tools/fetchers/fetch_bist_market.py --days 30

NOT: Bu script tools/data_engine.py'nin işlevini Wrapper olarak sunar.
     Mevcut data_engine.py geriye dönük uyumluluk için korunur.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import argparse


def main():
    parser = argparse.ArgumentParser(description='BIST Market Verisi Çekici')
    parser.add_argument('--days', type=int, default=None,
                        help='Son N gün (varsayılan: tam 10 yıllık tarama)')
    parser.add_argument('--output', type=str, default=None,
                        help='Çıktı parquet yolu (varsayılan: config.DB_PATH)')
    args = parser.parse_args()

    # tools/data_engine.py içindeki data_engine() fonksiyonunu çağır
    from tools.data_engine import data_engine
    data_engine()


if __name__ == "__main__":
    main()
