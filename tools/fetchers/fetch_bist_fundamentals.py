"""
fetch_bist_fundamentals.py — BIST Temel Veri Çekici
====================================================
İş Yatırım'dan tüm BIST hisseleri için fiyat ve finansal raporları çeker,
TTM hesapları ve oran analizlerini yapar, parquet'e kaydeder.

İki aşama:
  1. fetch_raw    → raw_data/market/{TICKER}.json ve raw_data/financials/{TICKER}.json
  2. process_raw  → BIST_Tarihsel_Temel_Analiz.parquet (config.FUND_PARQUET_PATH)

Çalıştırma:
  # Tüm BIST hisseleri için ham çek + işle (saatler):
  venv/bin/python3 tools/fetchers/fetch_bist_fundamentals.py --full

  # Sadece ham veri çek (process yapmadan):
  venv/bin/python3 tools/fetchers/fetch_bist_fundamentals.py --fetch-only

  # Sadece mevcut ham veriden parquet üret:
  venv/bin/python3 tools/fetchers/fetch_bist_fundamentals.py --process-only

  # Belirli hisseler:
  venv/bin/python3 tools/fetchers/fetch_bist_fundamentals.py --full --symbols KUYAS,GARAN,THYAO
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import argparse
import json
import time
import warnings
import requests
from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")

from config import FUND_PARQUET_PATH, RAW_DATA_DIR


def get_bist_symbols():
    """İş Yatırım'dan BIST hisse listesini çek."""
    url = "https://www.isyatirim.com.tr/tr-tr/analiz/hisse/Sayfalar/Temel-Degerler-Ve-Oranlar.aspx"
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(url, headers=headers)
    soup = BeautifulSoup(resp.text, 'html.parser')
    symbols = []
    for option in soup.find_all('option'):
        val = option.get('value')
        if val and isinstance(val, str):
            val = val.strip()
            if 2 <= len(val) <= 6 and val.isupper():
                symbols.append(val)
    return sorted(set(symbols))


def fetch_raw_market_data(symbol, start_date, end_date):
    """İş Yatırım fiyat geçmişi."""
    url = (
        f"https://www.isyatirim.com.tr/_layouts/15/IsYatirim.Website/Common/Data.aspx/"
        f"HisseTekil?hisse={symbol}&startdate={start_date}&enddate={end_date}"
    )
    try:
        resp = requests.get(url, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            if 'value' in data and data['value']:
                return data['value']
    except Exception as e:
        print(f"  [Market Error] {symbol}: {e}")
    return None


def fetch_raw_financials(symbol, start_year, end_year):
    """isyatirimhisse paketi üzerinden finansal raporlar."""
    try:
        import isyatirimhisse
    except ImportError:
        print("❌ 'isyatirimhisse' paketi yüklü değil. Yükle: pip install isyatirimhisse")
        sys.exit(1)

    for grp in ['1', '2', '3']:
        try:
            df = isyatirimhisse.fetch_financials(
                symbols=symbol, start_year=start_year, end_year=end_year,
                exchange='TRY', financial_group=grp
            )
            if df is not None and not df.empty:
                return df.to_dict(orient='records')
        except Exception:
            continue
    return None


def run_fetch_raw(start_date="01-01-2016", end_date=None, symbols_filter=None,
                  start_year=2015, end_year=None):
    """Ham veri çek (market + financials)."""
    end_date = end_date or pd.Timestamp.today().strftime('%d-%m-%Y')
    end_year = end_year or pd.Timestamp.today().year

    symbols = symbols_filter or get_bist_symbols()
    os.makedirs(os.path.join(RAW_DATA_DIR, "market"), exist_ok=True)
    os.makedirs(os.path.join(RAW_DATA_DIR, "financials"), exist_ok=True)

    print(f"📡 {len(symbols)} hisse için ham veri çekimi başlatılıyor")
    print(f"   Market: {start_date} → {end_date}")
    print(f"   Finansal: {start_year} → {end_year}")

    for sym in tqdm(symbols, desc="Ham Veri"):
        market_path = os.path.join(RAW_DATA_DIR, "market", f"{sym}.json")
        fin_path = os.path.join(RAW_DATA_DIR, "financials", f"{sym}.json")

        if not os.path.exists(market_path):
            m_data = fetch_raw_market_data(sym, start_date, end_date)
            if m_data:
                with open(market_path, 'w', encoding='utf-8') as f:
                    json.dump(m_data, f, ensure_ascii=False)
            time.sleep(0.5)

        if not os.path.exists(fin_path):
            f_data = fetch_raw_financials(sym, start_year, end_year)
            if f_data:
                with open(fin_path, 'w', encoding='utf-8') as f:
                    json.dump(f_data, f, ensure_ascii=False)
            time.sleep(0.5)

    print("✅ Ham veri çekildi.")


def run_process():
    """Ham veriyi parquet'e işle (oran hesapları + TTM)."""
    from . import _process_bist_raw as proc

    market_dir = os.path.join(RAW_DATA_DIR, "market")
    if not os.path.exists(market_dir):
        print(f"❌ Ham veri yok: {market_dir}. Önce --fetch-only çalıştır.")
        sys.exit(1)

    symbols = [f.replace('.json', '') for f in os.listdir(market_dir) if f.endswith('.json')]
    if not symbols:
        print("❌ Hiç ham veri bulunamadı.")
        sys.exit(1)

    print(f"⚙️  {len(symbols)} hisse işleniyor (TTM + oran hesapları)...")
    all_dfs = []

    # _process_bist_raw modülü `process_hisse(symbol)` fonksiyonu içerir
    # Ancak `raw_data/` klasör yolu hardcoded; geçici cwd değiştir
    original_cwd = os.getcwd()
    try:
        # process_hisse `raw_data/market/{sym}.json` paths bekler → cwd Veri/ olmalı
        os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        os.chdir('..')  # Veri/ kökü
        for sym in tqdm(symbols, desc="İşleme"):
            try:
                df_sym = proc.process_hisse(sym)
                if df_sym is not None and not df_sym.empty:
                    all_dfs.append(df_sym)
            except Exception as e:
                tqdm.write(f"   ⚠️  {sym}: {e}")
    finally:
        os.chdir(original_cwd)

    if not all_dfs:
        print("❌ Hiçbir hisse işlenemedi.")
        sys.exit(1)

    final_df = pd.concat(all_dfs, ignore_index=True)
    print(f"   Toplam {len(final_df):,} satır × {len(final_df.columns)} kolon")

    final_df.to_parquet(FUND_PARQUET_PATH, index=False)
    print(f"✅ Kaydedildi: {FUND_PARQUET_PATH}")


def main():
    parser = argparse.ArgumentParser(description='BIST Temel Veri Çekici + İşleyici')
    parser.add_argument('--full', action='store_true', help='Ham çek + işle')
    parser.add_argument('--fetch-only', action='store_true', help='Sadece ham veri çek')
    parser.add_argument('--process-only', action='store_true', help='Sadece mevcut ham veriyi işle')
    parser.add_argument('--symbols', type=str, default=None, help='Virgülle ayrılmış hisse listesi')
    parser.add_argument('--start-date', type=str, default='01-01-2016', help='Market başlangıcı (DD-MM-YYYY)')
    parser.add_argument('--end-date',   type=str, default=None, help='Market bitişi (DD-MM-YYYY, varsayılan: bugün)')
    args = parser.parse_args()

    symbols_filter = None
    if args.symbols:
        symbols_filter = [s.strip().upper() for s in args.symbols.split(',')]

    if args.process_only:
        run_process()
    elif args.fetch_only:
        run_fetch_raw(args.start_date, args.end_date, symbols_filter)
    elif args.full:
        run_fetch_raw(args.start_date, args.end_date, symbols_filter)
        run_process()
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
