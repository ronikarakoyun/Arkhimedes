"""
fetch_kap_archive.py — KAP Duyuru Arşivi Çekici
================================================
MKK VYK API üzerinden tüm BIST hisseleri için KAP duyurularını
çeker ve parquet formatında saklar.

Çıktı (config.KAP_DATA_PATH = kap_disclosures.parquet):
  Ticker | Date | DisclosureIndex | CompanyId | Subject | RelatedTickers | Link

Çalıştırma:
  # Tüm BIST hisseleri için son 10 yıl (ilk çalıştırma, saatler sürer):
  venv/bin/python3 tools/fetchers/fetch_kap_archive.py --full

  # Incremental update (son disclosureIndex'ten itibaren, dakikalar):
  venv/bin/python3 tools/fetchers/fetch_kap_archive.py --incremental

  # Tek hisse:
  venv/bin/python3 tools/fetchers/fetch_kap_archive.py --ticker KUYAS --limit 50

  # Test (sadece 100 disclosure):
  venv/bin/python3 tools/fetchers/fetch_kap_archive.py --full --limit 100
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import argparse
import time
import pandas as pd
import polars as pl
import requests
from requests.auth import HTTPBasicAuth
from datetime import datetime
from tqdm import tqdm

from config import (
    MKK_API_KEY, MKK_API_SECRET, MKK_BASE_URL,
    KAP_DATA_PATH,
)

# ─── Sabitler ────────────────────────────────────────────────────────────────
MIN_INDEX_2015 = 538004      # ~2015 başı (10 yıllık geçmiş)
REQUEST_TIMEOUT = 20         # saniye
RATE_LIMIT_SLEEP = 0.15      # API rate limit
RETRY_MAX = 3

auth = HTTPBasicAuth(MKK_API_KEY, MKK_API_SECRET)


# ─── Yardımcı Fonksiyonlar ───────────────────────────────────────────────────

def _safe_get(url, params=None, retries=RETRY_MAX):
    """API çağrısı + rate limit + retry."""
    for attempt in range(retries):
        try:
            resp = requests.get(url, auth=auth, params=params, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code in (429, 500, 502, 503):
                time.sleep(2 ** attempt)
                continue
            return None
        except Exception:
            time.sleep(1)
    return None


def get_bist_members():
    """Tüm BIST üyelerini çek (ticker → companyId, name mapping)."""
    data = _safe_get(f"{MKK_BASE_URL}/members")
    if not data:
        raise RuntimeError("MKK /members endpoint erişilemedi. API key kontrol et.")
    members = []
    for m in data:
        stock_code = m.get('stockCode')
        company_id = m.get('id')
        title = m.get('title')
        if stock_code and company_id:
            members.append({
                'Ticker': stock_code,
                'CompanyId': company_id,
                'CompanyName': title or '',
            })
    return pd.DataFrame(members)


def get_last_disclosure_index():
    """API'deki son disclosureIndex'i al."""
    data = _safe_get(f"{MKK_BASE_URL}/lastDisclosureIndex")
    if not data:
        return None
    return int(data.get('lastDisclosureIndex', 0))


def fetch_disclosure_detail(disclosure_index):
    """Tek bir disclosure'ın detayını çek (subject, time, related tickers)."""
    data = _safe_get(
        f"{MKK_BASE_URL}/disclosureDetail/{disclosure_index}",
        params={"fileType": "data"}
    )
    if not data:
        return None

    subject = data.get('subject', {})
    if isinstance(subject, dict):
        subject_str = subject.get('tr', '') or subject.get('en', '')
    else:
        subject_str = str(subject or '')

    related = data.get('relatedStocks', []) or []
    if isinstance(related, list):
        related_str = ','.join(str(r) for r in related if r)
    else:
        related_str = str(related or '')

    return {
        'DisclosureIndex': int(disclosure_index),
        'Time': data.get('time', ''),
        'Subject': subject_str,
        'RelatedTickers': related_str,
        'CompanyId': data.get('companyId'),
    }


def fetch_disclosures_for_company(company_id, ticker, max_index=None, min_index=None, limit=None):
    """
    Bir şirket için disclosure'ları çek. Geriye doğru tarayarak indeks indekse iniyor.
    max_index: tarama başlangıcı (None = son indeks)
    min_index: tarama bitişi (None = MIN_INDEX_2015)
    limit: maksimum disclosure sayısı (None = sınırsız)
    """
    if max_index is None:
        max_index = get_last_disclosure_index()
    if min_index is None:
        min_index = MIN_INDEX_2015

    current_idx = max_index
    found = []
    seen_indices = set()

    while current_idx > min_index:
        if limit and len(found) >= limit:
            break

        time.sleep(RATE_LIMIT_SLEEP)
        params = {
            "disclosureIndex": str(current_idx),
            "companyId": str(company_id)
        }
        data = _safe_get(f"{MKK_BASE_URL}/disclosures", params=params)

        if not data:
            current_idx -= 5000
            continue

        if not isinstance(data, list):
            data = [data]

        if not data:
            current_idx -= 5000
            continue

        min_in_block = current_idx
        new_in_block = 0
        for disc in data:
            idx = int(disc.get('disclosureIndex', 0))
            if idx in seen_indices:
                continue
            seen_indices.add(idx)
            if idx < min_in_block:
                min_in_block = idx

            # Detay çek
            time.sleep(RATE_LIMIT_SLEEP)
            detail = fetch_disclosure_detail(idx)
            if detail:
                detail['Ticker'] = ticker
                found.append(detail)
                new_in_block += 1

                if limit and len(found) >= limit:
                    break

        # Bir sonraki tarama: bu blokta bulunan en küçük indeksin 1 altı
        if new_in_block == 0:
            current_idx -= 2000  # boş blok, atla
        else:
            current_idx = min_in_block - 1

    return found


def _parse_kap_time(time_str):
    """KAP API time formatını datetime'a çevir."""
    if not time_str:
        return None
    try:
        # Genelde "DD.MM.YYYY HH:MM:SS" veya benzeri
        for fmt in ("%d.%m.%Y %H:%M:%S", "%d.%m.%Y %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
            try:
                return pd.to_datetime(time_str, format=fmt)
            except (ValueError, TypeError):
                continue
        return pd.to_datetime(time_str, errors='coerce')
    except Exception:
        return None


def disclosures_to_dataframe(disclosures):
    """List of dict → pandas DataFrame."""
    if not disclosures:
        return pd.DataFrame(columns=['Ticker', 'Date', 'DisclosureIndex', 'CompanyId', 'Subject', 'RelatedTickers'])

    df = pd.DataFrame(disclosures)
    df['Date'] = df['Time'].apply(_parse_kap_time)
    df = df.dropna(subset=['Date']).copy()
    df['Date'] = df['Date'].dt.tz_localize(None) if df['Date'].dt.tz is not None else df['Date']
    df['DisclosureIndex'] = df['DisclosureIndex'].astype('int64')

    cols = ['Ticker', 'Date', 'DisclosureIndex', 'CompanyId', 'Subject', 'RelatedTickers']
    df = df[[c for c in cols if c in df.columns]]
    df = df.sort_values(['Ticker', 'Date'], ascending=[True, False]).reset_index(drop=True)
    return df


def load_existing_archive(path=KAP_DATA_PATH):
    """Mevcut parquet'i oku (varsa)."""
    if not os.path.exists(path):
        return pd.DataFrame(columns=['Ticker', 'Date', 'DisclosureIndex', 'CompanyId', 'Subject', 'RelatedTickers'])
    try:
        return pl.read_parquet(path).to_pandas()
    except Exception as e:
        print(f"⚠️  Mevcut arşiv okunamadı: {e}")
        return pd.DataFrame(columns=['Ticker', 'Date', 'DisclosureIndex', 'CompanyId', 'Subject', 'RelatedTickers'])


def save_archive(df, path=KAP_DATA_PATH):
    """DataFrame → parquet (Polars)."""
    if len(df) == 0:
        print("⚠️  Boş DataFrame, parquet'e yazılmadı.")
        return
    df = df.drop_duplicates(subset=['DisclosureIndex']).reset_index(drop=True)
    pl.from_pandas(df).write_parquet(path)
    print(f"✅ {len(df):,} disclosure {path} dosyasına kaydedildi.")


# ─── Ana Akışlar ─────────────────────────────────────────────────────────────

def run_full_archive(limit=None, members_filter=None):
    """Tüm BIST hisseleri için sıfırdan tarama (saatler sürer)."""
    print("📡 BIST üye listesi çekiliyor...")
    members = get_bist_members()
    print(f"   {len(members)} şirket bulundu.")

    if members_filter:
        members = members[members['Ticker'].isin(members_filter)]
        print(f"   Filtre uygulandı: {len(members)} şirket.")

    last_idx = get_last_disclosure_index()
    print(f"📊 Son disclosureIndex: {last_idx:,}  |  Hedef alt sınır: {MIN_INDEX_2015:,}")

    all_disclosures = []
    for _, row in tqdm(members.iterrows(), total=len(members), desc="Hisseler"):
        ticker = row['Ticker']
        company_id = row['CompanyId']
        try:
            disc = fetch_disclosures_for_company(
                company_id, ticker,
                max_index=last_idx,
                min_index=MIN_INDEX_2015,
                limit=limit
            )
            all_disclosures.extend(disc)
        except Exception as e:
            tqdm.write(f"   ⚠️  {ticker}: {e}")

    df = disclosures_to_dataframe(all_disclosures)
    save_archive(df)
    return df


def run_incremental():
    """Son disclosureIndex'ten itibaren incremental tarama."""
    existing = load_existing_archive()
    if len(existing) == 0:
        print("⚠️  Mevcut arşiv boş — önce --full ile tam tarama yap.")
        return None

    existing_max_idx = int(existing['DisclosureIndex'].max())
    api_last_idx = get_last_disclosure_index()
    print(f"📊 Mevcut max indeks: {existing_max_idx:,}  |  API son indeks: {api_last_idx:,}")

    if api_last_idx <= existing_max_idx:
        print("✅ Arşiv zaten güncel.")
        return existing

    delta = api_last_idx - existing_max_idx
    print(f"   {delta:,} yeni disclosure taranacak.")

    members = get_bist_members()
    new_disclosures = []
    for _, row in tqdm(members.iterrows(), total=len(members), desc="Incremental"):
        ticker = row['Ticker']
        company_id = row['CompanyId']
        try:
            disc = fetch_disclosures_for_company(
                company_id, ticker,
                max_index=api_last_idx,
                min_index=existing_max_idx
            )
            new_disclosures.extend(disc)
        except Exception as e:
            tqdm.write(f"   ⚠️  {ticker}: {e}")

    new_df = disclosures_to_dataframe(new_disclosures)
    if len(new_df) > 0:
        merged = pd.concat([existing, new_df], ignore_index=True)
        save_archive(merged)
    else:
        print("   Yeni disclosure bulunamadı.")
    return merged if len(new_df) > 0 else existing


def run_single_ticker(ticker, limit=50):
    """Tek hisse için son N disclosure (debug/test)."""
    members = get_bist_members()
    row = members[members['Ticker'] == ticker.upper()]
    if row.empty:
        print(f"❌ Ticker bulunamadı: {ticker}")
        return None

    company_id = int(row.iloc[0]['CompanyId'])
    name = row.iloc[0]['CompanyName']
    print(f"📊 {ticker} ({name}) — son {limit} disclosure çekiliyor...")

    disc = fetch_disclosures_for_company(
        company_id, ticker,
        max_index=get_last_disclosure_index(),
        min_index=MIN_INDEX_2015,
        limit=limit
    )
    df = disclosures_to_dataframe(disc)
    print(f"\n✅ {len(df)} disclosure çekildi:\n")
    if len(df) > 0:
        print(df[['Date', 'DisclosureIndex', 'Subject']].head(10).to_string(index=False))
    return df


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='KAP Duyuru Arşivi Çekici')
    parser.add_argument('--full', action='store_true', help='Sıfırdan tüm arşiv (saatler)')
    parser.add_argument('--incremental', action='store_true', help='Incremental update')
    parser.add_argument('--ticker', type=str, help='Tek hisse')
    parser.add_argument('--limit', type=int, default=None, help='Maksimum disclosure (test için)')
    parser.add_argument('--symbols', type=str, default=None,
                        help='Virgülle ayrılmış hisse listesi (sadece --full için)')
    args = parser.parse_args()

    if not MKK_API_KEY or not MKK_API_SECRET:
        print("❌ MKK_API_KEY/SECRET eksik. .env dosyasını kontrol et.")
        sys.exit(1)

    if args.ticker:
        run_single_ticker(args.ticker.upper(), limit=args.limit or 50)
    elif args.incremental:
        run_incremental()
    elif args.full:
        symbols_filter = None
        if args.symbols:
            symbols_filter = [s.strip().upper() for s in args.symbols.split(',')]
        run_full_archive(limit=args.limit, members_filter=symbols_filter)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
