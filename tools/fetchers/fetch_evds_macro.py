"""
fetch_evds_macro.py — EVDS (TCMB) Makro Veri Çekici
====================================================
TCMB EVDS API'sinden makro ekonomik verileri (TÜFE, ÜFE, faiz,
para arzı, kredi, vb.) çeker ve Excel formatında saklar.

Çıktı: config.MACRO_EXCEL_PATH (varsayılan: harici yol)

Çalıştırma:
  venv/bin/python3 tools/fetchers/fetch_evds_macro.py
  venv/bin/python3 tools/fetchers/fetch_evds_macro.py --start 2016-01-01 --end 2026-12-31 --output evds.xlsx

Önkoşul:
  pip install evds  (TCMB resmi paketi)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import argparse
import requests
import pandas as pd

from config import EVDS_API_KEY, MACRO_EXCEL_PATH

# ─── Veri Grupları (mevcut sistemden) ────────────────────────────────────────
DATAGROUPS = [
    'bie_mknethar',   # Yabancı Menkul Kıymet Portföyü
    'bie_pbpanal1',   # Parasal Durum
    'bie_pbpanal2',   # Para Arzı / Karşılıklar
    'bie_mt100h',     # Mevduat Faiz Akımı
    'bie_rkgema',     # Reel Kesim Güven MA
    'bie_enfbek',     # Enflasyon Beklentileri
    'bie_krehacbs',   # Krediler Bankacılık
]

SHEET_NAMES = {
    'bie_mknethar': 'YD_Menkul_Kiymet_Portfoyu',
    'bie_pbpanal1': 'Parasal_Durum',
    'bie_pbpanal2': 'Para_Arzi_Karsilik',
    'bie_mt100h':   'Mevduat_Faiz_Akim',
    'bie_rkgema':   'Reel_Kesim_Guven_MA',
    'bie_enfbek':   'Enflasyon_Beklentileri',
    'bie_krehacbs': 'Krediler_Bankacilik',
}

BASIC_SERIES = {
    "TÜFE":           "TP.FG.J0",
    "Yİ-ÜFE":         "TP.TUFE1YI.T1",
    "Politika_Faizi": "TP.APIFON4",
}

BASE_URL = "https://evds3.tcmb.gov.tr/igmevdsms-dis/"


def get_data_in_chunks(api, codes, start, end, chunk_size=8):
    """EVDS URL uzunluk limitini aşmamak için seri kodlarını parçalara böl."""
    dfs = []
    for i in range(0, len(codes), chunk_size):
        chunk = codes[i:i + chunk_size]
        try:
            df = api.get_data(chunk, startdate=start, enddate=end)
            if df is not None and not df.empty:
                cols_to_drop = [c for c in df.columns if 'UNIXTIME' in c or 'YEARWEEK' in c]
                df = df.drop(columns=cols_to_drop)
                dfs.append(df)
        except Exception as e:
            print(f"    Chunk hatası {chunk}: {e}")
    if not dfs:
        return pd.DataFrame()
    final_df = dfs[0]
    for df in dfs[1:]:
        if 'Tarih' in df.columns and 'Tarih' in final_df.columns:
            common_cols = list(set(final_df.columns) & set(df.columns) - {'Tarih'})
            df = df.drop(columns=common_cols)
            final_df = pd.merge(final_df, df, on='Tarih', how='outer')
    return final_df


def fetch_evds(start_date, end_date, output_path):
    """EVDS verilerini çek ve Excel'e yaz."""
    try:
        import evds as evds_module
    except ImportError:
        print("❌ 'evds' paketi yüklü değil. Yüklemek için: pip install evds")
        sys.exit(1)

    if not EVDS_API_KEY:
        print("❌ EVDS_API_KEY eksik. .env dosyasını kontrol et.")
        sys.exit(1)

    api = evds_module.evdsAPI(EVDS_API_KEY)
    headers = {'key': EVDS_API_KEY}

    print(f"📊 EVDS verileri çekiliyor: {start_date} → {end_date}")
    print(f"   Çıktı: {output_path}")

    writer = pd.ExcelWriter(output_path, engine='openpyxl')

    # 1. Temel makro seriler
    print("\n→ Temel makro (TÜFE, ÜFE, Politika Faizi)...")
    try:
        df_basic = api.get_data(list(BASIC_SERIES.values()), startdate=start_date, enddate=end_date)
        cols_to_drop = [c for c in df_basic.columns if 'UNIXTIME' in c]
        df_basic = df_basic.drop(columns=cols_to_drop)
        df_basic.rename(columns={
            "TP_FG_J0":      "TÜFE_Genel",
            "TP_TUFE1YI_T1": "Yİ_ÜFE",
            "TP_APIFON4":    "Politika_Faizi_AOFM",
        }, inplace=True)
        df_basic.to_excel(writer, sheet_name="Temel_Makro", index=False)
        print(f"   ✅ {len(df_basic)} satır")
    except Exception as e:
        print(f"   ❌ Hata: {e}")

    # 2. Veri grupları
    for dg in DATAGROUPS:
        print(f"\n→ [{dg}] grubu...")
        url = BASE_URL + f"serieList/code={dg}&type=json"
        try:
            resp = requests.get(url, headers=headers, timeout=20)
            series_data = resp.json()
            if not isinstance(series_data, list):
                print(f"   ❌ Geçersiz format")
                continue
            series_codes = [s['SERIE_CODE'] for s in series_data]
            if not series_codes:
                print(f"   ⚠️  Seri yok")
                continue
            print(f"   {len(series_codes)} alt seri indiriliyor...")
            df_dg = get_data_in_chunks(api, series_codes, start_date, end_date)
            if df_dg.empty:
                print("   ⚠️  Veri yok")
                continue
            code_to_name = {}
            for s in series_data:
                evds_col_format = s['SERIE_CODE'].replace('.', '_')
                clean_name = s.get('SERIE_NAME_TR', s.get('SERIE_NAME', evds_col_format))
                code_to_name[evds_col_format] = clean_name
            df_dg.rename(columns=code_to_name, inplace=True)
            sheet_name = SHEET_NAMES.get(dg, dg)[:31]
            df_dg.to_excel(writer, sheet_name=sheet_name, index=False)
            print(f"   ✅ {sheet_name}: {len(df_dg)} satır")
        except Exception as e:
            print(f"   ❌ Beklenmeyen hata: {e}")

    writer.close()
    print(f"\n✅ Tamamlandı: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='EVDS Makro Veri Çekici')
    parser.add_argument('--start', type=str, default='2016-01-01', help='Başlangıç tarihi (YYYY-MM-DD)')
    parser.add_argument('--end',   type=str, default=None,         help='Bitiş tarihi (YYYY-MM-DD, varsayılan: bugün)')
    parser.add_argument('--output', type=str, default=MACRO_EXCEL_PATH, help='Çıktı Excel yolu')
    args = parser.parse_args()

    end_date = args.end or pd.Timestamp.today().strftime('%Y-%m-%d')
    # EVDS API formatı: DD-MM-YYYY
    start_fmt = pd.to_datetime(args.start).strftime('%d-%m-%Y')
    end_fmt = pd.to_datetime(end_date).strftime('%d-%m-%Y')

    fetch_evds(start_fmt, end_fmt, args.output)


if __name__ == "__main__":
    main()
