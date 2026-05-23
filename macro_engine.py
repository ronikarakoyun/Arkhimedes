"""
Macro Engine — EVDS makro veri loader'ı
=========================================
TCMB EVDS Excel'ini okuyup günlük forward-filled makro feature DataFrame'i üretir.

Look-ahead bias kuralı:
  Her seriye kendi LAG_MAP değeri uygulanır. Veri "ay sonu" (örn 2016-01-31)
  tarihinde dönem-sonu değeri taşır; bu tarihe lag eklenir → fiilen
  piyasaya açıklanma tarihinden sonra kullanılır hale gelir.

Türetilmiş feature'lar lag UYGULANMIŞ base'lerden hesaplanır.
"""
import pandas as pd
import numpy as np
from pathlib import Path

from config import MACRO_EXCEL_PATH

# ─── Sayfa → kolon eşleme (Excel kolon başlıkları birebir) ───────────────────
SHEET_COLS = {
    "Temel_Makro": {
        "Politika_Faizi_AOFM": "m_policy_rate",
        "TÜFE_Genel":          "_tufe_level",       # YoY hesaplanacak
    },
    "Reel_Kesim_Guven_MA": {
        "1.Reel Kesim Güven Endeksi-MA": "m_business_conf",
    },
    "Enflasyon_Beklentileri": {
        "Piyasa katılımcılarının 12 ay sonrası yıllık enflasyon beklentileri (%)": "m_cpi_expect_12m",
    },
    "Para_Arzi_Karsilik": {
        "2.M2": "_m2_level",                         # YoY hesaplanacak
    },
    "Krediler_Bankacilik": {
        "1. TOPLAM YURT İÇİ KREDİ HACMİ": "_credit_level",  # YoY hesaplanacak
    },
    "Mevduat_Faiz_Akim": {
        "Toplam (TL Mevduat, Akım, %)": "m_deposit_rate",
    },
}

# Per-seri lag (gün, takvim günü). Türetilmişler base'in lag'ini miras alır.
LAG_MAP = {
    "m_policy_rate":      1,
    "m_cpi_yoy":          5,
    "m_business_conf":    7,
    "m_cpi_expect_12m":   7,
    "m_m2_yoy":          20,
    "m_credit_yoy":      30,
    "m_deposit_rate":     5,
}


def _parse_tarih(s):
    """'2016-1' veya '01-01-2016' → ay sonu Timestamp."""
    s = str(s).strip()
    if "-" in s and len(s.split("-")[0]) == 4:           # '2016-1' formatı
        y, m = s.split("-")
        return pd.Timestamp(int(y), int(m), 1) + pd.offsets.MonthEnd(0)
    # 'dd-mm-yyyy' formatı
    return pd.to_datetime(s, dayfirst=True) + pd.offsets.MonthEnd(0)


def _load_sheet(xl, sheet, col_map):
    df = pd.read_excel(xl, sheet_name=sheet, usecols=["Tarih", *col_map.keys()])
    df = df.dropna(subset=["Tarih"])
    df["Date"] = df["Tarih"].apply(_parse_tarih)
    df = df.drop(columns=["Tarih"]).rename(columns=col_map)
    df = df.sort_values("Date").drop_duplicates("Date", keep="last")
    return df.set_index("Date")


def _yoy(series: pd.Series) -> pd.Series:
    """12 ay önceki değere göre yüzdesel değişim (%)."""
    return (series / series.shift(12) - 1.0) * 100.0


def load_macro_features(excel_path: str = MACRO_EXCEL_PATH,
                        start_date: str = "2016-01-01",
                        end_date: str | None = None) -> pd.DataFrame:
    """
    EVDS Excel → günlük (Date, m_*) DataFrame.

    Adımlar:
      1) Her sayfayı oku, ay-sonu tarihe ata
      2) Level serileri YoY'a çevir (TÜFE, M2, Kredi)
      3) Per-seri lag uygula (LAG_MAP, calendar gün)
      4) Günlük date range'e reindex + ffill (max 60 gün)
      5) Türetilmiş feature'ları lag sonrası hesapla
    """
    if not Path(excel_path).exists():
        raise FileNotFoundError(f"EVDS Excel bulunamadı: {excel_path}")

    xl = pd.ExcelFile(excel_path)

    # 1) Sayfaları yükle, tek monthly df'e birleştir
    monthly_frames = []
    for sheet, col_map in SHEET_COLS.items():
        if sheet not in xl.sheet_names:
            print(f"[macro_engine] uyarı: {sheet} sayfası yok, atlandı")
            continue
        monthly_frames.append(_load_sheet(xl, sheet, col_map))

    df_monthly = pd.concat(monthly_frames, axis=1).sort_index()

    # Reel getiri deflatörü için ham TÜFE seviyesi (lag YOK, feature DEĞİL)
    cpi_level_monthly = df_monthly["_tufe_level"].copy()

    # 2) Level → YoY dönüşümleri
    df_monthly["m_cpi_yoy"]    = _yoy(df_monthly["_tufe_level"])
    df_monthly["m_m2_yoy"]     = _yoy(df_monthly["_m2_level"])
    df_monthly["m_credit_yoy"] = _yoy(df_monthly["_credit_level"])
    df_monthly = df_monthly.drop(columns=["_tufe_level", "_m2_level", "_credit_level"])

    base_cols = list(LAG_MAP.keys())
    df_monthly = df_monthly[base_cols]

    # 3) Per-seri lag uygula — index'i lag kadar ileri kaydır (effective availability date)
    lagged_frames = []
    for col, lag in LAG_MAP.items():
        s = df_monthly[col].copy()
        s.index = s.index + pd.Timedelta(days=lag)
        lagged_frames.append(s.rename(col))
    df_lagged = pd.concat(lagged_frames, axis=1).sort_index()

    # 4) Günlük reindex + ffill
    if end_date is None:
        end_date = pd.Timestamp.today().normalize()
    daily_idx = pd.date_range(start=start_date, end=end_date, freq="D")
    df_daily = df_lagged.reindex(daily_idx).ffill(limit=60)
    df_daily.index.name = "Date"

    # CPI seviye — reel getiri deflatörü (lagsız; 'm_' öneki yok → feature olmaz)
    df_daily["cpi_level"] = cpi_level_monthly.reindex(daily_idx, method="ffill")

    # 5) Türetilmiş feature'lar (lag sonrası)
    df_daily["m_real_rate"]   = df_daily["m_policy_rate"] - df_daily["m_cpi_yoy"]
    df_daily["m_real_m2"]     = df_daily["m_m2_yoy"]    - df_daily["m_cpi_yoy"]
    df_daily["m_expect_gap"]  = df_daily["m_cpi_expect_12m"] - df_daily["m_cpi_yoy"]
    df_daily["m_conf_mom_6"]  = df_daily["m_business_conf"].pct_change(periods=180) * 100.0  # ~6 ay
    df_daily["m_rate_chg_3m"] = df_daily["m_policy_rate"].diff(periods=90)

    return df_daily.reset_index()


if __name__ == "__main__":
    df = load_macro_features()
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print("Date range:", df["Date"].min(), "→", df["Date"].max())
    print("\nNull oranı (%):")
    print((df.isnull().mean() * 100).round(1))
    print("\nSon 5 satır:")
    print(df.tail())
