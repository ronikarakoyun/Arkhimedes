"""
Fundamental Engine — BIST temel analiz verisi loader'ı
========================================================
Aylık temel veriyi (F/K, ROE, marj, FD/FAVÖK, vs.) çekip:
  1) Sektör-içi robust z-score normalizasyon (per-ay, per-sektör)
  2) Safe math türetilmiş feature'lar (negatif özkaynak koruması)

Performans notu: Veri zaten aylık. Günlük expand etmek yerine aylık gözlem
frekansında bırakıyoruz — `snapshot_at(date)` yardımcısı merge_asof ile
herhangi bir tarihten geriye en son bilinen değeri alır.

Look-ahead bias: parquet'teki Tarih = aylık gözlem günü; finansallar
process_bist_raw.available_period() ile KAP konsolide bildirim tarihlerine
(Q1→11 May, Q2→19 Ağu, Q3→9 Kas, Q4→11 Mar) göre zaten lag-güvenli eşlenmiş.
Bu yüzden Date = Tarih, EK LAG UYGULANMAZ.
"""
import pandas as pd
import numpy as np
from pathlib import Path

from config import FUND_PARQUET_PATH, SECTOR_MAP_PATH

USABLE_RATIO_COLS = [
    "F_K", "PD_DD", "HAO_PD",
    "ROE_%", "ROA_%",
    "Brut_Kar_Marji_%", "Net_Kar_Marji_%",
    "Cari_Oran", "FD_FAVOK",
]
NORMALIZE_BY_SECTOR = USABLE_RATIO_COLS.copy()

RAW_LEVEL_COLS = [
    "Firma_Degeri", "Net_Borc", "Ozkaynaklar",
    "Satis_Gelirleri_TTM", "Net_Kar_TTM",
    "Toplam_Varliklar", "FAVOK_TTM",
]

DERIVED_COLS = ["f_ev_to_sales", "f_net_debt_to_eq",
                "f_earnings_yield", "f_asset_turnover"]

# Temel ivme feature'ları — period-over-period değişim (Step 2b)
# f_ öneki → CombinedFilter.build_training_set tarafından otomatik alınır
MOMENTUM_DERIVED_COLS = [
    "f_roe_chg",           # ROE_% çeyreksel değişim
    "f_net_margin_chg",    # Net Kâr Marjı % çeyreksel değişim
    "f_favok_margin_chg",  # FAVÖK marjı (FAVOK_TTM/Satis) çeyreksel değişim
    "f_satis_growth",      # Satış Gelirleri TTM büyüme (pct_change)
]


def _safe_div(numer, denom):
    arr_n = np.asarray(numer, dtype=float)
    arr_d = np.asarray(denom, dtype=float)
    out = np.full(arr_n.shape, np.nan, dtype=float)
    mask = (arr_d > 0) & np.isfinite(arr_d) & np.isfinite(arr_n)
    out[mask] = arr_n[mask] / arr_d[mask]
    return out


def load_fundamentals(
    path: str = FUND_PARQUET_PATH,
    sector_map_path: str = SECTOR_MAP_PATH,
) -> pd.DataFrame:
    """
    AvailDate frekansında (Ticker × Date) DataFrame. Daily expand YOK
    — performans için sparse. Downstream consumer merge_asof kullanır.

    Returns columns:
        Ticker, Date(=aylık gözlem günü), Sektor,
        F_K, PD_DD, HAO_PD, ROE_%, ROA_%, Brut_Kar_Marji_%,
        Net_Kar_Marji_%, Cari_Oran, FD_FAVOK,          (orijinal seviyeler)
        F_K_z, PD_DD_z, ..., FD_FAVOK_z,               (sektör-içi robust z)
        f_ev_to_sales, f_net_debt_to_eq, f_earnings_yield, f_asset_turnover
    """
    if not Path(path).exists():
        raise FileNotFoundError(f"Fundamental parquet bulunamadı: {path}")

    raw = pd.read_parquet(path)
    raw = raw.rename(columns={"Symbol": "Ticker", "Tarih": "Date"})
    raw["Date"] = pd.to_datetime(raw["Date"])

    # FD/FAVÖK: process_bist_raw global p1/p99 winsorize'lı sürümü kullan
    if "FD_FAVOK_Winsorized" in raw.columns:
        raw["FD_FAVOK"] = raw["FD_FAVOK_Winsorized"]

    keep_cols = ["Ticker", "Date"] + USABLE_RATIO_COLS + RAW_LEVEL_COLS
    keep_cols = [c for c in keep_cols if c in raw.columns]
    raw = raw[keep_cols].copy()

    # Sektör join (sector_map.csv: Ticker, Sector, SuperSector)
    sm = pd.read_csv(sector_map_path).rename(columns={"Sector": "Sektor"})
    raw = raw.merge(sm[["Ticker", "Sektor"]], on="Ticker", how="left")
    raw["Sektor"] = raw["Sektor"].fillna("BİLİNMİYOR")

    # Look-ahead: Tarih zaten KAP-tarihli lag uygulanmış aylık gözlem günü
    # (process_bist_raw.available_period). EK LAG YOK — Date = Tarih.
    raw = raw.sort_values(["Ticker", "Date"])
    raw = raw.drop_duplicates(["Ticker", "Date"], keep="last")

    # Türetilmiş feature'lar (safe math)
    raw["f_ev_to_sales"]    = _safe_div(raw["Firma_Degeri"],         raw["Satis_Gelirleri_TTM"])
    raw["f_net_debt_to_eq"] = _safe_div(raw["Net_Borc"],             raw["Ozkaynaklar"])
    raw["f_earnings_yield"] = _safe_div(np.ones(len(raw)),            raw["F_K"])
    raw["f_asset_turnover"] = _safe_div(raw["Satis_Gelirleri_TTM"],  raw["Toplam_Varliklar"])

    # Sonsuzları temizle
    for col in DERIVED_COLS + USABLE_RATIO_COLS:
        if col in raw.columns:
            raw[col] = raw[col].replace([np.inf, -np.inf], np.nan)

    # ── Temel ivme feature'ları (Step 2b) ──────────────────────────────────
    # Per-ticker period-over-period değişim. Veri zaten sıralı (Ticker, Date).
    # 1. FAVÖK marjı = FAVOK_TTM / Satis_Gelirleri_TTM
    if "FAVOK_TTM" in raw.columns and "Satis_Gelirleri_TTM" in raw.columns:
        raw["_favok_margin"] = _safe_div(raw["FAVOK_TTM"], raw["Satis_Gelirleri_TTM"])
    else:
        raw["_favok_margin"] = np.nan

    # 2. Period-over-period değişim (diff veya pct_change) per ticker
    for src_col, dst_col, use_pct in [
        ("ROE_%",                "f_roe_chg",          False),
        ("Net_Kar_Marji_%",      "f_net_margin_chg",   False),
        ("_favok_margin",        "f_favok_margin_chg", False),
        ("Satis_Gelirleri_TTM",  "f_satis_growth",     True),
    ]:
        if src_col not in raw.columns:
            raw[dst_col] = np.nan
            continue
        if use_pct:
            raw[dst_col] = (raw.groupby("Ticker")[src_col]
                              .transform(lambda s: s.pct_change()))
        else:
            raw[dst_col] = (raw.groupby("Ticker")[src_col]
                              .transform(lambda s: s.diff()))

    # 3. Winsorize momentum feature'ları (p1-p99) — aykırı değer koruması
    for col in MOMENTUM_DERIVED_COLS:
        if col not in raw.columns:
            continue
        lo = raw[col].quantile(0.01)
        hi = raw[col].quantile(0.99)
        raw[col] = raw[col].clip(lower=lo, upper=hi)
        raw[col] = raw[col].replace([np.inf, -np.inf], np.nan)

    # Sektör-içi robust z-score (per Date × Sektor)
    # AvailDate "cross-section"unda diğer ticker'lar farklı tarihlerde update olmuş
    # olabilir — bu yüzden önce her (Ticker, ay) için en son değeri al, sonra
    # ay-bazında sektör median/MAD hesapla.
    raw["YearMonth"] = raw["Date"].dt.to_period("M")

    for col in NORMALIZE_BY_SECTOR:
        if col not in raw.columns:
            continue
        # Ay × sektör grup median ve MAD
        grp = raw.groupby(["YearMonth", "Sektor"])[col]
        med = grp.transform("median")
        # MAD = median(|x - median|)
        abs_dev = (raw[col] - med).abs()
        mad_grp = abs_dev.groupby([raw["YearMonth"], raw["Sektor"]]).transform("median")
        mad_grp = mad_grp.replace(0, np.nan)
        raw[f"{col}_z"] = (raw[col] - med) / mad_grp

    # Z-score'da kalan NaN'leri sektör-ay median ile değil global 0 ile doldur
    z_cols = [f"{c}_z" for c in NORMALIZE_BY_SECTOR if f"{c}_z" in raw.columns]
    for col in z_cols:
        raw[col] = raw[col].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Derived NaN'leri ay-sektör median ile doldur (boş kalanı 0)
    for col in DERIVED_COLS:
        if col not in raw.columns:
            continue
        med = raw.groupby(["YearMonth", "Sektor"])[col].transform("median")
        raw[col] = raw[col].fillna(med).fillna(0.0)

    # Momentum NaN'leri: sektör-ay medianı yoksa 0 ile doldur (ilk dönem eksikliği)
    for col in MOMENTUM_DERIVED_COLS:
        if col not in raw.columns:
            continue
        med = raw.groupby(["YearMonth", "Sektor"])[col].transform("median")
        raw[col] = raw[col].fillna(med).fillna(0.0)

    out_cols = (["Ticker", "Date", "Sektor"] + USABLE_RATIO_COLS
                + z_cols + DERIVED_COLS + MOMENTUM_DERIVED_COLS)
    out_cols = [c for c in out_cols if c in raw.columns]
    return raw[out_cols].reset_index(drop=True)


def snapshot_at(fund_df: pd.DataFrame,
                queries: pd.DataFrame) -> pd.DataFrame:
    """
    queries: DataFrame with columns (Ticker, Date).
    Returns: queries + tüm fundamental kolonları, her satır için o
             Ticker'ın o Date'ten önceki en son bilinen değeriyle doldurulmuş.
    """
    if queries.empty:
        return queries.copy()
    fund_df = fund_df.copy()
    fund_df["Date"] = pd.to_datetime(fund_df["Date"]).astype("datetime64[ns]")
    fund_df = fund_df.sort_values("Date").reset_index(drop=True)
    queries = queries.copy()
    queries["Date"] = pd.to_datetime(queries["Date"]).astype("datetime64[ns]")
    queries = queries.sort_values("Date").reset_index(drop=True)
    result = pd.merge_asof(
        queries, fund_df,
        on="Date", by="Ticker", direction="backward",
    )
    return result


if __name__ == "__main__":
    import time
    t0 = time.time()
    df = load_fundamentals()
    print(f"Loader süresi: {time.time()-t0:.1f}s")
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print("Ticker count:", df["Ticker"].nunique())
    print("Date range:", df["Date"].min(), "→", df["Date"].max())

    print("\nNull oranı (%) [sadece >0]:")
    nulls = (df.isnull().mean() * 100).round(1)
    print(nulls[nulls > 0].sort_values(ascending=False))

    # Örnek: AKBNK son birkaç dönem
    print("\nAKBNK son 5 dönem:")
    sub = df[df["Ticker"] == "AKBNK"].tail(5)
    print(sub[["Date", "F_K", "F_K_z", "ROE_%", "ROE_%_z", "f_earnings_yield"]])

    # Snapshot testi
    queries = pd.DataFrame({
        "Ticker": ["AKBNK", "EREGL", "SISE"],
        "Date":   ["2024-08-15", "2024-08-15", "2024-08-15"],
    })
    snap = snapshot_at(df, queries)
    print("\nsnapshot_at 2024-08-15:")
    print(snap[["Ticker", "Date", "F_K", "F_K_z", "ROE_%"]])
