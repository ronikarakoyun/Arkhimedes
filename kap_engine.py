"""
kap_engine.py — KAP Duyuru Analiz Motoru (Faz 1)
=================================================
KAP (Kamuyu Aydınlatma Platformu) verilerini Arkhimedes'e bağlar.

LLM/NLP YOK — sadece regex + kategori sayım + recency (hafif, deterministik).

Ana fonksiyonlar:
  load_kap_data(path)              → DataFrame
  categorize_event(subject)        → EVENT_TYPE (str)
  compute_kap_features(ticker, as_of_date, kap_df) → dict (6 feature)
  kap_summary_str(ticker, as_of_date, kap_df)      → "📰 Son 30g: ..." (str)

Olay Türleri:
  CAPITAL_POS   → bedelsiz, geri alım, hisse geri alım programı
  CAPITAL_NEG   → bedelli, sermaye artırım (bedelli)
  DIVIDEND      → temettü, kar payı
  BUYBACK       → aktif geri alım programı bildirimleri
  LEGAL_RISK    → SPK soruşturma, geçici durdurma, iflas, kayyum
  EARNINGS      → finansal rapor, bilanço, sorumluluk beyanı
  GOVERNANCE    → yönetim kurulu, genel kurul, ortaklık yapısı
  INSIDER_TRADE → pay alım/satım bildirimleri, yönetici işlemleri
  OTHER         → yukarıdakilerle eşleşmeyen duyurular
"""
import re
import os
import pandas as pd
import numpy as np

try:
    import polars as pl
    _HAVE_POLARS = True
except ImportError:
    _HAVE_POLARS = False

from config import KAP_DATA_PATH, KAP_LOOKBACK_30, KAP_LOOKBACK_90, KAP_ANOMALY_Z_THR


# ─── Regex Kategorileri ───────────────────────────────────────────────────────

_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("LEGAL_RISK",    re.compile(
        r"spk\s*(soruşturma|inceleme|karar)|geçici\s*durdurma|iflas|kayyum"
        r"|cezai|hukuki\s*süreç|dava\s*(açıldı|sonuç)|borcun\s*ödenmemesi",
        re.IGNORECASE
    )),
    ("CAPITAL_POS",   re.compile(
        r"bedelsiz\s*(sermaye|pay|hisse)|hisse\s*geri\s*alım\s*programı"
        r"|geri\s*alım\s*(programı|başlatıldı|tamamlandı)",
        re.IGNORECASE
    )),
    ("BUYBACK",       re.compile(
        r"geri\s*alım|hisse\s*geri\s*satın|pay\s*geri\s*alım",
        re.IGNORECASE
    )),
    ("CAPITAL_NEG",   re.compile(
        r"bedelli\s*(sermaye|pay)|nakdi\s*sermaye\s*artırım"
        r"|rüçhan\s*hakkı\s*kullanım|yeni\s*pay\s*(ihrac|arz)",
        re.IGNORECASE
    )),
    ("DIVIDEND",      re.compile(
        r"temettü|kâr\s*payı|kar\s*payı|nakit\s*kâr|nakit\s*kar"
        r"|kâr\s*dağıtım|kar\s*dağıtım",
        re.IGNORECASE
    )),
    ("INSIDER_TRADE", re.compile(
        r"pay\s*(alım|satım|edinim)|yönetici\s*(işlem|alım|satım)"
        r"|içeriden\s*öğrenen|içsel\s*bilgi",
        re.IGNORECASE
    )),
    ("EARNINGS",      re.compile(
        r"finansal\s*rapor|bilanço|gelir\s*tablosu|sorumluluk\s*beyanı"
        r"|(yıllık|çeyrek|ara)\s*(dönem\s*)?finansal|(q[1-4]|[1-4]\.\s*çeyrek)\s*sonuç"
        r"|bağımsız\s*denet",
        re.IGNORECASE
    )),
    ("GOVERNANCE",    re.compile(
        r"yönetim\s*kurulu|genel\s*kurul|olağan\s*genel|ortaklık\s*yapısı"
        r"|pay\s*sahipleri|denetim\s*komitesi|bağımsız\s*üye\s*(atama|seçim)"
        r"|ceo|genel\s*müdür\s*(atama|değişiklik)",
        re.IGNORECASE
    )),
]


def categorize_event(subject: str) -> str:
    """
    Duyuru başlığını (subject) regex ile 9 kategoriden birine eşleştir.

    Öncelik sırası önemli: LEGAL_RISK > CAPITAL_POS > BUYBACK > CAPITAL_NEG
    > DIVIDEND > INSIDER_TRADE > EARNINGS > GOVERNANCE > OTHER
    """
    if not subject or not isinstance(subject, str):
        return "OTHER"
    for event_type, pattern in _PATTERNS:
        if pattern.search(subject):
            return event_type
    return "OTHER"


# ─── Veri Yükleme ─────────────────────────────────────────────────────────────

def load_kap_data(path: str = KAP_DATA_PATH) -> pd.DataFrame:
    """
    kap_disclosures.parquet yükle → temizlenmiş DataFrame.

    Dönen sütunlar: Ticker, Date (datetime64), DisclosureIndex, Subject,
                    RelatedTickers, CompanyId, EventType
    """
    if not os.path.exists(path):
        return pd.DataFrame(columns=[
            "Ticker", "Date", "DisclosureIndex", "Subject",
            "RelatedTickers", "CompanyId", "EventType"
        ])

    if _HAVE_POLARS:
        df = pl.read_parquet(path).to_pandas()
    else:
        df = pd.read_parquet(path)

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date", "Ticker"]).copy()
    df["Ticker"] = df["Ticker"].astype(str).str.strip().str.upper()

    # Kategori ekle (daha önce yoksa)
    if "EventType" not in df.columns:
        df["EventType"] = df["Subject"].apply(categorize_event)

    return df.reset_index(drop=True)


# ─── Feature Hesaplama ────────────────────────────────────────────────────────

_POSITIVE_TYPES = {"CAPITAL_POS", "BUYBACK", "DIVIDEND"}
_NEGATIVE_TYPES = {"CAPITAL_NEG", "LEGAL_RISK"}


def compute_kap_features(
    ticker: str,
    as_of_date,
    kap_df: pd.DataFrame,
) -> dict:
    """
    Bir hisse için KAP tabanlı 6 özellik hesapla.

    Parametreler
    ------------
    ticker      : Hisse kodu (örn. "KUYAS")
    as_of_date  : Referans tarihi — bu günden önceki duyurulara bak (look-ahead yok)
    kap_df      : load_kap_data() çıktısı

    Dönen dict
    ----------
    kap_count_30d       : Son 30 gün toplam duyuru
    kap_count_90d       : Son 90 gün toplam duyuru
    kap_positive_30d    : Son 30 günde pozitif olay (bedelsiz/geri alım/temettü)
    kap_negative_30d    : Son 30 günde negatif olay (bedelli/SPK)
    kap_days_since_last : Son herhangi olaydan bu yana gün (NaN = hiç yok)
    kap_anomaly_z       : 90-gün ortalamaya göre Z-skor (anormal yoğunluk)
    """
    as_of = pd.to_datetime(as_of_date)
    empty = {
        "kap_count_30d": 0,
        "kap_count_90d": 0,
        "kap_positive_30d": 0,
        "kap_negative_30d": 0,
        "kap_days_since_last": np.nan,
        "kap_anomaly_z": 0.0,
    }

    if kap_df is None or len(kap_df) == 0:
        return empty

    mask = (kap_df["Ticker"] == ticker.upper()) & (kap_df["Date"] < as_of)
    t_df = kap_df.loc[mask].copy()

    if len(t_df) == 0:
        return empty

    cutoff_30 = as_of - pd.Timedelta(days=KAP_LOOKBACK_30)
    cutoff_90 = as_of - pd.Timedelta(days=KAP_LOOKBACK_90)

    recent_30 = t_df[t_df["Date"] >= cutoff_30]
    recent_90 = t_df[t_df["Date"] >= cutoff_90]

    # Sayımlar
    count_30 = len(recent_30)
    count_90 = len(recent_90)

    # Pozitif / Negatif
    pos_30 = int(recent_30["EventType"].isin(_POSITIVE_TYPES).sum())
    neg_30 = int(recent_30["EventType"].isin(_NEGATIVE_TYPES).sum())

    # Son olaydan bu yana gün
    latest = t_df["Date"].max()
    days_since = (as_of - latest).days if pd.notna(latest) else np.nan

    # Anomaly Z-skor: 90-günlük pencerelerde haftalık disclosure hızı
    # 90 günü 4 haftalık parçalara böl → her parça için sayım → std/mean hesapla
    anomaly_z = 0.0
    if len(recent_90) >= 3:
        # Son 90 günü 6 x 15-günlük dilimlere böl
        bins = pd.cut(
            (as_of - recent_90["Date"]).dt.days,
            bins=list(range(0, 105, 15)),
            labels=False,
            right=False
        )
        bin_counts = bins.value_counts().reindex(range(6), fill_value=0).values.astype(float)
        mean_c = bin_counts.mean()
        std_c  = bin_counts.std(ddof=1)
        if std_c > 0 and count_30 > 0:
            # Son 15-gün dilimindeki sayım = count_30 (yaklaşık)
            anomaly_z = round((count_30 - mean_c) / std_c, 2)

    return {
        "kap_count_30d":       count_30,
        "kap_count_90d":       count_90,
        "kap_positive_30d":    pos_30,
        "kap_negative_30d":    neg_30,
        "kap_days_since_last": days_since,
        "kap_anomaly_z":       anomaly_z,
    }


# ─── Rapor Özeti ─────────────────────────────────────────────────────────────

_EVENT_EMOJI = {
    "CAPITAL_POS":   "✅",
    "BUYBACK":       "✅",
    "DIVIDEND":      "💰",
    "CAPITAL_NEG":   "⚠️",
    "LEGAL_RISK":    "🚨",
    "EARNINGS":      "📊",
    "GOVERNANCE":    "🏛️",
    "INSIDER_TRADE": "👤",
    "OTHER":         "📄",
}

_EVENT_LABEL = {
    "CAPITAL_POS":   "bedelsiz/geri alım",
    "BUYBACK":       "geri alım",
    "DIVIDEND":      "temettü",
    "CAPITAL_NEG":   "bedelli arт.",
    "LEGAL_RISK":    "hukuki risk",
    "EARNINGS":      "finansal rapor",
    "GOVERNANCE":    "yönetim",
    "INSIDER_TRADE": "içeriden işlem",
    "OTHER":         "diğer",
}


def kap_summary_str(
    ticker: str,
    as_of_date,
    kap_df: pd.DataFrame,
    lookback_days: int = KAP_LOOKBACK_30,
) -> str:
    """
    Raporlarda kullanılmak üzere tek satırlık KAP özeti üret.

    Örnek çıktılar:
      "📰 Son 30g: 3 duyuru (✅ geri alım × 1, 📊 finansal rapor × 1, 📄 diğer × 1)"
      "📰 Son 30g: duyuru yok"
      "🚨 Son 30g: 1 hukuki risk!"
    """
    as_of = pd.to_datetime(as_of_date)

    if kap_df is None or len(kap_df) == 0:
        return "📰 KAP verisi yok"

    mask = (
        (kap_df["Ticker"] == ticker.upper()) &
        (kap_df["Date"] < as_of) &
        (kap_df["Date"] >= as_of - pd.Timedelta(days=lookback_days))
    )
    recent = kap_df.loc[mask].copy()

    if len(recent) == 0:
        return f"📰 Son {lookback_days}g: duyuru yok"

    # LEGAL_RISK öncelikli uyarı
    legal_count = int((recent["EventType"] == "LEGAL_RISK").sum())
    if legal_count > 0:
        prefix = f"🚨 Son {lookback_days}g: {legal_count} hukuki risk! "
    else:
        prefix = f"📰 Son {lookback_days}g: {len(recent)} duyuru "

    # Kategori dökümü (en fazla 4 kategori göster)
    counts = recent["EventType"].value_counts()
    parts = []
    for etype, cnt in counts.head(4).items():
        emoji = _EVENT_EMOJI.get(etype, "📄")
        label = _EVENT_LABEL.get(etype, etype.lower())
        parts.append(f"{emoji} {label} × {cnt}" if cnt > 1 else f"{emoji} {label}")

    if parts:
        return prefix + "(" + ", ".join(parts) + ")"
    return prefix.strip()


# ─── Toplu Feature Hesaplama (backtest / run_tracking için) ──────────────────

def batch_kap_features(
    tickers: list[str],
    as_of_date,
    kap_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Birden fazla hisse için kap_features hesapla → DataFrame döner.
    Satır: Ticker | kap_count_30d | ... | kap_anomaly_z
    """
    rows = []
    for ticker in tickers:
        feat = compute_kap_features(ticker, as_of_date, kap_df)
        feat["Ticker"] = ticker
        rows.append(feat)

    cols = ["Ticker", "kap_count_30d", "kap_count_90d",
            "kap_positive_30d", "kap_negative_30d",
            "kap_days_since_last", "kap_anomaly_z"]
    return pd.DataFrame(rows)[cols]


# ─── CLI (basit test) ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    ticker = sys.argv[1].upper() if len(sys.argv) > 1 else "KUYAS"
    date   = sys.argv[2] if len(sys.argv) > 2 else pd.Timestamp.today().strftime("%Y-%m-%d")

    print(f"\n🔍 KAP Engine — {ticker} @ {date}")
    print("─" * 50)

    kap = load_kap_data()
    if len(kap) == 0:
        print("⚠️  kap_disclosures.parquet bulunamadı veya boş.")
        print("   Önce: make fetch-kap-full")
        sys.exit(0)

    features = compute_kap_features(ticker, date, kap)
    summary  = kap_summary_str(ticker, date, kap)

    print(f"   {summary}")
    print()
    for k, v in features.items():
        print(f"   {k:<28}: {v}")

    # Son 5 duyuru
    mask = (kap["Ticker"] == ticker) & (kap["Date"] < pd.to_datetime(date))
    recent = kap[mask].sort_values("Date", ascending=False).head(5)
    if len(recent) > 0:
        print(f"\n   Son 5 duyuru ({ticker}):")
        for _, row in recent.iterrows():
            emoji = _EVENT_EMOJI.get(row.get("EventType", "OTHER"), "📄")
            print(f"   {row['Date'].date()}  {emoji}  {str(row['Subject'])[:70]}")
