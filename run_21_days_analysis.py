"""
run_21_days_analysis.py — Arkhimedes Son 21 Gün Geriye Dönük Tarama
====================================================================
Mevcut GMM + LambdaRank modeliyle son 21 işlem gününün her biri için
ayrı ayrı tarama yapar.

Çıktı: Gunluk_Raporlar/Son_21_Gun_Analizi/
  ├── Analiz_YYYY-MM-DD.md       (her gün — tüm adaylar + detaylar)
  ├── Analiz_YYYY-MM-DD.xlsx     (her gün — Excel tablosu)
  ├── Son_1_Ay_Konsolide.md      (21 günün tek sayfalık özeti)
  └── YapayZeka_Yorumlari.md     (stratejik yorum + insan kontrolü listesi)

Çalıştırma:
  venv/bin/python3 run_21_days_analysis.py
"""
import warnings
warnings.filterwarnings("ignore")

import polars as pl
import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path
from collections import defaultdict

from model_core import apply_preprocessing, FEATURES_FOR_CLUSTERING
from combined_filter import TECH_FEATURES
from macro_engine import load_macro_features
from fundamental_engine import load_fundamentals, snapshot_at
from backtest_engine import _assign_clusters
from config import DB_PATH, FEATURES_PATH as FEAT_PATH, ARTIFACTS_PATH, SECTOR_MAP_PATH

OUTPUT_DIR = Path("Gunluk_Raporlar/Son_21_Gun_Analizi")
TOP_N      = 30   # her gün için radar listesi büyüklüğü
ENKAZ_K    = 2.5  # peak/base >= 2.5 + price < %55 peak → enkaz
ENKAZ_DROP = 0.55

# Gösterilecek teknik feature'lar (özet tablo için)
DISPLAY_COLS = [
    "mom_120", "mom_60", "mom_30",
    "sector_rel_60", "rel_strength_60",
    "cv_compression", "v_roc",
    "vol_120",
]

# ─────────────────────────────────────────────────────────────────────────────
# YÜKLE
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("  Arkhimedes — Son 21 Gün Geriye Dönük Tarama")
print("=" * 70)
print("\n📂 Model ve veriler yükleniyor...")

artifacts   = joblib.load(ARTIFACTS_PATH)
df_feat_pl  = pl.read_parquet(FEAT_PATH).with_columns(pl.col("Date").cast(pl.Date))
df_mkt_pl   = pl.read_parquet(DB_PATH).with_columns(pl.col("Date").cast(pl.Date))
macro_df    = load_macro_features()
macro_df["Date"] = pd.to_datetime(macro_df["Date"])
fund_df     = load_fundamentals()

# Sektör haritası (opsiyonel)
sector_map: dict = {}
if os.path.exists(SECTOR_MAP_PATH):
    sm = pd.read_csv(SECTOR_MAP_PATH)
    if "Ticker" in sm.columns and "Sector" in sm.columns:
        sector_map = dict(zip(sm["Ticker"], sm["Sector"]))

# Model bileşenleri
model_bundle = {
    "cluster_method": artifacts.get("clustering_method", "gmm"),
    "clusterer":      artifacts.get("clusterer", artifacts.get("kmeans")),
}
cf          = artifacts.get("combined_filter")
scaler      = artifacts["scaler"]
weight_vec  = artifacts["feature_weights"]
q_low       = artifacts["q_low"]
q_high      = artifacts["q_high"]
cluster_info = artifacts["cluster_info"]
cluster_map_art = artifacts["cluster_map"]
win_rates_art = {c: info["win_rate"] for c, info in cluster_info.items()}
m_cols = [c for c in macro_df.columns if c.startswith("m_")]

# Son 21 gün
all_dates     = sorted(df_feat_pl["Date"].unique().to_list())
last21_dates  = all_dates[-21:]
print(f"   Son 21 gün: {last21_dates[0]} → {last21_dates[-1]}")

# Market verisi pandas'a çevir (enkaz + XU100 için)
df_mkt_pd = df_mkt_pl.to_pandas()
df_mkt_pd["Date"] = pd.to_datetime(df_mkt_pd["Date"])

# Feature DB pandas'a çevir (filtre için)
df_feat_pd = df_feat_pl.to_pandas()
df_feat_pd["Date"] = pd.to_datetime(df_feat_pd["Date"])

# ─────────────────────────────────────────────────────────────────────────────
# TOPLU FUNDAMENTAL SNAPSHOT (21 gün × tüm hisseler — tek merge_asof)
# ─────────────────────────────────────────────────────────────────────────────
print("   Fundamental snapshot (toplu) hazırlanıyor...")
all_tickers = [t for t in df_feat_pd["Ticker"].unique() if t != "XU100"]
bulk_queries = pd.DataFrame(
    [(t, pd.to_datetime(d)) for d in last21_dates for t in all_tickers],
    columns=["Ticker", "Date"],
)
bulk_snap = snapshot_at(fund_df, bulk_queries)
bulk_snap = bulk_snap.set_index(["Ticker", "Date"])
f_cols = [c for c in bulk_snap.columns if c.startswith("f_") or
          (c.endswith("_z") and not c.startswith("m_"))]
print(f"   ✅ Toplu snapshot: {len(bulk_snap):,} satır, {len(f_cols)} fundamental feature")

# ─────────────────────────────────────────────────────────────────────────────
# YARDIMCI FONKSİYONLAR
# ─────────────────────────────────────────────────────────────────────────────

def _enkaz_filter(df_mkt: pd.DataFrame, as_of_date) -> list:
    """Enkaz filtresi — as_of_date'e kadar olan veriyle hesapla."""
    sub = df_mkt[df_mkt["Date"] <= as_of_date]
    grp = (
        sub.groupby("Ticker")
        .agg(
            cur=("Pclose", "last"),
            hi252=("Phigh", lambda x: x.tail(252).max()),
            lo252=("Plow",  lambda x: x.tail(252).min()),
        )
        .reset_index()
    )
    grp = grp[grp["Ticker"] != "XU100"]
    survivor_mask = ~(
        (grp["hi252"] > grp["lo252"] * ENKAZ_K) &
        (grp["cur"] < grp["hi252"] * ENKAZ_DROP)
    )
    return grp[survivor_mask]["Ticker"].tolist()


def _xu100_context(df_mkt: pd.DataFrame, as_of_date) -> dict:
    """XU100 kapanış + günlük getiri."""
    sub = (df_mkt[(df_mkt["Ticker"] == "XU100") & (df_mkt["Date"] <= as_of_date)]
           .sort_values("Date").tail(2))
    if len(sub) == 0:
        return {"close": None, "daily_ret": None}
    close = float(sub.iloc[-1]["Pclose"])
    daily = (float(sub.iloc[-1]["Pclose"]) / float(sub.iloc[-2]["Pclose"]) - 1) * 100 if len(sub) >= 2 else None
    return {"close": close, "daily_ret": daily}


def _screen_day(day_date) -> dict | None:
    """Tek bir gün için GMM + LambdaRank taraması."""
    day_dt = pd.to_datetime(day_date)
    date_str = str(day_date)[:10]

    # Enkaz filtresi
    survivors = _enkaz_filter(df_mkt_pd, day_dt)
    if not survivors:
        return None

    # Son bilinen feature satırı — o güne kadar
    feats = (
        df_feat_pd[df_feat_pd["Date"] <= day_dt]
        .sort_values("Date")
        .groupby("Ticker").last()
        .reset_index()
    )
    feats = feats[feats["Ticker"].isin(survivors)].dropna(subset=FEATURES_FOR_CLUSTERING)
    if len(feats) == 0:
        return None

    # Preprocessing + GMM
    pp = apply_preprocessing(feats[FEATURES_FOR_CLUSTERING], FEATURES_FOR_CLUSTERING, q_low, q_high)
    valid = pp.notna().all(axis=1)
    feats = feats[valid].copy().reset_index(drop=True)
    pp    = pp[valid].reset_index(drop=True)

    X = scaler.transform(pp[FEATURES_FOR_CLUSTERING]) * weight_vec
    labels, uyum = _assign_clusters(model_bundle, X)
    feats["Cluster"]  = labels
    feats["Karakter"] = feats["Cluster"].map(cluster_map_art)
    feats["WinRate"]  = feats["Cluster"].map(win_rates_art).fillna(0.0)
    feats["Uyum"]     = uyum
    feats["Score"]    = 0.7 * feats["WinRate"] + 0.3 * feats["Uyum"]

    # Sektör etiketi
    if "Sector" not in feats.columns:
        feats["Sector"] = feats["Ticker"].map(sector_map).fillna("Bilinmiyor")

    # LambdaRank RankScore
    if cf is not None and cf.is_useful():
        # Macro snapshot
        macro_row = macro_df[macro_df["Date"] <= day_dt].sort_values("Date").tail(1)
        for col in m_cols:
            feats[col] = macro_row.iloc[0][col] if not macro_row.empty else np.nan

        # Fundamental snapshot — toplu batch'ten çek
        for col in f_cols:
            if col in bulk_snap.columns:
                vals = []
                for tk in feats["Ticker"]:
                    try:
                        vals.append(bulk_snap.at[(tk, day_dt), col])
                    except KeyError:
                        vals.append(np.nan)
                feats[col] = vals
            else:
                feats[col] = np.nan

        feat_cols = cf.feature_cols
        for c in feat_cols:
            if c not in feats.columns:
                feats[c] = np.nan
        try:
            feats["RankScore"] = cf.predict_rank_score(feats[feat_cols])
            feats["Score"]     = feats["RankScore"]
        except Exception as e:
            feats["RankScore"] = np.nan
    else:
        feats["RankScore"] = np.nan

    # Top N
    sort_col = "RankScore" if feats["RankScore"].notna().any() else "WinRate"
    top = feats.nlargest(TOP_N, sort_col).reset_index(drop=True)

    xu100 = _xu100_context(df_mkt_pd, day_dt)
    return {
        "date":      date_str,
        "top":       top,
        "n_scanned": len(feats),
        "xu100":     xu100,
    }


def _fmt_pct(v, digits=1):
    if pd.isna(v):
        return "—"
    return f"{v*100:+.{digits}f}%"


def _fmt_f(v, digits=2):
    if pd.isna(v):
        return "—"
    return f"{v:+.{digits}f}"


def _cluster_badge(wr: float) -> str:
    if wr >= 0.65:
        return "🌟"
    elif wr <= 0.35:
        return "⚠️"
    return "⚪"


# ─────────────────────────────────────────────────────────────────────────────
# GÜNLÜK MARKDOWN
# ─────────────────────────────────────────────────────────────────────────────

def write_daily_md(day: dict, out_dir: Path):
    top = day["top"]
    date_str = day["date"]
    xu = day["xu100"]

    xu_str = "—"
    if xu["close"]:
        sign = f"{xu['daily_ret']:+.2f}%" if xu["daily_ret"] is not None else "—"
        xu_str = f"{xu['close']:,.0f}  ({sign})"

    lines = []
    lines.append(f"# Arkhimedes Tarama — {date_str}\n")
    lines.append(f"**XU100:** {xu_str}  |  "
                 f"**Taranan hisse:** {day['n_scanned']}  |  "
                 f"**Radar listesi:** {len(top)}\n")

    # Küme dağılımı
    lines.append("## 📊 Küme Dağılımı\n")
    lines.append("| Küme | WinRate | Adaylar | Ort. RankScore |")
    lines.append("|---|---|---|---|")
    for cl_id, grp in top.groupby("Cluster"):
        wr = win_rates_art.get(int(cl_id), 0.0)
        avg_rs = grp["RankScore"].mean()
        badge = _cluster_badge(wr)
        lines.append(f"| {cluster_map_art.get(int(cl_id), cl_id)} | "
                     f"{badge} %{wr*100:.1f} | {len(grp)} | "
                     f"{_fmt_f(avg_rs)} |")

    # Radar listesi
    lines.append("\n## 📡 Radar Listesi (RankScore sıralı)\n")
    lines.append("| # | Ticker | Sektör | Küme | WR | Uyum | RankScore | "
                 "mom120 | mom60 | sRel60 | rsXU60 | cvComp | vRoc |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")

    for i, row in top.iterrows():
        wr   = row.get("WinRate", np.nan)
        uyum = row.get("Uyum", np.nan)
        rs   = row.get("RankScore", np.nan)
        badge = _cluster_badge(wr)
        sector = str(row.get("Sector", "—"))[:18]
        karakter = str(row.get("Karakter", "—"))[:18]

        def g(col):
            v = row.get(col, np.nan)
            return _fmt_pct(v) if not pd.isna(v) else "—"

        lines.append(
            f"| {i+1} | **{row['Ticker']}** | {sector} | {karakter} | "
            f"{badge} {wr*100:.0f}% | {uyum:.3f} | **{_fmt_f(rs)}** | "
            f"{g('mom_120')} | {g('mom_60')} | {g('sector_rel_60')} | "
            f"{g('rel_strength_60')} | {g('cv_compression')} | {g('v_roc')} |"
        )

    # Sektör konsantrasyonu
    if "Sector" in top.columns:
        lines.append("\n## 🏭 Sektör Konsantrasyonu\n")
        sc = top["Sector"].value_counts()
        for sec, cnt in sc.items():
            bar = "▓" * cnt
            lines.append(f"- **{sec}**: {cnt} aday  {bar}")

    lines.append("\n---\n*Arkhimedes — GMM + LambdaRank  |  Look-ahead yok*\n")

    out_path = out_dir / f"Analiz_{date_str}.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# GÜNLÜK EXCEL
# ─────────────────────────────────────────────────────────────────────────────

def write_daily_xlsx(day: dict, out_dir: Path):
    top = day["top"]
    date_str = day["date"]

    keep_cols = ["Ticker", "Sector", "Cluster", "Karakter", "WinRate",
                 "Uyum", "RankScore"] + [c for c in DISPLAY_COLS if c in top.columns]
    out_cols = [c for c in keep_cols if c in top.columns]
    df_out = top[out_cols].copy()

    # Format
    for col in ["WinRate"]:
        if col in df_out:
            df_out[col] = (df_out[col] * 100).round(1)
    for col in ["Uyum", "RankScore"]:
        if col in df_out:
            df_out[col] = df_out[col].round(4)
    for col in DISPLAY_COLS:
        if col in df_out:
            df_out[col] = (df_out[col] * 100).round(2)

    xu = day["xu100"]
    meta = pd.DataFrame({
        "Bilgi": ["Tarih", "XU100 Kapanış", "XU100 Günlük %", "Taranan Hisse"],
        "Değer": [
            date_str,
            f"{xu['close']:,.0f}" if xu["close"] else "—",
            f"{xu['daily_ret']:+.2f}%" if xu["daily_ret"] else "—",
            day["n_scanned"],
        ]
    })

    out_path = out_dir / f"Analiz_{date_str}.xlsx"
    with pd.ExcelWriter(str(out_path), engine="openpyxl") as writer:
        df_out.to_excel(writer, sheet_name="Radar", index=False)
        meta.to_excel(writer, sheet_name="Özet", index=False)


# ─────────────────────────────────────────────────────────────────────────────
# KONSOLİDE ÖZET
# ─────────────────────────────────────────────────────────────────────────────

def write_consolidated_md(all_days: list, out_dir: Path):
    lines = []
    lines.append("# Son 21 Gün Konsolide Özet — Arkhimedes\n")

    valid = [d for d in all_days if d is not None]
    start = valid[0]["date"]
    end   = valid[-1]["date"]
    lines.append(f"**Dönem:** {start} → {end}  |  **{len(valid)} işlem günü**\n")

    # XU100 serisi
    xu100_rets = [d["xu100"]["daily_ret"] for d in valid if d["xu100"]["daily_ret"] is not None]
    xu100_closes = [d["xu100"]["close"] for d in valid if d["xu100"]["close"] is not None]
    if xu100_closes:
        period_ret = (xu100_closes[-1] / xu100_closes[0] - 1) * 100
        lines.append(f"**XU100 Dönem Getirisi:** {period_ret:+.2f}%  "
                     f"({xu100_closes[0]:,.0f} → {xu100_closes[-1]:,.0f})\n")

    # Günlük özet tablo
    lines.append("## 📅 Günlük Radar Özeti\n")
    lines.append("| Tarih | XU100 | Gün% | Taranan | Adaylar | Ort.RS | En yüksek RS |")
    lines.append("|---|---|---|---|---|---|---|")
    for d in valid:
        xu = d["xu100"]
        close_s = f"{xu['close']:,.0f}" if xu["close"] else "—"
        ret_s   = f"{xu['daily_ret']:+.2f}%" if xu["daily_ret"] is not None else "—"
        top = d["top"]
        avg_rs = top["RankScore"].mean() if "RankScore" in top.columns else np.nan
        max_rs = top["RankScore"].max()  if "RankScore" in top.columns else np.nan
        lines.append(f"| {d['date']} | {close_s} | {ret_s} | "
                     f"{d['n_scanned']} | {len(top)} | "
                     f"{_fmt_f(avg_rs)} | {_fmt_f(max_rs)} |")

    # En çok radar'a giren hisseler
    ticker_days: dict = defaultdict(list)
    for d in valid:
        for _, row in d["top"].iterrows():
            tk = row["Ticker"]
            rs = row.get("RankScore", np.nan)
            wr = row.get("WinRate", np.nan)
            sec = row.get("Sector", "—")
            ticker_days[tk].append({"rs": rs, "wr": wr, "sec": sec})

    freq = {tk: len(v) for tk, v in ticker_days.items()}
    sorted_freq = sorted(freq.items(), key=lambda x: x[1], reverse=True)

    lines.append("\n## 🔁 En Sık Radar'a Giren Hisseler (≥5 gün)\n")
    lines.append("| Ticker | Sektör | Gün | Ort. RS | Ort. WR | Trend |")
    lines.append("|---|---|---|---|---|---|")
    for tk, cnt in sorted_freq:
        if cnt < 5:
            break
        entries = ticker_days[tk]
        avg_rs = np.nanmean([e["rs"] for e in entries])
        avg_wr = np.nanmean([e["wr"] for e in entries])
        sec    = entries[0]["sec"]

        # Trend: ilk yarı vs son yarı RS
        rs_vals = [e["rs"] for e in entries if not np.isnan(e["rs"])]
        if len(rs_vals) >= 4:
            mid = len(rs_vals) // 2
            trend_dir = "📈" if np.mean(rs_vals[mid:]) > np.mean(rs_vals[:mid]) else "📉"
        else:
            trend_dir = "—"

        lines.append(f"| **{tk}** | {sec} | {cnt} | {_fmt_f(avg_rs)} | "
                     f"%{avg_wr*100:.1f} | {trend_dir} |")

    # Sektör konsantrasyonu (toplam 21 gün)
    all_sectors: dict = defaultdict(int)
    for d in valid:
        for _, row in d["top"].iterrows():
            s = row.get("Sector", "Bilinmiyor")
            all_sectors[str(s)] += 1
    sorted_sec = sorted(all_sectors.items(), key=lambda x: x[1], reverse=True)

    lines.append("\n## 🏭 21 Gün Kümülatif Sektör Konsantrasyonu\n")
    lines.append("| Sektör | Toplam Giriş |")
    lines.append("|---|---|")
    for sec, cnt in sorted_sec[:15]:
        lines.append(f"| {sec} | {cnt} |")

    # Cluster dağılımı
    cluster_freq: dict = defaultdict(int)
    for d in valid:
        for _, row in d["top"].iterrows():
            cl = row.get("Cluster")
            cluster_freq[cl] += 1
    lines.append("\n## 🧩 21 Gün Küme Dağılımı\n")
    lines.append("| Küme | Karakter | WinRate | Toplam Giriş |")
    lines.append("|---|---|---|---|")
    for cl_id, cnt in sorted(cluster_freq.items(), key=lambda x: x[1], reverse=True):
        wr = win_rates_art.get(int(cl_id) if cl_id is not None else -1, 0.0)
        karakter = cluster_map_art.get(int(cl_id) if cl_id is not None else -1, "—")
        lines.append(f"| {cl_id} | {karakter} | %{wr*100:.1f} | {cnt} |")

    lines.append("\n---\n*Arkhimedes — GMM + LambdaRank*\n")
    out_path = out_dir / "Son_1_Ay_Konsolide.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"   ✅ {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# YAPAY ZEKA YORUMU
# ─────────────────────────────────────────────────────────────────────────────

def write_ai_commentary(all_days: list, out_dir: Path):
    """Kuantitatif verileri analiz ederek stratejik yorum üretir."""
    valid = [d for d in all_days if d is not None]
    if not valid:
        return

    # Veri topla
    xu100_closes = [(d["date"], d["xu100"]["close"]) for d in valid if d["xu100"]["close"]]
    xu100_rets   = [d["xu100"]["daily_ret"] for d in valid if d["xu100"]["daily_ret"] is not None]
    avg_rs_series = []
    n_candidates_series = []
    for d in valid:
        avg_rs = d["top"]["RankScore"].mean() if "RankScore" in d["top"] else np.nan
        avg_rs_series.append((d["date"], avg_rs))
        n_candidates_series.append(len(d["top"]))

    # XU100 analizi
    period_ret = (xu100_closes[-1][1] / xu100_closes[0][1] - 1) * 100 if len(xu100_closes) >= 2 else 0
    neg_days  = sum(1 for r in xu100_rets if r < 0)
    pos_days  = sum(1 for r in xu100_rets if r >= 0)
    max_drop  = min(xu100_rets) if xu100_rets else 0
    max_rise  = max(xu100_rets) if xu100_rets else 0

    # Piyasa rejim tespiti
    if period_ret <= -10:
        rejim = "GÜÇLÜ AYIK (Derin Düzeltme)"
        rejim_emoji = "🐻🔴"
    elif period_ret <= -5:
        rejim = "AYIK (Orta Düzeltme)"
        rejim_emoji = "🐻🟠"
    elif period_ret <= 0:
        rejim = "NÖTR-NEGATİF (Zayıf)"
        rejim_emoji = "⚖️🟡"
    elif period_ret <= 5:
        rejim = "NÖTR-POZİTİF (Toparlanma)"
        rejim_emoji = "⚖️🟢"
    else:
        rejim = "BOĞANEGATİF (Yükseliş)"
        rejim_emoji = "🐂🟢"

    # Cluster karakteri
    cluster_freq: dict = defaultdict(int)
    for d in valid:
        for _, row in d["top"].iterrows():
            cl = row.get("Cluster")
            cluster_freq[cl] += 1
    total_entries = sum(cluster_freq.values())
    dominant_cluster = max(cluster_freq, key=cluster_freq.get) if cluster_freq else None
    dominant_wr = win_rates_art.get(int(dominant_cluster) if dominant_cluster is not None else -1, 0)
    dominant_name = cluster_map_art.get(int(dominant_cluster) if dominant_cluster is not None else -1, "—")

    if dominant_wr >= 0.60:
        char = "AGRESIF (Yüksek win-rate kümeleri dominant)"
        char_emoji = "⚡"
    elif dominant_wr >= 0.45:
        char = "ORTA (Karışık sinyal)"
        char_emoji = "🎯"
    else:
        char = "DEFANSİF (Düşük win-rate kümeleri dominant)"
        char_emoji = "🛡️"

    # RS trend (ilk yarı vs son yarı)
    rs_first = np.nanmean([v for _, v in avg_rs_series[:len(avg_rs_series)//2]])
    rs_last  = np.nanmean([v for _, v in avg_rs_series[len(avg_rs_series)//2:]])
    rs_trend = rs_last - rs_first
    rs_trend_str = f"{'İYİLEŞİYOR 📈' if rs_trend > 0.1 else 'KÖTÜLEŞIYOR 📉' if rs_trend < -0.1 else 'YATAY ↔️'}"

    # Sık tekrar edenler
    ticker_days: dict = defaultdict(list)
    for d in valid:
        for _, row in d["top"].iterrows():
            ticker_days[row["Ticker"]].append(row.get("RankScore", np.nan))
    recurring = [(tk, len(v), np.nanmean(v)) for tk, v in ticker_days.items() if len(v) >= 10]
    recurring.sort(key=lambda x: x[1], reverse=True)

    # Sektör konsantrasyonu
    all_sectors: dict = defaultdict(int)
    for d in valid:
        for _, row in d["top"].iterrows():
            all_sectors[str(row.get("Sector", "Bilinmiyor"))] += 1
    top5_sectors = sorted(all_sectors.items(), key=lambda x: x[1], reverse=True)[:5]

    # Aday sayısı trendi
    avg_cand_first = np.mean(n_candidates_series[:len(n_candidates_series)//2])
    avg_cand_last  = np.mean(n_candidates_series[len(n_candidates_series)//2:])

    # ─── YORUM YAZAR ───
    lines = []
    lines.append("# 🤖 Yapay Zeka Stratejik Yorumları — Son 21 Gün\n")
    lines.append(f"*Üretildi: {valid[-1]['date']} itibarıyla Arkhimedes (GMM + LambdaRank) kantitatif analizi*\n")
    lines.append("> **Önemli Uyarı:** Bu bölüm bir LLM görüşü değil, sistemin kendi çıktılarına dayanan ")
    lines.append("> kuantitatif bir analizdir. Nihai karar her zaman insana aittir.\n")

    lines.append("---\n")
    lines.append("## 1. Piyasa Rejimi Değerlendirmesi\n")
    lines.append(f"**Tespit:** {rejim_emoji} **{rejim}**\n")
    lines.append(f"| Metrik | Değer |")
    lines.append(f"|---|---|")
    lines.append(f"| Dönem getirisi | {period_ret:+.2f}% |")
    lines.append(f"| Pozitif gün | {pos_days}/{len(xu100_rets)} |")
    lines.append(f"| Negatif gün | {neg_days}/{len(xu100_rets)} |")
    lines.append(f"| En sert düşüş | {max_drop:+.2f}% |")
    lines.append(f"| En güçlü yükseliş | {max_rise:+.2f}% |\n")

    if period_ret <= -8:
        lines.append("**Yorum:** Dönem piyasa açısından ciddi baskı altında geçmiştir. XU100'deki ")
        lines.append(f"**{period_ret:+.1f}%**'lik düşüş özellikle dönem sonundaki sert satış dalgasıyla ")
        lines.append("derinleşmiştir. Bu ortamda sistemin ürettiği sinyaller normalden daha yüksek ")
        lines.append("risk taşımaktadır: piyasa genelinin düştüğü ortamda göreli güç gösteren ")
        lines.append("adaylar ya gerçekten güçlüdür ya da henüz satışa maruz kalmamıştır.\n")
    elif period_ret <= -3:
        lines.append("**Yorum:** Orta şiddetli bir düzeltme yaşanmıştır. Sistemin sinyalleri bu ")
        lines.append("ortamda seçici olunmasını gerektirmektedir.\n")
    else:
        lines.append("**Yorum:** Dönem görece stabil geçmiştir. Sistem sinyalleri normal güvenilirlik ")
        lines.append("aralığında değerlendirilebilir.\n")

    lines.append("\n## 2. Sistem Karakteri — Agresif mi, Defansif mi?\n")
    lines.append(f"**Tespit:** {char_emoji} **{char}**\n")
    lines.append(f"Dönem boyunca dominant küme: **{dominant_name}** (WinRate: %{dominant_wr*100:.1f})\n")

    if dominant_wr < 0.45:
        lines.append("**Yorum:** Yüksek win-rate'li kümeler (%65+) 21 gün boyunca neredeyse hiç ")
        lines.append("aday üretmemiştir. Bu, piyasanın 'kaliteli ralli formasyonu' gereksinimlerini ")
        lines.append("karşılamadığı anlamına gelir. Mevcut adaylar **fırsatçı/spekülatif** ")
        lines.append("karakterde olup portföy büyüklükleri buna göre küçük tutulmalıdır.\n")
    elif dominant_wr >= 0.55:
        lines.append("**Yorum:** Sistem ağırlıklı olarak orta-yüksek win-rate kümelerinden aday ")
        lines.append("üretmiştir. Mevcut piyasa koşulları kaliteli formasyon oluşumuna izin ")
        lines.append("vermekte, ancak XU100'deki zayıflık dikkatli pozisyon yönetimini gerektirmektedir.\n")

    lines.append("\n## 3. LambdaRank RankScore Trendi\n")
    lines.append(f"**Tespit:** {rs_trend_str}\n")
    lines.append(f"- Dönem ilk yarısı ort. RS: **{rs_first:+.3f}**")
    lines.append(f"- Dönem son yarısı ort. RS: **{rs_last:+.3f}**")
    lines.append(f"- Fark: **{rs_trend:+.3f}**\n")

    if rs_trend < -0.1:
        lines.append("**Yorum:** Dönem ilerledikçe sistemin ürettiği adayların RankScore'u ")
        lines.append("düşmüştür. Bu, piyasadaki kötüleşmenin sisteme de yansıdığının göstergesidir: ")
        lines.append("giderek daha az sayıda hisse sistemin 'ralli öncesi formasyonu' kriterlerini ")
        lines.append("karşılamaktadır.\n")
    elif rs_trend > 0.1:
        lines.append("**Yorum:** Dönem ilerledikçe RankScore'lar artmıştır. Sistem giderek ")
        lines.append("daha kaliteli adaylar tespit etmektedir — bu bir toparlanma sinyali olabilir.\n")
    else:
        lines.append("**Yorum:** RankScore trendi yatay seyretmiştir. Tutarlı ama sınırlı bir ")
        lines.append("sinyal kalitesi gözlemlenmiştir.\n")

    lines.append("\n## 4. En İstikrarlı Adaylar (İnsan Kontrolü Öncelikli)\n")
    if recurring:
        lines.append("21 günün ≥10'unda radara giren hisseler sistem tarafından tutarlı biçimde ")
        lines.append("'ralli öncesi formasyon' taşıyor olarak değerlendirilmiştir:\n")
        lines.append("| Ticker | Gün | Ort. RS |")
        lines.append("|---|---|---|")
        for tk, cnt, avg_rs in recurring[:10]:
            lines.append(f"| **{tk}** | {cnt}/21 | {avg_rs:+.3f} |")
        lines.append("")
        lines.append("**Önerilen kontrol adımları:**")
        lines.append("1. Teknik grafiği incele — gerçekten sıkışma/birikim var mı?")
        lines.append("2. Temel analiz: son çeyrek açıklamaları, borç/özkaynak trendi")
        lines.append("3. Haber akışı: SPK bildirimleri, yönetim değişikliği var mı?")
        lines.append("4. Likidite kontrolü: günlük ortalama hacim yeterli mi?\n")
    else:
        lines.append("21 günün ≥10'unda radara giren istikrarlı bir aday tespit edilmemiştir. ")
        lines.append("Bu, piyasanın fırsatçı ve spekülatif bir karakterde olduğuna işaret etmektedir.\n")

    lines.append("\n## 5. Sektörel Konsantrasyon Analizi\n")
    lines.append("Dönem boyunca sistemin en çok aday ürettiği sektörler:\n")
    lines.append("| Sektör | Toplam Giriş |")
    lines.append("|---|---|")
    for sec, cnt in top5_sectors:
        share = cnt / total_entries * 100
        lines.append(f"| {sec} | {cnt} (%{share:.0f}) |")
    lines.append("")
    lines.append("**Yorum:** Belirli bir sektörde aşırı konsantrasyon (toplam girişlerin >%30'u) ")
    lines.append("sistematik bir sektör faktörünün (kur, faiz, regülasyon) etkisi altında olunduğuna ")
    lines.append("işaret edebilir. Bu durumda pozisyonları farklı sektörlere dağıtmak önem kazanır.\n")

    lines.append("\n## 6. ⚠️ İnsan Kontrolü Aşamasında Dikkat Edilmesi Gerekenler\n")

    checks = []
    if period_ret <= -8:
        checks.append("🔴 **Endeks çok sert düştü** — stop-loss seviyeleri gözden geçirilmeli; "
                      "pozisyon büyüklükleri normalin %50-60'ına düşürülmeli")
    if max_drop <= -4:
        checks.append(f"🔴 **Dönemde en az bir günlük -{abs(max_drop):.1f}% düşüş yaşandı** — "
                      "bu tür günlerde açık pozisyonların nasıl davrandığı test edilmeli")
    if dominant_wr < 0.45:
        checks.append("🟡 **Sistem ağırlıklı düşük win-rate kümelerinden sinyal üretti** — "
                      "her bir sinyale daha küçük pozisyon açılmalı")
    checks.append("🟡 **Sektörel risk** — en çok aday üreten sektörde tek bir portföyde "
                  ">%30 konsantrasyon olmaması önerilir")
    checks.append("🟡 **Temel kataliz** — sistemin teknik sinyallerine ek olarak beklenen bir "
                  "şirket haberi (kâr açıklaması, yatırım kararı) var mı kontrol edilmeli")
    checks.append("🟢 **LambdaRank sıralaması** — RankScore yüksek olsun, ancak WinRate'i "
                  "düşük kümelerden gelen sinyaller daha temkinli değerlendirilmeli")
    checks.append("🟢 **İkiz eşleşmesi** — run_full_analysis.py ile tam ikiz analizi yaparak "
                  "tarihsel senaryo karşılaştırması yapılması önerilir")

    for check in checks:
        lines.append(f"- {check}")

    lines.append("\n---\n")
    lines.append("## 7. Özet Skorcard\n")
    lines.append("| Boyut | Değerlendirme |")
    lines.append("|---|---|")
    lines.append(f"| Piyasa Rejimi | {rejim_emoji} {rejim} |")
    lines.append(f"| Sistem Karakteri | {char_emoji} {char} |")
    lines.append(f"| RS Trendi | {rs_trend_str} |")
    lines.append(f"| İstikrarlı Aday | {'VAR ✅' if recurring else 'YOK ⚠️'} |")
    lines.append(f"| Genel Risk Seviyesi | {'YÜKSEK 🔴' if period_ret <= -8 else 'ORTA 🟡' if period_ret <= -3 else 'NORMAL 🟢'} |")

    lines.append("\n---\n*Arkhimedes kuantitatif analiz çıktısı — Nihai karar insana aittir.*\n")

    out_path = out_dir / "YapayZeka_Yorumlari.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"   ✅ {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# ANA AKIŞ
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n📁 Çıktı dizini: {OUTPUT_DIR.resolve()}\n")

    all_days_data = []
    for i, day_date in enumerate(last21_dates, 1):
        date_str = str(day_date)[:10]
        print(f"  [{i:>2}/21] {date_str} taranıyor...", end="", flush=True)
        result = _screen_day(day_date)
        if result is None:
            print(" ⚠️ veri yok")
            all_days_data.append(None)
            continue

        write_daily_md(result, OUTPUT_DIR)
        write_daily_xlsx(result, OUTPUT_DIR)
        rs_mean = result["top"]["RankScore"].mean()
        xu = result["xu100"]
        xu_s = f"XU100={xu['close']:,.0f}({xu['daily_ret']:+.2f}%)" if xu["daily_ret"] else ""
        print(f" ✅ {len(result['top'])} aday  RS_ort={rs_mean:+.2f}  {xu_s}")
        all_days_data.append(result)

    print("\n📊 Konsolide özet yazılıyor...")
    write_consolidated_md(all_days_data, OUTPUT_DIR)

    print("🤖 Yapay zeka yorumu yazılıyor...")
    write_ai_commentary(all_days_data, OUTPUT_DIR)

    valid = [d for d in all_days_data if d is not None]
    print(f"\n{'='*70}")
    print(f"  ✅ TAMAMLANDI")
    print(f"{'='*70}")
    print(f"  📁 {OUTPUT_DIR.resolve()}/")
    print(f"     ├── Analiz_YYYY-MM-DD.md   × {len(valid)}")
    print(f"     ├── Analiz_YYYY-MM-DD.xlsx × {len(valid)}")
    print(f"     ├── Son_1_Ay_Konsolide.md")
    print(f"     └── YapayZeka_Yorumlari.md")
