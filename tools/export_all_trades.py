import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""
Tüm 10 yıllık işlemleri tek dosyada toplar:
- MD: kronolojik, okunabilir
- Excel: filtrelenebilir + yıllık özet sheet'leri
"""
import pandas as pd
import numpy as np
import os
import joblib

from config import REPORT_BACKTEST_DIR as REPORT_DIR, ARTIFACTS_PATH
OUT_DIR = REPORT_DIR

# Yükle
ts_trades   = pd.read_excel(os.path.join(REPORT_DIR, "Backtest_Sonuc.xlsx"), sheet_name='Trades')
peak_trades = pd.read_excel(os.path.join(REPORT_DIR, "PeakExit/PeakExit_Trades.xlsx"))

# Tarihleri normalize et
for df in [ts_trades, peak_trades]:
    df['EntryDate'] = pd.to_datetime(df['EntryDate'])
    df['ExitDate']  = pd.to_datetime(df['ExitDate'])
    df['EntryYear'] = df['EntryDate'].dt.year
    df['EntryMonth'] = df['EntryDate'].dt.month

# Sırala
ts_trades = ts_trades.sort_values('EntryDate').reset_index(drop=True)
ts_trades.insert(0, 'No', range(1, len(ts_trades)+1))

peak_trades = peak_trades.sort_values('EntryDate').reset_index(drop=True)
peak_trades.insert(0, 'No', range(1, len(peak_trades)+1))

# Küme isimleri (mevcut artifacts'tan)
try:
    artifacts = joblib.load(ARTIFACTS_PATH)
    cluster_names = {c: info['name'] for c, info in artifacts['cluster_info'].items()}
except (FileNotFoundError, KeyError) as e:
    print(f"⚠️  Cluster names yüklenemedi ({type(e).__name__}: {e}); boş dict kullanılacak.")
    cluster_names = {}


# ─── Excel: zengin format ────────────────────────────────────────────────────
xlsx_path = os.path.join(OUT_DIR, "Tum_Islemler.xlsx")

with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
    # Sheet 1: Trailing %20 strateji (gerçek)
    ts_display = ts_trades.copy()
    ts_display['GainPct_str']     = ts_display['GainPct'].apply(lambda x: f"{x*100:+.1f}%")
    ts_display['PeakGainPct_str'] = ts_display['PeakGainPct'].apply(lambda x: f"{x*100:+.1f}%")
    ts_display['WinRate_str']     = ts_display['WinRate'].apply(lambda x: f"%{x*100:.0f}")
    ts_display['LeftOnTable_pct'] = (ts_display['PeakClose']/ts_display['EntryPrice'] - (1+ts_display['GainPct'])) / (ts_display['PeakClose']/ts_display['EntryPrice']) * 100
    ts_display['ClusterName']     = ts_display['Cluster'].map(cluster_names).fillna(ts_display['Cluster'].astype(str))

    cols_main = ['No', 'Ticker', 'EntryDate', 'ExitDate', 'HoldingDays',
                 'EntryPrice', 'ExitPrice', 'PeakClose',
                 'GainPct_str', 'PeakGainPct_str', 'LeftOnTable_pct',
                 'Cluster', 'ClusterName', 'WinRate_str', 'Score',
                 'GainPct', 'PeakGainPct']
    ts_display[cols_main].to_excel(writer, sheet_name='Trailing_20pct', index=False)

    # Sheet 2: Peak Oracle
    pk_display = peak_trades.copy()
    pk_display['GainPct_str']     = pk_display['GainPct'].apply(lambda x: f"{x*100:+.1f}%")
    pk_display['PeakGainPct_str'] = pk_display['PeakGainPct'].apply(lambda x: f"{x*100:+.1f}%")
    pk_display['WinRate_str']     = pk_display['WinRate'].apply(lambda x: f"%{x*100:.0f}")
    pk_display['ClusterName']     = pk_display['Cluster'].map(cluster_names).fillna(pk_display['Cluster'].astype(str))
    cols_peak = ['No', 'Ticker', 'EntryDate', 'ExitDate', 'HoldingDays',
                 'EntryPrice', 'ExitPrice', 'PeakClose',
                 'GainPct_str', 'PeakGainPct_str',
                 'Cluster', 'ClusterName', 'WinRate_str', 'Score',
                 'GainPct', 'PeakGainPct']
    pk_display[cols_peak].to_excel(writer, sheet_name='Peak_Oracle', index=False)

    # Sheet 3: Yıllık özet (trailing)
    yearly = ts_trades.groupby('EntryYear').agg(
        Trade_Count   = ('GainPct', 'size'),
        Wins          = ('GainPct', lambda x: (x > 0).sum()),
        Losses        = ('GainPct', lambda x: (x <= 0).sum()),
        Win_Rate      = ('GainPct', lambda x: (x > 0).mean()),
        Avg_Gain      = ('GainPct', 'mean'),
        Max_Gain      = ('GainPct', 'max'),
        Min_Gain      = ('GainPct', 'min'),
        Avg_Hold_Days = ('HoldingDays', 'mean'),
        Avg_Peak_Gain = ('PeakGainPct', 'mean'),
    ).round(3)
    yearly.to_excel(writer, sheet_name='Yillik_Ozet')

    # Sheet 4: Küme bazlı özet
    cluster_summary = ts_trades.groupby('Cluster').agg(
        Trade_Count   = ('GainPct', 'size'),
        Wins          = ('GainPct', lambda x: (x > 0).sum()),
        Win_Rate      = ('GainPct', lambda x: (x > 0).mean()),
        Avg_Gain      = ('GainPct', 'mean'),
        Median_Gain   = ('GainPct', 'median'),
        Avg_Peak_Gain = ('PeakGainPct', 'mean'),
        Avg_Hold_Days = ('HoldingDays', 'mean'),
    ).round(3)
    cluster_summary.insert(0, 'ClusterName', cluster_summary.index.map(cluster_names).fillna('?'))
    cluster_summary.to_excel(writer, sheet_name='Kume_Ozet')

    # Sheet 5: Top 20 & Bottom 20
    top20 = ts_trades.nlargest(20, 'GainPct')[['No','Ticker','EntryDate','ExitDate','HoldingDays','GainPct','PeakGainPct','Cluster']].copy()
    top20['GainPct'] = top20['GainPct'].apply(lambda x: f"{x*100:+.1f}%")
    top20['PeakGainPct'] = top20['PeakGainPct'].apply(lambda x: f"{x*100:+.1f}%")
    bottom20 = ts_trades.nsmallest(20, 'GainPct')[['No','Ticker','EntryDate','ExitDate','HoldingDays','GainPct','PeakGainPct','Cluster']].copy()
    bottom20['GainPct'] = bottom20['GainPct'].apply(lambda x: f"{x*100:+.1f}%")
    bottom20['PeakGainPct'] = bottom20['PeakGainPct'].apply(lambda x: f"{x*100:+.1f}%")
    top20.to_excel(writer, sheet_name='Top_20', index=False)
    bottom20.to_excel(writer, sheet_name='Bottom_20', index=False)

print(f"💾 Excel: {xlsx_path}")


# ─── Markdown: kronolojik tam liste ─────────────────────────────────────────
md_path = os.path.join(OUT_DIR, "Tum_Islemler.md")
lines = []
lines.append("# BIST 10-Yıllık Backtest — Tüm İşlemler\n")
lines.append(f"*Strateji: Trailing %20 trailing stop, 5 paralel pozisyon, %20 ağırlık her*\n")
lines.append(f"*Periyot: 2018-05-23 → 2026-05-20  |  Toplam {len(ts_trades)} trade*\n")

# Özet metrikler
wins  = ts_trades[ts_trades['GainPct'] > 0]
losses = ts_trades[ts_trades['GainPct'] <= 0]
lines.append("\n## 📊 Genel Özet\n")
lines.append("| | |")
lines.append("|---|---|")
lines.append(f"| Toplam trade | **{len(ts_trades)}** |")
lines.append(f"| Karlı trade | {len(wins)} (%{len(wins)/len(ts_trades)*100:.1f}) |")
lines.append(f"| Zararlı trade | {len(losses)} (%{len(losses)/len(ts_trades)*100:.1f}) |")
lines.append(f"| Ortalama kazanç (karlılar) | **+%{wins['GainPct'].mean()*100:.1f}** |")
lines.append(f"| Ortalama zarar (zararlılar) | **%{losses['GainPct'].mean()*100:.1f}** |")
lines.append(f"| En iyi trade | {ts_trades.loc[ts_trades['GainPct'].idxmax(), 'Ticker']} +%{ts_trades['GainPct'].max()*100:.1f} |")
lines.append(f"| En kötü trade | {ts_trades.loc[ts_trades['GainPct'].idxmin(), 'Ticker']} %{ts_trades['GainPct'].min()*100:.1f} |")
lines.append(f"| Ortalama tutma | {ts_trades['HoldingDays'].mean():.0f} gün |")
lines.append(f"| En uzun tutma | {ts_trades['HoldingDays'].max()} gün ({ts_trades.loc[ts_trades['HoldingDays'].idxmax(), 'Ticker']}) |")
lines.append(f"| En kısa tutma | {ts_trades['HoldingDays'].min()} gün ({ts_trades.loc[ts_trades['HoldingDays'].idxmin(), 'Ticker']}) |")

# Yıllık dağılım
lines.append("\n## 📅 Yıllık Trade Dağılımı\n")
lines.append("| Yıl | Trade | Karlı | Win% | Avg Gain | Max Gain | Min Gain |")
lines.append("|---|---|---|---|---|---|---|")
for year, g in ts_trades.groupby('EntryYear'):
    wins_y = (g['GainPct'] > 0).sum()
    lines.append(
        f"| {year} | {len(g)} | {wins_y} | %{wins_y/len(g)*100:.1f} | "
        f"%{g['GainPct'].mean()*100:+.1f} | %{g['GainPct'].max()*100:+.1f} | %{g['GainPct'].min()*100:+.1f} |"
    )

# Küme bazlı
lines.append("\n## 🎯 Küme Bazlı Dağılım\n")
lines.append("| Küme | İsim | Trade | Win% | Avg Gain | Avg Peak | Avg Hold |")
lines.append("|---|---|---|---|---|---|---|")
for c, g in ts_trades.groupby('Cluster'):
    wins_c = (g['GainPct'] > 0).sum()
    name = cluster_names.get(c, '?')
    lines.append(
        f"| K{c} | {name} | {len(g)} | %{wins_c/len(g)*100:.1f} | "
        f"%{g['GainPct'].mean()*100:+.1f} | %{g['PeakGainPct'].mean()*100:+.1f} | "
        f"{g['HoldingDays'].mean():.0f}g |"
    )

# Kronolojik tüm trade'ler
lines.append("\n---\n\n## 📜 Kronolojik Tüm İşlemler\n")
lines.append("Sütunlar:")
lines.append("- **Giriş** / **Çıkış**: tarih")
lines.append("- **Tut**: tutma süresi (gün)")
lines.append("- **Fiyat**: giriş → çıkış")
lines.append("- **Getiri**: gerçek getiri (trailing stop ile)")
lines.append("- **Tepe**: peak'teki teorik getiri (peak oracle olsaydı)")
lines.append("- **Bırakılan**: trailing stop'un masada bıraktığı para (%)")
lines.append("- **K**: küme no\n")

cur_year = None
for _, t in ts_trades.iterrows():
    year = t['EntryDate'].year
    if year != cur_year:
        lines.append(f"\n### {year}\n")
        lines.append(f"| # | Hisse | Giriş | Çıkış | Tut | Fiyat | Getiri | Tepe | Bırakılan | K |")
        lines.append(f"|---|---|---|---|---|---|---|---|---|---|")
        cur_year = year

    peak_ratio = t['PeakClose'] / t['EntryPrice']
    actual_ratio = 1 + t['GainPct']
    left = (peak_ratio - actual_ratio) / peak_ratio * 100 if peak_ratio > 0 else 0

    gain_emoji = "🟢" if t['GainPct'] > 0.5 else "🟢" if t['GainPct'] > 0 else "🔴"

    lines.append(
        f"| {t['No']} | **{t['Ticker']}** | {t['EntryDate'].date()} | "
        f"{t['ExitDate'].date()} | {t['HoldingDays']}g | "
        f"{t['EntryPrice']:.2f} → {t['ExitPrice']:.2f} | "
        f"{gain_emoji} **%{t['GainPct']*100:+.1f}** | %{t['PeakGainPct']*100:+.1f} | "
        f"%{left:.1f} | {t['Cluster']} |"
    )

lines.append("\n---\n")
lines.append(f"\n*Toplam {len(ts_trades)} işlem. 'Bırakılan' sütunu trailing %20 stratejisinin masada bıraktığı para — peak oracle ile %22.4 ortalama fark.*")

with open(md_path, 'w', encoding='utf-8') as f:
    f.write("\n".join(lines))

print(f"💾 Markdown: {md_path}")

# CSV de versinler — temiz, sade
csv_path = os.path.join(OUT_DIR, "Tum_Islemler.csv")
ts_trades.to_csv(csv_path, index=False, encoding='utf-8')
print(f"💾 CSV: {csv_path}")

print(f"\n✅ {len(ts_trades)} trade üç formatta hazır.")