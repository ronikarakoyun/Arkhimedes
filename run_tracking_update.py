"""
run_tracking_update.py — Arkhimedes Seyir Defteri Güncelleyici
==============================================================
Günlük tarama sonucunu alır, durum makinesini çalıştırır, rapor üretir.

Çalıştırma:
  venv/bin/python3 run_tracking_update.py              # bugün
  venv/bin/python3 run_tracking_update.py --date YYYY-MM-DD  # belirli gün
"""
import argparse
import warnings
import polars as pl
import pandas as pd

warnings.filterwarnings('ignore')

from config import DB_PATH, TRACKING_DB_PATH, TRACKING_REPORT_DIR, TRACKING_TOP_N
from run_daily_analysis import run_screener_and_dashboard
from journey_tracker import update_tracking_engine, generate_report, load_tracking_db


def main(date_str=None):
    print("🗺️  Arkhimedes Seyir Defteri Güncelleniyor...")

    # ── Piyasa verisi → bugünün fiyatları ────────────────────────────────────
    df_market = pl.read_parquet(DB_PATH).with_columns(pl.col("Date").cast(pl.Date))

    if date_str:
        try:
            target_date = pd.to_datetime(date_str).date()
        except Exception as e:
            print(f"❌ Tarih formatı hatası: {e}")
            return
    else:
        target_date = df_market["Date"].max()

    target_date_str = str(target_date)
    print(f"📅 Tarih: {target_date_str}")

    # Kapanış fiyatları — tüm ticker'lar için {ticker: fiyat}
    df_today = df_market.filter(pl.col("Date") == target_date)
    prices = dict(zip(df_today["Ticker"].to_list(), df_today["Pclose"].to_list()))
    print(f"💰 {len(prices)} ticker için kapanış fiyatı yüklendi.")

    # ── Günlük tarama → sıralı aday listesi ──────────────────────────────────
    print("\n🔍 Günlük tarama çalıştırılıyor...")
    final_df = run_screener_and_dashboard(target_date_str=target_date_str, return_df=True)

    if final_df is None or final_df.empty:
        print("❌ Günlük tarama çıktısı boş — tracking güncellenmedi.")
        return

    print(f"✅ {len(final_df)} aday sıralandı. İlk {TRACKING_TOP_N}'i takip edilecek.")

    # ── State Machine güncelle ────────────────────────────────────────────────
    print(f"\n⚙️  Seyir Defteri güncelleniyor ({TRACKING_DB_PATH})...")
    stats = update_tracking_engine(
        daily_top_df=final_df,
        current_market_prices=prices,
        current_date=target_date,
        db_path=TRACKING_DB_PATH,
    )

    # ── Rapor üret ve ekrana bas ──────────────────────────────────────────────
    db = load_tracking_db(TRACKING_DB_PATH)
    report_text = generate_report(db, stats, report_dir=TRACKING_REPORT_DIR)
    print("\n" + report_text)

    print(f"\n✅ Rapor kaydedildi: {TRACKING_REPORT_DIR}/Seyir_Defteri_{target_date_str.replace('-', '')}.txt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arkhimedes Seyir Defteri Güncelleyici')
    parser.add_argument('--date', type=str, help='Tarama tarihi (YYYY-MM-DD)', default=None)
    args = parser.parse_args()
    main(date_str=args.date)
