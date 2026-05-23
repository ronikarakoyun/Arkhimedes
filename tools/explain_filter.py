import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""
CombinedFilter SHAP Açıklama Scripti
=====================================
Belirtilen tarihteki filtre-geçen hisseler için hangi feature'ların
P_rally'yi yukarı/aşağı çektiğini gösterir.
"""
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import polars as pl
import pickle
import shap

sys.path.insert(0, "/Users/unalronikarakoyun/Desktop/Veri")
from macro_engine import load_macro_features
from fundamental_engine import load_fundamentals, snapshot_at
from combined_filter import CombinedFilter, TECH_FEATURES, build_training_set
from config import (FEATURES_PATH as FEAT_PATH, SUCCESS_POOL_PATH, FAILURE_POOL_PATH,
                    FINAL_P_THRESHOLD)

TARGET_DATE = pd.Timestamp("2026-05-21")
TICKERS = ["NTGAZ", "AVTUR", "OBASE", "ATAGY", "INGRM", "FORTE", "TTKOM"]

def build_feature_row(ticker, date, macro_df, fund_df, feat_pd):
    from macro_engine import load_macro_features
    # Macro snapshot
    macro_day = macro_df[macro_df['Date'] <= date].sort_values('Date').iloc[-1]
    m_cols = [c for c in macro_df.columns if c.startswith("m_")]
    macro_vals = macro_day[m_cols].to_dict()

    # Fundamental snapshot
    q = pd.DataFrame([{"Ticker": ticker, "Date": date}])
    q["Date"] = pd.to_datetime(q["Date"])
    snap = snapshot_at(fund_df, q)
    f_cols = [c for c in snap.columns if (c.endswith("_z") or c.startswith("f_")) and c not in TECH_FEATURES]
    fund_vals = {}
    if len(snap) > 0:
        fund_vals = snap.iloc[0][f_cols].to_dict()

    # Technical snapshot
    tech_row = feat_pd[(feat_pd['Ticker'] == ticker) & (feat_pd['Date'] <= date)]
    tech_vals = {}
    if len(tech_row) > 0:
        tech_row = tech_row.sort_values('Date').iloc[-1]
        for c in TECH_FEATURES:
            if c in tech_row.index:
                tech_vals[c] = tech_row[c]

    row = {**macro_vals, **fund_vals, **tech_vals}
    return row


def main():
    print("📥 Veriler yükleniyor...")
    macro_df = load_macro_features()
    macro_df['Date'] = pd.to_datetime(macro_df['Date'])
    fund_df = load_fundamentals()

    feat_pd = pl.read_parquet(FEAT_PATH).to_pandas()
    feat_pd['Date'] = pd.to_datetime(feat_pd['Date'])

    print("🤖 CombinedFilter eğitiliyor (tüm geçmiş)...")
    X_train, y_train, dates_train = build_training_set(
        success_pool_path=SUCCESS_POOL_PATH,
        failure_pool_path=FAILURE_POOL_PATH,
        features_path=FEAT_PATH,
    )
    cf = CombinedFilter()
    cf.fit(X_train, y_train, dates_train)
    print(f"   cv_auc={cf.cv_auc:.3f}  n={cf.n_train}")

    print(f"\n📊 {TARGET_DATE.date()} için feature vektörleri hazırlanıyor...")
    rows = []
    for ticker in TICKERS:
        row = build_feature_row(ticker, TARGET_DATE, macro_df, fund_df, feat_pd)
        row['Ticker'] = ticker
        rows.append(row)

    df_explain = pd.DataFrame(rows).set_index('Ticker')
    df_explain = df_explain.reindex(columns=cf.feature_cols)
    df_explain = df_explain.fillna(value=cf.feature_medians_).fillna(0.0)
    df_explain = df_explain.replace([np.inf, -np.inf], 0.0)

    p_vals = cf.predict_proba(df_explain)
    for t, p in zip(TICKERS, p_vals):
        print(f"   {t}: P_rally={p:.3f} {'✅' if p >= FINAL_P_THRESHOLD else '⚠️'}")

    print("\n🔬 SHAP değerleri hesaplanıyor...")
    # TreeExplainer for LightGBM
    explainer = shap.TreeExplainer(cf.model)
    shap_values = explainer.shap_values(df_explain)
    # For binary: shap_values shape depends on version
    if isinstance(shap_values, list):
        sv = shap_values[1]  # positive class
    else:
        sv = shap_values

    print("\n" + "=" * 70)
    print(f"📈 SHAP AÇIKLAMASI — {TARGET_DATE.date()} — Filtre Geçen Hisseler")
    print("=" * 70)

    # Global medians for comparison
    medians = pd.Series(cf.feature_medians_)

    for i, ticker in enumerate(TICKERS):
        p = p_vals[i]
        shap_row = sv[i]
        feat_row = df_explain.iloc[i]

        # Top 5 positive (yukarı çeken) and top 3 negative (aşağı çeken)
        shap_series = pd.Series(shap_row, index=cf.feature_cols)
        top_pos = shap_series.nlargest(5)
        top_neg = shap_series.nsmallest(3)

        print(f"\n{'─'*60}")
        print(f"  {ticker}   P_rally={p:.3f}  {'✅ GEÇTİ' if p >= FINAL_P_THRESHOLD else '⚠️ VETO'}")
        print(f"{'─'*60}")

        print("  ▲ P_rally'yi YUKARI çeken feature'lar:")
        for feat, sv_val in top_pos.items():
            val = feat_row[feat]
            med = medians.get(feat, np.nan)
            vs = f"{val:+.3f}  (medyan: {med:.3f})" if not pd.isna(med) else f"{val:+.3f}"
            print(f"     {feat:28s}  SHAP={sv_val:+.4f}   değer={vs}")

        print("  ▼ P_rally'yi AŞAĞI çeken feature'lar:")
        for feat, sv_val in top_neg.items():
            val = feat_row[feat]
            med = medians.get(feat, np.nan)
            vs = f"{val:+.3f}  (medyan: {med:.3f})" if not pd.isna(med) else f"{val:+.3f}"
            print(f"     {feat:28s}  SHAP={sv_val:+.4f}   değer={vs}")


if __name__ == "__main__":
    main()
