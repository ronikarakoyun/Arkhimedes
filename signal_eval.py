"""
signal_eval.py — Sinyal Kalitesi Değerlendirmesi
=================================================
Sistemi bir HİSSE-BULUCU olarak değerlendirir — portföy P&L değil.
Walk-forward, look-ahead'siz. Sistemin tek işi: ralli yapacak hisseleri
sıralı liste olarak önermek. Bu harness onu şu metriklerle ölçer:

  * Precision@K   : her ay top-K önerinin kaçı ralli yaptı
  * Recall@K      : o ayki rallilerin kaçı top-K'da yakalandı
  * P/R eğrisi    : P_rally eşiği taradıkça precision/recall
  * Lift          : precision ÷ taban oran
  * Kalibrasyon   : P_rally %X diyen sinyaller gerçekte ~%X mi tuttu

Deterministik (portföy yolu yok) → tekrarlanabilir, optimize edilebilir hedef.
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import polars as pl

from backtest_engine import (train_at_date, precompute_signals_for_epoch,
                             RETRAIN_DATES, BACKTEST_END, FEAT_PATH)
from common import compute_forward_relative_gain
from model_core import RALLY_WINDOW_DAYS, RALLY_GAIN_THRESHOLD
from macro_engine import load_macro_features
from fundamental_engine import load_fundamentals
from config import FINAL_P_THRESHOLD


def build_signal_set():
    """Walk-forward tüm epoch'ların sinyallerini toplar, ralli etiketini ekler."""
    df_feat = pl.read_parquet(FEAT_PATH).with_columns(pl.col("Date").cast(pl.Datetime("ms")))
    df_outcomes = compute_forward_relative_gain(df_feat, RALLY_WINDOW_DAYS)
    df_feat_pd = df_outcomes.to_pandas()
    df_feat_pd['Date'] = pd.to_datetime(df_feat_pd['Date'])

    macro_df = load_macro_features()
    macro_df['Date'] = pd.to_datetime(macro_df['Date'])
    fund_df = load_fundamentals()

    bounds = list(RETRAIN_DATES) + [BACKTEST_END + pd.Timedelta(days=1)]
    all_sig, prev_clusters = [], {}
    for i, rd in enumerate(RETRAIN_DATES):
        bundle = train_at_date(df_feat_pd, rd, verbose=False,
                               macro_df=macro_df, fund_df=fund_df)
        if bundle is None:
            continue
        sig, prev_clusters = precompute_signals_for_epoch(
            bundle, df_feat_pd, rd, bounds[i + 1], prev_clusters, epoch_idx=i,
            macro_df=macro_df, fund_df=fund_df)
        if len(sig) > 0:
            sig['Epoch'] = i + 1
            all_sig.append(sig)
        print(f"   epoch {i+1}: {len(sig):,} sinyal")

    S = pd.concat(all_sig, ignore_index=True)
    S['Date'] = pd.to_datetime(S['Date'])

    # Ralli etiketi
    lab = df_feat_pd[['Ticker', 'Date', 'future_max_gain']].drop_duplicates(['Ticker', 'Date'])
    S = S.merge(lab, on=['Ticker', 'Date'], how='left')
    S['rallied'] = (S['future_max_gain'] >= RALLY_GAIN_THRESHOLD).astype(float)

    # İleri penceresi eksik (son RALLY_WINDOW_DAYS) sinyalleri at — etiket güvenilmez
    safe_cutoff = df_feat_pd['Date'].max() - pd.Timedelta(days=RALLY_WINDOW_DAYS)
    n_before = len(S)
    S = S[(S['Date'] <= safe_cutoff) & S['future_max_gain'].notna()].copy()
    print(f"   {n_before - len(S):,} sinyal elendi (eksik ileri pencere / etiketsiz)")
    return S


def report(S):
    base = S['rallied'].mean()
    print("\n" + "=" * 70)
    print("📊 SİNYAL KALİTESİ DEĞERLENDİRMESİ")
    print("=" * 70)
    print(f"Değerlendirilen sinyal : {len(S):,}")
    print(f"Taban ralli oranı      : {base:.3f}  (rastgele seçim bu)")

    SP = S[S['P_rally'].notna()].copy()
    print(f"P_rally'li sinyal      : {len(SP):,}  ({len(SP)/len(S)*100:.0f}%)"
          f"  — kalanlar filtre-DROP epoch'larından")

    # ── P_rally (RankScore) eşik sweep — LambdaRank skoru olasılık değil;
    # ── bu bölüm yoruma alındı (binary threshold anlamlı değil ranking için)
    # tot_rally = SP['rallied'].sum()
    # for thr in [0.0, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]: ...

    # ── Precision@K / Recall@K — aylık ──
    S = S.copy()
    S['ym'] = S['Date'].dt.to_period('M')
    for rankcol in ['Score', 'P_rally']:
        print(f"\n── Precision@K / Recall@K — aylık, '{rankcol}' sıralamasıyla ──")
        for K in [5, 10, 20]:
            precs, recs = [], []
            for _, g in S.groupby('ym'):
                g = g.dropna(subset=[rankcol])
                if len(g) == 0:
                    continue
                topk = g.nlargest(K, rankcol)
                precs.append(topk['rallied'].mean())
                tot = g['rallied'].sum()
                if tot > 0:
                    recs.append(topk['rallied'].sum() / tot)
            print(f"   K={K:>2}: precision@K={np.mean(precs):.3f}  "
                  f"recall@K={np.mean(recs):.3f}  lift={np.mean(precs)/base:.2f}x  "
                  f"({len(precs)} ay)")

    # ── Kalibrasyon — LambdaRank skoru olasılık değil; bu bölüm devre dışı ──
    # ── Bunun yerine NDCG@K metriği kullanılıyor (aşağıda) ──
    # SP['bin'] = pd.qcut(SP['P_rally'], 10, duplicates='drop') ...

    # ── NDCG@K — RankScore'un kalitesini ölç (ranking'in asıl hedefi) ──
    from sklearn.metrics import ndcg_score as sk_ndcg
    print(f"\n── NDCG@K — aylık, 'P_rally' (RankScore) sıralamasıyla ──")
    for K in [5, 10, 20]:
        ndcg_vals = []
        for _, g in S.groupby('ym'):
            g = g.dropna(subset=['P_rally', 'future_max_gain'])
            if len(g) < 2:
                continue
            # Relevance: future_max_gain'e göre normalised rank (0-1)
            rel = g['future_max_gain'].rank(pct=True).values.reshape(1, -1)
            sc  = g['P_rally'].values.reshape(1, -1)
            try:
                ndcg_vals.append(sk_ndcg(rel, sc, k=min(K, len(g))))
            except Exception:
                pass
        if ndcg_vals:
            print(f"   K={K:>2}: NDCG@K={np.mean(ndcg_vals):.4f}  ({len(ndcg_vals)} ay)")

    # ── Epoch bazlı stabilite ──
    print(f"\n── Epoch bazlı (sinyal kalitesi zaman içinde stabil mi) ──")
    for ep, g in S.groupby('Epoch'):
        gp = g.dropna(subset=['P_rally'])
        pr = gp['rallied'].mean() if len(gp) else float('nan')
        print(f"   epoch {ep}: {len(g):>5,} sinyal  taban={g['rallied'].mean():.3f}"
              f"  P_rally'li-precision={pr:.3f}")


if __name__ == "__main__":
    print("🔄 Walk-forward sinyal seti üretiliyor...")
    S = build_signal_set()
    report(S)
