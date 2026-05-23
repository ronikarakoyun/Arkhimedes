"""
CombinedFilter — LambdaRank (Learning to Rank)
================================================
Binary sınıflandırma KALDIRILDI. Yerine LambdaRank (LGBMRanker):

  * Aynı YearMonth içindeki hisseler 0-4 alaka puanıyla kıyaslanır.
  * "En çok kazananı listenin en tepesine koy" → NDCG@K metriği.
  * Çıktı: ham sıralama puanı (ranking score) — olasılık DEĞİL.

Kalibrasyon problemi ortadan kalkar çünkü model olasılık iddia etmiyor.
Bilgi kaybı ortadan kalkar: %49 ve %400 kazanan artık eşit değil.

Feature setleri:
  * 12 makro feature (m_* kolonları)
  * ≤14 fundamental feature (f_* + _z kolonları)
  * 18 teknik feature (mom, cv, xu100, sektörel göreli güç)

KEEP/DROP gate: ENV_FILTER_NDCG_THRESHOLD = 0.0 → her zaman KEEP.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from lightgbm import LGBMRanker

from config import ENV_FILTER_NDCG_THRESHOLD, COMBINED_CV_WINDOW_YEARS

# Teknik feature listesi — model_core'daki FEATURES_FOR_CLUSTERING'in
# bir alt-kümesi (K-Means/GMM'tekilerle aynı feature'lar ama onları kirletmez)
TECH_FEATURES = [
    "mom_120", "mom_60", "mom_30", "mom_252",
    "cv_120", "cv_60", "cv_compression",
    "rel_strength_120", "rel_strength_60", "rel_vol_120",
    "xu100_mom_60", "xu100_drawdown", "xu100_above_ma200",
    "usd_mom_30", "v_roc", "vwap_dist_avg",
    "sector_rel_60", "sector_rel_120",          # Step 2a: sektörel göreli güç
]

# LightGBM LambdaRank parametreleri
LGB_RANK_PARAMS = dict(
    objective='lambdarank',
    metric='ndcg',
    ndcg_eval_at=[5, 10, 20],
    num_leaves=31,
    learning_rate=0.05,
    n_estimators=300,
    max_depth=6,
    min_child_samples=50,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=5,
    verbosity=-1,
    random_state=42,
    num_threads=-1,
)


def _compute_relevance_labels(df: pd.DataFrame, gain_col: str,
                               group_col: str = 'YearMonth') -> np.ndarray:
    """
    Her YearMonth grubu içinde future_max_gain'i alaka seviyelerine dönüştür.
    Bins: [0%,10%) → 0 | [10%,30%) → 1 | [30%,70%) → 2 | [70%,90%) → 3 | [90%,100%] → 4
    Minimum 5 üyeli gruplarda çalışır; küçük gruplarda tüm satırlara 2 verilir (nötr).

    NOT: df.reset_index(drop=True) yapılmış olmalı (0-tabanlı integer index).
    """
    labels = np.full(len(df), 2, dtype=np.int32)
    for grp, idx in df.groupby(group_col).groups.items():
        if len(idx) < 5:
            continue
        pos = df.index.get_indexer(idx)  # 0-tabanlı pozisyonlar
        gains = df.loc[idx, gain_col].values
        # rank(pct=True) → sabit 5 sınıf, yığılmadan etkilenmez
        pct = pd.Series(gains).rank(pct=True, method='average').values
        rel = np.where(pct <= 0.10, 0,
              np.where(pct <= 0.30, 1,
              np.where(pct <= 0.70, 2,
              np.where(pct <= 0.90, 3, 4)))).astype(np.int32)
        labels[pos] = rel
    return labels


def _get_group_array(sorted_group_col: pd.Series) -> np.ndarray:
    """Ardışık grup boyutları dizisi (LightGBM group parametresi için)."""
    return sorted_group_col.value_counts(sort=False).reindex(
        sorted_group_col.unique()).fillna(0).values.astype(np.int32)


def _rolling_walk_forward_cv_ndcg(X: pd.DataFrame, relevance: np.ndarray,
                                   group_col: pd.Series, dates: pd.Series,
                                   window_years: int = 3,
                                   k: int = 10) -> 'tuple[float | None, list[float]]':
    """
    Rolling CV: window_years yıl train, 1 yıl test. NDCG@k döner.
    X, relevance, group_col, dates aynı index sırasında olmalı.
    """
    dates = pd.to_datetime(dates).reset_index(drop=True)
    years = dates.dt.year.values
    unique_years = sorted(np.unique(years))
    ndcgs = []
    for test_year in unique_years:
        if test_year - window_years < unique_years[0]:
            continue
        train_m = (years >= test_year - window_years) & (years < test_year)
        test_m = (years == test_year)
        if train_m.sum() < 200 or test_m.sum() < 50:
            continue
        rel_train = relevance[train_m]
        rel_test = relevance[test_m]
        if len(np.unique(rel_train)) < 2 or len(np.unique(rel_test)) < 2:
            continue

        gc_train = group_col.values[train_m]
        gc_test = group_col.values[test_m]

        order_tr = np.argsort(gc_train, kind='stable')
        order_te = np.argsort(gc_test, kind='stable')

        Xtr = X[train_m].iloc[order_tr].reset_index(drop=True)
        ytr = rel_train[order_tr]
        Xte = X[test_m].iloc[order_te].reset_index(drop=True)
        yte = rel_test[order_te]

        gc_tr_sorted = gc_train[order_tr]
        gc_te_sorted = gc_test[order_te]

        g_tr = _get_group_array(pd.Series(gc_tr_sorted))
        g_te = _get_group_array(pd.Series(gc_te_sorted))

        if g_tr.sum() == 0 or g_te.sum() == 0:
            continue

        try:
            m = LGBMRanker(**LGB_RANK_PARAMS)
            m.fit(Xtr, ytr, group=g_tr,
                  eval_set=[(Xte, yte)], eval_group=[g_te],
                  callbacks=[lgb.early_stopping(30, verbose=False)])
            # LightGBM C++ çekirdeğinde hesapladığı NDCG'yi doğrudan oku
            ndcg = m.best_score_.get('valid_0', {}).get('ndcg@10')
            if ndcg is not None:
                ndcgs.append(float(ndcg))
        except Exception:
            pass

    return (float(np.mean(ndcgs)) if ndcgs else None), ndcgs


class CombinedFilter:
    """
    LambdaRank filtresi: binary sınıflandırma YOK.
    Aynı YearMonth içindeki hisseleri 0-4 alaka seviyesiyle kıyaslar.
    Çıktı: ham sıralama puanı (ranking score) — olasılık DEĞİL.

    fit() çağrısı:
      X        : DataFrame, kolonlar = macro_cols + fund_cols + tech_cols
      relevance: np.ndarray[int32], 0-4 alaka seviyesi
      dates    : pd.Series of pd.Timestamp (walk-forward CV için)
      group_col: pd.Series of str (YearMonth — grup sınırları için)
    """

    def __init__(self, ndcg_threshold: float = ENV_FILTER_NDCG_THRESHOLD,
                 lgb_params: 'dict | None' = None,
                 cv_window_years: int = 3):
        self.ndcg_threshold = ndcg_threshold
        self.lgb_params = lgb_params or LGB_RANK_PARAMS
        self.cv_window_years = cv_window_years

        self.model: 'LGBMRanker | None' = None
        self.feature_cols: list = []
        self.feature_medians_: dict = {}
        self.cv_ndcg: 'float | None' = None
        self.cv_fold_ndcgs: list = []
        self.n_train: int = 0

    def fit(self, X: pd.DataFrame, relevance: np.ndarray,
            dates: pd.Series, group_col: pd.Series):
        X = X.copy().reset_index(drop=True)
        relevance = np.asarray(relevance, dtype=np.int32)
        dates = pd.to_datetime(dates).reset_index(drop=True)
        group_col = group_col.reset_index(drop=True)

        self.feature_cols = list(X.columns)
        self.feature_medians_ = X.median(numeric_only=True).to_dict()
        X = X.fillna(value=self.feature_medians_).replace([np.inf, -np.inf], 0.0)

        # Rolling CV (NDCG)
        self.cv_ndcg, self.cv_fold_ndcgs = _rolling_walk_forward_cv_ndcg(
            X, relevance, group_col, dates,
            window_years=self.cv_window_years)

        # Final model: tüm veri, YearMonth'a göre sıralı
        gc = group_col.values
        order = np.argsort(gc, kind='stable')
        X_sorted = X.iloc[order].reset_index(drop=True)
        rel_sorted = relevance[order]
        gc_sorted = gc[order]
        groups = _get_group_array(pd.Series(gc_sorted))

        self.model = LGBMRanker(**self.lgb_params)
        self.model.fit(X_sorted, rel_sorted, group=groups)
        self.n_train = len(X)
        return self

    def predict_rank_score(self, X: pd.DataFrame) -> np.ndarray:
        """Ham sıralama puanı — büyük = daha iyi sıralamalı."""
        if self.model is None:
            raise RuntimeError("fit() önce çağrılmalı")
        if isinstance(X, pd.Series):
            X = X.to_frame().T
        X = X.reindex(columns=self.feature_cols)
        X = X.fillna(value=self.feature_medians_).fillna(0.0)
        X = X.replace([np.inf, -np.inf], 0.0)
        return self.model.predict(X)

    # Geriye uyumluluk için takma ad — backtest_engine bazı yerlerde kullanıyor olabilir
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Geriye uyumluluk: predict_rank_score'a yönlendir."""
        return self.predict_rank_score(X)

    def is_useful(self) -> bool:
        return self.model is not None and (
            self.ndcg_threshold <= 0 or
            (self.cv_ndcg is not None and self.cv_ndcg >= self.ndcg_threshold))

    def summary(self) -> str:
        status = "✅ KEEP" if self.is_useful() else "❌ DROP"
        lines = [
            f"LambdaRankFilter {status}",
            f"  n_train  : {self.n_train}",
            f"  features : {len(self.feature_cols)}",
        ]
        if self.cv_ndcg is not None:
            lines.append(f"  cv_ndcg  : {self.cv_ndcg:.4f}")
        else:
            lines.append("  cv_ndcg  : N/A")
        if self.cv_fold_ndcgs:
            lines.append("  per-year : " + ", ".join(f"{v:.3f}" for v in self.cv_fold_ndcgs))
        return "\n".join(lines)


# ─── Yardımcı: build_training_set — geriye uyumluluk için korunur ────────────
def build_training_set(
    success_pool_path: str = "knowledge_success_pool.parquet",
    failure_pool_path: str = "knowledge_failure_pool.parquet",
    features_path: str = "features_db.parquet",
    tech_features: 'list[str] | None' = None,
):
    """
    [DEPRECATED] Binary eğitim seti — artık CombinedFilter tarafından kullanılmıyor.
    LambdaRank eğitimi için backtest_engine._train_combined_filter kullanılır.
    Geriye uyumluluk için bırakıldı.
    """
    import polars as pl
    from macro_engine import load_macro_features
    from fundamental_engine import load_fundamentals, snapshot_at

    tech_features = tech_features or TECH_FEATURES

    sp = pd.read_parquet(success_pool_path)[["Ticker", "Date"]].copy()
    fp = pd.read_parquet(failure_pool_path)[["Ticker", "Date"]].copy()
    sp["y"] = 1
    fp["y"] = 0
    labels = pd.concat([sp, fp], ignore_index=True)
    labels["Date"] = pd.to_datetime(labels["Date"])

    macro = load_macro_features()
    macro["Date"] = pd.to_datetime(macro["Date"])
    m_cols = [c for c in macro.columns if c.startswith("m_")]
    labs = labels.merge(macro, on="Date", how="left")

    fund = load_fundamentals()
    snap = snapshot_at(fund, labs[["Ticker", "Date"]])
    labs = labs.merge(snap.drop_duplicates(["Ticker", "Date"]),
                      on=["Ticker", "Date"], how="left")
    f_cols = [c for c in labs.columns
              if (c.endswith("_z") and c not in m_cols) or
                 (c.startswith("f_") and c not in tech_features)]

    feat = pl.read_parquet(features_path).to_pandas()
    feat["Date"] = pd.to_datetime(feat["Date"])
    tech_keep = [c for c in tech_features if c in feat.columns]
    labs = labs.merge(feat[["Ticker", "Date"] + tech_keep],
                      on=["Ticker", "Date"], how="left")

    labs = labs.dropna(subset=m_cols + tech_keep, how="any").reset_index(drop=True)
    labs = labs.sort_values("Date").reset_index(drop=True)

    feat_cols = m_cols + f_cols + tech_keep
    X = labs[feat_cols].copy()
    y = labs["y"].values
    dates = labs["Date"]
    return X, y, dates


if __name__ == "__main__":
    print("CombinedFilter (LambdaRank) import testi:")
    print(f"  LGB_RANK_PARAMS.objective = {LGB_RANK_PARAMS['objective']}")
    print(f"  TECH_FEATURES = {len(TECH_FEATURES)} feature")
    print(f"  ENV_FILTER_NDCG_THRESHOLD = {ENV_FILTER_NDCG_THRESHOLD}")
    cf = CombinedFilter()
    print(f"  CombinedFilter() init OK — ndcg_threshold={cf.ndcg_threshold}")
