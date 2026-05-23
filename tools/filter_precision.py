import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""
CombinedFilter precision/recall — walk-forward CV ile.
P_rally >= threshold → 'geçer'. Geçenlerin kaçı gerçek başarı (y=1)?
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, precision_score, recall_score

from combined_filter import build_training_set, LGB_PARAMS

X, y, dates = build_training_set()
y = np.asarray(y)
X = X.fillna(X.median(numeric_only=True)).replace([np.inf, -np.inf], 0.0)
dates = pd.to_datetime(dates)
years = dates.dt.year.values

print(f"Eğitim seti: {len(X)} sinyal, {X.shape[1]} feature")
print(f"Taban başarı oranı (tüm havuz): {y.mean():.3f}  ({y.sum()} başarı / {len(y)})")
print()

# Walk-forward: son 5 yıl train, sonraki yıl test
all_y, all_p = [], []
for test_year in sorted(np.unique(years)):
    tr = (years >= test_year - 5) & (years < test_year)
    te = years == test_year
    if tr.sum() < 500 or te.sum() < 100:
        continue
    if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
        continue
    m = lgb.LGBMClassifier(**LGB_PARAMS)
    m.fit(X[tr], y[tr])
    p = m.predict_proba(X[te])[:, 1]
    all_y.append(y[te]); all_p.append(p)

Y = np.concatenate(all_y)
P = np.concatenate(all_p)
print(f"Walk-forward test örneği: {len(Y)}")
print(f"Genel AUC: {roc_auc_score(Y, P):.3f}")
print()

print(f"{'Eşik':>6} {'Geçen%':>8} {'Precision':>10} {'Recall':>8} {'Lift':>7}")
base = Y.mean()
for thr in [0.30, 0.35, 0.40, 0.45, 0.50]:
    passed = P >= thr
    if passed.sum() == 0:
        continue
    prec = Y[passed].mean()              # geçenlerin başarı oranı
    rec = (Y[passed].sum()) / Y.sum()    # tüm başarıların kaçı geçti
    lift = prec / base
    print(f"{thr:>6.2f} {passed.mean()*100:>7.1f}% {prec:>10.3f} {rec:>8.3f} {lift:>6.2f}x")
