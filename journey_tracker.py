"""
journey_tracker.py — Arkhimedes Yaşayan Seyir Defteri
=======================================================
Hisse tespitlerini (HOOK) olay güdümlü durum makinesiyle izler.

Durumlar:
  ACTIVE         — bugünkü top_n listesinde mevcut
  GHOST          — listeden düştü ama henüz arşivlenmedi
  ARCHIVED_STALE — 45 gün GHOST → historical_archives'e taşındı

JSON şeması (tracking_db.json):
  {
    "last_updated": "YYYY-MM-DD",
    "active_journeys":   { "KUYAS_1": {...}, ... },
    "historical_archives": { "KUYAS_0": {...}, ... }
  }

Hiçbir kayıt SİLİNMEZ — sadece bloklar arasında taşınır.
"""
import json
import os
import numpy as np
import pandas as pd
from datetime import datetime

from config import (
    TRACKING_DB_PATH, TRACKING_REPORT_DIR,
    TRACKING_TOP_N, TRACKING_GHOST_TIMEOUT,
)


# ─── DB I/O ─────────────────────────────────────────────────────────────────

def load_tracking_db(db_path: str = TRACKING_DB_PATH) -> dict:
    if os.path.exists(db_path):
        with open(db_path, 'r', encoding='utf-8') as f:
            db = json.load(f)
        db.setdefault('active_journeys', {})
        db.setdefault('historical_archives', {})
        return db
    return {'last_updated': None, 'active_journeys': {}, 'historical_archives': {}}


def save_tracking_db(db: dict, db_path: str = TRACKING_DB_PATH) -> None:
    with open(db_path, 'w', encoding='utf-8') as f:
        json.dump(db, f, ensure_ascii=False, indent=2, default=str)


# ─── Yardımcı Fonksiyonlar ───────────────────────────────────────────────────

def _to_date_str(d) -> str:
    """Herhangi bir tarih tipini 'YYYY-MM-DD' string'ine çevir."""
    return pd.to_datetime(d).strftime('%Y-%m-%d')


def _ghost_days(journey: dict, current_date) -> int:
    """GHOST modunda geçen gün sayısı. ghost_since string olabilir → güvenli parse."""
    gs = journey.get('ghost_since')
    if not gs:
        return 0
    curr  = pd.to_datetime(current_date).date()
    ghost = pd.to_datetime(gs).date()
    return (curr - ghost).days


def _next_cycle_id(db: dict, ticker: str) -> int:
    """Bu ticker için active + historical'daki max cycle_id + 1."""
    max_cycle = 0
    for jid, j in {**db['active_journeys'], **db['historical_archives']}.items():
        if j['ticker'] == ticker:
            max_cycle = max(max_cycle, j.get('cycle_id', 0))
    return max_cycle + 1


def _get_list_rank(df: pd.DataFrame, ticker: str) -> 'int | None':
    """df içindeki sırasını bul (1-tabanlı). Yoksa None."""
    matches = df.index[df['Ticker'] == ticker].tolist()
    if not matches:
        return None
    return int(matches[0]) + 1  # 0-tabanlı → 1-tabanlı


def _archive_journey(db: dict, journey_id: str, reason: str, current_date) -> None:
    """active_journeys'ten historical_archives'e taşı."""
    journey = db['active_journeys'].pop(journey_id)
    journey['status'] = 'ARCHIVED_STALE'
    journey['archived_date'] = _to_date_str(current_date)
    journey['archive_reason'] = reason
    db['historical_archives'][journey_id] = journey


# ─── Ana Motor ───────────────────────────────────────────────────────────────

def update_tracking_engine(
    daily_top_df: pd.DataFrame,
    current_market_prices: dict,
    current_date,
    top_n: int = TRACKING_TOP_N,
    db_path: str = TRACKING_DB_PATH,
) -> dict:
    """
    Günlük durum makinesi güncelleme motoru.

    Parametreler
    ------------
    daily_top_df         : Günlük taramanın sıralı çıktısı.
                           Zorunlu kolonlar: Ticker, RankScore, Cluster, Karakter, WinRate
    current_market_prices: {ticker: float} — bugünün kapanış fiyatları (detection_price için)
    current_date         : str veya datetime — o günün tarihi
    top_n                : Kaç hisseyi takip et (varsayılan TRACKING_TOP_N=30)
    db_path              : tracking_db.json dosya yolu

    Döndürür
    --------
    dict: n_active, n_ghost, n_new, n_archived, n_ghost_transitions, new_entries, archived_ids
    """
    current_date_str = _to_date_str(current_date)
    db = load_tracking_db(db_path)

    top_df = daily_top_df.head(top_n).reset_index(drop=True)
    today_tickers = set(top_df['Ticker'].tolist())
    active_tickers = {j['ticker'] for j in db['active_journeys'].values()}

    stats = {
        'date': current_date_str,
        'n_new': 0,
        'n_active': 0,
        'n_ghost': 0,
        'n_ghost_transitions': 0,
        'n_archived': 0,
        'new_entries': [],
        'archived_ids': [],
    }

    # ── ADIM 1: Çöp Toplayıcı (GC) — GHOST zaman aşımı ──────────────────────
    stale_ids = []
    for jid, journey in list(db['active_journeys'].items()):
        if journey['status'] == 'GHOST':
            gd = _ghost_days(journey, current_date)
            if gd > TRACKING_GHOST_TIMEOUT:
                stale_ids.append(jid)

    for jid in stale_ids:
        journey = db['active_journeys'][jid]
        gd = _ghost_days(journey, current_date)
        journey['daily_logs'].append({
            'day': len(journey['daily_logs']) + 1,
            'date': current_date_str,
            'list_rank': None,
            'rankscore': None,
            'cum_return_pct': None,
            'action': 'ARCHIVED_STALE',
            'note': f'{gd} gün hayalet modunda kaldı — arşive taşındı.',
        })
        _archive_journey(db, jid, reason='ghost_timeout', current_date=current_date)
        stats['n_archived'] += 1
        stats['archived_ids'].append(jid)

    # ── ADIM 2: Mevcut ACTIVE/GHOST yolculukları güncelle ────────────────────
    for jid, journey in list(db['active_journeys'].items()):
        ticker = journey['ticker']
        prev_status = journey['status']
        day_n = len(journey['daily_logs']) + 1
        # Hata 9 düzeltmesi — detection_price None ise saçma getiri üretme:
        detection_price = journey.get('detection_price')
        current_price = current_market_prices.get(ticker)
        if detection_price and current_price:
            cum_return = round(((current_price / detection_price) - 1) * 100, 2)
        else:
            cum_return = None

        if ticker in today_tickers:
            # → ACTIVE
            journey['status'] = 'ACTIVE'
            if prev_status == 'GHOST':
                journey['ghost_since'] = None  # Ghost modundan çıktı

            rank = _get_list_rank(top_df, ticker)
            prev_rank = journey['daily_logs'][-1].get('list_rank') if journey['daily_logs'] else rank
            if prev_rank and rank:
                action = 'CLIMBING' if rank < prev_rank else ('SLIDING' if rank > prev_rank else 'HOLDING')
            else:
                action = 'ACTIVE'
            note = f"{rank}. sıraya {'tırmandı' if action=='CLIMBING' else 'geriledi' if action=='SLIDING' else 'tutunuyor'}."

            row = top_df[top_df['Ticker'] == ticker].iloc[0]
            journey['daily_logs'].append({
                'day': day_n,
                'date': current_date_str,
                'list_rank': rank,
                'rankscore': float(row.get('RankScore', float('nan'))) if not pd.isna(row.get('RankScore', float('nan'))) else None,
                'cum_return_pct': cum_return,
                'action': action,
                'note': note,
            })

        else:
            # → GHOST
            if prev_status == 'ACTIVE':
                journey['status'] = 'GHOST'
                journey['ghost_since'] = current_date_str
                stats['n_ghost_transitions'] += 1
                note = 'Radardan düştü, arka planda izleniyor.'
                action = 'GHOST_START'
            else:
                gd = _ghost_days(journey, current_date)
                kalan = TRACKING_GHOST_TIMEOUT - gd
                note = f'Hayalet modunda {gd}. gün. Zaman aşımına {kalan} gün kaldı.'
                action = 'GHOST_CONTINUE'

            journey['daily_logs'].append({
                'day': day_n,
                'date': current_date_str,
                'list_rank': None,
                'rankscore': None,
                'cum_return_pct': cum_return,
                'action': action,
                'note': note,
            })

    # ── ADIM 3: Yeni HOOK — bugünkü top_n'de ACTIVE/GHOST olmayan ticker'lar ──
    # Hata 8 düzeltmesi — active_tickers'i GC sonrası SIFIRDAN inşa et.
    # Aksi halde bugün arşivlenen hisse hâlâ "aktif" sanılır, yeni cycle (AAA_2) açılamaz.
    active_tickers = {j['ticker'] for j in db['active_journeys'].values()
                      if j['status'] in ('ACTIVE', 'GHOST')}

    for _, row in top_df.iterrows():
        ticker = row['Ticker']
        if ticker in active_tickers:
            continue

        cycle_id = _next_cycle_id(db, ticker)
        journey_id = f"{ticker}_{cycle_id}"
        detection_price = current_market_prices.get(ticker, float('nan'))
        rankscore = row.get('RankScore', float('nan'))
        if pd.isna(rankscore):
            rankscore = None
        else:
            rankscore = float(rankscore)

        # Geçmiş operasyon kontrolü
        past_cycles = [j for j in db['historical_archives'].values() if j['ticker'] == ticker]
        history_note = ''
        if past_cycles:
            history_note = f' [Geçmiş Op. Bulundu: Cycle {max(j["cycle_id"] for j in past_cycles)}]'

        rank = _get_list_rank(top_df, ticker)
        db['active_journeys'][journey_id] = {
            'journey_id': journey_id,
            'ticker': ticker,
            'cycle_id': cycle_id,
            'status': 'ACTIVE',
            'detection_date': current_date_str,
            'detection_price': float(detection_price) if not (isinstance(detection_price, float) and np.isnan(detection_price)) else None,
            'detection_cluster': int(row.get('Cluster', -1)),
            'detection_cluster_name': str(row.get('Karakter', '')),
            'detection_rankscore': rankscore,
            'ghost_since': None,
            'daily_logs': [{
                'day': 1,
                'date': current_date_str,
                'list_rank': rank,
                'rankscore': rankscore,
                'cum_return_pct': 0.0,
                'action': 'INITIAL_DETECTION',
                'note': f"{row.get('Karakter', '?')} kümesinden radara girdi.{history_note}",
            }],
        }
        stats['n_new'] += 1
        stats['new_entries'].append({'ticker': ticker, 'journey_id': journey_id,
                                     'detection_price': detection_price,
                                     'rankscore': rankscore,
                                     'cluster_name': str(row.get('Karakter', '')),
                                     'history_note': history_note})
        active_tickers.add(ticker)

    # ── Sayımları güncelle ───────────────────────────────────────────────────
    for j in db['active_journeys'].values():
        if j['status'] == 'ACTIVE':
            stats['n_active'] += 1
        elif j['status'] == 'GHOST':
            stats['n_ghost'] += 1

    db['last_updated'] = current_date_str
    save_tracking_db(db, db_path)
    return stats


# ─── Rapor Üretici ──────────────────────────────────────────────────────────

def generate_report(db: dict, stats: dict, report_dir: str = TRACKING_REPORT_DIR) -> str:
    """Günlük seyir defteri raporunu oluştur ve dosyaya yaz."""
    os.makedirs(report_dir, exist_ok=True)
    date_str = stats['date']
    date_fname = date_str.replace('-', '')

    lines = [
        f"🎯 ARKHIMEDES SEYİR DEFTERİ — {date_str}",
        "─" * 50,
        f"AKTİF: {stats['n_active']}   HAYALET: {stats['n_ghost']}   "
        f"Bugün arşivlendi: {stats['n_archived']}   Yeni tespit: {stats['n_new']}",
        "",
    ]

    # Yeni tespitler
    if stats['new_entries']:
        lines.append(f"🆕 YENİ TESPİT ({len(stats['new_entries'])})")
        for e in stats['new_entries']:
            price_str = f"₺{e['detection_price']:.2f}" if e['detection_price'] else "₺?"
            rs_str = f"RS:{e['rankscore']:+.2f}" if e['rankscore'] is not None else "RS:N/A"
            lines.append(f"  {e['ticker']:<8} {e['cluster_name']:<4} {rs_str:<12} Tespit:{price_str}{e['history_note']}")
        lines.append("")

    # Aktif hisseler (günlük log'un son satırına göre sırala)
    active = [(jid, j) for jid, j in db['active_journeys'].items() if j['status'] == 'ACTIVE']
    active.sort(key=lambda x: len(x[1]['daily_logs']), reverse=True)
    if active:
        lines.append(f"🟢 AKTİF ({len(active)})")
        for jid, j in active:
            last = j['daily_logs'][-1]
            rank = last.get('list_rank')
            rs = last.get('rankscore')
            cr = last.get('cum_return_pct')
            rank_str = f"Sıra:{rank}" if rank else "Sıra:?"
            rs_str = f"RS:{rs:+.2f}" if rs is not None else "RS:N/A"
            cr_str = f"Kümülatif:{cr:+.1f}%" if cr is not None else ""
            p = j.get('detection_price')
            price_str = f"₺{p:.2f}" if p else ""
            lines.append(
                f"  {j['ticker']:<8} Gün:{len(j['daily_logs']):<4} {rank_str:<10} {rs_str:<14} {cr_str:<18} Tespit:{price_str}"
            )
        lines.append("")

    # Hayalet hisseler
    ghosts = [(jid, j) for jid, j in db['active_journeys'].items() if j['status'] == 'GHOST']
    ghosts.sort(key=lambda x: _ghost_days(x[1], stats['date']), reverse=True)
    if ghosts:
        lines.append(f"👻 HAYALET ({len(ghosts)})")
        for jid, j in ghosts:
            gd = _ghost_days(j, stats['date'])
            kalan = TRACKING_GHOST_TIMEOUT - gd
            cr = j['daily_logs'][-1].get('cum_return_pct') if j['daily_logs'] else None
            cr_str = f"Kümülatif:{cr:+.1f}%" if cr is not None else ""
            p = j.get('detection_price')
            price_str = f"₺{p:.2f}" if p else ""
            lines.append(
                f"  {j['ticker']:<8} Gün:{len(j['daily_logs']):<4} Hayalet:{gd}gün  Kalan:{kalan}gün  {cr_str:<18} Tespit:{price_str}"
            )
        lines.append("")

    # Arşivlenenler
    if stats['archived_ids']:
        lines.append(f"📦 BUGÜN ARŞİVLENDİ — STALE ({len(stats['archived_ids'])})")
        for jid in stats['archived_ids']:
            j = db['historical_archives'].get(jid, {})
            lines.append(f"  {j.get('ticker', jid):<8} {TRACKING_GHOST_TIMEOUT} gün hayalet → ARCHIVED_STALE")
        lines.append("")

    report_text = "\n".join(lines)

    # Dosyaya yaz
    fpath = os.path.join(report_dir, f"Seyir_Defteri_{date_fname}.txt")
    with open(fpath, 'w', encoding='utf-8') as f:
        f.write(report_text)

    return report_text
