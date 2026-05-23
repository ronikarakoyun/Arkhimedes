import os
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

# KAP konsolide bilanço bildirim son tarihleri — look-ahead güvenli (geç tarih)
def _period_available_date(year, q_month):
    if q_month == 3:  return pd.Timestamp(year, 5, 11)        # Q1  → 11 Mayıs
    if q_month == 6:  return pd.Timestamp(year, 8, 19)        # Q2  → 19 Ağustos
    if q_month == 9:  return pd.Timestamp(year, 11, 9)        # Q3  → 9 Kasım
    return pd.Timestamp(year + 1, 3, 11)                      # Q4  → 11 Mart (ertesi yıl)

def available_period(obs_date):
    """obs_date itibarıyla KAP'ta yayınlanmış en güncel bilanço dönemi: 'YYYY/Q'."""
    best = None
    for yr in (obs_date.year - 2, obs_date.year - 1, obs_date.year):
        for q_month in (3, 6, 9, 12):
            avail = _period_available_date(yr, q_month)
            if avail <= obs_date and (best is None or avail > best[0]):
                best = (avail, f"{yr}/{q_month}")
    return best[1] if best else None

def winsorize_series(s, lower_p=0.01, upper_p=0.99):
    if s.empty: return s
    lower = s.quantile(lower_p)
    upper = s.quantile(upper_p)
    return s.clip(lower, upper)

def process_hisse(symbol):
    market_path = f"raw_data/market/{symbol}.json"
    fin_path = f"raw_data/financials/{symbol}.json"
    
    if not os.path.exists(market_path):
        return pd.DataFrame()
    
    # 1. Load Market Data
    with open(market_path, 'r', encoding='utf-8') as f:
        m_data = json.load(f)
    
    df_market = pd.DataFrame(m_data)
    cols = ['HGDG_TARIH', 'HGDG_KAPANIS', 'PD', 'HAO_PD', 'PD_USD', 'SERMAYE']
    df_market = df_market[[c for c in cols if c in df_market.columns]]
    df_market['Tarih'] = pd.to_datetime(df_market['HGDG_TARIH'], format='%d-%m-%Y')
    df_market = df_market.sort_values('Tarih')
    df_market['YearMonth'] = df_market['Tarih'].dt.to_period('M')
    monthly_df = df_market.groupby('YearMonth').tail(1).copy()
    
    if 'HAO_PD' in monthly_df.columns and 'PD' in monthly_df.columns:
        monthly_df['Halka_Aciklik_Orani'] = (monthly_df['HAO_PD'] / monthly_df['PD']) * 100
    
    monthly_df['Symbol'] = symbol
    
    # 2. Initialize all potential columns to avoid KeyErrors
    ratio_cols = ['F_K', 'PD_DD', 'FD_FAVOK', 'Net_Borc', 'Firma_Degeri', 'ROE_%', 'ROA_%', 'Brut_Kar_Marji_%', 'Net_Kar_Marji_%', 'Cari_Oran', 'Negative_EBITDA', 'Kisa_Vadeli_Borclanmalar', 'Uzun_Vadeli_Borclanmalar', 'Net_Kar_TTM', 'Satis_Gelirleri_TTM', 'Brut_Kar_TTM', 'Esas_Faaliyet_Kari_TTM', 'FAVOK_TTM', 'Ozkaynaklar', 'Toplam_Varliklar', 'Donen_Varliklar', 'Kisa_Vadeli_Yukumlulukler']
    for col in ratio_cols:
        monthly_df[col] = np.nan
    
    if os.path.exists(fin_path):
        with open(fin_path, 'r', encoding='utf-8') as f:
            f_data = json.load(f)
        
        df_fin = pd.DataFrame(f_data)
        if not df_fin.empty:
            item_col = None
            for c in df_fin.columns:
                if 'name' in c.lower() or 'desc' in c.lower():
                    if 'tr' in c.lower() or 'tr' in c:
                        item_col = c
                        break
                    item_col = c
            if not item_col:
                item_col = df_fin.columns[1] if len(df_fin.columns) > 1 else None

            if item_col:
                targets = {
                    'Net_Kar': ['Dönem Net Kar/Zararı', 'Net Dönem Karı/Zararı', 'Ana Ortaklık Payları', 'Net Kar'],
                    'Ozkaynaklar': ['Özkaynaklar', 'Ana Ortaklığa Ait Özkaynaklar'],
                    'Toplam_Varliklar': ['Toplam Varlıklar', 'TOPLAM VARLIKLAR', 'Toplam Aktifler'],
                    'Satis_Gelirleri': ['Hasılat', 'Satış Gelirleri', 'Faiz Gelirleri'], 
                    'Brut_Kar': ['Brüt Kar/Zarar', 'Brüt Kar'],
                    'Esas_Faaliyet_Kari': ['FAALİYET KARI (ZARARI)', 'Esas Faaliyet Karı/Zararı'],
                    'FAVOK_Direct': ['FAVÖK', 'FVAÖK'],
                    'Amortisman': ['Amortisman & İtfa Payları', 'Amortisman Giderleri'],
                    'Kisa_Vadeli_Yukumlulukler': ['Kısa Vadeli Yükümlülükler'],
                    'Donen_Varliklar': ['Dönen Varlıklar'],
                    'Toplam_Yukumlulukler': ['Toplam Yükümlülükler'],
                    'Nakit_Benzerleri': ['Nakit ve Nakit Benzerleri']
                }

                df_fin['clean_names'] = df_fin[item_col].astype(str).str.strip().str.upper()
                q_cols = [c for c in df_fin.columns if '/' in c and c[0].isdigit()]
                extracted_data = []
                
                for q_col in q_cols:
                    q_data = {'Period': q_col}
                    for var_name, search_terms in targets.items():
                        val = None
                        for term in search_terms:
                            term_upper = term.strip().upper()
                            mask = df_fin['clean_names'] == term_upper
                            if mask.any():
                                val = df_fin[mask].iloc[0][q_col]
                                break
                            mask = df_fin['clean_names'].str.startswith(term_upper)
                            if mask.any():
                                val = df_fin[mask].iloc[0][q_col]
                                break
                        
                        try:
                            if val is not None and not pd.isna(val):
                                raw_val = str(val).replace('.', '').replace(',', '.')
                                q_data[var_name] = float(raw_val)
                            else:
                                q_data[var_name] = None
                        except:
                            q_data[var_name] = None
                    
                    debt_mask = df_fin['clean_names'] == 'Finansal Borçlar'.upper()
                    debt_rows = df_fin[debt_mask]
                    if len(debt_rows) >= 2:
                        try:
                            q_data['Kisa_Vadeli_Borclanmalar'] = float(str(debt_rows.iloc[0][q_col]).replace('.', '').replace(',', '.'))
                            q_data['Uzun_Vadeli_Borclanmalar'] = float(str(debt_rows.iloc[1][q_col]).replace('.', '').replace(',', '.'))
                        except: pass

                    f_dir, ebit, amort = q_data.get('FAVOK_Direct'), q_data.get('Esas_Faaliyet_Kari'), q_data.get('Amortisman')
                    if f_dir is not None and not pd.isna(f_dir): q_data['FAVOK'] = f_dir
                    elif ebit is not None and amort is not None: q_data['FAVOK'] = ebit + amort
                    elif ebit is not None: q_data['FAVOK'] = ebit
                    else: q_data['FAVOK'] = None
                    extracted_data.append(q_data)
                    
                df_res = pd.DataFrame(extracted_data)
                if not df_res.empty:
                    df_res[['Year', 'Quarter']] = df_res['Period'].str.split('/', expand=True).astype(int)
                    df_res = df_res.sort_values(['Year', 'Quarter']).reset_index(drop=True)
                    income_items = ['Net_Kar', 'Satis_Gelirleri', 'Brut_Kar', 'Esas_Faaliyet_Kari', 'FAVOK']
                    for item in income_items:
                        df_res[f'{item}_TTM'] = None
                        for i, row in df_res.iterrows():
                            y, q = row['Year'], row['Quarter']
                            val_current = row[item]
                            if pd.isna(val_current): continue
                            if q == 12: df_res.at[i, f'{item}_TTM'] = val_current
                            else:
                                q4_prev = df_res[(df_res['Year'] == y-1) & (df_res['Quarter'] == 12)]
                                q_prev = df_res[(df_res['Year'] == y-1) & (df_res['Quarter'] == q)]
                                if not q4_prev.empty and not q_prev.empty:
                                    v4p, vp = q4_prev.iloc[0][item], q_prev.iloc[0][item]
                                    if pd.notnull(v4p) and pd.notnull(vp): df_res.at[i, f'{item}_TTM'] = val_current + v4p - vp
                                    else: df_res.at[i, f'{item}_TTM'] = (val_current / q) * 12
                                else: df_res.at[i, f'{item}_TTM'] = (val_current / q) * 12

                    fin_cols = [c for c in df_res.columns if c not in ['Period', 'Year', 'Quarter']]
                    for index, row in monthly_df.iterrows():
                        dt = row['Tarih']
                        target_period = available_period(dt)
                        if target_period is None:
                            continue
                        fin_row = df_res[df_res['Period'] == target_period]
                        if not fin_row.empty:
                            fin_row = fin_row.iloc[0]
                            for c in fin_cols:
                                monthly_df.at[index, c] = fin_row[c]

    # 4. Ratio Calculations
    try: monthly_df['F_K'] = monthly_df['PD'] / monthly_df['Net_Kar_TTM'].astype(float)
    except: pass
    try: monthly_df['PD_DD'] = monthly_df['PD'] / monthly_df['Ozkaynaklar'].astype(float)
    except: pass
    try:
        kisa = monthly_df['Kisa_Vadeli_Borclanmalar'].fillna(0).astype(float)
        uzun = monthly_df['Uzun_Vadeli_Borclanmalar'].fillna(0).astype(float)
        nakit = monthly_df['Nakit_Benzerleri'].fillna(0).astype(float)
        monthly_df['Net_Borc'] = (kisa + uzun) - nakit
        monthly_df['Firma_Degeri'] = monthly_df['PD'] + monthly_df['Net_Borc']
    except: pass
    try:
        favok_ttm = monthly_df['FAVOK_TTM'].astype(float)
        monthly_df['FD_FAVOK'] = monthly_df['Firma_Degeri'] / favok_ttm
        monthly_df['Negative_EBITDA'] = favok_ttm < 0
    except: pass
    try: monthly_df['ROE_%'] = (monthly_df['Net_Kar_TTM'].astype(float) / monthly_df['Ozkaynaklar'].astype(float)) * 100
    except: pass
    try: monthly_df['ROA_%'] = (monthly_df['Net_Kar_TTM'].astype(float) / monthly_df['Toplam_Varliklar'].astype(float)) * 100
    except: pass
    try: monthly_df['Brut_Kar_Marji_%'] = (monthly_df['Brut_Kar_TTM'].astype(float) / monthly_df['Satis_Gelirleri_TTM'].astype(float)) * 100
    except: pass
    try: monthly_df['Net_Kar_Marji_%'] = (monthly_df['Net_Kar_TTM'].astype(float) / monthly_df['Satis_Gelirleri_TTM'].astype(float)) * 100
    except: pass
    try: monthly_df['Cari_Oran'] = monthly_df['Donen_Varliklar'].astype(float) / monthly_df['Kisa_Vadeli_Yukumlulukler'].astype(float)
    except: pass
    
    monthly_df['History_Count'] = len(monthly_df)
    return monthly_df

if __name__ == "__main__":
    market_files = [f.replace('.json', '') for f in os.listdir('raw_data/market') if f.endswith('.json')]
    symbols = market_files 
    all_dfs = []
    print(f"İşleniyor ({len(symbols)} hisse)...")
    for sym in tqdm(symbols):
        df = process_hisse(sym)
        if not df.empty: all_dfs.append(df)
            
    if all_dfs:
        final_df = pd.concat(all_dfs, ignore_index=True)
        print("Winsorization uygulanıyor...")
        ratio_cols = ['F_K', 'PD_DD', 'FD_FAVOK', 'ROE_%', 'ROA_%', 'Brut_Kar_Marji_%', 'Net_Kar_Marji_%', 'Cari_Oran']
        for col in ratio_cols:
            if col in final_df.columns:
                final_df[f'{col}_Winsorized'] = winsorize_series(final_df[col])
        
        if 'YearMonth' in final_df.columns: final_df['YearMonth'] = final_df['YearMonth'].astype(str)
        drop_cols = ['HGDG_TARIH', 'FAVOK_Direct']
        final_df = final_df.drop(columns=[c for c in drop_cols if c in final_df.columns])
        final_df.to_parquet("BIST_Tarihsel_Temel_Analiz.parquet", engine='pyarrow')
        print(f"\n✅ İşlem tamamlandı. Veri: {final_df.shape}.")
    else:
        print("İşlenecek veri bulunamadı.")
