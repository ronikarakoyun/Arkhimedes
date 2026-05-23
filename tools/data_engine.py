import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import polars as pl
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import os
import warnings

warnings.filterwarnings('ignore')

# ─── Configuration ────────────────────────────────────────────────────────────
from config import DB_PATH
LOOKBACK_YEARS = 10

# Makro veri: forex, emtia — ".IS" suffix yok, ticker adı doğrudan kullanılır
MACRO_TICKERS = {
    "USDTRY=X": "USDTRY",
    "XBANK.IS": "XBANK",   # Bankacılık (Yahoo'da tam geçmiş var)
    "XUSIN.IS": "XUSIN",   # Sanayi (Yahoo'da tam geçmiş var)
}

# Full list provided by user
TICKERS = [
    "XU100.IS", "THYAO.IS", "GARAN.IS", "ASELS.IS", "EREGL.IS", "KCHOL.IS", "SAHOL.IS", "SISE.IS", 
    "TOASO.IS", "AKBNK.IS", "YKBNK.IS", "BIMAS.IS", "FROTO.IS", "PGSUS.IS", "TAVHL.IS", "ADESE.IS",
    "ADLVY.IS", "ADGYO.IS", "AFYON.IS", "AGHOL.IS", "AGESA.IS", "AGROT.IS", "AAGYO.IS", "AHSGY.IS",
    "AHGAZ.IS", "AKSFA.IS", "AKFK.IS", "AKM.IS", "AKMEN.IS", "AKCVR.IS", "AKCNS.IS", "AKDFA.IS",
    "AKYHO.IS", "AKENR.IS", "AKFGY.IS", "AKFIS.IS", "AKFYE.IS", "AKHAN.IS", "ATEKS.IS", "AKSGY.IS",
    "AKMGY.IS", "AKSA.IS", "AKSEN.IS", "AKGRT.IS", "AKSUE.IS", "AKTVK.IS", "AFB.IS", "AKTIF.IS",
    "ALCAR.IS", "ALGYO.IS", "ALARK.IS", "ALBRK.IS", "ALK.IS", "ALCTL.IS", "ALFAS.IS", "ALJF.IS",
    "ALKIM.IS", "ALKA.IS", "ALNUS.IS", "ANC.IS", "AYCES.IS", "ALTNY.IS", "ALKLC.IS", "ALVES.IS",
    "ANSGR.IS", "AEFES.IS", "ANHYT.IS", "ASUZU.IS", "ANGEN.IS", "ANELE.IS", "ARCLK.IS", "ARDYZ.IS",
    "ARENA.IS", "ARFYE.IS", "ARMGD.IS", "ARSAN.IS", "ARSVY.IS", "ARTMS.IS", "ARZUM.IS", "ASGYO.IS",
    "ASTOR.IS", "ATAGY.IS", "ATATR.IS", "ATAVK.IS", "ATA.IS", "ATAYM.IS", "ATAKP.IS", "AGYO.IS",
    "ATLFA.IS", "ATSYH.IS", "ATLAS.IS", "ATATP.IS", "AVOD.IS", "AVGYO.IS", "AVTUR.IS", "AVHOL.IS",
    "AVPGY.IS", "AYDEM.IS", "AYEN.IS", "AYES.IS", "AYGAZ.IS", "AZTEK.IS", "A1CAP.IS", "ACP.IS",
    "A1YEN.IS", "BAGFS.IS", "BAHKM.IS", "BAKAB.IS", "BALAT.IS", "BALSU.IS", "BNTAS.IS", "BANVT.IS",
    "BARMA.IS", "BSRFK.IS", "BASGZ.IS", "BASCM.IS", "BEGYO.IS", "BTCIM.IS", "BSOKE.IS", "BYDNR.IS",
    "BAYRK.IS", "BERA.IS", "BRKT.IS", "BRKSN.IS", "BESLR.IS", "BESTE.IS", "BJKAS.IS", "BEYAZ.IS",
    "BIENY.IS", "BIGTK.IS", "BLCYT.IS", "BLKOM.IS", "BINBN.IS", "BIOEN.IS", "BRKVY.IS", "BRKO.IS",
    "BIGEN.IS", "BRLSM.IS", "BRMEN.IS", "BIZIM.IS", "BLUME.IS", "BMSTL.IS", "BMSCH.IS", "BNPPI.IS",
    "BOBET.IS", "BORSK.IS", "BORLS.IS", "BRSAN.IS", "BRYAT.IS", "BFREN.IS", "BOSSA.IS", "BRISA.IS",
    "BULGS.IS", "BLS.IS", "BLSMD.IS", "BURCE.IS", "BURVA.IS", "BRGAN.IS", "BUR.IS", "BRGFK.IS",
    "BUCIM.IS", "BVSAN.IS", "BIGCH.IS", "CRFSA.IS", "CASA.IS", "CEMZY.IS", "CEOEM.IS", "CCOLA.IS",
    "CONSE.IS", "COSMO.IS", "CRDFA.IS", "CVKMD.IS", "CWENE.IS", "CGCAM.IS", "CAGFA.IS", "CMSAN.IS",
    "CANTE.IS", "CATES.IS", "CLEBI.IS", "CELHA.IS", "CLKMT.IS", "CEMAS.IS", "CEMTS.IS", "CMBTN.IS",
    "CMENT.IS", "CIMSA.IS", "CUSAN.IS", "DVRLK.IS", "DYBNK.IS", "DAGI.IS", "DAPGM.IS", "DARDL.IS",
    "DGATE.IS", "DCTTR.IS", "DGRVK.IS", "DMSAS.IS", "DENVA.IS", "DENGE.IS", "DNZEN.IS", "DENFA.IS",
    "DZGYO.IS", "DERIM.IS", "DERHL.IS", "DESA.IS", "DESPC.IS", "DSTKF.IS", "DMD.IS", "DSYAT.IS",
    "DEVA.IS", "DNISI.IS", "DIRIT.IS", "DITAS.IS", "DKVRL.IS", "DMRGD.IS", "DOCO.IS", "DOFRB.IS",
    "DOFER.IS", "DOHOL.IS", "DGNMO.IS", "DOGVY.IS", "ARASE.IS", "DOGUB.IS", "DGGYO.IS", "DOAS.IS",
    "DFKTR.IS", "DOKTA.IS", "DURDO.IS", "DURKN.IS", "DUNYH.IS", "DNYVA.IS", "DYOBY.IS", "EBEBK.IS",
    "ECOGR.IS", "ECZYT.IS", "EDATA.IS", "EDIP.IS", "EFOR.IS", "EGEEN.IS", "EGGUB.IS", "EGPRO.IS",
    "EGSER.IS", "EPLAS.IS", "EGEGY.IS", "ECZIP.IS", "ECILC.IS", "EKER.IS", "EKIZ.IS", "EKOFA.IS",
    "EKOS.IS", "EKSUN.IS", "ELITE.IS", "EMKEL.IS", "EMNIS.IS", "EMIRV.IS", "EKTVK.IS", "DMLKT.IS",
    "EKGYO.IS", "EMVAR.IS", "EMPAE.IS", "ENDAE.IS", "ENJSA.IS", "ENERY.IS", "ENKAI.IS", "ENPRA.IS",
    "ENSRI.IS", "ERBOS.IS", "ERCB.IS", "ERGLI.IS", "KIMMR.IS", "ERSU.IS", "ESCAR.IS", "ESCOM.IS",
    "ESEN.IS", "ETILR.IS", "EUKYO.IS", "EUYO.IS", "ETYAT.IS", "EUHOL.IS", "TEZOL.IS", "EUREN.IS",
    "EUPWR.IS", "EYGYO.IS", "FADE.IS", "FMIZP.IS", "FENER.IS", "FBB.IS", "FBBNK.IS", "FKPET.IS",
    "FLAP.IS", "FONET.IS", "FORMT.IS", "FRMPL.IS", "FORTE.IS", "FRIGO.IS", "FZLGY.IS", "GWIND.IS",
    "GSRAY.IS", "GARFA.IS", "GARFL.IS", "GRNYO.IS", "GATEG.IS", "GEDIK.IS", "GEDZA.IS", "GLCVY.IS",
    "GENIL.IS", "GENTS.IS", "GENKM.IS", "GEREL.IS", "GZNMI.IS", "GIPTA.IS", "GMTAS.IS", "GESAN.IS",
    "GLB.IS", "GLBMD.IS", "GLYHO.IS", "GGBVK.IS", "GSIPD.IS", "GOODY.IS", "GOKNR.IS", "GOLTS.IS",
    "GRTHO.IS", "GSDDE.IS", "GSDHO.IS", "GUBRF.IS", "GLRYH.IS", "GLRMK.IS", "GUNDG.IS", "GRSEL.IS",
    "HALKF.IS", "HLGYO.IS", "HLVKS.IS", "HALKI.IS", "HLY.IS", "HRKET.IS", "HATEK.IS", "HATSN.IS",
    "HAYVK.IS", "HDFFL.IS", "HDFGS.IS", "HEDEF.IS", "HDFVK.IS", "HDFYB.IS", "HYB.IS", "HEKTS.IS",
    "HKTM.IS", "HTTBT.IS", "HOROZ.IS", "HUBVC.IS", "HUNER.IS", "HUZFA.IS", "HURGZ.IS", "ICB.IS",
    "ICBCT.IS", "ICUGS.IS", "INGRM.IS", "INVEO.IS", "IAZ.IS", "INVAZ.IS", "INVES.IS", "ISKPL.IS",
    "IEYHO.IS", "IDGYO.IS", "IHEVA.IS", "IHLGM.IS", "IHGZT.IS", "IHAAS.IS", "IHLAS.IS", "IHYAY.IS",
    "IMASM.IS", "INDES.IS", "INFO.IS", "IYF.IS", "INTEK.IS", "INTEM.IS", "ISDMR.IS", "ISTFK.IS",
    "ISTVY.IS", "ISFAK.IS", "ISFIN.IS", "ISGYO.IS", "ISGSY.IS", "ISMEN.IS", "IYM.IS", "ISYAT.IS",
    "ISBIR.IS", "ISSEN.IS", "IZINV.IS", "IZENR.IS", "IZMDC.IS", "IZFAS.IS", "JANTS.IS", "KFEIN.IS",
    "KLKIM.IS", "KLSER.IS", "KLVKS.IS", "KLYPV.IS", "KTEST.IS", "KAPLM.IS", "KRDMA.IS", "KRDMB.IS",
    "KRDMD.IS", "KAREL.IS", "KARSN.IS", "KRTEK.IS", "KARTN.IS", "KATVK.IS", "KTLEV.IS", "KATMR.IS",
    "KFILO.IS", "KAYSE.IS", "KNTFA.IS", "KENT.IS", "KRVGD.IS", "KERVN.IS", "TCKRC.IS", "KZBGY.IS",
    "KLGYO.IS", "KLRHO.IS", "KMPUR.IS", "KLMSN.IS", "KCAER.IS", "KOCFN.IS", "KOCMT.IS", "KSFIN.IS",
    "KLSYN.IS", "KNFRT.IS", "KONTR.IS", "KONYA.IS", "KONKA.IS", "KGYO.IS", "KORDS.IS", "KRPLS.IS",
    "KORTS.IS", "KOTON.IS", "KOPOL.IS", "KRGYO.IS", "KRSTL.IS", "KRONT.IS", "KTKVK.IS", "KTSVK.IS",
    "KSTUR.IS", "KUVVA.IS", "KUYAS.IS", "KBORU.IS", "KZGYO.IS", "KUTPO.IS", "KTSKR.IS", "LIDER.IS",
    "LIDFA.IS", "LILAK.IS", "LMKDC.IS", "LINK.IS", "LOGO.IS", "LKMNH.IS", "LRSHO.IS", "LXGYO.IS",
    "LUKSK.IS", "LYDHO.IS", "LYDYE.IS", "MACKO.IS", "MAKIM.IS", "MAKTK.IS", "MANAS.IS", "MRBAS.IS",
    "MRS.IS", "MAGEN.IS", "MRMAG.IS", "MARKA.IS", "MARMR.IS", "MAALT.IS", "MRSHL.IS", "MRGYO.IS",
    "MARTI.IS", "MTRKS.IS", "MAVI.IS", "MZHLD.IS", "MEDTR.IS", "MEGMT.IS", "MEGAP.IS", "MEKAG.IS",
    "MEKMD.IS", "MSA.IS", "MNDRS.IS", "MEPET.IS", "MERCN.IS", "MRBKF.IS", "MBFTR.IS", "MERIT.IS",
    "MERKO.IS", "METRO.IS", "MTRYO.IS", "MCARD.IS", "MEYSU.IS", "MHRGY.IS", "MIATK.IS", "MDASM.IS",
    "MDS.IS", "MGROS.IS", "MINTF.IS", "MSGYO.IS", "MSY.IS", "MSYBN.IS", "MPARK.IS", "MMCAS.IS",
    "MNGFA.IS", "MOBTL.IS", "MOGAN.IS", "MNDTR.IS", "MOPAS.IS", "EGEPO.IS", "NATEN.IS", "NTGAZ.IS",
    "NTHOL.IS", "NETAS.IS", "NETCD.IS", "NIBAS.IS", "NUHCM.IS", "NUGYO.IS", "NURVK.IS", "NRBNK.IS",
    "NYB.IS", "OBAMS.IS", "OBASE.IS", "ODAS.IS", "ODINE.IS", "OFSYM.IS", "ONCSM.IS", "ONRYT.IS",
    "OPET.IS", "ORCAY.IS", "ORFIN.IS", "ORGE.IS", "ORMA.IS", "OSVKS.IS", "OMD.IS", "OSMEN.IS",
    "OSTIM.IS", "OTKAR.IS", "OTOKC.IS", "OTOSR.IS", "OTTO.IS", "OYAKC.IS", "OYA.IS", "OYYAT.IS",
    "OYAYO.IS", "OYLUM.IS", "OZKGY.IS", "OZATD.IS", "OZGYO.IS", "OZRDN.IS", "OZSUB.IS", "OZYSR.IS",
    "PAMEL.IS", "PNLSN.IS", "PAGYO.IS", "PAPIL.IS", "PRFFK.IS", "PRDGS.IS", "PRKME.IS", "PARSN.IS",
    "PBT.IS", "PBTR.IS", "PASEU.IS", "PSGYO.IS", "PAHOL.IS", "PATEK.IS", "PCILT.IS", "PEKGY.IS",
    "PENGD.IS", "PENTA.IS", "PSDTC.IS", "PETKM.IS", "PKENT.IS", "PETUN.IS", "PINSU.IS", "PNSUT.IS",
    "PKART.IS", "PLTUR.IS", "POLHO.IS", "POLTK.IS", "PRZMA.IS", "QFINF.IS", "QYATB.IS", "YBQ.IS",
    "QYHOL.IS", "FIN.IS", "QNBTR.IS", "QNBFF.IS", "QNBFK.IS", "QNBVK.IS", "QUAGR.IS", "QUFIN.IS",
    "RNPOL.IS", "RALYH.IS", "RAYSG.IS", "REEDR.IS", "RYGYO.IS", "RYSAS.IS", "RODRG.IS", "ROYAL.IS",
    "RGYAS.IS", "RTALB.IS", "RUBNS.IS", "RUZYE.IS", "SAFKR.IS", "SANEL.IS", "SNICA.IS", "SANFM.IS",
    "SANKO.IS", "SAMAT.IS", "SARKY.IS", "SASA.IS", "SVGYO.IS", "SAYAS.IS", "SDTTR.IS", "SEGMN.IS",
    "SEKUR.IS", "SELEC.IS", "SELVA.IS", "SERNT.IS", "SRVGY.IS", "SEYKM.IS", "SILVR.IS", "SNGYO.IS",
    "SKYLP.IS", "SMRTG.IS", "SMART.IS", "SODSN.IS", "SOKE.IS", "SKTAS.IS", "SONME.IS", "SNPAM.IS",
    "SUMAS.IS", "SUNTK.IS", "SURGY.IS", "SUWEN.IS", "SMRFA.IS", "SMRVA.IS", "SEKFK.IS", "SEGYO.IS",
    "SKY.IS", "SKYMD.IS", "SEK.IS", "SKBNK.IS", "SOKM.IS", "TABGD.IS", "TAC.IS", "TCRYT.IS",
    "TAMFA.IS", "TNZTP.IS", "TARKM.IS", "TATGD.IS", "TATEN.IS", "DRPHN.IS", "TEBFA.IS", "TEKTU.IS",
    "TKFEN.IS", "TKNSA.IS", "TMPOL.IS", "TRHOL.IS", "TEVKS.IS", "TAE.IS", "TRBNK.IS", "TERA.IS",
    "TRA.IS", "TEHOL.IS", "TFNVK.IS", "TGSAS.IS", "TIMUR.IS", "TRYKI.IS", "TRGYO.IS", "TRMET.IS",
    "TRENJ.IS", "TLMAN.IS", "TSPOR.IS", "TDGYO.IS", "TRMEN.IS", "TVM.IS", "TSGYO.IS", "TUCLK.IS",
    "TUKAS.IS", "TRCAS.IS", "TUREX.IS", "MARBL.IS", "TRKFN.IS", "TRILC.IS", "TCELL.IS", "TRKNT.IS",
    "TMSN.IS", "TUPRS.IS", "TRALT.IS", "PRKAB.IS", "TTKOM.IS", "TTRAK.IS", "TBORG.IS", "TURGG.IS",
    "TGB.IS", "HALKB.IS", "THL.IS", "EXIMB.IS", "THR.IS", "ISATR.IS", "ISBTR.IS", "ISCTR.IS",
    "ISKUR.IS", "TIB.IS", "KLN.IS", "KLNMA.IS", "TSK.IS", "TSKB.IS", "TURSG.IS", "TVB.IS",
    "VAKBN.IS", "TV8TV.IS", "UFUK.IS", "ULAS.IS", "ULUFA.IS", "ULUSE.IS", "ULUUN.IS", "UMPAS.IS",
    "USAK.IS", "UCAYM.IS", "ULKER.IS", "UNLU.IS", "VAKFA.IS", "VAKFN.IS", "VKGYO.IS", "VKFYO.IS",
    "VAKVK.IS", "VAKKO.IS", "VANGD.IS", "VBTYZ.IS", "VDFLO.IS", "VRGYO.IS", "VERUS.IS", "VERTU.IS",
    "VESBE.IS", "VESTL.IS", "VKING.IS", "VSNMD.IS", "VDFAS.IS", "YKFKT.IS", "YKFIN.IS", "YKR.IS",
    "YKYAT.IS", "YKB.IS", "YAPRK.IS", "YATAS.IS", "YYLGD.IS", "YAYLA.IS", "YGGYO.IS", "YEOTK.IS",
    "YYAPI.IS", "YESIL.IS", "YBTAS.IS", "YIGIT.IS", "YONGA.IS", "YKSLN.IS", "YUNSA.IS", "ZGYO.IS",
    "ZEDUR.IS", "ZERGY.IS", "ZRGYO.IS", "ZKBVK.IS", "ZKBVR.IS", "ZOREN.IS", "BINHO.IS",
]

def fetch_ticker_data(symbol, start, end):
    try:
        df = yf.download(symbol, start=start, end=end, auto_adjust=True, progress=False)
        if df.empty: return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.reset_index()
        df["Ticker"] = symbol.replace(".IS", "")
        col_map = {"Open": "Popen", "High": "Phigh", "Low": "Plow", "Close": "Pclose", "Volume": "Vlot"}
        df = df.rename(columns=col_map)
        keep = ["Date", "Ticker", "Popen", "Phigh", "Plow", "Pclose", "Vlot"]
        df = df[[c for c in keep if c in df.columns]]
        return pl.from_pandas(df)
    except Exception as e:
        print(f"   ⚠️  Bist ticker fetch hatası: {type(e).__name__}: {e}")
        return None

def _fetch_macro_data(start_date, end_date):
    """Makro tickerları (forex, emtia) çeker ve standart formata dönüştürür."""
    frames = []
    for yf_symbol, clean_name in MACRO_TICKERS.items():
        try:
            df = yf.download(yf_symbol, start=start_date, end=end_date,
                             auto_adjust=True, progress=False)
            if df.empty:
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.reset_index()
            df["Ticker"] = clean_name
            col_map = {"Open": "Popen", "High": "Phigh", "Low": "Plow",
                       "Close": "Pclose", "Volume": "Vlot"}
            df = df.rename(columns=col_map)
            if "Vlot" not in df.columns:
                df["Vlot"] = 0.0
            keep = ["Date", "Ticker", "Popen", "Phigh", "Plow", "Pclose", "Vlot"]
            df = df[[c for c in keep if c in df.columns]]
            frames.append(pl.from_pandas(df))
            print(f"  ✅ Makro veri çekildi: {clean_name} ({len(df)} satır)")
        except Exception as e:
            print(f"  ⚠️ Makro veri hatası ({yf_symbol}): {e}")
    return frames


def data_engine():
    print(f"🚀 Data Engine Starting for {len(TICKERS)} tickers...")
    end_date = (datetime.today() + timedelta(days=1)).strftime("%Y-%m-%d")
    start_date = (datetime.today() - timedelta(days=LOOKBACK_YEARS * 365)).strftime("%Y-%m-%d")
    
    existing_df = None
    if os.path.exists(DB_PATH):
        existing_df = pl.read_parquet(DB_PATH)
        # Force a full sync check for the NEW tickers provided
        existing_tickers = existing_df["Ticker"].unique().to_list()
        missing_tickers = [t for t in TICKERS if t.replace(".IS", "") not in existing_tickers]
        
        if missing_tickers:
            print(f"📥 Found {len(missing_tickers)} NEW tickers to sync from history.")
            # For simplicity in this large update, we'll fetch missing ones from 10y back 
            # and existing ones from their last date. 
        else:
            last_date = existing_df["Date"].max()
            start_date = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
            if start_date >= end_date:
                print("✅ All tickers are up to date.")
                return

    new_frames = []
    total = len(TICKERS)
    for i, symbol in enumerate(TICKERS):
        if (i+1) % 50 == 0 or i == 0:
            print(f"  Syncing {i+1}/{total}: {symbol}...")
        
        # Determine specific start date for this symbol if it's new
        sym_clean = symbol.replace(".IS", "")
        if existing_df is not None and sym_clean in existing_df["Ticker"].to_list():
            # Incremental for existing
            last_date_sym = existing_df.filter(pl.col("Ticker") == sym_clean)["Date"].max()
            sym_start = (last_date_sym + timedelta(days=1)).strftime("%Y-%m-%d")
        else:
            # Full 10y for new
            sym_start = (datetime.today() - timedelta(days=LOOKBACK_YEARS * 365)).strftime("%Y-%m-%d")
        
        if sym_start < end_date:
            df = fetch_ticker_data(symbol, sym_start, end_date)
            if df is not None:
                new_frames.append(df)
            
    # Makro verileri çek (USDTRY vb.)
    print("📡 Makro veriler çekiliyor...")
    macro_start = (datetime.today() - timedelta(days=LOOKBACK_YEARS * 365)).strftime("%Y-%m-%d")
    macro_frames = _fetch_macro_data(macro_start, end_date)
    new_frames.extend(macro_frames)

    if not new_frames:
        print("⚠️ No new data fetched.")
        return

    new_df = pl.concat(new_frames)
    
    # Calculate Pvwap for new_df BEFORE concat with existing_df to match columns
    new_df = new_df.with_columns(
        ((pl.col("Phigh") + pl.col("Plow") + pl.col("Pclose")) / 3).alias("Pvwap")
    )

    if existing_df is not None:
        # Ensure column order matches
        new_df = new_df.select(existing_df.columns)
        combined = pl.concat([existing_df, new_df])
    else:
        combined = new_df

    combined = combined.unique(subset=["Date", "Ticker"]).sort(["Ticker", "Date"])
    combined = combined.filter(pl.col("Pclose") > 0)

    combined.write_parquet(DB_PATH)
    print(f"📦 {DB_PATH} updated. Total rows: {len(combined):,}, Total tickers: {combined['Ticker'].n_unique()}")

if __name__ == "__main__":
    data_engine()
