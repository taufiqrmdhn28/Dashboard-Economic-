import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import google.generativeai as genai
from statsmodels.tsa.statespace.dynamic_factor_mq import DynamicFactorMQ
import os
import pickle
import warnings
import hashlib

# Abaikan warning agar terminal bersih
warnings.filterwarnings('ignore')

file_makro = "Makro Indikator AI.xlsx"
file_adb = "INO_02022026.xlsx"

# ==========================================
# 0. KONFIGURASI API KEY (SECURE)
# ==========================================
try:
    USER_API_KEY = st.secrets["GEMINI_API_KEY"]
except:
    USER_API_KEY = ""

# ==========================================
# SETUP CACHE AI (Biar Abadi)
# ==========================================
CACHE_FILE = "policy_cache.pkl"

if 'policy_cache' not in st.session_state:
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "rb") as f:
            st.session_state.policy_cache = pickle.load(f)
    else:
        st.session_state.policy_cache = {}

# FUNGSI SIGNATURE: Mengingat Data Makro (Lengkap) & Data Harian
def make_signature(view, avg, target, monthly_info, daily_info):
    raw_str = f"{view}_{avg:.2f}_{target}_{monthly_info}_{daily_info}"
    import hashlib
    return hashlib.md5(raw_str.encode()).hexdigest()

# ==========================================
# 1. SETUP & DESIGN
# ==========================================
st.set_page_config(page_title="Macro AI Command Center", layout="wide", page_icon="🇮🇩")

st.markdown("""
<style>
    .stApp { background: radial-gradient(circle at 10% 20%, rgb(242, 243, 247) 0%, rgb(215, 221, 232) 90.2%); }
    .glass-card {
        background: rgba(255, 255, 255, 0.65);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.7);
        padding: 24px;
        margin-bottom: 24px;
    }
    .card-title { font-size: 13px; color: #444; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px; }
    .card-value { font-size: 26px; color: #111; font-weight: 800; margin: 4px 0; }
    .badge { display: inline-block; padding: 4px 10px; border-radius: 8px; font-size: 11px; font-weight: 700; margin-right: 6px; }
    .badge-green { background: rgba(212, 237, 218, 0.8); color: #155724; }
    .badge-red { background: rgba(248, 215, 218, 0.8); color: #721c24; }
    .badge-neutral { background: rgba(226, 227, 229, 0.8); color: #383d41; }
    h1 { color: #002d72 !important; }
</style>
""", unsafe_allow_html=True)

st.title("🇮🇩 National Economic Command Center")
st.markdown("##### Engine: Holt-Winters (Econometric Forecasting) | Standard: Official Statistics")

# ==========================================
# 2. DATA LOADING
# ==========================================
@st.cache_data
def load_data():
    try:
        df_target = pd.read_excel(file_makro, sheet_name=0)
        df_triwulan = pd.read_excel(file_makro, sheet_name=1)
        df_makro = pd.read_excel(file_makro, sheet_name=2)
        df_hist_gdp = pd.read_excel(file_adb, sheet_name=2)
        return df_target, df_triwulan, df_makro, df_hist_gdp
    except Exception as e:
        st.error(f"Error Loading Data: {e}")
        return None, None, None, None

df_target, df_triwulan, df_makro, df_hist_gdp = load_data()

# ==========================================
# 2.5. DATA LOADING (GOOGLE SHEETS HARIAN)
# ==========================================
@st.cache_data(ttl=3600)
def load_daily_data():
    try:
        url = "https://docs.google.com/spreadsheets/d/1wM0lHYqNTgf4Jo4AMCDakWnwqF1lVg-7/export?format=xlsx&gid=1981545536"
        df_daily = pd.read_excel(url, engine="openpyxl")
        
        date_col = 'Tanggal' if 'Tanggal' in df_daily.columns else df_daily.columns[0]
        df_daily[date_col] = pd.to_datetime(df_daily[date_col])
        df_daily = df_daily.sort_values(by=date_col)
        
        return df_daily, date_col
    except Exception as e:
        st.warning(f"⚠️ Gagal sinkronisasi data Google Sheets. Info Error: {e}")
        return None, None

df_daily, date_col_daily = load_daily_data()

# ==========================================
# 3. ENGINE DFM NOWCASTING (TARGET 4 KUARTAL 2026)
# ==========================================
def apply_matlab_transformation(series, j1, j2, j3, freq='M'):
    out = series.copy().astype(float)
    if j1 == 1:
        out = out.mask(out <= 0, np.nan)
        out = 100 * np.log(out)
    if j2 == 1:
        out = out.diff(1)
    elif j3 == 1:
        lags = 12 if freq == 'M' else 4
        out = out.diff(lags)
    return out

def build_ragged_vintage(data_full, df_cal, indicator_col, vintage_cols, v_date, obs_cutoff):
    vintage = data_full[data_full.index <= obs_cutoff].copy()
    vcols_sorted = sorted([c for c in vintage_cols if c <= obs_cutoff])
    v_col_key = vcols_sorted[-1] if vcols_sorted else None
    if v_col_key is None: return vintage
    for _, row in df_cal.iterrows():
        ind = row[indicator_col]
        if ind not in vintage.columns: continue
        rd = pd.to_datetime(row[v_col_key], errors="coerce")
        if pd.notna(rd) and rd > v_date:
            mask_from = rd.replace(day=1) - pd.DateOffset(months=1)
            vintage.loc[vintage.index >= mask_from, ind] = np.nan
    return vintage

def get_prediction_value(pred_means, target, quarter):
    if target not in pred_means.columns: return np.nan
    q_end = quarter.to_timestamp(how="end").normalize()
    q_start = quarter.to_timestamp(how="start")
    for candidate in [q_start, q_end, q_start.replace(day=1), q_end.replace(day=1)]:
        if candidate in pred_means.index:
            return float(pred_means.loc[candidate, target])
    return np.nan

def calculate_annual_nowcast(pred_means, target_var, cutoff):
    year = cutoff.year
    vals = [get_prediction_value(pred_means, target_var, pd.Period(year=year, quarter=q, freq='Q')) for q in range(1, 5)]
    vals = [v for v in vals if pd.notna(v)]
    return np.mean(vals) if len(vals) == 4 else np.nan

@st.cache_data(show_spinner="⚙️ DFM Nowcasting: Memproses 4 Titik Kalender Terakhir di Tiap Kuartal 2026...")
def run_full_dfm_replication():
    try:
        # 1. Load Data
        df_m_raw = pd.read_excel(file_adb, sheet_name='MonthlyData', index_col=0, parse_dates=True)
        df_q_raw = pd.read_excel(file_adb, sheet_name='QuarterlyData', index_col=0, parse_dates=True)
        df_cal = pd.read_excel(file_adb, sheet_name='Calendar')
        info_m = pd.read_excel(file_adb, sheet_name='InfoM')
        info_q = pd.read_excel(file_adb, sheet_name='InfoQ')
        
        if "INCLUDE" in df_cal.columns: df_cal = df_cal[df_cal["INCLUDE"] == 1].reset_index(drop=True)
        indicator_col = df_cal.columns[0]
        vintage_cols = [pd.to_datetime(c) for c in df_cal.columns[2:]]

        processed_data = {}
        for _, row in info_m[info_m['INCLUDED'] == 1].iterrows():
            name = row['Indicator Code']
            if name in df_m_raw.columns:
                processed_data[name] = apply_matlab_transformation(df_m_raw[name], row['log'], row['MoM'], row['YoY'], 'M')
        for _, row in info_q[info_q['INCLUDED'] == 1].iterrows():
            name = row['Indicator Code']
            if name in df_q_raw.columns:
                s = apply_matlab_transformation(df_q_raw[name], row['log'], row['QoQ'], row['YoY'], 'Q')
                processed_data[name] = s

        # 2. RATA-KAN FREKUENSI KE 'MS'
        data_full = pd.DataFrame(processed_data).replace([np.inf, -np.inf], np.nan)
        data_full.index = pd.to_datetime(data_full.index)
        data_full = data_full.resample('MS').first() 
        target_var = 'RGDP_growth'
        if target_var not in data_full.columns: return pd.DataFrame()
        
        # 3. Kumpulkan Jadwal Rilis KHUSUS TAHUN 2026
        jobs = []
        seen = set()
        for vc in vintage_cols:
            col_name = vc.strftime('%Y-%m-%d 00:00:00') if vc.strftime('%Y-%m-%d 00:00:00') in df_cal.columns else df_cal.columns[2 + vintage_cols.index(vc)]
            release_dates = pd.to_datetime(df_cal[col_name], errors="coerce").dropna().unique()
            for rd in sorted(release_dates):
                if rd.year == 2026 and (rd, vc) not in seen:
                    seen.add((rd, vc)); jobs.append((rd, vc))
        jobs.sort(key=lambda x: x[0])
        
        # 4. Filter: Ambil Tanggal Paling Terakhir di Tiap Kuartal (Q1, Q2, Q3, Q4)
        target_jobs = []
        for q in [1, 2, 3, 4]:
            q_jobs = [j for j in jobs if j[0].quarter == q]
            if q_jobs: target_jobs.append(q_jobs[-1]) 
                
        if not target_jobs: return pd.DataFrame()

        # 5. Eksekusi Iterasi
        results_table = []
        for actual_v_date, v_date_base in target_jobs:
            obs_cutoff = v_date_base.replace(day=1)
            ref_q = pd.Period(actual_v_date, freq='Q')
            
            v_data = build_ragged_vintage(data_full, df_cal, indicator_col, vintage_cols, actual_v_date, obs_cutoff).dropna(axis=1, how='all')
            end_m = v_data.drop(columns=[target_var])
            end_q = data_full[[target_var]]
            
            model = DynamicFactorMQ(endog=end_m, endog_quarterly=end_q, k_factors=1, factor_orders=1, idiosyncratic_ar=1, standardize=True)
            res = model.fit(method='em', maxiter=1000, tolerance=1e-5, disp=False)
            means = res.get_prediction(end=res.model.nobs + 24).predicted_mean
            
            results_table.append({
                'Day Prediction': actual_v_date,
                'Reference Quarter': ref_q.strftime('%YQ%q'),
                'Backcast': get_prediction_value(means, target_var, ref_q - 1),
                'Nowcast': get_prediction_value(means, target_var, ref_q),
                'Forecast': get_prediction_value(means, target_var, ref_q + 1),
                '2-step': get_prediction_value(means, target_var, ref_q + 2),
                '3-step': get_prediction_value(means, target_var, ref_q + 3),
                'Annual Nowcast': calculate_annual_nowcast(means, target_var, actual_v_date)
            })

        return pd.DataFrame(results_table)
    except Exception as e:
        st.error(f"Error Replikasi DFM: {e}")
        return pd.DataFrame()

# ==========================================
# 4. EXECUTION DASHBOARD PDB
# ==========================================
if df_target is not None:
    t_2025 = df_target[df_target['Tahun'] == 2025]['Target'].values[0]
    row_2025 = df_triwulan[df_triwulan['Tahun'] == 2025].iloc[0]

    real_2025, now_2025, combined_2025 = [], [], []
    for q in ['Q1', 'Q2', 'Q3', 'Q4']:
        r = row_2025.get(f'Realisasi {q}', np.nan)
        n = row_2025.get(f'Nowcasting {q}', np.nan)
        real_2025.append(r if pd.notna(r) else None)
        now_2025.append(n if pd.notna(n) else None)
        val = r if pd.notna(r) else (n if pd.notna(n) else 5.0)
        combined_2025.append(val)

    t_2026 = df_target[df_target['Tahun'] == 2026]['Target'].values[0] if 2026 in df_target['Tahun'].values else 5.4
    
    # EKSEKUSI REPLIKASI FULL DFM (Hasilnya Tepat 4 Baris: Q1-Q4 2026)
    df_full_results = run_full_dfm_replication()
    
    if not df_full_results.empty:
        preds_2026 = []
        # Memaksa AI hanya mengambil HANYA nilai NOWCAST secara berurutan (Q1, Q2, Q3, Q4)
        for q_str in ['2026Q1', '2026Q2', '2026Q3', '2026Q4']:
            row = df_full_results[df_full_results['Reference Quarter'] == q_str]
            if not row.empty:
                preds_2026.append(row.iloc[0]['Nowcast'])
            else:
                preds_2026.append(np.nan)
        
        # Jaga-jaga jika ada kuartal yang kosong di kalender
        s = pd.Series(preds_2026)
        s = s.ffill().bfill().fillna(5.2)
        preds_2026 = s.tolist()
    else:
        preds_2026 = [5.1, 5.2, 5.3, 5.4]

    real_2026 = [None, None, None, None]
    now_2026 = preds_2026

    st.sidebar.header("⚙️ Control Panel")
    selected_view = st.sidebar.selectbox("Pilih Periode Monitoring", ["2025", "2026", "Full Trajectory"])

    final_x, final_real, final_now, final_target = [], [], [], []
    current_avg, current_target = 0, 0

    if selected_view == "2025":
        final_x = ['Q1', 'Q2', 'Q3', 'Q4']
        final_real, final_now, final_target = real_2025, now_2025, [t_2025]*4
        vals = [v for v in [r if r is not None else n for r, n in zip(real_2025, now_2025)] if v is not None]
        current_avg, current_target = np.mean(vals) if vals else 0, t_2025
    elif selected_view == "2026":
        final_x = ['Q1', 'Q2', 'Q3', 'Q4']
        final_real, final_now, final_target = real_2026, now_2026, [t_2026]*4
        current_avg, current_target = np.mean(now_2026), t_2026
    else: 
        df_h = df_hist_gdp.copy()
        try:
            if pd.api.types.is_numeric_dtype(df_h.iloc[:, 0]): df_h.iloc[:, 0] = pd.to_datetime(df_h.iloc[:, 0], unit='D', origin='1899-12-30')
            else: df_h.iloc[:, 0] = pd.to_datetime(df_h.iloc[:, 0])
        except: pass
        df_h.set_index(df_h.columns[0], inplace=True)
        col_target = 'RGDP_growth' if 'RGDP_growth' in df_h.columns else df_h.columns[1]
        series_hist = df_h[col_target].dropna()
        series_hist = series_hist[series_hist.index >= '2010-01-01']
        
        try: x_hist = [f"{d.year}-Q{(d.month-1)//3 + 1}" for d in series_hist.index]
        except: x_hist = [str(i) for i in range(len(series_hist))]
            
        y_hist = series_hist.values.tolist()
        x_2025 = ['2025-Q1', '2025-Q2', '2025-Q3', '2025-Q4']
        x_2026 = ['2026-Q1', '2026-Q2', '2026-Q3', '2026-Q4']
        
        full_x_real, full_y_real = x_hist + x_2025, y_hist + combined_2025
        full_x_proj, full_y_proj = [x_2025[-1]] + x_2026, [combined_2025[-1]] + preds_2026
        current_avg, current_target = np.mean(preds_2026), t_2026

    # UPDATE JUDUL CHART DFM
    title_text = f"Outlook Ekonomi: {selected_view}"
    if selected_view == "2026": title_text += " (Model: Dynamic Factor MQ)"
    elif selected_view == "Full Trajectory": title_text = "Historis & Proyeksi Ekonomi (DFM Model)"

    st.markdown(f"### {title_text}")
    fig = go.Figure()

    if selected_view == "Full Trajectory":
        fig.add_trace(go.Scatter(x=full_x_real, y=full_y_real, name='Realisasi (2010-2025)', mode='lines', line=dict(color='#f1c40f', width=2.5)))
        fig.add_trace(go.Scatter(x=full_x_proj, y=full_y_proj, name='Proyeksi DFM 2026', mode='lines', line=dict(color='#27ae60', width=2.5, dash='dot')))
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', legend=dict(orientation="h", y=1.1), height=450)
    else:
        fig.add_trace(go.Bar(x=final_x, y=final_real, name='Realisasi (BPS)', marker_color='#2980b9', text=[f"{v:.2f}%" if v else "" for v in final_real], textposition='auto'))
        if selected_view == "2026":
            fig.add_trace(go.Scatter(x=final_x, y=final_now, name='DFM Nowcasting', mode='lines+markers', line=dict(color='#f39c12', width=4, shape='spline'), text=[f"{v:.2f}%" for v in final_now], textposition='top center'))
        else:
            fig.add_trace(go.Bar(x=final_x, y=final_now, name='Nowcasting', marker_color='#f39c12', text=[f"{v:.2f}%" if v else "" for v in final_now], textposition='auto'))
        fig.add_trace(go.Scatter(x=final_x, y=final_target, name='Target APBN', mode='lines', line=dict(color='#c0392b', width=3, dash='dash')))
        fig.update_layout(barmode='group', plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', legend=dict(orientation="h", y=1.1), height=450)

    c1, c2, c3 = st.columns(3)
    c1.metric("Target Acuan", f"{current_target}%")
    gap = current_avg - current_target
    c2.metric("Realisasi/Proyeksi Avg", f"{current_avg:.2f}%", delta=f"{gap:.2f}%")
    status = "✅ ON TRACK" if gap >= -0.1 else "❌ MELESET / BELOW TARGET"
    c3.metric("Status Capaian", status, delta_color="normal" if gap >= -0.1 else "inverse")

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True)
    
    # --- TOMBOL DOWNLOAD HASIL REPLIKASI (EXCEL) ---
    if not df_full_results.empty:
        import io
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            df_full_results.to_excel(writer, index=False, sheet_name='Nowcast Results')
        
        st.download_button(
            label="📥 Download Full Nowcast Results (Excel)",
            data=buffer.getvalue(),
            file_name="Replikasi_Final_MATLAB_Elaborated.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    st.markdown('</div>', unsafe_allow_html=True)

    # ==========================================
    # --- MONITORING DATA HARIAN (DTD & YTD) ---
    # ==========================================
    st.markdown("### 📈 Monitoring Data Harian")
    
    daily_summary_list = [] # <-- VARIABEL INI WAJIB ADA BIAR AI GAK ERROR
    daily_summary_str = "Data harian tidak tersedia."

    if 'df_daily' in locals() and df_daily is not None:
        daily_cols = st.columns(4)
        daily_indicators = ['IHSG', 'Saham Daily', 'Obligasi Daily', 'Brent', 'WTI', 'CPO', 'Emas', 'Batubara', 'Natural Gas', 'Nikel']
        
        idx = 0
        for col in daily_indicators:
            if col not in df_daily.columns: continue
                
            valid_series = df_daily[[date_col_daily, col]].dropna()
            if valid_series.empty: continue
                
            latest_row = valid_series.iloc[-1]
            val = latest_row[col]
            date_obj = latest_row[date_col_daily]
            date_str = date_obj.strftime("%d %b %Y")
            
            if len(valid_series) > 1:
                prev_row = valid_series.iloc[-2]
                val_prev = prev_row[col]
                dtd = ((val - val_prev) / val_prev) * 100 if val_prev != 0 else 0
            else: dtd = 0
                
            current_year = date_obj.year
            prev_year_data = valid_series[valid_series[date_col_daily].dt.year == current_year - 1]
            
            if not prev_year_data.empty:
                ytd_base_val = prev_year_data.iloc[-1][col]
                ytd = ((val - ytd_base_val) / ytd_base_val) * 100 if ytd_base_val != 0 else 0
                ytd_str = f"YTD: {ytd:+.2f}%"
            else:
                ytd = 0
                ytd_str = "YTD: -"
                
            color_dtd = "badge-red" if dtd < 0 else "badge-green"
            color_ytd = "badge-red" if ytd < 0 else "badge-green"
            disp = f"{val:,.2f}" if val > 10 else f"{val:.2f}"
            
            # --- MASUKKAN KE DALAM DAFTAR UNTUK AI ---
            daily_summary_list.append(f"{col}: {disp} (DTD: {dtd:+.2f}%, {ytd_str})")

            html = f"""
            <div class="glass-card" style="padding: 15px; margin-bottom: 10px;">
                <div class="card-title">{col}</div>
                <div class="card-value">{disp}</div>
                <div style="font-size: 11px; color: #666; margin-bottom: 8px; font-style: italic;">Data: {date_str}</div>
                <span class="badge {color_dtd}">DTD: {dtd:+.2f}%</span>
                <span class="badge {color_ytd}">{ytd_str}</span>
            </div>
            """
            with daily_cols[idx % 4]: st.markdown(html, unsafe_allow_html=True)
            idx += 1
            
        if daily_summary_list:
            daily_summary_str = " | ".join(daily_summary_list)
            
    st.markdown("<br>", unsafe_allow_html=True)

    # ==========================================
    # --- ATURAN GLOBAL WARNA (WAJIB DI SINI) ---
    # ==========================================
    ATURAN_WARNA = {
        'PMI Manufaktur Negara Berkembang': True, 
        'Jumlah Uang Yang Beredar': True, 
        'Penjualan Mobil': True, 
        'Penjualan semen': True, 
        'Ekspor Barang': True, 
        'Impor Barang Modal': True, 
        'Impor Bahan Baku': True, 
        'Kredit Perbankan': True,       # NAIK = HIJAU
        'Penjualan Motor': True,        # NAIK = HIJAU
        'Indeks Keyakinan Konsumen': True, 
        'Impor Barang Konsumsi': True, 
        'Inflasi': False, 
        'Nilai Tukar terhadap Dolar AS': False, 
        'Suku Bunga': False
    }

    # ==========================================
    # --- DEEP DIVE (FIXED YoY & MtM FORMATTING) ---
    # ==========================================
    st.markdown("### 🔍 Deep Dive: Indikator Makro (Real Sector)")
    
    df_makro['Tanggal'] = pd.to_datetime(df_makro['Tanggal'])
    df_makro = df_makro.sort_values(by='Tanggal')
    
    cols = st.columns(4)
    monthly_summary_list = [] # <-- WADAH BARU UNTUK REKAP DATA BULANAN AI
    monthly_summary_str = "Data bulanan tidak tersedia."
    
    indicator_cols = [c for c in df_makro.columns if c != 'Tanggal']

    for i, col in enumerate(indicator_cols):
        valid_series = df_makro[['Tanggal', col]].dropna()
        if valid_series.empty: continue

        latest_row = valid_series.iloc[-1]
        val = latest_row[col]
        date_obj = latest_row['Tanggal']
        date_str = date_obj.strftime("%b %Y")
        
        # Hitung MtM 
        if len(valid_series) > 1:
            prev_row = valid_series.iloc[-2]
            val_prev_mtm = prev_row[col]
            mtm_diff = val - val_prev_mtm
            mtm_pct = (mtm_diff / abs(val_prev_mtm)) * 100 if val_prev_mtm != 0 else 0
        else:
            mtm_diff, mtm_pct = 0, 0

        # Hitung YoY 
        target_date_yoy = date_obj - pd.DateOffset(years=1)
        row_yoy = df_makro[(df_makro['Tanggal'].dt.year == target_date_yoy.year) & (df_makro['Tanggal'].dt.month == target_date_yoy.month)]
        
        if not row_yoy.empty and pd.notna(row_yoy.iloc[0][col]):
            val_yoy = row_yoy.iloc[0][col]
            yoy_diff = val - val_yoy
            yoy_pct = (yoy_diff / abs(val_yoy)) * 100 if val_yoy != 0 else 0
            has_yoy = True
        else:
            yoy_diff, yoy_pct, has_yoy = 0, 0, False

        # --- AMBIL ATURAN DARI ATURAN_WARNA ---
        rule_naik_bagus = ATURAN_WARNA.get(col, True)
        is_level_indicator = any(k in col for k in ["PMI", "Inflasi", "Suku Bunga", "Nilai Tukar", "Indeks Keyakinan Konsumen"])

        # FORMAT TAMPILAN DEEP DIVE
        if is_level_indicator:
            disp = f"{val:,.2f}" if val > 1000 else f"{val:.2f}"
            badge_1 = f"MtM: {mtm_diff:+.2f}"
            badge_2 = f"YoY: {yoy_diff:+.2f}" if has_yoy else "YoY: -"
            
            is_bad_mtm = (rule_naik_bagus and mtm_diff < 0) or (not rule_naik_bagus and mtm_diff > 0)
            is_bad_yoy = (rule_naik_bagus and yoy_diff < 0) or (not rule_naik_bagus and yoy_diff > 0)
        else:
            disp = f"{val:,.2f}"
            badge_1 = f"MtM: {mtm_pct:+.2f}%"
            badge_2 = f"YoY: {yoy_pct:+.2f}%" if has_yoy else "YoY: -"
            
            is_bad_mtm = (rule_naik_bagus and mtm_pct < 0) or (not rule_naik_bagus and mtm_pct > 0)
            is_bad_yoy = (rule_naik_bagus and yoy_pct < 0) or (not rule_naik_bagus and yoy_pct > 0)

        if "PMI" in col and val < 50: 
            is_bad_mtm, is_bad_yoy = True, True
            
        color_1 = "badge-red" if is_bad_mtm else "badge-green"
        color_2 = "badge-red" if is_bad_yoy else "badge-green"
        
        # --- REKAP DATA BULANAN KE DALAM WADAH AI (MtM dan YoY Lengkap) ---
        status_mtm = "Melemah" if is_bad_mtm else "Membaik"
        status_yoy = "Melemah" if is_bad_yoy else "Membaik"
        monthly_summary_list.append(f"[{col}] Data: {disp} | {badge_1} ({status_mtm}) | {badge_2} ({status_yoy})")

        html = f"""
        <div class="glass-card" style="padding: 15px; margin-bottom: 10px;">
            <div class="card-title">{col}</div>
            <div class="card-value">{disp}</div>
            <div style="font-size: 11px; color: #666; margin-bottom: 8px; font-style: italic;">Data: {date_str}</div>
            <span class="badge {color_1}">{badge_1}</span>
            <span class="badge {color_2}">{badge_2}</span>
        </div>
        """
        with cols[i%4]: st.markdown(html, unsafe_allow_html=True)
        
    if monthly_summary_list:
        monthly_summary_str = "\n".join(monthly_summary_list)

   # ==========================================
    # --- HEATMAP BULANAN (YOY TRACKER) ---
    # ==========================================
    st.markdown("### 🗺️ Heatmap Tracker (Tren YoY)")
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    df_hm = df_makro[df_makro['Tanggal'] >= '2025-01-01'].copy()
    
    heatmap_summary_list = [] # <-- WADAH BARU UNTUK REKAP HEATMAP
    heatmap_summary_str = "Data Heatmap tidak tersedia."
    
    if not df_hm.empty:
        dates_hm = df_hm['Tanggal'].tolist()
        x_labels = df_hm['Tanggal'].dt.strftime('%b %Y').tolist()
        z_data, text_data = [], []
        
        for col in indicator_cols:
            col_z, col_text = [], []
            rule_naik_bagus = ATURAN_WARNA.get(col.strip(), True)
            is_level_indicator = any(k in col for k in ["PMI", "Inflasi", "Suku Bunga", "Nilai Tukar", "Indeks Keyakinan Konsumen"])
                
            for d in dates_hm:
                curr_row = df_makro[df_makro['Tanggal'] == d]
                val = curr_row[col].values[0] if not curr_row.empty else np.nan
                
                prev_d = d - pd.DateOffset(years=1)
                prev_row = df_makro[(df_makro['Tanggal'].dt.year == prev_d.year) & (df_makro['Tanggal'].dt.month == prev_d.month)]
                val_prev = prev_row[col].values[0] if not prev_row.empty else np.nan
                
                prev_prev_d = prev_d - pd.DateOffset(years=1)
                prev_prev_row = df_makro[(df_makro['Tanggal'].dt.year == prev_prev_d.year) & (df_makro['Tanggal'].dt.month == prev_prev_d.month)]
                val_prev_prev = prev_prev_row[col].values[0] if not prev_prev_row.empty else np.nan
                
                if pd.isna(val) or pd.isna(val_prev):
                    col_z.append(0) 
                    col_text.append("-")
                else:
                    if is_level_indicator:
                        txt = f"{val:,.2f}" if val > 1000 else f"{val:.2f}"
                        diff = val - val_prev
                    else:
                        yoy_curr = (val - val_prev) / abs(val_prev) * 100 if val_prev != 0 else 0
                        txt = f"{yoy_curr:+.2f}%"
                        if pd.isna(val_prev_prev):
                            diff = yoy_curr
                        else:
                            yoy_prev = (val_prev - val_prev_prev) / abs(val_prev_prev) * 100 if val_prev_prev != 0 else 0
                            diff = yoy_curr - yoy_prev
                        
                    is_green = False
                    if diff == 0: 
                        col_z.append(0) 
                    elif rule_naik_bagus: 
                        is_green = diff > 0
                        col_z.append(1 if is_green else -1)
                    else: 
                        is_green = diff < 0
                        col_z.append(1 if is_green else -1)
                        
                    col_text.append(txt)
                    
                    # Cuma masukkan bulan terbaru ke AI untuk tau sentimen momentum
                    if d == dates_hm[-1]:
                        sentimen = "Positif (Hijau)" if is_green else "Negatif (Merah)"
                        heatmap_summary_list.append(f"{col}: Momentum {sentimen} ({txt})")
            
            z_data.append(col_z)
            text_data.append(col_text)
            
        if heatmap_summary_list:
            heatmap_summary_str = " | ".join(heatmap_summary_list)
            
        fig_hm = go.Figure(data=go.Heatmap(
            z=z_data, x=x_labels, y=indicator_cols, text=text_data,
            texttemplate="<b>%{text}</b>", 
            textfont=dict(size=14, color='#111'),
            colorscale=[[0.0, '#e74c3c'], [0.5, '#ecf0f1'], [1.0, '#2ecc71']], 
            zmin=-1, zmax=1, showscale=False, xgap=3, ygap=3
        ))
        fig_hm.update_layout(
            height=150 + len(indicator_cols)*35,
            margin=dict(l=220, r=20, t=30, b=20),
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            yaxis=dict(autorange="reversed", tickfont=dict(size=12, color='#333', weight='bold')) 
        )
        st.plotly_chart(fig_hm, use_container_width=True)
        st.markdown("<p style='font-size: 11px; color: #666; text-align: center;'>Keterangan Warna: 🟩 Mengalami Perbaikan Momentum (vs Tahun Lalu) | 🟥 Mengalami Perlambatan Momentum | ⬜ Stagnan / Belum Rilis</p>", unsafe_allow_html=True)
    else:
        st.info("Belum ada data bulanan untuk ditampilkan.")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # ==========================================
    # --- AI ADVISOR ---
    # ==========================================
    st.markdown("### 🧠 AI Policy Generator")
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)

    # SIGNATURE MEMBACA DATA BULANAN & HARIAN LENGKAP
    signature = make_signature(selected_view, current_avg, current_target, monthly_summary_str, daily_summary_str)

    if signature in st.session_state.policy_cache:
        st.success("✅ Menggunakan hasil kebijakan sebelumnya (Data Harian & Makro belum berubah)")
        st.markdown(st.session_state.policy_cache[signature])
    else:
        if st.button("Generate Kebijakan Strategis (AI)"):
            genai.configure(api_key=USER_API_KEY)
            with st.spinner('AI sedang mensimulasikan skenario ekonomi dan volatilitas pasar...'):
                try:
                    avail = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
                    model_name = next((m for m in avail if 'flash' in m), avail[0] if avail else None)

                    if not model_name:
                        st.error("Gagal mendeteksi model. Cek API Key atau Region.")
                    else:
                        generation_config = genai.types.GenerationConfig(
                            temperature=0.4, # <-- AI LEBIH KREATIF & OUT-OF-THE-BOX
                            top_p=0.8
                        )
                        model = genai.GenerativeModel(model_name)

                        prompt = f"""
Anda berperan sebagai CHIEF ECONOMIST & AHLI GLOBAL MACRO di Bappenas RI. 
Gaya analisis Anda tajam, melihat *blind-spots*, dan setara dengan analis di *elite hedge fund* internasional.

=====================
KONDISI PDB & PERTUMBUHAN
=====================
Fokus Indikator: {selected_view}
Rata-rata saat ini: {current_avg:.2f}% (Target APBN: {current_target}%)

=====================
DINAMIKA SEKTOR RIIL BULANAN (MtM & YoY)
=====================
Berikut adalah rincian kinerja indikator makro bulanan terakhir:
{monthly_summary_str}

=====================
MOMENTUM BULANAN (HEATMAP YOY)
=====================
Sentimen perbaikan/perlambatan (Heatmap bulan terbaru):
{heatmap_summary_str}

=====================
VOLATILITAS PASAR HARIAN (DTD & YTD)
=====================
{daily_summary_str}

=====================
TUGAS ANALISIS & SINTESIS
=====================
1. ANALISIS PARADOKS & PERSILANGAN: 
   - Kontraskan data jangka pendek (DTD/MtM) dengan data jangka panjang (YTD/YoY).
   - Contoh: Jika komoditas turun secara DTD namun masih naik tajam secara YTD, atau jika ekspor naik MtM namun melambat YoY. Temukan "hidden danger" atau "hidden opportunity" dari persilangan data ini.
2. TRANSMISI KEBIJAKAN: Hubungkan secara logis bagaimana volatilitas pasar harian (IHSG, Nilai Tukar, Komoditas) sedang merembet dan menekan sektor riil bulanan.
3. 5 REKOMENDASI KEBIJAKAN INOVATIF:
   - 2 Kebijakan "Quick Win" (Taktis meredam kepanikan pasar/syok inflasi jangka pendek).
   - 2 Kebijakan Reformasi Struktural (Fokus ke efisiensi/industrialisasi sektor yang merah YoY-nya).
   - 1 Kebijakan Unorthodox / Out-of-the-box (Solusi radikal namun rasional yang mendobrak kebiasaan birokrat konvensional).

=====================
FORMAT WAJIB UNTUK SETIAP KEBIJAKAN
=====================
- Nama Kebijakan: (Actionable dan Tegas)
- Rasionalisasi Macro: (Penjelasan mengapa ini menyelesaikan masalah di Sektor Riil maupun Pasar Harian)
- Dasar Akademis: [Nomor]. Dasar Teori - Penulis (Tahun) - Link Scholar: https://scholar.google.com/scholar?q=kata+kunci
"""

                        res = model.generate_content(prompt, generation_config=generation_config)
                        policy_text = res.text

                        # SIMPAN KE CACHE & FILE
                        st.session_state.policy_cache[signature] = policy_text
                        with open(CACHE_FILE, "wb") as f:
                            pickle.dump(st.session_state.policy_cache, f)

                        st.success(f"Analisis Selesai (Engine: {model_name})")
                        st.markdown(policy_text)

                except Exception as e:
                    st.error(f"Error AI: {e}")

    st.markdown('</div>', unsafe_allow_html=True)
