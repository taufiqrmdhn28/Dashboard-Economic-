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

@st.cache_data(show_spinner="⚙️ DFM Engine: Menghasilkan Histori Prediksi & Sinkronisasi Data Actual...")
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

        data_full = pd.DataFrame(processed_data).replace([np.inf, -np.inf], np.nan).sort_index()
        data_full.index = pd.to_datetime(data_full.index)
        data_full_resampled = data_full.resample('MS').first() 
        target_var = 'RGDP_growth'

        # --- FUNGSI HELPER UNTUK MENGAMBIL NILAI ACTUAL (Excel Only) ---
        def get_actual_value(ref_period):
            # Berdasarkan file INO, data kuartalan ada di bln 3, 6, 9, 12 tanggal 1
            # Contoh: 2023Q1 -> target 2023-03-01
            target_date = ref_period.to_timestamp(how='end').replace(day=1).normalize()
            if target_date in data_full.index:
                val = data_full.loc[target_date, target_var]
                return val if pd.notna(val) else np.nan
            return np.nan

        # 2. Kumpulkan Jadwal Rilis (2023 - 2026)
        jobs = []
        seen = set()
        
        # a. Tentukan Lasteval: Tanggal persis saat dashboard dibuka
        hari_ini = pd.Timestamp.today().normalize()
        
        for vc in vintage_cols:
            col_name = vc.strftime('%Y-%m-%d 00:00:00') if vc.strftime('%Y-%m-%d 00:00:00') in df_cal.columns else df_cal.columns[2 + vintage_cols.index(vc)]
            
            # b. Ambil semua jadwal rilis dari kolom Kalender
            release_dates = pd.to_datetime(df_cal[col_name], errors="coerce").dropna().unique()
            
            for rd in sorted(release_dates):
                # c. LOGIKA PENGEREMAN (Sesuai Maksud Min):
                # Hanya simpan jadwal rilis (rd) yang TANGGALNYA KURANG DARI ATAU SAMA DENGAN HARI INI
                if 2023 <= rd.year <= 2026 and rd <= hari_ini and (rd, vc) not in seen:
                    seen.add((rd, vc))
                    jobs.append((rd, vc)) # Masukkan ke antrean proses MATLAB
                    
        jobs.sort(key=lambda x: x[0])

        # 3. Eksekusi Iterasi
        results_table = []
        for actual_v_date, v_date_base in jobs:
            obs_cutoff = v_date_base.replace(day=1)
            ref_q = pd.Period(actual_v_date, freq='Q')
            
            v_data = build_ragged_vintage(data_full_resampled, df_cal, indicator_col, vintage_cols, actual_v_date, obs_cutoff).dropna(axis=1, how='all')
            end_m = v_data.drop(columns=[target_var], errors='ignore')
            
            q_freq = "QE" if pd.__version__ >= "2.2.0" else "Q"
            if target_var in v_data.columns:
                end_q = v_data[[target_var]].resample(q_freq).last()
            else:
                end_q = data_full_resampled.loc[data_full_resampled.index <= obs_cutoff, [target_var]].resample(q_freq).last()
            
            model = DynamicFactorMQ(endog=end_m, endog_quarterly=end_q, k_factors=1, factor_orders=1, idiosyncratic_ar=1, standardize=True)
            res = model.fit(method='em', maxiter=500, tolerance=1e-5, disp=False)
            means = res.get_prediction(end=res.model.nobs + 24).predicted_mean
            
            results_table.append({
                'Day Prediction': actual_v_date,
                'Reference Quarter': ref_q.strftime('%YQ%q'),
                'Actual': get_actual_value(ref_q), # <--- KOLOM ACTUAL MASUK DISINI
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
    
    # EKSEKUSI DFM
    df_full_results = run_full_dfm_replication()
    
    if not df_full_results.empty:
        preds_2026 = []
        
        # 1. Tentukan "lasteval" (Hari ini / Tanggal rilis data paling update)
        latest_row = df_full_results.sort_values('Day Prediction').iloc[-1]
        ref_q_str = latest_row['Reference Quarter'] # Contoh output: '2026Q2'
        ref_year = int(ref_q_str[:4])
        ref_q_num = int(ref_q_str[-1])
        
        # 2. Petakan horizon proyeksi untuk Q1 - Q4 2026 dari perspektif "lasteval"
        for target_q in [1, 2, 3, 4]:
            # Hitung jarak kuartal target dengan kuartal saat ini (lasteval)
            distance = (2026 - ref_year) * 4 + (target_q - ref_q_num)
            
            # Mapping jarak ke terminologi kolom output DFM / MATLAB
            mapping_kolom = {
                -1: 'Backcast',
                0:  'Nowcast',
                1:  'Forecast',
                2:  '2-step',
                3:  '3-step'
            }
            
            nama_kolom = mapping_kolom.get(distance)
            
            # Jika kolom yang dituju (misal 'Backcast' untuk Q1) ada di baris lasteval, ambil nilainya!
            if nama_kolom and nama_kolom in latest_row and pd.notna(latest_row[nama_kolom]):
                preds_2026.append(float(latest_row[nama_kolom]))
            else:
                # Fallback aman: Jika jarak lewat terlalu jauh (misal Q1 ditanya saat sudah di Q4)
                # Kita ambil hasil Nowcast paling terakhir yang direkam untuk kuartal tersebut
                fallback_df = df_full_results[df_full_results['Reference Quarter'] == f"2026Q{target_q}"]
                if not fallback_df.empty:
                    preds_2026.append(float(fallback_df.sort_values('Day Prediction').iloc[-1]['Nowcast']))
                else:
                    preds_2026.append(np.nan)
        
        # Bersihkan data (ffill/bfill) jika ada kuartal yang blank
        s_preds = pd.Series(preds_2026)
        preds_2026 = s_preds.ffill().bfill().fillna(5.2).tolist()
    else:
        preds_2026 = [5.1, 5.2, 5.3, 5.4]

    real_2026 = [None, None, None, None]
    now_2026 = preds_2026

    # =======================================================
    # JURUS UI: MENYIAPKAN WADAH ATAS UNTUK JUDUL & METRIK
    # =======================================================
    header_ui = st.container()

    # TOMBOL PILIHAN (Sekarang posisinya ada di bawah Metrik, tepat di atas Grafik)
    selected_view = st.radio(
        "Pilih Rentang Waktu Analisis:",
        ["2026", "2010 - 2026"],
        horizontal=True,
        index=0
    )

    final_x, final_real, final_now, final_target = [], [], [], []
    current_avg, current_target = 0, 0

    if selected_view == "2026":
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

    # =======================================================
    # MENGISI WADAH ATAS DENGAN JUDUL DAN ANGKA
    # =======================================================
    title_text = f"Outlook Ekonomi: {selected_view}"
    if selected_view == "2026": title_text += " (Model: Dynamic Factor MQ)"
    else: title_text = "Historis & Proyeksi Ekonomi (DFM Model)"

    with header_ui:
        st.markdown(f"### {title_text}")
        
        # 🔥 AREA INPUT MANUAL REALISASI BPS (c-t-c) 🔥
        # Ubah kata None menjadi angka jika data BPS sudah rilis (Contoh: 5.11)
        realisasi_bps_ctc = None 
        
        # Kita ubah jadi 4 kolom agar muat semua
        c1, c2, c3, c4 = st.columns(4)
        
        # 1. KOTAK TARGET
        c1.metric("Target Acuan", f"{current_target}%")
        
        # 2. KOTAK REALISASI BPS (MANUAL)
        if realisasi_bps_ctc is not None:
            gap_realisasi = realisasi_bps_ctc - current_target
            c2.metric("Realisasi BPS (c-t-c)", f"{realisasi_bps_ctc}%", delta=f"{gap_realisasi:.2f}%")
        else:
            c2.metric("Realisasi BPS (c-t-c)", "Belum Rilis", delta="-", delta_color="off")
            
        # 3. KOTAK PROYEKSI DFM (OTOMATIS)
        gap_proyeksi = current_avg - current_target
        c3.metric("Proyeksi DFM (Avg)", f"{current_avg:.2f}%", delta=f"{gap_proyeksi:.2f}%")
        
        # 4. KOTAK STATUS CAPAIAN
        # Logika Pintar: Jika BPS sudah rilis, status pakai data BPS. Jika belum, pakai data Proyeksi AI.
        angka_acuan_status = realisasi_bps_ctc if realisasi_bps_ctc is not None else current_avg
        gap_status = angka_acuan_status - current_target
        
        status = "✅ SESUAI TARGET" if gap_status >= -0.1 else "❌ BELOW TARGET"
        c4.metric("Status Capaian", status, delta_color="normal" if gap_status >= -0.1 else "inverse")

    # =======================================================
    # MEMBANGUN GRAFIK DI BAWAH TOMBOL
    # =======================================================
    fig = go.Figure()

    if selected_view == "2010 - 2026":
        fig.add_trace(go.Scatter(x=full_x_real, y=full_y_real, name='Realisasi (2010-2025)', mode='lines', line=dict(color='#f1c40f', width=2.5)))
        fig.add_trace(go.Scatter(x=full_x_proj, y=full_y_proj, name='Proyeksi DFM 2026', mode='lines', line=dict(color='#27ae60', width=2.5, dash='dot')))
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', legend=dict(orientation="h", y=1.1), height=450)
    else:
        fig.add_trace(go.Bar(x=final_x, y=final_real, name='Realisasi (BPS)', marker_color='#2980b9', text=[f"{v:.2f}%" if v else "" for v in final_real], textposition='auto'))
        fig.add_trace(go.Scatter(x=final_x, y=final_now, name='DFM Nowcasting', mode='lines+markers', line=dict(color='#f39c12', width=4, shape='spline'), text=[f"{v:.2f}%" for v in final_now], textposition='top center'))
        fig.add_trace(go.Scatter(x=final_x, y=final_target, name='Target APBN', mode='lines', line=dict(color='#c0392b', width=3, dash='dash')))
        fig.update_layout(barmode='group', plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', legend=dict(orientation="h", y=1.1), height=450)

    # =======================================================
    # SUNTIKAN MAGIC PLOTLY: MUNCULKAN ANGKA ATAS-BAWAH (ANTI-DEMPET)
    # =======================================================
    for trace in fig.data:
        trace_name = getattr(trace, 'name', '')
        
        # 1. TRACE REALISASI (Garis Utama Kuning)
        if trace_name == 'Realisasi (2010-2025)':
            text_labels, marker_sizes, text_pos = [], [], []
            if trace.x is not None and trace.y is not None:
                for i, y_val in enumerate(trace.y):
                    # HANYA aktifkan di titik paling terakhir (Q4 2025)
                    if i == len(trace.x) - 1 and pd.notna(y_val): 
                        text_labels.append(f"<b>{float(y_val):.2f}%</b>")
                        marker_sizes.append(10)
                        text_pos.append("top center") # Taruh tegak lurus di atas
                    else:
                        text_labels.append(""); marker_sizes.append(0); text_pos.append("top center")
                        
                trace.mode = "lines+markers+text"
                trace.text = text_labels
                trace.textposition = text_pos 
                trace.textfont = dict(size=13, color="#0f172a") # Font di-bold dan sedikit diperbesar
                
                if not hasattr(trace, 'marker') or trace.marker is None: trace.marker = dict()
                trace.marker.size = marker_sizes
                trace.marker.symbol = "circle"
                trace.marker.color = "#f1c40f" # KUNING/GOLD
                trace.marker.line = dict(width=2, color="white")
                
        # 2. TRACE PROYEKSI (Garis Putus-putus Hijau)
        elif trace_name == 'Proyeksi DFM 2026':
            text_labels, marker_sizes, text_pos = [], [], []
            if trace.x is not None and trace.y is not None:
                # Karena titik Q4 2025 (Kuning) di Atas, titik Q1 2026 (Hijau) kita mulai dari BAWAH
                pos_toggle = True # True = bawah, False = atas
                for x_val, y_val in zip(trace.x, trace.y):
                    if '2026' in str(x_val) and pd.notna(y_val):
                        text_labels.append(f"<b>{float(y_val):.2f}%</b>")
                        marker_sizes.append(10)
                        # Taruh selang-seling murni Atas/Bawah Center
                        text_pos.append("bottom center" if pos_toggle else "top center")
                        pos_toggle = not pos_toggle
                    else:
                        text_labels.append(""); marker_sizes.append(0); text_pos.append("top center")
                        
                trace.mode = "lines+markers+text"
                trace.text = text_labels
                trace.textposition = text_pos 
                trace.textfont = dict(size=13, color="#0f172a")
                
                if not hasattr(trace, 'marker') or trace.marker is None: trace.marker = dict()
                trace.marker.size = marker_sizes
                trace.marker.symbol = "circle"
                trace.marker.color = "#27ae60" # HIJAU
                trace.marker.line = dict(width=2, color="white")
                
        # 3. TRACE NOWCASTING PENDEK (Untuk Tampilan Menu "2026" Saja)
        elif trace_name == 'DFM Nowcasting':
            text_labels, marker_sizes = [], []
            if trace.x is not None and trace.y is not None:
                for y_val in trace.y:
                    if pd.notna(y_val):
                        text_labels.append(f"<b>{float(y_val):.2f}%</b>")
                        marker_sizes.append(11)
                    else:
                        text_labels.append(""); marker_sizes.append(0)
                        
                trace.mode = "lines+markers+text"
                trace.text = text_labels
                trace.textposition = "top center" # Tampilan 2026 jarak antar kuartalnya lebar, aman di atas semua
                trace.textfont = dict(size=14, color="#0f172a")
                
                if not hasattr(trace, 'marker') or trace.marker is None: trace.marker = dict()
                trace.marker.size = marker_sizes
                trace.marker.symbol = "circle"
                trace.marker.color = "#f39c12" # ORANGE
                trace.marker.line = dict(width=2, color="white")
                
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
    
    # 1. TOMBOL PILIHAN MENU UI (Sesuai Arahan Koor)
    selected_daily_view = st.radio(
        "Pilih Mode Tampilan Pasar:",
        ["Data Berjalan", "Data Rata-Rata"],
        horizontal=True,
        key="daily_view_toggle"
    )
    
    # 🔥 WADAH KERANJANG DATA (Kita siapkan 3 sekaligus!)
    daily_summary_list = []  # Untuk dikirim ke otak AI (berubah sesuai UI)
    daily_berjalan_list = [] # Spesial untuk diselundupkan ke Tab "Data Berjalan" di HTML
    daily_rata_list = []     # Spesial untuk diselundupkan ke Tab "Rata-rata" di HTML

    daily_summary_str = "Data harian tidak tersedia."
    daily_berjalan_str = "Data harian tidak tersedia."
    daily_rata_str = "Data harian tidak tersedia."

    if 'df_daily' in locals() and df_daily is not None:
        daily_cols = st.columns(4)
        daily_indicators = ['IHSG', 'Saham Daily', 'Obligasi Daily', 'Brent', 'WTI', 'CPO', 'Emas', 'Batubara', 'Natural Gas', 'Nikel']
        
        idx = 0
        for col in daily_indicators:
            if col not in df_daily.columns: continue
                
            valid_series = df_daily[[date_col_daily, col]].dropna()
            if valid_series.empty: continue
                
            latest_row = valid_series.iloc[-1]
            val = latest_row[col] # Nilai Spot Terakhir
            date_obj = latest_row[date_col_daily]
            date_str = date_obj.strftime("%d %b %Y")
            current_year = date_obj.year
            
            # --- A. PERHITUNGAN DATA BERJALAN ---
            if len(valid_series) > 1:
                prev_row = valid_series.iloc[-2]
                val_prev = prev_row[col]
                dtd = ((val - val_prev) / val_prev) * 100 if val_prev != 0 else 0
            else: dtd = 0
                
            prev_year_data = valid_series[valid_series[date_col_daily].dt.year == current_year - 1]
            if not prev_year_data.empty:
                ytd_base_val = prev_year_data.iloc[-1][col]
                ytd = ((val - ytd_base_val) / ytd_base_val) * 100 if ytd_base_val != 0 else 0
                ytd_str = f"YTD: {ytd:+.2f}%"
            else:
                ytd = 0; ytd_str = "YTD: -"
            
            # --- B. PERHITUNGAN DATA RATA-RATA ---
            current_year_data = valid_series[valid_series[date_col_daily].dt.year == current_year]
            avg_current = current_year_data[col].mean() if not current_year_data.empty else val
            avg_prev = prev_year_data[col].mean() if not prev_year_data.empty else 0
            
            if avg_prev != 0:
                avg_growth = ((avg_current - avg_prev) / avg_prev) * 100  # Persen Perubahan (Δ)
            else:
                avg_growth = 0

            # --- 🔥 RAHASIANYA DI SINI MIN! SIMPAN KE WADAH EXPORT SECARA BERSAMAAN 🔥 ---
            disp_val_b = f"{val:,.2f}" if val > 10 else f"{val:.2f}"
            
            # PERBAIKAN: Masukkan YTD ke dalam kurung
            daily_berjalan_list.append(f"{col}: {disp_val_b} (DTD: {dtd:+.2f}%, {ytd_str})")
            
            disp_val_r = f"{avg_current:,.2f}" if avg_current > 10 else f"{avg_current:.2f}"
            daily_rata_list.append(f"{col}: Avg {current_year} = {disp_val_r} (Perubahan vs Avg 2025: {avg_growth:+.2f}%)")

            # --- C. SWITCH TAMPILAN UI STREAMLIT & DATA UNTUK AI ---
            if "Berjalan" in selected_daily_view:
                disp_val = disp_val_b
                color_1 = "badge-red" if dtd < 0 else "badge-green"
                color_2 = "badge-red" if ytd < 0 else "badge-green"
                badge_1_str = f"DTD: {dtd:+.2f}%"
                badge_2_str = ytd_str
                subtitle_str = f"Data Spot: {date_str}"
                daily_summary_list.append(f"{col}: {disp_val_b} (DTD: {dtd:+.2f}%)")
            else:
                disp_val = disp_val_r
                color_1 = "badge-neutral" 
                color_2 = "badge-red" if avg_growth < 0 else "badge-green"
                avg_prev_disp = f"{avg_prev:,.2f}" if avg_prev > 10 else f"{avg_prev:.2f}"
                badge_1_str = f"Avg '25: {avg_prev_disp}"
                badge_2_str = f"Δ {avg_growth:+.2f}%"
                subtitle_str = f"Rata-rata YTD {current_year}"
                daily_summary_list.append(f"{col}: Avg {current_year} = {disp_val_r} (Perubahan vs Avg 2025: {avg_growth:+.2f}%)")

            # --- D. RENDER KOTAK KACA HTML (Di layar Streamlit) ---
            html = f"""
            <div class="glass-card" style="padding: 15px; margin-bottom: 10px;">
                <div class="card-title">{col}</div>
                <div class="card-value">{disp_val}</div>
                <div style="font-size: 11px; color: #666; margin-bottom: 8px; font-style: italic;">{subtitle_str}</div>
                <span class="badge {color_1}">{badge_1_str}</span>
                <span class="badge {color_2}">{badge_2_str}</span>
            </div>
            """
            with daily_cols[idx % 4]: st.markdown(html, unsafe_allow_html=True)
            idx += 1
            
        # Tutup semua keranjang saat perulangan selesai
        if daily_summary_list: daily_summary_str = " | ".join(daily_summary_list)
        if daily_berjalan_list: daily_berjalan_str = " | ".join(daily_berjalan_list)
        if daily_rata_list: daily_rata_str = " | ".join(daily_rata_list)
            
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
    # --- HEATMAP BULANAN (YOY TRACKER & THRESHOLD) ---
    # ==========================================
    st.markdown("### 🗺️ Heatmap Tracker (Tren YoY & Threshold Target)")
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    df_hm = df_makro[df_makro['Tanggal'] >= '2025-01-01'].copy()
    
    heatmap_summary_list = [] # <-- WADAH UNTUK REKAP HEATMAP AI
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
                    # Penentuan Text Label (Value / %)
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
                    
                    # ==========================================
                    # LOGIKA KHUSUS DARI KOOR (TARGET THRESHOLD)
                    # ==========================================
                    is_special_indicator = False
                    
                    if "PMI" in col:
                        is_special_indicator = True
                        is_green = val >= 50.0
                    elif "Inflasi" in col:
                        is_special_indicator = True
                        is_green = 1.5 <= val <= 3.5
                    elif "Nilai Tukar" in col:
                        is_special_indicator = True
                        is_green = val <= 16900
                        
                    # EKSEKUSI WARNA HEATMAP
                    if is_special_indicator:
                        col_z.append(1 if is_green else -1)
                    else:
                        # Logika Momentum Biasa (Untuk indikator di luar 3 yang spesial)
                        if diff == 0: 
                            col_z.append(0) 
                        elif rule_naik_bagus: 
                            is_green = diff > 0
                            col_z.append(1 if is_green else -1)
                        else: 
                            is_green = diff < 0
                            col_z.append(1 if is_green else -1)
                        
                    col_text.append(txt)
                    
                    # Cuma masukkan bulan terbaru ke AI untuk memicu sentimen yang tepat
                    if d == dates_hm[-1]:
                        sentimen = "Positif/Aman (Hijau)" if is_green else "Negatif/Waspada (Merah)"
                        heatmap_summary_list.append(f"{col}: Kondisi {sentimen} ({txt})")
            
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
        
        # ==========================================
        # KETERANGAN LEGEND BARU YANG LEBIH ELEGAN
        # ==========================================
        st.markdown("""
        <div style='font-size: 11.5px; color: #475569; background: #f8fafc; padding: 12px 15px; border-radius: 10px; border: 1px solid #cbd5e1; line-height: 1.6;'>
            <strong>Keterangan Momentum Umum:</strong> 🟩 Mengalami Perbaikan Momentum (vs Tahun Lalu) | 🟥 Mengalami Perlambatan Momentum | ⬜ Stagnan / Belum Rilis <br>
            <strong>Keterangan Threshold Khusus:</strong> 
            <span style='background:#dcfce7; color:#166534; padding:2px 6px; border-radius:4px;'>🟩 PMI Manufaktur (≥ 50)</span> | 
            <span style='background:#dcfce7; color:#166534; padding:2px 6px; border-radius:4px;'>🟩 Inflasi (1.5% - 3.5%)</span> | 
            <span style='background:#dcfce7; color:#166534; padding:2px 6px; border-radius:4px;'>🟩 Nilai Tukar (< 16.900)</span>
            <br><em>*Khusus 3 indikator di atas, warna merah 🟥 menandakan realisasi keluar dari batas rentang sasaran wajar (Threshold).</em>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("Belum ada data bulanan untuk ditampilkan.")
    st.markdown('</div>', unsafe_allow_html=True)
    
   # ==========================================
    # --- AI ADVISOR & EDITOR (DRAF EDITABLE) ---
    # ==========================================
    st.markdown("### 🧠 AI Policy Generator")
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)

    signature = make_signature(selected_view, current_avg, current_target, monthly_summary_str, daily_summary_str)
    editor_key = f"editor_{signature}"
    
    # Inisialisasi awal variabel agar tidak error
    final_policy_text = ""

    # 1. LOGIKA CACHE & INISIALISASI EDITOR
    if signature in st.session_state.policy_cache:
        # Jika ada di cache tapi belum masuk ke editor session, pindahkan ke editor
        if editor_key not in st.session_state:
            st.session_state[editor_key] = st.session_state.policy_cache[signature]
        
        st.success("✅ Draf tersedia. Silakan lakukan penyesuaian narasi pada kotak di bawah.")

    # 2. TOMBOL GENERATE (Jika belum ada data atau ingin generate ulang)
    if signature not in st.session_state.policy_cache:
        if st.button("Generate Kebijakan Strategis (AI)"):
            genai.configure(api_key=USER_API_KEY)
            with st.spinner('AI sedang menyusun rekomendasi kebijakan teknokratis...'):
                try:
                    # Logika deteksi model sesuai permintaan Min
                    avail = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
                    model_name = next((m for m in avail if 'flash' in m), avail[0] if avail else None)

                    if not model_name: 
                        st.error("Gagal mendeteksi model. Cek API Key.")
                    else:
                        generation_config = genai.types.GenerationConfig(temperature=0.4, top_p=0.8)
                        model = genai.GenerativeModel(model_name)
                        
                        # Prompt asli Min tanpa perubahan
                        prompt = f"""
Anda adalah Perencana Pembangunan Nasional Ahli Utama di Bappenas RI. 
Tugas Anda adalah menyusun Catatan Strategis (Executive Summary) yang ditujukan kepada pimpinan kementerian mengenai prospek ekonomi makro dan arahan kebijakan ke depan.

=====================
GAYA BAHASA WAJIB:
=====================
Gunakan gaya bahasa resmi, baku, dan teknokratis khas dokumen perencanaan pembangunan nasional (seperti narasi KEM RKP, RPJMN, dan RPJPN). Gunakan terminologi perencanaan yang lugas, terstruktur, dan visioner (misalnya: "Arah Kebijakan", "Strategi Pembangunan", "Transformasi Ekonomi", "Hilirisasi Berkelanjutan", "Ketahanan Fundamental", "Penciptaan Nilai Tambah"). Hindari gaya bahasa pasar modal yang berlebihan, ganti dengan narasi ketahanan dan akselerasi ekonomi struktural.

=====================
KONDISI PDB & PERTUMBUHAN
=====================
Fokus Indikator: {selected_view}
Rata-rata Proyeksi DFM saat ini: {current_avg:.2f}% (Target APBN: {current_target}%)

=====================
DINAMIKA SEKTOR RIIL & MOMENTUM (Berdasarkan Data BPS & BI Terkini)
=====================
Ringkasan Bulanan: {monthly_summary_str}
Status Heatmap (Momentum/Threshold): {heatmap_summary_str}

=====================
VOLATILITAS INDIKATOR HARIAN
=====================
{daily_summary_str}

=====================
STRUKTUR OUTPUT DOKUMEN:
=====================
Bagian Utama: ARAH KEBIJAKAN DAN STRATEGI PEMBANGUNAN
Sajikan 5 Rekomendasi Kebijakan dengan format perencanaan pembangunan:
- Arah Kebijakan: (Tegas, berorientasi solusi, bernada RKP/RPJMN)
- Strategi Kebijakan dan Mekanisme Kebijakan: (Penjelasan teknokratis mengapa kebijakan ini krusial untuk mengejar target {current_target}%, dikaitkan dengan realisasi indikator makro saat ini).
- Referensi Akademis: [Nomor]. Dasar Teori - Penulis (Tahun) - Link Scholar: https://scholar.google.com/scholar?q=kata+kunci

*Komposisi Kebijakan:*
- 2 Kebijakan Stabilisasi Jangka Pendek.
- 2 Kebijakan Transformasi Struktural.
- 1 Kebijakan Terobosan yang Inovatif.

---
Bagian Bawah: LAMPIRAN ANALISIS TEKNIS
(Buat pemisah visual, lalu berikan 2 analisis teknis singkat namun mendalam)
- 1. Analisis Dinamika Makroekonomi: (Uraikan perbandingan antara data berjalan dengan data rata-rata).
- 2. Identifikasi Risiko Transmisi Sektor Riil: (Uraikan jalur transmisi bagaimana volatilitas pasar/global saat ini berpotensi berdampak pada sektor manufaktur, daya beli, atau investasi bulanan).
"""
                        res = model.generate_content(prompt, generation_config=generation_config)
                        
                        # Simpan ke cache file dan session state editor
                        st.session_state.policy_cache[signature] = res.text
                        with open(CACHE_FILE, "wb") as f: 
                            pickle.dump(st.session_state.policy_cache, f)
                        
                        st.session_state[editor_key] = res.text
                        st.success(f"Analisis Selesai (Engine: {model_name})")
                        
                        # Refresh halaman agar kotak editor langsung muncul
                        st.rerun()

                except Exception as e: 
                    st.error(f"Error AI: {e}")

    # 3. TAMPILAN EDITOR (Hanya muncul jika sudah ada data)
    if editor_key in st.session_state:
        st.markdown("---")
        # Editor utama yang bisa diketik bebas
        st.session_state[editor_key] = st.text_area(
            "✍️ Ruang Editor Laporan:",
            value=st.session_state[editor_key],
            height=500,
            help="Anda bisa mengubah, menambah, atau menghapus narasi AI di sini sebelum laporan difinalisasi."
        )
        
        # Pratinjau agar tetap terlihat rapi secara visual
        with st.expander("🔍 Pratinjau Hasil Akhir Laporan", expanded=True):
            st.markdown(st.session_state[editor_key])
        
        # Variabel ini yang akan ditarik oleh fitur ekspor/download
        final_policy_text = st.session_state[editor_key]

    st.markdown('</div>', unsafe_allow_html=True)

    # =========================================================
    # FITUR MAGIC: EXPORT KE EXECUTIVE BRIEF (NOTEBOOKLM STYLE)
    # =========================================================
    if final_policy_text:
        st.markdown("<br><hr style='border:1px dashed #ccc;'><br>", unsafe_allow_html=True)
        st.markdown("#### 📑 Export Executive Brief")
        st.caption("Download laporan berformat presentasi eksekutif (HTML Interaktif). Bisa di-Save as PDF saat dibuka.")
        
        try:
            import markdown
            import re
            import copy
            
            # 1. Bypass Kaleido: Penyesuaian Grafik Khusus Export
            fig_export = copy.deepcopy(fig) if 'fig' in locals() else go.Figure()
            
            for trace in fig_export.data:
                trace_type = getattr(trace, 'type', 'scatter')
                if trace_type == 'scatter':
                    if trace.name and "Proyeksi" in trace.name:
                        jml_titik = len(trace.x) if trace.x is not None else len(trace.y)
                        pola_posisi = ['bottom center', 'top center'] * (jml_titik // 2 + 1)
                        trace.textposition = pola_posisi[:jml_titik]
                        trace.textfont = dict(size=9, color='#065f46', weight='bold')
                    elif trace.name and "Realisasi" in trace.name:
                        jml_titik = len(trace.x) if trace.x is not None else len(trace.y)
                        pola_posisi = ['top center'] * jml_titik
                        if jml_titik > 0:
                            pola_posisi[-1] = 'top left'
                        trace.textposition = pola_posisi
                        trace.textfont = dict(size=9, color='#92400e', weight='bold')

            fig_export.update_layout(margin=dict(t=60, b=60, l=30, r=80))
            chart_html = fig_export.to_html(full_html=False, include_plotlyjs='cdn', default_height='450px')
            
            # 2. RENDER TEKS AI 
            html_policy = markdown.markdown(final_policy_text)
            html_policy = html_policy.replace("<ul>", "<ul class='premium-list'>")
            html_policy = html_policy.replace("<li>", "<li>")
            html_policy = html_policy.replace("<h3>", "<h3 class='policy-title'>✨ ")
            html_policy = html_policy.replace("<strong>", "<strong class='highlight-text'>")
            
            # 3. FUNGSI PINTAR: Mengubah teks jadi kotak-kotak HTML
            def parse_to_html_list(data_str, is_market=False):
                if not data_str: return "<p>Data tidak tersedia.</p>"
                clean_data = data_str.replace('\n', ' | ')
                html_list = "<ul class='data-list'>"
                for item in clean_data.split(' | '):
                    item_clean = item.strip()
                    if item_clean and "tidak tersedia" not in item_clean.lower():
                        if is_market:
                            html_list += f"<li><span class='bullet-blue'></span> {item_clean}</li>"
                        else:
                            if "-" in item_clean:
                                html_list += f"<li><span class='badge-red'>▼</span> {item_clean}</li>"
                            else:
                                html_list += f"<li><span class='badge-blue'>▲</span> {item_clean}</li>"
                html_list += "</ul>"
                return html_list if "<li" in html_list else "<p>Data tidak tersedia.</p>"

            # 4. SIAPKAN DATA UNTUK RENDER
            # Sektor Riil (Cukup 1 Tampilan, langsung pakai variabel dari dashboard)
            html_monthly = parse_to_html_list(monthly_summary_str, False)
            
            # Pasar Harian (Butuh 2 Tampilan untuk fitur Tab)
            db_str = locals().get('daily_berjalan_str', daily_summary_str)
            dr_str = locals().get('daily_rata_str', daily_summary_str)
            html_daily_berjalan = parse_to_html_list(db_str, True)
            html_daily_rata = parse_to_html_list(dr_str, True)

            # 5. TEMPLATE HTML PREMIUM
            html_template = f"""
            <!DOCTYPE html>
            <html lang="id">
            <head>
                <meta charset="UTF-8">
                <title>Brief: Macroeconomic Update RI</title>
                <style>
                    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;700;800&display=swap');
                    
                    body {{ font-family: 'Plus Jakarta Sans', sans-serif; background-color: #cbd5e1; color: #334155; padding: 50px 20px; line-height: 1.6; margin: 0; }}
                    .report-container {{ max-width: 1100px; margin: 0 auto; background: #ffffff; border-radius: 20px; box-shadow: 0 25px 50px -12px rgba(0,0,0,0.3); overflow: hidden; }}
                    
                    .header {{ background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 100%); padding: 50px 70px; color: white; }}
                    .header h1 {{ font-size: 40px; font-weight: 800; margin: 0 0 10px 0; color: #ffffff; letter-spacing: -0.5px; line-height: 1.2; }}
                    .header p {{ font-size: 16px; color: #94a3b8; margin: 0; letter-spacing: 0.5px; }}
                    
                    .content-body {{ padding: 40px 70px 60px 70px; }}
                    
                    /* TAMPILAN LABEL & TAB INTERAKTIF */
                    .section-header {{ display: flex; align-items: center; justify-content: space-between; margin: 40px 0 20px 0; }}
                    .section-label {{ font-size: 22px; font-weight: 800; color: #0f172a; display: flex; align-items: center; gap: 12px; margin: 0; }}
                    .section-label span {{ background: #eff6ff; border: 1px solid #bfdbfe; padding: 8px 12px; border-radius: 10px; font-size: 18px; }}
                    
                    .tab-container {{ display: flex; gap: 10px; background: #f8fafc; padding: 6px; border-radius: 12px; border: 1px solid #e2e8f0; }}
                    .tab-btn {{ background: transparent; border: none; padding: 8px 16px; border-radius: 8px; cursor: pointer; font-weight: 700; font-family: inherit; color: #64748b; transition: 0.3s; font-size: 14px; }}
                    .tab-btn:hover {{ background: #e2e8f0; color: #0f172a; }}
                    .tab-btn.active {{ background: #3b82f6; color: white; box-shadow: 0 2px 4px rgba(59,130,246,0.3); }}
                    
                    .chart-wrapper {{ background: #ffffff; border: 1px solid #e2e8f0; border-radius: 16px; padding: 10px; box-shadow: 0 4px 10px rgba(0,0,0,0.03); margin-bottom: 50px; }}
                    .data-list {{ list-style: none; padding: 0; margin: 0; display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; }}
                    .data-list li {{ background: #ffffff; padding: 14px 16px; border-radius: 10px; border: 1px solid #e2e8f0; font-size: 13.5px; color: #334155; display: flex; align-items: center; box-shadow: 0 2px 4px rgba(0,0,0,0.02); font-weight: 600; margin: 0; }}
                    .bullet-blue {{ display: inline-block; width: 10px; height: 10px; background: #3b82f6; border-radius: 50%; flex-shrink: 0; margin-right: 10px; }}
                    .badge-blue {{ background: #dbeafe; color: #1e40af; padding: 4px 8px; border-radius: 6px; font-weight: 800; font-size: 11px; flex-shrink: 0; }}
                    .badge-red {{ background: #fee2e2; color: #991b1b; padding: 4px 8px; border-radius: 6px; font-weight: 800; font-size: 11px; flex-shrink: 0; }}
                    
                    .ai-box {{ background: linear-gradient(145deg, #f8fafc, #eff6ff); border: 1px solid #bfdbfe; border-radius: 20px; padding: 40px; margin-top: 60px; position: relative; }}
                    .ai-box::before {{ content:''; position: absolute; top:0; left:0; width:100%; height:6px; background: linear-gradient(90deg, #2563eb, #9333ea); border-radius: 20px 20px 0 0; }}
                    .policy-title {{ color: #1e3a8a; font-size: 20px; font-weight: 800; border-bottom: 2px dashed #cbd5e1; padding-bottom: 12px; margin-top: 35px; margin-bottom: 20px; }}
                    .premium-list {{ list-style: none; padding: 0; margin: 0; }}
                    .premium-list li {{ background: #ffffff; border: 1px solid #e2e8f0; border-left: 5px solid #3b82f6; padding: 25px 30px; border-radius: 12px; margin-bottom: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.03); font-size: 15.5px; color: #1e293b; line-height: 1.8; }}
                    .highlight-text {{ color: #2563eb; font-weight: 800; }}
                    
                    .footer {{ text-align: center; padding: 30px; margin-top: 50px; color: #94a3b8; font-size: 13px; border-top: 1px solid #e2e8f0; }}
                </style>
                <script>
                    function openTabPH(evt, tabName) {{
                        var i, tabcontent, tablinks;
                        tabcontent = document.getElementsByClassName("tab-content-ph");
                        for (i = 0; i < tabcontent.length; i++) {{ tabcontent[i].style.display = "none"; }}
                        tablinks = document.getElementsByClassName("tab-btn-ph");
                        for (i = 0; i < tablinks.length; i++) {{ tablinks[i].className = tablinks[i].className.replace(" active", ""); }}
                        document.getElementById(tabName).style.display = "block";
                        evt.currentTarget.className += " active";
                    }}
                </script>
            </head>
            <body>
                <div class="report-container">
                    
                    <div class="header">
                        <h1>Macroeconomic Brief</h1>
                        <p>Analisis Perkembangan Ekonomi Makro Bappenas RI</p>
                    </div>

                    <div class="content-body">
                        
                        <div class="section-header">
                            <div class="section-label"><span>📈</span> Proyeksi Pertumbuhan Ekonomi (DFM)</div>
                        </div>
                        <div class="chart-wrapper">
                            {chart_html}
                        </div>

                        <!-- SEKTOR RIIL (TANPA TAB, STATIS SESUAI DASHBOARD) -->
                        <div class="section-header">
                            <div class="section-label"><span>🏢</span> Kinerja Seluruh Sektor Riil</div>
                        </div>
                        {html_monthly}

                        <!-- TAB PASAR HARIAN -->
                        <div class="section-header">
                            <div class="section-label"><span>⚡</span> Volatilitas Pasar Harian</div>
                            <div class="tab-container">
                                <button class="tab-btn tab-btn-ph active" onclick="openTabPH(event, 'ph-berjalan')">Data Berjalan</button>
                                <button class="tab-btn tab-btn-ph" onclick="openTabPH(event, 'ph-rata')">Rata-rata</button>
                            </div>
                        </div>
                        <div id="ph-berjalan" class="tab-content-ph" style="display: block;">{html_daily_berjalan}</div>
                        <div id="ph-rata" class="tab-content-ph" style="display: none;">{html_daily_rata}</div>

                        <div class="ai-box">
                            <div class="section-label" style="margin-top: 0; margin-bottom: 25px; border:none; padding:0;"><span>🧠</span> Rekomendasi Kebijakan</div>
                            {html_policy}
                        </div>
                        
                        <div class="footer">
                            Dokumen ini dihasilkan oleh model AI.<br>
                            Dicetak pada: <strong>{pd.Timestamp.now().strftime('%d %B %Y %H:%M')} WIB</strong>
                        </div>
                    </div>

                </div>
            </body>
            </html>
            """
            
            # 6. Tombol Download HTML
            st.download_button(
                label="📥 Download Laporan Eksekutif (.html)",
                data=html_template,
                file_name="Laporan_Brief_Bappenas.html",
                mime="text/html",
                type="primary"
            )
            
        except Exception as e:
            st.warning(f"Gagal menyiapkan dokumen HTML. Error detail: {e}")
