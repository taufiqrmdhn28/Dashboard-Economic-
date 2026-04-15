import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import google.generativeai as genai
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import os
import pickle

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

# ---> FUNGSI INI WAJIB ADA DI ATAS BIAR GAK ERROR NameError <---
def make_signature(view, avg, target, probs, daily_info):
    return f"{view}_{avg:.2f}_{target}_{probs}_{daily_info}"

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
# 3. ROBUST ECONOMETRIC ENGINE (HOLT-WINTERS)
# ==========================================
def calculate_econometric_projection(df_historical, data_2025_list, target_2026):
    df_h = df_historical.copy()
    try:
        if pd.api.types.is_numeric_dtype(df_h.iloc[:, 0]):
             df_h.iloc[:, 0] = pd.to_datetime(df_h.iloc[:, 0], unit='D', origin='1899-12-30')
        else:
             df_h.iloc[:, 0] = pd.to_datetime(df_h.iloc[:, 0])
    except: pass

    df_h.set_index(df_h.columns[0], inplace=True)
    col_target = 'RGDP_growth' if 'RGDP_growth' in df_h.columns else df_h.columns[1]
    series_hist = df_h[col_target].dropna()

    idx_2025 = pd.date_range(start='2025-03-31', periods=4, freq='QE')
    series_2025 = pd.Series(data_2025_list, index=idx_2025)

    full_series = pd.concat([series_hist, series_2025])
    full_series = full_series.sort_index()
    full_series.index = pd.DatetimeIndex(full_series.index).to_period('Q-DEC')

    try:
        model = ExponentialSmoothing(
            full_series, trend='add', seasonal='add', seasonal_periods=4, damped_trend=True
        ).fit()

        forecast_2026 = model.forecast(4)
        final_preds = []
        for val in forecast_2026:
            clipped_val = np.clip(val, 4.5, 5.8)
            final_preds.append(clipped_val)
        return list(final_preds)
    except Exception as e:
        return [5.1, 5.3, 5.2, 5.4]

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
    preds_2026 = calculate_econometric_projection(df_hist_gdp, combined_2025, t_2026)

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

    title_text = f"Outlook Ekonomi: {selected_view}"
    if selected_view == "2026": title_text += " (Proyeksi Holt-Winters)"
    elif selected_view == "Full Trajectory": title_text = "Historis & Proyeksi Ekonomi (2010 - 2026)"

    st.markdown(f"### {title_text}")
    fig = go.Figure()

    if selected_view == "Full Trajectory":
        fig.add_trace(go.Scatter(x=full_x_real, y=full_y_real, name='Realisasi (2010-2025)', mode='lines', line=dict(color='#f1c40f', width=2.5)))
        fig.add_trace(go.Scatter(x=full_x_proj, y=full_y_proj, name='Proyeksi 2026', mode='lines', line=dict(color='#27ae60', width=2.5, dash='dot')))
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', legend=dict(orientation="h", y=1.1), height=450)
    else:
        fig.add_trace(go.Bar(x=final_x, y=final_real, name='Realisasi (BPS)', marker_color='#2980b9', text=[f"{v:.2f}%" if v else "" for v in final_real], textposition='auto'))
        if selected_view == "2026":
            fig.add_trace(go.Scatter(x=final_x, y=final_now, name='Proyeksi Model (Seasonal)', mode='lines+markers', line=dict(color='#f39c12', width=4, shape='spline'), text=[f"{v:.2f}%" for v in final_now], textposition='top center'))
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
    probs = []
    
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
        
        if is_bad_mtm or is_bad_yoy: probs.append(f"{col} (Weak Trend)")

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

   # ==========================================
    # --- HEATMAP BULANAN (YOY TRACKER) ---
    # ==========================================
    st.markdown("### 🗺️ Heatmap Tracker (Tren YoY)")
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    df_hm = df_makro[df_makro['Tanggal'] >= '2025-01-01'].copy()
    
    if not df_hm.empty:
        dates_hm = df_hm['Tanggal'].tolist()
        x_labels = df_hm['Tanggal'].dt.strftime('%b %Y').tolist()
        z_data, text_data = [], []
        
        for col in indicator_cols:
            col_z, col_text = [], []
            
            # AMBIL ATURAN DARI KAMUS SAKTI (Naik=Hijau, Turun=Merah)
            rule_naik_bagus = ATURAN_WARNA.get(col.strip(), True)
            
            is_level_indicator = any(k in col for k in ["PMI", "Inflasi", "Suku Bunga", "Nilai Tukar", "Indeks Keyakinan Konsumen"])
                
            for d in dates_hm:
                curr_row = df_makro[df_makro['Tanggal'] == d]
                val = curr_row[col].values[0] if not curr_row.empty else np.nan
                
                # Tarik Data Tahun Lalu
                prev_d = d - pd.DateOffset(years=1)
                prev_row = df_makro[(df_makro['Tanggal'].dt.year == prev_d.year) & (df_makro['Tanggal'].dt.month == prev_d.month)]
                val_prev = prev_row[col].values[0] if not prev_row.empty else np.nan
                
                # Tarik Data Dua Tahun Lalu (Untuk membandingkan pertumbuhan YoY tahun lalu)
                prev_prev_d = prev_d - pd.DateOffset(years=1)
                prev_prev_row = df_makro[(df_makro['Tanggal'].dt.year == prev_prev_d.year) & (df_makro['Tanggal'].dt.month == prev_prev_d.month)]
                val_prev_prev = prev_prev_row[col].values[0] if not prev_prev_row.empty else np.nan
                
                if pd.isna(val) or pd.isna(val_prev):
                    col_z.append(0) 
                    col_text.append("-")
                else:
                    if is_level_indicator:
                        # 1. INDIKATOR LEVEL (PMI, IKK, dll)
                        # Teks: Nilai Asli
                        txt = f"{val:,.2f}" if val > 1000 else f"{val:.2f}"
                        # Warna: Selisih poin tahun ini vs tahun lalu
                        diff = val - val_prev
                    else:
                        # 2. INDIKATOR PERTUMBUHAN (Ekspor, Mobil, Kredit, Motor, dll)
                        # Teks: Pertumbuhan YoY Tahun Ini
                        yoy_curr = (val - val_prev) / abs(val_prev) * 100 if val_prev != 0 else 0
                        txt = f"{yoy_curr:+.2f}%"
                        
                        # Warna: Selisih Momentum (YoY Tahun Ini dikurangi YoY Tahun Lalu)
                        if pd.isna(val_prev_prev):
                            diff = yoy_curr # Default jika data 2 tahun lalu tidak ada
                        else:
                            yoy_prev = (val_prev - val_prev_prev) / abs(val_prev_prev) * 100 if val_prev_prev != 0 else 0
                            diff = yoy_curr - yoy_prev
                        
                    # EKSEKUSI WARNA MUTLAK
                    if diff == 0: 
                        col_z.append(0) 
                    elif rule_naik_bagus: 
                        col_z.append(1 if diff > 0 else -1) # Perbaikan Momentum -> Hijau, Melambat -> Merah
                    else: 
                        col_z.append(1 if diff < 0 else -1) # Perbaikan Momentum -> Merah (Khusus Inflasi, dll)
                        
                    col_text.append(txt)
            
            z_data.append(col_z)
            text_data.append(col_text)
            
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

    prob_str = ", ".join(probs) if probs else "None (Stabil)"
    
    # AI MENGUNCI DATA HARIAN JUGA SEBAGAI SIDIK JARI
    signature = make_signature(selected_view, current_avg, current_target, prob_str, daily_summary_str)

    # CEK APAKAH SUDAH ADA CACHE
    if signature in st.session_state.policy_cache:
        st.success("✅ Menggunakan hasil kebijakan sebelumnya (data belum berubah)")
        st.markdown(st.session_state.policy_cache[signature])
    else:
        if st.button("Generate Kebijakan Strategis (AI)"):
            genai.configure(api_key=USER_API_KEY)
            with st.spinner('AI sedang mensimulasikan skenario ekonomi...'):
                try:
                    avail = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
                    model_name = next((m for m in avail if 'flash' in m), avail[0] if avail else None)

                    if not model_name:
                        st.error("Gagal mendeteksi model. Cek API Key atau Region.")
                    else:
                        generation_config = genai.types.GenerationConfig(
                            temperature=0.4,
                            top_p=0.8
                        )
                        model = genai.GenerativeModel(model_name)

                        prompt = f"""
Anda berperan sebagai PERENCANA EKONOMI MAKRO BAPPENAS RI berbasis data monitoring.

=====================
KONTEKS DATA PDB & BULANAN
=====================
Indikator: {selected_view}
Rata-rata saat ini: {current_avg:.2f}%
Target pertumbuhan 2026: {current_target}%
Pelemahan YoY Terdeteksi: {prob_str}

=====================
DINAMIKA PASAR HARIAN TERBARU
=====================
{daily_summary_str}

=====================
TUGAS ANALISIS & SINTESIS
=====================
1. ANALISIS PARADOKS: Jangan hanya melihat data harian (DTD). Kontraskan dengan tren tahun berjalan (YTD). (Misal: Jika komoditas turun DTD tapi naik tajam YTD, apa bahaya inflasi tersembunyinya bagi daya beli / indikator makro yang sedang melemah?).
2. TRANSMISI KEBIJAKAN: Hubungkan secara logis bagaimana volatilitas pasar harian ini akan merembet dan memperparah pelemahan di sektor riil (indikator YoY yang merah).
3. 5 REKOMENDASI KEBIJAKAN INOVATIF:
   - 2 Kebijakan "Quick Win" (Taktis untuk meredam syok pasar jangka pendek).
   - 2 Kebijakan Reformasi Struktural (Fokus ke industrialisasi, efisiensi, atau *green economy*).
   - 1 Kebijakan Unorthodox / *Out-of-the-box* (Solusi radikal namun rasional yang jarang dipikirkan birokrat konvensional).

=====================
FORMAT WAJIB
=====================
Untuk setiap kebijakan:
- Masalah ekonomi yang ditangani
- Mekanisme teori
- Rekomendasi kebijakan actionable
- Dampak jangka pendek
- Dampak struktural jangka panjang

Dasar Akademis:
[Nomor]. Dasar Teori
- Penulis (Tahun)
- Link Scholar: https://scholar.google.com/scholar?q=kata+kunci
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
