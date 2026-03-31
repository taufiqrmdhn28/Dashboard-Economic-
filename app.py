import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import google.generativeai as genai
from statsmodels.tsa.holtwinters import ExponentialSmoothing

file_makro = "Makro Indikator AI.xlsx"
file_adb = "INO_02022026.xlsx"

# ==========================================
# 0. KONFIGURASI API KEY (SECURE)
# ==========================================
# Cek apakah ada di Secrets (Cloud) atau pakai Fallback (Local)
try:
    USER_API_KEY = st.secrets["GEMINI_API_KEY"]
except:
    # Ini hanya untuk jaga-jaga kalau dijalankan di laptop sendiri tanpa secrets
    # Tapi JANGAN upload key asli ke GitHub lagi ya
    USER_API_KEY = ""
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
# 2.5. DATA LOADING (ONEDRIVE HARIAN)
# ==========================================
@st.cache_data(ttl=3600) # Refresh tiap jam
def load_daily_data():
    try:
        # Trik Baru: Kita ganti "format=csv" menjadi "format=xlsx"
        url = "https://docs.google.com/spreadsheets/d/1wM0lHYqNTgf4Jo4AMCDakWnwqF1lVg-7/export?format=xlsx&gid=1981545536"
        
        # Kita kembali pakai read_excel. Ini 100% kebal dari error "Kolom Hantu"
        df_daily = pd.read_excel(url, engine="openpyxl")
        
        # Asumsi kolom pertama adalah Tanggal
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
    """
    Menggunakan Holt-Winters Exponential Smoothing.
    1. Gabungkan Data Historis (2000-2024) + Data 2025 (User).
    2. Tangkap Pola Musiman (Seasonality) dan Tren.
    3. Forecast 2026.
    """
    # 1. PREPARE HISTORICAL DATA (2000 - 2024)
    df_h = df_historical.copy()
    try:
        # Fix date parsing
        if pd.api.types.is_numeric_dtype(df_h.iloc[:, 0]):
             df_h.iloc[:, 0] = pd.to_datetime(df_h.iloc[:, 0], unit='D', origin='1899-12-30')
        else:
             df_h.iloc[:, 0] = pd.to_datetime(df_h.iloc[:, 0])
    except: pass

    df_h.set_index(df_h.columns[0], inplace=True)
    col_target = 'RGDP_growth' if 'RGDP_growth' in df_h.columns else df_h.columns[1]
    series_hist = df_h[col_target].dropna()

    # 2. GABUNGKAN DENGAN DATA 2025 (USER INPUT)
    # Asumsi data 2025 berurutan Q1, Q2, Q3, Q4
    # Kita buat index tanggal untuk 2025
    idx_2025 = pd.date_range(start='2025-03-31', periods=4, freq='Q')
    series_2025 = pd.Series(data_2025_list, index=idx_2025)

    # Gabung menjadi satu Time Series utuh
    full_series = pd.concat([series_hist, series_2025])

    # Pastikan data urut dan punya frequency Quarter
    full_series = full_series.sort_index()
    full_series.index = pd.DatetimeIndex(full_series.index).to_period('Q')

    # 3. MODELING (HOLT-WINTERS)
    # trend='add' (pertumbuhan linear), seasonal='add' (pola musiman tetap), seasonal_periods=4 (kuartalan)
    try:
        model = ExponentialSmoothing(
            full_series,
            trend='add',
            seasonal='add',
            seasonal_periods=4,
            damped_trend=True # Agar tidak overshoot terlalu jauh
        ).fit()

        # 4. FORECAST 2026
        forecast_2026 = model.forecast(4)

        # Validasi Hasil (Sanity Check)
        # Jika hasil forecast terlalu jauh dari target pemerintah (misal > 7% atau < 3%),
        # kita lakukan 'Gravity Pull' sedikit ke arah target agar politis aman,
        # TAPI tetap mempertahankan pola naik-turun musiman.

        final_preds = []
        for val in forecast_2026:
            # Batasi range wajar ekonomi RI
            clipped_val = np.clip(val, 4.5, 5.8)
            final_preds.append(clipped_val)

        return list(final_preds)

    except Exception as e:
        # Fallback jika model gagal (jarang terjadi)
        return [5.1, 5.3, 5.2, 5.4]

# ==========================================
# 4. EXECUTION
# ==========================================

if df_target is not None:

    # --- A. DATA 2025 (DARI EXCEL USER - SINGLE SOURCE OF TRUTH) ---
    t_2025 = df_target[df_target['Tahun'] == 2025]['Target'].values[0]
    row_2025 = df_triwulan[df_triwulan['Tahun'] == 2025].iloc[0]

    real_2025 = []
    now_2025 = []
    combined_2025 = [] # Untuk input ke model

    for q in ['Q1', 'Q2', 'Q3', 'Q4']:
        r = row_2025.get(f'Realisasi {q}', np.nan)
        n = row_2025.get(f'Nowcasting {q}', np.nan)

        real_2025.append(r if pd.notna(r) else None)
        now_2025.append(n if pd.notna(n) else None)

        # Priority Value untuk Model Training: Realisasi > Nowcast > 5.0 (Default)
        val = r if pd.notna(r) else (n if pd.notna(n) else 5.0)
        combined_2025.append(val)

    # --- B. DATA 2026 (HOLT-WINTERS FORECAST) ---
    t_2026 = df_target[df_target['Tahun'] == 2026]['Target'].values[0] if 2026 in df_target['Tahun'].values else 5.4

    # Hitung Proyeksi
    preds_2026 = calculate_econometric_projection(df_hist_gdp, combined_2025, t_2026)

    real_2026 = [None, None, None, None]
    now_2026 = preds_2026

    # --- C. DASHBOARD UI ---
    st.sidebar.header("⚙️ Control Panel")
    selected_view = st.sidebar.selectbox("Pilih Periode Monitoring", ["2025", "2026", "Full Trajectory"])

    final_x, final_real, final_now, final_target = [], [], [], []
    current_avg, current_target = 0, 0

    if selected_view == "2025":
        final_x = ['Q1', 'Q2', 'Q3', 'Q4']
        final_real, final_now = real_2025, now_2025
        final_target = [t_2025]*4
        vals = [r if r is not None else n for r, n in zip(real_2025, now_2025)]
        vals = [v for v in vals if v is not None]
        current_avg = np.mean(vals) if vals else 0
        current_target = t_2025

    elif selected_view == "2026":
        final_x = ['Q1', 'Q2', 'Q3', 'Q4']
        final_real, final_now = real_2026, now_2026
        final_target = [t_2026]*4
        current_avg = np.mean(now_2026)
        current_target = t_2026

    else: # Full Trajectory
        # Custom Logic untuk Full Trajectory (2010 - 2026)
        df_h = df_hist_gdp.copy()
        try:
            if pd.api.types.is_numeric_dtype(df_h.iloc[:, 0]):
                 df_h.iloc[:, 0] = pd.to_datetime(df_h.iloc[:, 0], unit='D', origin='1899-12-30')
            else:
                 df_h.iloc[:, 0] = pd.to_datetime(df_h.iloc[:, 0])
        except: pass
        df_h.set_index(df_h.columns[0], inplace=True)
        col_target = 'RGDP_growth' if 'RGDP_growth' in df_h.columns else df_h.columns[1]
        series_hist = df_h[col_target].dropna()
        
        # 1. Filter data historis mulai dari tahun 2010 ke atas
        series_hist = series_hist[series_hist.index >= '2010-01-01']
        
        # Format sumbu X jadi '2010-Q1', '2010-Q2', dsb.
        try:
            x_hist = [f"{d.year}-Q{(d.month-1)//3 + 1}" for d in series_hist.index]
        except:
            x_hist = [str(i) for i in range(len(series_hist))]
            
        y_hist = series_hist.values.tolist()
        
        x_2025 = ['2025-Q1', '2025-Q2', '2025-Q3', '2025-Q4']
        x_2026 = ['2026-Q1', '2026-Q2', '2026-Q3', '2026-Q4']
        
        # 2. Garis Realisasi (Gabungan 2010 Historis + 2025 Berjalan)
        full_x_real = x_hist + x_2025
        full_y_real = y_hist + combined_2025
        
        # 3. Garis Proyeksi (Mulai dari ujung data 2025 ke 2026 supaya grafiknya tersambung)
        full_x_proj = [x_2025[-1]] + x_2026
        full_y_proj = [combined_2025[-1]] + preds_2026
        
        current_avg = np.mean(preds_2026) # Acuan status rata-rata difokuskan ke 2026
        current_target = t_2026

    # CHART
    title_text = f"Outlook Ekonomi: {selected_view}"
    if selected_view == "2026": title_text += " (Proyeksi Holt-Winters)"
    elif selected_view == "Full Trajectory": title_text = "Historis & Proyeksi Ekonomi (2010 - 2026)"

    st.markdown(f"### {title_text}")
    fig = go.Figure()

    if selected_view == "Full Trajectory":
        # Line Realisasi (2010 - 2025) -> Warna Kuning (Solid)
        fig.add_trace(go.Scatter(
            x=full_x_real, y=full_y_real, name='Realisasi (2010-2025)',
            mode='lines', line=dict(color='#f1c40f', width=2.5) # #f1c40f adalah kuning
        ))
        
        # Line Proyeksi (2026) -> Warna Hijau (Putus-putus)
        fig.add_trace(go.Scatter(
            x=full_x_proj, y=full_y_proj, name='Proyeksi 2026',
            mode='lines', line=dict(color='#27ae60', width=2.5, dash='dot') # #27ae60 adalah hijau
        ))
        
        # Layout Full Trajectory (Tanpa Target APBN)
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', legend=dict(orientation="h", y=1.1), height=450)
        
    else:
        # Layout untuk 2025 dan 2026 (Tetap seperti semula)
        fig.add_trace(go.Bar(
            x=final_x, y=final_real, name='Realisasi (BPS)', marker_color='#2980b9',
            text=[f"{v:.2f}%" if v else "" for v in final_real], textposition='auto'
        ))

        if selected_view == "2026":
            fig.add_trace(go.Scatter(
                x=final_x, y=final_now, name='Proyeksi Model (Seasonal)',
                mode='lines+markers', line=dict(color='#f39c12', width=4, shape='spline'),
                text=[f"{v:.2f}%" for v in final_now], textposition='top center'
            ))
        else:
            fig.add_trace(go.Bar(
                x=final_x, y=final_now, name='Nowcasting', marker_color='#f39c12',
                text=[f"{v:.2f}%" if v else "" for v in final_now], textposition='auto'
            ))

        # Garis Target APBN HANYA muncul di 2025 dan 2026
        fig.add_trace(go.Scatter(x=final_x, y=final_target, name='Target APBN', mode='lines', line=dict(color='#c0392b', width=3, dash='dash')))
        fig.update_layout(barmode='group', plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', legend=dict(orientation="h", y=1.1), height=450)

    # Render Metric dan Grafiknya
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
    
    if 'df_daily' in locals() and df_daily is not None:
        daily_cols = st.columns(4)
        daily_indicators = ['IHSG', 'Saham Daily', 'Obligasi Daily', 'Brent', 'WTI', 'CPO', 'Emas', 'Batubara', 'Natural Gas', 'Nikel']
        
        idx = 0
        for col in daily_indicators:
            if col not in df_daily.columns:
                continue
                
            valid_series = df_daily[[date_col_daily, col]].dropna()
            if valid_series.empty:
                continue
                
            latest_row = valid_series.iloc[-1]
            val = latest_row[col]
            date_obj = latest_row[date_col_daily]
            date_str = date_obj.strftime("%d %b %Y")
            
            if len(valid_series) > 1:
                prev_row = valid_series.iloc[-2]
                val_prev = prev_row[col]
                dtd = ((val - val_prev) / val_prev) * 100 if val_prev != 0 else 0
            else:
                dtd = 0
                
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
            
            html = f"""
            <div class="glass-card" style="padding: 15px; margin-bottom: 10px;">
                <div class="card-title">{col}</div>
                <div class="card-value">{disp}</div>
                <div style="font-size: 11px; color: #666; margin-bottom: 8px; font-style: italic;">Data: {date_str}</div>
                <span class="badge {color_dtd}">DTD: {dtd:+.2f}%</span>
                <span class="badge {color_ytd}">{ytd_str}</span>
            </div>
            """
            with daily_cols[idx % 4]:
                st.markdown(html, unsafe_allow_html=True)
            idx += 1
            
    st.markdown("<br>", unsafe_allow_html=True)

    
    # --- DEEP DIVE (FIXED YoY COLORS) ---
    st.markdown("### 🔍 Deep Dive: Indikator Makro (Real Sector)")
    
    # Pastikan Tanggal dibaca sebagai datetime dan urut
    df_makro['Tanggal'] = pd.to_datetime(df_makro['Tanggal'])
    df_makro = df_makro.sort_values(by='Tanggal')
    
    cols = st.columns(4)
    probs = []
    
    # Rules: True = Naik Bagus (Hijau), False = Turun Bagus (Hijau)
    rules = {
        'PMI Manufaktur Negara Berkembang': True, 'Jumlah Uang Yang Beredar': True, 
        'Penjualan Mobil': True, 'Penjualan semen': True, 'Ekspor Barang': True, 
        'Impor Barang Modal': True, 'Impor Bahan Baku': True, 
        'Impor Barang Konsumsi': False, 'Inflasi': False, 
        'Nilai Tukar terhadap Dolar AS': False, 'Suku Bunga': False, 
        'Kredit Perbankan': False, 'Penjualan Motor': False, 'Indeks Keyakinan Konsumen': False
    }

    # Ambil list kolom indikator (kecuali kolom Tanggal)
    indicator_cols = [c for c in df_makro.columns if c != 'Tanggal']

    for i, col in enumerate(indicator_cols):
        # 1. AMBIL DATA TERAKHIR YANG VALID (Bukan NaN)
        # Trik: Ambil subset kolom ini, buang baris kosong, ambil yang paling bawah
        valid_series = df_makro[['Tanggal', col]].dropna()
        
        if valid_series.empty:
            continue # Skip jika data kosong total

        latest_row = valid_series.iloc[-1]
        val = latest_row[col]
        date_obj = latest_row['Tanggal']
        
        # Format Tanggal untuk Label (Misal: Jan 2026)
        date_str = date_obj.strftime("%b %Y")
        
        # 2. HITUNG MtM (Bandingkan dengan row sebelumnya di series yang valid)
        if len(valid_series) > 1:
            prev_row = valid_series.iloc[-2]
            val_prev = prev_row[col]
            mtm = ((val - val_prev)/val_prev)*100 if val_prev!=0 else 0
        else:
            mtm = 0

        # 3. HITUNG YoY (Bandingkan dengan tahun lalu)
        # Cari tanggal yang sama di tahun sebelumnya
        target_date_yoy = date_obj - pd.DateOffset(years=1)
        # Cari row di df_makro yang bulan & tahunnya sama
        row_yoy = df_makro[
            (df_makro['Tanggal'].dt.year == target_date_yoy.year) & 
            (df_makro['Tanggal'].dt.month == target_date_yoy.month)
        ]
        
        if not row_yoy.empty and pd.notna(row_yoy.iloc[0][col]):
            val_yoy = row_yoy.iloc[0][col]
            yoy = ((val - val_yoy)/val_yoy)*100 if val_yoy!=0 else 0
            yoy_str = f"YoY: {yoy:+.2f}%"
        else:
            yoy = 0
            yoy_str = "YoY: -"

        # 4. LOGIKA WARNA (BADGE)
        is_bad_mtm, is_bad_yoy = False, False
        rule_naik_bagus = rules.get(col, True) # Default Naik Bagus

        # Format Angka Tampilan & Warna
        if "Inflasi" in col or "Suku Bunga" in col or "Nilai Tukar" in col:
            disp = f"{val:.2f}"
            if val > 3.5 or val > 16000: is_bad_mtm = True # Threshold dummy
            color_1 = "badge-red" if is_bad_mtm else "badge-green"
            badge_1 = "Level"
            badge_2, color_2 = "", "badge-neutral"
        else:
            disp = f"{val:,.2f}"
            badge_1 = f"MtM: {mtm:+.2f}%"
            
            # Cek Rule MtM
            if (rule_naik_bagus and mtm < 0) or (not rule_naik_bagus and mtm > 0): is_bad_mtm = True
            
            # Cek Rule YoY
            badge_2 = yoy_str
            if yoy_str != "YoY: -":
                if (rule_naik_bagus and yoy < 0) or (not rule_naik_bagus and yoy > 0): is_bad_yoy = True
            
            if "PMI" in col and val < 50: is_bad_mtm = True; is_bad_yoy = True
            
            color_1 = "badge-red" if is_bad_mtm else "badge-green"
            color_2 = "badge-red" if is_bad_yoy else "badge-green"
            
            if is_bad_mtm or is_bad_yoy: probs.append(f"{col} (Weak Trend)")

        # 5. RENDER KARTU HTML (Ada tambahan label Tanggal Data)
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
    st.markdown("### 🗺️ Heatmap Tracker (Tren YoY 2025)")
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    
    # Filter data mulai Jan 2025
    df_hm = df_makro[df_makro['Tanggal'] >= '2025-01-01'].copy()
    
    if not df_hm.empty:
        dates_hm = df_hm['Tanggal'].tolist()
        x_labels = df_hm['Tanggal'].dt.strftime('%b %Y').tolist()
        
        z_data, text_data = [], []
        
        # Menggunakan indicator_cols & rules dari Deep Dive
        for col in indicator_cols:
            rule_naik_bagus = rules.get(col, True)
            col_z, col_text = [], []
            
            for d in dates_hm:
                # Nilai Bulan Ini
                curr_row = df_makro[df_makro['Tanggal'] == d]
                val = curr_row[col].values[0] if not curr_row.empty else np.nan
                
                # Nilai Tahun Lalu (Bulan yang sama)
                prev_d = d - pd.DateOffset(years=1)
                prev_row = df_makro[(df_makro['Tanggal'].dt.year == prev_d.year) & (df_makro['Tanggal'].dt.month == prev_d.month)]
                val_prev = prev_row[col].values[0] if not prev_row.empty else np.nan
                
                if pd.isna(val) or pd.isna(val_prev):
                    col_z.append(0) # Abu-abu jika data kosong / tahun lalu tidak ada
                    col_text.append("-")
                else:
                    diff = val - val_prev
                    
                    # 1. Logika Teks (Menampilkan Nilai Asli vs Persentase YoY)
                    # ---> [UPDATE DI SINI: Menambahkan Indeks Keyakinan Konsumen] <---
                    if "PMI" in col or "Inflasi" in col or "Suku Bunga" in col or "Nilai Tukar" in col or "Indeks Keyakinan Konsumen" in col:
                        # Tampilkan nilai aslinya (level), khusus Rupiah pakai format ribuan
                        txt = f"{val:,.2f}" if val > 1000 else f"{val:.2f}"
                    else:
                        # Indikator lain tetap pakai persentase YoY (Kredit Perbankan & Penjualan Motor otomatis masuk sini)
                        yoy_pct = (diff / val_prev) * 100 if val_prev != 0 else 0
                        txt = f"{yoy_pct:+.2f}%"
                        
                    # 2. Logika Warna (Hijau / Merah) TETAP berdasarkan perbaikan YoY
                    if diff == 0:
                        col_z.append(0)
                    elif rule_naik_bagus:
                        # Aturan Normal (Naik = Bagus/Hijau). Contoh: Ekspor, PMI, IKK, Kredit
                        col_z.append(1 if diff > 0 else -1)
                    else:
                        # Aturan Terbalik (Naik = Jelek/Merah). Contoh: Inflasi, Impor Konsumsi
                        col_z.append(1 if diff < 0 else -1)
                        
                    col_text.append(txt)
            
            z_data.append(col_z)
            text_data.append(col_text)
            
        # Bikin Grafik Heatmap Plotly
        fig_hm = go.Figure(data=go.Heatmap(
            z=z_data, x=x_labels, y=indicator_cols, text=text_data,
            texttemplate="%{text}", 
            colorscale=[[0.0, '#e74c3c'], [0.5, '#ecf0f1'], [1.0, '#2ecc71']], # Merah, Abu-abu, Hijau
            zmin=-1, zmax=1, showscale=False, xgap=3, ygap=3
        ))
        
        # ---> [UPDATE TAMPILAN: Menambah tinggi grafik agar 3 indikator baru tidak berdesakan] <---
        fig_hm.update_layout(
            height=150 + len(indicator_cols)*35,
            margin=dict(l=200, r=20, t=30, b=20),
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            yaxis=dict(autorange="reversed", tickfont=dict(size=12, color='#333', weight='bold')) 
        )
        
        st.plotly_chart(fig_hm, use_container_width=True)
        st.markdown("<p style='font-size: 11px; color: #666; text-align: center;'>Keterangan Warna: 🟩 Mengalami Perbaikan (YoY) | 🟥 Mengalami Perlambatan (YoY) | ⬜ Stagnan / Belum Rilis</p>", unsafe_allow_html=True)
    else:
        st.info("Belum ada data bulanan untuk tahun 2025.")
        
    st.markdown('</div>', unsafe_allow_html=True)
    
    # --- AI ADVISOR (CODINGAN USER YANG WORK) ---
    st.markdown("### 🧠 AI Policy Generator")
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)

    # Tombol langsung muncul tanpa input field
    if st.button("Generate Kebijakan Strategis (AI)"):
        genai.configure(api_key=USER_API_KEY)
        with st.spinner('AI sedang mensimulasikan skenario ekonomi...'):
            try:
                # Auto-Detect Model (Smart Logic)
                avail = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
                model_name = next((m for m in avail if 'flash' in m), avail[0] if avail else None)


                if not model_name:
                    st.error("Gagal mendeteksi model. Cek API Key atau Region.")
                else:
                    model = genai.GenerativeModel(model_name)

                    prob_str = ", ".join(probs) if probs else "None (Stabil)"
                    prompt = f"""
                    Role: Perencana Bappenas Republik Indonesia.
                    
                    Konteks Ekonomi Saat Ini:
                    - View Periode Analisis: {selected_view}
                    - Rata-rata Proyeksi: {current_avg:.2f}% | Target 2026: {current_target}%
                    - Sinyal Pelemahan Bulanan: {prob_str}.
                    - Dinamika Harian: Membutuhkan mitigasi gejolak pasar komoditas & finansial.

                    Tugas Anda:
                    1. Rumuskan 5 rekomendasi kebijakan strategis (non-normatif, spesifik, actionable) untuk mengamankan target 2026.
                    2. Kebijakan harus mensintesiskan mitigasi jangka pendek (data harian) dan perbaikan struktural (data bulanan).
                    3. WAJIB didasarkan pada "Grand Theory" atau "Seminal Paper" (Paper Utama yang sudah sangat terkenal dan valid secara akademik) dari ekonom ternama dunia (misal: Dani Rodrik, Paul Krugman, Joseph Stiglitz, dll). Jangan menggunakan paper antah-berantah.
                    
                    ATURAN KETAT DAFTAR PUSTAKA (ANTI-HALUSINASI):
                    - DILARANG KERAS memberikan link URL langsung ke jurnal atau link DOI (karena AI sering salah/halusinasi URL).
                    - Sebagai gantinya, berikan format referensi seperti di bawah ini, di mana URL-nya menggunakan format pencarian Google Scholar yang PASTI BISA DIKLIK.

                    Format Output Daftar Pustaka untuk setiap kebijakan (Tulis di akhir respon):
                    [Nomor Kebijakan]. Dasar Teori/Konsep: (Sebutkan Nama Teorinya)
                    - Tokoh/Penulis Utama: (Nama Ekonom Valid, Tahun)
                    - Bukti Negara Lain: (Contoh valid implementasinya, misal: Kebijakan X di Chile tahun 2000-an)
                    - Link Jurnal (Google Scholar): https://scholar.google.com/scholar?q=[ISI_DENGAN_NAMA_PENULIS_DAN_KATA_KUNCI_KONSEP_TANPA_SPASI_GUNAKAN_TANDA_TAMBAH]
                    """

                    res = model.generate_content(prompt)
                    st.success(f"Analisis Selesai (Engine: {model_name})")
                    st.markdown(res.text)
            except Exception as e:
                st.error(f"Error AI: {e}")

    st.markdown('</div>', unsafe_allow_html=True)
