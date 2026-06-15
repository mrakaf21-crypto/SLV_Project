"""
============================================================
SLV - SmartLivestock Vision
MAIN APP: dashboard.py (SPOTIFY EDITION - WHATSAPP COMPATIBLE H.264)
============================================================
Jalankan dengan:  streamlit run src/dashboard.py
============================================================
"""

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import json
import os
import sys
import time
import platform
import tempfile
import matplotlib.pyplot as plt
from datetime import datetime

# Tambahkan path supaya import modul lokal bisa jalan
sys.path.insert(0, os.path.dirname(__file__))

from measurement import calculate_uniformity, get_livestock_status
from detection import LivestockDetector

# ── Config Halaman ────────────────────────────────────────
st.set_page_config(
    page_title="SmartLivestock Vision",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS (Spotify Dark Theme & Montserrat Font Fix) ───
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght=400;500;600;700;800&display=swap');

    html, body, [class*="css"], .stMarkdown, p, div, span, h1, h2, h3, h4, h5, h6, button, label, input, select, textarea {
        font-family: 'Montserrat', sans-serif;
    }
    
    [data-testid="stSidebarCollapseButton"] span, 
    .st-emotion-cache-6qob1r, 
    [data-testid="stExpanderToggleIcon"] svg,
    [class*="Icon"] {
        font-family: inherit !important;
    }

    .stApp { 
        background-color: #121212; 
        color: #ffffff;
    }
    section[data-testid="stSidebar"] {
        background-color: #000000 !important;
        border-right: 1px solid #282828;
    }

    .app-header {
        background: linear-gradient(90deg, #181818, #282828);
        border-radius: 8px;
        padding: 24px 32px;
        margin-bottom: 24px;
        border-left: 5px solid #1DB954; /* Spotify Green */
    }
    .app-title {
        color: #FFFFFF;
        font-size: 1.8rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .app-subtitle {
        color: #b3b3b3;
        font-size: 0.9rem;
        margin-top: 6px;
    }

    .stButton > button {
        background-color: #282828 !important;
        color: #ffffff !important;
        border: 1px solid #404040 !important;
        border-radius: 20px !important; 
        padding: 6px 24px !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        transition: all 0.2s ease-in-out !important;
    }
    .stButton > button:hover {
        background-color: #ffffffff !important;
        color: #000000 !important;
        border-color: #ffffffff !important;
        scale: 1.03;
    }
    
    div.stButton > button[kind="primary"] {
        background-color: #1DB954 !important;
        color: #ffffff !important;
        border: none !important;
    }
    div.stButton > button[kind="primary"]:hover {
        background-color: #1ed760 !important;
        color: #ffffff !important;
        scale: 1.03;
        box-shadow: 0 0 12px #1DB954;
    }

    .stSelectbox div[data-baseweb="select"], .stSlider div[data-testid="stSliderRoot"] {
        background-color: #121212 !important;
    }
    
    hr {
        border-color: #282828 !important;
    }

    div[data-testid="stDataFrame"] {
        background-color: #181818 !important;
        border: 1px solid #282828 !important;
        border-radius: 8px !important;
    }

    .stExpander {
        background-color: #181818 !important;
        border: 1px solid #282828 !important;
        border-radius: 8px !important;
    }
</style>
""", unsafe_allow_html=True)

# ── Session State ─────────────────────────────────────────
def init_state():
    defaults = {
        "detector"      : None,
        "camera_active" : False,
        "detections"    : [],
        "history"       : [],       
        "cv_trend"      : [],       
        "frame_count"   : 0,
        "skip_frame"    : 3,        
        "annotated"     : None,
        "cached_img_cv" : None,  
        "cached_img_uni": None   
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ── Load Konfigurasi ──────────────────────────────────────
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")
def load_config():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH) as f:
            return json.load(f)
    return {}

# ── Header ────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <p class="app-title">SmartLivestock Vision (SLV)</p>
    <p class="app-subtitle">
        Sistem Monitoring Dimensi, Bobot & Keseragaman Ternak Berbasis AI - Real-time
    </p>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### Kontrol Sistem")
    st.divider()

    cam_index = st.selectbox("Index Kamera", [0, 1, 2], help="0 = kamera bawaan laptop")
    
    st.markdown("**Resolusi Kamera**")
    cam_res = st.selectbox(
        "Pilih Kualitas",
        options=["VGA (640x480) - Cepat", "HD (1280x720) - Sedang", "FHD (1920x1080) - Berat"],
        index=0,
        label_visibility="collapsed"
    )
    
    if "VGA" in cam_res:
        st.session_state["cam_w"], st.session_state["cam_h"] = 640, 480
    elif "HD" in cam_res:
        st.session_state["cam_w"], st.session_state["cam_h"] = 1280, 720
    else:
        st.session_state["cam_w"], st.session_state["cam_h"] = 1920, 1080

    skip = st.slider("Skip Frame (hemat CPU)", min_value=1, max_value=10, value=3)
    st.session_state["skip_frame"] = skip
    
    st.divider()

    st.markdown("### Tuning AI")
    ai_conf = st.slider(
        "Batas Deteksi (Confidence)", 
        min_value=0.10, 
        max_value=0.95, 
        value=0.60, 
        step=0.05
    )
    st.session_state["ai_conf"] = ai_conf
    
    st.divider()

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("START", width="stretch", type="primary"):
            st.session_state["camera_active"] = True
            st.session_state["cached_img_cv"] = None
            st.session_state["cached_img_uni"] = None
            if st.session_state["detector"] is None:
                with st.spinner("Memuat model AI..."):
                    st.session_state["detector"] = LivestockDetector()
            st.success("Kamera aktif!")

    with col_b:
        if st.button("STOP", width="stretch"):
            st.session_state["camera_active"] = False
            st.info("Kamera dihentikan.")

    st.divider()
    input_mode = st.radio("Mode Input", ["Kamera Live", "Upload Video", "Upload Foto"])

    uploaded_file = None
    if input_mode == "Upload Video":
        uploaded_file = st.file_uploader("Upload Video (.mp4, 'avi')", type=["mp4", "avi", "mov"])
    elif input_mode == "Upload Foto":
        uploaded_file = st.file_uploader("Upload Foto (.jpg, 'png')", type=["jpg", "jpeg", "png"])

    st.divider()
    
    st.markdown("### Kalibrasi Aktif")
    cfg = load_config()
    
    with st.expander("Ubah Parameter Kalibrasi"):
        with st.form("form_kalibrasi"):
            new_mm_px = st.number_input("mm/pixel", value=float(cfg.get('mm_per_pixel', 0.27)), step=0.01)
            new_scale_p = st.number_input("Skala Panjang", value=float(cfg.get('scale_length', 48.38)), step=1.0)
            new_scale_t = st.number_input("Skala Tinggi", value=float(cfg.get('scale_height', 63.15)), step=1.0)
            
            submit_kalibrasi = st.form_submit_button("Simpan Kalibrasi")
            
            if submit_kalibrasi:
                new_cfg = {
                    "mm_per_pixel": new_mm_px,
                    "scale_length": new_scale_p,
                    "scale_height": new_scale_t
                }
                with open(CONFIG_PATH, "w") as f:
                    json.dump(new_cfg, f, indent=4)
                st.success("Tersimpan!")
                time.sleep(0.5)
                st.rerun()

    st.code(f"mm/pixel : {cfg.get('mm_per_pixel', 'N/A')}\nSkala P  : 1 : {cfg.get('scale_length', 'N/A')}\nSkala T  : 1 : {cfg.get('scale_height', 'N/A')}")

    st.divider()
    if st.button("Reset Semua Data", width="stretch"):
        st.session_state["history"] = []
        st.session_state["detections"] = []
        st.session_state["cv_trend"] = []
        st.session_state["cached_img_cv"] = None
        st.session_state["cached_img_uni"] = None
        st.rerun()

# ── LAYOUT UTAMA ──────────────────────────────────────────
st.markdown("#### Live Feed Kamera AI")
video_placeholder = st.empty()
box_status_placeholder = st.empty()

st.divider()
col_list, col_uni = st.columns([2, 3], gap="medium")

with col_list:
    st.markdown("#### Daftar Sapi Terdeteksi Aktual")
    list_placeholder = st.empty()

with col_uni:
    st.markdown("#### Analisis Keseragaman Kelompok")
    uni_placeholder = st.empty()

st.divider()
col_hist, col_chart = st.columns([1, 1], gap="medium")

with col_hist:
    st.markdown("#### Riwayat Deteksi & Spesifikasi Fisik")
    history_placeholder = st.empty()

with col_chart:
    st.markdown("#### Grafik Tren Keseragaman Kandang (CV)")
    chart_placeholder = st.empty()

# ── FUNGSI BACKEND GRAFIK VIDEO ───────────────────────────
def get_baked_chart_image(cv_trend, width_px=300, height_px=110):
    if not cv_trend or len(cv_trend) < 2:
        return None, None
        
    df = pd.DataFrame(cv_trend)
    dpi = 100
    figsize_inch = (width_px / dpi, height_px / dpi)
    
    # 1. Render Grafik Tren CV (%)
    fig_cv, ax_cv = plt.subplots(figsize=figsize_inch, dpi=dpi, facecolor='#181818')
    ax_cv.set_facecolor('#181818')
    ax_cv.plot(df["CV"], color='#ffbb33', linewidth=2.5)
    ax_cv.set_title("TREN CV (%)", color='#FFFFFF', fontsize=7, loc='left', fontweight='bold', pad=4)
    ax_cv.axis('off')
    latest_cv = df["CV"].iloc[-1]
    fig_cv.text(0.65, 0.15, f"CV: {latest_cv}%", color='#ffbb33', fontsize=7, fontweight='bold')
    plt.tight_layout()
    
    fig_cv.canvas.draw()
    img_cv = np.asarray(fig_cv.canvas.buffer_rgba())
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGBA2BGR)
    img_cv = cv2.resize(img_cv, (width_px, height_px))
    plt.close(fig_cv)
    
    # 2. Render Grafik Indeks Keseragaman (%)
    fig_uni, ax_uni = plt.subplots(figsize=figsize_inch, dpi=dpi, facecolor='#181818')
    ax_uni.set_facecolor('#181818')
    ax_uni.plot(df["Uniformity"], color='#1DB954', linewidth=2.5)
    ax_uni.set_title("INDEKS KESERAGAMAN (%)", color='#FFFFFF', fontsize=7, loc='left', fontweight='bold', pad=4)
    ax_uni.axis('off')
    latest_uni = df["Uniformity"].iloc[-1]
    fig_uni.text(0.62, 0.15, f"Uni: {latest_uni}%", color='#1DB954', fontsize=7, fontweight='bold')
    plt.tight_layout()
    
    fig_uni.canvas.draw()
    img_uni = np.asarray(fig_uni.canvas.buffer_rgba())
    img_uni = cv2.cvtColor(img_uni, cv2.COLOR_RGBA2BGR)
    img_uni = cv2.resize(img_uni, (width_px, height_px))
    plt.close(fig_uni)
    
    return img_cv, img_uni

# ── FUNGSI PROSES FRAME ───────────────────────────────────
def process_frame(frame: np.ndarray):
    fc = st.session_state["frame_count"]
    st.session_state["frame_count"] = fc + 1

    if fc % st.session_state["skip_frame"] != 0:
        return st.session_state.get("annotated", frame)

    det = st.session_state["detector"]
    if det is None: return frame

    result = det.detect(frame, selected_id=None)
    st.session_state["detections"] = result["detections"]
    st.session_state["annotated"]  = result["annotated"]

    for d in result["detections"]:
        st.session_state["history"].append({
            "Waktu"      : datetime.now().strftime("%H:%M:%S"),
            "ID Sapi"    : f"#{d['id']}",
            "Panjang(cm)": d["panjang_cm"],
            "Tinggi(cm)" : d["tinggi_cm"],
            "Lingkar(cm)": d["lingkar_cm"],
            "Bobot(kg)"  : d["bobot_kg"],
            "Status"     : d["status"]["label"].replace("✅ ", "").replace("📈 ", "").replace("🌱 ", "").replace("⚠️ ", ""),
        })
        if len(st.session_state["history"]) > 200:
            st.session_state["history"] = st.session_state["history"][-200:]

    return result["annotated"]

# ── FUNGSI UPDATE DASHBOARD ───────────────────────────────
def update_side_panels():
    detections = st.session_state["detections"]
    history    = st.session_state["history"]
    cv_trend   = st.session_state["cv_trend"]
    current_time = datetime.now().strftime("%H:%M:%S")
    unique_ms = int(time.time() * 1000)

    with list_placeholder.container():
        if detections:
            for d in detections:
                clean_status = d['status']['label'].replace("✅ ", "").replace("📈 ", "").replace("🌱 ", "").replace("⚠️ ", "")
                st.markdown(f"Sapi #{d['id']} - `{d['bobot_kg']} kg` - {clean_status}")
        else:
            st.markdown("<p style='color:#b3b3b3;'>Belum ada sapi terdeteksi di meja.</p>", unsafe_allow_html=True)

    weights = [d["bobot_kg"] for d in detections] if detections else []
    with uni_placeholder.container():
        uni = calculate_uniformity(weights)
        clean_uni_status = uni["status"].replace("🟢 ", "").replace("🟡 ", "").replace("🟠 ", "").replace("🔴 ", "")
        
        if uni["count"] < 2:
            st.markdown(
                f'<div style="background-color: #1DB954; color: #000000; padding: 14px 20px; border-radius: 6px; font-weight: 600; font-size: 0.85rem; border: 1px solid #282828;">{clean_uni_status}</div>', 
                unsafe_allow_html=True
            )
        else:
            st.markdown(f"##### Status Kelompok: {clean_uni_status}")
            m_col1, m_col2, m_col3 = st.columns(3)
            with m_col1: st.metric(label="Rata-rata Bobot", value=f"{uni['mean']} kg")
            with m_col2: st.metric(label="Standar Deviasi", value=f"±{uni['std']} kg")
            with m_col3: st.metric(label="Koefisien Variasi (CV)", value=f"{uni['cv']}%")
            st.progress(int(uni["uniformity"]) / 100, text=f"Index Keseragaman: {uni['uniformity']}%")
            st.caption(f"Rentang Data Kelompok: Min {uni['min']} kg | Max {uni['max']} kg ({uni['count']} Sapi Aktif)")

            if not cv_trend or cv_trend[-1]["Waktu"] != current_time:
                cv_trend.append({
                    "Waktu": current_time, 
                    "CV": float(uni["cv"]),
                    "Uniformity": float(uni["uniformity"])
                })
                if len(cv_trend) > 40: cv_trend.pop(0)
                st.session_state["cv_trend"] = cv_trend

    with history_placeholder.container():
        if history:
            df_full = pd.DataFrame(history)
            st.markdown(f"Total Data: `{len(df_full)}` | Terberat: `{df_full['Bobot(kg)'].max()} kg`")
            st.dataframe(df_full[-30:].iloc[::-1], width="stretch", height=160)
            
            c1, c2 = st.columns(2)
            with c1:
                st.download_button("Export CSV", data=df_full.to_csv(index=False), file_name="SLV_data.csv", mime="text/csv", key=f"dl_btn_{unique_ms}", width="stretch")
            with c2:
                if st.button("Hapus Baris Terakhir", key=f"del_btn_{unique_ms}", width="stretch"):
                    if st.session_state["history"]:
                        st.session_state["history"].pop()
                        if st.session_state["cv_trend"]: st.session_state["cv_trend"].pop()
                        st.success("Baris terakhir sukses dihapus!")
                        time.sleep(0.3)
                        st.rerun()
        else:
            st.markdown("<p style='color:#b3b3b3;'>Belum ada riwayat data.</p>", unsafe_allow_html=True)

    with chart_placeholder.container():
        if cv_trend and len(cv_trend) >= 2:
            df_chart = pd.DataFrame(cv_trend)
            st.line_chart(df_chart.set_index("Waktu"), height=210)
        else:
            st.markdown(
                '<div style="background-color: #1DB954; color: #000000; padding: 14px 20px; border-radius: 6px; font-weight: 600; font-size: 0.85rem; border: 1px solid #282828;">Menunggu data terkumpul untuk memplot grafik tren variasi kandang...</div>', 
                unsafe_allow_html=True
            )

# ── MODE EKSEKUSI ─────────────────────────────────────────
if input_mode == "Upload Foto" and uploaded_file:
    if st.session_state["detector"] is None: st.session_state["detector"] = LivestockDetector()
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    annotated = process_frame(frame)
    with video_placeholder: st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), width="stretch")
    update_side_panels()

elif input_mode == "Upload Video" and uploaded_file:
    if st.session_state["detector"] is None: st.session_state["detector"] = LivestockDetector()
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
        tfile.write(uploaded_file.read())
        tfile_path = tfile.name

    cap = cv2.VideoCapture(tfile_path)
    
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    orig_fps = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 20.0
    
    out_dir = tempfile.gettempdir()
    out_video_path = os.path.join(out_dir, f"slv_output_{int(time.time())}.mp4")
    
    # ── PERBAIKAN SAKTI ANTI-BLOKIR WA: Menggunakan ImageIO libx264 ──
    import imageio
    try:
        out_writer = imageio.get_writer(out_video_path, fps=orig_fps, codec='libx264', quality=7)
        use_imageio = True
    except Exception:
        # Fallback darurat jika ada kendala library
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(out_video_path, fourcc, orig_fps, (orig_w, orig_h))
        use_imageio = False
    
    progress_bar = st.empty()
    progress_bar.info("AI sedang melakukan tracking objek dan menjahit grafik ke dalam file video. Silakan tunggu...")
    
    chart_w, chart_h = int(orig_w * 0.28), int(orig_h * 0.18) 
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        current_fc = st.session_state["frame_count"]
        annotated_frame = process_frame(frame)
        update_side_panels()
        
        if current_fc % st.session_state["skip_frame"] == 0 or st.session_state["cached_img_cv"] is None:
            img_cv, img_uni = get_baked_chart_image(st.session_state["cv_trend"], width_px=chart_w, height_px=chart_h)
            st.session_state["cached_img_cv"] = img_cv
            st.session_state["cached_img_uni"] = img_uni
        
        img_cv = st.session_state["cached_img_cv"]
        img_uni = st.session_state["cached_img_uni"]
        
        if img_cv is not None and img_uni is not None:
            margin = 15
            y_offset = orig_h - chart_h - margin
            
            x_offset_cv = margin
            annotated_frame[y_offset:y_offset+chart_h, x_offset_cv:x_offset_cv+chart_w] = img_cv
            
            x_offset_uni = orig_w - chart_w - margin
            annotated_frame[y_offset:y_offset+chart_h, x_offset_uni:x_offset_uni+chart_w] = img_uni
            
        # Tulis frame sesuai dengan mesin pengekspor yang aktif
        if use_imageio:
            # ImageIO membutuhkan skema warna RGB
            frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            out_writer.append_data(frame_rgb)
        else:
            out_writer.write(annotated_frame)
        
        with video_placeholder: 
            st.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), width="stretch")
            
    cap.release()
    if use_imageio:
        out_writer.close()
    else:
        out_writer.release()
        
    progress_bar.empty()
    
    st.success("Proses rendering video selesai!")
    if os.path.exists(out_video_path):
        with open(out_video_path, "rb") as vid_file:
            st.download_button(
                label="Download Hasil Video AI",
                data=vid_file,
                file_name=f"SLV_Analisis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4",
                mime="video/mp4",
                use_container_width=True,
                key=f"instant_dl_v_btn_{int(time.time())}"
            )
            
    try:
        os.unlink(tfile_path)
    except Exception:
        pass

elif input_mode == "Kamera Live":
    if st.session_state["camera_active"]:
        if platform.system() == "Windows":
            cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(cam_index)
            
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, st.session_state.get("cam_w", 640))
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, st.session_state.get("cam_h", 480))
            
            stop_cam = st.button("Stop Kamera", key="stop_cam_live")
            fps_display = box_status_placeholder.empty()

            while st.session_state["camera_active"] and not stop_cam:
                t0 = time.time()
                
                ret, frame = cap.read()
                if not ret: break

                annotated = process_frame(frame)
                with video_placeholder: st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), width="stretch")

                update_side_panels()
                fps = 1 / (time.time() - t0 + 1e-9)
                fps_display.markdown(f"<p style='color:#1DB954; font-size:0.8rem; font-weight:600;'>FPS: {fps:.1f}</p>", unsafe_allow_html=True)
                
                time.sleep(0.001)
                
            cap.release()
            if stop_cam:
                st.session_state["camera_active"] = False
                st.rerun()
    else:
        with video_placeholder:
            st.markdown("""
            <div style="background:#181818; border-radius:8px; padding:80px 20px; text-align:center; border: 1px dashed #282828; min-height:350px; display:flex; flex-direction:column; align-items:center; justify-content:center;">
                <div style="color:#1DB954; font-size:1.3rem; font-weight:700; margin:12px 0; letter-spacing:-0.3px;">SmartLivestock Vision</div>
                <div style="color:#b3b3b3; font-size:0.85rem;">Tekan <b>START</b> di sidebar untuk mengaktifkan kamera</div>
            </div>
            """, unsafe_allow_html=True)
        update_side_panels()