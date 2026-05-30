import streamlit as st
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import io
import os
import sys
from huggingface_hub import hf_hub_download

sys.path.append(os.path.dirname(__file__))

# ── Model Download from Hugging Face ─────────────────────────────────────────
MODEL_PATH  = "model/best_model_resnet50.pth"
HF_REPO_ID  = "meli9/geoai-disastermapper"
HF_FILENAME = "best_model_resnet50.pth"

@st.cache_resource
def download_model():
    """Download model from Hugging Face if not present locally."""
    if not os.path.exists(MODEL_PATH):
        os.makedirs("model", exist_ok=True)
        with st.spinner("Downloading model from Hugging Face... (452MB — first load only)"):
            hf_hub_download(
                repo_id   = HF_REPO_ID,
                filename  = HF_FILENAME,
                local_dir = "model"
            )
        st.success("Model ready ✓")
    return MODEL_PATH

# Download on startup
download_model()

from utils.inference import (
    load_model, run_inference, get_damage_stats,
    prediction_to_colored_image, export_geojson, LABELS, COLORS
)
from utils.climate import (
    get_climate_data, parse_climate_data,
    calculate_anomaly, plot_climate_trends, get_climate_summary
)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title  = "GeoAI Disaster Mapper",
    page_icon   = "🛰️",
    layout      = "wide",
    initial_sidebar_state = "expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg-primary:   #0a0e1a;
    --bg-secondary: #111827;
    --bg-card:      #1a2035;
    --accent:       #00d4ff;
    --accent-warm:  #ff6b35;
    --text-primary: #e2e8f0;
    --text-muted:   #64748b;
    --border:       #1e293b;
    --green:        #2ecc71;
    --yellow:       #f1c40f;
    --orange:       #e67e22;
    --red:          #e74c3c;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg-primary);
    color: var(--text-primary);
}

h1, h2, h3 {
    font-family: 'Space Mono', monospace !important;
}

.main-header {
    background: linear-gradient(135deg, #0a0e1a 0%, #1a2035 50%, #0f1e3d 100%);
    border-bottom: 1px solid var(--accent);
    padding: 2rem;
    margin-bottom: 2rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}

.main-header::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    background: radial-gradient(ellipse at 50% 0%, rgba(0,212,255,0.1) 0%, transparent 70%);
}

.main-title {
    font-family: 'Space Mono', monospace;
    font-size: 2.5rem;
    font-weight: 700;
    color: var(--accent);
    letter-spacing: -1px;
    margin: 0;
    text-shadow: 0 0 30px rgba(0,212,255,0.3);
}

.main-subtitle {
    color: var(--text-muted);
    font-size: 1rem;
    margin-top: 0.5rem;
    letter-spacing: 2px;
    text-transform: uppercase;
}

.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
    transition: border-color 0.2s;
}

.metric-card:hover {
    border-color: var(--accent);
}

.metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 1.8rem;
    font-weight: 700;
    color: var(--accent);
}

.metric-label {
    font-size: 0.75rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 0.3rem;
}

.damage-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
    margin: 2px;
}

.section-header {
    font-family: 'Space Mono', monospace;
    font-size: 1rem;
    color: var(--accent);
    text-transform: uppercase;
    letter-spacing: 2px;
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.5rem;
    margin-bottom: 1rem;
}

.info-box {
    background: rgba(0,212,255,0.05);
    border: 1px solid rgba(0,212,255,0.2);
    border-radius: 8px;
    padding: 1rem;
    margin: 1rem 0;
    font-size: 0.9rem;
}

.warning-box {
    background: rgba(255,107,53,0.05);
    border: 1px solid rgba(255,107,53,0.2);
    border-radius: 8px;
    padding: 1rem;
    margin: 1rem 0;
    font-size: 0.9rem;
}

.stButton > button {
    background: linear-gradient(135deg, var(--accent), #0099cc);
    color: #0a0e1a;
    font-family: 'Space Mono', monospace;
    font-weight: 700;
    border: none;
    border-radius: 8px;
    padding: 0.6rem 2rem;
    letter-spacing: 1px;
    transition: all 0.2s;
    width: 100%;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(0,212,255,0.3);
}

[data-testid="stSidebar"] {
    background: var(--bg-secondary);
    border-right: 1px solid var(--border);
}

.stFileUploader {
    background: var(--bg-card);
    border: 2px dashed var(--border);
    border-radius: 12px;
    padding: 1rem;
}

div[data-testid="stMetric"] {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem;
}
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <p class="main-title">🛰️ GeoAI DisasterMapper</p>
    <p class="main-subtitle">Automated Building Damage Detection & Climate Vulnerability Analysis</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="section-header">⚙️ Configuration</p>', unsafe_allow_html=True)

    # Model path — auto set from HF download
    model_path = MODEL_PATH
    st.markdown(f"""
    <div style='background:#1a2035; border:1px solid #00d4ff33;
                border-radius:8px; padding:0.8rem; font-size:0.8rem;'>
        <b style='color:#00d4ff;'>Model</b><br>
        U-Net ResNet50<br>
        <span style='color:#64748b;'>Auto-loaded from Hugging Face</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<p class="section-header">📍 Location</p>', unsafe_allow_html=True)

    # Disaster presets
    disaster_presets = {
        "Custom"                  : (0.0, 0.0, None),
        "Morocco Earthquake 2023" : (31.07, -8.41, "2023-09-08"),
        "Turkey Earthquake 2023"  : (37.17, 37.03, "2023-02-06"),
        "Nepal Earthquake 2015"   : (27.73, 85.33, "2015-04-25"),
        "Palu Tsunami 2018"       : (-0.90, 119.88, "2018-09-28"),
        "Hurricane Michael 2018"  : (30.19, -85.68, "2018-10-10"),
    }

    selected_preset = st.selectbox("Disaster Preset", list(disaster_presets.keys()))
    preset_lat, preset_lon, preset_date = disaster_presets[selected_preset]

    col1, col2 = st.columns(2)
    with col1:
        lat = st.number_input("Latitude",  value=preset_lat, format="%.4f")
    with col2:
        lon = st.number_input("Longitude", value=preset_lon, format="%.4f")

    disaster_date = st.text_input(
        "Disaster Date (YYYY-MM-DD)",
        value=preset_date if preset_date else ""
    )

    st.markdown("---")
    st.markdown('<p class="section-header">🌡️ Climate Settings</p>', unsafe_allow_html=True)

    start_year = st.slider("Climate Data Start Year", 2000, 2020, 2010)
    end_year   = st.slider("Climate Data End Year",   2015, 2024, 2023)

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.75rem; color:#64748b; line-height:1.6;'>
    <b style='color:#00d4ff;'>About</b><br>
    U-Net ResNet50 trained on xBD dataset<br>
    Macro F1: 0.76 | 5 damage classes<br>
    Climate data: NASA POWER API<br><br>
    <b style='color:#00d4ff;'>Project</b><br>
    GeoAI + Climate Change Research<br>
    2015 Gorkha Earthquake, Nepal
    </div>
    """, unsafe_allow_html=True)

# ── Main content ──────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🛰️ Damage Detection", "🌡️ Climate Analysis", "📊 Model Info"])

# ── Tab 1: Damage Detection ───────────────────────────────────────────────────
with tab1:
    st.markdown('<p class="section-header">Upload Satellite Images</p>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Before Disaster**")
        pre_file = st.file_uploader(
            "Upload pre-disaster image",
            type=['png', 'jpg', 'jpeg', 'tif', 'tiff'],
            key="pre"
        )
        if pre_file:
            pre_image = Image.open(pre_file)
            st.image(pre_image, caption="Before Disaster", use_column_width=True)

    with col2:
        st.markdown("**After Disaster**")
        post_file = st.file_uploader(
            "Upload post-disaster image",
            type=['png', 'jpg', 'jpeg', 'tif', 'tiff'],
            key="post"
        )
        if post_file:
            post_image = Image.open(post_file)
            st.image(post_image, caption="After Disaster", use_column_width=True)

    st.markdown("---")

    if pre_file and post_file:
        if st.button("🔍 RUN DAMAGE DETECTION"):
            with st.spinner("Loading model and running inference..."):
                try:
                    # Load model
                    if not os.path.exists(model_path):
                        st.error(f"Model not found at: {model_path}")
                        st.stop()

                    device = 'cuda' if torch.cuda.is_available() else 'cpu'
                    model  = load_model(model_path, device)

                    # Run inference
                    pre_image  = Image.open(pre_file)
                    post_image = Image.open(post_file)
                    pred       = run_inference(model, pre_image, post_image, device)

                    # Get stats
                    stats = get_damage_stats(pred)

                    st.success("Inference complete ✓")

                    # ── Damage metrics ────────────────────────────────────
                    st.markdown('<p class="section-header">Damage Statistics</p>',
                                unsafe_allow_html=True)

                    metric_cols = st.columns(5)
                    damage_colors_hex = {
                        'background' : '#64748b',
                        'no-damage'  : '#2ecc71',
                        'minor'      : '#f1c40f',
                        'major'      : '#e67e22',
                        'destroyed'  : '#e74c3c'
                    }

                    for i, label in enumerate(LABELS):
                        with metric_cols[i]:
                            pct = stats.get(label, {}).get('percent', 0)
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-value" style="color:{damage_colors_hex[label]}">
                                    {pct:.1f}%
                                </div>
                                <div class="metric-label">{label}</div>
                            </div>
                            """, unsafe_allow_html=True)

                    # ── Damage map visualization ──────────────────────────
                    st.markdown('<p class="section-header">Damage Map</p>',
                                unsafe_allow_html=True)

                    colored_pred = prediction_to_colored_image(pred)

                    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                    fig.patch.set_facecolor('#0a0e1a')

                    for ax in axes:
                        ax.set_facecolor('#0a0e1a')
                        ax.axis('off')

                    axes[0].imshow(np.array(pre_image.convert('RGB')))
                    axes[0].set_title('Before Disaster',
                                      color='white', fontsize=12, pad=10)

                    axes[1].imshow(np.array(post_image.convert('RGB')))
                    axes[1].set_title('After Disaster',
                                      color='white', fontsize=12, pad=10)

                    axes[2].imshow(np.array(post_image.convert('RGB')))
                    axes[2].imshow(colored_pred, alpha=0.6)
                    axes[2].set_title('Damage Prediction Overlay',
                                      color='white', fontsize=12, pad=10)

                    # Legend
                    patches = [
                        mpatches.Patch(color=COLORS[i], label=LABELS[i])
                        for i in range(len(LABELS))
                    ]
                    fig.legend(
                        handles=patches,
                        loc='lower center',
                        ncol=5,
                        facecolor='#1a2035',
                        labelcolor='white',
                        fontsize=10,
                        bbox_to_anchor=(0.5, -0.05)
                    )

                    plt.tight_layout()

                    buf = io.BytesIO()
                    plt.savefig(buf, format='png', dpi=150,
                                bbox_inches='tight', facecolor='#0a0e1a')
                    buf.seek(0)
                    plt.close()

                    st.image(buf, use_column_width=True)

                    # ── GeoJSON export ────────────────────────────────────
                    st.markdown('<p class="section-header">Export Results</p>',
                                unsafe_allow_html=True)

                    geojson_path = export_geojson(pred)

                    if geojson_path:
                        with open(geojson_path, 'r') as f:
                            geojson_data = f.read()

                        st.download_button(
                            label     = "⬇️ Download GeoJSON",
                            data      = geojson_data,
                            file_name = "damage_map.geojson",
                            mime      = "application/json"
                        )
                        st.markdown("""
                        <div class="info-box">
                        📌 Open the GeoJSON file in QGIS or ArcGIS to view
                        damage polygons with exact coordinates on a map.
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.info("No significant damage detected in uploaded images.")

                except Exception as e:
                    st.error(f"Error during inference: {str(e)}")
    else:
        st.markdown("""
        <div class="info-box">
        👆 Upload both <b>before</b> and <b>after</b> disaster satellite images
        to run damage detection. Use the xBD test images or any high-resolution
        satellite image pair.
        </div>
        """, unsafe_allow_html=True)

# ── Tab 2: Climate Analysis ───────────────────────────────────────────────────
with tab2:
    st.markdown('<p class="section-header">Climate Vulnerability Analysis</p>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
    Fetches temperature, rainfall and humidity data from
    <b>NASA POWER API</b> for the selected disaster location.
    Analyzes climate trends to assess long-term vulnerability.
    </div>
    """, unsafe_allow_html=True)

    if lat == 0.0 and lon == 0.0:
        st.warning("Please set latitude and longitude in the sidebar first.")
    else:
        st.markdown(f"**Location:** {lat}°N, {lon}°E | **Period:** {start_year}–{end_year}")

        if st.button("🌡️ FETCH CLIMATE DATA"):
            with st.spinner("Fetching data from NASA POWER API..."):
                try:
                    raw_data = get_climate_data(lat, lon, start_year, end_year)

                    if raw_data is None:
                        st.error("Failed to fetch climate data. Check your internet connection.")
                        st.stop()

                    df      = parse_climate_data(raw_data)
                    df      = calculate_anomaly(df)
                    summary = get_climate_summary(
                        df,
                        disaster_date if disaster_date else None
                    )

                    # ── Climate metrics ───────────────────────────────────
                    st.markdown('<p class="section-header">Climate Summary</p>',
                                unsafe_allow_html=True)

                    mcols = st.columns(4)

                    with mcols[0]:
                        st.metric(
                            "Avg Temperature",
                            f"{summary['avg_temp']}°C"
                        )
                    with mcols[1]:
                        st.metric(
                            "Max Temperature",
                            f"{summary['max_temp']}°C"
                        )
                    with mcols[2]:
                        st.metric(
                            "Avg Rainfall",
                            f"{summary['avg_rainfall']} mm/day"
                        )
                    with mcols[3]:
                        warming = summary.get('warming_rate', 0) or 0
                        st.metric(
                            "Warming Rate",
                            f"{warming:+.3f}°C/year",
                            delta=f"{'↑ Warming' if warming > 0 else '↓ Cooling'}"
                        )

                    # Pre vs post disaster
                    if disaster_date and 'temp_change' in summary:
                        st.markdown('<p class="section-header">Pre vs Post Disaster Climate</p>',
                                    unsafe_allow_html=True)

                        pcols = st.columns(3)
                        with pcols[0]:
                            st.metric(
                                "Pre-disaster Avg Temp",
                                f"{summary['pre_disaster_avg_temp']}°C"
                            )
                        with pcols[1]:
                            st.metric(
                                "Post-disaster Avg Temp",
                                f"{summary['post_disaster_avg_temp']}°C"
                            )
                        with pcols[2]:
                            change = summary['temp_change']
                            st.metric(
                                "Temperature Change",
                                f"{change:+.2f}°C",
                                delta=f"{'Warmer' if change > 0 else 'Cooler'} after disaster"
                            )

                    # ── Climate charts ────────────────────────────────────
                    st.markdown('<p class="section-header">Climate Trends</p>',
                                unsafe_allow_html=True)

                    location_name = selected_preset if selected_preset != "Custom" \
                        else f"{lat}°N, {lon}°E"

                    chart_buf = plot_climate_trends(
                        df,
                        location_name,
                        disaster_date if disaster_date else None
                    )
                    st.image(chart_buf, use_column_width=True)

                    # Download CSV
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label     = "⬇️ Download Climate Data (CSV)",
                        data      = csv,
                        file_name = "climate_data.csv",
                        mime      = "text/csv"
                    )

                    # Vulnerability assessment
                    st.markdown('<p class="section-header">Vulnerability Assessment</p>',
                                unsafe_allow_html=True)

                    warming_rate = summary.get('warming_rate') or 0
                    if warming_rate > 0.02:
                        risk_level = "🔴 HIGH"
                        risk_color = "#e74c3c"
                        risk_msg   = f"Region is warming at {warming_rate:.3f}°C/year — significantly above global average."
                    elif warming_rate > 0.01:
                        risk_level = "🟠 MEDIUM"
                        risk_color = "#e67e22"
                        risk_msg   = f"Region shows moderate warming trend of {warming_rate:.3f}°C/year."
                    else:
                        risk_level = "🟡 LOW-MEDIUM"
                        risk_color = "#f1c40f"
                        risk_msg   = f"Region shows relatively stable temperature trend."

                    st.markdown(f"""
                    <div style='background:#1a2035; border:1px solid {risk_color};
                                border-radius:8px; padding:1rem; margin:1rem 0;'>
                        <b style='color:{risk_color}; font-size:1.1rem;'>
                            Climate Risk: {risk_level}
                        </b><br><br>
                        <span style='color:#e2e8f0;'>{risk_msg}</span><br><br>
                        <span style='color:#64748b; font-size:0.85rem;'>
                        Areas damaged by disasters become increasingly vulnerable
                        to climate-induced secondary hazards such as landslides,
                        floods, and extreme heat events.
                        </span>
                    </div>
                    """, unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"Error fetching climate data: {str(e)}")

# ── Tab 3: Model Info ─────────────────────────────────────────────────────────
with tab3:
    st.markdown('<p class="section-header">Model Architecture</p>',
                unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="info-box">
        <b style='color:#00d4ff;'>Architecture</b><br>
        U-Net with ResNet50 encoder<br>
        Pretrained on ImageNet<br>
        6 input channels (3 pre + 3 post)<br>
        5 output classes<br><br>

        <b style='color:#00d4ff;'>Training</b><br>
        Dataset: xBD (xView2 Challenge)<br>
        Optimizer: AdamW (lr=0.0005)<br>
        Loss: BCE + Dice combined<br>
        Epochs: 30 with early stopping<br>
        Augmentation: flip + rotation<br><br>

        <b style='color:#00d4ff;'>Performance</b><br>
        Overall Accuracy: 93%<br>
        Macro F1 Score: 0.76<br>
        Destroyed F1: 0.71
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="info-box">
        <b style='color:#00d4ff;'>Damage Classes</b><br><br>
        🟢 No Damage — intact structures<br>
        🟡 Minor Damage — superficial damage<br>
        🟠 Major Damage — structural damage<br>
        🔴 Destroyed — complete collapse<br><br>

        <b style='color:#00d4ff;'>Training Data</b><br>
        19 disaster events worldwide<br>
        Guatemala volcano eruption<br>
        Hurricane Michael, Harvey, Florence<br>
        Midwest flooding, Santa Rosa wildfire<br>
        Palu tsunami, Mexico earthquake<br><br>

        <b style='color:#00d4ff;'>Climate Data</b><br>
        Source: NASA POWER API<br>
        Parameters: Temperature, Rainfall, Humidity<br>
        Resolution: Monthly averages<br>
        Coverage: Global, 2000–present
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<p class="section-header">Training Curves</p>',
                unsafe_allow_html=True)

    curve_path = "outputs/training_curves.png"
    cm_path    = "outputs/confusion_matrix.png"

    if os.path.exists(curve_path):
        st.image(curve_path, caption="Training Loss & Accuracy Curves",
                 use_column_width=True)
    else:
        st.info("Place training_curves.png in outputs/ folder to display here.")

    if os.path.exists(cm_path):
        st.image(cm_path, caption="Confusion Matrix",
                 use_column_width=True)
    else:
        st.info("Place confusion_matrix.png in outputs/ folder to display here.")

    st.markdown('<p class="section-header">Project Pipeline</p>',
                unsafe_allow_html=True)

    st.markdown("""
    ```
    xBD Dataset (USA/Guatemala disasters)
            ↓
    Preprocessing → 256x256 chips with augmentation
            ↓
    U-Net ResNet50 Training (Google Colab T4 GPU)
            ↓
    Inference on unseen test disasters
            ↓
    Apply to target disaster location
            ↓
    GeoJSON damage map export
            ↓
    NASA POWER climate vulnerability overlay
    ```
    """)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#64748b; font-size:0.8rem; padding:1rem;'>
    GeoAI-DisasterMapper | U-Net ResNet50 | xBD Dataset | NASA POWER API<br>
    Built for Graduate Research Assistant Application — GeoAI & Climate Change
</div>
""", unsafe_allow_html=True)