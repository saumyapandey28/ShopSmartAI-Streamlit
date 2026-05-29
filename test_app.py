# ============================================================
# AI-Powered E-Commerce Analytics & Recommendation System
# ============================================================

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(
    page_title="ShopSmart AI",
    page_icon="🔺",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;500;600;700&family=Inconsolata:wght@300;400;500&display=swap');

/* ── Root palette ── */
:root {
    --bg-base:      #0a0a0b;
    --bg-surface:   #111114;
    --bg-card:      #18181c;
    --bg-raised:    #1f1f25;
    --border:       #2a2a32;
    --border-accent:#8b0000;
    --crimson:      #c0392b;
    --crimson-light:#e74c3c;
    --crimson-glow: rgba(192,57,43,0.18);
    --crimson-dim:  rgba(192,57,43,0.08);
    --text-primary: #f0eee8;
    --text-secondary:#9a9a9f;
    --text-muted:   #55555f;
    --gold:         #c9a84c;
    --success:      #2d7a4f;
    --success-text: #5ac888;
}

/* ── App shell ── */
.stApp, [data-testid="stAppViewContainer"] {
    background: var(--bg-base) !important;
    font-family: 'Inconsolata', monospace;
}

/* ── Main content area ── */
[data-testid="stMain"] > div, .main .block-container {
    background: var(--bg-base) !important;
    padding-top: 1rem !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--bg-surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * {
    color: var(--text-primary) !important;
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stMultiselect label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stRadio label,
[data-testid="stSidebar"] .stNumberInput label,
[data-testid="stSidebar"] .stCheckbox label {
    color: var(--text-secondary) !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
}

/* ── Global text ── */
h1, h2, h3, h4, h5, h6 {
    font-family: 'Rajdhani', sans-serif !important;
    color: var(--text-primary) !important;
    letter-spacing: 0.04em;
}
p, span, div, li, td, th { color: var(--text-primary); }
label { color: var(--text-secondary) !important; }
.stMarkdown p { color: var(--text-secondary); }

/* ── Inputs ── */
input, textarea, [data-baseweb="input"] input,
[data-baseweb="textarea"] textarea {
    background: var(--bg-raised) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-primary) !important;
    border-radius: 4px !important;
}
input:focus, textarea:focus {
    border-color: var(--crimson) !important;
    box-shadow: 0 0 0 2px var(--crimson-glow) !important;
}

/* ── Select / Multiselect ── */
[data-baseweb="select"] > div {
    background: var(--bg-raised) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-primary) !important;
}
[data-baseweb="popover"] { background: var(--bg-raised) !important; }
[data-baseweb="menu"] { background: var(--bg-card) !important; }
[data-baseweb="option"]:hover { background: var(--crimson-dim) !important; }
[data-baseweb="tag"] {
    background: var(--border-accent) !important;
    color: var(--text-primary) !important;
}

/* ── Slider ── */
[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {
    background: var(--crimson) !important;
}
.st-by { background: var(--crimson) !important; }

/* ── Radio ── */
[data-testid="stRadio"] label { font-size: 0.82rem !important; }
[data-testid="stRadio"] [data-testid="stMarkdownContainer"] p {
    color: var(--text-primary) !important;
}

/* ── Buttons ── */
.stButton > button {
    background: transparent !important;
    border: 1px solid var(--border-accent) !important;
    color: var(--crimson-light) !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    border-radius: 3px !important;
    transition: all 0.2s !important;
    padding: 0.35rem 0.9rem !important;
}
.stButton > button:hover {
    background: var(--crimson) !important;
    color: #fff !important;
    border-color: var(--crimson) !important;
    box-shadow: 0 0 16px var(--crimson-glow) !important;
}
.stButton > button[kind="primary"] {
    background: var(--crimson) !important;
    color: #fff !important;
    border-color: var(--crimson) !important;
}
.stButton > button[kind="primary"]:hover {
    background: var(--crimson-light) !important;
    box-shadow: 0 0 24px rgba(231,76,60,0.4) !important;
}

/* ── Tabs ── */
[data-baseweb="tab-list"] {
    background: var(--bg-surface) !important;
    border-bottom: 1px solid var(--border) !important;
    gap: 0 !important;
}
[data-baseweb="tab"] {
    background: transparent !important;
    color: var(--text-muted) !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    border-bottom: 2px solid transparent !important;
    padding: 0.6rem 1.4rem !important;
    transition: all 0.2s !important;
}
[aria-selected="true"][data-baseweb="tab"] {
    color: var(--crimson-light) !important;
    border-bottom: 2px solid var(--crimson) !important;
    background: var(--crimson-dim) !important;
}
[data-baseweb="tab-panel"] {
    background: var(--bg-base) !important;
    padding-top: 1.2rem !important;
}

/* ── Metrics ── */
[data-testid="stMetric"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-top: 2px solid var(--crimson) !important;
    border-radius: 4px !important;
    padding: 1rem 1.2rem !important;
}
[data-testid="stMetricLabel"] {
    color: var(--text-muted) !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    font-family: 'Rajdhani', sans-serif !important;
}
[data-testid="stMetricValue"] {
    color: var(--text-primary) !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 1.6rem !important;
    font-weight: 700 !important;
}
[data-testid="stMetricDelta"] { font-size: 0.78rem !important; }

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: 4px !important;
}
[data-testid="stDataFrame"] iframe { background: var(--bg-card) !important; }

/* ── Alerts / messages ── */
[data-testid="stAlert"] {
    background: var(--bg-card) !important;
    border-radius: 4px !important;
}
.stSuccess { border-left: 3px solid var(--success-text) !important; }
.stInfo    { border-left: 3px solid #4a9eca !important; }
.stWarning { border-left: 3px solid var(--gold) !important; }
.stError   { border-left: 3px solid var(--crimson) !important; }

/* ── Expander ── */
[data-testid="stExpander"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 4px !important;
}
[data-testid="stExpander"] summary {
    color: var(--text-secondary) !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.06em !important;
}

/* ── Checkbox ── */
[data-testid="stCheckbox"] label { color: var(--text-secondary) !important; }

/* ── Number input ── */
[data-testid="stNumberInput"] input { text-align: right; }

/* ── Divider ── */
hr { border-color: var(--border) !important; margin: 1.5rem 0 !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: var(--bg-base); }
::-webkit-scrollbar-thumb { background: var(--border-accent); border-radius: 2px; }

/* ══════════════════════════════════
   Custom Components
══════════════════════════════════ */

.page-header {
    border-left: 4px solid var(--crimson);
    padding: 0.8rem 1.2rem;
    background: linear-gradient(90deg, var(--crimson-dim) 0%, transparent 100%);
    margin-bottom: 1.5rem;
    border-radius: 0 4px 4px 0;
}
.page-header h1 {
    margin: 0;
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 1.8rem !important;
    font-weight: 700 !important;
    color: var(--text-primary) !important;
    letter-spacing: 0.06em;
}
.page-header p {
    margin: 0.2rem 0 0;
    font-size: 0.78rem;
    color: var(--text-muted) !important;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}

.section-title {
    font-family: 'Rajdhani', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    color: var(--text-secondary) !important;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.5rem;
    margin-bottom: 1rem;
}

.product-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 1.1rem;
    margin-bottom: 0.8rem;
    transition: border-color 0.2s, box-shadow 0.2s;
    position: relative;
    overflow: hidden;
}
.product-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--crimson) 0%, transparent 100%);
    opacity: 0;
    transition: opacity 0.2s;
}
.product-card:hover {
    border-color: var(--border-accent);
    box-shadow: 0 4px 24px var(--crimson-glow);
}
.product-card:hover::before { opacity: 1; }

.product-name {
    font-family: 'Rajdhani', sans-serif;
    font-size: 1.05rem;
    font-weight: 700;
    color: var(--text-primary) !important;
    letter-spacing: 0.03em;
    margin: 0 0 2px;
}
.product-brand {
    font-size: 0.72rem;
    color: var(--text-muted) !important;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin: 0 0 8px;
}
.product-desc {
    font-size: 0.8rem;
    color: var(--text-secondary) !important;
    line-height: 1.5;
    margin: 6px 0 10px;
}
.product-price {
    font-family: 'Rajdhani', sans-serif;
    font-size: 1.4rem;
    font-weight: 700;
    color: var(--text-primary) !important;
}
.product-original {
    font-size: 0.82rem;
    color: var(--text-muted) !important;
    text-decoration: line-through;
    margin-left: 6px;
}
.product-savings {
    font-size: 0.72rem;
    color: var(--success-text) !important;
    letter-spacing: 0.06em;
}
.product-stock {
    font-size: 0.7rem;
    color: var(--text-muted) !important;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-top: 4px;
}

.badge {
    display: inline-block;
    padding: 1px 8px;
    border-radius: 2px;
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-right: 4px;
    margin-bottom: 4px;
    font-family: 'Rajdhani', sans-serif;
}
.badge-cat  { background: rgba(139,0,0,0.3); color: #e87070 !important; border: 1px solid rgba(139,0,0,0.5); }
.badge-disc { background: rgba(45,122,79,0.25); color: var(--success-text) !important; border: 1px solid rgba(45,122,79,0.4); }
.badge-star { background: rgba(201,168,76,0.2); color: var(--gold) !important; border: 1px solid rgba(201,168,76,0.35); }
.badge-in-cart { background: var(--crimson); color: #fff !important; border: 1px solid var(--crimson); }

.cart-row {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-left: 3px solid var(--crimson);
    border-radius: 0 4px 4px 0;
    padding: 0.65rem 1rem;
    margin-bottom: 0.4rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
}

.rec-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-left: 3px solid var(--crimson);
    border-radius: 0 4px 4px 0;
    padding: 1rem;
    margin-bottom: 0.6rem;
    transition: box-shadow 0.2s;
}
.rec-card:hover { box-shadow: 0 4px 20px var(--crimson-glow); }
.rec-name {
    font-family: 'Rajdhani', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    color: var(--text-primary) !important;
    margin: 0 0 4px;
}
.rec-meta { font-size: 0.75rem; color: var(--text-secondary) !important; margin: 2px 0; }

.stat-box {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 1rem 1.2rem;
    text-align: center;
}
.stat-label {
    font-size: 0.68rem;
    color: var(--text-muted) !important;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    font-family: 'Rajdhani', sans-serif;
    margin-bottom: 4px;
}
.stat-value {
    font-family: 'Rajdhani', sans-serif;
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--text-primary) !important;
}
.stat-value.accent { color: var(--crimson-light) !important; }

.insight-box {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-left: 3px solid var(--gold);
    border-radius: 0 4px 4px 0;
    padding: 0.9rem 1.1rem;
    margin-top: 0.8rem;
}
.insight-box p { font-size: 0.82rem; color: var(--text-secondary) !important; margin: 0; }

.qty-badge {
    display: inline-block;
    background: var(--crimson);
    color: #fff !important;
    font-family: 'Rajdhani', sans-serif;
    font-weight: 700;
    font-size: 0.85rem;
    width: 28px; height: 28px;
    line-height: 28px;
    text-align: center;
    border-radius: 2px;
}

.main-brand {
    font-family: 'Rajdhani', sans-serif;
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--text-primary) !important;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}
.main-brand span { color: var(--crimson-light) !important; }
.brand-tagline {
    font-size: 0.65rem;
    color: var(--text-muted) !important;
    letter-spacing: 0.2em;
    text-transform: uppercase;
}

.sidebar-section {
    font-size: 0.65rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--crimson) !important;
    font-family: 'Rajdhani', sans-serif;
    font-weight: 700;
    margin: 1rem 0 0.5rem;
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.3rem;
}

.found-count {
    font-family: 'Rajdhani', sans-serif;
    font-size: 0.75rem;
    color: var(--text-muted) !important;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}
.found-count span { color: var(--crimson-light) !important; font-weight: 700; font-size: 0.95rem; }

.checkout-total {
    background: linear-gradient(135deg, var(--bg-card), var(--bg-raised));
    border: 1px solid var(--border-accent);
    border-radius: 4px;
    padding: 1.2rem 1.5rem;
    text-align: center;
}
.checkout-total .amount {
    font-family: 'Rajdhani', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    color: var(--text-primary) !important;
}
.checkout-total .label {
    font-size: 0.7rem;
    color: var(--text-muted) !important;
    letter-spacing: 0.15em;
    text-transform: uppercase;
}
.savings-pill {
    display: inline-block;
    background: rgba(45,122,79,0.2);
    border: 1px solid rgba(90,200,136,0.3);
    border-radius: 20px;
    padding: 2px 12px;
    font-size: 0.75rem;
    color: var(--success-text) !important;
    margin-top: 4px;
}
</style>
""", unsafe_allow_html=True)

PLOTLY_TEMPLATE = go.layout.Template()
PLOTLY_TEMPLATE.layout = go.Layout(
    paper_bgcolor="#111114",
    plot_bgcolor="#111114",
    font=dict(family="Rajdhani, sans-serif", color="#9a9a9f", size=12),
    title_font=dict(family="Rajdhani, sans-serif", color="#f0eee8", size=16),
    xaxis=dict(gridcolor="#2a2a32", linecolor="#2a2a32", tickcolor="#55555f", zerolinecolor="#2a2a32"),
    yaxis=dict(gridcolor="#2a2a32", linecolor="#2a2a32", tickcolor="#55555f", zerolinecolor="#2a2a32"),
    legend=dict(bgcolor="#18181c", bordercolor="#2a2a32", borderwidth=1, font=dict(color="#9a9a9f")),
    margin=dict(t=32, b=24, l=12, r=12),
    colorway=["#c0392b","#e74c3c","#922b21","#641e16","#f1948a","#fadbd8","#7b241c","#a93226"],
)
CRIMSON_SCALE = [[0,"#1a0505"],[0.25,"#5c1010"],[0.5,"#8b0000"],[0.75,"#c0392b"],[1,"#e74c3c"]]

# ══════════════════════════════════════════════════════════════
# EMBEDDED PRODUCT CATALOGUE  — 35 products, 6 categories
# ══════════════════════════════════════════════════════════════
PRODUCTS = [
    # ── Electronics ────────────────────────────────────────────
    {"id":"P001","name":"ProBook Laptop 15","category":"Electronics","brand":"TechEdge",
     "price":54999,"discount":0.10,"rating":4.5,"stock":23,
     "description":"15.6 FHD IPS · Intel i5-12th Gen · 16GB DDR5 · 512GB NVMe SSD · Backlit KB · Win11 Pro",
     "tags":["laptop","work","student","office"]},
    {"id":"P002","name":"UltraPhone X12","category":"Electronics","brand":"NovaMobile",
     "price":32999,"discount":0.15,"rating":4.3,"stock":45,
     "description":"6.7 AMOLED 120Hz · 108MP Triple Cam · 5000mAh · 65W Fast Charge · 5G · 128GB",
     "tags":["phone","5g","camera","mobile"]},
    {"id":"P003","name":"SoundPro ANC Headphones","category":"Electronics","brand":"AudioMax",
     "price":4999,"discount":0.20,"rating":4.6,"stock":80,
     "description":"Active Noise Cancellation · 30hr battery · Bluetooth 5.2 · Hi-Res Audio · Foldable",
     "tags":["audio","wireless","anc","headphones"]},
    {"id":"P004","name":"SlimTab Pro 10","category":"Electronics","brand":"TechEdge",
     "price":22999,"discount":0.08,"rating":4.2,"stock":17,
     "description":"10.5 2K Display · Octa-core 2.4GHz · 4GB RAM · 128GB · Stylus Support · USB-C",
     "tags":["tablet","drawing","portable","student"]},
    {"id":"P005","name":"SmartWatch Elite S3","category":"Electronics","brand":"WristTech",
     "price":8999,"discount":0.12,"rating":4.4,"stock":55,
     "description":"1.4 AMOLED · SpO2 & HR Monitor · GPS · 7-day battery · IP68 · 100+ sport modes",
     "tags":["fitness","watch","health","gps"]},
    {"id":"P006","name":"4K StreamCam Pro","category":"Electronics","brand":"PixelView",
     "price":3499,"discount":0.05,"rating":4.1,"stock":34,
     "description":"4K 30fps · Autofocus · Dual noise-cancelling mics · USB-C · Ring fill light · Clip mount",
     "tags":["webcam","streaming","wfh","content"]},
    {"id":"P007","name":"Mech Keyboard RGB TKL","category":"Electronics","brand":"KeyMaster",
     "price":2999,"discount":0.18,"rating":4.7,"stock":60,
     "description":"Tenkeyless · Cherry MX Blue · Per-key RGB · USB-A · N-Key Rollover · Aluminium top",
     "tags":["keyboard","gaming","rgb","mechanical"]},
    {"id":"P008","name":"Portable SSD 1TB","category":"Electronics","brand":"DataSpeed",
     "price":6499,"discount":0.10,"rating":4.5,"stock":40,
     "description":"USB 3.2 Gen2 · 1050MB/s read · 1000MB/s write · Shock-proof · Palm-sized · Plug & Play",
     "tags":["storage","backup","portable","speed"]},
    # ── Clothing ────────────────────────────────────────────────
    {"id":"P009","name":"Classic Cotton T-Shirt","category":"Clothing","brand":"UrbanWear",
     "price":699,"discount":0.30,"rating":4.2,"stock":200,
     "description":"100% Combed Cotton · Pre-shrunk · Reinforced seams · S–3XL · 14 colour options",
     "tags":["casual","cotton","everyday","unisex"]},
    {"id":"P010","name":"Slim Stretch Denim","category":"Clothing","brand":"DenimCo",
     "price":1899,"discount":0.25,"rating":4.3,"stock":90,
     "description":"98% Cotton 2% Elastane · 5-pocket · Mid-rise · Stone-washed · Machine washable",
     "tags":["denim","jeans","casual","slim"]},
    {"id":"P011","name":"Winter Puffer Jacket","category":"Clothing","brand":"FrostLine",
     "price":3499,"discount":0.20,"rating":4.6,"stock":35,
     "description":"Water-resistant shell · Hooded · 600-fill lightweight down · Inner fleece lining · Packable",
     "tags":["winter","warm","outdoor","jacket"]},
    {"id":"P012","name":"Chiffon Wrap Dress","category":"Clothing","brand":"BlosomStyle",
     "price":1299,"discount":0.15,"rating":4.4,"stock":55,
     "description":"Lightweight chiffon · Adjustable tie waist · Midi length · Floral print · Machine washable",
     "tags":["summer","dress","feminine","floral"]},
    {"id":"P013","name":"Air Running Sneakers","category":"Clothing","brand":"SpeedStep",
     "price":2799,"discount":0.10,"rating":4.5,"stock":70,
     "description":"Air-cushion midsole · Engineered mesh upper · Reflective strip · Unisex · EU 36–46",
     "tags":["running","sports","shoes","unisex"]},
    {"id":"P014","name":"Non-Iron Oxford Shirt","category":"Clothing","brand":"BoardRoom",
     "price":1599,"discount":0.05,"rating":4.1,"stock":80,
     "description":"100% Fine Cotton · Non-iron finish · Button-down collar · Slim fit · 8 colours",
     "tags":["formal","office","cotton","business"]},
    # ── Home & Kitchen ──────────────────────────────────────────
    {"id":"P015","name":"Pro Series Blender 1200W","category":"Home & Kitchen","brand":"ChefMate",
     "price":4299,"discount":0.18,"rating":4.4,"stock":28,
     "description":"1200W motor · 6 stainless blades · 1.5L BPA-free jar · 5 speeds + pulse · Self-clean",
     "tags":["blender","smoothie","kitchen","appliance"]},
    {"id":"P016","name":"Barista Espresso Maker","category":"Home & Kitchen","brand":"BrewKing",
     "price":7999,"discount":0.12,"rating":4.7,"stock":20,
     "description":"15-bar Italian pump · Steam milk frother · 1.2L removable tank · Auto shut-off · 2-cup",
     "tags":["coffee","espresso","morning","appliance"]},
    {"id":"P017","name":"Digital Air Fryer XL 5L","category":"Home & Kitchen","brand":"CrispAir",
     "price":3999,"discount":0.22,"rating":4.5,"stock":42,
     "description":"5L basket · 8 digital presets · 200C max · Non-stick ceramic · Dishwasher-safe parts",
     "tags":["airfryer","healthy","cooking","appliance"]},
    {"id":"P018","name":"Smart 4-Slice Toaster","category":"Home & Kitchen","brand":"ToastPro",
     "price":1199,"discount":0.10,"rating":4.0,"stock":65,
     "description":"4-slice wide-slot · 7 browning levels · Stainless crumb tray · Reheat & defrost modes",
     "tags":["toaster","breakfast","kitchen","appliance"]},
    {"id":"P019","name":"Cyclone Cordless Vacuum","category":"Home & Kitchen","brand":"CleanSweep",
     "price":9499,"discount":0.15,"rating":4.6,"stock":18,
     "description":"22Kpa cyclone suction · 45-min runtime · HEPA filter · Flexi-reach · Wall dock included",
     "tags":["vacuum","cleaning","cordless","home"]},
    {"id":"P020","name":"Granite Cookware Set 5pc","category":"Home & Kitchen","brand":"ChefMate",
     "price":2999,"discount":0.25,"rating":4.3,"stock":30,
     "description":"5-piece set · Granite non-stick coating · Induction + gas compatible · Oven safe to 200C",
     "tags":["cookware","nonstick","cooking","kitchen"]},
    # ── Books ───────────────────────────────────────────────────
    {"id":"P021","name":"Atomic Habits","category":"Books","brand":"Penguin",
     "price":499,"discount":0.20,"rating":4.8,"stock":150,
     "description":"James Clear · #1 NYT Bestseller · Build good habits · Break bad ones · Proven framework",
     "tags":["selfhelp","habits","productivity","bestseller"]},
    {"id":"P022","name":"Python for Data Science","category":"Books","brand":"OReilly",
     "price":899,"discount":0.15,"rating":4.6,"stock":75,
     "description":"Wes McKinney · NumPy · Pandas · Matplotlib · Scikit-learn · 3rd Edition · 600 pages",
     "tags":["python","datascience","programming","technical"]},
    {"id":"P023","name":"The Midnight Library","category":"Books","brand":"Canongate",
     "price":399,"discount":0.10,"rating":4.5,"stock":120,
     "description":"Matt Haig · Booker Prize longlisted · Life choices · Regret · Infinite possibility · Fiction",
     "tags":["fiction","novel","inspiring","booker"]},
    {"id":"P024","name":"Sapiens","category":"Books","brand":"Harper",
     "price":549,"discount":0.18,"rating":4.7,"stock":90,
     "description":"Yuval Noah Harari · A brief history of humankind · 30+ languages · Global bestseller",
     "tags":["history","nonfiction","bestseller","philosophy"]},
    {"id":"P025","name":"The Complete Cookbook","category":"Books","brand":"DK",
     "price":799,"discount":0.12,"rating":4.4,"stock":60,
     "description":"500+ recipes · Step-by-step photography · World cuisines · Techniques & tips · Hardcover",
     "tags":["cooking","recipes","food","reference"]},
    # ── Sports ──────────────────────────────────────────────────
    {"id":"P026","name":"Pro Yoga Mat 6mm","category":"Sports","brand":"ZenFlex",
     "price":1299,"discount":0.15,"rating":4.5,"stock":85,
     "description":"Non-slip TPE · 6mm cushion · Alignment print · 183x61cm · Carry strap · Eco-friendly",
     "tags":["yoga","fitness","mat","exercise"]},
    {"id":"P027","name":"Adjustable Dumbbell Set","category":"Sports","brand":"IronGrip",
     "price":6999,"discount":0.10,"rating":4.7,"stock":22,
     "description":"5–25kg per dumbbell · Quick-adjust spin dial · Compact ABS tray · Replaces 9 pairs",
     "tags":["weights","gym","strength","home-gym"]},
    {"id":"P028","name":"Gore-Tex Trail Shoes","category":"Sports","brand":"TrailBeast",
     "price":3499,"discount":0.08,"rating":4.4,"stock":48,
     "description":"Gore-Tex waterproof · Vibram outsole · 8mm heel drop · Rock plate · Sizes EU 38–47",
     "tags":["running","trail","outdoor","waterproof"]},
    {"id":"P029","name":"21-Speed Mountain Bike","category":"Sports","brand":"TerraCycle",
     "price":15999,"discount":0.12,"rating":4.3,"stock":10,
     "description":"6061 Aluminium frame · Shimano Altus 21-speed · Hydraulic disc brakes · 26 inch alloy wheels",
     "tags":["cycling","outdoor","bike","mountain"]},
    {"id":"P030","name":"2-Person Hiking Tent","category":"Sports","brand":"WildCamp",
     "price":4499,"discount":0.20,"rating":4.5,"stock":25,
     "description":"3000mm HH waterproof · 3-season · Freestanding · 2.1kg · Vestibule · Stakes included",
     "tags":["camping","outdoor","travel","hiking"]},
    # ── Beauty ──────────────────────────────────────────────────
    {"id":"P031","name":"Vitamin C Brightening Serum","category":"Beauty","brand":"GlowLab",
     "price":899,"discount":0.25,"rating":4.6,"stock":110,
     "description":"15% L-Ascorbic Acid · Niacinamide · Hyaluronic Acid · 30ml · All skin types · Paraben-free",
     "tags":["skincare","serum","brightening","vitaminc"]},
    {"id":"P032","name":"Velvet Matte Lip Set x6","category":"Beauty","brand":"ColourPop",
     "price":599,"discount":0.30,"rating":4.3,"stock":95,
     "description":"6 curated shades · 8hr long-lasting formula · Non-drying · Vegan · Cruelty-free · 3.5ml each",
     "tags":["lips","makeup","matte","vegan"]},
    {"id":"P033","name":"Volume Boost Mascara","category":"Beauty","brand":"LashQueen",
     "price":749,"discount":0.20,"rating":4.4,"stock":80,
     "description":"Buildable volume · Lengthening hourglass brush · Smudge-proof · 10ml · Ophthalmologist tested",
     "tags":["eyes","mascara","makeup","volume"]},
    {"id":"P034","name":"Oud & Sandalwood EDP 100ml","category":"Beauty","brand":"Aromas",
     "price":2499,"discount":0.10,"rating":4.7,"stock":40,
     "description":"Oud Wood top · Sandalwood heart · Amber base · 12hr projection · Unisex · Gift-boxed",
     "tags":["fragrance","oud","luxury","perfume"]},
    {"id":"P035","name":"Day Glow Moisturiser SPF30","category":"Beauty","brand":"GlowLab",
     "price":699,"discount":0.15,"rating":4.5,"stock":120,
     "description":"SPF 30 PA+++ · Hyaluronic Acid · 50ml · Lightweight gel-cream · Non-comedogenic · Unscented",
     "tags":["skincare","spf","moisturiser","daily"]},
]

PRODUCT_DF = pd.DataFrame(PRODUCTS)
PRODUCT_DF["final_price"]  = (PRODUCT_DF["price"] * (1 - PRODUCT_DF["discount"])).round(0).astype(int)
PRODUCT_DF["discount_pct"] = (PRODUCT_DF["discount"] * 100).astype(int)
PRODUCT_DF["savings"]      = PRODUCT_DF["price"] - PRODUCT_DF["final_price"]

ICONS = {"Electronics":"▣","Clothing":"◈","Home & Kitchen":"◉","Books":"◆","Sports":"▲","Beauty":"◇"}
CAT_COLORS = ["#c0392b","#922b21","#7b241c","#641e16","#a93226","#e74c3c"]
all_cats = PRODUCT_DF["category"].unique().tolist()

# ══════════════════════════════════════════════════════════════
# ORDER DATA  (analytics)
# ══════════════════════════════════════════════════════════════

@st.cache_data
def generate_orders(n: int = 1000, seed: int = 42):
    rng   = np.random.default_rng(seed)
    cats  = all_cats
    regs  = ["North","South","East","West","Central"]
    stats = ["Completed","Pending","Cancelled","Returned"]
    cat_arr  = rng.choice(cats, n)
    prod_arr = [rng.choice(PRODUCT_DF[PRODUCT_DF["category"]==c]["name"].values) for c in cat_arr]
    prices   = np.round(rng.uniform(200, 60000, n), 2)
    qty      = rng.integers(1, 8, n)
    disc     = np.round(rng.uniform(0, 0.4, n), 2)
    revenue  = np.round(prices * qty * (1 - disc), 2)
    ratings  = np.clip(rng.normal(4.1, 0.7, n), 1, 5).round(1)
    dates    = [datetime(2024,1,1)+timedelta(days=int(d)) for d in rng.integers(0,365,n)]
    df = pd.DataFrame({
        "order_id": [f"ORD-{10000+i}" for i in range(n)],
        "date": dates, "category": cat_arr, "product": prod_arr,
        "price": prices, "quantity": qty, "discount": disc,
        "revenue": revenue, "rating": ratings,
        "region": rng.choice(regs, n),
        "status": rng.choice(stats, n, p=[0.70,0.15,0.08,0.07]),
        "lat": rng.uniform(8.0, 37.0, n),
        "lon": rng.uniform(68.0, 97.0, n),
    })
    df["month"] = df["date"].dt.strftime("%b %Y")
    return df

@st.cache_data
def generate_customers(n: int = 300, seed: int = 7):
    rng  = np.random.default_rng(seed)
    segs = ["Premium","Regular","New","At-Risk"]
    return pd.DataFrame({
        "customer_id":   [f"CUST-{1000+i}" for i in range(n)],
        "age":           rng.integers(18, 70, n),
        "total_spent":   np.round(rng.exponential(500, n), 2),
        "orders":        rng.integers(1, 50, n),
        "segment":       rng.choice(segs, n, p=[0.15,0.45,0.25,0.15]),
        "avg_rating":    np.clip(rng.normal(4.1, 0.7, n), 1, 5).round(1),
        "last_purchase": [datetime(2024,1,1)+timedelta(days=int(d)) for d in rng.integers(0,365,n)],
    })

# ══════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════
for key, val in [("cart",{}),("ai_log",[]),("feedback",{})]:
    if key not in st.session_state:
        st.session_state[key] = val

# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="padding: 0.8rem 0 0.5rem">
        <div class="main-brand">SHOP<span>SMART</span></div>
        <div class="brand-tagline">AI Commerce Intelligence</div>
    </div>""", unsafe_allow_html=True)
    st.markdown("---")

    st.markdown('<div class="sidebar-section">Navigate</div>', unsafe_allow_html=True)
    page = st.selectbox("", [
        "Dashboard", "Product Browser", "Product Analytics",
        "AI Recommendations", "Customer Insights",
        "Geographic View",
    ], label_visibility="collapsed")

    st.markdown('<div class="sidebar-section">Analytics Filters</div>', unsafe_allow_html=True)
    sel_cats      = st.multiselect("Categories", all_cats, default=all_cats)
    month_range   = st.slider("Month Range", 1, 12, (1, 12))
    status_filter = st.radio("Order Status", ["All","Completed","Pending","Cancelled","Returned"])
    min_revenue   = st.number_input("Min Revenue (Rs)", 0, 50000, 0, step=500)
    show_raw      = st.checkbox("Show Raw Tables", False)

    st.markdown("---")
    total_items = sum(st.session_state.cart.values())
    if total_items:
        cart_total = sum(
            int(PRODUCT_DF[PRODUCT_DF["id"]==pid]["final_price"].values[0]) * q
            for pid, q in st.session_state.cart.items()
            if len(PRODUCT_DF[PRODUCT_DF["id"]==pid]) > 0
        )
        st.markdown(f"""
        <div style="background:#18181c;border:1px solid #2a2a32;border-left:3px solid #c0392b;
             border-radius:0 4px 4px 0;padding:0.7rem 0.9rem">
          <div style="font-size:0.65rem;letter-spacing:0.18em;text-transform:uppercase;
               color:#55555f;font-family:Rajdhani,sans-serif">Cart</div>
          <div style="font-family:Rajdhani,sans-serif;font-size:1.1rem;font-weight:700;
               color:#f0eee8">{total_items} item(s) &mdash; Rs {cart_total:,}</div>
        </div>""", unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.65rem;color:#55555f;letter-spacing:0.1em;margin-top:1.5rem;text-transform:uppercase">Built with Streamlit · NumPy · Pandas</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# LOAD & FILTER ORDERS
# ══════════════════════════════════════════════════════════════
df_full = generate_orders()
df_cust = generate_customers()
df = df_full.copy()
if sel_cats:   df = df[df["category"].isin(sel_cats)]
df = df[df["date"].dt.month.between(*month_range)]
if status_filter != "All": df = df[df["status"] == status_filter]
df = df[df["revenue"] >= min_revenue]

def make_fig(fig, h=340):
    fig.update_layout(template=PLOTLY_TEMPLATE, height=h, margin=dict(t=28,b=16,l=8,r=8))
    return fig

# ══════════════════════════════════════════════════════════════
# PAGE: DASHBOARD
# ══════════════════════════════════════════════════════════════
if page == "Dashboard":
    st.markdown('<div class="page-header"><h1>Executive Dashboard</h1><p>Real-time analytics overview</p></div>', unsafe_allow_html=True)

    total_rev    = df["revenue"].sum()
    total_orders = len(df)
    avg_order    = df["revenue"].mean() if len(df) else 0
    avg_rating   = df["rating"].mean()  if len(df) else 0
    conv_rate    = (df["status"]=="Completed").mean()*100 if len(df) else 0

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Total Revenue",   f"Rs {total_rev:,.0f}",   "+12.4%")
    c2.metric("Total Orders",    f"{total_orders:,}",       "+8.1%")
    c3.metric("Avg Order Value", f"Rs {avg_order:,.0f}",    "+3.7%")
    c4.metric("Avg Rating",      f"{avg_rating:.2f} / 5",  "+0.2")
    c5.metric("Conversion",      f"{conv_rate:.1f}%",       "+1.5%")

    st.markdown("---")
    col1, col2 = st.columns([3,2])
    with col1:
        st.markdown('<div class="section-title">Monthly Revenue Trend</div>', unsafe_allow_html=True)
        monthly = df.groupby(df["date"].dt.to_period("M"))["revenue"].sum().reset_index()
        monthly["date"] = monthly["date"].dt.to_timestamp()
        fig = px.line(monthly, x="date", y="revenue", markers=True)
        fig.update_traces(line_color="#c0392b", line_width=2, marker_color="#e74c3c", marker_size=5)
        fig.add_scatter(x=monthly["date"], y=monthly["revenue"], fill='tozeroy',
                        fillcolor="rgba(192,57,43,0.07)", line=dict(width=0), showlegend=False)
        st.plotly_chart(make_fig(fig, 300), use_container_width=True)
    with col2:
        st.markdown('<div class="section-title">Revenue by Category</div>', unsafe_allow_html=True)
        cat_rev = df.groupby("category")["revenue"].sum().reset_index()
        fig = px.pie(cat_rev, names="category", values="revenue",
                     color_discrete_sequence=CAT_COLORS, hole=0.55)
        fig.update_traces(textfont_color="#f0eee8", textfont_family="Rajdhani")
        st.plotly_chart(make_fig(fig, 300), use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<div class="section-title">Order Status Breakdown</div>', unsafe_allow_html=True)
        sc = df["status"].value_counts().reset_index(); sc.columns = ["status","count"]
        cmap = {"Completed":"#c0392b","Pending":"#7b241c","Cancelled":"#55555f","Returned":"#2a2a32"}
        fig = px.bar(sc, x="status", y="count", color="status", color_discrete_map=cmap)
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(make_fig(fig, 280), use_container_width=True)
    with col_b:
        st.markdown('<div class="section-title">Rating Distribution</div>', unsafe_allow_html=True)
        fig = px.histogram(df, x="rating", nbins=20, color_discrete_sequence=["#8b0000"])
        fig.update_traces(marker_line_color="#c0392b", marker_line_width=0.5)
        st.plotly_chart(make_fig(fig, 280), use_container_width=True)

    if show_raw:
        st.markdown('<div class="section-title">Raw Orders</div>', unsafe_allow_html=True)
        st.dataframe(df.head(100), use_container_width=True)

# ══════════════════════════════════════════════════════════════
# PAGE: PRODUCT BROWSER
# ══════════════════════════════════════════════════════════════
elif page == "Product Browser":
    st.markdown('<div class="page-header"><h1>Product Browser</h1><p>35 products across 6 categories — search & filter</p></div>', unsafe_allow_html=True)

    # ── Filter Bar ────────────────────────────────────────────
    s1, s2, s3, s4 = st.columns([3,2,2,2])
    search_q   = s1.text_input("Search", placeholder="product, brand, tag... e.g. laptop, yoga, serum")
    cat_filter = s2.selectbox("Category", ["All"] + all_cats)
    sort_by    = s3.selectbox("Sort", ["Relevance","Price: Low→High","Price: High→Low","Rating","Discount %"])
    price_max  = s4.number_input("Max Price (Rs)", 399, 100000, 100000, step=500)

    f1, f2, f3 = st.columns([2,2,4])
    min_rating = f1.slider("Min Rating", 1.0, 5.0, 1.0, 0.1)
    show_disc  = f2.checkbox("Discounted Only (>10%)", False)

    # ── Filter logic ──────────────────────────────────────────
    fdf = PRODUCT_DF.copy()
    if search_q:
        q = search_q.lower().strip()
        fdf = fdf[
            fdf["name"].str.lower().str.contains(q, na=False) |
            fdf["description"].str.lower().str.contains(q, na=False) |
            fdf["brand"].str.lower().str.contains(q, na=False) |
            fdf["tags"].apply(lambda t: any(q in tag for tag in t))
        ]
    if cat_filter != "All": fdf = fdf[fdf["category"] == cat_filter]
    fdf = fdf[(fdf["final_price"] <= price_max) & (fdf["rating"] >= min_rating)]
    if show_disc: fdf = fdf[fdf["discount_pct"] > 10]

    if sort_by == "Price: Low→High":   fdf = fdf.sort_values("final_price")
    elif sort_by == "Price: High→Low": fdf = fdf.sort_values("final_price", ascending=False)
    elif sort_by == "Rating":          fdf = fdf.sort_values("rating", ascending=False)
    elif sort_by == "Discount %":      fdf = fdf.sort_values("discount_pct", ascending=False)
    fdf = fdf.reset_index(drop=True)

    st.markdown(f'<div class="found-count"><span>{len(fdf)}</span> product(s) found</div>', unsafe_allow_html=True)
    st.markdown("---")

    if len(fdf) == 0:
        st.info("No products match your filters. Try broadening the search or adjusting filters.")
    else:
        rows = [fdf.iloc[i:i+3] for i in range(0, len(fdf), 3)]
        for row_df in rows:
            cols = st.columns(3)
            for col, (_, p) in zip(cols, row_df.iterrows()):
                icon = ICONS.get(p["category"], "◈")

                with col:
                    st.markdown(f"""
<div class="product-card">
  <div class="product-name">{icon}&nbsp; {p['name']}</div>
  <div class="product-brand">{p['brand']}</div>
  <div>
    <span class="badge badge-cat">{p['category']}</span>
    <span class="badge badge-star">{p['rating']} ★</span>
    {"<span class='badge badge-disc'>-"+str(p['discount_pct'])+"%</span>" if p['discount_pct']>0 else ""}
  </div>
  <div class="product-desc">{p['description']}</div>
  <div>
    <span class="product-price">Rs {p['final_price']:,}</span>
    {"<span class='product-original'>Rs "+f"{p['price']:,}"+"</span>" if p['savings']>0 else ""}
  </div>
  {"<div class='product-savings'>Save Rs "+str(p['savings'])+"</div>" if p['savings']>0 else ""}
  <div class="product-stock">Stock: {p['stock']} units</div>
</div>""", unsafe_allow_html=True)

    # ── Cart ──────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-title">Cart Summary</div>', unsafe_allow_html=True)

    if not st.session_state.cart:
        st.markdown('<div style="padding:1.2rem;background:#18181c;border:1px solid #2a2a32;border-radius:4px;color:#55555f;font-size:0.82rem;letter-spacing:0.06em">Cart is empty.</div>', unsafe_allow_html=True)
    else:
        cart_rows = []
        for pid, qty in st.session_state.cart.items():
            m = PRODUCT_DF[PRODUCT_DF["id"] == pid]
            if len(m) == 0: continue
            r = m.iloc[0]
            cart_rows.append({
                "Product":       r["name"],
                "Category":      r["category"],
                "Unit (Rs)":     r["final_price"],
                "Qty":           qty,
                "Subtotal (Rs)": r["final_price"] * qty,
            })
        cart_df = pd.DataFrame(cart_rows)
        grand   = int(cart_df["Subtotal (Rs)"].sum())
        total_savings = sum(
            (PRODUCT_DF[PRODUCT_DF["id"]==pid].iloc[0]["savings"]) * q
            for pid, q in st.session_state.cart.items()
            if len(PRODUCT_DF[PRODUCT_DF["id"]==pid]) > 0
        )

        st.dataframe(cart_df, use_container_width=True, hide_index=True)
        st.markdown("<br>", unsafe_allow_html=True)

        ca, cb, cc = st.columns([3,2,2])
        with ca:
            st.markdown(f"""
<div class="checkout-total">
  <div class="label">Grand Total</div>
  <div class="amount">Rs {grand:,}</div>
  <div class="savings-pill">You save Rs {int(total_savings):,}</div>
</div>""", unsafe_allow_html=True)
        with cb:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Clear Cart", use_container_width=True):
                st.session_state.cart = {}
        with cc:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Checkout →", use_container_width=True, type="primary"):
                st.success(f"Order placed! Total Rs {grand:,}. Thank you.")
                st.balloons()
                st.session_state.cart = {}

# ══════════════════════════════════════════════════════════════
# PAGE: PRODUCT ANALYTICS
# ══════════════════════════════════════════════════════════════
elif page == "Product Analytics":
    st.markdown('<div class="page-header"><h1>Product Analytics</h1><p>Performance, discount impact, returns</p></div>', unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["Top Products","Discount Impact","Returns Analysis"])

    with tab1:
        st.markdown('<div class="section-title">Top 15 Products by Revenue</div>', unsafe_allow_html=True)
        top = df.groupby(["category","product"])["revenue"].sum().reset_index()\
                .sort_values("revenue", ascending=False).head(15)
        fig = px.bar(top, x="product", y="revenue", color="category",
                     color_discrete_sequence=CAT_COLORS)
        fig.update_layout(xaxis_tickangle=-40)
        st.plotly_chart(make_fig(fig, 420), use_container_width=True)
        if show_raw: st.dataframe(top, use_container_width=True)

    with tab2:
        st.markdown('<div class="section-title">Discount vs Revenue</div>', unsafe_allow_html=True)
        fig = px.scatter(df.sample(min(400,len(df))), x="discount", y="revenue",
                         color="category", size="quantity", opacity=0.75,
                         color_discrete_sequence=CAT_COLORS)
        st.plotly_chart(make_fig(fig, 400), use_container_width=True)
        corr = np.corrcoef(df["discount"], df["revenue"])[0,1]
        st.markdown(f'<div class="insight-box"><p>Discount–Revenue Pearson correlation: <strong style="color:#e74c3c">{corr:.3f}</strong>. {"Heavy discounting is negatively impacting revenue." if corr < -0.3 else "Moderate discount strategy detected."}</p></div>', unsafe_allow_html=True)

    with tab3:
        st.markdown('<div class="section-title">Returns & Cancellations Heatmap</div>', unsafe_allow_html=True)
        neg = df[df["status"].isin(["Cancelled","Returned"])]
        pivot = neg.pivot_table(index="category", columns="status", values="order_id", aggfunc="count", fill_value=0)
        fig = px.imshow(pivot, text_auto=True, color_continuous_scale=CRIMSON_SCALE)
        st.plotly_chart(make_fig(fig, 360), use_container_width=True)
        st.markdown('<div class="insight-box"><p>Electronics & Clothing show the highest return rates. Improving product descriptions and size guides can significantly reduce return volumes.</p></div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# PAGE: AI RECOMMENDATIONS
# ══════════════════════════════════════════════════════════════
elif page == "AI Recommendations":
    st.markdown('<div class="page-header"><h1>AI Recommendations</h1><p>Personalised product engine</p></div>', unsafe_allow_html=True)

    with st.form("rec_form"):
        fc1, fc2 = st.columns(2)
        with fc1:
            user_age    = st.slider("Age", 18, 70, 28)
            user_budget = st.number_input("Budget per item (Rs)", 100, 100000, 5000, step=500)
            user_cats   = st.multiselect("Preferred Categories", all_cats, default=["Electronics","Sports"])
        with fc2:
            user_gender = st.radio("Gender", ["Male","Female","Prefer not to say"])
            user_region = st.selectbox("Region", ["North","South","East","West","Central"])
            user_rating = st.slider("Minimum Rating", 1.0, 5.0, 4.0, 0.1)
        user_notes = st.text_input("Specific needs (optional)", placeholder="e.g. gift for mom, home office setup")
        submitted  = st.form_submit_button("Generate Recommendations →", use_container_width=True)

    if submitted:
        pool = PRODUCT_DF[
            PRODUCT_DF["category"].isin(user_cats) &
            (PRODUCT_DF["final_price"] <= user_budget) &
            (PRODUCT_DF["rating"] >= user_rating)
        ].copy()
        if len(pool) == 0:
            st.warning("No products match. Try relaxing budget or rating filters.")
        else:
            pool["score"] = (0.6*(pool["rating"]/5) + 0.4*(pool["discount_pct"]/40)).round(3)
            recs = pool.sort_values("score", ascending=False).head(6)
            st.session_state.ai_log.append({"time":datetime.now().strftime("%H:%M:%S"),"budget":user_budget,"cats":str(user_cats),"results":len(recs)})
            st.success(f"{len(recs)} personalised recommendations generated.")
            st.markdown("---")
            cols3 = st.columns(3)
            for i, (_, r) in enumerate(recs.iterrows()):
                with cols3[i%3]:
                    st.markdown(f"""
<div class="rec-card">
  <div class="rec-name">{ICONS.get(r['category'],'◈')} {r['name']}</div>
  <div class="rec-meta">{r['brand']} · {r['category']}</div>
  <div class="rec-meta">Rs {r['final_price']:,} &nbsp;|&nbsp; {r['rating']} ★</div>
  <div class="rec-meta" style="color:#5ac888 !important">{r['discount_pct']}% off &nbsp;·&nbsp; {r['stock']} in stock</div>
</div>""", unsafe_allow_html=True)

    with st.expander("Recommendation Session Log"):
        if st.session_state.ai_log:
            st.dataframe(pd.DataFrame(st.session_state.ai_log), use_container_width=True)
        else:
            st.info("No searches in this session yet.")

# ══════════════════════════════════════════════════════════════
# PAGE: CUSTOMER INSIGHTS
# ══════════════════════════════════════════════════════════════
elif page == "Customer Insights":
    st.markdown('<div class="page-header"><h1>Customer Intelligence</h1><p>Segmentation, lifetime value, churn risk</p></div>', unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total Customers",    f"{len(df_cust):,}")
    c2.metric("Avg Lifetime Value", f"Rs {df_cust['total_spent'].mean():,.0f}")
    c3.metric("Avg Orders / Cust",  f"{df_cust['orders'].mean():.1f}")
    c4.metric("Premium Customers",  f"{(df_cust['segment']=='Premium').sum():,}")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-title">Customer Segments</div>', unsafe_allow_html=True)
        seg = df_cust["segment"].value_counts().reset_index(); seg.columns = ["Segment","Count"]
        fig = px.pie(seg, names="Segment", values="Count", hole=0.5,
                     color_discrete_sequence=["#c0392b","#7b241c","#55555f","#2a2a32"])
        fig.update_traces(textfont_color="#f0eee8")
        st.plotly_chart(make_fig(fig, 320), use_container_width=True)
    with col2:
        st.markdown('<div class="section-title">Age vs Spend</div>', unsafe_allow_html=True)
        fig = px.scatter(df_cust, x="age", y="total_spent", color="segment", size="orders", opacity=0.72,
                         color_discrete_sequence=["#c0392b","#7b241c","#922b21","#55555f"])
        st.plotly_chart(make_fig(fig, 320), use_container_width=True)

    st.markdown('<div class="section-title">At-Risk Customer Alerts</div>', unsafe_allow_html=True)
    at_risk = df_cust[df_cust["segment"]=="At-Risk"].sort_values("total_spent", ascending=False)
    if len(at_risk):
        st.dataframe(at_risk[["customer_id","age","total_spent","orders","avg_rating"]].head(10), use_container_width=True, hide_index=True)
        st.warning(f"{len(at_risk)} customers flagged as at-risk of churning. Recommend targeted retention campaign.")

    st.markdown("---")
    st.markdown('<div class="section-title">Submit Feedback</div>', unsafe_allow_html=True)
    with st.form("feedback_form"):
        cid, fb = st.columns(2)
        cust_id  = cid.text_input("Customer ID", placeholder="CUST-1234")
        feedback = fb.text_input("Feedback")
        stars    = st.slider("Rating", 1, 5, 4)
        if st.form_submit_button("Submit →", use_container_width=True):
            if feedback:
                st.session_state.feedback[cust_id or "Anonymous"] = {"feedback":feedback,"stars":stars}
                st.success("Feedback recorded.")
    if st.session_state.feedback:
        with st.expander("All Feedback"):
            st.json(st.session_state.feedback)

# ══════════════════════════════════════════════════════════════
# PAGE: GEOGRAPHIC VIEW
# ══════════════════════════════════════════════════════════════
elif page == "Geographic View":
    st.markdown('<div class="page-header"><h1>Geographic Distribution</h1><p>Order heatmap across India</p></div>', unsafe_allow_html=True)
    col1, col2 = st.columns([3,1])
    with col2:
        map_cat = st.selectbox("Category", ["All"]+all_cats)
        map_min = st.number_input("Min Revenue", 0, 5000, 0, 100)
    map_df = df.copy()
    if map_cat != "All": map_df = map_df[map_df["category"]==map_cat]
    map_df = map_df[map_df["revenue"]>=map_min].dropna(subset=["lat","lon"])
    with col1:
        st.map(map_df[["lat","lon"]].rename(columns={"lat":"latitude","lon":"longitude"}))

    st.markdown("---")
    st.markdown('<div class="section-title">Revenue by Region</div>', unsafe_allow_html=True)
    reg = df.groupby("region")["revenue"].sum().reset_index()
    fig = px.bar(reg, x="region", y="revenue", color="revenue", color_continuous_scale=CRIMSON_SCALE)
    fig.update_traces(marker_line_width=0)
    st.plotly_chart(make_fig(fig, 320), use_container_width=True)

# ── Footer ────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div style="text-align:center;font-size:0.65rem;color:#55555f;letter-spacing:0.15em;text-transform:uppercase;font-family:Rajdhani,sans-serif">ShopSmart AI &nbsp;·&nbsp; Built with Streamlit &nbsp;·&nbsp; NumPy &amp; Pandas</div>', unsafe_allow_html=True)
