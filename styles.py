"""CSS theme for the KONKAE dexterity assessment dashboard."""

DASHBOARD_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    .stApp {
        background-color: #0F172A;
        color: #F8FAFC;
        font-family: 'Inter', sans-serif;
    }
    [data-testid="stSidebar"] {
        background-color: #1E293B;
        border-right: 1px solid #334155;
    }
    [data-testid="stSidebar"] .stMarkdown h2, [data-testid="stSidebar"] label {
        color: #E2E8F0 !important;
        font-family: 'Outfit', sans-serif;
    }

    /* --- Global readable typography on the dark theme --- */
    h1, h2, h3, h4, h5, h6 {
        color: #F8FAFC !important;
        font-family: 'Outfit', sans-serif;
    }
    .stApp p, .stApp li, .stApp span, .stApp label,
    [data-testid="stMarkdownContainer"] p {
        color: #E2E8F0;
    }
    [data-testid="stCaptionContainer"], .stCaption, small {
        color: #94A3B8 !important;
    }
    /* Form widget labels + values (selectbox, slider, number, uploader) */
    .stApp [data-baseweb="select"] *,
    .stApp [data-testid="stWidgetLabel"] *,
    .stApp .stSlider label,
    .stApp [data-testid="stNumberInput"] input,
    .stApp [data-testid="stTextInput"] input {
        color: #E2E8F0 !important;
    }
    /* Selectbox / inputs: solid dark surface so options never clash with bg */
    .stApp [data-baseweb="select"] > div,
    .stApp [data-testid="stNumberInput"] input,
    .stApp [data-testid="stTextInput"] input {
        background-color: #0F172A !important;
        border: 1px solid #334155 !important;
    }
    /* Dropdown popover */
    [data-baseweb="popover"] [role="listbox"] {
        background-color: #1E293B !important;
        color: #E2E8F0 !important;
    }
    [data-baseweb="popover"] [role="option"]:hover {
        background-color: #334155 !important;
    }

    /* --- Alert boxes: enforce contrast (Streamlit default text can wash out) --- */
    [data-testid="stAlert"] {
        border-radius: 10px;
        border: 1px solid #334155;
    }
    [data-testid="stAlert"] * { color: #F1F5F9 !important; }

    /* Primary buttons */
    .stApp .stButton > button,
    .stApp .stDownloadButton > button {
        background-color: #0EA5E9;
        color: #0F172A !important;
        border: none;
        font-weight: 700;
        border-radius: 10px;
        transition: all 0.2s ease-in-out;
    }
    .stApp .stButton > button:hover,
    .stApp .stDownloadButton > button:hover {
        background-color: #38BDF8;
        transform: translateY(-1px);
        box-shadow: 0 4px 14px rgba(56,189,248,0.35);
    }

    /* DataFrame / table legibility */
    .stApp [data-testid="stDataFrame"] { background-color: #1E293B; }

    /* Video preview frame */
    .preview-card {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 14px;
        padding: 12px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    .preview-title {
        color: #38BDF8;
        font-family: 'Outfit', sans-serif;
        font-weight: 600;
        font-size: 14px;
        margin: 0 0 8px 0;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .section-divider {
        border: none;
        border-top: 1px solid #1E293B;
        margin: 18px 0;
    }

    .log-container {
        height: 380px;
        overflow-y: auto;
        font-family: 'JetBrains Mono', monospace;
        font-size: 11.5px;
        background-color: #090D16;
        color: #38BDF8;
        padding: 14px;
        border: 1px solid #1E293B;
        border-radius: 12px;
        line-height: 1.6;
        box-shadow: inset 0 0 10px rgba(0,0,0,0.5);
    }
    .report-card {
        background-color: #1E293B;
        border: 1px solid #334155;
        padding: 24px;
        border-radius: 16px;
        margin-top: 15px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.25);
    }
    .metric-card {
        background-color: #1E293B;
        border: 1px solid #334155;
        padding: 14px;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.15);
        transition: all 0.2s ease-in-out;
    }
    .metric-card:hover {
        border-color: #475569;
        transform: translateY(-2px);
    }
    .metric-val {
        font-size: 26px;
        font-weight: 700;
        color: #38BDF8;
        font-family: 'Outfit', sans-serif;
    }
    .badge-left {
        background-color: rgba(56, 189, 248, 0.15);
        color: #38BDF8;
        border: 1px solid rgba(56, 189, 248, 0.3);
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 12px;
    }
    .badge-right {
        background-color: rgba(251, 146, 60, 0.15);
        color: #FB923C;
        border: 1px solid rgba(251, 146, 60, 0.3);
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 12px;
    }
    .status-badge-recording {
        background-color: rgba(239, 68, 68, 0.15);
        color: #F87171;
        border: 1px solid rgba(239, 68, 68, 0.3);
        padding: 6px 12px;
        border-radius: 8px;
        font-weight: 600;
        text-align: center;
    }
    .status-badge-demo {
        background-color: rgba(245, 158, 11, 0.15);
        color: #FBBF24;
        border: 1px solid rgba(245, 158, 11, 0.3);
        padding: 6px 12px;
        border-radius: 8px;
        font-weight: 600;
        text-align: center;
    }
</style>
"""
