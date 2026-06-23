import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile
import os
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="KONKAE.COM",
    page_icon="🦾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS Styling for Dashboard ---
st.markdown("""
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
""", unsafe_allow_html=True)


# --- Helper Math Functions ---
def calculate_shoulder_angle(shoulder, elbow, hip):
    """Calculates shoulder abduction/elevation angle relative to vertical torso."""
    v_arm = np.array([elbow[0] - shoulder[0], elbow[1] - shoulder[1]])
    v_torso = np.array([hip[0] - shoulder[0], hip[1] - shoulder[1]])
    
    norm_arm = np.linalg.norm(v_arm)
    norm_torso = np.linalg.norm(v_torso)
    
    if norm_arm == 0 or norm_torso == 0:
        return 0.0
        
    dot_product = np.dot(v_arm, v_torso)
    cos_theta = dot_product / (norm_arm * norm_torso)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    
    theta = np.arccos(cos_theta)
    return float(np.degrees(theta))


class TemporalSmoother:
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.history = []
        
    def smooth(self, pt):
        """pt is a dictionary with keys 'x', 'y', 'z'."""
        self.history.append(pt)
        if len(self.history) > self.window_size:
            self.history.pop(0)
        avg_x = sum(p['x'] for p in self.history) / len(self.history)
        avg_y = sum(p['y'] for p in self.history) / len(self.history)
        avg_z = sum(p['z'] for p in self.history) / len(self.history)
        return {'x': avg_x, 'y': avg_y, 'z': avg_z}


def dist_3d_norm(p1, p2):
    """Calculates 3D Euclidean distance in normalized coordinate space."""
    dx = p1['x'] - p2['x']
    dy = p1['y'] - p2['y']
    dz = p1['z'] - p2['z']
    return float(np.sqrt(dx**2 + dy**2 + dz**2))


def get_kinematics_card_html(
    frame_idx,
    fps,
    total_hits,
    left_speeds,
    right_speeds,
    left_jerks,
    right_jerks,
    left_rom_min,
    left_rom_max,
    right_rom_min,
    right_rom_max,
    left_current_rom,
    right_current_rom,
    left_straightness_val,
    right_straightness_val,
    dominant_side
):
    duration_sec = frame_idx / fps if fps > 0 else 0.0
    
    # Calculate stats
    left_avg_speed = np.mean(left_speeds) if left_speeds else 0.0
    right_avg_speed = np.mean(right_speeds) if right_speeds else 0.0
    left_max_speed = np.max(left_speeds) if left_speeds else 0.0
    right_max_speed = np.max(right_speeds) if right_speeds else 0.0
    left_avg_jerk = np.mean(left_jerks) if left_jerks else 0.0
    right_avg_jerk = np.mean(right_jerks) if right_jerks else 0.0
    
    left_rom_range = left_rom_max - left_rom_min if left_rom_max != float('-inf') and left_rom_min != float('inf') else 0.0
    right_rom_range = right_rom_max - right_rom_min if right_rom_max != float('-inf') and right_rom_min != float('inf') else 0.0
    
    # SVG circular gauge properties
    if total_hits > 0:
        stroke_color = "#34d399" # green
        glow_color = "rgba(52, 211, 153, 0.3)"
        offset = 0.0
    else:
        stroke_color = "#64748b" # gray
        glow_color = "rgba(100, 116, 139, 0.15)"
        offset = 301.59
        
    dominant_side_text = dominant_side.upper()
    
    return f"""<div style="background-color: #1E293B; border: 1px solid #334155; border-radius: 16px; padding: 20px; font-family: 'Inter', sans-serif; box-shadow: 0 4px 20px rgba(0,0,0,0.3); width: 100%; margin: 10px 0;">
<!-- Header -->
<div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 15px; border-bottom: 1px solid #334155; padding-bottom: 10px;">
<div>
<h3 style="margin: 0; font-size: 14px; font-weight: 700; color: #E2E8F0; letter-spacing: 0.05em; text-transform: uppercase;">ARM KINEMATICS</h3>
<p style="margin: 4px 0 0 0; font-size: 11px; color: #94A3B8; font-family: 'JetBrains Mono', monospace;">{frame_idx} frames · {duration_sec:.1f}s</p>
</div>
<div style="display: flex; gap: 12px; font-size: 11px; font-weight: 700; text-transform: uppercase; margin-top: 4px;">
<span style="color: #22D3EE; display: flex; align-items: center; gap: 4px;"><span style="display: inline-block; width: 6px; height: 6px; border-radius: 50%; background-color: #22D3EE;"></span>LEFT</span>
<span style="color: #FB7185; display: flex; align-items: center; gap: 4px;"><span style="display: inline-block; width: 6px; height: 6px; border-radius: 50%; background-color: #FB7185;"></span>RIGHT</span>
</div>
</div>

<!-- Hits Gauge -->
<div style="display: flex; justify-content: center; margin-bottom: 8px;">
<div style="position: relative; width: 120px; height: 120px; filter: drop-shadow(0 0 12px {glow_color});">
<svg width="120" height="120" viewBox="0 0 120 120" style="transform: rotate(-90deg); width: 120px; height: 120px;">
<!-- Track -->
<circle cx="60" cy="60" r="48" fill="none" stroke="rgba(255, 255, 255, 0.05)" stroke-width="8" />
<!-- Value -->
<circle cx="60" cy="60" r="48" fill="none" stroke="{stroke_color}" stroke-width="8" stroke-linecap="round"
stroke-dasharray="301.59" stroke-dashoffset="{offset}" style="transition: stroke-dashoffset 0.5s ease, stroke 0.5s ease;" />
</svg>
<div style="position: absolute; inset: 0; display: flex; flex-direction: column; justify-content: center; align-items: center;">
<span style="font-size: 26px; font-weight: 800; color: #FFFFFF; line-height: 1;">{total_hits}</span>
<span style="font-size: 9px; font-weight: 700; color: #94A3B8; letter-spacing: 0.05em; margin-top: 2px;">HITS</span>
</div>
</div>
</div>
<div style="text-align: center; font-size: 11px; font-weight: 700; color: #94A3B8; letter-spacing: 0.05em; text-transform: uppercase; margin-bottom: 12px;">TARGET HITS</div>

<!-- Dominant Side Badge -->
<div style="display: flex; justify-content: center; margin-bottom: 15px;">
<div style="border: 1px solid rgba(245, 158, 11, 0.4); background-color: rgba(245, 158, 11, 0.08); color: #FBBF24; font-size: 10px; font-weight: 700; letter-spacing: 0.05em; padding: 4px 12px; border-radius: 20px; text-transform: uppercase;">
DOMINANT: {dominant_side_text} ARM
</div>
</div>

<!-- Metrics List -->
<div style="display: flex; flex-direction: column;">
<!-- Row: AVG SPEED -->
<div style="display: flex; justify-content: space-between; align-items: center; padding: 8px 0; border-bottom: 1px solid rgba(255,255,255,0.05);">
<span style="font-size: 10px; font-weight: 600; color: #94A3B8; letter-spacing: 0.05em;">AVG SPEED</span>
<div style="display: flex; gap: 16px;">
<span style="color: #22D3EE; font-family: 'JetBrains Mono', monospace; font-size: 12px; font-weight: 700; width: 75px; text-align: right;">{left_avg_speed:.1f}<span style="font-size: 8px; font-weight: 500; color: #64748B; margin-left: 2px;">px/s</span></span>
<span style="color: #FB7185; font-family: 'JetBrains Mono', monospace; font-size: 12px; font-weight: 700; width: 75px; text-align: right;">{right_avg_speed:.1f}<span style="font-size: 8px; font-weight: 500; color: #64748B; margin-left: 2px;">px/s</span></span>
</div>
</div>

<!-- Row: MAX SPEED -->
<div style="display: flex; justify-content: space-between; align-items: center; padding: 8px 0; border-bottom: 1px solid rgba(255,255,255,0.05);">
<span style="font-size: 10px; font-weight: 600; color: #94A3B8; letter-spacing: 0.05em;">MAX SPEED</span>
<div style="display: flex; gap: 16px;">
<span style="color: #22D3EE; font-family: 'JetBrains Mono', monospace; font-size: 12px; font-weight: 700; width: 75px; text-align: right;">{left_max_speed:.1f}<span style="font-size: 8px; font-weight: 500; color: #64748B; margin-left: 2px;">px/s</span></span>
<span style="color: #FB7185; font-family: 'JetBrains Mono', monospace; font-size: 12px; font-weight: 700; width: 75px; text-align: right;">{right_max_speed:.1f}<span style="font-size: 8px; font-weight: 500; color: #64748B; margin-left: 2px;">px/s</span></span>
</div>
</div>

<!-- Row: AVG JERK -->
<div style="display: flex; justify-content: space-between; align-items: center; padding: 8px 0; border-bottom: 1px solid rgba(255,255,255,0.05);">
<span style="font-size: 10px; font-weight: 600; color: #94A3B8; letter-spacing: 0.05em;">AVG JERK</span>
<div style="display: flex; gap: 16px;">
<span style="color: #22D3EE; font-family: 'JetBrains Mono', monospace; font-size: 12px; font-weight: 700; width: 75px; text-align: right;">{left_avg_jerk:.1f}<span style="font-size: 8px; font-weight: 500; color: #64748B; margin-left: 2px;">px/s³</span></span>
<span style="color: #FB7185; font-family: 'JetBrains Mono', monospace; font-size: 12px; font-weight: 700; width: 75px; text-align: right;">{right_avg_jerk:.1f}<span style="font-size: 8px; font-weight: 500; color: #64748B; margin-left: 2px;">px/s³</span></span>
</div>
</div>

<!-- Row: ROM RANGE -->
<div style="display: flex; justify-content: space-between; align-items: center; padding: 8px 0; border-bottom: 1px solid rgba(255,255,255,0.05);">
<span style="font-size: 10px; font-weight: 600; color: #94A3B8; letter-spacing: 0.05em;">ROM RANGE</span>
<div style="display: flex; gap: 16px;">
<span style="color: #22D3EE; font-family: 'JetBrains Mono', monospace; font-size: 12px; font-weight: 700; width: 75px; text-align: right;">{left_rom_range:.0f}<span style="font-size: 8px; font-weight: 500; color: #64748B; margin-left: 2px;">°</span></span>
<span style="color: #FB7185; font-family: 'JetBrains Mono', monospace; font-size: 12px; font-weight: 700; width: 75px; text-align: right;">{right_rom_range:.0f}<span style="font-size: 8px; font-weight: 500; color: #64748B; margin-left: 2px;">°</span></span>
</div>
</div>

<!-- Row: CURRENT ROM -->
<div style="display: flex; justify-content: space-between; align-items: center; padding: 8px 0; border-bottom: 1px solid rgba(255,255,255,0.05);">
<span style="font-size: 10px; font-weight: 600; color: #94A3B8; letter-spacing: 0.05em;">CURRENT ROM</span>
<div style="display: flex; gap: 16px;">
<span style="color: #22D3EE; font-family: 'JetBrains Mono', monospace; font-size: 12px; font-weight: 700; width: 75px; text-align: right;">{left_current_rom:.0f}<span style="font-size: 8px; font-weight: 500; color: #64748B; margin-left: 2px;">°</span></span>
<span style="color: #FB7185; font-family: 'JetBrains Mono', monospace; font-size: 12px; font-weight: 700; width: 75px; text-align: right;">{right_current_rom:.0f}<span style="font-size: 8px; font-weight: 500; color: #64748B; margin-left: 2px;">°</span></span>
</div>
</div>

<!-- Row: STRAIGHTNESS -->
<div style="display: flex; justify-content: space-between; align-items: center; padding: 8px 0; border-bottom: 1px solid rgba(255,255,255,0.05);">
<span style="font-size: 10px; font-weight: 600; color: #94A3B8; letter-spacing: 0.05em;">STRAIGHTNESS</span>
<div style="display: flex; gap: 16px;">
<span style="color: #22D3EE; font-family: 'JetBrains Mono', monospace; font-size: 12px; font-weight: 700; width: 75px; text-align: right;">{left_straightness_val:.0f}<span style="font-size: 8px; font-weight: 500; color: #64748B; margin-left: 2px;">%</span></span>
<span style="color: #FB7185; font-family: 'JetBrains Mono', monospace; font-size: 12px; font-weight: 700; width: 75px; text-align: right;">{right_straightness_val:.0f}<span style="font-size: 8px; font-weight: 500; color: #64748B; margin-left: 2px;">%</span></span>
</div>
</div>

<!-- Row: DOMINANT SIDE -->
<div style="display: flex; justify-content: space-between; align-items: center; padding: 10px 0; border-bottom: none;">
<span style="font-size: 10px; font-weight: 700; color: #94A3B8; letter-spacing: 0.05em;">DOMINANT SIDE</span>
<span style="color: #FBBF24; font-family: 'JetBrains Mono', monospace; font-size: 12px; font-weight: 700; text-transform: uppercase;">{dominant_side_text}</span>
</div>
</div>
</div>"""



# --- App Headers ---
st.markdown("<h4 style='color: #94A3B8; margin-bottom: 0; font-family: Outfit, sans-serif; font-weight: 400;'>By Toto and King</h4>", unsafe_allow_html=True)
st.markdown("<h1 style='margin-top: 0; color: #FFFFFF; font-family: Outfit, sans-serif; font-weight: 700;'>KONKAE.COM</h1>", unsafe_allow_html=True)

# --- Sidebar ---
st.sidebar.markdown("""<div style="background: rgba(6, 182, 212, 0.1); border: 1px solid rgba(6, 182, 212, 0.3); padding: 12px; border-radius: 10px; margin-bottom: 15px; text-align: center;">
<span style="color: #38BDF8; font-weight: bold; font-size: 13px; display: block; margin-bottom: 8px;">🌐 ระบบประเมินร่วมทางคลินิก</span>
<a href="http://localhost:3000" target="_self" style="text-decoration: none;">
<button style="background: #06B6D4; color: #0F172A; border: none; padding: 8px 12px; border-radius: 8px; font-weight: bold; font-size: 12px; cursor: pointer; transition: 0.3s; width: 100%;">
🎥 เปิดระบบกล้องสด (Live Webcam)
</button>
</a>
</div>""", unsafe_allow_html=True)

st.sidebar.title("Assessment Settings")

min_frames_in_box = st.sidebar.slider("Min Frames for HIT (Sensitivity)", 2, 20, 3, step=1)
skip_seconds = st.sidebar.slider("ข้ามช่วงเริ่มต้นวิดีโอ/สอนใช้งาน (วินาที)", 0.0, 30.0, 0.0, step=0.5)
target_color = st.sidebar.selectbox("สีกล่องเป้าหมาย (Target Color)", ["Green", "Red", "Blue"])
ref_point_mode = st.sidebar.selectbox("จุดอ้างอิงของมือ (Tracking Point)", ["Wrist", "Index Finger Tip"])

mirror_view = st.sidebar.checkbox("กลับด้านวิดีโอ (Mirror/Flip Video)", value=False)
save_video = st.sidebar.checkbox("Save Annotated Video for Download", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("📐 Grid Bounds Alignment")
grid_preset = st.sidebar.selectbox("Grid Layout Preset", ["Fit Video Frame (Default)", "Center 4:3 (Pillarbox)", "Custom Margins"])

if grid_preset == "Center 4:3 (Pillarbox)":
    left_margin, right_margin, top_margin, bottom_margin = 12.5, 12.5, 0.0, 0.0
elif grid_preset == "Custom Margins":
    left_margin = st.sidebar.slider("Left Margin (%)", 0.0, 50.0, 12.5, step=0.5)
    right_margin = st.sidebar.slider("Right Margin (%)", 0.0, 50.0, 12.5, step=0.5)
    top_margin = st.sidebar.slider("Top Margin (%)", 0.0, 50.0, 0.0, step=0.5)
    bottom_margin = st.sidebar.slider("Bottom Margin (%)", 0.0, 50.0, 0.0, step=0.5)
else: # Fit Video Frame
    left_margin, right_margin, top_margin, bottom_margin = 0.0, 0.0, 0.0, 0.0

uploaded_file = st.sidebar.file_uploader("Upload Assessment Video", type=["mp4", "avi", "mov", "webm"])

if uploaded_file is not None:
    st.sidebar.success("Video Loaded successfully!")
else:
    st.sidebar.info("Upload a video to start the assessment.")

# --- Layout Columns ---
col1, col2 = st.columns([1.8, 1.2])

with col1:
    st.subheader("Assessment Frame Feed")
    frame_placeholder = st.empty()
    progress_bar = st.progress(0)
    status_placeholder = st.empty()

with col2:
    st.subheader("Real-Time Telemetry & Status")
    status_col1, status_col2 = st.columns(2)
    with status_col1:
        st.markdown("<b>สถานะการบันทึก:</b>", unsafe_allow_html=True)
        rec_status_placeholder = st.empty()
    with status_col2:
        st.markdown("<b>เป้าหมายขณะนี้:</b>", unsafe_allow_html=True)
        active_cell_placeholder = st.empty()
        
    st.markdown("<hr style='margin: 10px 0; border-color: #143D66;'>", unsafe_allow_html=True)
    
    st.markdown("<hr style='margin: 10px 0; border-color: #143D66;'>", unsafe_allow_html=True)
    
    # Kinematics Dashboard Card Placeholder
    kinematics_card_placeholder = st.empty()
    
    st.markdown("<hr style='margin: 10px 0; border-color: #143D66;'>", unsafe_allow_html=True)

    st.subheader("Real-Time Event Logs")
    log_filter = st.selectbox(
        "กรองประเภท Log (Filter Logs)",
        ["ทั้งหมด (Show All)", "เฉพาะเป้าหมายปรากฏ (Appearances)", "เฉพาะการชนเป้าหมาย (Hits)", "เฉพาะเป้าหมายพลาด (Misses)", "เฉพาะการยกแขน (Arm Raises)"],
        key="log_filter_select"
    )
    log_placeholder = st.empty()
    log_download_placeholder = st.empty()

# Initialize log lists in session state for persistence across Streamlit reruns
if "log_lines" not in st.session_state:
    st.session_state.log_lines = [
        "Initializing Automated Dexterity Assessment Pipeline...",
        "Ready. Upload a video and press 'Start Analysis'."
    ]

def filter_and_format_logs(lines, filter_type):
    filtered_lines = []
    for line in lines:
        if filter_type == "เฉพาะเป้าหมายปรากฏ (Appearances)" and "appeared" not in line:
            continue
        elif filter_type == "เฉพาะการชนเป้าหมาย (Hits)" and "HIT" not in line:
            continue
        elif filter_type == "เฉพาะเป้าหมายพลาด (Misses)" and "disappeared" not in line:
            continue
        elif filter_type == "เฉพาะการยกแขน (Arm Raises)" and "Raise" not in line:
            continue
        
        # Color highlighting based on log categories
        if "appeared" in line:
            colored = f"<span style='color: #38BDF8;'>{line}</span>"
        elif "HIT" in line:
            colored = f"<span style='color: #34D399; font-weight: bold;'>{line}</span>"
        elif "disappeared" in line:
            colored = f"<span style='color: #F87171;'>{line}</span>"
        elif "Raise" in line:
            colored = f"<span style='color: #FBBF24;'>{line}</span>"
        else:
            colored = f"<span style='color: #94A3B8;'>{line}</span>"
        filtered_lines.append(colored)
    return filtered_lines

def render_logs(lines, filter_type, show_download=True, key=None):
    colored_lines = filter_and_format_logs(lines, filter_type)
    formatted = "<br>".join(colored_lines[::-1])  # Reverse logs (latest on top)
    log_placeholder.markdown(f"<div class='log-container'>{formatted}</div>", unsafe_allow_html=True)
    
    if show_download:
        # Text-only format download of the full log lines
        log_text = "\n".join(lines)
        button_key = key if key is not None else f"dl_btn_{len(lines)}"
        log_download_placeholder.download_button(
            label="📥 ดาวน์โหลด Event Logs (.txt)",
            data=log_text,
            file_name="dexterity_event_logs.txt",
            mime="text/plain",
            key=button_key
        )
    else:
        log_download_placeholder.empty()

log_lines = st.session_state.log_lines
render_logs(log_lines, log_filter, key="dl_btn_top")
rec_status_placeholder.markdown("<div class='status-badge-demo'>WAITING</div>", unsafe_allow_html=True)
active_cell_placeholder.markdown("<div class='metric-card' style='padding:5px; margin:0;'>ไม่มี</div>", unsafe_allow_html=True)

# Initialize live kinematics card
initial_card_html = get_kinematics_card_html(
    frame_idx=0,
    fps=30.0,
    total_hits=0,
    left_speeds=[],
    right_speeds=[],
    left_jerks=[],
    right_jerks=[],
    left_rom_min=float('inf'),
    left_rom_max=float('-inf'),
    right_rom_min=float('inf'),
    right_rom_max=float('-inf'),
    left_current_rom=0.0,
    right_current_rom=0.0,
    left_straightness_val=0.0,
    right_straightness_val=0.0,
    dominant_side="WAITING"
)
kinematics_card_placeholder.markdown(initial_card_html, unsafe_allow_html=True)

# Placeholder for final summary report
report_placeholder = st.empty()

# --- Telemetry Charts Section ---
st.markdown("<hr style='border-color: #143D66;'>", unsafe_allow_html=True)
st.subheader("📊 กราฟวิเคราะห์การเคลื่อนไหว (Kinematics Telemetry Charts)")

chart_col1, chart_col2, chart_col3 = st.columns(3)
with chart_col1:
    st.markdown("<b>ความเร็วของแขน (Arm Speed - px/s)</b>", unsafe_allow_html=True)
    speed_chart_placeholder = st.empty()
with chart_col2:
    st.markdown("<b>ความราบเรียบของข้อต่อ (Movement Jerk)</b>", unsafe_allow_html=True)
    jerk_chart_placeholder = st.empty()
with chart_col3:
    st.markdown("<b>องศาการขยับไหล่ (Shoulder ROM - degrees)</b>", unsafe_allow_html=True)
    rom_chart_placeholder = st.empty()

def update_telemetry_charts(history_list):
    if not history_list:
        return
    import pandas as pd
    df = pd.DataFrame(history_list)
    if 'Timestamp (sec)' in df.columns:
        df_speed = df[['Timestamp (sec)', 'Left Arm Speed (px/s)', 'Right Arm Speed (px/s)']].set_index('Timestamp (sec)')
        df_jerk = df[['Timestamp (sec)', 'Left Movement Jerk (px/s3)', 'Right Movement Jerk (px/s3)']].set_index('Timestamp (sec)')
        df_rom = df[['Timestamp (sec)', 'Left Shoulder Angle (deg)', 'Right Shoulder Angle (deg)']].set_index('Timestamp (sec)')
        
        df_speed.columns = ['Left Arm', 'Right Arm']
        df_jerk.columns = ['Left Arm', 'Right Arm']
        df_rom.columns = ['Left Arm', 'Right Arm']
        
        speed_chart_placeholder.line_chart(df_speed, height=200)
        jerk_chart_placeholder.line_chart(df_jerk, height=200)
        rom_chart_placeholder.line_chart(df_rom, height=200)

# Initialize chart placeholders
if st.session_state.get("analysis_completed", False) and st.session_state.get("frame_history"):
    update_telemetry_charts(st.session_state.frame_history)
else:
    speed_chart_placeholder.info("รอข้อมูลการเคลื่อนไหว...")
    jerk_chart_placeholder.info("รอข้อมูลการเคลื่อนไหว...")
    rom_chart_placeholder.info("รอข้อมูลการเคลื่อนไหว...")


# --- Processing Loop ---
if uploaded_file is not None:
    start_analysis = st.sidebar.button("Start Analysis", use_container_width=True)
    
    if start_analysis:
        st.session_state.analysis_completed = False
        st.session_state.report_html = None
        st.session_state.frame_history = []
        st.session_state.video_bytes = None
        
        # Save upload to temp file
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())
        video_path = tfile.name
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        st.session_state.log_lines = [
            "Initializing Dual Computer Vision ...",
            f"Video format: {width}x{height} @ {fps:.1f} fps",
            "Processing frames... (Pose & Target)"
        ]
        log_lines = st.session_state.log_lines
        render_logs(log_lines, log_filter, show_download=False)
        
        # Setup VideoWriter if needed
        writer = None
        out_path = None
        if save_video:
            out_tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            out_path = out_tfile.name
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

        # Setup MediaPipe Pose
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles

        # Tracking variables
        local_frame_history = []
        frame_idx = 0
        active_targets_map = {} # cell_num (1-9) -> target_instance_id
        
        # Hit stats
        hits_log = {}         # target_instance -> {'Left': bool, 'Right': bool}
        any_hit_logged = {}  # target_instance -> bool
        target_instance_counter = 0
        
        # Reaction times and velocity tracking
        trial_start_times = {} # target_instance -> start_time
        left_wrist_history = []
        right_wrist_history = []
        
        # Temporal smoothers for joint coordinates
        left_shoulder_smoother = TemporalSmoother(5)
        right_shoulder_smoother = TemporalSmoother(5)
        left_elbow_smoother = TemporalSmoother(5)
        right_elbow_smoother = TemporalSmoother(5)
        left_wrist_smoother = TemporalSmoother(5)
        right_wrist_smoother = TemporalSmoother(5)
        left_index_smoother = TemporalSmoother(5)
        right_index_smoother = TemporalSmoother(5)

        # Kinematic metrics tracking variables
        prev_left_wrist_smooth = None
        prev_right_wrist_smooth = None
        prev_left_speed = 0.0
        prev_right_speed = 0.0
        prev_left_accel = 0.0
        prev_right_accel = 0.0
        
        # Accumulators for physical speed and jerk
        left_speeds = []
        right_speeds = []
        left_jerks = []
        right_jerks = []
        
        # Range of motion bounds and current values
        left_rom_min = float('inf')
        left_rom_max = float('-inf')
        right_rom_min = float('inf')
        right_rom_max = float('-inf')
        left_current_rom = 0.0
        right_current_rom = 0.0
        
        # normalized wrist histories and path lengths for straightness calculation
        left_wrist_norm_history = []
        right_wrist_norm_history = []
        left_path_length = 0.0
        right_path_length = 0.0
        
        # Repetitions states
        left_arm_raised = False
        right_arm_raised = False
        left_raise_count = 0
        right_raise_count = 0
        max_left_angle = 0.0
        max_right_angle = 0.0
        
        # Jitter/Smoothness sum variables
        left_jitter_sum = 0.0
        right_jitter_sum = 0.0
        left_jitter_frames = 0
        right_jitter_frames = 0
        
        # Frame counter for consecutive frames in grid
        consecutive_frames_counter = {
            i: {'Left': 0, 'Right': 0} for i in range(1, 10)
        }
        
        # Hits/Misses counter
        total_hits = 0
        total_misses = 0
        left_hits_count = 0
        right_hits_count = 0
        reaction_times = []
        left_reaction_times = []
        right_reaction_times = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_idx += 1
            current_time_sec = frame_idx / fps
            is_recording = current_time_sec >= skip_seconds
            
            # Flip Frame if Mirror View is enabled
            if mirror_view:
                frame = cv2.flip(frame, 1)

            annotated_frame = frame.copy()
            
            # Calculate active grid area coordinates
            x_start = int(width * (left_margin / 100.0))
            x_end = int(width * (1.0 - right_margin / 100.0))
            y_start = int(height * (top_margin / 100.0))
            y_end = int(height * (1.0 - bottom_margin / 100.0))
            
            grid_w = x_end - x_start
            grid_h = y_end - y_start
            
            # 1. 3x3 Grid Target Coordinates Definition
            cell_w = grid_w // 3
            cell_h = grid_h // 3
            cell_area = cell_w * cell_h
            
            # Draw outer grid boundary on frame
            cv2.rectangle(annotated_frame, (x_start, y_start), (x_end, y_end), (120, 120, 120), 1)
            
            # Draw 3x3 Grid borders on frame
            for i in range(1, 3):
                cv2.line(annotated_frame, (x_start + i * cell_w, y_start), (x_start + i * cell_w, y_end), (120, 120, 120), 1)
                cv2.line(annotated_frame, (x_start, y_start + i * cell_h), (x_end, y_start + i * cell_h), (120, 120, 120), 1)
                
            # Number the grid cells 1 to 9 visually
            for r in range(3):
                for c in range(3):
                    cell_num = r * 3 + c + 1
                    cv2.putText(annotated_frame, str(cell_num), (x_start + c * cell_w + 10, y_start + r * cell_h + 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 120, 120), 1)

            # 2. Detect Active Targets by color threshold scan
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            if target_color == "Green":
                color_mask = cv2.inRange(hsv, np.array([35, 40, 40]), np.array([85, 255, 255]))
            elif target_color == "Blue":
                color_mask = cv2.inRange(hsv, np.array([100, 40, 40]), np.array([140, 255, 255]))
            else: # Red
                mask1 = cv2.inRange(hsv, np.array([0, 40, 40]), np.array([10, 255, 255]))
                mask2 = cv2.inRange(hsv, np.array([170, 40, 40]), np.array([180, 255, 255]))
                color_mask = mask1 | mask2
                
            current_active_cells = []
            
            for r in range(3):
                for c in range(3):
                    cell_num = r * 3 + c + 1
                    x1 = x_start + c * cell_w
                    y1 = y_start + r * cell_h
                    x2 = x_start + (c + 1) * cell_w
                    y2 = y_start + (r + 1) * cell_h
                    
                    cell_mask = color_mask[y1:y2, x1:x2]
                    pixel_count = np.sum(cell_mask > 0)
                    ratio = (pixel_count / cell_area) * 100
                    
                    # 10% area ratio threshold filters noise and catches target box activations perfectly
                    if ratio > 10.0:
                        current_active_cells.append(cell_num)
            
            # Active Cell Telemetry Display
            if current_active_cells:
                cells_str = ", ".join([f"ช่องที่ {c}" for c in current_active_cells])
                active_cell_placeholder.markdown(f"<div class='metric-card' style='padding:5px; margin:0; border-color:#EF4444; color:#EF4444;'>{cells_str}</div>", unsafe_allow_html=True)
            else:
                active_cell_placeholder.markdown("<div class='metric-card' style='padding:5px; margin:0;'>ไม่มี</div>", unsafe_allow_html=True)

            # Recording status telemetry
            if is_recording:
                rec_status_placeholder.markdown("<div class='status-badge-recording'>REC ACTIVE (บันทึกข้อมูล)</div>", unsafe_allow_html=True)
            else:
                rec_status_placeholder.markdown("<div class='status-badge-demo'>DEMO MODE (ช่วงสอน/สาธิต)</div>", unsafe_allow_html=True)
                
                # Draw a beautiful loader on the video frame
                overlay = annotated_frame.copy()
                cv2.rectangle(overlay, (0, 0), (width, height), (48, 27, 6), -1) # Dark navy
                alpha = 0.8
                annotated_frame = cv2.addWeighted(overlay, alpha, annotated_frame, 1 - alpha, 0)
                
                # Draw a loading text and progress bar
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(annotated_frame, "DEMO / TEACHING PHASE (ช่วงสาธิตการใช้งาน)", (width//2 - 220, height//2 - 40),
                            font, 0.6, (244, 162, 89), 2, cv2.LINE_AA)
                cv2.putText(annotated_frame, f"System will start analysis in {max(0.0, skip_seconds - current_time_sec):.1f}s", (width//2 - 170, height//2 - 10),
                            font, 0.5, (243, 244, 246), 1, cv2.LINE_AA)
                
                # Draw progress bar container
                bar_w = 360
                bar_h = 8
                bar_x = (width - bar_w) // 2
                bar_y = height // 2 + 15
                cv2.rectangle(annotated_frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (80, 80, 80), -1)
                fill_w = int(bar_w * (current_time_sec / max(0.1, skip_seconds)))
                cv2.rectangle(annotated_frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), (244, 162, 89), -1)
            
            # Target Appearance State Machine (only triggers if in recording phase)
            if is_recording:
                # Check for disappearances
                for cell_num in list(active_targets_map.keys()):
                    if cell_num not in current_active_cells:
                        inst_id = active_targets_map[cell_num]
                        if not any_hit_logged.get(inst_id, False):
                            log_lines.append(f"[Frame {frame_idx}] Target {inst_id} disappeared (MISSED)")
                            total_misses += 1
                        del active_targets_map[cell_num]
                        
                # Check for appearances
                for cell_num in current_active_cells:
                    if cell_num not in active_targets_map:
                        target_instance_counter += 1
                        active_targets_map[cell_num] = target_instance_counter
                        hits_log[target_instance_counter] = {'Left': False, 'Right': False}
                        any_hit_logged[target_instance_counter] = False
                        
                        col = (cell_num - 1) % 3
                        side = "LEFT" if col == 0 else "RIGHT" if col == 2 else "CENTER"
                        log_lines.append(f"[Frame {frame_idx}] Target {target_instance_counter} appeared on {side}")
                        trial_start_times[target_instance_counter] = frame_idx / fps
            else:
                # If during warm-up phase, clear active targets tracking to avoid spillover
                active_targets_map = {}
                
            # Draw border outlines on all active grid cells
            for cell_num, inst_id in active_targets_map.items():
                r = (cell_num - 1) // 3
                c = (cell_num - 1) % 3
                # Glow green if hit, red if not hit yet
                glow_color = (0, 255, 0) if any_hit_logged.get(inst_id, False) else (0, 0, 255)
                cv2.rectangle(annotated_frame, (x_start + c * cell_w + 3, y_start + r * cell_h + 3), 
                              (x_start + (c + 1) * cell_w - 3, y_start + (r + 1) * cell_h - 3), glow_color, 3)

            # Initialize frame kinematics variables
            left_speed = 0.0
            right_speed = 0.0
            left_jerk = 0.0
            right_jerk = 0.0
            left_angle = 0.0
            right_angle = 0.0

            # 3. AI Pose Tracking
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)
            
            left_wrist_pt = None
            right_wrist_pt = None
            
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                
                l_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
                l_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
                l_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
                l_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
                l_index = landmarks[mp_pose.PoseLandmark.LEFT_INDEX]
                
                r_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                r_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]
                r_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
                r_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
                r_index = landmarks[mp_pose.PoseLandmark.RIGHT_INDEX]
                
                # Convert to dictionaries for smoothing
                l_shoulder_dict = {'x': l_shoulder.x, 'y': l_shoulder.y, 'z': l_shoulder.z}
                r_shoulder_dict = {'x': r_shoulder.x, 'y': r_shoulder.y, 'z': r_shoulder.z}
                l_elbow_dict = {'x': l_elbow.x, 'y': l_elbow.y, 'z': l_elbow.z}
                r_elbow_dict = {'x': r_elbow.x, 'y': r_elbow.y, 'z': r_elbow.z}
                l_wrist_dict = {'x': l_wrist.x, 'y': l_wrist.y, 'z': l_wrist.z}
                r_wrist_dict = {'x': r_wrist.x, 'y': r_wrist.y, 'z': r_wrist.z}
                l_index_dict = {'x': l_index.x, 'y': l_index.y, 'z': l_index.z}
                r_index_dict = {'x': r_index.x, 'y': r_index.y, 'z': r_index.z}
                
                # Temporal smoothing
                l_shoulder_smooth = left_shoulder_smoother.smooth(l_shoulder_dict)
                r_shoulder_smooth = right_shoulder_smoother.smooth(r_shoulder_dict)
                l_elbow_smooth = left_elbow_smoother.smooth(l_elbow_dict)
                r_elbow_smooth = right_elbow_smoother.smooth(r_elbow_dict)
                l_wrist_smooth = left_wrist_smoother.smooth(l_wrist_dict)
                r_wrist_smooth = right_wrist_smoother.smooth(r_wrist_dict)
                l_index_smooth = left_index_smoother.smooth(l_index_dict)
                r_index_smooth = right_index_smoother.smooth(r_index_dict)
                
                # Calculate shoulder angles using smoothed coordinates
                left_angle = calculate_shoulder_angle(
                    (l_shoulder_smooth['x'], l_shoulder_smooth['y']),
                    (l_elbow_smooth['x'], l_elbow_smooth['y']),
                    (l_hip.x, l_hip.y)
                )
                right_angle = calculate_shoulder_angle(
                    (r_shoulder_smooth['x'], r_shoulder_smooth['y']),
                    (r_elbow_smooth['x'], r_elbow_smooth['y']),
                    (r_hip.x, r_hip.y)
                )
                
                if is_recording:
                    max_left_angle = max(max_left_angle, left_angle)
                    max_right_angle = max(max_right_angle, right_angle)
                    
                    left_rom_min = min(left_rom_min, left_angle)
                    left_rom_max = max(left_rom_max, left_angle)
                    right_rom_min = min(right_rom_min, right_angle)
                    right_rom_max = max(right_rom_max, right_angle)
                    
                    left_current_rom = left_angle
                    right_current_rom = right_angle
                    
                    left_wrist_norm_history.append(l_wrist_smooth)
                    right_wrist_norm_history.append(r_wrist_smooth)
                    
                    if len(left_wrist_norm_history) >= 2:
                        left_path_length += dist_3d_norm(left_wrist_norm_history[-1], left_wrist_norm_history[-2])
                    if len(right_wrist_norm_history) >= 2:
                        right_path_length += dist_3d_norm(right_wrist_norm_history[-1], right_wrist_norm_history[-2])
                    
                    # Left Arm Rep
                    if left_angle > 60.0:
                        if not left_arm_raised:
                            left_arm_raised = True
                            left_raise_count += 1
                            log_lines.append(f"[Frame {frame_idx}] Left Arm Raise #{left_raise_count} (Angle: {left_angle:.1f}°)")
                    elif left_angle < 30.0:
                        left_arm_raised = False
                        
                    # Right Arm Rep
                    if right_angle > 60.0:
                        if not right_arm_raised:
                            right_arm_raised = True
                            right_raise_count += 1
                            log_lines.append(f"[Frame {frame_idx}] Right Arm Raise #{right_raise_count} (Angle: {right_angle:.1f}°)")
                    elif right_angle < 30.0:
                        right_arm_raised = False
                
                # dist_3d_norm is defined globally
                
                dt_safe = 1.0 / fps if fps > 0 else 1.0 / 30.0
                
                # Instantaneous speed and jerk in normalized coordinates
                if prev_left_wrist_smooth is not None:
                    left_speed = dist_3d_norm(l_wrist_smooth, prev_left_wrist_smooth) / dt_safe
                    left_accel = abs(left_speed - prev_left_speed) / dt_safe
                    left_jerk = abs(left_accel - prev_left_accel) / dt_safe
                    prev_left_accel = left_accel
                else:
                    left_accel = 0.0
                    left_jerk = 0.0
                    prev_left_accel = 0.0
                    
                if prev_right_wrist_smooth is not None:
                    right_speed = dist_3d_norm(r_wrist_smooth, prev_right_wrist_smooth) / dt_safe
                    right_accel = abs(right_speed - prev_right_speed) / dt_safe
                    right_jerk = abs(right_accel - prev_right_accel) / dt_safe
                    prev_right_accel = right_accel
                else:
                    right_accel = 0.0
                    right_jerk = 0.0
                    prev_right_accel = 0.0
                    
                prev_left_wrist_smooth = l_wrist_smooth
                prev_right_wrist_smooth = r_wrist_smooth
                prev_left_speed = left_speed
                prev_right_speed = right_speed
                
                if is_recording:
                    left_speeds.append(left_speed)
                    right_speeds.append(right_speed)
                    left_jerks.append(left_jerk)
                    right_jerks.append(right_jerk)
                
                l_wrist_px = (int(l_wrist_smooth['x'] * width), int(l_wrist_smooth['y'] * height))
                r_wrist_px = (int(r_wrist_smooth['x'] * width), int(r_wrist_smooth['y'] * height))
                
                if is_recording:
                    left_wrist_history.append(l_wrist_px)
                    right_wrist_history.append(r_wrist_px)
                    
                    # Left Jitter (retained for backward compatibility of custom smoothness)
                    if len(left_wrist_history) >= 3:
                        v1 = np.array(left_wrist_history[-1]) - np.array(left_wrist_history[-2])
                        v2 = np.array(left_wrist_history[-2]) - np.array(left_wrist_history[-3])
                        left_jitter_sum += np.linalg.norm(v1 - v2)
                        left_jitter_frames += 1
                    # Right Jitter
                    if len(right_wrist_history) >= 3:
                        v1 = np.array(right_wrist_history[-1]) - np.array(right_wrist_history[-2])
                        v2 = np.array(right_wrist_history[-2]) - np.array(right_wrist_history[-3])
                        right_jitter_sum += np.linalg.norm(v1 - v2)
                        right_jitter_frames += 1
                
                # Set tracking coordinate points (using smoothed positions)
                if ref_point_mode == "Index Finger Tip":
                    left_wrist_pt = (int(l_index_smooth['x'] * width), int(l_index_smooth['y'] * height))
                    right_wrist_pt = (int(r_index_smooth['x'] * width), int(r_index_smooth['y'] * height))
                else:
                    left_wrist_pt = l_wrist_px
                    right_wrist_pt = r_wrist_px
                
                # Pose telemetry updated globally at the end of loop iteration

                # Draw skeleton overlays
                mp_drawing.draw_landmarks(
                    annotated_frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                )
                
            # 4. Collision Checking with 3x3 Grid target boxes (only if in recording phase)
            hands_pts = {'Left': left_wrist_pt, 'Right': right_wrist_pt}
            
            if is_recording:
                for cell_num, inst_id in list(active_targets_map.items()):
                    r = (cell_num - 1) // 3
                    c = (cell_num - 1) % 3
                    x1 = x_start + c * cell_w
                    y1 = y_start + r * cell_h
                    x2 = x_start + (c + 1) * cell_w
                    y2 = y_start + (r + 1) * cell_h
                    
                    for hand_side, pt in hands_pts.items():
                        if pt is not None:
                            px, py = pt
                            in_cell = x1 <= px <= x2 and y1 <= py <= y2
                            
                            if in_cell:
                                consecutive_frames_counter[cell_num][hand_side] += 1
                                
                                # Check if threshold is met and it is the active cell
                                if consecutive_frames_counter[cell_num][hand_side] >= min_frames_in_box:
                                    if not hits_log[inst_id][hand_side]:
                                        hits_log[inst_id][hand_side] = True
                                        any_hit_logged[inst_id] = True
                                        
                                        trial_start = trial_start_times.get(inst_id, frame_idx / fps)
                                        reaction = (frame_idx / fps) - trial_start
                                        log_lines.append(f"[Frame {frame_idx}] Target {inst_id} HIT by {hand_side.upper()} Hand")
                                        
                                        total_hits += 1
                                        reaction_times.append(reaction)
                                        if hand_side == 'Left':
                                            left_hits_count += 1
                                            left_reaction_times.append(reaction)
                                        else:
                                            right_hits_count += 1
                                            right_reaction_times.append(reaction)
                                            
                                        cv2.putText(annotated_frame, "HIT!", (x1 + 10, y1 + cell_h - 10),
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                                        
                                        consecutive_frames_counter[cell_num][hand_side] = 0
                            else:
                                consecutive_frames_counter[cell_num][hand_side] = 0
                
            # Draw tracking bubble label near wrist coordinates
            for hand_side, pt in hands_pts.items():
                if pt is not None:
                    px, py = pt
                    color = (200, 209, 31) if hand_side == 'Left' else (89, 162, 244)
                    
                    # Draw visual tracker cursor
                    cv2.circle(annotated_frame, (px, py), 10, color, -1)
                    cv2.circle(annotated_frame, (px, py), 15, (255, 255, 255), 2)
                    
                    # Label text
                    text = f"{'มือซ้าย' if hand_side == 'Left' else 'มือขวา'}"
                    cv2.putText(annotated_frame, text, (px - 25, py - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Output frame to dashboard
            display_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(display_frame, use_container_width=True)
            
            pct_complete = min(1.0, frame_idx / total_frames)
            progress_bar.progress(pct_complete)
            status_placeholder.text(f"Processed Frame: {frame_idx}/{total_frames} ({pct_complete * 100:.1f}%)")
            
            # Update live kinematics card
            left_speed_sum = sum(left_speeds) if left_speeds else 0.0
            right_speed_sum = sum(right_speeds) if right_speeds else 0.0
            if left_speed_sum == 0.0 and right_speed_sum == 0.0:
                dominant_side = "WAITING"
            else:
                dominant_side = "left" if left_speed_sum > right_speed_sum else "right"

            left_straightness_val = 0.0
            right_straightness_val = 0.0
            if left_wrist_norm_history:
                left_disp = dist_3d_norm(left_wrist_norm_history[-1], left_wrist_norm_history[0])
                left_straightness_val = (left_disp / left_path_length * 100) if left_path_length > 0.001 else 0.0
            if right_wrist_norm_history:
                right_disp = dist_3d_norm(right_wrist_norm_history[-1], right_wrist_norm_history[0])
                right_straightness_val = (right_disp / right_path_length * 100) if right_path_length > 0.001 else 0.0

            live_card_html = get_kinematics_card_html(
                frame_idx=frame_idx,
                fps=fps,
                total_hits=total_hits,
                left_speeds=left_speeds,
                right_speeds=right_speeds,
                left_jerks=left_jerks,
                right_jerks=right_jerks,
                left_rom_min=left_rom_min,
                left_rom_max=left_rom_max,
                right_rom_min=right_rom_min,
                right_rom_max=right_rom_max,
                left_current_rom=left_current_rom,
                right_current_rom=right_current_rom,
                left_straightness_val=left_straightness_val,
                right_straightness_val=right_straightness_val,
                dominant_side=dominant_side
            )
            kinematics_card_placeholder.markdown(live_card_html, unsafe_allow_html=True)
            
            render_logs(log_lines, log_filter, show_download=False)
            
            if writer:
                writer.write(annotated_frame)
                
            # Record frame history for frame lookup and CSV report (relative to active grid area)
            left_hand_x = (left_wrist_pt[0] - x_start) / grid_w * 100 if left_wrist_pt is not None else None
            left_hand_y = (left_wrist_pt[1] - y_start) / grid_h * 100 if left_wrist_pt is not None else None
            right_hand_x = (right_wrist_pt[0] - x_start) / grid_w * 100 if right_wrist_pt is not None else None
            right_hand_y = (right_wrist_pt[1] - y_start) / grid_h * 100 if right_wrist_pt is not None else None
            
            t_x1 = 0.0
            t_y1 = 0.0
            t_x2 = 0.0
            t_y2 = 0.0
            active_cells_list = list(active_targets_map.keys())
            primary_active_cell = active_cells_list[0] if active_cells_list else None
            t_x1 = 0.0
            t_y1 = 0.0
            t_x2 = 0.0
            t_y2 = 0.0
            if primary_active_cell is not None:
                tr = (primary_active_cell - 1) // 3
                tc = (primary_active_cell - 1) % 3
                t_x1 = tc * 33.33
                t_y1 = tr * 33.33
                t_x2 = (tc + 1) * 33.33
                t_y2 = (tr + 1) * 33.33
                
            left_hit_this_frame = any(hits_log.get(inst_id, {}).get('Left', False) for inst_id in active_targets_map.values())
            right_hit_this_frame = any(hits_log.get(inst_id, {}).get('Right', False) for inst_id in active_targets_map.values())
            
            local_frame_history.append({
                'Frame Index': frame_idx,
                'Timestamp (sec)': current_time_sec,
                'Active Target Cell': ", ".join(map(str, active_cells_list)) if active_cells_list else "None",
                'Target X1 (%)': t_x1,
                'Target Y1 (%)': t_y1,
                'Target X2 (%)': t_x2,
                'Target Y2 (%)': t_y2,
                'Left Hand X (%)': left_hand_x if left_hand_x is not None else "N/A",
                'Left Hand Y (%)': left_hand_y if left_hand_y is not None else "N/A",
                'Right Hand X (%)': right_hand_x if right_hand_x is not None else "N/A",
                'Right Hand Y (%)': right_hand_y if right_hand_y is not None else "N/A",
                'Left Hand Hit': "Yes" if left_hit_this_frame else "No",
                'Right Hand Hit': "Yes" if right_hit_this_frame else "No",
                'Left Arm Speed (px/s)': round(left_speed, 1) if (results.pose_landmarks and prev_left_wrist_smooth is not None) else 0.0,
                'Right Arm Speed (px/s)': round(right_speed, 1) if (results.pose_landmarks and prev_right_wrist_smooth is not None) else 0.0,
                'Left Movement Jerk (px/s3)': round(left_jerk, 1) if (results.pose_landmarks and prev_left_wrist_smooth is not None) else 0.0,
                'Right Movement Jerk (px/s3)': round(right_jerk, 1) if (results.pose_landmarks and prev_right_wrist_smooth is not None) else 0.0,
                'Left Shoulder Angle (deg)': round(left_angle, 1) if results.pose_landmarks else 0.0,
                'Right Shoulder Angle (deg)': round(right_angle, 1) if results.pose_landmarks else 0.0
            })
            
            if frame_idx % 30 == 0:
                update_telemetry_charts(local_frame_history)
                
        # --- End of Processing pipeline ---
        cap.release()
        if writer:
            writer.release()
        st.session_state.frame_history = local_frame_history
        
        # Final charts update
        update_telemetry_charts(local_frame_history)
            
        log_lines.append("Assessment completed successfully!")
        render_logs(log_lines, log_filter, key="dl_btn_finished")
        
        # Calculate kinematics summaries (Reaction Time and physical movement speed/jerk)
        left_avg_reaction_time = np.mean(left_reaction_times) if left_reaction_times else 0.0
        right_avg_reaction_time = np.mean(right_reaction_times) if right_reaction_times else 0.0
        
        left_smoothness = max(0, min(100, 100 - (left_jitter_sum / max(1, left_jitter_frames) * 1200)))
        right_smoothness = max(0, min(100, 100 - (right_jitter_sum / max(1, right_jitter_frames) * 1200)))
        
        # Calculate straightness
        def get_displacement_and_path_length(history):
            if len(history) < 2:
                return 0.0, 0.0
            def dist_3d_dict(a, b):
                return np.sqrt((a['x'] - b['x'])**2 + (a['y'] - b['y'])**2 + (a['z'] - b['z'])**2)
            start = history[0]
            end = history[-1]
            displacement = dist_3d_dict(start, end)
            path_length = 0.0
            for i in range(1, len(history)):
                path_length += dist_3d_dict(history[i], history[i-1])
            return displacement, path_length

        left_disp, left_path = get_displacement_and_path_length(left_wrist_norm_history)
        left_straightness_val = (left_disp / left_path if left_path > 0.001 else 1.0) * 100
        
        right_disp, right_path = get_displacement_and_path_length(right_wrist_norm_history)
        right_straightness_val = (right_disp / right_path if right_path > 0.001 else 1.0) * 100

        dominant_side_en = "LEFT" if left_speed_sum > right_speed_sum else "RIGHT"

        # New kinematics metrics summaries
        left_avg_speed_px = np.mean(left_speeds) if left_speeds else 0.0
        right_avg_speed_px = np.mean(right_speeds) if right_speeds else 0.0
        left_max_speed_px = np.max(left_speeds) if left_speeds else 0.0
        right_max_speed_px = np.max(right_speeds) if right_speeds else 0.0
        
        left_avg_jerk_px = np.mean(left_jerks) if left_jerks else 0.0
        right_avg_jerk_px = np.mean(right_jerks) if right_jerks else 0.0
        
        left_rom_range = left_rom_max - left_rom_min if left_rom_max != float('-inf') else 0.0
        right_rom_range = right_rom_max - right_rom_min if right_rom_max != float('-inf') else 0.0
        
        left_speed_sum = sum(left_speeds) if left_speeds else 0.0
        right_speed_sum = sum(right_speeds) if right_speeds else 0.0
        
        if left_speed_sum == 0.0 and right_speed_sum == 0.0:
            dominant_side_thai = "ไม่ระบุ (ไม่มีข้อมูลเคลื่อนไหว)"
        else:
            dominant_side_thai = "มือซ้าย (LEFT)" if left_speed_sum > right_speed_sum else "มือขวา (RIGHT)"
            
        # Learned Non-Use Risk Assessment
        lnu_risk = "Low (ความเสี่ยงต่ำ)"
        lnu_color = "#1FD1C8"
        
        total_trials = target_instance_counter
        
        if total_trials > 2:
            left_ratio = left_hits_count / max(1, total_hits)
            right_ratio = right_hits_count / max(1, total_hits)
            
            if (right_ratio > 0.8 and left_hits_count <= 1 and max_left_angle < 45.0) or \
               (left_ratio > 0.8 and right_hits_count <= 1 and max_right_angle < 45.0):
                lnu_risk = "High (ความเสี่ยงสูง - ตรวจพบภาวะฝืนไม่ใช้งานแขนข้างที่อ่อนแรง)"
                lnu_color = "#EF4444"
            elif (right_ratio > 0.65 and left_hits_count <= 2) or (left_ratio > 0.65 and right_hits_count <= 2):
                lnu_risk = "Moderate (ความเสี่ยงปานกลาง - มีแนวโน้มชดเชยการใช้กำลังสองฝั่งไม่สมดุล)"
                lnu_color = "#F59E0B"
                
        # Display Final Assessment Report matching the real-time panel style
        report_html = f"""<div style="background-color: #1E293B; border: 1px solid #334155; border-radius: 16px; padding: 24px; font-family: 'Inter', sans-serif; box-shadow: 0 4px 20px rgba(0,0,0,0.3); max-width: 600px; margin: 20px auto;">
<!-- Header -->
<div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 20px; border-bottom: 1px solid #334155; padding-bottom: 12px;">
<div>
<h3 style="margin: 0; font-size: 15px; font-weight: 700; color: #E2E8F0; letter-spacing: 0.05em; text-transform: uppercase;">ARM KINEMATICS</h3>
<p style="margin: 4px 0 0 0; font-size: 11px; color: #94A3B8; font-family: 'JetBrains Mono', monospace;">{frame_idx} frames · {frame_idx / fps:.1f}s</p>
</div>
<div style="display: flex; gap: 12px; font-size: 11px; font-weight: 700; text-transform: uppercase; margin-top: 4px;">
<span style="color: #22D3EE; display: flex; align-items: center; gap: 4px;"><span style="display: inline-block; width: 6px; height: 6px; border-radius: 50%; background-color: #22D3EE;"></span>LEFT</span>
<span style="color: #FB7185; display: flex; align-items: center; gap: 4px;"><span style="display: inline-block; width: 6px; height: 6px; border-radius: 50%; background-color: #FB7185;"></span>RIGHT</span>
</div>
</div>

<!-- Hero Hit Ring -->
<div style="display: flex; justify-content: center; margin-bottom: 12px;">
<div style="width: 100px; height: 100px; border-radius: 50%; border: 4px solid #334155; display: flex; flex-direction: column; justify-content: center; align-items: center; background: radial-gradient(circle, rgba(15,23,42,1) 60%, rgba(30,41,59,1) 100%); box-shadow: 0 4px 10px rgba(0,0,0,0.3);">
<span style="font-size: 32px; font-weight: 800; color: #FFFFFF; line-height: 1;">{total_hits}</span>
<span style="font-size: 9px; font-weight: 700; color: #94A3B8; letter-spacing: 0.05em; margin-top: 4px;">HITS</span>
</div>
</div>
<div style="text-align: center; font-size: 11px; font-weight: 700; color: #94A3B8; letter-spacing: 0.05em; text-transform: uppercase; margin-bottom: 15px;">TARGET HITS</div>

<!-- Dominant Side Badge -->
<div style="display: flex; justify-content: center; margin-bottom: 24px;">
<div style="border: 1px solid rgba(245, 158, 11, 0.4); background-color: rgba(245, 158, 11, 0.08); color: #FBBF24; font-size: 11px; font-weight: 700; letter-spacing: 0.05em; padding: 6px 16px; border-radius: 20px; text-transform: uppercase;">
DOMINANT: {dominant_side_en} ARM
</div>
</div>

<!-- Metrics List -->
<div style="display: flex; flex-direction: column;">
<!-- Row: AVG SPEED -->
<div style="display: flex; justify-content: space-between; align-items: center; padding: 10px 0; border-bottom: 1px solid #334155;">
<span style="font-size: 11px; font-weight: 600; color: #94A3B8; letter-spacing: 0.05em;">AVG SPEED</span>
<div style="display: flex; gap: 24px;">
<span style="color: #22D3EE; font-family: 'JetBrains Mono', monospace; font-size: 13px; font-weight: 700; width: 85px; text-align: right;">{left_avg_speed_px:.1f}<span style="font-size: 9px; font-weight: 500; color: #64748B; margin-left: 2px;">px/s</span></span>
<span style="color: #FB7185; font-family: 'JetBrains Mono', monospace; font-size: 13px; font-weight: 700; width: 85px; text-align: right;">{right_avg_speed_px:.1f}<span style="font-size: 9px; font-weight: 500; color: #64748B; margin-left: 2px;">px/s</span></span>
</div>
</div>

<!-- Row: MAX SPEED -->
<div style="display: flex; justify-content: space-between; align-items: center; padding: 10px 0; border-bottom: 1px solid #334155;">
<span style="font-size: 11px; font-weight: 600; color: #94A3B8; letter-spacing: 0.05em;">MAX SPEED</span>
<div style="display: flex; gap: 24px;">
<span style="color: #22D3EE; font-family: 'JetBrains Mono', monospace; font-size: 13px; font-weight: 700; width: 85px; text-align: right;">{left_max_speed_px:.1f}<span style="font-size: 9px; font-weight: 500; color: #64748B; margin-left: 2px;">px/s</span></span>
<span style="color: #FB7185; font-family: 'JetBrains Mono', monospace; font-size: 13px; font-weight: 700; width: 85px; text-align: right;">{right_max_speed_px:.1f}<span style="font-size: 9px; font-weight: 500; color: #64748B; margin-left: 2px;">px/s</span></span>
</div>
</div>

<!-- Row: AVG JERK -->
<div style="display: flex; justify-content: space-between; align-items: center; padding: 10px 0; border-bottom: 1px solid #334155;">
<span style="font-size: 11px; font-weight: 600; color: #94A3B8; letter-spacing: 0.05em;">AVG JERK</span>
<div style="display: flex; gap: 24px;">
<span style="color: #22D3EE; font-family: 'JetBrains Mono', monospace; font-size: 13px; font-weight: 700; width: 85px; text-align: right;">{left_avg_jerk_px:.1f}<span style="font-size: 9px; font-weight: 500; color: #64748B; margin-left: 2px;">px/s³</span></span>
<span style="color: #FB7185; font-family: 'JetBrains Mono', monospace; font-size: 13px; font-weight: 700; width: 85px; text-align: right;">{right_avg_jerk_px:.1f}<span style="font-size: 9px; font-weight: 500; color: #64748B; margin-left: 2px;">px/s³</span></span>
</div>
</div>

<!-- Row: ROM RANGE -->
<div style="display: flex; justify-content: space-between; align-items: center; padding: 10px 0; border-bottom: 1px solid #334155;">
<span style="font-size: 11px; font-weight: 600; color: #94A3B8; letter-spacing: 0.05em;">ROM RANGE</span>
<div style="display: flex; gap: 24px;">
<span style="color: #22D3EE; font-family: 'JetBrains Mono', monospace; font-size: 13px; font-weight: 700; width: 85px; text-align: right;">{left_rom_range:.0f}<span style="font-size: 9px; font-weight: 500; color: #64748B; margin-left: 2px;">°</span></span>
<span style="color: #FB7185; font-family: 'JetBrains Mono', monospace; font-size: 13px; font-weight: 700; width: 85px; text-align: right;">{right_rom_range:.0f}<span style="font-size: 8px; font-weight: 500; color: #64748B; margin-left: 2px;">°</span></span>
</div>
</div>

<!-- Row: CURRENT ROM -->
<div style="display: flex; justify-content: space-between; align-items: center; padding: 10px 0; border-bottom: 1px solid #334155;">
<span style="font-size: 11px; font-weight: 600; color: #94A3B8; letter-spacing: 0.05em;">CURRENT ROM</span>
<div style="display: flex; gap: 24px;">
<span style="color: #22D3EE; font-family: 'JetBrains Mono', monospace; font-size: 13px; font-weight: 700; width: 85px; text-align: right;">{left_current_rom:.0f}<span style="font-size: 9px; font-weight: 500; color: #64748B; margin-left: 2px;">°</span></span>
<span style="color: #FB7185; font-family: 'JetBrains Mono', monospace; font-size: 13px; font-weight: 700; width: 85px; text-align: right;">{right_current_rom:.0f}<span style="font-size: 9px; font-weight: 500; color: #64748B; margin-left: 2px;">°</span></span>
</div>
</div>

<!-- Row: STRAIGHTNESS -->
<div style="display: flex; justify-content: space-between; align-items: center; padding: 10px 0; border-bottom: 1px solid #334155;">
<span style="font-size: 11px; font-weight: 600; color: #94A3B8; letter-spacing: 0.05em;">STRAIGHTNESS</span>
<div style="display: flex; gap: 24px;">
<span style="color: #22D3EE; font-family: 'JetBrains Mono', monospace; font-size: 13px; font-weight: 700; width: 85px; text-align: right;">{left_straightness_val:.0f}<span style="font-size: 9px; font-weight: 500; color: #64748B; margin-left: 2px;">%</span></span>
<span style="color: #FB7185; font-family: 'JetBrains Mono', monospace; font-size: 13px; font-weight: 700; width: 85px; text-align: right;">{right_straightness_val:.0f}<span style="font-size: 9px; font-weight: 500; color: #64748B; margin-left: 2px;">%</span></span>
</div>
</div>

<!-- Row: DOMINANT SIDE -->
<div style="display: flex; justify-content: space-between; align-items: center; padding: 12px 0; border-bottom: 1px solid #334155;">
<span style="font-size: 11px; font-weight: 700; color: #94A3B8; letter-spacing: 0.05em;">DOMINANT SIDE</span>
<span style="color: #FBBF24; font-family: 'JetBrains Mono', monospace; font-size: 13px; font-weight: 700; text-transform: uppercase;">{dominant_side_en}</span>
</div>
</div>

<!-- Learned Non-Use Summary Box -->
<div style="margin-top: 20px; padding: 16px; background-color: #0F172A; border-radius: 12px; border: 1px solid #334155;">
<p style="margin: 0; font-size: 14px; color: #FFFFFF;">
🔍 <b>สรุปการคาดการณ์ภาวะ Learned Non-Use:</b> 
<span style="color: {lnu_color}; font-weight: bold;">{lnu_risk}</span>
</p>
<p style="margin: 5px 0 0 0; font-size: 11px; color: #94A3B8;">
* เกณฑ์คำนวณจากความสมดุลของการสลับยื่นมือ, ความเร็วสัมผัส, องศาไหล่ (Shoulder Elevation), และระดับความแกว่งไหว (Jitter) ระหว่างการประเมิน
</p>
</div>
</div>"""
        report_placeholder.markdown(report_html, unsafe_allow_html=True)
        st.balloons()
        
        # Save results to session state for persistence
        st.session_state.report_html = report_html
        st.session_state.video_bytes = None
        
        # Display Download Button for output video
        if save_video and out_path and os.path.exists(out_path):
            with open(out_path, "rb") as f:
                video_bytes = f.read()
            st.session_state.video_bytes = video_bytes
            st.sidebar.download_button(
                label="📥 Download Annotated Video",
                data=video_bytes,
                file_name="annotated_dexterity_assessment.mp4",
                mime="video/mp4",
                use_container_width=True
            )
            
        st.session_state.analysis_completed = True
            
        # Cleanup temp files
        try:
            os.unlink(video_path)
            if out_path:
                os.unlink(out_path)
        except Exception as e:
            pass

# Display final results and Frame Inspector if analysis is complete
if st.session_state.get("analysis_completed", False):
    st.markdown("<hr style='border-color: #143D66;'>", unsafe_allow_html=True)
    st.markdown(st.session_state.report_html, unsafe_allow_html=True)
    
    if st.session_state.video_bytes:
        st.sidebar.download_button(
            label="📥 Download Annotated Video",
            data=st.session_state.video_bytes,
            file_name="annotated_dexterity_assessment.mp4",
            mime="video/mp4",
            use_container_width=True
        )
        
    st.markdown("<hr style='border-color: #143D66;'>", unsafe_allow_html=True)
    st.subheader("🔍 ค้นหาและตรวจสอบรายเฟรม (Frame Inspector)")
    
    import pandas as pd
    df_history = pd.DataFrame(st.session_state.frame_history)
    
    col_dl, col_search = st.columns([1, 2])
    
    with col_dl:
        csv_data = df_history.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            label="📥 ดาวน์โหลดข้อมูลทุกเฟรม (CSV)",
            data=csv_data,
            file_name="detailed_frame_log.csv",
            mime="text/csv",
            use_container_width=True
        )
        
    with col_search:
        search_frame = st.number_input("ป้อนหมายเลขเฟรมที่ต้องการค้นหา (เช่น 120):", min_value=1, step=1, value=1)
        if search_frame is not None:
            max_processed = df_history['Frame Index'].max()
            if search_frame > max_processed:
                st.error(f"❌ ยังประมวลผลไปไม่ถึงเฟรมนี้ (ประมวลผลถึงเฟรมสูงสุด: {max_processed})")
            else:
                match = df_history[df_history['Frame Index'] == search_frame]
                if not match.empty:
                    row = match.iloc[0]
                    st.success(f"พบข้อมูลเฟรมที่ {search_frame} (เวลา: {row['Timestamp (sec)']:.2f} วินาที)")
                    
                    m_col1, m_col2 = st.columns(2)
                    with m_col1:
                        st.write(f"**เป้าหมายช่องที่:** {row['Active Target Cell']}")
                        if row['Active Target Cell'] != "None":
                            st.caption(f"พื้นที่เป้าหมาย: X: {row['Target X1 (%)']:.1f}%-{row['Target X2 (%)']:.1f}% | Y: {row['Target Y1 (%)']:.1f}%-{row['Target Y2 (%)']:.1f}%")
                        st.write(f"**สถานะชนมือซ้าย (Left Hit):** {row['Left Hand Hit']}")
                        st.write(f"**สถานะชนมือขวา (Right Hit):** {row['Right Hand Hit']}")
                    with m_col2:
                        st.write(f"**พิกัดมือซ้าย (Left Hand):** X: {row['Left Hand X (%)']} Y: {row['Left Hand Y (%)']}")
                        st.write(f"**พิกัดมือขวา (Right Hand):** X: {row['Right Hand X (%)']} Y: {row['Right Hand Y (%)']}")
                else:
                    st.warning("⚠️ ไม่พบข้อมูลสำหรับเฟรมนี้ (อาจโดนข้ามระหว่างประมวลผล)")
