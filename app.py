"""KONKAE.COM - Automated dexterity / sarcopenia assessment dashboard.

Thin Streamlit orchestration layer. The heavy computer-vision pipeline lives in
``video_processor.py`` and the math in ``kinematics.py``; presentation helpers
are in ``styles.py`` and ``ui_components.py``.
"""

import os
import tempfile
import cv2
import numpy as np

import pandas as pd
import streamlit as st

from styles import DASHBOARD_CSS
from ui_components import get_kinematics_card_html, get_report_card_html
from video_processor import ProcessorConfig, VideoProcessor


def get_preview_frame(video_path, skip_seconds, left_margin, right_margin, top_margin, bottom_margin, mirror_view):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    
    # Seek to skip_seconds
    frame_to_seek = int(skip_seconds * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_to_seek)
    
    ret, frame = cap.read()
    if not ret:
        # Fallback to first frame if seek failed
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()
        
    cap.release()
    
    if not ret or frame is None:
        return None
    
    height, width, _ = frame.shape
    if mirror_view:
        frame = cv2.flip(frame, 1)
    
    annotated = frame.copy()
    
    # Draw grid geometry matching video_processor.py
    x_start = int(width * (left_margin / 100.0))
    x_end = int(width * (1.0 - right_margin / 100.0))
    y_start = int(height * (top_margin / 100.0))
    y_end = int(height * (1.0 - bottom_margin / 100.0))
    grid_w = x_end - x_start
    grid_h = y_end - y_start
    cell_w = grid_w // 3
    cell_h = grid_h // 3
    
    # Draw grid box
    cv2.rectangle(annotated, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)  # Green box
    for i in range(1, 3):
        cv2.line(annotated, (x_start + i * cell_w, y_start), (x_start + i * cell_w, y_end), (0, 255, 0), 1)
        cv2.line(annotated, (x_start, y_start + i * cell_h), (x_end, y_start + i * cell_h), (0, 255, 0), 1)
    
    # Label cells
    for r in range(3):
        for c in range(3):
            cell_num = r * 3 + c + 1
            cv2.putText(annotated, f"Cell {cell_num}", (x_start + c * cell_w + 10, y_start + r * cell_h + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
    # Convert BGR to RGB
    return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

# --- Page Configuration ---
st.set_page_config(
    page_title="KONKAE.COM",
    page_icon="\U0001F9BE",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(DASHBOARD_CSS, unsafe_allow_html=True)


# --- Session State Defaults ---
_DEFAULTS = {
    "log_lines": [
        "Initializing Automated Dexterity Assessment Pipeline...",
        "Ready. Upload a video and press 'Start Analysis'.",
    ],
    "analysis_completed": False,
    "processing": False,
    "report_html": None,
    "frame_history": [],
    "video_bytes": None,
    # Persisted UI snapshots (survive reruns so video/telemetry don't vanish)
    "last_frame_rgb": None,
    "last_kinematics_html": None,
    "last_progress": 0.0,
    "last_status_text": "",
    "last_rec_status_html": None,
    "last_active_cell_html": None,
}
for key, default in _DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default


# --- App Headers ---
st.markdown("<h4 style='color: #94A3B8; margin-bottom: 0; font-family: Outfit, sans-serif; font-weight: 400;'>By Toto and King</h4>", unsafe_allow_html=True)
st.markdown("<h1 style='margin-top: 0; color: #FFFFFF; font-family: Outfit, sans-serif; font-weight: 700;'>KONKAE.COM</h1>", unsafe_allow_html=True)


# --- Sidebar ---
st.sidebar.markdown("""<div style="background: rgba(6, 182, 212, 0.1); border: 1px solid rgba(6, 182, 212, 0.3); padding: 12px; border-radius: 10px; margin-bottom: 15px; text-align: center;">
<span style="color: #38BDF8; font-weight: bold; font-size: 13px; display: block; margin-bottom: 8px;">\U0001F310 \u0e23\u0e30\u0e1a\u0e1a\u0e1b\u0e23\u0e30\u0e40\u0e21\u0e34\u0e19\u0e23\u0e48\u0e27\u0e21\u0e17\u0e32\u0e07\u0e04\u0e25\u0e34\u0e19\u0e34\u0e01</span>
<a href="http://localhost:3000" target="_self" style="text-decoration: none;">
<button style="background: #06B6D4; color: #0F172A; border: none; padding: 8px 12px; border-radius: 8px; font-weight: bold; font-size: 12px; cursor: pointer; transition: 0.3s; width: 100%;">
\U0001F3A5 \u0e40\u0e1b\u0e34\u0e14\u0e23\u0e30\u0e1a\u0e1a\u0e01\u0e25\u0e49\u0e2d\u0e07\u0e2a\u0e14 (Live Webcam)
</button>
</a>
</div>""", unsafe_allow_html=True)

st.sidebar.title("Assessment Settings")

min_frames_in_box = st.sidebar.slider("Min Frames for HIT (Sensitivity)", 2, 20, 3, step=1)
skip_seconds = st.sidebar.slider("\u0e02\u0e49\u0e32\u0e21\u0e0a\u0e48\u0e27\u0e07\u0e40\u0e23\u0e34\u0e48\u0e21\u0e15\u0e49\u0e19\u0e27\u0e34\u0e14\u0e35\u0e42\u0e2d/\u0e2a\u0e2d\u0e19\u0e43\u0e0a\u0e49\u0e07\u0e32\u0e19 (\u0e27\u0e34\u0e19\u0e32\u0e17\u0e35)", 0.0, 30.0, 0.0, step=0.5)
target_color = st.sidebar.selectbox("\u0e2a\u0e35\u0e01\u0e25\u0e48\u0e2d\u0e07\u0e40\u0e1b\u0e49\u0e32\u0e2b\u0e21\u0e32\u0e22 (Target Color)", ["Green", "Red", "Blue"])
ref_point_mode = st.sidebar.selectbox("\u0e08\u0e38\u0e14\u0e2d\u0e49\u0e32\u0e07\u0e2d\u0e34\u0e07\u0e02\u0e2d\u0e07\u0e21\u0e37\u0e2d (Tracking Point)", ["Wrist", "Index Finger Tip"])

mirror_view = st.sidebar.checkbox("\u0e01\u0e25\u0e31\u0e1a\u0e14\u0e49\u0e32\u0e19\u0e27\u0e34\u0e14\u0e35\u0e42\u0e2d (Mirror/Flip Video)", value=False)
save_video = st.sidebar.checkbox("Save Annotated Video for Download", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("\U0001F4D0 Grid Bounds Alignment")
grid_preset = st.sidebar.selectbox("Grid Layout Preset", ["Fit Video Frame (Default)", "Center 4:3 (Pillarbox)", "Custom Margins"])

if grid_preset == "Center 4:3 (Pillarbox)":
    left_margin, right_margin, top_margin, bottom_margin = 12.5, 12.5, 0.0, 0.0
elif grid_preset == "Custom Margins":
    grid_width = st.sidebar.slider("ความกว้างตาราง Grid Width (%)", 10.0, 100.0, 75.0, step=0.5)
    grid_height = st.sidebar.slider("ความสูงตาราง Grid Height (%)", 10.0, 100.0, 100.0, step=0.5)
    grid_center_x = st.sidebar.slider("ตำแหน่งกึ่งกลางตารางแนวนอน Grid Center X (%)", 0.0, 100.0, 50.0, step=0.5)
    grid_center_y = st.sidebar.slider("ตำแหน่งกึ่งกลางตารางแนวตั้ง Grid Center Y (%)", 0.0, 100.0, 50.0, step=0.5)
    
    # Calculate margins from size and position
    left_margin = max(0.0, grid_center_x - grid_width / 2)
    right_margin = max(0.0, 100.0 - (grid_center_x + grid_width / 2))
    top_margin = max(0.0, grid_center_y - grid_height / 2)
    bottom_margin = max(0.0, 100.0 - (grid_center_y + grid_height / 2))
else:  # Fit Video Frame
    left_margin, right_margin, top_margin, bottom_margin = 0.0, 0.0, 0.0, 0.0

uploaded_file = st.sidebar.file_uploader("Upload Assessment Video", type=["mp4", "avi", "mov", "webm"])
if uploaded_file is not None:
    st.sidebar.success("Video Loaded successfully!")
    if "temp_video_path" not in st.session_state or st.session_state.get("last_uploaded_name") != uploaded_file.name:
        # Clean up old temp file if exists
        if "temp_video_path" in st.session_state and os.path.exists(st.session_state.temp_video_path):
            try:
                os.remove(st.session_state.temp_video_path)
            except Exception:
                pass
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())
        tfile.close()
        st.session_state.temp_video_path = tfile.name
        st.session_state.last_uploaded_name = uploaded_file.name
        st.session_state.analysis_completed = False
        st.session_state.last_frame_rgb = None
        st.session_state.last_progress = 0.0
        st.session_state.last_status_text = "New video uploaded. Adjust settings above and click 'Start Analysis'."
        st.session_state.last_rec_status_html = None
        st.session_state.last_active_cell_html = None
        st.session_state.frame_history = []
else:
    st.sidebar.info("Upload a video to start the assessment.")
    if "temp_video_path" in st.session_state:
        if os.path.exists(st.session_state.temp_video_path):
            try:
                os.remove(st.session_state.temp_video_path)
            except Exception:
                pass
        del st.session_state.temp_video_path
        if "last_uploaded_name" in st.session_state:
            del st.session_state.last_uploaded_name

# --- Generate preview frame if not processing ---
if not st.session_state.processing and uploaded_file is not None and "temp_video_path" in st.session_state:
    preview_img = get_preview_frame(
        st.session_state.temp_video_path,
        skip_seconds=skip_seconds,
        left_margin=left_margin,
        right_margin=right_margin,
        top_margin=top_margin,
        bottom_margin=bottom_margin,
        mirror_view=mirror_view
    )
    if preview_img is not None:
        st.session_state.last_frame_rgb = preview_img


def _filter_and_format_logs(lines, filter_type):
    filtered_lines = []
    for line in lines:
        if filter_type == "\u0e40\u0e09\u0e1e\u0e32\u0e30\u0e40\u0e1b\u0e49\u0e32\u0e2b\u0e21\u0e32\u0e22\u0e1b\u0e23\u0e32\u0e01\u0e0f (Appearances)" and "appeared" not in line:
            continue
        elif filter_type == "\u0e40\u0e09\u0e1e\u0e32\u0e30\u0e01\u0e32\u0e23\u0e0a\u0e19\u0e40\u0e1b\u0e49\u0e32\u0e2b\u0e21\u0e32\u0e22 (Hits)" and "HIT" not in line:
            continue
        elif filter_type == "\u0e40\u0e09\u0e1e\u0e32\u0e30\u0e40\u0e1b\u0e49\u0e32\u0e2b\u0e21\u0e32\u0e22\u0e1e\u0e25\u0e32\u0e14 (Misses)" and "disappeared" not in line:
            continue
        elif filter_type == "\u0e40\u0e09\u0e1e\u0e32\u0e30\u0e01\u0e32\u0e23\u0e22\u0e01\u0e41\u0e02\u0e19 (Arm Raises)" and "Raise" not in line:
            continue

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


# --- Layout Columns ---
col1, col2 = st.columns([1.8, 1.2])

with col1:
    st.subheader("Assessment Frame Feed")
    preview_placeholder = st.empty()
    frame_placeholder = st.empty()
    progress_bar = st.progress(st.session_state.last_progress)
    status_placeholder = st.empty()

with col2:
    st.subheader("Real-Time Telemetry & Status")
    status_col1, status_col2 = st.columns(2)
    with status_col1:
        st.markdown("<b>\u0e2a\u0e16\u0e32\u0e19\u0e30\u0e01\u0e32\u0e23\u0e1a\u0e31\u0e19\u0e17\u0e36\u0e01:</b>", unsafe_allow_html=True)
        rec_status_placeholder = st.empty()
    with status_col2:
        st.markdown("<b>\u0e40\u0e1b\u0e49\u0e32\u0e2b\u0e21\u0e32\u0e22\u0e02\u0e13\u0e30\u0e19\u0e35\u0e49:</b>", unsafe_allow_html=True)
        active_cell_placeholder = st.empty()

    st.markdown("<hr style='margin: 10px 0; border-color: #143D66;'>", unsafe_allow_html=True)

    # Kinematics Dashboard Card Placeholder
    kinematics_card_placeholder = st.empty()

    st.markdown("<hr style='margin: 10px 0; border-color: #143D66;'>", unsafe_allow_html=True)

    st.subheader("Real-Time Event Logs")
    log_display_placeholder = st.empty()

    # ---------------------------------------------------------------
    # Log filter lives inside a @st.fragment so changing it does NOT
    # trigger a full-page rerun (which was the root cause of the video
    # disappearing).
    # ---------------------------------------------------------------
    @st.fragment
    def _log_panel():
        """Self-contained log viewer that reruns only itself on filter change."""
        log_filter = st.selectbox(
            "กรองประเภท Log (Filter Logs)",
            ["ทั้งหมด (Show All)", "เฉพาะเป้าหมายปรากฏ (Appearances)", "เฉพาะการชนเป้าหมาย (Hits)", "เฉพาะเป้าหมายพลาด (Misses)", "เฉพาะการยกแขน (Arm Raises)"],
            key="log_filter_select",
        )
        lines = st.session_state.log_lines
        colored_lines = _filter_and_format_logs(lines, log_filter)
        formatted = "<br>".join(colored_lines[::-1])  # latest on top
        log_display_placeholder.markdown(f"<div class='log-container'>{formatted}</div>", unsafe_allow_html=True)

        # Download button
        log_text = "\n".join(lines)
        st.download_button(
            label="📥 ดาวน์โหลด Event Logs (.txt)",
            data=log_text,
            file_name="dexterity_event_logs.txt",
            mime="text/plain",
            key="dl_btn_fragment",
        )

    _log_panel()



# --- Restore persisted UI state from session_state ---
# This is the key fix: after a rerun the placeholders are recreated empty,
# so we immediately re-populate them from whatever we last stored.

if st.session_state.last_frame_rgb is not None:
    frame_placeholder.image(st.session_state.last_frame_rgb, use_container_width=True)

if st.session_state.last_status_text:
    status_placeholder.text(st.session_state.last_status_text)

if st.session_state.last_rec_status_html:
    rec_status_placeholder.markdown(st.session_state.last_rec_status_html, unsafe_allow_html=True)
else:
    rec_status_placeholder.markdown("<div class='status-badge-demo'>WAITING</div>", unsafe_allow_html=True)

if st.session_state.last_active_cell_html:
    active_cell_placeholder.markdown(st.session_state.last_active_cell_html, unsafe_allow_html=True)
else:
    active_cell_placeholder.markdown("<div class='metric-card' style='padding:5px; margin:0;'>\u0e44\u0e21\u0e48\u0e21\u0e35</div>", unsafe_allow_html=True)

if st.session_state.last_kinematics_html:
    kinematics_card_placeholder.markdown(st.session_state.last_kinematics_html, unsafe_allow_html=True)
else:
    _init_kin_html = get_kinematics_card_html(
        frame_idx=0, fps=30.0, total_hits=0,
        left_hits=0, right_hits=0,
        left_speeds=[], right_speeds=[], left_jerks=[], right_jerks=[],
        left_rom_min=float("inf"), left_rom_max=float("-inf"),
        right_rom_min=float("inf"), right_rom_max=float("-inf"),
        left_current_rom=0.0, right_current_rom=0.0,
        left_straightness_val=0.0, right_straightness_val=0.0,
        dominant_side="WAITING",
    )
    kinematics_card_placeholder.markdown(_init_kin_html, unsafe_allow_html=True)

# Placeholder for final summary report
report_placeholder = st.empty()


# --- Telemetry Charts Section ---
st.markdown("<hr style='border-color: #143D66;'>", unsafe_allow_html=True)
st.subheader("\U0001F4CA \u0e01\u0e23\u0e32\u0e1f\u0e27\u0e34\u0e40\u0e04\u0e23\u0e32\u0e30\u0e2b\u0e4c\u0e01\u0e32\u0e23\u0e40\u0e04\u0e25\u0e37\u0e48\u0e2d\u0e19\u0e44\u0e2b\u0e27 (Kinematics Telemetry Charts)")

chart_col1, chart_col2, chart_col3 = st.columns(3)
with chart_col1:
    st.markdown("<b>\u0e04\u0e27\u0e32\u0e21\u0e40\u0e23\u0e47\u0e27\u0e02\u0e2d\u0e07\u0e41\u0e02\u0e19 (Arm Speed - px/s)</b>", unsafe_allow_html=True)
    speed_chart_placeholder = st.empty()
with chart_col2:
    st.markdown("<b>\u0e04\u0e27\u0e32\u0e21\u0e23\u0e32\u0e1a\u0e40\u0e23\u0e35\u0e22\u0e1a\u0e02\u0e2d\u0e07\u0e02\u0e49\u0e2d\u0e15\u0e48\u0e2d (Movement Jerk)</b>", unsafe_allow_html=True)
    jerk_chart_placeholder = st.empty()
with chart_col3:
    st.markdown("<b>\u0e2d\u0e07\u0e28\u0e32\u0e01\u0e32\u0e23\u0e02\u0e22\u0e31\u0e1a\u0e44\u0e2b\u0e25\u0e48 (Shoulder ROM - degrees)</b>", unsafe_allow_html=True)
    rom_chart_placeholder = st.empty()


def update_telemetry_charts(history_list):
    if not history_list:
        return
    df = pd.DataFrame(history_list)
    if "Timestamp (sec)" not in df.columns:
        return
    df_speed = df[["Timestamp (sec)", "Left Arm Speed (px/s)", "Right Arm Speed (px/s)"]].set_index("Timestamp (sec)")
    df_jerk = df[["Timestamp (sec)", "Left Movement Jerk (px/s3)", "Right Movement Jerk (px/s3)"]].set_index("Timestamp (sec)")
    df_rom = df[["Timestamp (sec)", "Left Shoulder Angle (deg)", "Right Shoulder Angle (deg)"]].set_index("Timestamp (sec)")
    df_speed.columns = ["Left Arm", "Right Arm"]
    df_jerk.columns = ["Left Arm", "Right Arm"]
    df_rom.columns = ["Left Arm", "Right Arm"]
    speed_chart_placeholder.line_chart(df_speed, height=200)
    jerk_chart_placeholder.line_chart(df_jerk, height=200)
    rom_chart_placeholder.line_chart(df_rom, height=200)


# Initialize chart placeholders
if st.session_state.analysis_completed and st.session_state.frame_history:
    update_telemetry_charts(st.session_state.frame_history)
else:
    speed_chart_placeholder.info("\u0e23\u0e2d\u0e02\u0e49\u0e2d\u0e21\u0e39\u0e25\u0e01\u0e32\u0e23\u0e40\u0e04\u0e25\u0e37\u0e48\u0e2d\u0e19\u0e44\u0e2b\u0e27...")
    jerk_chart_placeholder.info("\u0e23\u0e2d\u0e02\u0e49\u0e2d\u0e21\u0e39\u0e25\u0e01\u0e32\u0e23\u0e40\u0e04\u0e25\u0e37\u0e48\u0e2d\u0e19\u0e44\u0e2b\u0e27...")
    rom_chart_placeholder.info("\u0e23\u0e2d\u0e02\u0e49\u0e2d\u0e21\u0e39\u0e25\u0e01\u0e32\u0e23\u0e40\u0e04\u0e25\u0e37\u0e48\u0e2d\u0e19\u0e44\u0e2b\u0e27...")


# --- Processing Loop ---
if uploaded_file is not None:
    # Disable the button while already processing to prevent re-entry
    start_disabled = st.session_state.processing
    start_analysis = st.sidebar.button(
        "Start Analysis",
        use_container_width=True,
        disabled=start_disabled,
    )

    if start_analysis and not st.session_state.processing:
        st.session_state.processing = True
        st.session_state.analysis_completed = False
        st.session_state.report_html = None
        st.session_state.frame_history = []
        st.session_state.video_bytes = None
        st.session_state.last_frame_rgb = None
        st.session_state.last_kinematics_html = None
        st.session_state.last_progress = 0.0
        st.session_state.last_status_text = ""
        st.session_state.last_rec_status_html = None
        st.session_state.last_active_cell_html = None

        video_path = st.session_state.temp_video_path

        config = ProcessorConfig(
            target_color=target_color,
            ref_point_mode=ref_point_mode,
            min_frames_in_box=min_frames_in_box,
            skip_seconds=skip_seconds,
            mirror_view=mirror_view,
            left_margin=left_margin,
            right_margin=right_margin,
            top_margin=top_margin,
            bottom_margin=bottom_margin,
        )

        processor = VideoProcessor(video_path, config)
        fps = processor.fps
        total_frames = processor.total_frames

        st.session_state.log_lines = [
            "Initializing Dual Computer Vision ...",
            f"Video format: {processor.width}x{processor.height} @ {fps:.1f} fps",
            "Processing frames... (Pose & Target)",
        ]

        # Optional annotated-video writer
        import cv2  # local import keeps the writer concern next to its use

        writer = None
        out_path = None
        if save_video:
            out_tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            out_path = out_tfile.name
            out_tfile.close()
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(out_path, fourcc, fps, (processor.width, processor.height))

        local_frame_history = []
        last_frame_idx = 0

        try:
            for result in processor.process():
                last_frame_idx = result.frame_idx
                st.session_state.log_lines.extend(result.new_log_lines)

                # Update real-time logs in the placeholder
                current_filter = st.session_state.get("log_filter_select", "ทั้งหมด (Show All)")
                colored_lines = _filter_and_format_logs(st.session_state.log_lines, current_filter)
                formatted = "<br>".join(colored_lines[::-1])
                log_display_placeholder.markdown(f"<div class='log-container'>{formatted}</div>", unsafe_allow_html=True)

                # Active-cell telemetry
                if result.active_cells:
                    cells_str = ", ".join([f"\u0e0a\u0e48\u0e2d\u0e07\u0e17\u0e35\u0e48 {c}" for c in result.active_cells])
                    cell_html = f"<div class='metric-card' style='padding:5px; margin:0; border-color:#EF4444; color:#EF4444;'>{cells_str}</div>"
                else:
                    cell_html = "<div class='metric-card' style='padding:5px; margin:0;'>\u0e44\u0e21\u0e48\u0e21\u0e35</div>"
                active_cell_placeholder.markdown(cell_html, unsafe_allow_html=True)
                st.session_state.last_active_cell_html = cell_html

                # Recording status telemetry
                if result.is_recording:
                    rec_html = "<div class='status-badge-recording'>REC ACTIVE (\u0e1a\u0e31\u0e19\u0e17\u0e36\u0e01\u0e02\u0e49\u0e2d\u0e21\u0e39\u0e25)</div>"
                else:
                    rec_html = "<div class='status-badge-demo'>DEMO MODE (\u0e0a\u0e48\u0e27\u0e07\u0e2a\u0e2d\u0e19/\u0e2a\u0e32\u0e18\u0e34\u0e15)</div>"
                rec_status_placeholder.markdown(rec_html, unsafe_allow_html=True)
                st.session_state.last_rec_status_html = rec_html

                # Video frame — display AND persist
                frame_placeholder.image(result.annotated_frame_rgb, use_container_width=True)
                st.session_state.last_frame_rgb = result.annotated_frame_rgb

                pct_complete = min(1.0, result.frame_idx / total_frames)
                progress_bar.progress(pct_complete)
                st.session_state.last_progress = pct_complete

                status_text = f"Processed Frame: {result.frame_idx}/{total_frames} ({pct_complete * 100:.1f}%)"
                status_placeholder.text(status_text)
                st.session_state.last_status_text = status_text

                # Live kinematics card
                kin_html = get_kinematics_card_html(
                    frame_idx=result.frame_idx,
                    fps=fps,
                    total_hits=result.total_hits,
                    left_hits=result.left_hits,
                    right_hits=result.right_hits,
                    left_speeds=result.left_speeds,
                    right_speeds=result.right_speeds,
                    left_jerks=result.left_jerks,
                    right_jerks=result.right_jerks,
                    left_rom_min=result.left_rom_min,
                    left_rom_max=result.left_rom_max,
                    right_rom_min=result.right_rom_min,
                    right_rom_max=result.right_rom_max,
                    left_current_rom=result.left_current_rom,
                    right_current_rom=result.right_current_rom,
                    left_straightness_val=result.left_straightness,
                    right_straightness_val=result.right_straightness,
                    dominant_side=result.dominant_side,
                )
                kinematics_card_placeholder.markdown(kin_html, unsafe_allow_html=True)
                st.session_state.last_kinematics_html = kin_html

                if writer is not None:
                    writer.write(cv2.cvtColor(result.annotated_frame_rgb, cv2.COLOR_RGB2BGR))

                local_frame_history.append(result.history_row)

                if result.frame_idx % 30 == 0:
                    update_telemetry_charts(local_frame_history)
        finally:
            processor.release()
            if writer is not None:
                writer.release()
            st.session_state.processing = False

        st.session_state.frame_history = local_frame_history
        update_telemetry_charts(local_frame_history)

        st.session_state.log_lines.append("Assessment completed successfully!")

        # --- Final summary report ---
        summary = processor.summarize()
        report_html = get_report_card_html(
            frame_idx=last_frame_idx,
            fps=fps,
            total_hits=summary["total_hits"],
            left_hits=summary["left_hits"],
            right_hits=summary["right_hits"],
            dominant_side_en=summary["dominant_side_en"],
            left_avg_speed=summary["left_avg_speed"],
            right_avg_speed=summary["right_avg_speed"],
            left_max_speed=summary["left_max_speed"],
            right_max_speed=summary["right_max_speed"],
            left_avg_jerk=summary["left_avg_jerk"],
            right_avg_jerk=summary["right_avg_jerk"],
            left_rom_range=summary["left_rom_range"],
            right_rom_range=summary["right_rom_range"],
            left_current_rom=summary["left_current_rom"],
            right_current_rom=summary["right_current_rom"],
            left_straightness_val=summary["left_straightness"],
            right_straightness_val=summary["right_straightness"],
            lnu_risk=summary["lnu_risk"],
            lnu_color=summary["lnu_color"],
        )
        report_placeholder.markdown(report_html, unsafe_allow_html=True)
        st.balloons()

        st.session_state.report_html = report_html
        st.session_state.video_bytes = None

        # Download button for annotated video
        if save_video and out_path and os.path.exists(out_path):
            with open(out_path, "rb") as f:
                video_bytes = f.read()
            st.session_state.video_bytes = video_bytes
            st.sidebar.download_button(
                label="\U0001F4E5 Download Annotated Video",
                data=video_bytes,
                file_name="annotated_dexterity_assessment.mp4",
                mime="video/mp4",
                use_container_width=True,
            )

        st.session_state.analysis_completed = True

        # Cleanup temp files
        try:
            os.unlink(video_path)
            if out_path:
                os.unlink(out_path)
        except OSError:
            pass
else:
    # No file uploaded — ensure Start Analysis button is not shown
    pass


# --- Final results + Frame Inspector ---
if st.session_state.analysis_completed:
    st.markdown("<hr style='border-color: #143D66;'>", unsafe_allow_html=True)
    st.markdown(st.session_state.report_html, unsafe_allow_html=True)

    if st.session_state.video_bytes:
        st.sidebar.download_button(
            label="\U0001F4E5 Download Annotated Video",
            data=st.session_state.video_bytes,
            file_name="annotated_dexterity_assessment.mp4",
            mime="video/mp4",
            use_container_width=True,
            key="dl_video_persist",
        )

    st.markdown("<hr style='border-color: #143D66;'>", unsafe_allow_html=True)
    st.subheader("\U0001F50D \u0e04\u0e49\u0e19\u0e2b\u0e32\u0e41\u0e25\u0e30\u0e15\u0e23\u0e27\u0e08\u0e2a\u0e2d\u0e1a\u0e23\u0e32\u0e22\u0e40\u0e1f\u0e23\u0e21 (Frame Inspector)")

    df_history = pd.DataFrame(st.session_state.frame_history)

    col_dl, col_search = st.columns([1, 2])
    with col_dl:
        csv_data = df_history.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            label="\U0001F4E5 \u0e14\u0e32\u0e27\u0e19\u0e4c\u0e42\u0e2b\u0e25\u0e14\u0e02\u0e49\u0e2d\u0e21\u0e39\u0e25\u0e17\u0e38\u0e01\u0e40\u0e1f\u0e23\u0e21 (CSV)",
            data=csv_data,
            file_name="detailed_frame_log.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with col_search:
        search_frame = st.number_input("\u0e1b\u0e49\u0e2d\u0e19\u0e2b\u0e21\u0e32\u0e22\u0e40\u0e25\u0e02\u0e40\u0e1f\u0e23\u0e21\u0e17\u0e35\u0e48\u0e15\u0e49\u0e2d\u0e07\u0e01\u0e32\u0e23\u0e04\u0e49\u0e19\u0e2b\u0e32 (\u0e40\u0e0a\u0e48\u0e19 120):", min_value=1, step=1, value=1)
        if search_frame is not None and not df_history.empty:
            max_processed = df_history["Frame Index"].max()
            if search_frame > max_processed:
                st.error(f"\u274c \u0e22\u0e31\u0e07\u0e1b\u0e23\u0e30\u0e21\u0e27\u0e25\u0e1c\u0e25\u0e44\u0e1b\u0e44\u0e21\u0e48\u0e16\u0e36\u0e07\u0e40\u0e1f\u0e23\u0e21\u0e19\u0e35\u0e49 (\u0e1b\u0e23\u0e30\u0e21\u0e27\u0e25\u0e1c\u0e25\u0e16\u0e36\u0e07\u0e40\u0e1f\u0e23\u0e21\u0e2a\u0e39\u0e07\u0e2a\u0e38\u0e14: {max_processed})")
            else:
                match = df_history[df_history["Frame Index"] == search_frame]
                if not match.empty:
                    row = match.iloc[0]
                    st.success(f"\u0e1e\u0e1a\u0e02\u0e49\u0e2d\u0e21\u0e39\u0e25\u0e40\u0e1f\u0e23\u0e21\u0e17\u0e35\u0e48 {search_frame} (\u0e40\u0e27\u0e25\u0e32: {row['Timestamp (sec)']:.2f} \u0e27\u0e34\u0e19\u0e32\u0e17\u0e35)")
                    m_col1, m_col2 = st.columns(2)
                    with m_col1:
                        st.write(f"**\u0e40\u0e1b\u0e49\u0e32\u0e2b\u0e21\u0e32\u0e22\u0e0a\u0e48\u0e2d\u0e07\u0e17\u0e35\u0e48:** {row['Active Target Cell']}")
                        if row["Active Target Cell"] != "None":
                            st.caption(f"\u0e1e\u0e37\u0e49\u0e19\u0e17\u0e35\u0e48\u0e40\u0e1b\u0e49\u0e32\u0e2b\u0e21\u0e32\u0e22: X: {row['Target X1 (%)']:.1f}%-{row['Target X2 (%)']:.1f}% | Y: {row['Target Y1 (%)']:.1f}%-{row['Target Y2 (%)']:.1f}%")
                        st.write(f"**\u0e2a\u0e16\u0e32\u0e19\u0e30\u0e0a\u0e19\u0e21\u0e37\u0e2d\u0e0b\u0e49\u0e32\u0e22 (Left Hit):** {row['Left Hand Hit']}")
                        st.write(f"**\u0e2a\u0e16\u0e32\u0e19\u0e30\u0e0a\u0e19\u0e21\u0e37\u0e2d\u0e02\u0e27\u0e32 (Right Hit):** {row['Right Hand Hit']}")
                    with m_col2:
                        st.write(f"**\u0e1e\u0e34\u0e01\u0e31\u0e14\u0e21\u0e37\u0e2d\u0e0b\u0e49\u0e32\u0e22 (Left Hand):** X: {row['Left Hand X (%)']} Y: {row['Left Hand Y (%)']}")
                        st.write(f"**\u0e1e\u0e34\u0e01\u0e31\u0e14\u0e21\u0e37\u0e2d\u0e02\u0e27\u0e32 (Right Hand):** X: {row['Right Hand X (%)']} Y: {row['Right Hand Y (%)']}")
                else:
                    st.warning("\u26a0\ufe0f \u0e44\u0e21\u0e48\u0e1e\u0e1a\u0e02\u0e49\u0e2d\u0e21\u0e39\u0e25\u0e2a\u0e33\u0e2b\u0e23\u0e31\u0e1a\u0e40\u0e1f\u0e23\u0e21\u0e19\u0e35\u0e49 (\u0e2d\u0e32\u0e08\u0e42\u0e14\u0e19\u0e02\u0e49\u0e32\u0e21\u0e23\u0e30\u0e2b\u0e27\u0e48\u0e32\u0e07\u0e1b\u0e23\u0e30\u0e21\u0e27\u0e25\u0e1c\u0e25)")
