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

gemini_key = os.environ.get("GEMINI_API_KEY", "")


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
    page_icon="🦾",
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
    "last_frame_rgb": None,
    "last_kinematics_html": None,
    "last_progress": 0.0,
    "last_status_text": "",
    "last_rec_status_html": None,
    "last_active_cell_html": None,
    "ai_report": None,
    "analysis_summary": None,
}
for key, default in _DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default


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
        if filter_type == "เฉพาะเป้าหมายปรากฏ (Appearances)" and "appeared" not in line:
            continue
        elif filter_type == "เฉพาะการชนเป้าหมาย (Hits)" and "HIT" not in line:
            continue
        elif filter_type == "เฉพาะเป้าหมายพลาด (Misses)" and "disappeared" not in line:
            continue
        elif filter_type == "เฉพาะการยกแขน (Arm Raises)" and "Raise" not in line:
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


def generate_game_clinical_report(summary: dict) -> str:
    lni_score = summary.get("lni_score", 0.0)
    lni_percent = f"{lni_score * 100:.1f}"
    
    if lni_score < 0.15:
        severity = "Minimal Asymmetry (ความไม่สมมาตรน้อยมาก)"
        risk_level = "LOW"
        emoji = "✅"
        recommendation = "การทำงานของแขนทั้งสองข้างมีความสมมาตรดีเยี่ยม ผู้ทดสอบแสดงความสมดุลของการประสานสัมพันธ์ ความเร็ว และเส้นทางการเคลื่อนไหวที่เป็นเส้นตรงระหว่างทั้งสองฝั่ง ไม่พบข้อบ่งชี้ทางคลินิกของภาวะ Learned Non-Use แนะนำให้ทำกิจกรรมทางกายที่ใช้ร่างกายสองฝั่งตามปกติเพื่อรักษาสมรรถนะ"
    elif lni_score < 0.35:
        severity = "Mild Asymmetry / Compensatory Behavior (ความไม่สมมาตรเล็กน้อย / พฤติกรรมชดเชย)"
        risk_level = "MODERATE-LOW"
        emoji = "⚠️"
        recommendation = "พบความไม่สมมาตรของการเคลื่อนไหวเล็กน้อย ผู้ทดสอบมีอาการลังเลหรือใช้ระยะเวลาเอื้อมนานขึ้นเล็กน้อยในข้างที่ใช้งานน้อยกว่า ซึ่งอาจเป็นตัวแทนของรูปแบบการเคลื่อนไหวเพื่อชดเชยในระยะเริ่มต้น แนะนำให้เน้นการทำกิจกรรมที่ใช้สองมือร่วมกัน โดยพยายามใช้แขนข้างที่ไม่ถนัด/ใช้งานน้อยก่อนในการทำภารกิจประจำวัน"
    elif lni_score < 0.55:
        severity = "Moderate Learned Non-Use (ภาวะ Learned Non-Use ระดับปานกลาง)"
        risk_level = "MODERATE"
        emoji = "🔶"
        recommendation = "ตรวจพบรูปแบบภาวะ Learned Non-Use ระดับปานกลาง มีการเลือกใช้แขนข้างเด่นอย่างชัดเจน ทั้งในด้านความเร็วการเอื้อม ความตรงของเส้นทาง และการเลือกข้างแขนเพื่อเอื้อมเป้าหมาย แนะนำให้พิจารณาใช้หลักการฝึกกระตุ้นการใช้งานแขนข้างที่อ่อนแรง (Constraint-Induced Movement Therapy - CIMT) โดยจำกัดการใช้งานแขนข้างเด่นในช่วงสั้นๆ ระหว่างทำกิจกรรมในชีวิตประจำวัน (เช่น การรับประทานอาหาร การเปิดประตู) เพื่อบังคับให้มีการใช้งานข้างที่อ่อนแรง"
    else:
        severity = "Significant Learned Non-Use & High Fall Risk (ภาวะ Learned Non-Use รุนแรงและความเสี่ยงหกล้มสูง)"
        risk_level = "HIGH"
        emoji = "🔴"
        recommendation = "พบภาวะ Learned Non-Use ระดับรุนแรง ผู้ทดสอบพึ่งพาแขนข้างเด่นเกือบทั้งหมด ในขณะที่แขนข้างที่ใช้งานน้อยมีความเร็วต่ำอย่างมาก เส้นทางการเคลื่อนไหวอ้อมหรือสั่น (jerk สูง) ความแตกต่างของการใช้กำลังนี้ส่งผลต่อการทรงตัวแบบสองฝั่งและปฏิกิริยาการป้องกันตัวเมื่อหกล้ม ทำให้มีความเสี่ยงหกล้มสูงขึ้นอย่างมาก แนะนำให้ส่งต่อเพื่อรับการประเมินทางกิจกรรมบำบัดหรือกายภาพบำบัดอย่างเร่งด่วน"

    left_reaches = summary.get("left_reaches", 0)
    right_reaches = summary.get("right_reaches", 0)
    total_reaches = max(1, summary.get("total_hits", 1))
    
    if left_reaches < right_reaches:
        less_active = "Left"
        more_active = "Right"
    elif left_reaches > right_reaches:
        less_active = "Right"
        more_active = "Left"
    else:
        if summary.get("left_avg_reach_time", 0.0) > summary.get("right_avg_reach_time", 0.0):
            less_active = "Left"
            more_active = "Right"
        else:
            less_active = "Right"
            more_active = "Left"

    less_stats = {
        "reaches": left_reaches if less_active == "Left" else right_reaches,
        "avg_reach_time": summary.get("left_avg_reach_time", 0.0) if less_active == "Left" else summary.get("right_avg_reach_time", 0.0),
        "min_reach_time": summary.get("left_min_reach_time", 0.0) if less_active == "Left" else summary.get("right_min_reach_time", 0.0),
        "max_reach_time": summary.get("left_max_reach_time", 0.0) if less_active == "Left" else summary.get("right_max_reach_time", 0.0),
        "avg_straightness": summary.get("left_avg_straightness", 0.0) if less_active == "Left" else summary.get("right_avg_straightness", 0.0),
        "avg_jerk": summary.get("left_avg_jerk", 0.0) if less_active == "Left" else summary.get("right_avg_jerk", 0.0),
    }
    
    more_stats = {
        "reaches": right_reaches if less_active == "Left" else left_reaches,
        "avg_reach_time": summary.get("right_avg_reach_time", 0.0) if less_active == "Left" else summary.get("left_avg_reach_time", 0.0),
        "min_reach_time": summary.get("right_min_reach_time", 0.0) if less_active == "Left" else summary.get("left_min_reach_time", 0.0),
        "max_reach_time": summary.get("right_max_reach_time", 0.0) if less_active == "Left" else summary.get("left_max_reach_time", 0.0),
        "avg_straightness": summary.get("right_avg_straightness", 0.0) if less_active == "Left" else summary.get("left_avg_straightness", 0.0),
        "avg_jerk": summary.get("right_avg_jerk", 0.0) if less_active == "Left" else summary.get("left_avg_jerk", 0.0),
    }

    usage_ratio = f"{(less_stats['reaches'] / total_reaches * 100):.0f}"
    more_usage_ratio = f"{(more_stats['reaches'] / total_reaches * 100):.0f}"
    
    speed_diff = f"{(((less_stats['avg_reach_time'] - more_stats['avg_reach_time']) / max(0.001, more_stats['avg_reach_time'])) * 100):.0f}" if more_stats['avg_reach_time'] > 0 else "0"

    less_active_th = "แขนซ้าย" if less_active == "Left" else "แขนขวา"
    more_active_th = "แขนซ้าย" if more_active == "Left" else "แขนขวา"

    fall_risk_desc = (
        "⚠️ **ความเสี่ยงหกล้มสูง (HIGH RISK):** ความไม่สมมาตรของการทำงานของแขนมีความสัมพันธ์อย่างสูงกับการสูญเสียการทรงตัวในการเคลื่อนไหว หากผู้ทดสอบลื่นล้ม ปฏิกิริยาการยื่นแขนเพื่อพยุงตัวในข้างที่ใช้งานน้อยอาจทำงานช้าหรือไม่มีกำลังพอ ส่งผลให้มีความเสี่ยงที่จะกระดูกหักจากแรงกระแทกได้สูง แนะนำให้แนะนำมาตรการป้องกันการหกล้มและติดตั้งราวพยุง"
        if lni_score > 0.4 else
        "ผู้ทดสอบแสดงปฏิกิริยาตอบสนองและการควบคุมที่สมมาตรเพียงพอ ความเสี่ยงต่อการหกล้มกะทันหันอันเนื่องมาจากการไม่ใช้งานแขนในปัจจุบันอยู่ในระดับต่ำ"
    )

    return f"""## {emoji} รายงานผลการเอื้อมมือ 9-Grid — {severity}

**ระยะเวลารวมในการทดสอบ:** {summary.get('duration_sec', 0.0):.1f} วินาที
**จำนวนการเอื้อมแตะทั้งหมด:** {summary.get('total_hits', 0)} ครั้ง
**ระดับความเสี่ยง:** {risk_level}
**ดัชนีการไม่ใช้งานแขนข้างที่ไม่ถนัด (LNI):** {lni_percent}%

## สรุปสมรรถนะการทดสอบ

**แขนข้างที่ใช้งานหลัก (More Active): {more_active_th}**
- จำนวนการเอื้อมแตะ: {more_stats['reaches']} ครั้ง ({more_usage_ratio}% ของทั้งหมด)
- เวลาในการเอื้อมเฉลี่ย: {(more_stats['avg_reach_time'] / 1000.0):.2f} วินาที (ต่ำสุด: {(more_stats['min_reach_time'] / 1000.0):.2f}s, สูงสุด: {(more_stats['max_reach_time'] / 1000.0):.2f}s)
- ความตรงของเส้นทางการเคลื่อนไหว: {more_stats['avg_straightness'] * 100.0:.0f}% (ค่าสูงหมายถึงการเอื้อมเป็นเส้นตรงได้ดี)
- ความสั่นของการเคลื่อนไหว (Jerk): {more_stats['avg_jerk']:.1f} (ค่าต่ำหมายถึงการเคลื่อนไหวที่สมูทและนิ่ง)

**แขนข้างที่ใช้งานน้อย (Less Active): {less_active_th}**
- จำนวนการเอื้อมแตะ: {less_stats['reaches']} ครั้ง ({usage_ratio}% ของทั้งหมด)
- เวลาในการเอื้อมเฉลี่ย: {(less_stats['avg_reach_time'] / 1000.0):.2f} วินาที (ต่ำสุด: {(less_stats['min_reach_time'] / 1000.0):.2f}s, สูงสุด: {(less_stats['max_reach_time'] / 1000.0):.2f}s)
- ความตรงของเส้นทางการเคลื่อนไหว: {less_stats['avg_straightness'] * 100.0:.0f}%
- ความสั่นของการเคลื่อนไหว (Jerk): {less_stats['avg_jerk']:.1f}

## ข้อค้นพบทางคลินิก

- **ความเหลื่อมล้ำในการใช้งาน (Usage Disparity):** {less_active_th} ถูกเลือกใช้เอื้อมเพียง {usage_ratio}% ของเป้าหมายทั้งหมด แสดงถึงความพึงพอใจที่จะเลือกใช้ {more_active_th} มากกว่าอย่างชัดเจน
- **เวลาเอื้อมที่ช้าลง (Reach Delay):** {less_active_th} ใช้เวลาในการเอื้อมช้ากว่า {more_active_th} เฉลี่ย {speed_diff}% ซึ่งบ่งชี้ถึงความล่าช้าในการวางแผนการเคลื่อนไหว (motor planning delay) หรือความอ่อนแรงของกล้ามเนื้อ
- **ความเบี่ยงเบนของเส้นทาง (Path Deviation):** {less_active_th} มีความตรงในการเคลื่อนไหวที่ {less_stats['avg_straightness'] * 100.0:.0f}% เปรียบเทียบกับ {more_stats['avg_straightness'] * 100.0:.0f}% ของอีกข้าง สะท้อนถึงปัญหาการประสานงานของกล้ามเนื้อหรืออาการสั่น
- **ความสั่นสะท้าน (Tremor / Spasticity):** ค่า jerk ของ {less_active_th} อยู่ที่ {less_stats['avg_jerk']:.1f} เปรียบเทียบกับ {more_stats['avg_jerk']:.1f} ของอีกฝั่ง

## แผนการฟื้นฟูและเป้าหมาย

1. **เป้าหมายระยะสั้น (2-4 สัปดาห์):** บังคับการใช้งานของ {less_active_th} ในภารกิจการฝึกซ้อมเป็นเวลา 15-20 นาทีต่อวัน และพยายามลดระดับคะแนน LNI ให้ต่ำกว่า 40% ในการทดสอบซ้ำ
2. **การประสานงานสองมือ:** ฝึกทำกิจกรรมที่ต้องการการทรงตัวและจับพยุงด้วย {more_active_th} และการควบคุมหยิบจับด้วย {less_active_th}
3. **การออกกำลังกายที่บ้าน:** ทำกายภาพยืดเหยียดช่วงไหล่ร่วมกับการเอื้อมแตะเป้าหมาย (เช่น การติดกระดาษสีสติกเกอร์เป็นจุดตารางบนกำแพงแล้วเอื้อมแตะทีละสี)

## ประเมินความเสี่ยงต่อการหกล้ม

{fall_risk_desc}

---
*รายงานนี้สร้างด้วยระบบคัดกรองอัตโนมัติจากการประเมินทางจลนศาสตร์ (Kinematics) เพื่อวัตถุประสงค์ในการคัดกรองเบื้องต้นเท่านั้น ไม่ใช่การวินิจฉัยทางการแพทย์ และไม่ทดแทนการประเมินวิเคราะห์โดยบุคลากรทางการแพทย์*"""


def build_game_prompt_py(summary: dict) -> str:
    lni_percent = f"{summary.get('lni_score', 0.0) * 100:.1f}"
    duration_sec = f"{summary.get('duration_sec', 0.0):.0f}"
    total_hits = max(1, summary.get("total_hits", 1))
    
    left_usage_pct = f"{(summary.get('left_reaches', 0) / total_hits * 100):.0f}"
    right_usage_pct = f"{(summary.get('right_reaches', 0) / total_hits * 100):.0f}"
    
    left_fmt = (
        f"  - จำนวนครั้งที่ใช้แขนนี้แตะเป้า (usage): {summary.get('left_reaches', 0)} ครั้ง ({left_usage_pct}% ของทั้งหมด)\n"
        f"  - เวลาเอื้อมเฉลี่ย (เร็ว = น้อยกว่า): {(summary.get('left_avg_reach_time', 0.0) / 1000.0):.2f} วินาที\n"
        f"  - เวลาเอื้อม ต่ำสุด/สูงสุด: {(summary.get('left_min_reach_time', 0.0) / 1000.0):.2f}s / {(summary.get('left_max_reach_time', 0.0) / 1000.0):.2f}s\n"
        f"  - ความตรงของเส้นทาง (สูง = ตรง/ควบคุมดี): {summary.get('left_avg_straightness', 1.0) * 100.0:.0f}%\n"
        f"  - ความสั่นของการเคลื่อนไหว jerk (ต่ำ = นุ่มนวลกว่า): {summary.get('left_avg_jerk', 0.0):.1f}"
    )
    
    right_fmt = (
        f"  - จำนวนครั้งที่ใช้แขนนี้แตะเป้า (usage): {summary.get('right_reaches', 0)} ครั้ง ({right_usage_pct}% ของทั้งหมด)\n"
        f"  - เวลาเอื้อมเฉลี่ย (เร็ว = น้อยกว่า): {(summary.get('right_avg_reach_time', 0.0) / 1000.0):.2f} วินาที\n"
        f"  - เวลาเอื้อม ต่ำสุด/สูงสุด: {(summary.get('right_min_reach_time', 0.0) / 1000.0):.2f}s / {(summary.get('right_max_reach_time', 0.0) / 1000.0):.2f}s\n"
        f"  - ความตรงของเส้นทาง (สูง = ตรง/ควบคุมดี): {summary.get('right_avg_straightness', 1.0) * 100.0:.0f}%\n"
        f"  - ความสั่นของการเคลื่อนไหว jerk (ต่ำ = นุ่มนวลกว่า): {summary.get('right_avg_jerk', 0.0):.1f}"
    )
    
    one_sided = ""
    if summary.get('left_reaches', 0) == 0 or summary.get('right_reaches', 0) == 0:
        one_sided = "\n- หมายเหตุสำคัญ: ผู้ทดสอบใช้แขนเพียงข้างเดียวในการแตะเป้าตลอดการทดสอบ ซึ่งเป็นสัญญาณพฤติกรรม Learned Non-Use ที่ชัดเจนที่สุด"
        
    return f"""คุณคือผู้ช่วยนักกายภาพบำบัด (virtual physiotherapist) ที่วิเคราะห์ผลการทดสอบการเอื้อมมือแตะเป้าหมายแบบ 9-Grid ของผู้สูงอายุ เพื่อคัดกรองภาวะ Learned Non-Use, ความเสี่ยงกล้ามเนื้อฝ่อลีบ (Sarcopenia) และความเสี่ยงการหกล้ม

ข้อมูลที่วัดได้จากเซสชันนี้ (telemetry จริงจากกล้องด้วย MediaPipe Pose):
- ระยะเวลาทดสอบ: {duration_sec} วินาที
- จำนวนการเอื้อมแตะทั้งหมด: {summary.get('total_hits', 0)} ครั้ง
- Learned Non-Use Index (LNI): {lni_percent}%{one_sided}
- แขนซ้าย (Left):
{left_fmt}
- แขนขวา (Right):
{right_fmt}

วิธีคำนวณ LNI (เพื่อให้ตีความตัวเลขถูกต้อง): LNI เป็นค่าความไม่สมมาตรระหว่างแขน 0–100% รวมจาก 4 ด้าน ได้แก่ ความต่างของเวลาเอื้อม (น้ำหนัก 30%), ความต่างของความตรงเส้นทาง (20%), ความต่างของ jerk (20%) และความต่างของจำนวนครั้งที่เลือกใช้แต่ละแขน/usage (30%) — ยิ่ง LNI สูงยิ่งไม่สมมาตรและเสี่ยงมาก โปรดวิเคราะห์โดยอ้างอิงทั้ง 4 ด้านนี้ โดยเฉพาะ usage และเวลาเอื้อม

จงเขียน "รายงานคัดกรองทางคลินิก" เป็นภาษาไทย โดยใช้รูปแบบ Markdown ตามโครงสร้างนี้พอดี (ใช้ ## เป็นหัวข้อ, ใช้ - เป็นบุลเล็ต):

## (อีโมจิระดับความเสี่ยง) รายงานผลทดสอบการเอื้อมมือ 9-Grid — (ระดับความรุนแรง)
(สรุประยะเวลา จำนวนครั้ง ระดับความเสี่ยง และ LNI)

## สรุปผลการเคลื่อนไหว
(เปรียบเทียบแขนข้างที่ใช้งานมากกับข้างที่ใช้งานน้อย ด้วยตัวเลขจริง)

## ข้อค้นพบทางคลินิก
(วิเคราะห์ความไม่สมมาตรของการใช้แขน ความเร็ว ความตรงของเส้นทาง และการสั่น/jerk)

## แผนการฟื้นฟู
(ข้อแนะนำกายภาพบำบัดที่ทำได้จริง เช่นหลักการ CIMT, การออกกำลังที่บ้าน เป็นข้อ ๆ)

## ประเมินความเสี่ยงการหกล้ม
(ประเมินความเสี่ยงการหกล้มจากความไม่สมมาตรของแขน)

ข้อกำหนด:
- อ้างอิงตัวเลขจริงจากข้อมูลข้างต้นเสมอ ห้ามแต่งตัวเลขใหม่
- เลือกอีโมจิตามความรุนแรง: LNI < 15% ใช้ ✅, < 35% ใช้ ⚠️, < 55% ใช้ 🔶, ตั้งแต่ 55% ขึ้นไปใช้ 🔴
- ใช้ภาษากระชับ เข้าใจง่าย เหมาะกับการอ่านโดยผู้ดูแลผู้สูงอายุ
- ปิดท้ายด้วยบรรทัด: "*รายงานนี้สร้างด้วย AI เพื่อการคัดกรองเบื้องต้นเท่านั้น ไม่ใช่การวินิจฉัยทางการแพทย์ และไม่ทดแทนการประเมินโดยบุคลากรทางการแพทย์*"
- ตอบเฉพาะตัวรายงาน ไม่ต้องมีคำนำหรือคำอธิบายเพิ่ม"""


def generate_gemini_report_py(summary: dict, api_key: str) -> str:
    import json
    import urllib.request
    
    prompt = build_game_prompt_py(summary)
    
    model = "gemini-2.5-flash"
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.4,
            "maxOutputTokens": 2048,
            "thinkingConfig": {"thinkingBudget": 0}
        }
    }
    
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    
    with urllib.request.urlopen(req, timeout=20) as response:
        res_body = response.read().decode("utf-8")
        res_json = json.loads(res_body)
        
        candidate = res_json.get("candidates", [{}])[0]
        finish_reason = candidate.get("finishReason")
        if finish_reason and finish_reason != "STOP":
            raise RuntimeError(f"Gemini response incomplete: {finish_reason}")
            
        parts = candidate.get("content", {}).get("parts", [])
        text = "".join(p.get("text", "") for p in parts).strip()
        if not text:
            raise RuntimeError("Gemini returned empty text")
        return text


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
        st.markdown("<b>สถานะการบันทึก:</b>", unsafe_allow_html=True)
        rec_status_placeholder = st.empty()
    with status_col2:
        st.markdown("<b>เป้าหมายขณะนี้:</b>", unsafe_allow_html=True)
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
    active_cell_placeholder.markdown("<div class='metric-card' style='padding:5px; margin:0;'>ไม่มี</div>", unsafe_allow_html=True)

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
st.subheader("📊 กราฟวิเคราะห์การเคลื่อนไหว (Kinematics Telemetry Charts)")

chart_col1, chart_col2, chart_col3 = st.columns(3)
with chart_col1:
    st.markdown("<b>ความเร็วของแขน (Arm Speed - px/s)</b>", unsafe_allow_html=True)
    speed_chart_placeholder = st.empty()
with chart_col2:
    st.markdown("<b>ความเรียบเนียนของข้อต่อ (Movement Jerk)</b>", unsafe_allow_html=True)
    jerk_chart_placeholder = st.empty()
with chart_col3:
    st.markdown("<b>องศาการขยับไหล่ (Shoulder ROM - degrees)</b>", unsafe_allow_html=True)
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
    speed_chart_placeholder.info("รอข้อมูลการเคลื่อนไหว...")
    jerk_chart_placeholder.info("รอข้อมูลการเคลื่อนไหว...")
    rom_chart_placeholder.info("รอข้อมูลการเคลื่อนไหว...")


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
        st.session_state.ai_report = None
        st.session_state.analysis_summary = None

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
        import cv2

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
                    cells_str = ", ".join([f"ช่องที่ {c}" for c in result.active_cells])
                    cell_html = f"<div class='metric-card' style='padding:5px; margin:0; border-color:#EF4444; color:#EF4444;'>{cells_str}</div>"
                else:
                    cell_html = "<div class='metric-card' style='padding:5px; margin:0;'>ไม่มี</div>"
                active_cell_placeholder.markdown(cell_html, unsafe_allow_html=True)
                st.session_state.last_active_cell_html = cell_html

                # Recording status telemetry
                if result.is_recording:
                    rec_html = "<div class='status-badge-recording'>REC ACTIVE (บันทึกข้อมูล)</div>"
                else:
                    rec_html = "<div class='status-badge-demo'>DEMO MODE (ช่วงสอน/สาธิต)</div>"
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
        st.session_state.analysis_summary = summary
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
                label="📥 Download Annotated Video",
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


# --- Final results + Frame Inspector ---
if st.session_state.analysis_completed:
    st.markdown("<hr style='border-color: #143D66;'>", unsafe_allow_html=True)
    st.markdown(st.session_state.report_html, unsafe_allow_html=True)

    summary = st.session_state.get("analysis_summary")
    if summary:
        st.markdown("<hr style='border-color: #143D66;'>", unsafe_allow_html=True)
        st.subheader("📊 กราฟเปรียบเทียบผลการทดสอบ (Performance Comparison Charts)")
        
        # 1. Bar Chart: Left vs Right Comparison
        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            st.markdown("<b>เปรียบเทียบแขนซ้าย vs แขนขวา (Left vs Right Comparison)</b>", unsafe_allow_html=True)
            left_jerk = summary.get("left_avg_jerk", 0.0)
            right_jerk = summary.get("right_avg_jerk", 0.0)
            
            bar_df = pd.DataFrame({
                "Metric": ["Reach Duration (s)", "Path Straightness (%)", "Smoothness Index"],
                "Left Arm": [
                    summary.get("left_avg_reach_time", 0.0) / 1000.0,
                    summary.get("left_avg_straightness", 1.0) * 100.0,
                    1.0 / (1.0 + left_jerk * 0.01) if left_jerk > 0 else 1.0
                ],
                "Right Arm": [
                    summary.get("right_avg_reach_time", 0.0) / 1000.0,
                    summary.get("right_avg_straightness", 1.0) * 100.0,
                    1.0 / (1.0 + right_jerk * 0.01) if right_jerk > 0 else 1.0
                ]
            }).set_index("Metric")
            st.bar_chart(bar_df)
            
        with chart_col2:
            st.markdown("<b>ระยะเวลาการเอื้อมแตะแต่ละครั้ง (Individual Reach Times)</b>", unsafe_allow_html=True)
            reaches_list = summary.get("reaches", [])
            if reaches_list:
                reaches_df = pd.DataFrame([
                    {
                        "Reach #": r["index"],
                        "Time (s)": r["reachTimeMs"] / 1000.0,
                        "Arm": "Left Arm" if r["arm"] == "left" else "Right Arm"
                    }
                    for r in reaches_list
                ])
                st.scatter_chart(reaches_df, x="Reach #", y="Time (s)", color="Arm")
            else:
                st.info("ไม่มีข้อมูลการเอื้อมแตะในเซสชันนี้")

        # 2. AI / Rule-based Clinical Screening Report
        st.markdown("<hr style='border-color: #143D66;'>", unsafe_allow_html=True)
        st.subheader("📋 รายงานสรุปผลคัดกรองทางคลินิก (Clinical Screening Report)")
        
        if st.session_state.get("ai_report") is None:
            api_info = " (ใช้โมเดล Gemini)" if gemini_key else " (แบบอิงเกณฑ์ทางคลินิก - Rule-based)"
            if st.button(f"📝 สร้างรายงานคัดกรอง{api_info}", key="gen_ai_report_btn", use_container_width=True):
                with st.spinner("กำลังวิเคราะห์ผลข้อมูลจลนศาสตร์..."):
                    try:
                        if gemini_key:
                            report_txt = generate_gemini_report_py(summary, gemini_key)
                        else:
                            report_txt = generate_game_clinical_report(summary)
                        st.session_state.ai_report = report_txt
                        st.rerun()
                    except Exception as e:
                        st.error(f"การเรียกใช้งาน AI ขัดข้อง: {e} (กำลังสลับมาสร้างรายงานแบบอิงเกณฑ์แทน...)")
                        report_txt = generate_game_clinical_report(summary)
                        st.session_state.ai_report = report_txt
                        st.rerun()
        else:
            st.markdown(
                f"<div style='background-color:#1E293B; border:1px solid #334155; padding:20px; border-radius:12px; margin-bottom:15px;'>"
                f"{st.session_state.ai_report}"
                f"</div>",
                unsafe_allow_html=True
            )
            
            dl_col1, dl_col2 = st.columns([1, 3])
            with dl_col1:
                st.download_button(
                    label="📥 ดาวน์โหลดรายงาน (.md)",
                    data=st.session_state.ai_report,
                    file_name="clinical_screening_report.md",
                    mime="text/markdown",
                    use_container_width=True
                )
            with dl_col2:
                if st.button("🔄 สร้างรายงานใหม่ (Regenerate)", key="regen_report_btn"):
                    st.session_state.ai_report = None
                    st.rerun()

    if st.session_state.video_bytes:
        st.sidebar.download_button(
            label="📥 Download Annotated Video",
            data=st.session_state.video_bytes,
            file_name="annotated_dexterity_assessment.mp4",
            mime="video/mp4",
            use_container_width=True,
            key="dl_video_persist",
        )

    st.markdown("<hr style='border-color: #143D66;'>", unsafe_allow_html=True)
    st.subheader("🔍 ค้นหาและตรวจสอบรายเฟรม (Frame Inspector)")

    df_history = pd.DataFrame(st.session_state.frame_history)

    col_dl, col_search = st.columns([1, 2])
    with col_dl:
        csv_data = df_history.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            label="📥 ดาวน์โหลดข้อมูลทุกเฟรม (CSV)",
            data=csv_data,
            file_name="detailed_frame_log.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with col_search:
        search_frame = st.number_input("ป้อนหมายเลขเฟรมที่ต้องการค้นหา (เช่น 120):", min_value=1, step=1, value=1)
        if search_frame is not None and not df_history.empty:
            max_processed = df_history["Frame Index"].max()
            if search_frame > max_processed:
                st.error(f"❌ ยังประมวลผลไปไม่ถึงเฟรมนี้ (ประมวลผลถึงเฟรมสูงสุด: {max_processed})")
            else:
                match = df_history[df_history["Frame Index"] == search_frame]
                if not match.empty:
                    row = match.iloc[0]
                    st.success(f"พบข้อมูลเฟรมที่ {search_frame} (เวลา: {row['Timestamp (sec)']:.2f} วินาที)")
                    m_col1, m_col2 = st.columns(2)
                    with m_col1:
                        st.write(f"**เป้าหมายช่องที่:** {row['Active Target Cell']}")
                        if row["Active Target Cell"] != "None":
                            st.caption(f"พื้นที่เป้าหมาย: X: {row['Target X1 (%)']:.1f}%-{row['Target X2 (%)']:.1f}% | Y: {row['Target Y1 (%)']:.1f}%-{row['Target Y2 (%)']:.1f}%")
                        st.write(f"**สถานะชนมือซ้าย (Left Hit):** {row['Left Hand Hit']}")
                        st.write(f"**สถานะชนมือขวา (Right Hit):** {row['Right Hand Hit']}")
                    with m_col2:
                        st.write(f"**พิกัดมือซ้าย (Left Hand):** X: {row['Left Hand X (%)']} Y: {row['Left Hand Y (%)']}")
                        st.write(f"**พิกัดมือขวา (Right Hand):** X: {row['Right Hand X (%)']} Y: {row['Right Hand Y (%)']}")
                else:
                    st.warning("⚠️ ไม่พบข้อมูลสำหรับเฟรมนี้ (อาจโดนข้ามระหว่างประมวลผล)")
