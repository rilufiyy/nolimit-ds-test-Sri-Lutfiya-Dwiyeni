"""
Page 1 - Upload and Transcribe

Steps:
  1. Choose provider (Local Whisper+Pyannote | AssemblyAI)
  2. Upload audio/video file
  3. Poll /api/status/{id} every 5 s until done
  4. Navigate to Results page to view transcript
"""
from __future__ import annotations

import time

import streamlit as st

st.set_page_config(
    page_title="Upload - MeetRecall AI",
    layout="wide",
)

from utils.styles import inject          
from utils.api_client import api_client  

inject()

# Session state
if "current_meeting_id" not in st.session_state:
    st.session_state.current_meeting_id = ""

# Page header
st.markdown("""
<h1 style="font-size:2.1rem; font-weight:900; margin-bottom:4px;">Upload and Transcribe</h1>
<p style="color:#64748b; font-size:1rem; margin-bottom:28px;">
    Upload an audio or video recording to start speaker-labelled transcription.
</p>
""", unsafe_allow_html=True)

# Step 1 - Provider selection
st.markdown("""
<div class="step-row">
    <span class="step-num">1</span>
    <span class="step-title">Choose Transcription Provider</span>
</div>
""", unsafe_allow_html=True)

col_l, col_r, _ = st.columns([1, 1, 1])
with col_l:
    provider_label = st.radio(
        "Provider",
        [
            "Local - Whisper + Pyannote (free, private)",
            "AssemblyAI - Universal-3 Pro (cloud, fast)",
        ],
        index=0,
        label_visibility="collapsed",
    )
provider = "assemblyai" if "AssemblyAI" in provider_label else "local"

if provider == "local":
    st.markdown("""<div class="info-box">
        <strong>openai/whisper-small</strong> (ASR) +
        <strong>pyannote/speaker-diarization-3.1</strong> (speaker ID).<br>
        Free and private - runs entirely in Docker. Requires
        <code>HUGGINGFACE_TOKEN</code> in <code>backend/.env</code>.
        GPU is detected automatically for faster processing.
    </div>""", unsafe_allow_html=True)
else:
    st.markdown("""<div class="info-box">
        <strong>AssemblyAI Universal-3 Pro</strong> - cloud transcription with
        built-in speaker labels.<br>
        Fast (2-4 min for a 60-min meeting). Requires
        <code>ASSEMBLY_API_KEY</code> in <code>backend/.env</code>.
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Step 2 - Upload file
st.markdown("""
<div class="step-row">
    <span class="step-num">2</span>
    <span class="step-title">Upload Recording</span>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Supported formats: MP4, MKV, MOV, AVI, MP3, WAV, M4A, WEBM",
    type=["mp4", "mkv", "mov", "avi", "mp3", "wav", "m4a", "webm"],
)

col_btn, _ = st.columns([1, 3])
with col_btn:
    start_btn = st.button(
        "Start Transcription",
        type="primary",
        disabled=uploaded_file is None,
        use_container_width=True,
    )

if start_btn and uploaded_file:
    with st.spinner("Uploading to backend..."):
        result = api_client.upload_file(uploaded_file, provider=provider)

    if "error" in result:
        st.error(f"Upload failed: {result['error']}")
    else:
        meeting_id = result["meeting_id"]
        st.session_state.current_meeting_id = meeting_id
        st.session_state["_poll_mid"] = meeting_id
        st.markdown(f"""<div class="success-box">
            <strong>Upload accepted.</strong><br>
            Meeting ID: <code>{meeting_id}</code><br>
            <small>Transcription started in background - status updates below.</small>
        </div>""", unsafe_allow_html=True)
        time.sleep(0.4)
        st.rerun()

st.divider()

# Step 3 - Status polling
st.markdown("""
<div class="step-row">
    <span class="step-num">3</span>
    <span class="step-title">Processing Status</span>
</div>
""", unsafe_allow_html=True)

mid_input = st.text_input(
    "Meeting ID",
    placeholder="Auto-filled after upload - or paste an existing ID",
    key="_poll_mid",
)

if not mid_input:
    st.markdown("""<div class="info-box">
        Upload a file above, or paste a Meeting ID to check status.
    </div>""", unsafe_allow_html=True)
    st.stop()

# Sync current_meeting_id with whatever is in the text input
st.session_state.current_meeting_id = mid_input

status_data = api_client.get_status(mid_input)
if "error" in status_data:
    st.error(status_data["error"])
    st.stop()

status = status_data.get("status", "unknown")

# Metrics row
c1, c2, c3, c4 = st.columns(4)
c1.metric("Status",   status.upper())
c2.metric("Provider", status_data.get("provider", "-").upper())
c3.metric("Duration", f"{status_data.get('duration', 0):.1f}s")
c4.metric("File",     (status_data.get("filename", "-") or "-")[:20])

st.markdown("<br>", unsafe_allow_html=True)

# Status-specific UI
if status == "processing":
    st.markdown("""
    <div class="proc-box">
        <h3 style="color:#b45309; margin:0 0 8px 0;">Transcription in Progress</h3>
        <p style="color:#78716c; font-size:14px; margin:0; line-height:1.7;">
            The AI is processing your recording.<br>
            This page auto-refreshes every <strong>5 seconds</strong>.<br>
            Local models may take several minutes depending on file length.
        </p>
    </div>
    """, unsafe_allow_html=True)
    time.sleep(5)
    st.rerun()

elif status == "failed":
    st.error("Transcription failed. Please re-upload the file and try again.")

elif status == "completed":
    st.markdown("""<div class="success-box">
        <strong>Transcription complete.</strong>
        Your meeting has been indexed for AI Chat and Topic Clusters.
    </div>""", unsafe_allow_html=True)
    st.markdown("""<div class="info-box" style="margin-top:12px;">
        Open <strong>2 Hasil</strong> in the sidebar to view the transcript and summary.<br>
        Or head to <strong>3 AI Chat</strong> to ask questions about this meeting.
    </div>""", unsafe_allow_html=True)