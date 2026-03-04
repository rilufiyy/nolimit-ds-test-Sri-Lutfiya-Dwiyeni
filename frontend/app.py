"""
MeetRecall AI - Landing page.
"""
import streamlit as st

st.set_page_config(
    page_title="MeetRecall AI",
    layout="wide",
    initial_sidebar_state="expanded",
)

from utils.styles import inject          # noqa: E402
from utils.api_client import api_client  # noqa: E402

inject()

# Session state defaults
if "current_meeting_id" not in st.session_state:
    st.session_state.current_meeting_id = ""

# Hero banner
st.markdown("""
<div style="
    background: linear-gradient(135deg, #1e1b4b 0%, #4f46e5 55%, #7c3aed 100%);
    border-radius: 20px;
    padding: 48px 44px;
    margin-bottom: 32px;
    color: white;
">
    <h1 style="font-size:2.8rem; font-weight:900; margin:0 0 10px 0; color:white; letter-spacing:-1px;">
        MeetRecall AI
    </h1>
    <p style="font-size:1.1rem; opacity:0.88; margin:0; max-width:580px; line-height:1.7;">
        Transform meeting recordings into actionable intelligence.
        Transcription, RAG chatbot, and topic clustering
        powered by open-source HuggingFace models.
    </p>
    <div style="margin-top:22px; display:flex; gap:10px; flex-wrap:wrap;">
        <span style="background:rgba(255,255,255,0.17); padding:5px 14px; border-radius:999px; font-size:13px; font-weight:600;">HuggingFace</span>
        <span style="background:rgba(255,255,255,0.17); padding:5px 14px; border-radius:999px; font-size:13px; font-weight:600;">Runs Locally</span>
        <span style="background:rgba(255,255,255,0.17); padding:5px 14px; border-radius:999px; font-size:13px; font-weight:600;">FAISS + RAG</span>
        <span style="background:rgba(255,255,255,0.17); padding:5px 14px; border-radius:999px; font-size:13px; font-weight:600;">Topic Clustering</span>
        <span style="background:rgba(255,255,255,0.17); padding:5px 14px; border-radius:999px; font-size:13px; font-weight:600;">GPU Accelerated</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Backend status
health = api_client.health()
status = health.get("status", "unreachable")

if status == "healthy":
    st.markdown("""<div class="success-box">
        Backend connected — ffmpeg ready, all systems operational.
    </div>""", unsafe_allow_html=True)
elif status == "degraded":
    st.markdown("""<div class="warn-box">
        Backend running but ffmpeg is missing — audio conversion may fail.<br>
        <small>Rebuild: <code>docker-compose up -d --build</code></small>
    </div>""", unsafe_allow_html=True)
else:
    st.error("Backend unreachable. Start with: docker-compose up -d")

st.markdown("<br>", unsafe_allow_html=True)

# Feature cards
col1, col2, col3 = st.columns(3, gap="large")

with col1:
    st.markdown("""
    <div class="feat-card">
        <div class="feat-title">Transcribe</div>
        <div class="feat-desc">
            Upload audio or video recordings and get a speaker-labelled transcript using
            <strong>openai/whisper-small</strong> + <strong>pyannote 3.1</strong>,
            or the fast <strong>AssemblyAI</strong> cloud provider.
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feat-card">
        <div class="feat-title">AI Chat (RAG)</div>
        <div class="feat-desc">
            Ask natural language questions about any meeting. Answers are grounded in
            the transcript via <strong>FAISS semantic search</strong> +
            <strong>google/flan-t5-base</strong> generation with cited sources.
        </div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="feat-card">
        <div class="feat-title">Topic Clusters</div>
        <div class="feat-desc">
            Discover recurring themes from transcript embeddings clustered with
            <strong>KMeans</strong> or <strong>HDBSCAN</strong>, using
            <strong>paraphrase-multilingual-MiniLM-L12-v2</strong>.
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Quick start guide
with st.expander("Quick Start Guide", expanded=False):
    st.markdown("""
**1. Configure environment**
```bash
cp backend/.env.example backend/.env
# HUGGINGFACE_TOKEN -- required for Pyannote speaker diarization
# ASSEMBLY_API_KEY  -- optional, for AssemblyAI cloud provider
```

**2. Start services**
```bash
docker-compose up -d --build
# UI  -> http://localhost:8502
# API -> http://localhost:8080/docs
```

**3. Transcribe a meeting**
- Open **1 Upload** from the sidebar
- Choose provider (Local = free; AssemblyAI = fast)
- Upload your file, click **Start Transcription**
- Switch to **2 Hasil** once processing completes

**4. Explore**
- **3 AI Chat** -- ask questions about the meeting
- **4 Topic Clusters** -- discover themes discussed

GPU: If you have an NVIDIA GPU, processing is significantly faster.
Ensure NVIDIA Container Toolkit is installed on your system.
    """)

# Sidebar
with st.sidebar:
    st.markdown("### Navigate")
    st.markdown("""
| Page | Function |
|---|---|
| **1 Upload** | Upload and transcribe |
| **2 Hasil** | View transcript and summary |
| **3 AI Chat** | RAG question answering |
| **4 Topics** | Theme discovery |
    """)
    st.divider()
    st.markdown("### Models")
    st.markdown("""
- openai/whisper-small
- pyannote/speaker-diarization-3.1
- paraphrase-multilingual-MiniLM-L12-v2
- google/flan-t5-base
    """)