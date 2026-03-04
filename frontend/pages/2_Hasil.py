"""
Page 2 - Hasil Transkripsi (Transcript Results)

Shows:
  - Processing status with auto-refresh every 5 s
  - Meeting summary (stats, speaker breakdown)
  - Dialog view (speaker-coloured segments, native Streamlit only)
  - Raw text view + download
"""
from __future__ import annotations

import time

import streamlit as st

st.set_page_config(
    page_title="Hasil - MeetRecall AI",
    layout="wide",
)

from utils.styles import inject          
from utils.api_client import api_client  

inject()

SPEAKER_COLORS = ["blue", "violet", "green", "orange", "red", "gray"]

# Session state
if "current_meeting_id" not in st.session_state:
    st.session_state.current_meeting_id = ""

# Sidebar - meeting selector
with st.sidebar:
    st.markdown("### Pilih Meeting")
    meetings = api_client.list_meetings()
    all_meetings = [
        m for m in meetings
        if m.get("status") in ("processing", "completed", "failed")
    ]

    if all_meetings:
        def _label(m: dict) -> str:
            icon = {
                "processing": "[PROSES]",
                "completed":  "[SELESAI]",
                "failed":     "[GAGAL]",
            }.get(m.get("status", ""), "[?]")
            return f"{icon} {m['filename']} ({m['meeting_id'][:8]}...)"

        options = {_label(m): m["meeting_id"] for m in all_meetings}
        default_key = next(
            (k for k, v in options.items()
             if v == st.session_state.current_meeting_id),
            list(options.keys())[0],
        )
        selected_label = st.selectbox(
            "Meeting",
            list(options.keys()),
            index=list(options.keys()).index(default_key),
            label_visibility="collapsed",
        )
        meeting_id = options[selected_label]
    else:
        meeting_id = st.text_input(
            "Meeting ID",
            value=st.session_state.current_meeting_id,
            placeholder="Paste meeting ID",
            label_visibility="collapsed",
        )

    if meeting_id:
        st.session_state.current_meeting_id = meeting_id

# Page header
st.title("Hasil Transkripsi")
st.caption("Ringkasan, speaker breakdown, dan transkrip lengkap hasil meeting.")

if not meeting_id:
    st.info(
        "Belum ada meeting yang dipilih.  \n"
        "Upload file di halaman **1 Upload** terlebih dahulu, "
        "lalu pilih meeting dari sidebar."
    )
    st.stop()

# Primary poll: GET /api/transcript/{id}
transcript_data = api_client.get_transcript(meeting_id)

# Processing state (HTTP 409)
if transcript_data.get("status") == "processing":
    status_data = api_client.get_status(meeting_id)
    fname      = status_data.get("filename", "-") if "error" not in status_data else "-"
    prov       = status_data.get("provider", "-").upper() if "error" not in status_data else "-"
    dur_so_far = status_data.get("duration", 0) if "error" not in status_data else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Status",   "PROCESSING")
    c2.metric("Provider", prov)
    c3.metric("File",     (fname or "-")[:20])
    c4.metric("Durasi",   f"{dur_so_far:.0f}s" if dur_so_far else "-")

    st.info(
        "**Transkripsi Sedang Berjalan...**\n\n"
        "AI sedang memproses rekaman Anda. "
        "Halaman ini **auto-refresh setiap 5 detik**.\n\n"
        "**Tahap proses:**\n"
        "- Mengonversi audio ke WAV 16 kHz mono (ffmpeg)\n"
        "- Whisper-small melakukan transkripsi (ASR)\n"
        "- Pyannote 3.1 mengidentifikasi speaker\n"
        "- Membangun FAISS index untuk RAG dan clustering"
    )

    time.sleep(5)
    st.rerun()

# Failed state (HTTP 500)
if transcript_data.get("status") == "failed":
    st.error(
        "**Transkripsi Gagal**\n\n"
        "Proses transkripsi mengalami error. "
        "Silakan upload ulang file di halaman **1 Upload**."
    )
    st.stop()

# Generic error (404 / network)
if "error" in transcript_data:
    st.error(transcript_data["error"])
    st.stop()

segments  = transcript_data.get("segments", [])
full_text = transcript_data.get("full_text", "")
duration  = transcript_data.get("duration", 0)
filename  = transcript_data.get("filename", "-")
provider  = transcript_data.get("provider", "-")

# Speaker index (0-5) for colour mapping
spk_idx: dict[str, int] = {}
for seg in segments:
    spk = seg["speaker"]
    if spk not in spk_idx:
        spk_idx[spk] = len(spk_idx) % len(SPEAKER_COLORS)

speakers = list(spk_idx.keys())

# ── Section 1: Summary ────────────────────────────────────────────────────────
st.subheader("1. Ringkasan Meeting")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Segmen",   len(segments))
c2.metric("Speaker",  len(speakers))
c3.metric("Durasi",   f"{duration:.0f}s")
c4.metric("Provider", provider.upper())

# Speaker breakdown
if speakers:
    st.markdown("#### Speaker Breakdown")
    spk_seg_counts: dict[str, int] = {}
    for seg in segments:
        spk_seg_counts[seg["speaker"]] = spk_seg_counts.get(seg["speaker"], 0) + 1

    total_segs = len(segments) or 1
    for spk, cnt in sorted(spk_seg_counts.items(), key=lambda x: -x[1]):
        pct   = cnt / total_segs * 100
        color = SPEAKER_COLORS[spk_idx.get(spk, 0)]
        col_name, col_bar, col_count = st.columns([2, 5, 2])
        col_name.markdown(f"**:{color}[{spk}]**")
        col_bar.progress(pct / 100)
        col_count.markdown(f"{cnt} segmen ({pct:.0f}%)")

# Meeting detail
with st.expander("Detail Meeting", expanded=False):
    st.markdown(f"""
| Field | Value |
|---|---|
| **Meeting ID** | `{meeting_id}` |
| **File** | {filename} |
| **Provider** | {provider} |
| **Durasi** | {duration:.1f} detik |
| **Jumlah Segmen** | {len(segments)} |
| **Jumlah Speaker** | {len(speakers)} |
    """)

st.divider()

# ── Section 2: Transcript ─────────────────────────────────────────────────────
st.subheader("2. Transkrip")

tab_dialog, tab_raw = st.tabs(["Dialog", "Teks Lengkap"])

with tab_dialog:
    if segments:
        for seg in segments:
            color = SPEAKER_COLORS[spk_idx.get(seg["speaker"], 0)]
            ts    = f"{seg['start_time']:.1f}s"
            col_spk, col_txt = st.columns([1, 5])
            with col_spk:
                st.markdown(f"**:{color}[{seg['speaker']}]**")
                st.caption(ts)
            with col_txt:
                st.write(seg["text"])
            st.divider()
    else:
        st.info("Tidak ada segmen yang ditemukan.")

with tab_raw:
    st.text_area("Teks lengkap", full_text, height=500, label_visibility="collapsed")
    st.download_button(
        "Download .txt",
        data=full_text,
        file_name=f"{meeting_id[:8]}_transcript.txt",
        mime="text/plain",
    )

st.divider()
st.info(
    f"Meeting **{meeting_id[:8]}...** siap digunakan.  \n"
    "Buka **3 AI Chat** untuk tanya jawab, "
    "atau **4 Topic Clusters** untuk eksplorasi topik."
)