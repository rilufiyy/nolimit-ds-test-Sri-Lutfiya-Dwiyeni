"""
Page 5 - Sentiment Analysis per Speaker

Menampilkan analisis sentimen (Positif / Netral / Negatif) per speaker
berdasarkan segmen transkripsi menggunakan model lokal HuggingFace.
"""
from __future__ import annotations

import streamlit as st
import pandas as pd

st.set_page_config(page_title="Sentiment - MeetRecall AI", layout="wide")

from utils.styles import inject
from utils.api_client import api_client

inject()

if "current_meeting_id" not in st.session_state:
    st.session_state.current_meeting_id = ""
if "sentiment_cache" not in st.session_state:
    st.session_state.sentiment_cache = {}

# Sidebar
with st.sidebar:
    st.markdown("### Pilih Meeting")
    meetings  = api_client.list_meetings()
    completed = [m for m in meetings if m.get("status") == "completed"]
    if completed:
        opts = {f"{m['filename']} ({m['meeting_id'][:8]}...)": m["meeting_id"] for m in completed}
        def_k = next(
            (k for k, v in opts.items() if v == st.session_state.current_meeting_id),
            list(opts.keys())[0],
        )
        sel_lbl = st.selectbox(
            "Meeting", list(opts.keys()),
            index=list(opts.keys()).index(def_k),
            label_visibility="collapsed",
        )
        meeting_id = opts[sel_lbl]
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
st.title("Sentiment Analysis")
st.caption("Analisis sentimen per speaker berdasarkan segmen transkripsi (model lokal HuggingFace).")

if not meeting_id:
    st.info("Pilih meeting dari sidebar, atau upload file di halaman 1 Upload.")
    st.stop()

st.caption(f"Meeting ID: `{meeting_id}`")
st.divider()

_sk = f"sent_{meeting_id}"

# Run / Refresh controls
col_run, col_refresh = st.columns([3, 1])
with col_run:
    run_btn = st.button("Jalankan Analisis Sentiment", type="primary", use_container_width=True)
with col_refresh:
    if st.button("Reset", use_container_width=True) and _sk in st.session_state.sentiment_cache:
        del st.session_state.sentiment_cache[_sk]
        st.rerun()

if run_btn:
    with st.spinner("Menjalankan sentiment analysis (model lokal)..."):
        result = api_client.get_sentiment(meeting_id)
    if "error" in result:
        st.error(f"Gagal: {result['error']}")
        st.stop()
    st.session_state.sentiment_cache[_sk] = result
    st.rerun()

# Display
if _sk not in st.session_state.sentiment_cache:
    st.info("Klik **Jalankan Analisis Sentiment** untuk memulai.")
    st.stop()

data = st.session_state.sentiment_cache[_sk]
speaker_sentiment: dict = data.get("speaker_sentiment", {})

if not speaker_sentiment:
    st.warning("Tidak ada data sentimen yang ditemukan.")
    st.stop()

# Summary metrics
num_speakers = len(speaker_sentiment)
avg_pos = sum(s.get("positive", 0) for s in speaker_sentiment.values()) / num_speakers
avg_neg = sum(s.get("negative", 0) for s in speaker_sentiment.values()) / num_speakers
avg_neu = sum(s.get("neutral",  0) for s in speaker_sentiment.values()) / num_speakers

c1, c2, c3, c4 = st.columns(4)
c1.metric("Speaker Dianalisis", num_speakers)
c2.metric("Rata-rata Positif",  f"{avg_pos*100:.1f}%")
c3.metric("Rata-rata Netral",   f"{avg_neu*100:.1f}%")
c4.metric("Rata-rata Negatif",  f"{avg_neg*100:.1f}%")

st.divider()

# Bar chart grouped by speaker
st.subheader("Distribusi Sentimen per Speaker")
rows = [
    {
        "Speaker":  sp,
        "Positif":  round(s.get("positive", 0) * 100, 1),
        "Netral":   round(s.get("neutral",  0) * 100, 1),
        "Negatif":  round(s.get("negative", 0) * 100, 1),
    }
    for sp, s in speaker_sentiment.items()
]
df = pd.DataFrame(rows).set_index("Speaker")
st.bar_chart(df, color=["#22c55e", "#94a3b8", "#ef4444"])

st.divider()

# Per-speaker detail
st.subheader("Detail per Speaker")
for sp, s in speaker_sentiment.items():
    pos = s.get("positive", 0)
    neu = s.get("neutral",  0)
    neg = s.get("negative", 0)
    dominant = max({"Positif": pos, "Netral": neu, "Negatif": neg}, key=lambda k: {"Positif": pos, "Netral": neu, "Negatif": neg}[k])

    with st.expander(f"{sp}  —  dominan: {dominant}"):
        ca, cb, cc = st.columns(3)
        ca.metric("Positif", f"{pos*100:.1f}%")
        cb.metric("Netral",  f"{neu*100:.1f}%")
        cc.metric("Negatif", f"{neg*100:.1f}%")

        col_lbl, col_bar = st.columns([1, 5])
        col_lbl.markdown("**Positif**")
        col_bar.progress(pos)
        col_lbl, col_bar = st.columns([1, 5])
        col_lbl.markdown("**Netral**")
        col_bar.progress(neu)
        col_lbl, col_bar = st.columns([1, 5])
        col_lbl.markdown("**Negatif**")
        col_bar.progress(neg)
