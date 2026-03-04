"""
Page 4 - Topic Clusters (Section C - semantic labels)

Cluster labels sekarang dihasilkan oleh LLM (flan-t5) sehingga
mencerminkan topik diskusi yang sebenarnya:
  Sebelum: "Topic 0", "Topic 1"
  Sesudah: "Model Deployment Discussion", "Project Timeline Planning"
"""
from __future__ import annotations

import streamlit as st
import pandas as pd

st.set_page_config(page_title="Topic Clusters - MeetRecall AI", layout="wide")

from utils.styles import inject
from utils.api_client import api_client

inject()

if "current_meeting_id" not in st.session_state:
    st.session_state.current_meeting_id = ""

with st.sidebar:
    st.markdown("### Pilih Meeting")
    meetings  = api_client.list_meetings()
    completed = [m for m in meetings if m.get("status") == "completed"]
    if completed:
        opts = {f"{m['filename']} ({m['meeting_id'][:8]}...)": m["meeting_id"] for m in completed}
        sel_lbl = st.selectbox("Meeting", list(opts.keys()), label_visibility="collapsed")
        meeting_id = opts[sel_lbl]
    else:
        meeting_id = st.text_input("Meeting ID", value=st.session_state.current_meeting_id, placeholder="Paste meeting ID", label_visibility="collapsed")
    if meeting_id:
        st.session_state.current_meeting_id = meeting_id
    st.divider()
    st.markdown("### Clustering Options")
    method = st.radio("Algorithm", ["kmeans", "hdbscan"], index=0)
    n_clusters = st.slider("Number of clusters (KMeans)", min_value=2, max_value=15, value=5, disabled=(method=="hdbscan"))
    run_btn = st.button("Run Clustering", type="primary", use_container_width=True)

st.markdown("<h1 style='font-size:2.1rem;font-weight:900;margin-bottom:4px;'>Topic Clusters</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:#64748b;font-size:1rem;margin-bottom:20px;'>Segment embeddings dicluster untuk menemukan tema utama meeting. Label dihasilkan LLM secara semantik.</p>", unsafe_allow_html=True)

if not meeting_id:
    st.info("Pilih meeting dari sidebar, atau upload file di halaman 1 Upload.")
    st.stop()

st.markdown(f"<p style='color:#94a3b8;font-size:13px;'>Meeting ID: <code>{meeting_id}</code></p>", unsafe_allow_html=True)

if run_btn:
    with st.spinner("Menjalankan clustering + LLM labeling..."):
        result = api_client.run_clustering(meeting_id, method=method, n_clusters=n_clusters)
    if "error" in result:
        st.error(f"Clustering gagal: {result['error']}")
        st.stop()
    st.session_state[f"clusters_{meeting_id}"] = result
    st.success("Clustering selesai. Label topik dihasilkan secara semantik.")
else:
    if f"clusters_{meeting_id}" not in st.session_state:
        cached = api_client.get_clusters(meeting_id)
        if "error" not in cached:
            st.session_state[f"clusters_{meeting_id}"] = cached

result = st.session_state.get(f"clusters_{meeting_id}")
if not result:
    st.info("Belum ada hasil clustering. Klik Run Clustering di sidebar.")
    st.stop()

topics = result.get("topics", [])
if not topics:
    st.warning("Tidak ada topik yang ditemukan (kemungkinan segmen terlalu sedikit).")
    st.stop()

st.divider()
c1, c2, c3 = st.columns(3)
c1.metric("Topik Ditemukan", result.get("n_clusters_found", len(topics)))
c2.metric("Algoritma", result.get("method","-").upper())
c3.metric("Total Segmen", sum(t["size"] for t in topics))

st.markdown("<br>", unsafe_allow_html=True)

# Bar chart - pakai label semantik bukan "Topik #N"
st.markdown("#### Distribusi Cluster")
df = pd.DataFrame([{"Label": t["label"][:40], "Segmen": t["size"]} for t in topics])
st.bar_chart(df.set_index("Label"), color="#6366f1")

st.divider()
st.markdown("#### Detail Setiap Topik")

COLORS = ["#6366f1","#ec4899","#22c55e","#f97316","#a855f7","#14b8a6","#eab308","#ef4444"]

for topic in topics:
    cid      = topic["cluster_id"]
    label    = topic.get("label", f"Cluster {cid}")
    size     = topic["size"]
    speakers = ", ".join(topic.get("speakers", [])) or "-"
    t_start  = topic.get("time_range", {}).get("start", 0)
    t_end    = topic.get("time_range", {}).get("end",   0)
    color    = COLORS[cid % len(COLORS)]

    with st.expander(f"{label}  |  {size} segmen  |  {t_start:.0f}s - {t_end:.0f}s"):
        col1, col2, col3 = st.columns(3)
        col1.markdown(f"**Speakers:** {speakers}")
        col2.markdown(f"**Range:** {t_start:.1f}s - {t_end:.1f}s")
        col3.markdown(f"**Segmen:** {size}")

        sample_texts = topic.get("sample_texts", [])
        if sample_texts:
            st.markdown("**Contoh teks dari cluster ini:**")
            for txt in sample_texts:
                st.markdown(f"> {txt}")