"""
Page 3 - AI Chat (RAG) - meeting intelligence edition.

Renders structured response (Section A/B):
  transcription_summary, key_actions, keywords, sources

Sidebar (Section E): speaker chart + sentiment visualization.
"""
from __future__ import annotations
import streamlit as st
import pandas as pd

st.set_page_config(page_title="AI Chat - MeetRecall AI", layout="wide")

from utils.styles import inject
from utils.api_client import api_client

inject()

for _k, _v in [("current_meeting_id",""),("chat_messages",[]),("chat_meeting_id","")]:
    if _k not in st.session_state:
        st.session_state[_k] = _v


def _render_assistant(msg: dict) -> None:
    """Render AI chat response — clean prose first, details in collapsed sections."""
    # Main answer as natural prose
    answer = msg.get("transcription_summary", "")
    st.markdown(answer)

    # Action items — collapsed by default unless there are any
    actions = msg.get("key_actions", [])
    if actions:
        with st.expander(f"Action Items ({len(actions)})", expanded=False):
            for item in actions:
                st.markdown(f"- {item}")

    # Keywords — always collapsed so chat feels clean
    kws = msg.get("keywords", [])
    if kws:
        with st.expander("Keywords", expanded=False):
            tags = " ".join(
                f'<span style="background:#ede9fe;color:#6d28d9;padding:2px 8px;'
                f'border-radius:12px;font-size:12px;margin:2px;display:inline-block;">'
                f'{kw}</span>'
                for kw in kws
            )
            st.markdown(tags, unsafe_allow_html=True)

    # Source citations — always collapsed
    srcs = msg.get("sources", [])
    if srcs:
        with st.expander(f"Sumber Transkripsi ({len(srcs)} kutipan)", expanded=False):
            for src in srcs:
                spk_name = src.get("speaker", "?")
                start    = src.get("start_time", 0)
                score    = src.get("score", 0)
                text     = src.get("text", "")
                st.markdown(
                    f"**{spk_name}** @ {start:.1f}s"
                    f" _(relevansi: {score*100:.1f}%)_\n\n"
                    f"> {text}"
                )


with st.sidebar:
    st.markdown("### Pilih Meeting")
    meetings  = api_client.list_meetings()
    completed = [m for m in meetings if m.get("status") == "completed"]
    if completed:
        opts = {f"{m['filename']} ({m['meeting_id'][:8]}...)": m["meeting_id"] for m in completed}
        def_k = next((k for k, v in opts.items() if v == st.session_state.current_meeting_id), list(opts.keys())[0])
        sel_lbl = st.selectbox("Meeting", list(opts.keys()), index=list(opts.keys()).index(def_k), label_visibility="collapsed")
        selected_id = opts[sel_lbl]
    else:
        selected_id = st.text_input("Meeting ID", value=st.session_state.current_meeting_id, placeholder="Paste meeting ID here", label_visibility="collapsed")
    st.divider()
    top_k = st.slider("Chunks to retrieve (top_k)", 1, 10, 3)
    if st.button("Clear chat history", use_container_width=True):
        st.session_state.chat_messages = []
        st.rerun()

    if selected_id:
        # --- Speaker distribution chart (Section E) ---
        st.divider()
        st.markdown("### Speaker Distribution")
        _t = api_client.get_transcript(selected_id)
        if _t.get("status") not in ("processing","failed") and "segments" in _t:
            _cnt: dict = {}
            for seg in _t["segments"]:
                spk = seg.get("speaker","Unknown")
                _cnt[spk] = _cnt.get(spk, 0) + 1
            if _cnt:
                st.bar_chart(pd.DataFrame({"Segmen": _cnt.values()}, index=list(_cnt.keys())), color="#6366f1")


if selected_id != st.session_state.chat_meeting_id:
    st.session_state.chat_messages = []
    st.session_state.chat_meeting_id = selected_id
if selected_id:
    st.session_state.current_meeting_id = selected_id

st.markdown("<h1 style='font-size:2.1rem;font-weight:900;'>AI Chat</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:#64748b;font-size:1rem;margin-bottom:8px;'>Tanya jawab seputar isi meeting menggunakan RAG (FAISS + Qwen2).</p>", unsafe_allow_html=True)

if not selected_id:
    st.info("Belum ada meeting yang dipilih. Upload file di halaman 1 Upload terlebih dahulu.")
    st.stop()

st.markdown(f"<p style='color:#94a3b8;font-size:13px;'>Meeting ID: <code>{selected_id}</code></p>", unsafe_allow_html=True)
st.divider()

for msg in st.session_state.chat_messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            _render_assistant(msg)
        else:
            st.markdown(msg["content"])

if prompt := st.chat_input("Tanyakan sesuatu tentang meeting ini..."):
    st.session_state.chat_messages.append({"role":"user","content":prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    # Extract last 3 Q&A turns for conversation memory
    _msgs = st.session_state.chat_messages
    _history = []
    for _i, _m in enumerate(_msgs[:-1] if _msgs else []):
        if _m["role"] == "user" and _i + 1 < len(_msgs) and _msgs[_i + 1]["role"] == "assistant":
            _history.append({
                "question": _m["content"],
                "answer": _msgs[_i + 1].get("transcription_summary", ""),
            })
    _history = _history[-3:] or None

    with st.chat_message("assistant"):
        with st.spinner("Mencari konteks dan membuat jawaban..."):
            resp = api_client.rag_chat(selected_id, prompt, top_k=top_k, history=_history)
        if "error" in resp:
            md = {"role": "assistant", "transcription_summary": f"Error: {resp.get('error', '')}", "key_actions": [], "keywords": [], "sources": []}
        else:
            md = {"role":"assistant","transcription_summary":resp.get("transcription_summary","Tidak ada jawaban."),"key_actions":resp.get("key_actions",[]),"keywords":resp.get("keywords",[]),"sources":resp.get("sources",[])}
        _render_assistant(md)
    st.session_state.chat_messages.append(md)