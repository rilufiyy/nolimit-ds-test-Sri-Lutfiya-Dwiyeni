"""Shared CSS injected into every MeetRecall AI page."""
from __future__ import annotations

import streamlit as st

_CSS = """
<style>
/* Page background */
[data-testid="stAppViewContainer"] > .main {
    background-color: #f8fafc;
}

/* Sidebar dark theme */
[data-testid="stSidebar"] {
    background: linear-gradient(175deg, #1e1b4b 0%, #312e81 100%) !important;
}
[data-testid="stSidebar"] * {
    color: rgba(255,255,255,0.9) !important;
}
[data-testid="stSidebar"] .stTextInput input {
    background: rgba(255,255,255,0.1) !important;
    border: 1px solid rgba(255,255,255,0.25) !important;
    color: white !important;
    border-radius: 8px !important;
}
[data-testid="stSidebar"] .stSelectbox > div > div {
    background: rgba(255,255,255,0.1) !important;
    border: 1px solid rgba(255,255,255,0.25) !important;
    border-radius: 8px !important;
}
[data-testid="stSidebar"] .stButton > button {
    background: rgba(255,255,255,0.12) !important;
    border: 1px solid rgba(255,255,255,0.28) !important;
    color: white !important;
    border-radius: 8px !important;
    width: 100%;
    font-weight: 600 !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(255,255,255,0.22) !important;
}

/* Metric cards */
[data-testid="stMetric"] {
    background: white;
    border-radius: 12px;
    padding: 16px 20px !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    border: 1px solid #e2e8f0;
}
[data-testid="stMetricValue"] {
    font-size: 1.7rem !important;
    font-weight: 800 !important;
    color: #1e293b !important;
}
[data-testid="stMetricLabel"] {
    font-size: 0.77rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.6px !important;
    text-transform: uppercase !important;
    color: #64748b !important;
}

/* Primary button */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    border: none !important;
    color: white !important;
    font-weight: 700 !important;
    border-radius: 10px !important;
    padding: 10px 28px !important;
    font-size: 15px !important;
    transition: transform 0.15s, box-shadow 0.15s !important;
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(99,102,241,0.4) !important;
}

/* Step header */
.step-row {
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 20px 0 10px 0;
}
.step-num {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    min-width: 34px;
    height: 34px;
    border-radius: 50%;
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: white;
    font-weight: 800;
    font-size: 16px;
    flex-shrink: 0;
}
.step-title {
    font-size: 18px;
    font-weight: 700;
    color: #1e293b;
}

/* Info / success / warning boxes */
.info-box {
    background: linear-gradient(135deg, #eef2ff, #e0e7ff);
    border: 1px solid #c7d2fe;
    border-radius: 12px;
    padding: 14px 18px;
    color: #3730a3;
    font-size: 14px;
    line-height: 1.6;
    margin: 10px 0;
}
.success-box {
    background: linear-gradient(135deg, #f0fdf4, #dcfce7);
    border: 1px solid #86efac;
    border-radius: 12px;
    padding: 14px 18px;
    color: #166534;
    font-size: 14px;
    line-height: 1.6;
    margin: 10px 0;
}
.warn-box {
    background: linear-gradient(135deg, #fffbeb, #fef3c7);
    border: 1px solid #fcd34d;
    border-radius: 12px;
    padding: 14px 18px;
    color: #92400e;
    font-size: 14px;
    line-height: 1.6;
    margin: 10px 0;
}

/* Speaker segments */
.seg-block {
    padding: 12px 16px;
    border-radius: 10px;
    margin-bottom: 8px;
    border-left: 4px solid;
}
.seg-block.spk-0 { background:#eff6ff; border-color:#3b82f6; }
.seg-block.spk-1 { background:#fdf4ff; border-color:#a855f7; }
.seg-block.spk-2 { background:#f0fdf4; border-color:#22c55e; }
.seg-block.spk-3 { background:#fff7ed; border-color:#f97316; }
.seg-block.spk-4 { background:#fdf2f8; border-color:#ec4899; }
.seg-block.spk-5 { background:#f0fdfa; border-color:#14b8a6; }
.spk-name { font-size:12px; font-weight:700; margin-bottom:3px; }
.spk-name.spk-0 { color:#1d4ed8; }
.spk-name.spk-1 { color:#7c3aed; }
.spk-name.spk-2 { color:#15803d; }
.spk-name.spk-3 { color:#c2410c; }
.spk-name.spk-4 { color:#be185d; }
.spk-name.spk-5 { color:#0f766e; }
.seg-ts   { font-size:11px; color:#94a3b8; margin-left:6px; }
.seg-text { font-size:14px; color:#334155; line-height:1.65; margin:0; }

/* Summary card */
.summary-card {
    background: white;
    border-radius: 16px;
    padding: 24px 28px;
    box-shadow: 0 4px 20px rgba(99,102,241,0.08);
    border: 1px solid #e2e8f0;
    margin-bottom: 20px;
}
.summary-item {
    display: flex;
    gap: 12px;
    padding: 10px 0;
    border-bottom: 1px solid #f1f5f9;
    font-size: 14px;
    color: #334155;
}
.summary-item:last-child { border-bottom: none; }
.summary-key {
    font-weight: 700;
    color: #6366f1;
    min-width: 130px;
    flex-shrink: 0;
}

/* Feature cards (landing) */
.feat-card {
    background: white;
    border-radius: 18px;
    padding: 28px 22px;
    text-align: center;
    box-shadow: 0 4px 20px rgba(0,0,0,0.07);
    border: 1px solid #e2e8f0;
    height: 100%;
}
.feat-title { font-size:17px; font-weight:700; color:#1e293b; margin-bottom:8px; }
.feat-desc  { font-size:13px; color:#64748b; line-height:1.65; }

/* Topic cards */
.topic-card {
    background: white;
    border-radius: 12px;
    padding: 18px 22px;
    margin: 10px 0;
    border-left: 5px solid;
    box-shadow: 0 2px 10px rgba(0,0,0,0.06);
}
.tc-0{border-color:#6366f1;} .tc-1{border-color:#ec4899;}
.tc-2{border-color:#22c55e;} .tc-3{border-color:#f97316;}
.tc-4{border-color:#a855f7;} .tc-5{border-color:#14b8a6;}
.tc-6{border-color:#eab308;} .tc-7{border-color:#ef4444;}

/* Processing box */
.proc-box {
    text-align: center;
    padding: 36px 24px;
    background: white;
    border-radius: 16px;
    border: 1px solid #fcd34d;
    box-shadow: 0 4px 16px rgba(245,158,11,0.12);
}

/* Divider */
hr { border-color: #e2e8f0 !important; margin: 24px 0 !important; }
</style>
"""


def inject() -> None:
    """Inject shared CSS into the current Streamlit page."""
    st.markdown(_CSS, unsafe_allow_html=True)
