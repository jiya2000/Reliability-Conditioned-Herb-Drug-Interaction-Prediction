"""Launch the presentation-ready research screen directly."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
from app.screening_demo import render_screening_demo

st.set_page_config(page_title='Evidence-Linked Herb–Drug Research System', layout='wide')
render_screening_demo(ROOT)
