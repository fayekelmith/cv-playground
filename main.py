import streamlit as st
from src.components.sidebar import sidebar
from src.components.details import details
from src.components.dashboard import dashboard
st.set_page_config(page_title="Computer Vision Playground", page_icon="🔭", layout="wide")

sidebar()

dashboard()

details()

def local_css(file_path:str):
    with open(file_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
local_css("style.css")
