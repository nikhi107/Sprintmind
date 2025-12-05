import streamlit as st
import requests
import pandas as pd

st.set_page_config(
    page_title="🚀 SprintMind-X",
    page_icon="🚀",
    layout="wide"
)

st.title("🚀 SprintMind-X")
st.markdown("### AI-Powered PR Story Mapping & Risk Prediction")

# Sidebar - Backend Status
st.sidebar.title("🔌 System Status")
try:
    response = requests.get("http://127.0.0.1:5000/status", timeout=5)
    if response.status_code == 200:
        st.sidebar.success("✅ Backend Online")
    else:
        st.sidebar.error("❌ Backend Offline")
except:
    st.sidebar.error("❌ Cannot connect to Backend (port 5000)")

# Main Dashboard
col1, col2 = st.columns([1, 2])

with col1:
    st.header("📊 Quick Stats")
    st.metric("Ranking Accuracy", "77.5%")
    st.metric("Risk Detection", "80.0%")
    st.info("👈 Try Smart Mapper or Risk Scanner")

with col2:
    st.header("🎯 What is SprintMind-X?")
    st.write("""
    - **🔍 Smart Mapper**: Automatically links PRs to User Stories
    - **🚨 Risk Scanner**: Predicts if PRs will cause sprint delays  
    - **Hybrid AI**: Combines BERT embeddings + XGBoost ranking
    """)

st.divider()
st.markdown("### 📱 Try the Modules 👇")
