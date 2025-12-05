import streamlit as st
import requests

st.title("🚨 SRP-Net Risk Scanner")
st.markdown("*Predict if this PR will cause sprint delays*")

# Input columns
col1, col2 = st.columns(2)

with col1:
    st.subheader("PR Details")
    title = st.text_input("Title", "Massive core refactor")
    body = st.text_area("Description", "Rewrote entire backend architecture", height=80)

with col2:
    st.subheader("Files Changed")
    files_text = st.text_area("One file per line", 
                             "backend/core.py\nbackend/config.py\napi/auth.py\nutils/helpers.py", 
                             height=200)
    files = [f.strip() for f in files_text.split('\n') if f.strip()]

if st.button("⚡ **Scan Risk Level**", type="primary"):
    payload = {
        "title": title,
        "body": body,
        "files": files
    }
    
    with st.spinner("🔍 Analyzing risk factors..."):
        try:
            response = requests.post("http://127.0.0.1:5000/predict_risk", 
                                   json=payload, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                # Risk Gauge
                st.metric("Risk Score", f"{data['risk_score']:.1%}")
                st.metric("Risk Level", data['risk_label'])
                
                # Risk factors
                col1, col2 = st.columns(2)
                with col1:
                    if data.get('is_critical', False):
                        st.error("🚨 **CRITICAL FILES DETECTED**")
                        st.write("`config`, `core`, `api`, `auth` files touched")
                    else:
                        st.success("✅ Safe file types")
                
                with col2:
                    st.info(f"Files analyzed: **{len(files)}**")
                    st.caption("Risk ↑ with more files & critical changes")
                    
            else:
                st.error("Backend API error")
                
        except Exception as e:
            st.error(f"Connection failed: {str(e)}")
            st.info("💡 Make sure `python backend/api.py` is running")
