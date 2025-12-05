import streamlit as st
import requests
import json

st.title("🔍 Smart PR-to-Story Mapper")
st.markdown("*Find the perfect user story for your PR*")

# Input Form
title = st.text_input("**PR Title**", "Fix login authentication bug")
body = st.text_area("**PR Description**", "Updated OAuth token validation logic in auth controller")

if st.button("🔮 **Find Matching Stories**", type="primary"):
    with st.spinner("🤖 AI analyzing semantic similarity..."):
        try:
            payload = {"title": title, "body": body}
            response = requests.post("http://127.0.0.1:5000/predict_story", 
                                   json=payload, timeout=10)
            
            if response.status_code == 200:
                results = response.json().get('matches', [])
                
                if results:
                    st.success(f"✅ Found {len(results)} matching stories!")
                    
                    for i, match in enumerate(results[:5]):
                        with st.expander(f"#{i+1} • {match['title'][:60]}... (Score: {match['score']:.3f})"):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Issue ID", f"#{match['issue_id']}")
                                st.metric("Confidence", f"{match['score']:.1%}")
                            with col2:
                                st.metric("Semantic Match", f"{match['cosine']:.1%}")
                            
                            if i == 0:
                                st.success("🏆 **TOP RECOMMENDATION**")
                else:
                    st.warning("No strong matches found.")
            else:
                st.error("Backend error. Check if API is running.")
                
        except requests.exceptions.RequestException:
            st.error("❌ Cannot connect to Backend API (port 5000)")
        except Exception as e:
            st.error(f"Error: {str(e)}")
