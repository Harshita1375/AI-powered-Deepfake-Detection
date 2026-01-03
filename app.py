import streamlit as st

st.set_page_config(
    page_title="Deepfake Detection AI",
    page_icon="🛡️",
    layout="centered"
)

st.sidebar.title("Navigation")
st.sidebar.info("Select a detection mode above to begin analysis.")

st.title("🛡️ Deepfake Detection System")
st.markdown("""
### Combatting Digital Misinformation with AI
Welcome to the Deepfake Detection System. This tool uses a **Robust ResNet18** deep learning model trained on the FaceForensics++ dataset to identify manipulated facial content.

#### How to use this app:
1. **Choose a Mode:** Use the sidebar to navigate to **Video Detection** or **Image Detection**.
2. **Upload Media:** Upload the file you want to verify.
3. **Analyze:** The AI will scan the pixels for forgery artifacts and provide a verdict.
""")

st.subheader("Project Highlights")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(label="Model", value="ResNet18")
with col2:
    st.metric(label="Accuracy", value="95%+")
with col3:
    st.metric(label="Dataset", value="FF++ (C23)")

st.divider()
st.caption("Developed by AI Researchers | Powered by PyTorch & Streamlit")

st.warning("Note: While highly accurate, no AI is 100% perfect. Always use multiple sources for verification.")