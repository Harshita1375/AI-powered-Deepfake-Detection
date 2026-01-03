import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
import os
import sys
import gdown

# --- 1. Setup Paths & Environment Detection ---
# Check if running on Streamlit Cloud
IS_CLOUD = os.getenv("STREAMLIT_RUNTIME_ENV") == "cloud" or os.path.exists("/app")

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.image_utils import get_resnet_model

# GDrive Configuration (Update with your ID)
FILE_ID = "1318HpyTgCkq4GD7U5KQ2S4GYYUum6UsY"
MODEL_PATH = "models/image_model.pth"

# --- 2. Smart Model Loader ---
@st.cache_resource
def load_detector():
    # Ensure directory exists
    if not os.path.exists("models"):
        os.makedirs("models")

    # If on Cloud and model is missing, download from GDrive
    if IS_CLOUD and not os.path.exists(MODEL_PATH):
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        with st.spinner("☁️ Cloud detected: Downloading model from Google Drive..."):
            try:
                gdown.download(url, MODEL_PATH, quiet=False)
            except Exception as e:
                st.error(f"Failed to download model from GDrive: {e}")
                return None
    
    # Final check: If still missing (local users who didn't train), show error
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Model file not found at {MODEL_PATH}. Please run train_image.py first.")
        return None
    
    # Load model using your existing utility
    try:
        model = get_resnet_model(checkpoint_path=MODEL_PATH)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model weights: {e}")
        return None

# Page Setup
st.set_page_config(page_title="Image Deepfake Detection", page_icon="🖼️")
st.title("🖼️ Image Forgery Detection")

detector = load_detector()

# --- 3. Rest of your UI Logic ---
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and detector is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    if st.button("Analyze Image"):
        with st.spinner("AI is analyzing pixels..."):
            input_tensor = preprocess(image).unsqueeze(0)
            with torch.no_grad():
                output = detector(input_tensor)
                confidence = output.item()
            
            is_fake = confidence > 0.5
            label = "DEEPFAKE" if is_fake else "REAL"
            color = "red" if is_fake else "green"
            
            st.subheader(f"Verdict: :{color}[{label}]")
            score = confidence if is_fake else (1 - confidence)
            st.progress(score)
            st.write(f"Confidence: **{score * 100:.2f}%**")