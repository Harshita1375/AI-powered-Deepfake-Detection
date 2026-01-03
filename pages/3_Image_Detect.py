import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
import os
import sys

# Add the project root to sys.path so we can import from 'src'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.image_utils import get_resnet_model

# Page Configuration
st.set_page_config(page_title="Image Deepfake Detection", page_icon="🖼️")

st.title("🖼️ Image Forgery Detection")
st.write("Upload an image to check if it is a Real photograph or a Deepfake.")

# --- 1. Load the Trained Model ---
MODEL_PATH = "models/image_model.pth"

@st.cache_resource
def load_detector():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found at {MODEL_PATH}. Please run train_image.py first.")
        return None
    
    model = get_resnet_model(checkpoint_path=MODEL_PATH)
    model.eval()
    return model

detector = load_detector()

# --- 2. Image Preprocessing (Must match Training transforms) ---
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- 3. UI for File Upload ---
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and detector is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    if st.button("Analyze Image"):
        with st.spinner("AI is analyzing pixels..."):
            # Prepare image for model
            input_tensor = preprocess(image).unsqueeze(0)
            
            # Prediction
            with torch.no_grad():
                output = detector(input_tensor)
                confidence = output.item()
            
            # Determine Result
            # Note: In our training, 0=Real, 1=Fake based on Folder Alphabetical order
            is_fake = confidence > 0.5
            label = "DEEPFAKE" if is_fake else "REAL"
            color = "red" if is_fake else "green"
            
            # Display Result
            st.subheader(f"Verdict: :{color}[{label}]")
            
            # Show confidence percentage
            score = confidence if is_fake else (1 - confidence)
            st.progress(score)
            st.write(f"Confidence: **{score * 100:.2f}%**")

            # Feedback context
            if is_fake:
                st.warning("Warning: This image shows patterns consistent with AI manipulation.")
            else:
                st.success("This image appears to be an authentic photograph.")