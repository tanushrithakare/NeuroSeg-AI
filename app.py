import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image

from model.model import NeuroSeg

# Streamlit App Configuration
st.set_page_config(
    page_title="NeuroSeg-AI Dashboard", 
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a better look
st.markdown("""
    <style>
    .title-text {
        text-align: center;
        color: #2c3e50;
        margin-bottom: 20px;
    }
    .subtitle-text {
        text-align: center;
        font-size: 1.1em;
        color: #555;
        margin-bottom: 30px;
    }
    </style>
""", unsafe_allow_html=True)

# Main Title
st.markdown("<h1 class='title-text'>🧠 NeuroSeg-AI: Brain Tumor Segmentation</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle-text'>Upload an MRI image to automatically detect and segment brain tumors using our advanced deep learning model.</p>", unsafe_allow_html=True)
st.divider()

# Sidebar for controls and info
with st.sidebar:
    st.header("⚙️ Settings")
    threshold = st.slider("Segmentation Threshold", min_value=0.1, max_value=0.9, value=0.5, step=0.05,
                          help="Adjust the confidence threshold for the tumor mask. Lower values might pick up more noise, while higher values are stricter.")
    alpha = st.slider("Overlay Transparency", min_value=0.1, max_value=0.9, value=0.5, step=0.05,
                      help="Adjust how transparent the tumor mask appears over the MRI.")
    
    st.divider()
    st.subheader("ℹ️ About")
    st.info("This application uses a PyTorch-based attention model (NeuroSeg) trained on MRI scans to predict tumor regions.")

# Cache the model loading to avoid reloading on every run
@st.cache_resource(show_spinner=False)
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = NeuroSeg().to(device)
    # Load from the checkpoints folder as neuroseg_best.pth is saved there
    model.load_state_dict(torch.load("checkpoints/neuroseg_best.pth", map_location=device))
    model.eval()
    return model, device

with st.spinner("Loading Model Checkpoint..."):
    try:
        model, device = load_model()
    except Exception as e:
        st.error(f"Failed to load model from 'checkpoints/neuroseg_best.pth'. Ensure the model is trained. Error: {e}")
        st.stop()

# File uploader
uploaded_file = st.file_uploader("📂 Choose an MRI image (PNG, JPG, TIFF)...", type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"])

if uploaded_file is not None:
    # Read the image from bytes
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        st.error("❌ Failed to decode image. Please try a valid image file.")
    else:
        with st.spinner("🧠 Analyzing image..."):
            # Preprocess
            original_size = image.shape
            image_resized = cv2.resize(image, (256, 256))
            image_norm = image_resized / 255.0
            
            # Convert to tensor
            input_tensor = torch.tensor(image_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            input_tensor = input_tensor.to(device)
            
            # Predict
            with torch.no_grad():
                output = model(input_tensor)
                pred = torch.sigmoid(output).cpu().numpy()[0, 0]
                
            # Threshold using sidebar value
            mask = (pred > threshold).astype(np.uint8)
            
            # Resize mask back to original size for display
            mask_resized = cv2.resize(mask, (original_size[1], original_size[0]), interpolation=cv2.INTER_NEAREST)
            
        # Visualization
        st.success("✅ Analysis Complete!")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 🖼️ Original MRI")
            st.image(image, use_container_width=True, channels="GRAY")
            
        with col2:
            st.markdown("#### 🎯 Predicted Mask")
            # Color the mask in red for better visibility
            mask_rgb = np.zeros((original_size[0], original_size[1], 3), dtype=np.uint8)
            mask_rgb[:, :, 0] = mask_resized * 255
            st.image(mask_rgb, use_container_width=True, channels="RGB")
            
        with col3:
            st.markdown("#### 🔍 Overlay Result")
            
            # Create overlay
            # Convert grayscale image to RGB
            img_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
            # Create a red mask overlay
            mask_overlay = np.zeros_like(img_rgb)
            mask_overlay[:, :, 0] = mask_resized * 255 # Red channel
            
            # Blend using sidebar alpha
            overlay = cv2.addWeighted(img_rgb, 1, mask_overlay, alpha, 0)
            
            st.image(overlay, use_container_width=True, channels="RGB")
            
        # AI Summary Report
        st.divider()
        st.markdown("### 🤖 AI Diagnostic Summary")
        
        # Simple heuristics for summary
        brain_pixels = np.sum(image > 20)  # rough estimate of brain tissue
        tumor_pixels = np.sum(mask_resized)
        
        if tumor_pixels > 50:
            percentage = (tumor_pixels / (brain_pixels + 1e-5)) * 100
            st.warning(f"**Findings:** A potential anomalous region was detected, covering approximately **{percentage:.2f}%** of the visible brain mass.")
            st.write("The model has localized a high-confidence region indicating potential structural abnormalities (highlighted in red). Clinical correlation is strongly advised.")
        else:
            st.success("**Findings:** No significant anomalous regions detected.")
            st.write("The model did not find any high-confidence tumor regions based on the current threshold settings.")
            
        # Interactive Chatbot section
        st.divider()
        st.markdown("### 💬 Ask the AI Assistant")
        st.write("Have questions about this scan? Chat with our diagnostic assistant.")
        
        if "messages" not in st.session_state:
            st.session_state.messages = []
            
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                
        if prompt := st.chat_input("Ask a question about the analysis..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
                
            # Mock AI response
            bot_response = f"This is an automated response to: '{prompt}'. In a full production system, I would connect to an LLM to analyze the MRI features and provide a detailed radiological assessment."
            st.session_state.messages.append({"role": "assistant", "content": bot_response})
            with st.chat_message("assistant"):
                st.markdown(bot_response)
else:
    # Show a placeholder when no image is uploaded
    st.info("👆 Please upload an MRI image to begin the analysis.")
