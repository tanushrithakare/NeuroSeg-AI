import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image

from model.model import NeuroSeg

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NeuroSeg-AI | Brain Tumor Segmentation",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Root theme ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Dark background */
.stApp {
    background: #0a0d14;
    color: #e2e8f0;
}

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0a0d14; }
::-webkit-scrollbar-thumb { background: #334155; border-radius: 3px; }

/* ── Hero banner ── */
.hero-banner {
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 40%, #0f172a 100%);
    border: 1px solid rgba(99, 102, 241, 0.25);
    border-radius: 20px;
    padding: 48px 40px 40px;
    margin-bottom: 32px;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 240px; height: 240px;
    background: radial-gradient(circle, rgba(99,102,241,0.18) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-banner::after {
    content: '';
    position: absolute;
    bottom: -80px; left: -40px;
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(139,92,246,0.12) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-size: 2.6rem;
    font-weight: 800;
    background: linear-gradient(90deg, #a78bfa, #818cf8, #60a5fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 10px 0;
    line-height: 1.15;
}
.hero-sub {
    font-size: 1.05rem;
    color: #94a3b8;
    font-weight: 400;
    margin: 0;
    max-width: 600px;
}
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(99,102,241,0.15);
    border: 1px solid rgba(99,102,241,0.3);
    color: #a78bfa;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    padding: 4px 12px;
    border-radius: 20px;
    margin-bottom: 16px;
}

/* ── Glass card ── */
.glass-card {
    background: rgba(15, 23, 42, 0.7);
    border: 1px solid rgba(99, 102, 241, 0.15);
    border-radius: 16px;
    padding: 24px;
    backdrop-filter: blur(12px);
    margin-bottom: 20px;
}
.card-title {
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #64748b;
    margin-bottom: 14px;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* ── Metric pill ── */
.metric-row {
    display: flex;
    gap: 14px;
    flex-wrap: wrap;
    margin-bottom: 28px;
}
.metric-pill {
    flex: 1;
    min-width: 130px;
    background: rgba(15,23,42,0.8);
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 14px;
    padding: 16px 20px;
    text-align: center;
}
.metric-pill .val {
    font-size: 1.7rem;
    font-weight: 800;
    background: linear-gradient(90deg, #a78bfa, #60a5fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
}
.metric-pill .lbl {
    font-size: 0.72rem;
    color: #64748b;
    font-weight: 500;
    margin-top: 4px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

/* ── Result label ── */
.result-label {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #64748b;
    text-align: center;
    margin-bottom: 8px;
}

/* ── Finding card ── */
.finding-positive {
    background: linear-gradient(135deg, rgba(239,68,68,0.08), rgba(220,38,38,0.04));
    border: 1px solid rgba(239,68,68,0.25);
    border-radius: 14px;
    padding: 20px 24px;
    margin: 16px 0;
}
.finding-negative {
    background: linear-gradient(135deg, rgba(16,185,129,0.08), rgba(5,150,105,0.04));
    border: 1px solid rgba(16,185,129,0.25);
    border-radius: 14px;
    padding: 20px 24px;
    margin: 16px 0;
}
.finding-title {
    font-size: 1rem;
    font-weight: 700;
    margin-bottom: 6px;
}
.finding-positive .finding-title { color: #f87171; }
.finding-negative .finding-title { color: #34d399; }
.finding-body {
    font-size: 0.88rem;
    color: #94a3b8;
    line-height: 1.6;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #080b12 !important;
    border-right: 1px solid rgba(99,102,241,0.12) !important;
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }

.sidebar-logo {
    text-align: center;
    padding: 8px 0 20px;
    border-bottom: 1px solid rgba(99,102,241,0.15);
    margin-bottom: 20px;
}
.sidebar-logo .logo-icon {
    font-size: 2.8rem;
    display: block;
    margin-bottom: 4px;
}
.sidebar-logo .logo-name {
    font-size: 1.1rem;
    font-weight: 700;
    background: linear-gradient(90deg, #a78bfa, #60a5fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.sidebar-logo .logo-tagline {
    font-size: 0.72rem;
    color: #475569;
    letter-spacing: 0.05em;
}

.sidebar-section {
    background: rgba(99,102,241,0.06);
    border: 1px solid rgba(99,102,241,0.12);
    border-radius: 12px;
    padding: 14px 16px;
    margin-bottom: 14px;
}
.sidebar-section-title {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #64748b !important;
    margin-bottom: 10px;
}

/* ── Uploader area ── */
[data-testid="stFileUploaderDropzone"] {
    background: rgba(15,23,42,0.6) !important;
    border: 2px dashed rgba(99,102,241,0.3) !important;
    border-radius: 16px !important;
    padding: 32px !important;
}
[data-testid="stFileUploaderDropzone"]:hover {
    border-color: rgba(99,102,241,0.6) !important;
    background: rgba(99,102,241,0.05) !important;
}

/* ── Slider ── */
[data-testid="stSlider"] > div > div > div > div {
    background: linear-gradient(90deg, #7c3aed, #4f46e5) !important;
}

/* ── Chat ── */
[data-testid="stBottom"] {
    background: transparent !important;
}
[data-testid="stChatMessage"] {
    background: rgba(15,23,42,0.7) !important;
    border: 1px solid rgba(99,102,241,0.15) !important;
    border-radius: 16px !important;
    padding: 16px !important;
    margin-bottom: 12px !important;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06) !important;
}
[data-testid="stChatInput"] {
    max-width: 800px !important;
    margin: 0 auto !important;
    padding-bottom: 24px !important;
}
[data-testid="stChatInput"] > div {
    background: rgba(15,23,42,0.95) !important;
    border: 1px solid rgba(99,102,241,0.4) !important;
    border-radius: 24px !important;
    padding: 2px 16px !important;
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3) !important;
}

/* ── Divider ── */
hr { border-color: rgba(99,102,241,0.12) !important; }

/* ── Success / Warning / Error ── */
[data-testid="stAlert"] {
    border-radius: 12px !important;
    border: none !important;
}
</style>
""", unsafe_allow_html=True)


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <span class="logo-icon">🧠</span>
        <div class="logo-name">NeuroSeg-AI</div>
        <div class="logo-tagline">BRAIN TUMOR SEGMENTATION</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section-title">⚙️ &nbsp;Inference Controls</div>', unsafe_allow_html=True)
    threshold = st.slider(
        "Segmentation Threshold",
        min_value=0.1, max_value=0.9, value=0.5, step=0.05,
        help="Probability cutoff for classifying a pixel as tumor. Lower = more sensitive, Higher = more specific."
    )
    alpha = st.slider(
        "Overlay Opacity",
        min_value=0.1, max_value=0.9, value=0.5, step=0.05,
        help="Transparency of the red tumor mask overlaid on the MRI."
    )

    st.markdown("---")

    st.markdown('<div class="sidebar-section-title">🏗️ &nbsp;Architecture</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sidebar-section">
        <div style="font-size:0.8rem; color:#94a3b8; line-height:1.8;">
            <div>🔵 <b>Encoder</b> — ResNet-50</div>
            <div>🟣 <b>Context</b> — ASPP (6×,12×,18×)</div>
            <div>🔴 <b>Attention</b> — CBAM</div>
            <div>🟢 <b>Decoder</b> — Skip-connection</div>
            <div>⚪ <b>Loss</b> — BCE + Dice</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.72rem; color:#334155; text-align:center; line-height:1.8;">
        <div>Dataset: LGG MRI Segmentation</div>
        <div>Input: 256×256 Grayscale</div>
        <div style="margin-top:6px; color:#4c1d95;">⚠️ For research use only</div>
    </div>
    """, unsafe_allow_html=True)


# ── Hero Banner ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
    <div class="hero-badge">🔬 Deep Learning · Medical Imaging</div>
    <h1 class="hero-title">Brain Tumor Segmentation</h1>
    <p class="hero-sub">
        Upload an MRI scan to instantly detect and segment tumor regions using an
        attention-based deep learning model trained on 3,929 clinical MRI scans.
    </p>
</div>
""", unsafe_allow_html=True)


# ── Model loading ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = NeuroSeg().to(device)
    model.load_state_dict(torch.load("checkpoints/neuroseg_best.pth", map_location=device, weights_only=True))
    model.eval()
    return model, device

model, device = None, "cpu"
model_ready = False

with st.spinner("Loading model checkpoint..."):
    try:
        model, device = load_model()
        model_ready = True
    except Exception as e:
        print(f"[NeuroSeg] Could not load model checkpoint: {e}")
        model_ready = False

device_label = ("GPU (CUDA)" if device == "cuda" else "CPU") if model_ready else "—"
status_val = "Ready" if model_ready else "Training..."
status_color = "#34d399" if model_ready else "#f59e0b"

st.markdown(f"""
<div class="metric-row">
    <div class="metric-pill"><div class="val">3,929</div><div class="lbl">Training Scans</div></div>
    <div class="metric-pill"><div class="val">50</div><div class="lbl">Epochs</div></div>
    <div class="metric-pill"><div class="val">BCE+Dice</div><div class="lbl">Loss Function</div></div>
    <div class="metric-pill">
        <div class="val" style="background: linear-gradient(90deg,{status_color},{status_color}); -webkit-background-clip:text; -webkit-text-fill-color:transparent;">{status_val}</div>
        <div class="lbl">Model Status</div>
    </div>
</div>
""", unsafe_allow_html=True)

if not model_ready:
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, rgba(245,158,11,0.08), rgba(217,119,6,0.04));
        border: 1px solid rgba(245,158,11,0.3);
        border-radius: 14px;
        padding: 18px 24px;
        margin-bottom: 24px;
        display: flex;
        align-items: center;
        gap: 14px;
    ">
        <div style="font-size:1.8rem;">⏳</div>
        <div>
            <div style="font-size:0.95rem; font-weight:700; color:#fbbf24; margin-bottom:4px;">Model Training In Progress</div>
            <div style="font-size:0.83rem; color:#92400e;">
                The checkpoint <code>checkpoints/neuroseg_best.pth</code> is being generated.
                Upload is disabled until training completes. Refresh the page once done.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── Upload section ─────────────────────────────────────────────────────────────
st.markdown('<div class="card-title">📂 &nbsp;Upload MRI Scan</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "Drop an MRI image here",
    type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"],
    label_visibility="collapsed",
    disabled=not model_ready,
)

if uploaded_file is not None:
    # ── Decode image ──
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    if image is None:
        st.error("Could not decode the uploaded file. Please try a valid MRI image.")
    else:
        with st.spinner("Analysing scan..."):
            original_h, original_w = image.shape

            # Preprocess
            image_resized = cv2.resize(image, (256, 256))
            image_norm = image_resized / 255.0
            input_tensor = torch.tensor(image_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

            # Inference with Test-Time Augmentation (TTA)
            with torch.no_grad():
                # 1. Original
                p1 = torch.sigmoid(model(input_tensor))
                
                # 2. Horizontal Flip
                input_hf = torch.flip(input_tensor, dims=[3])
                p2 = torch.flip(torch.sigmoid(model(input_hf)), dims=[3])
                
                # 3. Vertical Flip
                input_vf = torch.flip(input_tensor, dims=[2])
                p3 = torch.flip(torch.sigmoid(model(input_vf)), dims=[2])
                
                # Average consensus
                pred = ((p1 + p2 + p3) / 3.0).cpu().numpy()[0, 0]

            # Mask
            mask = (pred > threshold).astype(np.uint8)
            mask_resized = cv2.resize(mask, (original_w, original_h), interpolation=cv2.INTER_NEAREST)

            # Overlay
            img_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            mask_overlay = np.zeros_like(img_rgb)
            mask_overlay[:, :, 0] = mask_resized * 255
            overlay = cv2.addWeighted(img_rgb, 1, mask_overlay, alpha, 0)

            # Colorised mask
            mask_rgb = np.zeros((original_h, original_w, 3), dtype=np.uint8)
            mask_rgb[:, :, 0] = mask_resized * 255

            # Stats
            brain_pixels = max(int(np.sum(image > 20)), 1)
            tumor_pixels = int(np.sum(mask_resized))
            coverage_pct = (tumor_pixels / brain_pixels) * 100

        # ── Results header ──
        st.markdown("---")
        st.markdown('<div class="card-title">🎯 &nbsp;Segmentation Results</div>', unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3, gap="medium")

        with col1:
            st.markdown('<div class="result-label">🖼️ Original MRI</div>', unsafe_allow_html=True)
            st.image(image, use_container_width=True, channels="GRAY")

        with col2:
            st.markdown('<div class="result-label">🎯 Predicted Mask</div>', unsafe_allow_html=True)
            st.image(mask_rgb, use_container_width=True, channels="RGB")

        with col3:
            st.markdown('<div class="result-label">🔍 Overlay</div>', unsafe_allow_html=True)
            st.image(overlay, use_container_width=True, channels="RGB")

        # ── Scan statistics ──
        st.markdown("---")
        st.markdown('<div class="card-title">📊 &nbsp;Scan Statistics</div>', unsafe_allow_html=True)

        s1, s2, s3, s4 = st.columns(4, gap="medium")
        with s1:
            st.markdown(f"""<div class="metric-pill">
                <div class="val">{coverage_pct:.1f}%</div>
                <div class="lbl">Tumor Coverage</div>
            </div>""", unsafe_allow_html=True)
        with s2:
            st.markdown(f"""<div class="metric-pill">
                <div class="val">{tumor_pixels:,}</div>
                <div class="lbl">Tumor Pixels</div>
            </div>""", unsafe_allow_html=True)
        with s3:
            st.markdown(f"""<div class="metric-pill">
                <div class="val">{original_w}×{original_h}</div>
                <div class="lbl">Image Resolution</div>
            </div>""", unsafe_allow_html=True)
        with s4:
            st.markdown(f"""<div class="metric-pill">
                <div class="val">{threshold:.2f}</div>
                <div class="lbl">Threshold Used</div>
            </div>""", unsafe_allow_html=True)

        # ── AI Diagnostic Summary ──
        st.markdown("---")
        st.markdown('<div class="card-title">🤖 &nbsp;AI Diagnostic Summary</div>', unsafe_allow_html=True)

        if tumor_pixels > 50:
            severity = "High" if coverage_pct > 10 else "Moderate" if coverage_pct > 3 else "Low"
            st.markdown(f"""
            <div class="finding-positive">
                <div class="finding-title">⚠️ Potential Anomalous Region Detected</div>
                <div class="finding-body">
                    The model has identified a high-confidence region of structural abnormality covering
                    approximately <b>{coverage_pct:.2f}%</b> of visible brain mass ({tumor_pixels:,} pixels).
                    Estimated severity signal: <b>{severity}</b>.<br><br>
                    The highlighted region (red overlay) indicates areas where pixel-level attention scores
                    exceeded the classification threshold of <b>{threshold}</b>.
                    Clinical correlation and radiological review is <b>strongly advised</b> before drawing any conclusions.
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="finding-negative">
                <div class="finding-title">✅ No Significant Anomaly Detected</div>
                <div class="finding-body">
                    The model did not identify any high-confidence tumor regions at the current threshold
                    of <b>{threshold}</b>. Fewer than 50 pixels were classified as anomalous.<br><br>
                    Consider lowering the segmentation threshold in the sidebar if you suspect a subtle lesion,
                    or consult a radiologist for a comprehensive assessment.
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.caption("⚠️ This tool is intended for research purposes only and does not constitute medical advice.")

        # ── AI Chat Assistant ──
        st.markdown("---")
        st.markdown('<div class="card-title">💬 &nbsp;Ask the AI Assistant</div>', unsafe_allow_html=True)
        st.markdown(
            '<div style="font-size:0.85rem; color:#64748b; margin-bottom:14px;">Have questions about this scan or the model? Chat with the diagnostic assistant.</div>',
            unsafe_allow_html=True
        )

        if "messages" not in st.session_state:
            st.session_state.messages = []

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if prompt := st.chat_input("Ask about the scan, the model, or tumor segmentation..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Context-aware mock responses
            p_lower = prompt.lower()
            if any(w in p_lower for w in ["tumor", "cancer", "anomaly", "region"]):
                response = (
                    f"Based on this scan, the model detected {'an anomalous region covering ' + f'{coverage_pct:.2f}% of brain mass' if tumor_pixels > 50 else 'no significant tumor region'}. "
                    "The segmentation uses a ResNet-50 + CBAM attention mechanism trained on the LGG MRI dataset. "
                    "Please consult a certified radiologist for clinical interpretation."
                )
            elif any(w in p_lower for w in ["model", "architecture", "how", "work", "train"]):
                response = (
                    "NeuroSeg-AI uses a ResNet-50 encoder to extract multi-scale features, "
                    "ASPP (Atrous Spatial Pyramid Pooling) to capture context at multiple scales, "
                    "and CBAM (Convolutional Block Attention Module) for both channel and spatial attention. "
                    "It was trained with BCE + Dice loss on 3,929 LGG MRI scans."
                )
            elif any(w in p_lower for w in ["threshold", "confidence", "sensitivity"]):
                response = (
                    f"The current threshold is set to {threshold}. Lowering it increases sensitivity "
                    "(catches more potential tumor regions but may include false positives). "
                    "Raising it increases specificity (fewer false positives, but may miss subtle lesions). "
                    "Adjust the slider in the sidebar to explore."
                )
            else:
                response = (
                    f"Thank you for your question about '{prompt}'. "
                    "In a production system, this assistant would connect to a clinical LLM fine-tuned on radiology reports. "
                    "For now, I can answer questions about tumor detection, model architecture, and threshold settings."
                )

            st.session_state.messages.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.markdown(response)

else:
    # ── Empty state ──
    st.markdown("""
    <div style="
        text-align: center;
        padding: 60px 40px;
        background: rgba(15,23,42,0.5);
        border: 2px dashed rgba(99,102,241,0.2);
        border-radius: 20px;
        margin-top: 10px;
    ">
        <div style="font-size: 3.5rem; margin-bottom: 16px;">🧠</div>
        <div style="font-size: 1.15rem; font-weight: 600; color: #e2e8f0; margin-bottom: 8px;">
            No Scan Uploaded Yet
        </div>
        <div style="font-size: 0.9rem; color: #475569; max-width: 400px; margin: 0 auto; line-height: 1.6;">
            Upload a brain MRI image using the file picker above to begin AI-powered tumor segmentation.
            Supported formats: PNG, JPG, TIFF, BMP.
        </div>
    </div>
    """, unsafe_allow_html=True)
