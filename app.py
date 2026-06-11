import os
import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
from huggingface_hub import hf_hub_download

st.set_page_config(
    page_title="Deepfake Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .verdict-real {
        background: #1a3a1a; border: 2px solid #2ecc71;
        border-radius: 10px; padding: 16px; text-align: center;
    }
    .verdict-fake {
        background: #3a1a1a; border: 2px solid #e74c3c;
        border-radius: 10px; padding: 16px; text-align: center;
    }
    .section-header {
        font-size: 1.1rem; font-weight: 600;
        color: #a0aec0; margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)

IMG_SIZE        = (299, 299)
CONFIDENCE_LOW  = 0.40
CONFIDENCE_HIGH = 0.60
HF_REPO_ID      = "shimu-i/deepfake-detector-model"
MODEL_FILENAME  = "ffpp_adapted_final.keras"

@st.cache_resource(show_spinner=False)
def load_model():
    with st.spinner("Loading model..."):
        try:
            path = MODEL_FILENAME
        except Exception:
            path = MODEL_FILENAME
            if not os.path.exists(path):
                st.error("Model not found. Place ffpp_adapted_final.keras in the project folder.")
                st.stop()
        return tf.keras.models.load_model(path)

def preprocess_rgb(img_array):
    return cv2.resize(img_array, IMG_SIZE).astype(np.float32) / 255.0

def preprocess_dct(img_array):
    gray = cv2.cvtColor(img_array.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, IMG_SIZE).astype(np.float32)
    dct  = cv2.dct(gray)
    dct  = np.log(np.abs(dct) + 1)
    dct  = cv2.normalize(dct, None, 0, 1, cv2.NORM_MINMAX)
    return np.stack([dct] * 3, axis=-1)

def make_gradcam(model, rgb_input, dct_input, last_conv_layer="block14_sepconv2_act"):
    try:
        spatial_model = model.get_layer('spatial')
        conv_layer    = spatial_model.get_layer(last_conv_layer)

        # Use spatial sub-model standalone — build grad model within its own graph
        grad_model = tf.keras.Model(
            inputs  = spatial_model.inputs,
            outputs = [conv_layer.output, spatial_model.output]
        )

        rgb_t = tf.cast(rgb_input[np.newaxis], tf.float32)

        with tf.GradientTape() as tape:
            conv_out, spatial_out = grad_model(rgb_t)
            # Use mean of spatial output as proxy loss
            loss = tf.reduce_mean(spatial_out)

        grads  = tape.gradient(loss, conv_out)
        pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
        cam    = tf.reduce_sum(tf.multiply(pooled, conv_out[0]), axis=-1).numpy()
        cam    = np.maximum(cam, 0)
        cam    = cv2.resize(cam, IMG_SIZE)
        cam    = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam
    except Exception as e:
        print("Grad-CAM error:", e)
        return None

def overlay_heatmap(original_rgb, heatmap, alpha=0.45):
    coloured = (cm.jet(heatmap)[:, :, :3] * 255).astype(np.uint8)
    base     = cv2.resize(original_rgb.astype(np.uint8), IMG_SIZE)
    return cv2.addWeighted(base, 1 - alpha, coloured, alpha, 0)

def visualise_dct(dct_map):
    vis = (dct_map[:, :, 0] * 255).astype(np.uint8)
    vis = cv2.applyColorMap(vis, cv2.COLORMAP_INFERNO)
    return cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)

with st.sidebar:
    st.title("🔍 Deepfake Detector")
    st.caption("Phase 17 — Final Deployment")
    st.divider()
    st.markdown("**Model**")
    st.info("Dual-Stream Xception\nFF++ C23 adapted\nArtiFact zero-shot tested")
    st.divider()
    st.markdown("**Training pipeline**")
    st.markdown("""
- CIFAKE (diffusion objects)
- FaceForensics++ C23 (6 manipulation types)
- Zero-shot: ArtiFact (25 generators)
    """)
    st.divider()
    threshold = st.slider("Decision boundary", 0.30, 0.70, 0.50, 0.01,
        help="Probability above this → FAKE.")
    st.divider()
    st.caption("Beyond Binary Detection\nXAI Deepfake Localization via\nFrequency-Aware Segmentation")

st.title("Beyond Binary Detection")
st.markdown("Upload an image to detect whether it is AI-generated and see **where** the model found evidence.")

uploaded = st.file_uploader("Drop an image here (JPG, PNG, WEBP)", type=["jpg", "jpeg", "png", "webp"])

if uploaded is None:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        #### 🧠 Dual-Stream Analysis
        Processes every image through RGB spatial stream and DCT frequency stream simultaneously.
        """)
    with col2:
        st.markdown("""
        #### 🗺️ Explainability
        Grad-CAM heatmap shows exactly which pixels drove the model's decision.
        """)
    with col3:
        st.markdown("""
        #### 📊 Evidence Panel
        DCT frequency map, confidence score, and probability breakdown alongside the verdict.
        """)
    st.stop()

model     = load_model()
pil_img   = Image.open(uploaded).convert("RGB")
img_array = np.array(pil_img)
rgb_input = preprocess_rgb(img_array)
dct_input = preprocess_dct(img_array)

with st.spinner("Analysing..."):
    prob = float(model.predict(
        [rgb_input[np.newaxis], dct_input[np.newaxis]], verbose=0
    )[0][0])

is_fake   = prob >= threshold
uncertain = CONFIDENCE_LOW < prob < CONFIDENCE_HIGH
label     = "FAKE" if is_fake else "REAL"

left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown('<p class="section-header">Input Image</p>', unsafe_allow_html=True)
    st.image(pil_img, use_container_width=True)

with right:
    card_class = "verdict-fake" if is_fake else "verdict-real"
    icon       = "🔴" if is_fake else "🟢"
    st.markdown(
        f'<div class="{card_class}">'
        f'<h1 style="margin:0">{icon} {label}</h1>'
        f'<p style="font-size:1.4rem;margin:4px 0">FAKE probability: <b>{prob:.1%}</b></p>'
        f'{"<p style=color:#f39c12>⚠️ Low confidence — result is uncertain</p>" if uncertain else ""}'
        f'</div>',
        unsafe_allow_html=True
    )
    st.divider()
    p1, p2 = st.columns(2)
    with p1:
        st.metric("REAL probability", f"{(1-prob):.1%}")
    with p2:
        st.metric("FAKE probability", f"{prob:.1%}")
    st.progress(prob)
    st.divider()
    if prob > 0.85:
        st.error("Strong AI generation signal detected in frequency domain.")
    elif prob > 0.65:
        st.warning("Moderate AI generation signal — likely manipulated.")
    elif prob > 0.35:
        st.info("Ambiguous signal — could be real or lightly processed.")
    else:
        st.success("No significant AI generation signal found.")

st.divider()
st.markdown("### Explainability — Where Did the Model Look?")

with st.spinner("Generating Grad-CAM heatmap..."):
    heatmap = make_gradcam(model, rgb_input, dct_input)

e1, e2, e3 = st.columns(3)

with e1:
    st.markdown('<p class="section-header">Original (resized)</p>', unsafe_allow_html=True)
    st.image(cv2.resize(img_array, IMG_SIZE), use_container_width=True)
    st.caption("Input as seen by the RGB stream")

with e2:
    st.markdown('<p class="section-header">Grad-CAM Heatmap</p>', unsafe_allow_html=True)
    if heatmap is not None:
        st.image(overlay_heatmap(img_array, heatmap), use_container_width=True)
        st.caption("🔴 Red = high attention  🔵 Blue = low attention")
    else:
        st.warning("Grad-CAM unavailable for this model.")

with e3:
    st.markdown('<p class="section-header">DCT Frequency Map</p>', unsafe_allow_html=True)
    st.image(visualise_dct(dct_input), use_container_width=True)
    st.caption("Bright = high-frequency AI generation artifacts")

with st.expander("Technical details"):
    st.markdown(f"""
| Property | Value |
|---|---|
| Model | Dual-Stream Xception |
| Decision threshold | {threshold:.2f} |
| Raw FAKE probability | {prob:.6f} |
| Training | CIFAKE + FaceForensics++ C23 |
| Zero-shot tested | ArtiFact (25 generators) |
| Explainability | Grad-CAM on last Xception conv layer |
    """)
