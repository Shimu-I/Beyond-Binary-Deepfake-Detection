import os
import traceback
import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image, UnidentifiedImageError

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

IMG_SIZE         = (299, 299)
CONFIDENCE_LOW   = 0.40
CONFIDENCE_HIGH  = 0.60
MODEL_FILENAME   = "ffpp_adapted_final.keras"
MAX_UPLOAD_MB    = 15   # reject absurdly large uploads before they hit PIL/cv2


# ── Model loading ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    """Load the model from disk. Returns None on failure instead of crashing."""
    with st.spinner("Loading model..."):
        if not os.path.exists(MODEL_FILENAME):
            return None, f"Model file not found: `{MODEL_FILENAME}`. Place it next to app.py."
        try:
            model = tf.keras.models.load_model(MODEL_FILENAME)
        except Exception as e:
            traceback.print_exc()  # full traceback goes to your terminal, not the user
            return None, f"Failed to load model ({type(e).__name__}). Check the terminal log for details."
        return model, None


# ── Preprocessing ────────────────────────────────────────────────────────────
def preprocess_rgb(img_array):
    resized = cv2.resize(img_array, IMG_SIZE).astype(np.float32) / 255.0
    return resized


def preprocess_dct(img_array):
    gray = cv2.cvtColor(img_array.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, IMG_SIZE).astype(np.float32)
    dct  = cv2.dct(gray)
    dct  = np.log(np.abs(dct) + 1)
    dct  = cv2.normalize(dct, None, 0, 1, cv2.NORM_MINMAX)
    return np.stack([dct] * 3, axis=-1)


# ── Grad-CAM ─────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def build_grad_model(_model, last_conv_layer="block14_sepconv2_act"):
    """Build the Grad-CAM sub-model once and cache it, instead of rebuilding
    it from scratch on every single upload."""
    spatial_model = _model.get_layer("spatial")
    conv_layer    = spatial_model.get_layer(last_conv_layer)
    return tf.keras.Model(
        inputs  = spatial_model.inputs,
        outputs = [conv_layer.output, spatial_model.output]
    )


def make_gradcam(model, rgb_input, last_conv_layer="block14_sepconv2_act"):
    """Returns a heatmap array, or None if anything goes wrong (never raises)."""
    try:
        grad_model = build_grad_model(model, last_conv_layer)
        rgb_t = tf.cast(rgb_input[np.newaxis], tf.float32)

        with tf.GradientTape() as tape:
            conv_out, spatial_out = grad_model(rgb_t)
            loss = tf.reduce_mean(spatial_out)

        grads = tape.gradient(loss, conv_out)
        if grads is None:
            return None

        pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
        cam    = tf.reduce_sum(tf.multiply(pooled, conv_out[0]), axis=-1).numpy()
        cam    = np.maximum(cam, 0)
        cam    = cv2.resize(cam, IMG_SIZE)
        cam_range = cam.max() - cam.min()
        if cam_range < 1e-8:
            return None
        cam = (cam - cam.min()) / (cam_range + 1e-8)
        return cam
    except Exception as e:
        print("Grad-CAM error:", repr(e))
        return None


def overlay_heatmap(original_rgb, heatmap, alpha=0.45):
    coloured = (cm.jet(heatmap)[:, :, :3] * 255).astype(np.uint8)
    base     = cv2.resize(original_rgb.astype(np.uint8), IMG_SIZE)
    return cv2.addWeighted(base, 1 - alpha, coloured, alpha, 0)


def visualise_dct(dct_map):
    vis = (dct_map[:, :, 0] * 255).astype(np.uint8)
    vis = cv2.applyColorMap(vis, cv2.COLORMAP_INFERNO)
    return cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)


# ── Sidebar ──────────────────────────────────────────────────────────────────
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

# ── Load model up front so we fail fast with a clean message ────────────────
model, load_error = load_model()
if load_error:
    st.error(f"⚠️ {load_error}")
    st.stop()

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

# ── Guard: upload size ────────────────────────────────────────────────────────
size_mb = uploaded.size / (1024 * 1024)
if size_mb > MAX_UPLOAD_MB:
    st.error(f"File is {size_mb:.1f} MB — please upload something under {MAX_UPLOAD_MB} MB.")
    st.stop()

# ── Guard: image decoding ─────────────────────────────────────────────────────
try:
    pil_img = Image.open(uploaded).convert("RGB")
    img_array = np.array(pil_img)
    if img_array.size == 0:
        raise ValueError("Empty image")
except UnidentifiedImageError:
    st.error("That file doesn't look like a valid image. Try a JPG, PNG, or WEBP.")
    st.stop()
except Exception as e:
    st.error(f"Couldn't read that image ({type(e).__name__}). Try a different file.")
    st.stop()

# ── Guard: preprocessing ───────────────────────────────────────────────────────
try:
    rgb_input = preprocess_rgb(img_array)
    dct_input = preprocess_dct(img_array)
except Exception as e:
    st.error(f"Failed while preparing the image for the model ({type(e).__name__}).")
    st.stop()

# ── Guard: inference ────────────────────────────────────────────────────────────
try:
    with st.spinner("Analysing..."):
        prob = float(model.predict(
            [rgb_input[np.newaxis], dct_input[np.newaxis]], verbose=0
        )[0][0])
except Exception as e:
    traceback.print_exc()
    st.error(f"Inference failed ({type(e).__name__}). Check the terminal log for details.")
    st.stop()

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

e1, e2, e3 = st.columns(3)

with e1:
    st.markdown('<p class="section-header">Original (resized)</p>', unsafe_allow_html=True)
    original_slot = st.empty()
    original_caption = st.empty()

with e2:
    st.markdown('<p class="section-header">Grad-CAM Heatmap</p>', unsafe_allow_html=True)
    gradcam_slot = st.empty()
    gradcam_slot.info("Computing...")
    st.caption("🔴 Red = high attention  🔵 Blue = low attention")

with e3:
    st.markdown('<p class="section-header">DCT Frequency Map</p>', unsafe_allow_html=True)
    dct_slot = st.empty()
    dct_slot.info("Computing...")
    st.caption("Bright = high-frequency AI generation artifacts")


original_slot.image(cv2.resize(img_array, IMG_SIZE), use_container_width=True)
original_caption.caption("Input as seen by the RGB stream")

with st.spinner("Generating Grad-CAM heatmap..."):
    heatmap = make_gradcam(model, rgb_input)

if heatmap is not None:
    try:
        gradcam_slot.image(overlay_heatmap(img_array, heatmap), use_container_width=True)
    except Exception:
        gradcam_slot.warning("Grad-CAM heatmap couldn't be rendered for this image.")
else:
    gradcam_slot.warning("Grad-CAM unavailable for this model/image.")

try:
    dct_slot.image(visualise_dct(dct_input), use_container_width=True)
except Exception:
    dct_slot.warning("DCT map couldn't be rendered for this image.")

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