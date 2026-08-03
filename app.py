import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"

import io

import numpy as np
import tensorflow as tf
from PIL import Image

import streamlit as st

st.set_page_config(
    page_title="PEPPER.SPOT — Bacterial Spot Detector",
    layout="wide",
    initial_sidebar_state="collapsed",
)

MODEL_PATH = "models/custom_cnn_best.keras"
# REFERENCE_PATH = "models/leaf_reference.npz"
CLASS_NAMES = ["Bacterial_Spot", "Healthy"]

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Archivo:wght@400;500;700&family=Archivo+Black&family=IBM+Plex+Mono:wght@400;500;700&display=swap');

:root {
  --paper: #FFF8E7;
  --ink: #141414;
  --leaf: #2F7D3F;
  --tomato: #D0342C;
  --mustard: #F5C518;
  --cream: #FFF3C9;
  --slate: #4A5568;
}

#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header[data-testid="stHeader"] { visibility: hidden; }
[data-testid="stToolbar"] { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }
[data-testid="stStatusWidget"] { display: none; }
.stDeployButton { display: none; }

.block-container {
  max-width: 1000px;
  margin: 0 auto;
  padding: 0 !important;
}

.stApp {
  background: var(--paper);
  color: var(--ink);
  font-family: 'Archivo', sans-serif;
}

.stApp a { color: var(--ink); }

.nb-header {
  background: var(--ink);
  color: var(--paper);
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  flex-wrap: wrap;
  padding: 20px 32px;
  margin-left: calc(50% - 50vw);
  margin-right: calc(50% - 50vw);
}

.nb-header .logo {
  font-family: 'Archivo Black', sans-serif;
  text-transform: uppercase;
  font-size: 24px;
  letter-spacing: -0.01em;
  color: var(--paper);
  display: flex;
  align-items: center;
  gap: 12px;
}

.nb-header .logo .mark { color: var(--mustard); display: inline-flex; }

.nb-header .logo span.accent { color: var(--mustard); }

.nb-badge {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 12px;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  border: 2px solid var(--paper);
  padding: 6px 12px;
  transform: rotate(1deg);
  display: inline-block;
}

.nb-ticker {
  background: var(--mustard);
  border-bottom: 3px solid var(--ink);
  overflow: hidden;
  white-space: nowrap;
  margin-left: calc(50% - 50vw);
  margin-right: calc(50% - 50vw);
}

.nb-ticker .track {
  display: inline-block;
  padding: 8px 0;
  font-family: 'IBM Plex Mono', monospace;
  font-size: 12px;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  animation: nb-marquee 22s linear infinite;
}

@keyframes nb-marquee {
  from { transform: translateX(0); }
  to { transform: translateX(-50%); }
}

.nb-hero { padding: 72px 24px 40px; }

.nb-hero h1 {
  font-family: 'Archivo Black', sans-serif;
  text-transform: uppercase;
  font-size: clamp(2.4rem, 6vw, 4.6rem);
  letter-spacing: -0.02em;
  line-height: 0.92;
  margin: 0 0 20px;
  color: var(--ink);
}

.nb-hero h1 .outline {
  background: var(--mustard);
  box-shadow: 6px 6px 0 0 var(--ink);
  padding: 0 12px;
  border: 3px solid var(--ink);
}

.nb-hero .mono-line {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 13px;
  letter-spacing: 0.09em;
  text-transform: uppercase;
  color: var(--ink);
  opacity: 0.7;
}

.nb-intro {
  font-family: 'Archivo', sans-serif;
  font-size: 17px;
  line-height: 1.6;
  color: var(--ink);
  opacity: 0.85;
  max-width: 520px;
  margin: 0 0 20px;
}

.nb-mono {
  font-family: 'IBM Plex Mono', monospace;
  text-transform: uppercase;
  letter-spacing: 0.08em;
}

[data-testid="stFileUploader"] label p {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 13px;
  letter-spacing: 0.09em;
  text-transform: uppercase;
  color: var(--ink) !important;
  font-weight: 500;
}

[data-testid="stFileUploaderDropzone"] {
  border: 3px dashed var(--ink) !important;
  border-radius: 4px !important;
  background: #FFFFFF !important;
  box-shadow: 6px 6px 0 0 var(--ink) !important;
  padding: 44px 24px !important;
}

[data-testid="stFileUploaderDropzone"]:hover {
  background: var(--cream) !important;
  border-style: solid !important;
}

[data-testid="stFileUploaderDropzone"] button {
  background: var(--mustard) !important;
  color: var(--ink) !important;
  border: 2px solid var(--ink) !important;
  border-radius: 4px !important;
  font-family: 'Archivo Black', sans-serif !important;
  text-transform: uppercase !important;
  box-shadow: 3px 3px 0 0 var(--ink) !important;
}

[data-testid="stFileUploaderFileName"] {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 12px;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  border: 2px solid var(--ink);
  border-radius: 4px;
  box-shadow: 3px 3px 0 0 var(--ink);
}

.stButton > button {
  background: var(--mustard) !important;
  color: var(--ink) !important;
  border: 3px solid var(--ink) !important;
  border-radius: 4px !important;
  box-shadow: 6px 6px 0 0 var(--ink) !important;
  font-family: 'Archivo Black', sans-serif !important;
  text-transform: uppercase !important;
  letter-spacing: 0.02em;
  font-size: 18px;
  padding: 14px 30px;
  transition: transform 0.05s ease, box-shadow 0.05s ease;
}

.stButton > button:hover {
  transform: translate(-2px, -2px);
  box-shadow: 8px 8px 0 0 var(--ink) !important;
  color: var(--ink) !important;
  border-color: var(--ink) !important;
  background: var(--mustard) !important;
}

.stButton > button:active {
  transform: translate(6px, 6px);
  box-shadow: 0 0 0 0 var(--ink) !important;
}

.stButton > button:focus {
  color: var(--ink) !important;
  border-color: var(--ink) !important;
  box-shadow: 6px 6px 0 0 var(--ink) !important;
  outline: none !important;
}

[data-testid="stImage"] img {
  border: 3px solid var(--ink);
  border-radius: 4px;
  box-shadow: 8px 8px 0 0 var(--ink);
}

.nb-stamp {
  border: 3px solid var(--ink);
  border-radius: 4px;
  background: #FFFFFF;
  box-shadow: 8px 8px 0 0 var(--ink);
  padding: 26px;
}

.nb-stamp-healthy { border-top: 14px solid var(--leaf); }
.nb-stamp-diseased { border-top: 14px solid var(--tomato); }
.nb-stamp-unknown { border-top: 14px solid var(--slate); }

.nb-stamp-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  flex-wrap: wrap;
  margin-bottom: 8px;
}

.nb-stamp-word {
  font-family: 'Archivo Black', sans-serif;
  text-transform: uppercase;
  font-size: clamp(1.5rem, 4vw, 2.6rem);
  letter-spacing: -0.01em;
  color: #FFFFFF;
  border: 3px solid var(--ink);
  border-radius: 4px;
  padding: 8px 18px;
  transform: rotate(-1deg);
  box-shadow: 4px 4px 0 0 var(--ink);
}

.nb-stamp-healthy .nb-stamp-word { background: var(--leaf); }
.nb-stamp-diseased .nb-stamp-word { background: var(--tomato); }
.nb-stamp-unknown .nb-stamp-word { background: var(--slate); }

.nb-stamp-code {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 12px;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  border: 2px solid var(--ink);
  padding: 5px 10px;
  border-radius: 4px;
}

.nb-conf-label {
  display: flex;
  justify-content: space-between;
  font-family: 'IBM Plex Mono', monospace;
  font-size: 12px;
  letter-spacing: 0.09em;
  text-transform: uppercase;
  margin: 18px 0 8px;
}

.nb-conf-track {
  display: flex;
  gap: 4px;
  border: 3px solid var(--ink);
  padding: 6px;
  background: #FFFFFF;
}

.nb-conf-track span {
  flex: 1;
  height: 24px;
  border: 2px solid var(--ink);
  border-radius: 2px;
  background: #FFFFFF;
}

.nb-conf-track.healthy span.on { background: var(--leaf); }
.nb-conf-track.diseased span.on { background: var(--tomato); }
.nb-conf-track.unknown span.on { background: var(--slate); }

.nb-blurb {
  font-family: 'Archivo', sans-serif;
  font-size: 15px;
  line-height: 1.6;
  margin: 18px 0 0;
  color: var(--ink);
}

.nb-meta {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 11px;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--ink);
  opacity: 0.65;
  border-top: 2px solid var(--ink);
  margin-top: 18px;
  padding-top: 12px;
}

.nb-section {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 12px;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  margin: 64px 0 14px;
  display: inline-block;
  background: var(--ink);
  color: var(--paper);
  padding: 6px 12px;
  border-radius: 4px;
}

.nb-info {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
  border: 3px solid var(--ink);
  border-radius: 4px;
  box-shadow: 6px 6px 0 0 var(--ink);
  background: #FFFFFF;
}

.nb-info > div {
  padding: 18px 22px;
  border-right: 2px solid var(--ink);
}

.nb-info > div:last-child { border-right: none; }

.nb-info .lbl {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 11px;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--ink);
  opacity: 0.6;
  margin-bottom: 6px;
}

.nb-info .val {
  font-family: 'Archivo', sans-serif;
  font-weight: 700;
  font-size: 16px;
  text-transform: uppercase;
  letter-spacing: 0.01em;
}

.nb-footer {
  margin-top: 80px;
  border-top: 3px solid var(--ink);
  padding: 22px 24px;
  font-family: 'IBM Plex Mono', monospace;
  font-size: 11px;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  display: flex;
  justify-content: space-between;
  gap: 12px;
  flex-wrap: wrap;
}

.nb-footer a {
  color: var(--ink);
  font-weight: 700;
  text-decoration: none;
  border-bottom: 2px solid var(--ink);
}

.nb-footer a:hover {
  background: var(--mustard);
}

@media (max-width: 768px) {
  .nb-header { padding: 16px 18px; }
  .nb-header .logo { font-size: 20px; }
  .nb-badge { font-size: 10px; padding: 5px 9px; }
  .nb-hero { padding: 48px 18px 28px; }
  .nb-hero h1 { font-size: clamp(2rem, 9vw, 3.2rem); }
  .nb-stamp { padding: 18px; }
  .nb-stamp-word { font-size: clamp(1.3rem, 7vw, 2rem); }
  .nb-info { grid-template-columns: 1fr 1fr; }
  .nb-footer { flex-direction: column; text-align: center; gap: 8px; }
  [data-testid="stFileUploaderDropzone"] { padding: 32px 16px !important; }
  [data-testid="stFileUploader"] { padding: 0 24px; }
  .stButton { padding: 0 24px; }
}

@media (max-width: 480px) {
  .nb-header { justify-content: center; text-align: center; }
  .nb-info { grid-template-columns: 1fr; }
  .nb-info > div { border-right: none; border-bottom: 2px solid var(--ink); }
  .nb-info > div:last-child { border-bottom: none; }
}
"""

TICKER_TEXT = "HEALTHY \u00b7 BACTERIAL SPOT \u00b7 PEPPER LEAF DETECTOR \u00b7 CO11 \u00b7 "
TICKER = (
    '<div class="nb-ticker"><div class="track">'
    + (TICKER_TEXT * 6)
    + "</div></div>"
)

LEAF_SVG = (
    '<svg class="mark" width="28" height="28" viewBox="0 0 32 32" fill="none" '
    'stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
    '<path d="M4 28 C4 12 16 4 28 4 C28 16 20 28 4 28 Z"/>'
    '<path d="M8 24 C14 18 20 12 24 8"/>'
    "</svg>"
)

HEADER = f"""
<div class="nb-header">
  <div class="logo">{LEAF_SVG}<span>PEPPER<span class="accent">.SPOT</span></span></div>
  <div class="nb-badge">GROUP CO11 &middot; MINI-PROJECT</div>
</div>
"""

HERO = """
<div class="nb-hero">
  <div class="mono-line">Bacterial Spot Detection System</div>
  <h1>Pepper Leaf<br/><span class="outline">Disease Scan</span></h1>
  <p class="nb-intro">Upload a photo of a pepper leaf. The model checks whether the leaf is healthy or infected with bacterial spot.</p>
</div>
"""


def build_css():
    st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    dummy = np.zeros((1, 128, 128, 3), dtype=np.float32)
    _ = model.predict(dummy, verbose=0)
    return model


# @st.cache_resource(show_spinner=False)
# def load_reference():
#     z = np.load(REFERENCE_PATH)
#     return z["embeddings"], float(z["threshold"])


def preprocess(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((128, 128))
    arr = np.array(img, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)
    return arr


def conf_bar(confidence, result_cls):
    filled = int(round(confidence * 10))
    segs = []
    for i in range(10):
        cls = "on" if i < filled else ""
        segs.append(f'<span class="{cls}"></span>')
    return f'<div class="nb-conf-track {result_cls}">{"".join(segs)}</div>'


def similarity_bar(distance, threshold, result_cls):
    frac = min(1.0, distance / (threshold * 2))
    filled = int(round((1.0 - frac) * 10))
    segs = []
    for i in range(10):
        cls = "on" if i < filled else ""
        segs.append(f'<span class="{cls}"></span>')
    return f'<div class="nb-conf-track {result_cls}">{"".join(segs)}</div>'


def unknown_html(distance, threshold):
    blurb = (
        "This photo does not resemble the pepper leaves this model was "
        "trained on, so a health result would not be trustworthy. "
        "Try a clear, close-up photo of a single pepper leaf."
    )
    return f"""
<div class="nb-stamp nb-stamp-unknown">
  <div class="nb-stamp-head">
    <span class="nb-stamp-word">Not a Leaf</span>
    <span class="nb-stamp-code">Uncertain</span>
  </div>
  <div class="nb-conf-label"><span>Leaf similarity</span><span>{distance:.2f} / {threshold:.2f}</span></div>
  {similarity_bar(distance, threshold, "unknown")}
  <p class="nb-blurb">{blurb}</p>
  <div class="nb-meta">MODEL: CUSTOM CNN &middot; INPUT: 128x128 &middot; ENGINE: TENSORFLOW</div>
</div>
"""


def result_html(label, confidence):
    diseased = label == "Bacterial_Spot"
    result_cls = "diseased" if diseased else "healthy"
    word = "BACTERIAL SPOT" if diseased else "HEALTHY"
    code = "SCAN COMPLETE"
    if diseased:
        blurb = (
            "The model detected visual characteristics consistent with "
            "bacterial spot on the uploaded leaf. Inspect the plant for "
            "spreading lesions and consider treatment."
        )
    else:
        blurb = (
            "No visible symptoms of bacterial spot were detected. "
            "The leaf presents as healthy."
        )
    return f"""
<div class="nb-stamp nb-stamp-{result_cls}">
  <div class="nb-stamp-head">
    <span class="nb-stamp-word">{word}</span>
    <span class="nb-stamp-code">{code}</span>
  </div>
  <div class="nb-conf-label"><span>Confidence</span><span>{confidence:.1%}</span></div>
  {conf_bar(confidence, result_cls)}
  <p class="nb-blurb">{blurb}</p>
  <div class="nb-meta">MODEL: CUSTOM CNN &middot; INPUT: 128x128 &middot; ENGINE: TENSORFLOW</div>
</div>
"""


INFO = """
<span class="nb-section">System Specs</span>
<div class="nb-info">
  <div><div class="lbl">Model</div><div class="val">Custom CNN</div></div>
  <div><div class="lbl">Classes</div><div class="val">Healthy &middot; Spot</div></div>
  <div><div class="lbl">Input Size</div><div class="val">128 x 128</div></div>
  <div><div class="lbl">Engine</div><div class="val">TensorFlow</div></div>
</div>
"""

FOOTER = """
<div class="nb-footer">
  <div>PEPPER.SPOT &copy; 2026 &middot; GROUP CO11</div>
  <div>DEPT. OF COMPUTER ENGINEERING &middot; GET324 &middot; UNIUYO</div>
  <div><a href="https://github.com/Idighekere/pepper-bacterial-spot-detector-CO11" target="_blank" rel="noopener noreferrer">SOURCE: GITHUB</a></div>
</div>
"""

build_css()
st.markdown(HEADER, unsafe_allow_html=True)
st.markdown(TICKER, unsafe_allow_html=True)
st.markdown(HERO, unsafe_allow_html=True)

model = load_model()
# extractor = tf.keras.Model(inputs=model.input, outputs=model.layers[-2].output)
# ref_embeddings, leaf_threshold = load_reference()

c1, c2, c3 = st.columns([1, 2, 1])
with c2:
    uploaded = st.file_uploader(
        "Upload Leaf Image", type=["png", "jpg", "jpeg"]
    )

    if uploaded is not None:
        img_bytes = uploaded.getvalue()
        st.session_state["img_bytes"] = img_bytes
        st.image(Image.open(io.BytesIO(img_bytes)), width="stretch")

    analyze = st.button("Analyze Leaf", width="stretch")

    if analyze:
        if "img_bytes" not in st.session_state:
            st.warning("No image uploaded. Attach a leaf photo first.")
        else:
            arr = preprocess(st.session_state["img_bytes"])
            # OOD guard disabled: always run the model and show the result.
            # feats = extractor.predict(arr, verbose=0)[0]
            # feats = feats / (np.linalg.norm(feats) + 1e-8)
            # distance = float(np.min(np.linalg.norm(ref_embeddings - feats, axis=1)))
            # if distance > leaf_threshold:
            #     st.markdown(unknown_html(distance, leaf_threshold), unsafe_allow_html=True)
            # else:
            probs = model.predict(arr, verbose=0)[0]
            pred_idx = int(np.argmax(probs))
            label = CLASS_NAMES[pred_idx]
            confidence = float(probs[pred_idx])
            st.markdown(result_html(label, confidence), unsafe_allow_html=True)

st.markdown('<div style="padding: 0 24px;">' + INFO + "</div>", unsafe_allow_html=True)
st.markdown(FOOTER, unsafe_allow_html=True)
