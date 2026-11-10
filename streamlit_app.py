"""Streamlit frontend for the chest X-ray pneumonia classifier.

Place this file in the project root (next to the `xray/` package and
`xray_model.pth`) and run:

    streamlit run streamlit_app.py
"""
from __future__ import annotations

import io
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torchvision.transforms as T
from PIL import Image, ImageOps

try:  # project layout: xray/ml/model/arch.py
    from xray.ml.model.arch import Net
except Exception:  # fallback: arch.py copied next to this file
    from arch import Net

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "xray_model.pth"
DEVICE = torch.device("cpu")
LABELS = {0: "Normal", 1: "Pneumonia"}

# Same preprocessing as app.py / the training pipeline's test transform.
CROP = T.Compose([T.Resize(224), T.CenterCrop(224)])
TO_TENSOR = T.Compose(
    [T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
)

st.set_page_config(page_title="Chest X-ray Screening", page_icon="🫁", layout="wide")

# --------------------------------------------------------------------------- #
# Styling
# --------------------------------------------------------------------------- #
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:wght@500;700&family=Source+Sans+3:wght@400;600&display=swap');

:root {
  --ink: #0F1B24;
  --panel: #172833;
  --line: #2A3F4D;
  --text: #E6EEF2;
  --muted: #8FA5B3;
  --clear: #4DB6A0;
  --flag: #EF6F6C;
}
html, body, .stApp, [data-testid="stSidebar"] { font-family: 'Source Sans 3', sans-serif; }
.stApp { background: var(--ink); color: var(--text); }
[data-testid="stSidebar"] { background: var(--panel); border-right: 1px solid var(--line); }
h1, h2, h3 { font-family: 'Bricolage Grotesque', sans-serif; letter-spacing: -0.01em; }
.block-container { padding-top: 2.2rem; max-width: 1200px; }

.verdict {
  border: 1px solid var(--line); border-left-width: 6px; border-radius: 6px;
  background: var(--panel); padding: 1.1rem 1.3rem; margin-bottom: 1rem;
}
.verdict.clear { border-left-color: var(--clear); }
.verdict.flag  { border-left-color: var(--flag); }
.verdict .kicker { color: var(--muted); font-size: 0.9rem; margin: 0; }
.verdict .result { font-family: 'Bricolage Grotesque', sans-serif; font-size: 2.2rem;
                   font-weight: 700; line-height: 1.15; margin: 0.15rem 0 0.2rem; }
.verdict.clear .result { color: var(--clear); }
.verdict.flag  .result { color: var(--flag); }
.verdict .share { color: var(--text); margin: 0; }

.bar-row { margin: 0.7rem 0; }
.bar-label { display: flex; justify-content: space-between; font-size: 0.95rem; margin-bottom: 0.25rem; }
.bar-track { height: 10px; background: #0B141B; border: 1px solid var(--line); border-radius: 3px; overflow: hidden; }
.bar-fill { height: 100%; }
.note { color: var(--muted); font-size: 0.88rem; }
.empty {
  border: 1px dashed var(--line); border-radius: 6px; padding: 2.2rem 1.5rem;
  color: var(--muted); text-align: center; margin-top: 0.5rem;
}
[data-testid="stHeader"] { background: var(--ink); }
[data-testid="stHeader"] * { color: var(--muted) !important; }
[data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] label { color: var(--text) !important; }
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { color: var(--text); }
.stTabs [data-baseweb="tab"] p { color: var(--muted); }
.stTabs [aria-selected="true"] p { color: var(--text); }
[data-testid="stFileUploader"] section { background: var(--panel); border: 1px dashed var(--line); }
[data-testid="stFileUploader"] section small,
[data-testid="stFileUploader"] section span { color: var(--muted); }
[data-testid="stFileUploaderFileName"] { color: var(--text); }
[data-testid="stMetricValue"], [data-testid="stMetricLabel"] p { color: var(--text) !important; }
@media (prefers-reduced-motion: no-preference) { .bar-fill { transition: width .4s ease; } }
</style>
""",
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------- #
# Model + inference
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading model…")
def load_model():
    if not MODEL_PATH.exists():
        st.error(f"Model file not found: {MODEL_PATH}")
        st.stop()
    obj = torch.load(MODEL_PATH, map_location=DEVICE)
    if isinstance(obj, torch.nn.Module):  # a fully pickled model
        model = obj
    else:  # a state_dict, which is what xray_model.pth contains
        model = Net()
        model.load_state_dict(obj)
    model.to(DEVICE).eval()
    return model


@st.cache_data(show_spinner=False)
def analyze(_model, data: bytes) -> dict:
    """Run the classifier and compute a Grad-CAM map for the predicted class."""
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
    crop = CROP(img)
    x = TO_TENSOR(crop).unsqueeze(0).to(DEVICE).requires_grad_(True)

    store = {}

    def hook(_module, _inp, out):
        out.retain_grad()
        store["act"] = out

    # Last conv block before global average pooling.
    handle = _model.convolution_block9.register_forward_hook(hook)
    try:
        _model.zero_grad()
        out = _model(x)  # shape (1, 2)
        idx = int(out.argmax(dim=1).item())
        out[0, idx].backward()
        act = store["act"]
        weights = act.grad.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * act).sum(dim=1)).squeeze(0).detach().numpy()
    finally:
        handle.remove()

    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)

    scores = out.detach().squeeze(0).numpy().astype(float)  # raw logits
    share = torch.softmax(out.detach(), dim=1).squeeze(0).numpy().astype(float)
    return {
        "crop": np.asarray(crop),
        "cam": cam,
        "idx": idx,
        "scores": scores,
        "share": share,
    }


# Small built-in colormaps (RGB stops), so matplotlib isn't needed.
COLORMAPS = {
    "inferno": [(0, 0, 4), (31, 12, 72), (85, 15, 109), (136, 34, 106), (186, 54, 85),
                (227, 89, 51), (249, 140, 10), (249, 201, 50), (252, 255, 164)],
    "viridis": [(68, 1, 84), (71, 44, 122), (59, 81, 139), (44, 113, 142), (33, 144, 141),
                (39, 173, 129), (92, 200, 99), (170, 220, 50), (253, 231, 37)],
    "magma": [(0, 0, 4), (28, 16, 68), (79, 18, 123), (129, 37, 129), (181, 54, 122),
              (229, 80, 100), (251, 135, 97), (254, 194, 135), (252, 253, 191)],
    "jet": [(0, 0, 128), (0, 0, 255), (0, 128, 255), (0, 255, 255), (128, 255, 128),
            (255, 255, 0), (255, 128, 0), (255, 0, 0), (128, 0, 0)],
}


def apply_colormap(values: np.ndarray, name: str) -> np.ndarray:
    """Map values in [0, 1] to RGB floats in [0, 1]."""
    stops = np.asarray(COLORMAPS[name], dtype=float) / 255.0
    xs = np.linspace(0.0, 1.0, len(stops))
    return np.stack([np.interp(values, xs, stops[:, c]) for c in range(3)], axis=-1)


def overlay(crop: np.ndarray, cam: np.ndarray, strength: float, cmap: str):
    heat = Image.fromarray((cam * 255).astype(np.uint8)).resize((224, 224), Image.BILINEAR)
    heat = np.asarray(heat) / 255.0
    rgb = apply_colormap(heat, cmap)
    w = (strength * heat)[..., None]  # low-activation areas stay see-through
    blended = crop / 255.0 * (1 - w) + rgb * w
    return (blended * 255).astype(np.uint8), (rgb * 255).astype(np.uint8)


def score_bar(label: str, value: float, color: str) -> str:
    pct = max(0.0, min(1.0, value)) * 100
    return (
        f'<div class="bar-row"><div class="bar-label"><span>{label}</span>'
        f"<span>{pct:.1f}%</span></div>"
        f'<div class="bar-track"><div class="bar-fill" style="width:{pct:.1f}%;background:{color}"></div></div></div>'
    )


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
model = load_model()

with st.sidebar:
    st.header("Display")
    strength = st.slider("Heatmap strength", 0.0, 1.0, 0.65, 0.05)
    cmap = st.selectbox("Heatmap colors", ["inferno", "viridis", "jet", "magma"])
    st.divider()
    st.subheader("Before you use this")
    st.markdown(
        '<p class="note">This is a research demo trained on a small dataset. '
        "It is not a medical device and must not be used to diagnose or rule out "
        "pneumonia. A qualified clinician has to review every X-ray.</p>",
        unsafe_allow_html=True,
    )

if "history" not in st.session_state:
    st.session_state.history = {}

# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
st.title("Chest X-ray screening")
st.write("Upload a chest X-ray to see the model's prediction and the lung regions that drove it.")

tab_analyze, tab_history, tab_about = st.tabs(["Analyze", "Session history", "About the model"])

with tab_analyze:
    files = st.file_uploader(
        "Chest X-ray images (JPEG or PNG)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )

    if not files:
        st.markdown(
            '<div class="empty">No image yet. Drop one or more chest X-rays above '
            "to get a prediction and a heatmap.</div>",
            unsafe_allow_html=True,
        )
    else:
        results = {}
        for f in files:
            data = f.getvalue()
            try:
                res = analyze(model, data)
            except Exception as exc:
                st.error(f"Could not read {f.name}: {exc}")
                continue
            results[f.name] = res
            key = (f.name, len(data))
            if key not in st.session_state.history:
                st.session_state.history[key] = {
                    "Time": datetime.now().strftime("%H:%M:%S"),
                    "File": f.name,
                    "Prediction": LABELS[res["idx"]],
                    "Normal probability": round(float(res["share"][0]), 4),
                    "Pneumonia probability": round(float(res["share"][1]), 4),
                }

        if results:
            names = list(results)
            chosen = names[0]
            if len(names) > 1:
                chosen = st.selectbox("Image to inspect", names)
            res = results[chosen]

            blended, heat_only = overlay(res["crop"], res["cam"], strength, cmap)
            left, right = st.columns([1.15, 1], gap="large")

            with left:
                view = st.radio(
                    "View", ["Overlay", "Heatmap only", "Original"], horizontal=True
                )
                shown = {"Overlay": blended, "Heatmap only": heat_only, "Original": res["crop"]}[view]
                st.image(shown, use_container_width=True, caption=f"{chosen} (resized to 224×224)")

            with right:
                is_flag = res["idx"] == 1
                st.markdown(
                    f"""<div class="verdict {'flag' if is_flag else 'clear'}">
<p class="kicker">Model prediction</p>
<p class="result">{LABELS[res['idx']]}</p>
<p class="share">{res['share'][res['idx']] * 100:.1f}% model probability</p>
</div>""",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    score_bar("Normal", float(res["share"][0]), "#4DB6A0")
                    + score_bar("Pneumonia", float(res["share"][1]), "#EF6F6C"),
                    unsafe_allow_html=True,
                )
                st.markdown(
                    '<p class="note">Percentages are softmax probabilities of the two model outputs. '
                    "A result close to 50/50 means the model is unsure. Bright areas on the "
                    "heatmap are where the model looked most for its predicted class.</p>",
                    unsafe_allow_html=True,
                )

            if len(results) > 1:
                st.subheader("Batch summary")
                rows = [
                    {
                        "File": n,
                        "Prediction": LABELS[r["idx"]],
                        "Normal": f"{r['share'][0] * 100:.1f}%",
                        "Pneumonia": f"{r['share'][1] * 100:.1f}%",
                    }
                    for n, r in results.items()
                ]
                st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

with tab_history:
    hist = list(st.session_state.history.values())
    if not hist:
        st.markdown(
            '<div class="empty">Predictions you run in this session will be listed here.</div>',
            unsafe_allow_html=True,
        )
    else:
        df = pd.DataFrame(hist)
        counts = df["Prediction"].value_counts()
        c1, c2, c3 = st.columns(3)
        c1.metric("Images analyzed", len(df))
        c2.metric("Normal", int(counts.get("Normal", 0)))
        c3.metric("Pneumonia", int(counts.get("Pneumonia", 0)))
        st.dataframe(df, hide_index=True, use_container_width=True)
        b1, b2, _ = st.columns([1, 1, 3])
        b1.download_button(
            "Download CSV", df.to_csv(index=False).encode(), "xray_predictions.csv", "text/csv"
        )
        if b2.button("Clear history"):
            st.session_state.history = {}
            st.rerun()

with tab_about:
    n_params = sum(p.numel() for p in model.parameters())
    st.subheader("How it works")
    st.markdown(
        f"""
The classifier is a custom CNN built in PyTorch ({n_params:,} parameters): nine small
convolution blocks with batch normalization, three max-pooling stages, global average pooling
and a final 4×4 convolution that outputs two class scores (Normal, Pneumonia).

**Preprocessing.** Each image is converted to RGB, resized to 224 on the short side, center-cropped
to 224×224 and normalized with ImageNet mean and standard deviation, matching the training pipeline.

**Heatmap (Grad-CAM).** The gradient of the predicted class score is traced back to the last
convolution block. Its channels are weighted by how much they influenced the score and summed
into a 19×19 map, which is scaled up to the image. It shows where the model looked, which is
not the same as proof of where disease is.

**Limits.** The center crop can cut off the edges of wide images. The model was trained on one
dataset from a single source, so images from other scanners or hospitals may behave differently.
"""
    )
