"""Streamlit frontend for the chest X-ray pneumonia classifier.

Place this file in the project root (next to the `xray/` package and
`xray_model.pth`) and run:

    streamlit run streamlit_app.py
"""
from __future__ import annotations

import io
import re
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

# Extra feature modules that sit next to this file. If one cannot be imported
# (for example a package is missing) the rest of the app still works.
_FEATURE_ERRORS: dict = {}
try:
    from severity import estimate_severity
    _HAS_SEVERITY = True
except Exception as _exc:
    _HAS_SEVERITY = False
    _FEATURE_ERRORS["severity"] = str(_exc)
try:
    from dicom_handler import is_dicom_name, read_dicom
    _HAS_DICOM = True
except Exception as _exc:
    _HAS_DICOM = False
    _FEATURE_ERRORS["dicom"] = str(_exc)
    def is_dicom_name(name: str) -> bool:
        return str(name).lower().endswith((".dcm", ".dicom"))
try:
    from report_generator import generate_pdf_report
    _HAS_REPORT = True
except Exception as _exc:
    _HAS_REPORT = False
    _FEATURE_ERRORS["report"] = str(_exc)

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "xray_model.pth"
SAMPLE_DIR = APP_DIR / "sample_images"
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
:root { --warn: #E8B04B; --blue: #7FB3D5; }
.card { border: 1px solid var(--line); border-radius: 6px; background: var(--panel);
        padding: 0.9rem 1.2rem; margin-bottom: 1rem; }
.card-title { font-family: 'Bricolage Grotesque', sans-serif; font-weight: 700;
              font-size: 1.02rem; margin: 0 0 0.35rem; color: var(--text); }
.sev-badge { display: inline-block; padding: 0.25rem 0.8rem; border-radius: 999px;
             font-weight: 600; font-size: 0.85rem; border: 1px solid; }
.sev-mild { color: var(--clear); background: rgba(77,182,160,.14); border-color: rgba(77,182,160,.4); }
.sev-moderate { color: var(--warn); background: rgba(232,176,75,.14); border-color: rgba(232,176,75,.4); }
.sev-severe { color: var(--flag); background: rgba(239,111,108,.14); border-color: rgba(239,111,108,.4); }
.region-row { display: grid; grid-template-columns: repeat(5, 1fr); gap: 6px; margin: 0.7rem 0 0.2rem; }
.region-cell { border: 1px solid var(--line); border-radius: 5px; padding: 0.35rem 0.2rem;
               text-align: center; font-size: 0.72rem; color: var(--muted); line-height: 1.35; }
.region-cell b { display: block; color: var(--text); font-size: 0.95rem; }
.metric-card { border: 1px solid var(--line); border-radius: 6px; background: var(--panel);
               padding: 0.9rem 1rem; text-align: center; }
.metric-card .mc-val { font-family: 'Bricolage Grotesque', sans-serif; font-size: 1.9rem;
                       font-weight: 700; margin: 0; line-height: 1.2; }
.metric-card .mc-lbl { color: var(--muted); font-size: 0.8rem; margin: 0.1rem 0 0;
                       letter-spacing: .05em; text-transform: uppercase; }
.timeline-row { border-bottom: 1px solid var(--line); padding: 0.5rem 0; font-size: 0.93rem; }
.stDownloadButton > button { border-color: var(--blue); color: var(--blue); }
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


def region_cell(label: str, pct: float) -> str:
    alpha = 0.06 + 0.5 * min(pct, 60.0) / 60.0
    return (
        f'<div class="region-cell" style="background:rgba(239,111,108,{alpha:.2f})">'
        f"{label}<b>{pct:.0f}%</b></div>"
    )


def patient_info() -> dict:
    """Patient details saved from the sidebar form (blank values stay blank)."""
    ss = st.session_state
    sex = ss.get("pi_sex", "")
    return {
        "Patient name": ss.get("pi_name", "").strip(),
        "Patient ID": ss.get("pi_id", "").strip(),
        "Age": ss.get("pi_age", "").strip(),
        "Sex": "" if sex == "Not specified" else sex,
        "Referring physician": ss.get("pi_doctor", "").strip(),
    }


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
model = load_model()

with st.sidebar:
    st.header("🎛️ Display Settings")
    strength = st.slider("Heatmap strength", 0.0, 1.0, 0.65, 0.05)
    cmap = st.selectbox("Heatmap colors", ["inferno", "viridis", "jet", "magma"])
    st.divider()

    st.subheader("👤 Patient Information")
    with st.form("patient_form", border=False):
        st.text_input("Patient name", key="pi_name", placeholder="e.g. Rahul Sharma")
        st.text_input("Patient ID", key="pi_id", placeholder="e.g. P-1024")
        st.text_input("Age (years)", key="pi_age", max_chars=3)
        st.selectbox("Sex", ["Not specified", "Male", "Female", "Other"], key="pi_sex")
        st.text_input("Referring physician", key="pi_doctor")
        st.form_submit_button("💾 Save patient details")
    _saved_name = st.session_state.get("pi_name", "").strip()
    if _saved_name:
        st.caption(f"✅ Saved for the report: {_saved_name}")
    else:
        st.caption(
            "Press Save (or Enter) so these details appear in the PDF report. "
            "Use English letters, as the PDF supports Latin characters."
        )
    st.divider()

    st.subheader("⚠️ Disclaimer")
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
st.title("Chest X-ray analysis")
st.write("Upload a chest X-ray to see the model's prediction and the lung regions that drove it.")

tab_analyze, tab_dashboard, tab_history, tab_about = st.tabs(
    ["🔬 Analyze", "📊 Dashboard", "📋 Session history", "ℹ️ About"]
)

# =========================================================================== #
# Analyze
# =========================================================================== #
with tab_analyze:
    files = st.file_uploader(
        "Chest X-ray images (JPEG, PNG or DICOM)",
        type=["jpg", "jpeg", "png", "dcm", "dicom"],
        accept_multiple_files=True,
    )

    if "demo_payload" not in st.session_state:
        st.session_state.demo_payload = None

    # A new upload replaces any sample that is currently shown.
    _sig = tuple((f.name, f.size) for f in files) if files else ()
    if _sig != st.session_state.get("upload_sig", ()):
        st.session_state.upload_sig = _sig
        if files:
            st.session_state.demo_payload = None

    st.markdown(
        '<p class="note">💡 <b>Quick demo:</b> no X-ray at hand? Try one of these sample images.</p>',
        unsafe_allow_html=True,
    )
    d1, d2, d3, _ = st.columns([1.1, 1.6, 1.4, 0.9])
    for col, label, sample_name, btn_key in [
        (d1, "🧪 Sample Normal", "sample_normal.jpg", "demo_normal"),
        (d2, "🧪 Sample Bacterial Pneumonia", "sample_bacterial_pneumonia.jpg", "demo_bacterial"),
        (d3, "🧪 Sample Viral Pneumonia", "sample_viral_pneumonia.jpg", "demo_viral"),
    ]:
        if col.button(label, key=btn_key):
            sample_path = SAMPLE_DIR / sample_name
            if sample_path.exists():
                st.session_state.demo_payload = (sample_name, sample_path.read_bytes())
            else:
                st.session_state.demo_payload = None
                st.warning(f"Sample image `{sample_name}` was not found in the `sample_images` folder.")

    inputs = []  # list of (file name, file bytes)
    if st.session_state.demo_payload:
        inputs = [st.session_state.demo_payload]
        st.caption(f"Showing sample image: {st.session_state.demo_payload[0]}")
        if st.button("✖ Clear sample", key="clear_demo"):
            st.session_state.demo_payload = None
            st.rerun()
    elif files:
        inputs = [(f.name, f.getvalue()) for f in files]

    if not inputs:
        st.markdown(
            '<div class="empty">📤 No image yet. Drop one or more chest X-rays above, '
            "or click a sample button, to get a prediction and a heatmap.</div>",
            unsafe_allow_html=True,
        )
    else:
        results = {}
        for name, data in inputs:
            dicom_meta = None
            model_bytes = data
            if is_dicom_name(name):
                if not _HAS_DICOM:
                    st.error(
                        f"Cannot open **{name}**: DICOM support is not installed "
                        f"({_FEATURE_ERRORS.get('dicom', 'pydicom missing')})."
                    )
                    continue
                try:
                    dicom_img, dicom_meta = read_dicom(data)
                    _buf = io.BytesIO()
                    dicom_img.save(_buf, format="PNG")
                    model_bytes = _buf.getvalue()
                except Exception as exc:
                    st.error(f"Could not read DICOM file {name}: {exc}")
                    continue
            try:
                res = dict(analyze(model, model_bytes))
            except Exception as exc:
                st.error(f"Could not read {name}: {exc}")
                continue

            res["severity"] = (
                estimate_severity(res["cam"]) if (_HAS_SEVERITY and res["idx"] == 1) else None
            )
            res["dicom_meta"] = dicom_meta
            results[name] = res

            key = (name, len(data))
            if key not in st.session_state.history:
                sev = res["severity"]
                st.session_state.history[key] = {
                    "Time": datetime.now().strftime("%H:%M:%S"),
                    "File": name,
                    "Prediction": LABELS[res["idx"]],
                    "Normal probability": round(float(res["share"][0]), 4),
                    "Pneumonia probability": round(float(res["share"][1]), 4),
                    "Severity": sev["severity_level"] if sev else "N/A",
                    "Affected area %": round(sev["affected_percentage"], 1) if sev else 0.0,
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
                st.image(shown, width="stretch", caption=f"{chosen} (resized to 224×224)")
                if res.get("dicom_meta"):
                    with st.expander("📋 DICOM metadata"):
                        for k, v in res["dicom_meta"].items():
                            st.markdown(f"**{k}:** {v}")

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

                # Confidence breakdown
                st.markdown(
                    '<div class="card"><p class="card-title">📊 Confidence Breakdown</p>'
                    + score_bar("Normal", float(res["share"][0]), "#4DB6A0")
                    + score_bar("Pneumonia", float(res["share"][1]), "#EF6F6C")
                    + "</div>",
                    unsafe_allow_html=True,
                )

                # Severity assessment
                sev = res.get("severity")
                if is_flag and sev:
                    level = sev["severity_level"]
                    badge = {"Mild": "sev-mild", "Moderate": "sev-moderate", "Severe": "sev-severe"}[level]
                    cells = "".join(
                        region_cell(n, sev["affected_regions"][n])
                        for n in ["Upper left", "Upper right", "Central", "Lower left", "Lower right"]
                    )
                    st.markdown(
                        '<div class="card"><p class="card-title">🩺 Severity Assessment</p>'
                        f'<span class="sev-badge {badge}">{level}</span>'
                        f'<span style="margin-left:0.7rem;">{sev["affected_percentage"]:.1f}% of the image highlighted</span>'
                        f'<div class="region-row">{cells}</div>'
                        f'<p class="note" style="margin:0.5rem 0 0;">{sev["clinical_note"]} '
                        "Regions refer to the picture as displayed.</p></div>",
                        unsafe_allow_html=True,
                    )
                elif is_flag:
                    st.markdown(
                        '<div class="card"><p class="card-title">🩺 Severity Assessment</p>'
                        '<p class="note" style="margin:0;">Severity estimation is unavailable in this deployment.</p></div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        '<div class="card"><p class="card-title">🩺 Severity Assessment</p>'
                        '<p class="note" style="margin:0;">Not applicable: the model did not predict pneumonia.</p></div>',
                        unsafe_allow_html=True,
                    )

                st.markdown(
                    '<p class="note">Percentages are softmax probabilities of the two model outputs. '
                    "A result close to 50/50 means the model is unsure. Bright areas on the "
                    "heatmap are where the model looked most for its predicted class.</p>",
                    unsafe_allow_html=True,
                )

                # PDF report
                if _HAS_REPORT:
                    try:
                        pdf_bytes = generate_pdf_report(
                            original_image=Image.fromarray(res["crop"]),
                            overlay_image=Image.fromarray(blended),
                            prediction=LABELS[res["idx"]],
                            probabilities={
                                "Normal": float(res["share"][0]),
                                "Pneumonia": float(res["share"][1]),
                            },
                            severity=res.get("severity"),
                            patient_info=patient_info(),
                            filename=chosen,
                            dicom_meta=res.get("dicom_meta"),
                        )
                        stem = re.sub(r"[^A-Za-z0-9_-]+", "_", chosen.rsplit(".", 1)[0]) or "report"
                        st.download_button(
                            "📄 Download PDF report",
                            data=pdf_bytes,
                            file_name=f"xray_report_{stem}.pdf",
                            mime="application/pdf",
                        )
                        if not patient_info()["Patient name"]:
                            st.caption("Tip: save patient details in the sidebar to include them in the report.")
                    except Exception as exc:
                        st.warning(f"The PDF report could not be created: {exc}")
                else:
                    st.info(
                        "PDF export is unavailable because the `reportlab` package is not installed "
                        f"({_FEATURE_ERRORS.get('report', '')})."
                    )

            if len(results) > 1:
                st.subheader("📋 Batch summary")
                rows = [
                    {
                        "File": n,
                        "Prediction": LABELS[r["idx"]],
                        "Normal": f"{r['share'][0] * 100:.1f}%",
                        "Pneumonia": f"{r['share'][1] * 100:.1f}%",
                        "Severity": r["severity"]["severity_level"] if r.get("severity") else "—",
                    }
                    for n, r in results.items()
                ]
                st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

# =========================================================================== #
# Dashboard
# =========================================================================== #
with tab_dashboard:
    dash = list(st.session_state.history.values())
    if not dash:
        st.markdown(
            '<div class="empty">📊 Analyze X-rays in the Analyze tab and this dashboard '
            "will fill with statistics for the current session.</div>",
            unsafe_allow_html=True,
        )
    else:
        total = len(dash)
        n_normal = sum(1 for h in dash if h["Prediction"] == "Normal")
        n_pneu = total - n_normal
        top_conf = [max(h["Normal probability"], h["Pneumonia probability"]) * 100 for h in dash]

        cards = [
            ("Total analyzed", total, "var(--blue)"),
            ("Normal", n_normal, "var(--clear)"),
            ("Pneumonia", n_pneu, "var(--flag)"),
            ("Avg confidence", f"{np.mean(top_conf):.1f}%", "var(--warn)"),
        ]
        for col, (lbl, value, color) in zip(st.columns(4), cards):
            col.markdown(
                f'<div class="metric-card"><p class="mc-val" style="color:{color}">{value}</p>'
                f'<p class="mc-lbl">{lbl}</p></div>',
                unsafe_allow_html=True,
            )
        st.write("")

        c_left, c_right = st.columns(2, gap="large")
        with c_left:
            st.markdown('<p class="card-title">📊 Prediction distribution</p>', unsafe_allow_html=True)
            st.bar_chart(
                pd.DataFrame({"Images": [n_normal, n_pneu]}, index=["Normal", "Pneumonia"]),
                color="#7FB3D5",
            )
        with c_right:
            st.markdown('<p class="card-title">📈 Confidence per image (%)</p>', unsafe_allow_html=True)
            st.bar_chart(
                pd.DataFrame(
                    {"Confidence %": top_conf},
                    index=[f"{i + 1:02d}. {h['File']}" for i, h in enumerate(dash)],
                ),
                color="#4DB6A0",
            )

        sev_counts = {"Mild": 0, "Moderate": 0, "Severe": 0}
        for h in dash:
            if h.get("Severity") in sev_counts:
                sev_counts[h["Severity"]] += 1
        if any(sev_counts.values()):
            st.markdown('<p class="card-title">🩺 Severity distribution (pneumonia cases)</p>', unsafe_allow_html=True)
            st.bar_chart(
                pd.DataFrame({"Cases": list(sev_counts.values())}, index=list(sev_counts.keys())),
                color="#E8B04B",
            )

        st.markdown('<p class="card-title">🕒 Session timeline</p>', unsafe_allow_html=True)
        rows_html = ""
        for h, conf in zip(dash, top_conf):
            dot = "🟢" if h["Prediction"] == "Normal" else "🔴"
            sev_txt = f" · {h['Severity']}" if h.get("Severity") not in (None, "N/A") else ""
            rows_html += (
                f'<div class="timeline-row"><b>{h["Time"]}</b> &nbsp; {dot} <b>{h["Prediction"]}</b> '
                f'({conf:.1f}%){sev_txt} &nbsp;·&nbsp; <span class="note">{h["File"]}</span></div>'
            )
        st.markdown(rows_html, unsafe_allow_html=True)

# =========================================================================== #
# Session history
# =========================================================================== #
with tab_history:
    hist = list(st.session_state.history.values())
    if not hist:
        st.markdown(
            '<div class="empty">📋 Predictions you run in this session will be listed here.</div>',
            unsafe_allow_html=True,
        )
    else:
        df = pd.DataFrame(hist)
        counts = df["Prediction"].value_counts()
        c1, c2, c3 = st.columns(3)
        c1.metric("Images analyzed", len(df))
        c2.metric("Normal", int(counts.get("Normal", 0)))
        c3.metric("Pneumonia", int(counts.get("Pneumonia", 0)))
        st.dataframe(df, hide_index=True, width="stretch")
        b1, b2, _ = st.columns([1, 1, 3])
        b1.download_button(
            "📥 Download CSV", df.to_csv(index=False).encode(), "xray_predictions.csv", "text/csv"
        )
        if b2.button("🗑️ Clear history"):
            st.session_state.history = {}
            st.rerun()

# =========================================================================== #
# About
# =========================================================================== #
with tab_about:
    n_params = sum(p.numel() for p in model.parameters())

    st.subheader("Purpose")
    st.markdown(
        """
This application is a **research prototype for chest X-ray screening**. For each image it tells you whether
the X-ray looks **Normal** or shows signs of **Pneumonia**, highlights the regions that drove that decision,
estimates how widespread the highlighted area is, and can export the result as a PDF report.

It is intended for learning, demonstration and as a base for further research. It is **not a diagnostic tool**.
"""
    )

    st.subheader("How to use it")
    st.markdown(
        """
1. *(Optional)* Enter patient details in the sidebar and press **Save patient details**.
2. Upload one or more X-rays (**JPEG, PNG or DICOM**) or click one of the **sample** buttons.
3. Read the **prediction**, the **confidence breakdown** and the **severity assessment**. Switch between the
   original image, the heatmap overlay and the heatmap alone, and adjust the heatmap in **Display Settings**.
4. Download the **PDF report** for the selected image.
5. Use the **Dashboard** and **Session history** tabs to review everything analyzed in the current session.
"""
    )

    st.subheader("How an image is processed")
    st.markdown(
        """
1. **Load.** JPEG and PNG files are opened directly. DICOM files are decoded, scaled to 8-bit using the window
   settings stored in the file, and inverted if they use the MONOCHROME1 convention.
2. **Prepare.** The image is converted to RGB, resized to 224 pixels on the short side, center-cropped to 224×224
   and normalized with ImageNet mean and standard deviation, exactly as in training.
3. **Predict.** The CNN returns two scores (Normal, Pneumonia). A softmax turns them into probabilities that add up to 100%.
4. **Explain.** Grad-CAM produces a heatmap of the regions that pushed the model toward its answer.
5. **Estimate severity** (pneumonia predictions only) from that heatmap.
6. **Report.** The results can be exported as a one-page PDF.
"""
    )

    st.subheader("The model")
    st.markdown(
        f"""
| | |
|---|---|
| Type | Custom convolutional neural network built with PyTorch |
| Structure | 9 convolution blocks (ReLU + batch normalization), 3 max-pooling stages, global average pooling and a final 4×4 convolution |
| Parameters | {n_params:,} (small enough to run on a CPU) |
| Input | RGB image, 224×224 |
| Output | Two scores (Normal, Pneumonia), converted to probabilities with softmax |
"""
    )

    st.subheader("Training and results")
    st.markdown(
        """
**Data.** 5,856 chest X-rays labelled Normal or Pneumonia, split into 3,982 for training, 702 for validation
(15% held out from the training folder) and 1,172 for testing. The training set contains about three times more
Pneumonia than Normal images, so class weights were used to keep the model from favouring Pneumonia.

**Training.** Cross-entropy loss with class weights, SGD optimizer (learning rate 0.01, momentum 0.9) with a step
learning-rate schedule, batch size 32 and 12 epochs. Training images were augmented with small colour changes,
horizontal flips and rotations of up to 10°. The epoch with the best balanced accuracy on the validation set was kept,
and the test set was used once at the end.

**Test results (1,172 images the model never saw during training):**

| Metric | Value |
|---|---|
| Accuracy | 95.14% |
| Balanced accuracy | 95.28% |
| Recall, Normal | 95.6% (303 of 317 correct) |
| Recall, Pneumonia | 95.0% (812 of 855 correct) |

| | Predicted Normal | Predicted Pneumonia |
|---|---|---|
| **Actually Normal** | 303 | 14 |
| **Actually Pneumonia** | 43 | 812 |

The 43 pneumonia images predicted as Normal are the most serious kind of error: **a Normal result does not rule out pneumonia.**
"""
    )

    st.subheader("Heatmap and severity estimate")
    st.markdown(
        """
**Grad-CAM heatmap.** The gradient of the predicted class score is traced back to the last convolution block.
Its channels are weighted by their influence and combined into a 19×19 map that is scaled up to the image.
Bright areas are where the model looked most. This shows what influenced the model, which is not proof of where disease is.

**Severity assessment.** For pneumonia predictions, the app measures how much of the image the heatmap highlights
(areas at 50% or more of the heatmap's peak):

| Highlighted area | Level |
|---|---|
| Below 20% | Mild |
| 20% to 40% | Moderate |
| 40% or more | Severe |

It also reports the highlighted share in five regions (upper left, upper right, central, lower left, lower right of the picture).
This is a simple heuristic based on the model's attention, **not a clinical severity grade**.
"""
    )

    st.subheader("PDF report")
    st.markdown(
        """
The one-page report contains the patient details you saved, the image name, the prediction with its confidence
breakdown, the severity assessment, the original image next to the Grad-CAM overlay, DICOM details when
the upload was a DICOM file, and the disclaimer.
"""
    )

    st.subheader("Limitations and responsible use")
    st.markdown(
        """
- **Not for clinical use.** The app is not a certified medical device and must not be used to diagnose or rule out pneumonia.
  A qualified radiologist or physician must review every X-ray.
- **One dataset.** The model was trained and tested on a single dataset and has not been validated on other hospitals,
  scanners or patient groups (for example, a different age range). Accuracy elsewhere may be lower.
- **Only two answers.** The model chooses between Normal and Pneumonia. It cannot detect other conditions, and it does not
  check whether an uploaded image is really a chest X-ray.
- **Missed cases.** About 5% of pneumonia images in the test set were predicted as Normal.
- **Estimates only.** Heatmaps and severity levels explain the model's behaviour and are not measurements of disease.
- **Image handling.** Images are cropped to the centre and reduced to 224×224, so fine detail and image edges can be lost.
- **Privacy.** The app does not save uploaded images; they are processed in memory during your session. Do not upload
  identifiable patient data to a public deployment.
"""
    )

    st.subheader("Built with")
    st.markdown("Python · PyTorch · torchvision · Streamlit · Pillow · NumPy · pandas · pydicom · ReportLab")
