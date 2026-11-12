"""One-page PDF report for a chest X-ray analysis (built with reportlab)."""
from __future__ import annotations

import io
from datetime import datetime
from xml.sax.saxutils import escape

from PIL import Image
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    HRFlowable,
    Image as RLImage,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

NAVY = colors.HexColor("#0F1B24")
SLATE = colors.HexColor("#5B6F7C")
LINE = colors.HexColor("#C9D4DB")
SOFT = colors.HexColor("#F3F6F8")
GREEN = colors.HexColor("#1E8E78")
RED = colors.HexColor("#D9534F")
AMBER = colors.HexColor("#C98A1B")

_REPLACE = {
    "\u2014": "-", "\u2013": "-", "\u2018": "'", "\u2019": "'", "\u201c": '"',
    "\u201d": '"', "\u2022": "-", "\u2026": "...", "\u00a0": " ",
}


def _clean(value) -> str:
    """Make text safe for the built-in PDF font (Latin-1 only) and for XML markup."""
    text = "" if value is None else str(value)
    for old, new in _REPLACE.items():
        text = text.replace(old, new)
    text = text.encode("latin-1", "replace").decode("latin-1")
    return escape(text.strip())


def _style(name, **kw) -> ParagraphStyle:
    base = dict(fontName="Helvetica", fontSize=9.5, leading=12.5, textColor=NAVY)
    base.update(kw)
    return ParagraphStyle(name, **base)


S_TITLE = _style("title", fontName="Helvetica-Bold", fontSize=19, leading=23)
S_SMALL = _style("small", fontSize=8, leading=10.5, textColor=SLATE)
S_SMALL_R = _style("smallr", fontSize=8, leading=10.5, textColor=SLATE, alignment=2)
S_H = _style("h", fontName="Helvetica-Bold", fontSize=11, leading=14, textColor=NAVY, spaceBefore=7, spaceAfter=3)
S_LABEL = _style("label", fontName="Helvetica-Bold", fontSize=8.5, textColor=SLATE)
S_VALUE = _style("value", fontSize=9.5)
S_NOTE = _style("note", fontSize=8, leading=10.5, textColor=SLATE)
S_CAP = _style("cap", fontSize=8, leading=10, textColor=SLATE, alignment=1)
S_CELL_C = _style("cellc", fontSize=8.5, leading=11, alignment=1)


def _png(img: Image.Image) -> io.BytesIO:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    buf.seek(0)
    return buf


def _footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(SLATE)
    canvas.drawString(16 * mm, 9 * mm, "Research prototype - not a medical device. Results must be reviewed by a qualified clinician.")
    canvas.drawRightString(A4[0] - 16 * mm, 9 * mm, f"Page {doc.page}")
    canvas.restoreState()


def generate_pdf_report(
    *,
    original_image: Image.Image,
    overlay_image: Image.Image,
    prediction: str,
    probabilities: dict,
    severity: dict | None,
    patient_info: dict | None,
    filename: str,
    dicom_meta: dict | None = None,
) -> bytes:
    """Build the report and return the PDF as bytes.

    patient_info keys: Patient name, Patient ID, Age, Sex, Referring physician
    (blank values are shown as "Not provided").
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=16 * mm, rightMargin=16 * mm, topMargin=13 * mm, bottomMargin=16 * mm,
        title="Chest X-ray Analysis Report", author="Chest X-ray Analysis",
    )
    width = A4[0] - 32 * mm
    story = []

    # ---- header -------------------------------------------------------------
    now = datetime.now().strftime("%d %b %Y, %H:%M")
    head = Table(
        [[Paragraph("Chest X-ray Analysis Report", S_TITLE),
          Paragraph(f"Generated<br/>{_clean(now)}", S_SMALL_R)]],
        colWidths=[width * 0.72, width * 0.28],
    )
    head.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "BOTTOM"),
                              ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0)]))
    story += [head, Spacer(1, 3), HRFlowable(width="100%", thickness=1.6, color=NAVY), Spacer(1, 4)]

    # ---- patient information -----------------------------------------------
    pi = patient_info or {}

    def val(key):
        return _clean(pi.get(key)) or "Not provided"

    story.append(Paragraph("Patient information", S_H))
    rows = [
        [Paragraph("Patient name", S_LABEL), Paragraph(val("Patient name"), S_VALUE),
         Paragraph("Patient ID", S_LABEL), Paragraph(val("Patient ID"), S_VALUE)],
        [Paragraph("Age", S_LABEL), Paragraph(val("Age"), S_VALUE),
         Paragraph("Sex", S_LABEL), Paragraph(val("Sex"), S_VALUE)],
        [Paragraph("Referring physician", S_LABEL), Paragraph(val("Referring physician"), S_VALUE),
         Paragraph("Image file", S_LABEL), Paragraph(_clean(filename), S_VALUE)],
    ]
    t = Table(rows, colWidths=[width * 0.20, width * 0.30, width * 0.17, width * 0.33])
    t.setStyle(TableStyle([
        ("BOX", (0, 0), (-1, -1), 0.6, LINE), ("INNERGRID", (0, 0), (-1, -1), 0.4, LINE),
        ("BACKGROUND", (0, 0), (0, -1), SOFT), ("BACKGROUND", (2, 0), (2, -1), SOFT),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(t)

    # ---- result -------------------------------------------------------------
    is_pneu = prediction == "Pneumonia"
    colour = RED if is_pneu else GREEN
    p_norm = float(probabilities.get("Normal", 0.0)) * 100
    p_pneu = float(probabilities.get("Pneumonia", 0.0)) * 100
    top = max(p_norm, p_pneu)

    story.append(Paragraph("Model result", S_H))
    hexcol = colour.hexval().replace("0x", "#")
    verdict = [
        Paragraph("PREDICTION", _style("v1", fontSize=8, leading=10, textColor=SLATE)),
        Paragraph(f'<font color="{hexcol}">{_clean(prediction)}</font>',
                  _style("v2", fontName="Helvetica-Bold", fontSize=20, leading=25)),
        Paragraph(f"{top:.1f}% model probability", _style("v3", fontSize=9, leading=12)),
    ]
    probs = Table(
        [[Paragraph("Confidence breakdown", S_LABEL), ""],
         [Paragraph("Normal", S_VALUE), Paragraph(f"{p_norm:.1f}%", S_VALUE)],
         [Paragraph("Pneumonia", S_VALUE), Paragraph(f"{p_pneu:.1f}%", S_VALUE)]],
        colWidths=[width * 0.28, width * 0.14],
    )
    probs.setStyle(TableStyle([("SPAN", (0, 0), (1, 0)), ("LINEBELOW", (0, 0), (-1, 0), 0.4, LINE),
                               ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                               ("LEFTPADDING", (0, 0), (-1, -1), 4)]))
    res = Table([[verdict, probs]], colWidths=[width * 0.50, width * 0.50])
    res.setStyle(TableStyle([
        ("BOX", (0, 0), (-1, -1), 0.6, LINE), ("LINEBEFORE", (0, 0), (0, -1), 3.5, colour),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (0, 0), 10),
    ]))
    story.append(res)

    # ---- severity -----------------------------------------------------------
    story.append(Paragraph("Severity assessment", S_H))
    if is_pneu and severity:
        level = severity.get("severity_level", "")
        lvl_col = {"Mild": GREEN, "Moderate": AMBER, "Severe": RED}.get(level, AMBER)
        story.append(Paragraph(
            f'<font name="Helvetica-Bold" color="{lvl_col.hexval().replace("0x", "#")}">{_clean(level)}</font>'
            f' &nbsp;-&nbsp; {severity.get("affected_percentage", 0):.1f}% of the image highlighted by the model', S_VALUE))
        regs = severity.get("affected_regions", {})
        names = ["Upper left", "Upper right", "Lower left", "Lower right", "Central"]
        grid = Table(
            [[Paragraph(n, S_CELL_C) for n in names],
             [Paragraph(f"<b>{regs.get(n, 0):.0f}%</b>", S_CELL_C) for n in names]],
            colWidths=[width / 5] * 5,
        )
        grid.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.6, LINE), ("INNERGRID", (0, 0), (-1, -1), 0.4, LINE),
                                  ("BACKGROUND", (0, 0), (-1, 0), SOFT),
                                  ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]))
        story += [Spacer(1, 3), grid, Spacer(1, 2), Paragraph(
            "Region names refer to the picture as displayed. Severity is estimated from the model's heatmap "
            "(share of the image it focused on). It is a heuristic, not a clinical severity grade.", S_NOTE)]
    else:
        story.append(Paragraph("Not applicable - the model did not predict pneumonia for this image.", S_VALUE))

    # ---- images -------------------------------------------------------------
    story.append(Paragraph("Images", S_H))
    size = 68 * mm
    imgs = Table(
        [[RLImage(_png(original_image), width=size, height=size), RLImage(_png(overlay_image), width=size, height=size)],
         [Paragraph("Chest X-ray (224x224)", S_CAP), Paragraph("Grad-CAM overlay (areas the model focused on)", S_CAP)]],
        colWidths=[width / 2, width / 2],
    )
    imgs.setStyle(TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER"), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                              ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]))
    story.append(imgs)

    # ---- DICOM metadata (only for DICOM uploads) ----------------------------
    if dicom_meta:
        story.append(Paragraph("DICOM details", S_H))
        items = list(dicom_meta.items())[:8]
        meta_rows = []
        for i in range(0, len(items), 2):
            a = items[i]
            b = items[i + 1] if i + 1 < len(items) else ("", "")
            meta_rows.append([Paragraph(_clean(a[0]), S_LABEL), Paragraph(_clean(a[1]), S_VALUE),
                              Paragraph(_clean(b[0]), S_LABEL), Paragraph(_clean(b[1]), S_VALUE)])
        mt = Table(meta_rows, colWidths=[width * 0.2, width * 0.3, width * 0.2, width * 0.3])
        mt.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.6, LINE), ("INNERGRID", (0, 0), (-1, -1), 0.4, LINE),
                                ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]))
        story.append(mt)

    # ---- method + disclaimer ------------------------------------------------
    story.append(Spacer(1, 5))
    story.append(Paragraph(
        "Method: custom 9-block convolutional neural network (PyTorch). The image is resized and center-cropped "
        "to 224x224; the heatmap is computed with Grad-CAM from the last convolution block.", S_NOTE))
    story.append(Spacer(1, 4))
    disc = Table([[Paragraph(
        "<b>Disclaimer:</b> This report was produced by an AI research prototype. It is not a certified medical "
        "device and must not be used to diagnose or rule out pneumonia. All findings must be reviewed and confirmed "
        "by a qualified radiologist or physician.", _style("disc", fontSize=8.5, leading=11))]], colWidths=[width])
    disc.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.6, RED), ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FDF1F0")),
                              ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                              ("LEFTPADDING", (0, 0), (-1, -1), 8), ("RIGHTPADDING", (0, 0), (-1, -1), 8)]))
    story.append(disc)

    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    return buf.getvalue()
