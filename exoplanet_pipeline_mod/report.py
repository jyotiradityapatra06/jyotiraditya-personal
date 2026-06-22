import os


def generate_report(results: list, summary_img: str, out_path: str):
    """Generate a 3-page PDF report using ReportLab."""
    try:
        import reportlab
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.platypus import (
            SimpleDocTemplate,
            Paragraph,
            Spacer,
            Table,
            TableStyle,
            Image as RLImage,
            PageBreak,
            HRFlowable,
        )
        from reportlab.lib.units import cm
        from reportlab.lib.enums import TA_CENTER
    except ImportError:
        print("  [report] ReportLab not available, skipping PDF.")
        return

    PALETTE = {

        "bg": "#0d0d0d",
        "accent": "#f0a500",
        "text": "#e8e8e8",
        "lb": "#00d4ff",
    }

    ORANGE = colors.HexColor(PALETTE["accent"])
    LBLUE = colors.HexColor(PALETTE["lb"])

    W, H = A4
    doc = SimpleDocTemplate(
        out_path,
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )

    styles = getSampleStyleSheet()

    def style(name, **kw):
        return ParagraphStyle(name, parent=styles["Normal"], **kw)

    title_style = style(
        "Title2",
        fontSize=18,
        textColor=ORANGE,
        spaceAfter=6,
        alignment=TA_CENTER,
        fontName="Helvetica-Bold",
    )
    h1_style = style(
        "H1",
        fontSize=13,
        textColor=ORANGE,
        spaceAfter=4,
        fontName="Helvetica-Bold",
    )
    h2_style = style(
        "H2",
        fontSize=10,
        textColor=LBLUE,
        spaceAfter=3,
        fontName="Helvetica-Bold",
    )
    body_style = style(
        "Body", fontSize=9, textColor=colors.black, leading=13, spaceAfter=4
    )
    small_style = style("Small", fontSize=8, textColor=colors.grey)

    story = []

    story.append(Spacer(1, 0.5 * cm))
    story.append(Paragraph("AI-Enabled Exoplanet Detection Pipeline", title_style))
    story.append(
        Paragraph(
            "Problem Statement 7 — Hackathon Submission",
            style("sub", fontSize=10, textColor=colors.grey, alignment=TA_CENTER),
        )
    )
    story.append(Spacer(1, 0.4 * cm))
    story.append(HRFlowable(width="100%", color=ORANGE, thickness=1.5))
    story.append(Spacer(1, 0.3 * cm))

    story.append(Paragraph("1. Methodology", h1_style))
    story.append(
        Paragraph(
            "<b>Data Acquisition:</b> TESS 2-minute cadence light curves are retrieved from the MAST "
            "archive (archive.stsci.edu/tess) via the public HTTP API. Each light curve contains "
            "~18 000 flux measurements spanning ~27 days.",
            body_style,
        )
    )
    story.append(
        Paragraph(
            "<b>Preprocessing:</b> (i) Normalisation by median flux. (ii) Iterative 4-σ sigma clipping "
            "to remove cosmic rays and instrumental artefacts. (iii) Savitzky-Golay smoothing (window "
            "= 3 hr) to remove stellar variability, producing a flat, detrended flux series.",
            body_style,
        )
    )

    story.append(
        Paragraph(
            "<b>Transit Search — BLS Periodogram:</b> A Box Least Squares (BLS) algorithm scans periods "
            "from 0.5 to 15 days and transit durations from 1% to 15% of the period. The BLS power "
            "spectrum peaks at the most likely orbital period.",
            body_style,
        )
    )

    story.append(
        Paragraph(
            "<b>Classification:</b> Feature-based rule system assigns probability scores to five classes: "
            "<i>transit, eclipse, blend, starspot, noise</i>.",
            body_style,
        )
    )

    story.append(PageBreak())

    story.append(Paragraph("3. Results Summary", h1_style))
    story.append(Spacer(1, 0.2 * cm))

    hdr = ["Target", "Class", "Conf.", "Period (d)", "Depth (ppm)", "Duration (hr)", "SNR (σ)"]
    rows = [hdr]
    for res in results:
        bls = res["bls"]
        clf = res["clf"]
        fit = res["fit"]
        dep_ppm = fit.get("depth_ppm", round(bls.get("depth", 0) * 1e6, 0))
        rows.append(
            [
                res["name"],
                clf["label"].upper(),
                f"{clf['confidence']:.1%}",
                f"{bls.get('period', 0):.4f}",
                f"{dep_ppm:.0f}",
                f"{fit.get('duration_hrs', bls.get('duration', 0) * 24):.2f}",
                f"{bls.get('snr', 0):.1f}",
            ]
        )

    col_w = [4.5 * cm, 2 * cm, 1.8 * cm, 2.4 * cm, 2.4 * cm, 2.5 * cm, 1.8 * cm]
    rt = Table(rows, colWidths=col_w)
    rt.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), ORANGE),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )
    story.append(rt)

    story.append(Spacer(1, 0.5 * cm))
    if os.path.exists(summary_img):
        img_w = 16.5 * cm
        img_h = img_w * 0.65
        story.append(RLImage(summary_img, width=img_w, height=img_h))

    story.append(PageBreak())

    story.append(Paragraph("4. Detailed Parameter Estimates", h1_style))
    for res in results:
        story.append(Spacer(1, 0.25 * cm))
        story.append(Paragraph(res["name"], h2_style))

        fit = res["fit"]
        clf = res["clf"]
        bls = res["bls"]

        param_rows = [
            ["Parameter", "Value", "Unit"],
            ["Classification", clf["label"].upper(), ""],
            ["Confidence", f"{clf['confidence']:.1%}", ""],
            ["Orbital Period", f"{bls.get('period', 0):.6f}", "days"],
            ["Transit Epoch t₀", f"{bls.get('t0', 0):.4f}", "BJD"],
            ["Transit Depth", f"{fit.get('depth_ppm', bls.get('depth',0)*1e6):.0f}", "ppm"],
            [
                "Transit Duration",
                f"{fit.get('duration_hrs', bls.get('duration', 0)*24):.3f}",
                "hrs",
            ],
            ["SNR", f"{bls.get('snr', 0):.1f}", "σ"],
            ["Fit Status", fit.get("status", "N/A"), ""],
        ]

        pt = Table(param_rows, colWidths=[6 * cm, 5 * cm, 4 * cm])
        pt.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#333333")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                    ("GRID", (0, 0), (-1, -1), 0.4, colors.lightgrey),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("TOPPADDING", (0, 0), (-1, -1), 3),
                ]
            )
        )
        story.append(pt)

    doc.build(story)
    print(f"  [report] PDF saved {out_path}")

