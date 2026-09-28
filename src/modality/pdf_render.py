"""Deterministic PDF renderer: structured_payload dict -> PDF file.

`payload` schema (produced by the STRATA attacker):
{
    "section_titles": ["Background", "Method", ...],
    "section_bodies":  ["body text 1", "body text 2", ...],
    "title": "optional document title"
}

The renderer is fully deterministic: same payload -> identical PDF bytes (modulo
reportlab's timestamp metadata, which we suppress).
"""
from pathlib import Path


def render_pdf(payload: dict, out_path: str) -> str:
    """Render the payload to a PDF and return the output path."""
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.enums import TA_LEFT

    out_path = str(out_path)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        out_path,
        pagesize=letter,
        rightMargin=0.75 * inch,
        leftMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
        title="",
        author="",
        creator="",
        producer="",
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "TitleS", parent=styles["Title"], fontSize=16, alignment=TA_LEFT, spaceAfter=12
    )
    h_style = ParagraphStyle(
        "HS", parent=styles["Heading2"], fontSize=12, spaceBefore=10, spaceAfter=6
    )
    body_style = ParagraphStyle(
        "BS", parent=styles["BodyText"], fontSize=10.5, leading=14, spaceAfter=8
    )

    story = []
    doc_title = payload.get("title")
    if doc_title:
        story.append(Paragraph(_escape(doc_title), title_style))
        story.append(Spacer(1, 0.1 * inch))

    titles = payload.get("section_titles", [])
    bodies = payload.get("section_bodies", [])
    if len(titles) != len(bodies):
        # Truncate to the shorter length; deterministic.
        n = min(len(titles), len(bodies))
        titles, bodies = titles[:n], bodies[:n]

    for title, body in zip(titles, bodies):
        story.append(Paragraph(_escape(title), h_style))
        # Split body into paragraphs by double-newline
        for para in body.split("\n\n"):
            para = para.strip()
            if para:
                story.append(Paragraph(_escape(para), body_style))

    doc.build(story)
    return out_path


def payload_to_text(payload: dict) -> str:
    """Flatten the payload to a plain text string — used for the copy-penalty
    similarity check and for the text-only counterfactual ablation."""
    parts = []
    if payload.get("title"):
        parts.append(payload["title"])
    titles = payload.get("section_titles", [])
    bodies = payload.get("section_bodies", [])
    for t, b in zip(titles, bodies):
        parts.append(t)
        parts.append(b)
    return "\n\n".join(parts)


def _escape(s: str) -> str:
    """Minimal escaping for reportlab Paragraph parser."""
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\r\n", "\n")
    )
