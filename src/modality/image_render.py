"""Deterministic image renderer: structured_payload -> PNG.

Renders a single A4-aspect image with the document's title and section blocks laid out
top-to-bottom. The output is a fixed-resolution PNG so the same payload yields the
same pixel bytes.
"""
import os
from pathlib import Path


def render_image(payload: dict, out_path: str) -> str:
    from PIL import Image, ImageDraw, ImageFont

    out_path = str(out_path)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    W, H = 1240, 1754  # 150 DPI A4-ish
    PAD = 80
    img = Image.new("RGB", (W, H), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Try multiple font candidates so this works on Mac, Linux clusters, and Rocky/EPYC nodes.
    title_font = _load_font(28)
    header_font = _load_font(20)
    body_font = _load_font(15)

    y = PAD
    title = payload.get("title", "")
    if title:
        draw.text((PAD, y), title, fill=(0, 0, 0), font=title_font)
        y += 50
        # underline
        draw.line([(PAD, y), (W - PAD, y)], fill=(60, 60, 60), width=2)
        y += 20

    titles = payload.get("section_titles", [])
    bodies = payload.get("section_bodies", [])
    n = min(len(titles), len(bodies))
    for i in range(n):
        if y > H - PAD - 60:
            break  # ran out of vertical room
        draw.text((PAD, y), titles[i], fill=(0, 0, 0), font=header_font)
        y += 32
        wrapped = _wrap(bodies[i], body_font, W - 2 * PAD, draw)
        for line in wrapped:
            if y > H - PAD - 30:
                break
            draw.text((PAD, y), line, fill=(20, 20, 20), font=body_font)
            y += 22
        y += 16

    img.save(out_path, format="PNG", optimize=False)
    return out_path


def _load_font(size: int):
    """Pick an available TrueType font; fall back to PIL's default bitmap font."""
    from PIL import ImageFont
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Rocky/Debian
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/liberation/LiberationSans-Regular.ttf",
        "/Library/Fonts/Arial.ttf",                          # macOS
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


def _wrap(text: str, font, max_width: int, draw) -> list:
    """Greedy word-wrap to fit max_width pixels."""
    lines = []
    for para in text.split("\n"):
        words = para.split()
        cur = ""
        for w in words:
            trial = (cur + " " + w).strip()
            bbox = draw.textbbox((0, 0), trial, font=font)
            width = bbox[2] - bbox[0]
            if width <= max_width:
                cur = trial
            else:
                if cur:
                    lines.append(cur)
                cur = w
        if cur:
            lines.append(cur)
        lines.append("")  # paragraph break
    return lines
