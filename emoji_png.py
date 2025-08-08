"""Rendu d'un emoji en PNG via Pillow."""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

FONT_PATH_CANDIDATES = [
    "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf",
    "/usr/share/fonts/truetype/joypixels/JoyPixels-Regular.ttf",
]


def _find_font() -> str:
    for p in FONT_PATH_CANDIDATES:
        if Path(p).exists():
            return p
    raise RuntimeError("Aucune police emoji couleur trouvée. Installez NotoColorEmoji.ttf")


FONT_PATH = _find_font()


def emoji_to_png(emoji_char: str, output_path: Path | str, size: int = 128):
    output_path = Path(output_path)
    img = Image.new("RGBA", (size, size), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(FONT_PATH, size - 10)
    draw.text((0, 0), emoji_char, font=font, embedded_color=True)
    img.save(output_path)
    return output_path
