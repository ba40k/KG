"""Create deterministic synthetic image samples for the laboratory work."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

OUTPUT_DIR = Path(__file__).resolve().parent / "samples"
SIZE = (720, 480)


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf"),
    )
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


def _paper(gradient: np.ndarray) -> Image.Image:
    values = np.clip(gradient, 0, 255).astype(np.uint8)
    return Image.fromarray(np.repeat(values[:, :, None], 3, axis=2), "RGB")


def _document() -> Image.Image:
    rng = np.random.default_rng(11)
    y, x = np.mgrid[0:SIZE[1], 0:SIZE[0]]
    illumination = 222 - 63 * (x / SIZE[0]) - 18 * (y / SIZE[1])
    illumination += rng.normal(0, 2.0, SIZE[::-1])
    image = _paper(illumination)
    draw = ImageDraw.Draw(image)
    draw.text((48, 28), "FIELD REPORT  /  2026", font=_font(28, True), fill=44)
    draw.line((48, 70, 670, 70), fill=75, width=2)
    draw.text((48, 91), "Inspection record", font=_font(20, True), fill=51)
    paragraphs = [
        "Station 04        Date: 18 October        Page 1 of 2",
        "The north window produced a gradual shadow across the page.",
        "Measurements were recorded under changing ambient light.",
        "A threshold based on one global value loses faint details.",
        "Local methods compensate for illumination changes by area.",
        "Reference   128.4       96.2       17.8       PASS",
        "Notes: the small print remains visible near the shaded edge.",
    ]
    for index, text in enumerate(paragraphs):
        y_pos = 133 + index * 42
        draw.text((48, y_pos), text, font=_font(16), fill=47 if index != 5 else 68)
        if index in (0, 2, 4):
            draw.line((48, y_pos + 29, 665, y_pos + 29), fill=122, width=1)
    draw.rectangle((48, 428, 248, 455), outline=58, width=2)
    draw.text((60, 432), "APPROVED", font=_font(14, True), fill=54)
    return image


def _receipt() -> Image.Image:
    rng = np.random.default_rng(22)
    y, x = np.mgrid[0:SIZE[1], 0:SIZE[0]]
    shadow = 25 * np.exp(-((x - 630) ** 2 / (2 * 135 ** 2)))
    illumination = 231 - shadow + 4 * np.sin(y / 47)
    image = _paper(illumination + rng.normal(0, 1.5, SIZE[::-1]))
    draw = ImageDraw.Draw(image)
    draw.text((220, 26), "MARKET  /  RECEIPT", font=_font(23, True), fill=55)
    draw.text((246, 59), "18 OCT 2026     10:42", font=_font(15), fill=72)
    draw.line((180, 91, 550, 91), fill=100, width=1)
    items = [
        ("Whole grain bread", "2.40"),
        ("Apples  x 4", "3.20"),
        ("Milk  1L", "1.85"),
        ("Coffee", "6.75"),
        ("Reusable bag", "0.15"),
    ]
    for index, (name, price) in enumerate(items):
        yy = 111 + index * 34
        draw.text((192, yy), name, font=_font(16), fill=65)
        draw.text((487, yy), price, font=_font(16), fill=61)
    draw.line((180, 286, 550, 286), fill=105, width=1)
    draw.text((192, 302), "TOTAL", font=_font(18, True), fill=49)
    draw.text((480, 302), "14.35", font=_font(18, True), fill=49)
    draw.text((210, 354), "Thank you for your visit", font=_font(15), fill=74)
    draw.text((229, 385), "Please keep this receipt", font=_font(14), fill=82)
    return image


def _diagram() -> Image.Image:
    image = Image.new("RGB", SIZE, (211, 216, 215))
    draw = ImageDraw.Draw(image)
    pale, dark = (161, 169, 169), (131, 139, 139)
    draw.text((38, 24), "PROCESS FLOW / LOW-CONTRAST ORIGINAL", font=_font(20, True), fill=dark)
    boxes = [(42, 148, 174, 220), (278, 148, 410, 220), (514, 148, 646, 220)]
    labels = ("INPUT", "FILTER", "OUTPUT")
    for box, label in zip(boxes, labels):
        draw.rounded_rectangle(box, radius=8, outline=pale, width=3)
        draw.text((box[0] + 28, box[1] + 24), label, font=_font(16, True), fill=dark)
    draw.line((174, 184, 278, 184), fill=pale, width=5)
    draw.polygon(((267, 176), (281, 184), (267, 192)), fill=pale)
    draw.line((410, 184, 514, 184), fill=pale, width=5)
    draw.polygon(((503, 176), (517, 184), (503, 192)), fill=pale)
    draw.text((48, 282), "Signal level:  0.43", font=_font(17), fill=(166, 173, 172))
    draw.text((48, 321), "Background varies slowly across the scan.", font=_font(16), fill=(158, 166, 165))
    draw.text((48, 366), "Thin borders and small text should be preserved.", font=_font(16), fill=(154, 162, 161))
    draw.ellipse((582, 287, 630, 335), outline=(158, 165, 164), width=3)
    draw.arc((574, 279, 638, 343), start=45, end=285, fill=(164, 171, 170), width=2)
    return image


def _noisy_text() -> tuple[Image.Image, Image.Image]:
    rng = np.random.default_rng(33)
    image = Image.new("L", SIZE, 197)
    draw = ImageDraw.Draw(image)
    draw.text((48, 37), "NOISY SMALL PRINT", font=_font(25, True), fill=62)
    mask = Image.new("L", SIZE, 255)
    mask_draw = ImageDraw.Draw(mask)
    mask_draw.text((48, 37), "NOISY SMALL PRINT", font=_font(25, True), fill=0)
    for index in range(8):
        yy = 94 + index * 39
        text = f"Row {index + 1:02d}   sample value {17 + index * 9:03d}   check status"
        draw.text(
            (48, yy), text, font=_font(17 if index % 2 else 15),
            fill=82 + index * 3,
        )
        mask_draw.text((48, yy), text, font=_font(17 if index % 2 else 15), fill=0)
        draw.line((48, yy + 24, 655, yy + 24), fill=168, width=1)
        mask_draw.line((48, yy + 24, 655, yy + 24), fill=0, width=1)
    pixels = np.asarray(image, dtype=np.float32)
    noisy = np.clip(pixels + rng.normal(0, 15, pixels.shape), 0, 255).astype(np.uint8)
    blurred = Image.fromarray(noisy, "L").filter(ImageFilter.GaussianBlur(0.65))
    return blurred.convert("RGB"), mask


def _texture() -> Image.Image:
    y, x = np.mgrid[0:SIZE[1], 0:SIZE[0]]
    illumination = 201 + 31 * np.sin(x / 125) + 15 * np.cos(y / 72)
    illumination += 12 * np.sin((x + y) / 47)
    image = _paper(illumination)
    draw = ImageDraw.Draw(image)
    draw.text((38, 26), "TEXTURE / UNEVEN LIGHTING", font=_font(21, True), fill=49)
    for index, radius in enumerate((13, 19, 25, 31, 37)):
        x0 = 78 + index * 122
        draw.ellipse((x0, 177 - radius, x0 + radius * 2, 177 + radius), outline=55, width=3)
    for index in range(7):
        yy = 267 + index * 23
        draw.line((42, yy, 680 - index * 12, yy), fill=72 + index * 3, width=2)
    draw.text((44, 438), "Varying background tests local threshold stability", font=_font(15), fill=59)
    return image


def _spotted_scan() -> Image.Image:
    rng = np.random.default_rng(44)
    y, x = np.mgrid[0:SIZE[1], 0:SIZE[0]]
    illumination = 234 - 0.055 * x - 0.045 * y
    image = _paper(illumination + rng.normal(0, 4, SIZE[::-1]))
    draw = ImageDraw.Draw(image)
    draw.text((44, 28), "ARCHIVE PAGE  /  FIELD NOTES", font=_font(22, True), fill=52)
    for index in range(7):
        yy = 87 + index * 47
        draw.text(
            (46, yy),
            f"{index + 1:02d}   Light level changed during scanning session.",
            font=_font(16), fill=57 + index * 2,
        )
    # Add low-frequency stains and small specks without obscuring the text.
    overlay = Image.new("L", SIZE, 0)
    stain = ImageDraw.Draw(overlay)
    for _ in range(28):
        cx, cy = int(rng.integers(15, 700)), int(rng.integers(12, 465))
        rx, ry = int(rng.integers(5, 23)), int(rng.integers(3, 14))
        stain.ellipse((cx - rx, cy - ry, cx + rx, cy + ry), fill=int(rng.integers(5, 23)))
    mask = np.asarray(overlay, dtype=np.float32)
    stained = np.asarray(image, dtype=np.float32) - mask[:, :, None]
    return Image.fromarray(np.clip(stained, 0, 255).astype(np.uint8), "RGB")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    generators = {
        "document_uneven_light.png": _document,
        "receipt_shadow.png": _receipt,
        "low_contrast_diagram.png": _diagram,
        "text_noise_blur.png": _noisy_text,
        "textured_illumination.png": _texture,
        "spotted_scan.png": _spotted_scan,
    }
    for filename, generator in generators.items():
        generated = generator()
        if isinstance(generated, tuple):
            image, mask = generated
            mask.save(OUTPUT_DIR / "ground_truth_text_noise_blur.png")
        else:
            image = generated
        image.save(OUTPUT_DIR / filename)
        print(f"Created {OUTPUT_DIR / filename}")


if __name__ == "__main__":
    main()
