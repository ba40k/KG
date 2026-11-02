"""Reproduce report comparison figures and numerical measurements."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from processing import linear_contrast, local_threshold

BASE_DIR = Path(__file__).resolve().parent
SAMPLE_DIR = BASE_DIR / "samples"
RESULT_DIR = BASE_DIR / "results"


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    path = Path("C:/Windows/Fonts/arial.ttf")
    return ImageFont.truetype(str(path), size) if path.is_file() else ImageFont.load_default()


def _metric(image: Image.Image) -> tuple[float, float, float]:
    values = np.asarray(image.convert("L"), dtype=np.uint8)
    p5, p95 = np.percentile(values, (5, 95))
    return float(p95 - p5), float(values.std()), float(np.mean(values < 128) * 100)


def _comparison_sheet(filename: str, output: str, offset: int) -> None:
    with Image.open(SAMPLE_DIR / filename) as opened:
        original = opened.copy()
    grayscale = original.convert("L")
    mean = local_threshold(grayscale, "mean", window_size=31, offset=offset)
    gaussian = local_threshold(grayscale, "gaussian", window_size=31, offset=offset)
    images = (
        ("Исходное изображение", grayscale),
        ("Линейное контрастирование", linear_contrast(grayscale)),
        ("Локальный порог: среднее", mean),
        ("Локальный порог: Гаусс", gaussian),
    )

    panel_width, panel_height, title_height = 480, 320, 38
    margin, gap = 14, 12
    sheet = Image.new(
        "RGB",
        (margin * 2 + panel_width * 2 + gap, margin * 2 + (panel_height + title_height) * 2 + gap),
        "#17232f",
    )
    draw = ImageDraw.Draw(sheet)
    for index, (title, image) in enumerate(images):
        col, row = index % 2, index // 2
        left = margin + col * (panel_width + gap)
        top = margin + row * (panel_height + title_height + gap)
        draw.text((left + 4, top + 7), title, font=_font(19), fill="#e5edf4")
        fitted = ImageOps.contain(image.convert("RGB"), (panel_width, panel_height))
        panel = Image.new("RGB", (panel_width, panel_height), "#0b1118")
        panel.paste(
            fitted,
            ((panel_width - fitted.width) // 2, (panel_height - fitted.height) // 2),
        )
        sheet.paste(panel, (left, top + title_height))
    RESULT_DIR.mkdir(exist_ok=True)
    sheet.save(RESULT_DIR / output)

    print(f"\n{filename}")
    print(f"Local threshold parameters: window=31, C={offset}")
    for name, image in images:
        contrast, deviation, dark_share = _metric(image)
        print(
            f"{name}: P95-P5={contrast:.1f}; sigma={deviation:.1f}; "
            f"pixels <128={dark_share:.2f}%"
        )
    disagreement = np.mean(np.asarray(mean) != np.asarray(gaussian)) * 100
    print(f"Different mean/Gaussian pixels: {disagreement:.2f}%")
    if filename == "text_noise_blur.png":
        truth = np.asarray(
            Image.open(SAMPLE_DIR / "ground_truth_text_noise_blur.png")
        ) < 128
        for name, result in (("mean", mean), ("gaussian", gaussian)):
            predicted = np.asarray(result) < 128
            true_positive = np.count_nonzero(predicted & truth)
            false_positive = np.count_nonzero(predicted & ~truth)
            false_negative = np.count_nonzero(~predicted & truth)
            precision = true_positive / (true_positive + false_positive)
            recall = true_positive / (true_positive + false_negative)
            f1 = 2 * precision * recall / (precision + recall)
            print(
                f"{name}: precision={precision:.4f}; recall={recall:.4f}; "
                f"F1={f1:.4f}"
            )


def main() -> None:
    _comparison_sheet("document_uneven_light.png", "comparison_document.png", 10)
    _comparison_sheet("text_noise_blur.png", "comparison_noisy_text.png", 20)


if __name__ == "__main__":
    main()
