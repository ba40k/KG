from PIL import Image
import numpy as np
import pytest

from main import ImageWorkbench
from processing import (
    grayscale_histogram_percent,
    linear_contrast,
    local_threshold,
    point_operation,
)


def test_point_operation_applies_linear_transform_and_clips():
    image = Image.fromarray(np.array([[0, 50, 200, 255]], dtype=np.uint8))
    result = np.asarray(point_operation(image, contrast=2, brightness=10))
    assert result.tolist() == [[10, 110, 255, 255]]


def test_point_operation_inverts_after_transform():
    image = Image.fromarray(np.array([[0, 128, 255]], dtype=np.uint8))
    result = np.asarray(point_operation(image, invert=True))
    assert result.tolist() == [[255, 127, 0]]


def test_point_operation_preserves_rgb_image_mode():
    image = Image.new("RGB", (2, 1), (10, 100, 250))
    result = point_operation(image, contrast=2, brightness=-5)
    assert result.mode == "RGB"
    assert result.getpixel((0, 0)) == (15, 195, 255)


def test_point_operation_rejects_invalid_parameters():
    image = Image.new("L", (2, 2), 100)
    with pytest.raises(ValueError):
        point_operation(image, contrast=-1)
    with pytest.raises(ValueError):
        point_operation(image, brightness=float("inf"))


def test_linear_contrast_stretches_requested_percentile_bounds():
    image = Image.fromarray(np.array([[50, 75, 100, 125, 150]], dtype=np.uint8))
    result = np.asarray(linear_contrast(image, 0, 100))
    assert result.tolist() == [[0, 64, 127, 191, 255]]


def test_linear_contrast_leaves_constant_image_unchanged():
    image = Image.new("L", (5, 4), 128)
    assert np.array_equal(np.asarray(linear_contrast(image)), np.asarray(image))


def test_linear_contrast_handles_flat_rgb_channel():
    image = Image.new("RGB", (4, 3), (100, 50, 200))
    image.putpixel((0, 0), (150, 50, 220))
    result = linear_contrast(image, 0, 100)
    assert result.mode == "RGB"
    assert all(0 <= component <= 255 for component in result.getpixel((2, 2)))
    assert result.getpixel((2, 2))[1] == 50


def test_linear_contrast_validates_percentiles():
    with pytest.raises(ValueError):
        linear_contrast(Image.new("L", (3, 3)), 99, 1)


@pytest.mark.parametrize("method", ["mean", "gaussian"])
def test_local_threshold_returns_binary_grayscale(method):
    pixels = np.full((41, 51), 220, dtype=np.uint8)
    pixels[19:22, 8:43] = 30
    result = np.asarray(
        local_threshold(Image.fromarray(pixels), method=method, window_size=15, offset=5)
    )
    assert set(np.unique(result)) == {0, 255}
    assert np.mean(result[19:22, 12:39] == 0) > 0.95
    assert np.mean(result[2:8, 2:8] == 255) > 0.95


def test_local_threshold_supports_light_foreground():
    pixels = np.full((41, 51), 30, dtype=np.uint8)
    pixels[19:22, 8:43] = 220
    result = np.asarray(
        local_threshold(
            Image.fromarray(pixels), method="mean", window_size=15,
            offset=5, foreground="light",
        )
    )
    assert np.mean(result[19:22, 12:39] == 0) > 0.95
    assert np.mean(result[2:8, 2:8] == 255) > 0.95


def test_local_mean_and_gaussian_methods_can_differ():
    rng = np.random.default_rng(4)
    pixels = np.clip(
        np.linspace(100, 180, 65)[None, :] + rng.normal(0, 18, (65, 65)),
        0, 255,
    ).astype(np.uint8)
    image = Image.fromarray(pixels)
    mean = np.asarray(local_threshold(image, "mean", 15, 8))
    gaussian = np.asarray(local_threshold(image, "gaussian", 15, 8))
    assert np.count_nonzero(mean != gaussian) > 0


@pytest.mark.parametrize("window", [0, 2, 10, 3.5, True])
def test_local_threshold_rejects_invalid_windows(window):
    with pytest.raises(ValueError):
        local_threshold(Image.new("L", (10, 10)), window_size=window)


def test_local_threshold_rejects_unknown_method():
    with pytest.raises(ValueError):
        local_threshold(Image.new("L", (10, 10)), method="median")


def test_grayscale_histogram_reports_true_pixel_percentages():
    image = Image.fromarray(np.array([[0, 0, 255, 255]], dtype=np.uint8))
    histogram = grayscale_histogram_percent(image)
    assert histogram.shape == (256,)
    assert histogram[0] == 50
    assert histogram[255] == 50
    assert histogram.sum() == pytest.approx(100)


def test_grayscale_histogram_combines_rgb_channels_as_luminance():
    image = Image.new("RGB", (2, 1))
    image.putdata([(0, 0, 0), (255, 255, 255)])
    histogram = grayscale_histogram_percent(image)
    assert histogram[0] == 50
    assert histogram[255] == 50


class FakeCanvas:
    def __init__(self):
        self.items = []
        self.delete_calls = []

    def winfo_width(self):
        return 600

    def winfo_height(self):
        return 150

    def delete(self, tag):
        self.delete_calls.append(tag)
        self.items.clear()

    def create_text(self, *args, **kwargs):
        self.items.append(("text", args, kwargs))

    def create_line(self, *args, **kwargs):
        self.items.append(("line", args, kwargs))

    def create_rectangle(self, *args, **kwargs):
        self.items.append(("rectangle", args, kwargs))


def test_histogram_redraws_source_and_result_separately_on_shared_scale():
    pixels = np.tile(np.array([0, 255], dtype=np.uint8), (10, 5))
    workbench = ImageWorkbench.__new__(ImageWorkbench)
    workbench.histogram = FakeCanvas()
    workbench.source = Image.new("L", (10, 10), 128)
    workbench.result = Image.fromarray(pixels)

    workbench._draw_histogram()
    first_draw = list(workbench.histogram.items)
    bars = [item for item in first_draw if item[0] == "rectangle"]

    assert [item[2]["fill"] for item in bars] == [
        "#5f8da3", "#67e8d2", "#67e8d2"
    ]
    source_bar_height = bars[0][1][3] - bars[0][1][1]
    result_bar_heights = [bar[1][3] - bar[1][1] for bar in bars[1:]]
    assert source_bar_height > max(result_bar_heights)
    assert any(
        item[0] == "text" and item[2].get("text") == "ИСХОДНОЕ"
        for item in first_draw
    )
    assert any(
        item[0] == "text" and item[2].get("text") == "РЕЗУЛЬТАТ"
        for item in first_draw
    )

    workbench._draw_histogram()
    assert workbench.histogram.delete_calls == ["all", "all"]
    assert workbench.histogram.items == first_draw
