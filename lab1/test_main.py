import pytest

from main import ColorStudio, cmyk_to_rgb, hls_to_rgb, rgb_to_cmyk, rgb_to_hls


@pytest.mark.parametrize(
    ("cmyk", "expected_rgb"),
    [
        ((0, 100, 100, 0), (255, 0, 0)),
        ((100, 0, 100, 0), (0, 255, 0)),
        ((100, 100, 0, 0), (0, 0, 255)),
        ((0, 0, 0, 0), (255, 255, 255)),
        ((0, 0, 0, 100), (0, 0, 0)),
    ],
)
def test_cmyk_to_rgb(cmyk, expected_rgb):
    assert cmyk_to_rgb(cmyk) == expected_rgb


@pytest.mark.parametrize(
    ("rgb", "expected_cmyk"),
    [
        ((255, 0, 0), (0, 100, 100, 0)),
        ((0, 255, 0), (100, 0, 100, 0)),
        ((0, 0, 255), (100, 100, 0, 0)),
        ((255, 255, 255), (0, 0, 0, 0)),
        ((0, 0, 0), (0, 0, 0, 100)),
    ],
)
def test_rgb_to_cmyk(rgb, expected_cmyk):
    assert rgb_to_cmyk(rgb) == expected_cmyk


@pytest.mark.parametrize(
    ("hls", "expected_rgb"),
    [
        ((0, 50, 100), (255, 0, 0)),
        ((120, 50, 100), (0, 255, 0)),
        ((240, 50, 100), (0, 0, 255)),
        ((360, 50, 100), (255, 0, 0)),
        ((0, 0, 100), (0, 0, 0)),
        ((0, 100, 100), (255, 255, 255)),
        ((240, 50, 0), (128, 128, 128)),
    ],
)
def test_hls_to_rgb(hls, expected_rgb):
    assert hls_to_rgb(hls) == expected_rgb


@pytest.mark.parametrize(
    ("rgb", "expected_hls"),
    [
        ((255, 0, 0), (0, 50, 100)),
        ((0, 255, 0), (120, 50, 100)),
        ((0, 0, 255), (240, 50, 100)),
        ((255, 255, 255), (0, 100, 0)),
        ((0, 0, 0), (0, 0, 0)),
        ((128, 128, 128), (0, 50.2, 0)),
    ],
)
def test_rgb_to_hls(rgb, expected_hls):
    assert rgb_to_hls(rgb) == expected_hls


@pytest.mark.parametrize(
    "rgb",
    [
        (0, 0, 0),
        (255, 255, 255),
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (38, 210, 230),
        (17, 83, 149),
        (254, 127, 3),
    ],
)
def test_rgb_hls_round_trip_has_at_most_one_channel_step_of_error(rgb):
    converted_back = hls_to_rgb(rgb_to_hls(rgb))

    assert all(abs(actual - expected) <= 1 for actual, expected in zip(converted_back, rgb))


class StubEntry:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value


def make_color_studio_with_values(model, values):
    studio = ColorStudio.__new__(ColorStudio)
    studio.entries = {
        model: [StubEntry(str(value)) for value in values],
    }
    return studio


@pytest.mark.parametrize(
    ("model", "values", "expected"),
    [
        ("CMYK", ("0", "50.5", "100", "0"), (0, 50.5, 100, 0)),
        ("RGB", ("0", "127", "255"), (0, 127, 255)),
        ("HLS", ("360", "50.25", "100"), (360, 50.25, 100)),
    ],
)
def test_values_from_entries_accepts_valid_values(model, values, expected):
    studio = make_color_studio_with_values(model, values)

    assert studio._values_from_entries(model) == expected


@pytest.mark.parametrize(
    ("model", "values", "message"),
    [
        ("RGB", ("", "0", "0"), "R: введите число"),
        ("RGB", ("abc", "0", "0"), "R: введите число"),
        ("RGB", ("nan", "0", "0"), "R: введите конечное число"),
        ("RGB", ("inf", "0", "0"), "R: введите конечное число"),
        ("RGB", ("-inf", "0", "0"), "R: введите конечное число"),
        ("RGB", ("1.5", "0", "0"), "R: RGB задаётся целым числом"),
        ("RGB", ("-1", "0", "0"), "R: допустимый диапазон 0–255"),
        ("RGB", ("256", "0", "0"), "R: допустимый диапазон 0–255"),
        ("CMYK", ("0", "101", "0", "0"), "M: допустимый диапазон 0–100"),
        ("HLS", ("361", "0", "0"), "H: допустимый диапазон 0–360"),
    ],
)
def test_values_from_entries_rejects_invalid_values(model, values, message):
    studio = make_color_studio_with_values(model, values)

    with pytest.raises(ValueError, match=message):
        studio._values_from_entries(model)
