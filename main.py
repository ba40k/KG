import math
import tkinter as tk
from tkinter import colorchooser


BACKGROUND = "#07152b"
PANEL = "#e8f1f4"
INK = "#10243b"
CYAN = "#52f5e8"
PINK = "#ff4da6"
YELLOW = "#f5eb72"

MODELS = {
    "CMYK": (("C", 100), ("M", 100), ("Y", 100), ("K", 100)),
    "RGB": (("R", 255), ("G", 255), ("B", 255)),
    "HLS": (("H", 360), ("L", 100), ("S", 100)),
}


def cmyk_to_rgb(values):
    cyan, magenta, yellow, black = (value / 100 for value in values)
    return (
        round(255 * (1 - cyan) * (1 - black)),
        round(255 * (1 - magenta) * (1 - black)),
        round(255 * (1 - yellow) * (1 - black)),
    )


def rgb_to_cmyk(rgb):
    red, green, blue = (value / 255 for value in rgb)
    black = 1 - max(red, green, blue)
    if black == 1:
        return (0, 0, 0, 100)
    return tuple(
        round(value * 100, 2)
        for value in (
            (1 - red - black) / (1 - black),
            (1 - green - black) / (1 - black),
            (1 - blue - black) / (1 - black),
            black,
        )
    )


def hls_to_rgb(values):
    hue, lightness, saturation = values
    hue %= 360
    lightness /= 100
    saturation /= 100

    chroma = (1 - abs(2 * lightness - 1)) * saturation
    sector = hue / 60
    second = chroma * (1 - abs(sector % 2 - 1))
    offset = lightness - chroma / 2

    if sector < 1:
        red, green, blue = chroma, second, 0
    elif sector < 2:
        red, green, blue = second, chroma, 0
    elif sector < 3:
        red, green, blue = 0, chroma, second
    elif sector < 4:
        red, green, blue = 0, second, chroma
    elif sector < 5:
        red, green, blue = second, 0, chroma
    else:
        red, green, blue = chroma, 0, second

    return tuple(
        round((component + offset) * 255)
        for component in (red, green, blue)
    )


def rgb_to_hls(rgb):
    red, green, blue = (value / 255 for value in rgb)
    maximum = max(red, green, blue)
    minimum = min(red, green, blue)
    difference = maximum - minimum
    lightness = (maximum + minimum) / 2

    if difference == 0:
        hue = 0
        saturation = 0
    else:
        if lightness <= 0.5:
            saturation = difference / (maximum + minimum)
        else:
            saturation = difference / (2 - maximum - minimum)

        if maximum == red:
            hue = ((green - blue) / difference) % 6
        elif maximum == green:
            hue = (blue - red) / difference + 2
        else:
            hue = (red - green) / difference + 4
        hue *= 60

    return (
        round(hue, 2) % 360,
        round(lightness * 100, 2),
        round(saturation * 100, 2),
    )


class ColorStudio:
    def __init__(self, root):
        self.root = root
        self.rgb = (38, 210, 230)
        self.updating = False
        self.entries = {}
        self.scales = {}

        root.title("COLOR//STUDIO 2000")
        root.configure(bg=BACKGROUND)
        root.minsize(920, 610)
        root.geometry("1120x690")

        header = tk.Frame(root, bg=BACKGROUND, padx=22, pady=16)
        header.pack(fill="x")
        tk.Label(
            header,
            text="COLOR//STUDIO",
            font=("Arial Black", 25),
            fg=CYAN,
            bg=BACKGROUND,
        ).pack(side="left")
        tk.Label(
            header,
            text="  Y2K EDITION  •  CMYK / RGB / HLS",
            font=("Courier New", 11, "bold"),
            fg=PINK,
            bg=BACKGROUND,
        ).pack(side="left", pady=(10, 0))

        tk.Label(
            root,
            text="ИЗМЕНЯЙТЕ ЦВЕТ В ЛЮБОЙ МОДЕЛИ — ОСТАЛЬНЫЕ ОБНОВЯТСЯ СРАЗУ",
            font=("Courier New", 10, "bold"),
            fg=YELLOW,
            bg=BACKGROUND,
        ).pack(anchor="w", padx=24, pady=(0, 12))

        workspace = tk.Frame(root, bg=BACKGROUND, padx=18)
        workspace.pack(fill="both", expand=True)
        for column, model in enumerate(MODELS):
            workspace.columnconfigure(column, weight=1, uniform="model")
            self._build_model_panel(workspace, model, column)

        footer = tk.Frame(root, bg=BACKGROUND, padx=22, pady=14)
        footer.pack(fill="x")
        self.preview = tk.Label(
            footer,
            text="",
            width=12,
            height=2,
            relief="raised",
            bd=4,
            bg="#26d2e6",
        )
        self.preview.pack(side="left")
        self.hex_label = tk.Label(
            footer,
            text="HEX  #26D2E6",
            font=("Courier New", 14, "bold"),
            fg=CYAN,
            bg=BACKGROUND,
            padx=12,
        )
        self.hex_label.pack(side="left")
        self.status = tk.Label(
            footer,
            text="READY  |  Выберите палитру, введите значения или двигайте ползунки",
            font=("Courier New", 9),
            fg="#b9c9d8",
            bg=BACKGROUND,
            anchor="e",
        )
        self.status.pack(side="right", fill="x", expand=True)

        self._sync_from_rgb(self.rgb)

    def _build_model_panel(self, parent, model, column):
        accent = {"CMYK": "#00a9b7", "RGB": "#168548", "HLS": "#a72f91"}[model]
        panel = tk.Frame(parent, bg=PANEL, bd=3, relief="ridge", padx=12, pady=12)
        panel.grid(row=0, column=column, sticky="nsew", padx=6)

        tk.Label(
            panel,
            text=f" {model} ",
            font=("Arial Black", 19),
            fg="white",
            bg=accent,
            padx=8,
            pady=4,
        ).pack(fill="x", pady=(0, 12))

        self.entries[model] = []
        self.scales[model] = []
        for index, (channel, maximum) in enumerate(MODELS[model]):
            row = tk.Frame(panel, bg=PANEL)
            row.pack(fill="x", pady=4)

            tk.Label(
                row,
                text=channel,
                width=3,
                anchor="w",
                font=("Courier New", 11, "bold"),
                fg=INK,
                bg=PANEL,
            ).pack(side="left")
            entry_value = tk.StringVar()
            entry = tk.Entry(
                row,
                textvariable=entry_value,
                width=7,
                justify="right",
                font=("Courier New", 11, "bold"),
                relief="sunken",
                bd=2,
            )
            entry.pack(side="right")
            entry.bind("<Return>", lambda _event, name=model: self.apply_entry(name))
            entry.bind("<FocusOut>", lambda _event, name=model: self.apply_entry(name))
            self.entries[model].append(entry_value)

            scale_value = tk.DoubleVar()
            scale = tk.Scale(
                panel,
                from_=0,
                to=maximum,
                orient="horizontal",
                variable=scale_value,
                command=lambda value, name=model, position=index: self.on_slider(
                    name, position, value
                ),
                showvalue=False,
                resolution=1,
                length=270,
                bd=1,
                relief="sunken",
                bg=PANEL,
                fg=INK,
                troughcolor="#b6cbd4",
                activebackground=accent,
                highlightthickness=0,
            )
            scale.pack(fill="x", pady=(0, 2))
            self.scales[model].append(scale_value)

        tk.Button(
            panel,
            text="▣  ВЫБРАТЬ ИЗ ПАЛИТРЫ",
            command=lambda name=model: self.choose_color(name),
            font=("Courier New", 9, "bold"),
            bg="#d0dce3",
            fg=INK,
            activebackground=YELLOW,
            relief="raised",
            bd=3,
            pady=5,
        ).pack(fill="x", pady=(12, 0))

    def _sync_from_rgb(self, rgb, source_model=None, source_values=None):
        self.rgb = rgb
        converted = {
            "CMYK": rgb_to_cmyk(rgb),
            "RGB": rgb,
            "HLS": rgb_to_hls(rgb),
        }
        if source_model is not None:
            converted[source_model] = source_values

        self.updating = True
        for model, values in converted.items():
            for index, value in enumerate(values):
                self.entries[model][index].set(self._format_value(value))
                self.scales[model][index].set(value)
        self.updating = False

        color = f"#{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X}"
        self.preview.configure(bg=color)
        self.hex_label.configure(text=f"HEX  {color}")
        self.status.configure(text="READY  |  Цвет пересчитан во всех трёх моделях", fg=CYAN)

    def _values_from_entries(self, model):
        values = []
        for index, (channel, maximum) in enumerate(MODELS[model]):
            text = self.entries[model][index].get().strip()
            try:
                value = float(text)
            except ValueError as error:
                raise ValueError(f"{channel}: введите число") from error
            if not math.isfinite(value):
                raise ValueError(f"{channel}: введите конечное число")
            if model == "RGB":
                if not value.is_integer():
                    raise ValueError(f"{channel}: RGB задаётся целым числом")
                value = int(value)
            if not 0 <= value <= maximum:
                raise ValueError(f"{channel}: допустимый диапазон 0–{maximum}")
            values.append(value)
        return tuple(values)

    @staticmethod
    def _format_value(value):
        if isinstance(value, float):
            return f"{value:.2f}".rstrip("0").rstrip(".")
        return str(value)

    def apply_entry(self, model):
        try:
            values = self._values_from_entries(model)
        except ValueError as error:
            self.status.configure(text=f"INPUT ERROR  |  {error}", fg=PINK)
            return
        self._apply_values(model, values)

    def on_slider(self, model, index, raw_value):
        if self.updating:
            return
        value = float(raw_value)
        self.entries[model][index].set(self._format_value(value))
        try:
            values = self._values_from_entries(model)
        except ValueError as error:
            self.status.configure(text=f"INPUT ERROR  |  {error}", fg=PINK)
            return
        self._apply_values(model, values)

    def _apply_values(self, model, values):
        if model == "CMYK":
            rgb = cmyk_to_rgb(values)
        elif model == "RGB":
            rgb = values
        else:
            rgb = hls_to_rgb(values)
        self._sync_from_rgb(rgb, model, values)

    def choose_color(self, model):
        color = colorchooser.askcolor(
            color="#%02X%02X%02X" % self.rgb,
            parent=self.root,
            title=f"Выбор цвета для {model}",
        )[0]
        if color is not None:
            rgb = tuple(round(component) for component in color)
            self._sync_from_rgb(rgb)


def main():
    root = tk.Tk()
    ColorStudio(root)
    root.mainloop()


if __name__ == "__main__":
    main()
