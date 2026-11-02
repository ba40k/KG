"""Interactive image-processing workbench for variant 11."""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageOps, ImageTk

from processing import (
    grayscale_histogram_percent,
    linear_contrast,
    local_threshold,
    point_operation,
)

BASE_DIR = Path(__file__).resolve().parent
SAMPLE_DIR = BASE_DIR / "samples"
SAMPLES = {
    "Документ с неравномерной засветкой": "document_uneven_light.png",
    "Чек с боковой тенью": "receipt_shadow.png",
    "Схема с низким контрастом": "low_contrast_diagram.png",
    "Мелкий текст с шумом и размытием": "text_noise_blur.png",
    "Текстура и перепады освещения": "textured_illumination.png",
    "Скан с пятнистым фоном": "spotted_scan.png",
}


class ImageWorkbench:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.source: Image.Image | None = None
        self.source_name = ""
        self.result: Image.Image | None = None
        self.result_name = ""
        self.preview_images: list[ImageTk.PhotoImage] = []
        self._resize_job: str | None = None

        root.title("Лабораторная работа 2 — ImageLab | Вариант 11")
        root.geometry("1280x850")
        root.minsize(980, 680)
        root.configure(bg="#101923")
        self._configure_style()
        self._build_toolbar()
        self._build_workspace()
        root.bind("<Configure>", self._schedule_preview)

        available = [
            label for label, filename in SAMPLES.items()
            if (SAMPLE_DIR / filename).is_file()
        ]
        self.sample_box.configure(values=available)
        if available:
            self.sample_box.current(0)
            self.load_sample()
        else:
            self.status.set(
                "Примеры не найдены. Запустите generate_samples.py для создания набора."
            )

    def _configure_style(self) -> None:
        style = ttk.Style()
        style.theme_use("clam")
        style.configure(
            "TFrame", background="#17232f"
        )
        style.configure(
            "TLabel", background="#17232f", foreground="#e5edf4",
            font=("Segoe UI", 10),
        )
        style.configure(
            "Header.TLabel", font=("Segoe UI", 19, "bold"),
            foreground="#67e8d2",
        )
        style.configure(
            "Subheader.TLabel", font=("Segoe UI", 11, "bold"),
            foreground="#97c6d5",
        )
        style.configure(
            "TButton", padding=(10, 7), font=("Segoe UI", 9, "bold")
        )
        style.configure(
            "TNotebook", background="#17232f", borderwidth=0
        )
        style.configure(
            "TNotebook.Tab", padding=(10, 8), font=("Segoe UI", 9, "bold")
        )
        style.configure(
            "Horizontal.TScale", background="#17232f"
        )

    def _build_toolbar(self) -> None:
        toolbar = ttk.Frame(self.root, padding=(18, 14))
        toolbar.pack(fill="x")
        ttk.Label(toolbar, text="IMAGELAB", style="Header.TLabel").pack(side="left")
        ttk.Label(
            toolbar,
            text="  •  ОБРАБОТКА ИЗОБРАЖЕНИЙ  •  ВАРИАНТ 11",
            style="Subheader.TLabel",
        ).pack(side="left", padx=(8, 18), pady=(5, 0))
        ttk.Button(toolbar, text="Открыть изображение", command=self.open_image).pack(
            side="right", padx=(6, 0)
        )
        self.save_button = ttk.Button(
            toolbar, text="Сохранить результат", command=self.save_result,
            state="disabled",
        )
        self.save_button.pack(side="right", padx=6)
        ttk.Button(toolbar, text="Сбросить", command=self.reset_result).pack(
            side="right", padx=6
        )
        self.sample_box = ttk.Combobox(toolbar, state="readonly", width=35)
        self.sample_box.pack(side="right", padx=6)
        self.sample_box.bind("<<ComboboxSelected>>", lambda _event: self.load_sample())
        ttk.Label(toolbar, text="Тестовое изображение:").pack(side="right", padx=4)

    def _build_workspace(self) -> None:
        workspace = ttk.Panedwindow(self.root, orient="horizontal")
        workspace.pack(fill="both", expand=True, padx=16, pady=(0, 10))
        controls = ttk.Frame(workspace, padding=10)
        display = ttk.Frame(workspace, padding=(12, 6))
        workspace.add(controls, weight=1)
        workspace.add(display, weight=4)

        self.tabs = ttk.Notebook(controls)
        self.tabs.pack(fill="both", expand=True)
        self.tone_tab = ttk.Frame(self.tabs, padding=12)
        self.threshold_tab = ttk.Frame(self.tabs, padding=12)
        self.tabs.add(self.tone_tab, text="Контраст")
        self.tabs.add(self.threshold_tab, text="Порог")
        self._build_tone_controls()
        self._build_threshold_controls()

        display.columnconfigure((0, 1), weight=1, uniform="preview")
        display.rowconfigure(1, weight=1)
        ttk.Label(display, text="ИСХОДНОЕ ИЗОБРАЖЕНИЕ", style="Subheader.TLabel").grid(
            row=0, column=0, sticky="w", padx=(4, 0), pady=(0, 6)
        )
        ttk.Label(display, text="РЕЗУЛЬТАТ", style="Subheader.TLabel").grid(
            row=0, column=1, sticky="w", padx=(12, 0), pady=(0, 6)
        )
        self.source_view = tk.Label(
            display, text="Откройте изображение\nили выберите пример",
            bg="#0b1118", fg="#a9bac7", font=("Segoe UI", 12),
            relief="flat", compound="center",
        )
        self.source_view.grid(row=1, column=0, sticky="nsew", padx=(0, 6))
        self.result_view = tk.Label(
            display, text="Результат обработки появится здесь",
            bg="#0b1118", fg="#a9bac7", font=("Segoe UI", 12),
            relief="flat", compound="center",
        )
        self.result_view.grid(row=1, column=1, sticky="nsew", padx=(6, 0))
        self.histogram = tk.Canvas(
            display, height=150, bg="#0b1118", highlightthickness=0
        )
        self.histogram.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(12, 6))
        self.metrics = ttk.Label(
            display, text="Метрики: загрузите изображение",
            foreground="#b7c9d7",
        )
        self.metrics.grid(row=3, column=0, columnspan=2, sticky="w")

        self.status = tk.StringVar(value="Готово к работе")
        ttk.Label(
            self.root, textvariable=self.status, anchor="w",
            padding=(18, 9), background="#0b1118", foreground="#67e8d2",
        ).pack(fill="x", side="bottom")

    def _build_tone_controls(self) -> None:
        ttk.Label(
            self.tone_tab, text="Поэлементная операция",
            style="Subheader.TLabel",
        ).pack(anchor="w", pady=(0, 5))
        ttk.Label(
            self.tone_tab,
            text="g(x) = a·x + b. Значение ограничивается диапазоном 0–255.",
            wraplength=260,
        ).pack(anchor="w", pady=(0, 14))
        self.contrast_value = tk.DoubleVar(value=1.0)
        self.brightness_value = tk.DoubleVar(value=0.0)
        self._add_scale(
            self.tone_tab, "Контраст a", self.contrast_value, 0.25, 2.5,
            resolution=0.05,
        )
        self._add_scale(
            self.tone_tab, "Яркость b", self.brightness_value, -100, 100,
            resolution=1,
        )
        self.invert_value = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            self.tone_tab, text="Инвертировать яркость",
            variable=self.invert_value,
        ).pack(anchor="w", pady=(7, 14))
        ttk.Button(
            self.tone_tab, text="Применить точечную операцию",
            command=self.apply_point_operation,
        ).pack(fill="x", pady=(0, 8))
        ttk.Separator(self.tone_tab).pack(fill="x", pady=10)
        ttk.Label(
            self.tone_tab, text="Линейное контрастирование",
            style="Subheader.TLabel",
        ).pack(anchor="w", pady=(0, 5))
        ttk.Label(
            self.tone_tab,
            text="Растягивает процентили 1–99 до полного диапазона 0–255.",
            wraplength=260,
        ).pack(anchor="w", pady=(0, 10))
        ttk.Button(
            self.tone_tab, text="Выполнить растяжение",
            command=self.apply_linear_contrast,
        ).pack(fill="x")

    def _build_threshold_controls(self) -> None:
        ttk.Label(
            self.threshold_tab, text="Локальная пороговая обработка",
            style="Subheader.TLabel",
        ).pack(anchor="w", pady=(0, 5))
        ttk.Label(
            self.threshold_tab,
            text="Порог каждого пикселя вычисляется по его окрестности. "
                 "Результаты среднего и гауссова методов можно сравнить.",
            wraplength=260,
        ).pack(anchor="w", pady=(0, 14))
        self.method_value = tk.StringVar(value="mean")
        ttk.Radiobutton(
            self.threshold_tab, text="Локальное среднее",
            variable=self.method_value, value="mean",
        ).pack(anchor="w", pady=3)
        ttk.Radiobutton(
            self.threshold_tab, text="Гауссово среднее",
            variable=self.method_value, value="gaussian",
        ).pack(anchor="w", pady=3)
        ttk.Separator(self.threshold_tab).pack(fill="x", pady=10)
        self.window_value = tk.IntVar(value=31)
        self.offset_value = tk.DoubleVar(value=10)
        self._add_scale(
            self.threshold_tab, "Размер локального окна (нечётный)",
            self.window_value, 3, 99, resolution=2,
        )
        self._add_scale(
            self.threshold_tab, "Смещение порога C", self.offset_value,
            -30, 40, resolution=1,
        )
        self.foreground_value = tk.StringVar(value="dark")
        ttk.Label(self.threshold_tab, text="Цвет переднего плана").pack(
            anchor="w", pady=(8, 2)
        )
        ttk.Radiobutton(
            self.threshold_tab, text="Тёмные объекты",
            variable=self.foreground_value, value="dark",
        ).pack(anchor="w", pady=2)
        ttk.Radiobutton(
            self.threshold_tab, text="Светлые объекты",
            variable=self.foreground_value, value="light",
        ).pack(anchor="w", pady=2)
        ttk.Button(
            self.threshold_tab, text="Выполнить локальную бинаризацию",
            command=self.apply_local_threshold,
        ).pack(fill="x", pady=(15, 0))

    @staticmethod
    def _add_scale(
        parent: ttk.Frame, label: str, variable: tk.Variable,
        start: float, end: float, resolution: float,
    ) -> None:
        ttk.Label(parent, text=label).pack(anchor="w", pady=(5, 0))
        scale = tk.Scale(
            parent, variable=variable, from_=start, to=end,
            orient="horizontal", resolution=resolution, showvalue=True,
            bg="#17232f", fg="#e5edf4", troughcolor="#344b5b",
            highlightthickness=0, activebackground="#67e8d2",
            bd=0, length=250,
        )
        scale.pack(fill="x", pady=(0, 4))

    def open_image(self) -> None:
        path = filedialog.askopenfilename(
            title="Открыть изображение",
            filetypes=[
                ("Изображения", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp"),
                ("Все файлы", "*.*"),
            ],
        )
        if not path:
            return
        try:
            with Image.open(path) as opened:
                self._set_source(opened.copy(), Path(path).name)
        except (OSError, ValueError) as error:
            messagebox.showerror("Не удалось открыть изображение", str(error))

    def load_sample(self) -> None:
        label = self.sample_box.get()
        filename = SAMPLES.get(label)
        if filename is None:
            return
        path = SAMPLE_DIR / filename
        try:
            with Image.open(path) as opened:
                self._set_source(opened.copy(), label)
        except (OSError, ValueError) as error:
            messagebox.showerror("Не удалось открыть тестовый пример", str(error))

    def _set_source(self, image: Image.Image, name: str) -> None:
        self.source = ImageOps.exif_transpose(image)
        self.source_name = name
        self.result = None
        self.result_name = ""
        self.save_button.configure(state="disabled")
        self.status.set(f"Загружено: {name}  •  {self.source.width} × {self.source.height}")
        self._refresh_display()

    def apply_point_operation(self) -> None:
        if not self._has_source():
            return
        assert self.source is not None
        self._set_result(
            point_operation(
                self.source,
                self.contrast_value.get(),
                self.brightness_value.get(),
                self.invert_value.get(),
            ),
            "Поэлементная операция",
        )

    def apply_linear_contrast(self) -> None:
        if not self._has_source():
            return
        assert self.source is not None
        self._set_result(linear_contrast(self.source), "Линейное контрастирование")

    def apply_local_threshold(self) -> None:
        if not self._has_source():
            return
        assert self.source is not None
        method = self.method_value.get()
        if method not in ("mean", "gaussian"):
            messagebox.showerror("Некорректный метод", "Выберите один из двух методов.")
            return
        foreground = self.foreground_value.get()
        if foreground not in ("dark", "light"):
            messagebox.showerror("Некорректный объект", "Выберите цвет переднего плана.")
            return
        try:
            result = local_threshold(
                self.source,
                method=method,
                window_size=int(self.window_value.get()),
                offset=self.offset_value.get(),
                foreground=foreground,
            )
        except ValueError as error:
            messagebox.showerror("Некорректные параметры", str(error))
            return
        method_label = "локальное среднее" if method == "mean" else "гауссово среднее"
        self._set_result(result, f"Локальный порог — {method_label}")

    def _has_source(self) -> bool:
        if self.source is None:
            messagebox.showinfo("Нет изображения", "Сначала откройте исходное изображение.")
            return False
        return True

    def _set_result(self, image: Image.Image, name: str) -> None:
        self.result = image
        self.result_name = name
        self.save_button.configure(state="normal")
        self.status.set(f"Готово: {name}")
        self._refresh_display()

    def reset_result(self) -> None:
        self.result = None
        self.result_name = ""
        self.save_button.configure(state="disabled")
        self.status.set("Результат сброшен")
        self._refresh_display()

    def save_result(self) -> None:
        if self.result is None:
            return
        path = filedialog.asksaveasfilename(
            title="Сохранить результат",
            defaultextension=".png",
            filetypes=[
                ("PNG", "*.png"), ("JPEG", "*.jpg *.jpeg"),
                ("TIFF", "*.tif *.tiff"), ("Все файлы", "*.*"),
            ],
        )
        if not path:
            return
        try:
            output = self.result
            if Path(path).suffix.lower() in (".jpg", ".jpeg"):
                output = output.convert("RGB")
            output.save(path)
        except (OSError, ValueError) as error:
            messagebox.showerror("Не удалось сохранить результат", str(error))
            return
        self.status.set(f"Сохранено: {path}")

    def _schedule_preview(self, _event: tk.Event) -> None:
        if self._resize_job is not None:
            self.root.after_cancel(self._resize_job)
        self._resize_job = self.root.after(120, self._refresh_display)

    def _refresh_display(self) -> None:
        self._resize_job = None
        width = max(250, self.source_view.winfo_width() - 8)
        height = max(200, self.source_view.winfo_height() - 8)
        self.preview_images = []
        self._show_preview(self.source_view, self.source, width, height)
        self._show_preview(self.result_view, self.result, width, height)
        self._draw_histogram()
        self._update_metrics()

    def _show_preview(
        self, widget: tk.Label, image: Image.Image | None, width: int, height: int,
    ) -> None:
        if image is None:
            widget.configure(image="", text="Откройте изображение\nили выберите пример")
            return
        preview = Image.new("RGB", (width, height), "#0b1118")
        contained = ImageOps.contain(image.convert("RGB"), (width - 16, height - 16))
        preview.paste(
            contained,
            ((width - contained.width) // 2, (height - contained.height) // 2),
        )
        photo = ImageTk.PhotoImage(preview)
        self.preview_images.append(photo)
        widget.configure(image=photo, text="")

    def _draw_histogram(self) -> None:
        canvas = self.histogram
        canvas.delete("all")
        width = max(200, canvas.winfo_width())
        height = max(150, canvas.winfo_height())
        canvas.create_text(
            10, 11,
            text="ГИСТОГРАММА ЯРКОСТИ · % ПИКСЕЛЕЙ · ОБЩИЙ МАСШТАБ",
            anchor="w", fill="#a9bac7", font=("Segoe UI", 9, "bold"),
        )
        if self.source is None:
            return
        histograms = [
            grayscale_histogram_percent(self.source),
            (
                grayscale_histogram_percent(self.result)
                if self.result is not None
                else None
            ),
        ]
        max_percent = max(
            float(histogram.max())
            for histogram in histograms
            if histogram is not None
        )
        max_percent = max(max_percent, 0.1)

        left_margin, right_margin, panel_gap = 42, 12, 32
        panel_width = (width - left_margin - right_margin - panel_gap) / 2
        plot_top, plot_bottom = 42, height - 28
        plot_height = plot_bottom - plot_top
        colors = ("#5f8da3", "#67e8d2")
        labels = ("ИСХОДНОЕ", "РЕЗУЛЬТАТ")

        for panel_index, (histogram, color, label) in enumerate(
            zip(histograms, colors, labels)
        ):
            panel_left = left_margin + panel_index * (panel_width + panel_gap)
            plot_left = panel_left
            plot_right = panel_left + panel_width
            canvas.create_text(
                plot_left, 30, text=label, anchor="w",
                fill=color, font=("Segoe UI", 8, "bold"),
            )

            if panel_index == 0:
                for fraction in (0.0, 0.5, 1.0):
                    y = plot_bottom - fraction * plot_height
                    canvas.create_line(
                        plot_left, y, plot_right, y,
                        fill="#293946", width=1,
                    )
                    canvas.create_text(
                        plot_left - 5, y,
                        text=f"{max_percent * fraction:.1f}%",
                        anchor="e", fill="#8295a4",
                        font=("Segoe UI", 7),
                    )

            if histogram is None:
                canvas.create_text(
                    (plot_left + plot_right) / 2,
                    (plot_top + plot_bottom) / 2,
                    text="нет результата", fill="#8295a4",
                    font=("Segoe UI", 9),
                )
            else:
                bin_count = min(256, max(1, int(panel_width)))
                grouped = np.array(
                    [
                        group.sum()
                        for group in np.array_split(histogram, bin_count)
                    ],
                    dtype=np.float64,
                )
                bin_width = panel_width / bin_count
                for bin_index, percent in enumerate(grouped):
                    if percent <= 0:
                        continue
                    x0 = plot_left + bin_index * bin_width
                    x1 = plot_left + (bin_index + 1) * bin_width
                    y = plot_bottom - percent / max_percent * plot_height
                    canvas.create_rectangle(
                        x0, y, x1, plot_bottom,
                        fill=color, outline="", tags=("histogram-bars",),
                    )

            for value, fraction in (("0", 0.0), ("128", 128 / 255), ("255", 1.0)):
                x = plot_left + fraction * panel_width
                canvas.create_text(
                    x, height - 10, text=value, anchor="center",
                    fill="#8295a4", font=("Segoe UI", 7),
                )

    def _update_metrics(self) -> None:
        if self.source is None:
            self.metrics.configure(text="Метрики: загрузите изображение")
            return
        original = self._metrics_for(self.source)
        if self.result is None:
            result_text = "результат отсутствует"
        else:
            result_text = self._metrics_for(self.result)
        self.metrics.configure(
            text=f"Исходное: {original}     |     Результат: {result_text}"
        )

    @staticmethod
    def _metrics_for(image: Image.Image) -> str:
        grayscale = np.asarray(image.convert("L"), dtype=np.uint8)
        low, high = np.percentile(grayscale, (5, 95))
        black_ratio = np.mean(grayscale < 128) * 100
        return (
            f"контраст P95−P5={high - low:.0f}, "
            f"σ={grayscale.std():.1f}, тёмные пиксели={black_ratio:.1f}%"
        )


def main() -> None:
    root = tk.Tk()
    ImageWorkbench(root)
    root.mainloop()


if __name__ == "__main__":
    main()
