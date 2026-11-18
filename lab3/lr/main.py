"""Visualization of basic rasterization algorithms for Laboratory 3."""

from __future__ import annotations

import argparse
import math
import statistics
import sys
import time
from collections.abc import Callable

Point = tuple[int, int]
LineAlgorithm = Callable[[int, int, int, int], list[Point]]


def _round_pixel(value: float) -> int:
    """Round to the nearest pixel, resolving half-pixel ties away from zero."""
    if value >= 0:
        return math.floor(value + 0.5)
    return math.ceil(value - 0.5)


def step_by_step_line(x0: int, y0: int, x1: int, y1: int) -> list[Point]:
    """Rasterize a segment by evaluating its parametric equation per pixel."""
    dx = x1 - x0
    dy = y1 - y0
    steps = max(abs(dx), abs(dy))
    if steps == 0:
        return [(x0, y0)]

    return [
        (
            _round_pixel(x0 + index * dx / steps),
            _round_pixel(y0 + index * dy / steps),
        )
        for index in range(steps + 1)
    ]


def dda_line(x0: int, y0: int, x1: int, y1: int) -> list[Point]:
    """Rasterize a segment with the Digital Differential Analyzer algorithm."""
    dx = x1 - x0
    dy = y1 - y0
    steps = max(abs(dx), abs(dy))
    if steps == 0:
        return [(x0, y0)]

    x_increment = dx / steps
    y_increment = dy / steps
    x = float(x0)
    y = float(y0)
    points = []
    for _ in range(steps + 1):
        points.append((_round_pixel(x), _round_pixel(y)))
        x += x_increment
        y += y_increment
    return points


def bresenham_line(x0: int, y0: int, x1: int, y1: int) -> list[Point]:
    """Rasterize a segment using integer error updates in any octant."""
    dx = abs(x1 - x0)
    sx = 1 if x0 < x1 else -1
    dy = -abs(y1 - y0)
    sy = 1 if y0 < y1 else -1
    error = dx + dy
    points = []

    while True:
        points.append((x0, y0))
        if x0 == x1 and y0 == y1:
            return points

        doubled_error = 2 * error
        if doubled_error >= dy:
            error += dy
            x0 += sx
        if doubled_error <= dx:
            error += dx
            y0 += sy


def bresenham_circle(center_x: int, center_y: int, radius: int) -> list[Point]:
    """Rasterize a circle with an integer decision parameter and 8-way symmetry."""
    if radius < 0:
        raise ValueError("Радиус не может быть отрицательным.")

    x = 0
    y = radius
    decision = 3 - 2 * radius
    points: list[Point] = []
    seen: set[Point] = set()

    while x <= y:
        symmetric_points = (
            (x, y), (y, x), (-x, y), (-y, x),
            (x, -y), (y, -x), (-x, -y), (-y, -x),
        )
        for offset_x, offset_y in symmetric_points:
            point = (center_x + offset_x, center_y + offset_y)
            if point not in seen:
                seen.add(point)
                points.append(point)

        if decision < 0:
            decision += 4 * x + 6
        else:
            decision += 4 * (x - y) + 10
            y -= 1
        x += 1

    return points


LINE_ALGORITHMS: dict[str, LineAlgorithm] = {
    "Пошаговый": step_by_step_line,
    "ЦДА": dda_line,
    "Брезенхем (отрезок)": bresenham_line,
}
CIRCLE_ALGORITHM_NAME = "Брезенхем (окружность)"


def benchmark(repeats: int = 500) -> dict[str, float]:
    """Return mean execution time per call in milliseconds."""
    cases: tuple[tuple[str, Callable[[], list[Point]]], ...] = (
        ("Пошаговый, отрезок (0, 0)–(1000, 600)",
         lambda: step_by_step_line(0, 0, 1000, 600)),
        ("ЦДА, отрезок (0, 0)–(1000, 600)",
         lambda: dda_line(0, 0, 1000, 600)),
        ("Брезенхем, отрезок (0, 0)–(1000, 600)",
         lambda: bresenham_line(0, 0, 1000, 600)),
        ("Брезенхем, окружность R=700",
         lambda: bresenham_circle(0, 0, 700)),
    )
    results = {}
    for name, algorithm in cases:
        samples = []
        for _ in range(5):
            started = time.perf_counter_ns()
            for _ in range(repeats):
                algorithm()
            samples.append((time.perf_counter_ns() - started) / repeats / 1_000_000)
        results[name] = statistics.median(samples)
    return results



def comparison_rows(x0, y0, x1, y1):
    """Trace actual arithmetic; keep it outside timed algorithm calls."""
    n = max(abs(x1 - x0), abs(y1 - y0))
    results = [f(x0, y0, x1, y1) for f in LINE_ALGORITHMS.values()]
    x, y = float(x0), float(y0)
    ix, iy = ((x1 - x0) / n, (y1 - y0) / n) if n else (0., 0.)
    bx, by = x0, y0
    dx, dy = abs(x1 - x0), -abs(y1 - y0)
    sx, sy = (1 if x0 < x1 else -1), (1 if y0 < y1 else -1)
    error = dx + dy
    rows = []
    for i in range(n + 1):
        direct = (x0 + i * (x1 - x0) / n, y0 + i * (y1 - y0) / n) if n else (float(x0), float(y0))
        pixels = [r[i] for r in results]
        rows.append((i, direct, (x, y), error, pixels, len(set(pixels)) != 1))
        x += ix
        y += iy
        e2 = 2 * error
        if e2 >= dy:
            error += dy
            bx += sx
        if e2 <= dx:
            error += dx
            by += sy
    return rows


class RasterizationApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.scale = 30
        self.points: list[Point] = []
        self.start_point: Point | None = None
        self.end_point: Point | None = None
        self.plane_extent = 100
        self._initial_center_pending = True

        root.title("Лабораторная работа 3 — растровые алгоритмы")
        root.geometry("1000x760")
        root.minsize(760, 560)

        self._build_controls()
        workspace = ttk.Frame(root)
        workspace.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 8))
        self.canvas = tk.Canvas(
            workspace,
            background="#ffffff",
            highlightthickness=1,
            highlightbackground="#cbd5e1",
        )
        self.horizontal_scrollbar = ttk.Scrollbar(
            workspace, orient=tk.HORIZONTAL, command=self._scroll_x
        )
        self.vertical_scrollbar = ttk.Scrollbar(
            workspace, orient=tk.VERTICAL, command=self._scroll_y
        )
        self.canvas.configure(
            xscrollcommand=self.horizontal_scrollbar.set,
            yscrollcommand=self.vertical_scrollbar.set,
        )
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.vertical_scrollbar.grid(row=0, column=1, sticky="ns")
        self.horizontal_scrollbar.grid(row=1, column=0, sticky="ew")
        workspace.rowconfigure(0, weight=1)
        workspace.columnconfigure(0, weight=1)
        self.canvas.bind("<Configure>", self._draw_scene)
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind("<Shift-MouseWheel>", self._on_shift_mousewheel)
        self.canvas.bind(
            "<Button-4>", lambda event: self._zoom_at(event.x, event.y, 1.2)
        )
        self.canvas.bind(
            "<Button-5>", lambda event: self._zoom_at(event.x, event.y, 1 / 1.2)
        )
        self.canvas.bind(
            "<Shift-Button-4>",
            lambda _event: self._scroll_x("scroll", "-3", "units"),
        )
        self.canvas.bind(
            "<Shift-Button-5>",
            lambda _event: self._scroll_x("scroll", "3", "units"),
        )
        self.canvas.bind("<ButtonPress-2>", self._start_pan)
        self.canvas.bind("<B2-Motion>", self._pan)
        self.canvas.bind("<ButtonRelease-2>", self._end_pan)

    def _build_controls(self) -> None:
        panel = ttk.Frame(self.root, padding=(12, 10))
        panel.pack(fill=tk.X)

        ttk.Label(panel, text="Алгоритм").grid(row=0, column=0, sticky=tk.W)
        self.algorithm = ttk.Combobox(
            panel,
            values=(*LINE_ALGORITHMS, CIRCLE_ALGORITHM_NAME),
            state="readonly",
            width=25,
        )
        self.algorithm.current(0)
        self.algorithm.grid(row=1, column=0, padx=(0, 14), sticky=tk.W)
        self.algorithm.bind("<<ComboboxSelected>>", self._update_input_states)

        self.entries: dict[str, ttk.Entry] = {}
        self.input_labels: dict[str, ttk.Label] = {}
        defaults = (
            ("x₀", "0"), ("y₀", "0"), ("x₁", "10"), ("y₁", "6"),
            ("R", "6"),
        )
        for column, (label, default) in enumerate(defaults, start=1):
            input_label = ttk.Label(panel, text=label)
            input_label.grid(row=0, column=column, sticky=tk.W)
            entry = ttk.Entry(panel, width=6)
            entry.insert(0, default)
            entry.grid(row=1, column=column, padx=(0, 8), sticky=tk.W)
            self.input_labels[label] = input_label
            self.entries[label] = entry

        self._update_input_states()
        ttk.Button(panel, text="Сравнить три алгоритма", command=self.compare).grid(row=2, column=0, columnspan=3, sticky=tk.W, pady=(8,0))
        ttk.Label(panel, text="Масштаб").grid(row=0, column=6, sticky=tk.W)
        zoom_controls = ttk.Frame(panel)
        zoom_controls.grid(row=1, column=6, padx=(0, 10), sticky=tk.W)
        ttk.Button(zoom_controls, text="−", width=3, command=self.zoom_out).pack(
            side=tk.LEFT
        )
        self.zoom_label = ttk.Label(zoom_controls, width=6, anchor=tk.CENTER)
        self.zoom_label.pack(side=tk.LEFT, padx=2)
        ttk.Button(zoom_controls, text="+", width=3, command=self.zoom_in).pack(
            side=tk.LEFT
        )
        ttk.Button(
            zoom_controls, text="100%", command=self.reset_zoom
        ).pack(side=tk.LEFT, padx=(4, 0))
        self._update_zoom_label()

        ttk.Button(panel, text="Построить", command=self.draw).grid(
            row=1, column=7, padx=(0, 6)
        )
        ttk.Button(panel, text="Очистить", command=self.clear).grid(
            row=1, column=8
        )


    def compare(self):
        try:
            case = tuple(self._integer(k) for k in ("x₀", "y₀", "x₁", "y₁"))
            if max(abs(case[2]-case[0]), abs(case[3]-case[1])) > 5000:
                raise ValueError("Для таблицы сравнения используйте не более 5000 шагов.")
        except ValueError as error:
            messagebox.showerror("Сравнение", str(error), parent=self.root)
            return
        window = tk.Toplevel(self.root)
        window.title("Сравнение: пиксели и промежуточные вычисления")
        window.geometry("1180x780")
        top = ttk.Frame(window, padding=8)
        top.pack(fill=tk.X)
        examples = {
            "Текущий отрезок": case,
            "Совпадение: (1,1) → (8,5)": (1,1,8,5),
            "Погрешность ЦДА: (0,0) → (14,1)": (0,0,14,1),
            "Равноудалённые пиксели: (-3,-2) → (3,1)": (-3,-2,3,1),
        }
        choice = ttk.Combobox(top, values=list(examples), state="readonly", width=57)
        choice.current(0)
        choice.pack(side=tk.LEFT)
        status = ttk.Label(window, padding=8, wraplength=1100)
        status.pack(fill=tk.X)
        pictures = ttk.Frame(window)
        pictures.pack(fill=tk.BOTH, expand=True)
        canvases = []
        for name in LINE_ALGORITHMS:
            frame = ttk.LabelFrame(pictures, text=name)
            frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4)
            canvas = tk.Canvas(frame, background="white", height=280, highlightthickness=0)
            canvas.pack(fill=tk.BOTH, expand=True)
            canvases.append(canvas)
        table_frame = ttk.Frame(window)
        table_frame.pack(fill=tk.BOTH, expand=True, padx=8)
        columns = ("i", "direct", "dda", "error", "p0", "p1", "p2", "diff")
        tree = ttk.Treeview(table_frame, columns=columns, show="headings", height=12)
        headings = ("Шаг", "Формула: x; y", "ЦДА: x; y", "e Брезенхема", "Пошаговый", "ЦДА", "Брезенхем", "Различие")
        for col, text in zip(columns, headings):
            tree.heading(col, text=text)
            tree.column(col, width=205 if col in ("direct", "dda") else 95, stretch=True)
        scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        tree.pack(fill=tk.BOTH, expand=True)
        tree.tag_configure("different", background="#fef3c7")
        ttk.Label(window, text="Жёлтые клетки / строки: не все методы выбрали один пиксель. Серый пунктир: идеальный отрезок. Значения float показаны с 17 значащими цифрами.", padding=8, wraplength=1100).pack(fill=tk.X)
        state = {}

        def paint(_event=None):
            if not state:
                return
            a,b,c,d = state["case"]
            results = state["results"]
            sets = [set(r) for r in results]
            common = set.intersection(*sets)
            xmin, xmax, ymin, ymax = min(a,c)-1, max(a,c)+1, min(b,d)-1, max(b,d)+1
            for canvas, points in zip(canvases, results):
                canvas.delete("all")
                w,h = max(canvas.winfo_width(),100), max(canvas.winfo_height(),100)
                scale = min((w-50)/(xmax-xmin+1), (h-50)/(ymax-ymin+1), 42)
                ox, oy = (w-(xmin+xmax)*scale)/2, (h+(ymin+ymax)*scale)/2
                def screen(x,y): return ox+x*scale, oy-y*scale
                stride = max(1, math.ceil(22/scale))
                for x in range(xmin, xmax+1, stride):
                    px,_ = screen(x,0)
                    canvas.create_line(px,10,px,h-20,fill="#e2e8f0")
                    canvas.create_text(px,h-10,text=str(x),font=("Arial",8))
                for y in range(ymin,ymax+1,stride):
                    _,py = screen(0,y)
                    canvas.create_line(25,py,w-10,py,fill="#e2e8f0")
                    canvas.create_text(12,py,text=str(y),font=("Arial",8))
                if xmin <= 0 <= xmax:
                    canvas.create_line(ox,10,ox,h-20,fill="#64748b",width=2)
                    canvas.create_text(ox+10,12,text="Y")
                if ymin <= 0 <= ymax:
                    canvas.create_line(25,oy,w-10,oy,fill="#64748b",width=2)
                    canvas.create_text(w-12,oy-10,text="X")
                r = max(.5,scale*.42)
                for p in points:
                    px,py = screen(*p)
                    canvas.create_rectangle(px-r,py-r,px+r,py+r,fill="#2563eb" if p in common else "#f59e0b",outline="")
                canvas.create_line(*screen(a,b),*screen(c,d),fill="#475569",dash=(4,3),width=2)
                selected = tree.selection()
                if selected:
                    i = int(tree.item(selected[0], "values")[0])
                    px,py = screen(*points[i])
                    canvas.create_rectangle(px-r-2,py-r-2,px+r+2,py+r+2,outline="#dc2626",width=2)

        def rebuild(_event=None):
            current = examples[choice.get()]
            results = [f(*current) for f in LINE_ALGORITHMS.values()]
            state.update(case=current, results=results)
            tree.delete(*tree.get_children())
            rows = comparison_rows(*current)
            for i, direct, dda, error, pixels, different in rows:
                fmt = lambda p: "; ".join(format(v,".17g") for v in p)
                tree.insert("", "end", values=(i,fmt(direct),fmt(dda),error,*map(str,pixels),"ДА" if different else "—"), tags=("different",) if different else ())
            count = sum(row[-1] for row in rows)
            status.configure(text=f"Отрезок {current[:2]} → {current[2:]}. Шагов с разными пикселями: {count} из {len(rows)}. Выберите строку для выделения шага. Совпадение результата допустимо: сравниваются также способы вычисления.")
            paint()
        choice.bind("<<ComboboxSelected>>", rebuild)
        tree.bind("<<TreeviewSelect>>", paint)
        for canvas in canvases:
            canvas.bind("<Configure>", paint)
        rebuild()

    def _update_input_states(
        self, _event: tk.Event[tk.Misc] | None = None
    ) -> None:
        is_circle = self.algorithm.get() == CIRCLE_ALGORITHM_NAME
        for label in ("x₁", "y₁"):
            state = "disabled" if is_circle else "normal"
            self.entries[label].configure(state=state)
            self.input_labels[label].configure(state=state)
        state = "normal" if is_circle else "disabled"
        self.entries["R"].configure(state=state)
        self.input_labels["R"].configure(state=state)

    def _update_zoom_label(self) -> None:
        self.zoom_label.configure(text=f"{self.scale / 30:.0%}")

    def _zoom_at(self, screen_x: float, screen_y: float, factor: float) -> None:
        new_scale = max(10.0, min(180.0, self.scale * factor))
        if math.isclose(new_scale, self.scale):
            return

        canvas = self.canvas
        world_x = canvas.canvasx(screen_x) / self.scale - self.plane_extent
        world_y = self.plane_extent - canvas.canvasy(screen_y) / self.scale
        self.scale = new_scale
        self._update_zoom_label()
        self._draw_scene()
        self._show_world_point_at(world_x, world_y, screen_x, screen_y)
        self._draw_scene()

    def zoom_in(self) -> None:
        self._zoom_at(
            self.canvas.winfo_width() / 2,
            self.canvas.winfo_height() / 2,
            1.2,
        )

    def zoom_out(self) -> None:
        self._zoom_at(
            self.canvas.winfo_width() / 2,
            self.canvas.winfo_height() / 2,
            1 / 1.2,
        )

    def reset_zoom(self) -> None:
        self.scale = 30.0
        self._update_zoom_label()
        self._draw_scene()
        self._center_on_world(0, 0)
        self._draw_scene()

    def _show_world_point_at(
        self, world_x: float, world_y: float, screen_x: float, screen_y: float
    ) -> None:
        canvas = self.canvas
        canvas.update_idletasks()
        total_width = 2 * self.plane_extent * self.scale
        total_height = 2 * self.plane_extent * self.scale
        left = (self.plane_extent + world_x) * self.scale - screen_x
        top = (self.plane_extent - world_y) * self.scale - screen_y
        max_left = max(0, total_width - canvas.winfo_width())
        max_top = max(0, total_height - canvas.winfo_height())
        canvas.xview_moveto(max(0, min(left, max_left)) / total_width)
        canvas.yview_moveto(max(0, min(top, max_top)) / total_height)

    def _start_pan(self, event: tk.Event[tk.Misc]) -> None:
        self.canvas.scan_mark(event.x, event.y)
        self.canvas.configure(cursor="fleur")

    def _pan(self, event: tk.Event[tk.Misc]) -> None:
        self.canvas.scan_dragto(event.x, event.y, gain=1)
        self._draw_scene()

    def _end_pan(self, _event: tk.Event[tk.Misc]) -> None:
        self.canvas.configure(cursor="")

    def _scroll_x(self, *args: str) -> None:
        self.canvas.xview(*args)
        self._draw_scene()

    def _scroll_y(self, *args: str) -> None:
        self.canvas.yview(*args)
        self._draw_scene()

    def _on_mousewheel(self, event: tk.Event[tk.Misc]) -> str:
        if event.delta:
            steps = (
                event.delta / 120
                if abs(event.delta) >= 120
                else math.copysign(1, event.delta)
            )
            self._zoom_at(event.x, event.y, 1.2 ** steps)
        return "break"

    def _on_shift_mousewheel(self, event: tk.Event[tk.Misc]) -> str:
        if event.delta:
            steps = (
                event.delta / 120
                if abs(event.delta) >= 120
                else math.copysign(1, event.delta)
            )
            self._scroll_x("scroll", str(-int(steps)), "units")
        return "break"

    def _center_on_world(self, x: float, y: float) -> None:
        if not hasattr(self, "canvas"):
            return
        canvas = self.canvas
        canvas.update_idletasks()
        total_width = 2 * self.plane_extent * self.scale
        total_height = 2 * self.plane_extent * self.scale
        left = (self.plane_extent + x) * self.scale - canvas.winfo_width() / 2
        top = (self.plane_extent - y) * self.scale - canvas.winfo_height() / 2
        max_left = max(0, total_width - canvas.winfo_width())
        max_top = max(0, total_height - canvas.winfo_height())
        canvas.xview_moveto(max(0, min(left, max_left)) / total_width)
        canvas.yview_moveto(max(0, min(top, max_top)) / total_height)

    def _integer(self, label: str) -> int:
        text = self.entries[label].get().strip()
        try:
            return int(text)
        except ValueError as error:
            raise ValueError(f"Поле «{label}» должно содержать целое число.") from error

    def draw(self) -> None:
        try:
            algorithm_name = self.algorithm.get()
            if algorithm_name == CIRCLE_ALGORITHM_NAME:
                center_x = self._integer("x₀")
                center_y = self._integer("y₀")
                radius = self._integer("R")
                self.points = bresenham_circle(center_x, center_y, radius)
                self.start_point = None
                self.end_point = None
            else:
                start = (self._integer("x₀"), self._integer("y₀"))
                end = (self._integer("x₁"), self._integer("y₁"))
                algorithm = LINE_ALGORITHMS[algorithm_name]
                self.points = algorithm(*start, *end)
                self.start_point = start
                self.end_point = end
        except (ValueError, KeyError) as error:
            messagebox.showerror("Некорректные данные", str(error), parent=self.root)
            return

        self.plane_extent = max(
            100,
            25 * math.ceil(
                (max(
                    (max(abs(x), abs(y)) for x, y in self.points),
                    default=0,
                ) + 25) / 25
            ),
        )
        self._draw_scene()
        min_x = min(x for x, _ in self.points)
        max_x = max(x for x, _ in self.points)
        min_y = min(y for _, y in self.points)
        max_y = max(y for _, y in self.points)
        self._center_on_world((min_x + max_x) / 2, (min_y + max_y) / 2)
        self._draw_scene()

    def clear(self) -> None:
        self.points = []
        self.start_point = None
        self.end_point = None
        self.plane_extent = 100
        self._draw_scene()
        self._center_on_world(0, 0)
        self._draw_scene()

    def _draw_scene(self, _event: tk.Event[tk.Misc] | None = None) -> None:
        if not hasattr(self, "canvas"):
            return
        canvas = self.canvas
        canvas.delete("all")
        width = max(canvas.winfo_width(), 2)
        height = max(canvas.winfo_height(), 2)
        plane_width = 2 * self.plane_extent * self.scale
        plane_height = 2 * self.plane_extent * self.scale
        canvas.configure(scrollregion=(0, 0, plane_width, plane_height))
        origin_x = self.plane_extent * self.scale
        origin_y = self.plane_extent * self.scale
        left = canvas.canvasx(0)
        right = canvas.canvasx(width)
        top = canvas.canvasy(0)
        bottom = canvas.canvasy(height)
        min_x = max(
            -self.plane_extent,
            math.ceil(left / self.scale - self.plane_extent),
        )
        max_x = min(
            self.plane_extent,
            math.floor(right / self.scale - self.plane_extent),
        )
        min_y = max(
            -self.plane_extent,
            math.ceil(self.plane_extent - bottom / self.scale),
        )
        max_y = min(
            self.plane_extent,
            math.floor(self.plane_extent - top / self.scale),
        )
        label_interval = max(1, math.ceil(24 / self.scale))

        for x in range(min_x, max_x + 1):
            screen_x = origin_x + x * self.scale
            color = "#64748b" if x == 0 else "#e2e8f0"
            canvas.create_line(screen_x, 0, screen_x, plane_height, fill=color)
            if x and x % label_interval == 0:
                canvas.create_text(
                    screen_x + 2, origin_y + 10, text=str(x),
                    fill="#64748b", anchor=tk.NW, font=("Segoe UI", 8),
                )

        for y in range(min_y, max_y + 1):
            screen_y = origin_y - y * self.scale
            color = "#64748b" if y == 0 else "#e2e8f0"
            canvas.create_line(0, screen_y, plane_width, screen_y, fill=color)
            if y and y % label_interval == 0:
                canvas.create_text(
                    origin_x + 4, screen_y + 2, text=str(y),
                    fill="#64748b", anchor=tk.NW, font=("Segoe UI", 8),
                )

        canvas.create_text(
            origin_x + 5, origin_y + 3, text="0",
            fill="#334155", anchor=tk.NW, font=("Segoe UI", 8, "bold"),
        )
        if top <= origin_y <= bottom:
            canvas.create_text(
                right - 12, origin_y - 10, text="X →",
                fill="#334155", anchor=tk.E, font=("Segoe UI", 9, "bold"),
            )
        if left <= origin_x <= right:
            canvas.create_text(
                origin_x + 8, top + 8, text="Y ↑",
                fill="#334155", anchor=tk.NW, font=("Segoe UI", 9, "bold"),
            )

        radius = max(2, self.scale // 2 - 2)
        visible_left = left - radius
        visible_right = right + radius
        visible_top = top - radius
        visible_bottom = bottom + radius
        for x, y in self.points:
            center_x = origin_x + x * self.scale
            center_y = origin_y - y * self.scale
            if not (
                visible_left <= center_x <= visible_right
                and visible_top <= center_y <= visible_bottom
            ):
                continue
            fill = "#2563eb"
            if (x, y) == self.start_point:
                fill = "#16a34a"
            if (x, y) == self.end_point:
                fill = "#dc2626"
            canvas.create_rectangle(
                center_x - radius,
                center_y - radius,
                center_x + radius,
                center_y + radius,
                fill=fill,
                outline="#1e293b",
            )

        if self._initial_center_pending:
            self._initial_center_pending = False
            self.root.after_idle(lambda: self._center_on_world(0, 0))


def run_benchmark() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    print("Среднее время вызова (медиана 5 серий, 500 вызовов в серии):")
    for name, elapsed_ms in benchmark().items():
        print(f"{name}: {elapsed_ms:.6f} мс")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark", action="store_true", help="измерить время работы алгоритмов"
    )
    parser.add_argument("--compare", action="store_true", help="вывести проверочные примеры без интерфейса")
    args = parser.parse_args()
    if args.compare:
        for case in ((1,1,8,5), (0,0,14,1), (-3,-2,3,1)):
            print("Отрезок:", case)
            for row in comparison_rows(*case):
                print(row)
        return
    if args.benchmark:
        run_benchmark()
        return

    global tk, ttk, messagebox
    import tkinter as tk
    from tkinter import ttk, messagebox

    root = tk.Tk()
    RasterizationApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
