"""Графическое приложение: python app.py (без аргументов)."""
import tkinter as tk
from tkinter import filedialog, messagebox
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.patches import Polygon as MPoly, Rectangle
from clipping import (liang_barsky, sutherland_hodgman, read_segments,
                      read_polygon)

DEMO_SEGS = ([(-2, 1, 12, 9), (1, 1, 4, 3), (-3, -2, -1, 8), (2, -3, 9, 12),
              (0, 7, 11, 7), (-1, 5, 3, -1), (8, 10, 12, 4), (2, 2, 6, 2)],
             (0, 0, 10, 8))
DEMO_POLY = ([(5, -2), (12, 4), (5, 10), (-2, 4)], (0, 0, 10, 8))


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Вариант 11: отсечение отрезков и многоугольников")
        self.geometry("1000x720")
        bar = tk.Frame(self); bar.pack(side=tk.TOP, fill=tk.X, padx=6, pady=6)
        for text, cmd in [("Демо: отрезки (Лианг-Барски)", self.demo_segs),
                          ("Демо: многоугольник (Сазерленд-Ходжман)", self.demo_poly),
                          ("Открыть файл отрезков…", self.open_segs),
                          ("Открыть файл многоугольника…", self.open_poly),
                          ("Сохранить PNG…", self.save)]:
            tk.Button(bar, text=text, command=cmd).pack(side=tk.LEFT, padx=3)
        self.info = tk.Label(self, anchor="w", justify="left")
        self.info.pack(side=tk.BOTTOM, fill=tk.X, padx=6)
        self.fig = Figure(figsize=(8, 6), layout="tight"); self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, self)
        NavigationToolbar2Tk(self.canvas, self).update()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.demo_segs()

    def _axes(self, title, win, pts):
        ax = self.ax; ax.clear()
        xs = [p[0] for p in pts] + [win[0], win[2]]
        ys = [p[1] for p in pts] + [win[1], win[3]]
        m = 0.1 * max(max(xs) - min(xs), max(ys) - min(ys), 1)
        ax.set_xlim(min(xs) - m, max(xs) + m); ax.set_ylim(min(ys) - m, max(ys) + m)
        ax.set_aspect("equal"); ax.grid(True, alpha=.3)
        ax.axhline(0, color="k", lw=.8); ax.axvline(0, color="k", lw=.8)
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_title(title)
        ax.add_patch(Rectangle((win[0], win[1]), win[2]-win[0], win[3]-win[1],
                               fill=True, fc="#dde8ff", ec="blue", lw=2, label="Окно"))

    def show_segs(self, segs, win):
        pts = [(s[0], s[1]) for s in segs] + [(s[2], s[3]) for s in segs]
        self._axes("Алгоритм Лианга-Барски", win, pts)
        vis = 0
        for i, s in enumerate(segs):
            self.ax.plot([s[0], s[2]], [s[1], s[3]], color="gray", ls="--", lw=1.2,
                         label="Исходные" if i == 0 else None)
            r = liang_barsky(*s, *win)
            if r:
                vis += 1
                self.ax.plot([r[0], r[2]], [r[1], r[3]], color="green", lw=3,
                             label="Видимые части" if vis == 1 else None)
        self.ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1)); self.canvas.draw()
        self.info.config(text=f"Отрезков: {len(segs)}, видимых: {vis}. Окно: {win}")

    def show_poly(self, poly, win):
        self._axes("Алгоритм Сазерленда-Ходжмана", win, poly)
        self.ax.add_patch(MPoly(poly, closed=True, fill=False, ec="gray", ls="--",
                                lw=1.5, label="Исходный"))
        r = sutherland_hodgman(poly, *win)
        if r:
            self.ax.add_patch(MPoly(r, closed=True, fc="#9be59b", ec="green", lw=3,
                                    alpha=.8, label="Видимая часть"))
        self.ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1)); self.canvas.draw()
        txt = ", ".join(f"({x:.2f}; {y:.2f})" for x, y in r) if r else "нет"
        self.info.config(text=f"Вершин: {len(poly)} → {len(r)}. Результат: {txt}")

    def demo_segs(self): self.show_segs(*DEMO_SEGS)
    def demo_poly(self): self.show_poly(*DEMO_POLY)

    def _open(self, reader, show):
        path = filedialog.askopenfilename(filetypes=[("Текст", "*.txt"), ("Все", "*")])
        if not path:
            return
        try:
            show(*reader(path))
        except Exception as e:
            messagebox.showerror("Ошибка чтения", str(e))

    def open_segs(self): self._open(read_segments, self.show_segs)
    def open_poly(self): self._open(read_polygon, self.show_poly)

    def save(self):
        p = filedialog.asksaveasfilename(defaultextension=".png")
        if p:
            self.fig.savefig(p, dpi=150)


if __name__ == "__main__":
    App().mainloop()
