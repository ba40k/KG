from __future__ import annotations

import math
import tkinter as tk
from tkinter import ttk


Vertex = tuple[float, float, float]
Matrix = list[list[float]]
Face = tuple[tuple[int, ...], str]


# ============================================================
# Матрицы преобразований
# ============================================================

def identity_matrix() -> Matrix:
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def multiply_matrices(a: Matrix, b: Matrix) -> Matrix:
    return [
        [
            sum(a[i][k] * b[k][j] for k in range(4))
            for j in range(4)
        ]
        for i in range(4)
    ]


def rotation_x(angle: float) -> Matrix:
    c, s = math.cos(angle), math.sin(angle)

    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, c, -s, 0.0],
        [0.0, s, c, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def rotation_y(angle: float) -> Matrix:
    c, s = math.cos(angle), math.sin(angle)

    return [
        [c, 0.0, s, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [-s, 0.0, c, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def rotation_z(angle: float) -> Matrix:
    c, s = math.cos(angle), math.sin(angle)

    return [
        [c, -s, 0.0, 0.0],
        [s, c, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def scaling(sx: float, sy: float, sz: float) -> Matrix:
    return [
        [sx, 0.0, 0.0, 0.0],
        [0.0, sy, 0.0, 0.0],
        [0.0, 0.0, sz, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def translation(dx: float, dy: float, dz: float) -> Matrix:
    return [
        [1.0, 0.0, 0.0, dx],
        [0.0, 1.0, 0.0, dy],
        [0.0, 0.0, 1.0, dz],
        [0.0, 0.0, 0.0, 1.0],
    ]


def transform_point(point: Vertex, matrix: Matrix) -> Vertex:
    x, y, z = point

    return (
        matrix[0][0] * x
        + matrix[0][1] * y
        + matrix[0][2] * z
        + matrix[0][3],

        matrix[1][0] * x
        + matrix[1][1] * y
        + matrix[1][2] * z
        + matrix[1][3],

        matrix[2][0] * x
        + matrix[2][1] * y
        + matrix[2][2] * z
        + matrix[2][3],
    )


def matrix_to_string(matrix: Matrix) -> str:
    return "\n".join(
        " ".join(f"{value:10.3f}" for value in row)
        for row in matrix
    )


# ============================================================
# Шаблоны букв
# ============================================================

GLYPHS: dict[str, list[str]] = {
    "Т": [
        "11111",
        "00100",
        "00100",
        "00100",
        "00100",
        "00100",
        "00100",
    ],

    "К": [
        "10001",
        "10010",
        "10100",
        "11000",
        "10100",
        "10010",
        "10001",
    ],

    "А": [
        "01110",
        "10001",
        "10001",
        "11111",
        "10001",
        "10001",
        "10001",
    ],

    "Ч": [
        "10001",
        "10001",
        "10001",
        "01111",
        "00001",
        "00001",
        "00001",
    ],

    "Е": [
        "11111",
        "10000",
        "10000",
        "11110",
        "10000",
        "10000",
        "11111",
    ],

    "Н": [
        "10001",
        "10001",
        "10001",
        "11111",
        "10001",
        "10001",
        "10001",
    ],

    "О": [
        "01110",
        "10001",
        "10001",
        "10001",
        "10001",
        "10001",
        "01110",
    ],
}


# ============================================================
# Построение геометрии
# ============================================================

def shade_color(color: str, factor: float) -> str:
    channels = [
        int(color[i:i + 2], 16)
        for i in (1, 3, 5)
    ]

    r, g, b = [
        max(0, min(255, round(channel * factor)))
        for channel in channels
    ]

    return f"#{r:02x}{g:02x}{b:02x}"


def build_box(
    cx: float,
    cy: float,
    cz: float,
    width: float,
    height: float,
    depth: float,
    color: str,
) -> tuple[list[Vertex], list[Face]]:
    hx = width / 2
    hy = height / 2
    hz = depth / 2

    vertices = [
        (cx - hx, cy - hy, cz - hz),
        (cx + hx, cy - hy, cz - hz),
        (cx + hx, cy + hy, cz - hz),
        (cx - hx, cy + hy, cz - hz),

        (cx - hx, cy - hy, cz + hz),
        (cx + hx, cy - hy, cz + hz),
        (cx + hx, cy + hy, cz + hz),
        (cx - hx, cy + hy, cz + hz),
    ]

    # Порядок вершин задаёт внешние нормали граней.
    faces = [
        ((0, 3, 2, 1), color),
        ((4, 5, 6, 7), color),
        ((0, 1, 5, 4), color),
        ((1, 2, 6, 5), color),
        ((2, 3, 7, 6), color),
        ((3, 0, 4, 7), color),
    ]

    return vertices, faces


def build_surname_mesh(
    name: str = "ТКАЧЕНКО",
    cell_size: float = 1.0,
    depth: float = 0.9,
) -> tuple[list[Vertex], list[Face]]:
    if not name:
        raise ValueError("Фамилия не должна быть пустой.")

    if cell_size <= 0 or depth <= 0:
        raise ValueError(
            "Размер блока и глубина должны быть положительными."
        )

    vertices: list[Vertex] = []
    faces: list[Face] = []

    palette = [
        "#58d6ff",
        "#fdd76b",
        "#5ef29a",
        "#ff8e6e",
        "#d7b4ff",
        "#ff8cc6",
        "#80baff",
        "#f5b85a",
    ]

    x_offset = 0.0

    for letter_index, symbol in enumerate(name.upper()):
        if symbol not in GLYPHS:
            raise ValueError(
                f"Нет шаблона для символа {symbol!r}."
            )

        glyph = GLYPHS[symbol]
        base_color = palette[letter_index % len(palette)]

        for row_index, row in enumerate(glyph):
            for col_index, value in enumerate(row):
                if value != "1":
                    continue

                # Узор закреплён за исходными блоками.
                factor = (
                    1.0
                    if (row_index + col_index) % 2 == 0
                    else 0.90
                )
                color = shade_color(base_color, factor)

                block_vertices, block_faces = build_box(
                    cx=x_offset + col_index * cell_size,
                    cy=-row_index * cell_size,
                    cz=0.0,
                    width=cell_size * 0.96,
                    height=cell_size * 0.96,
                    depth=depth,
                    color=color,
                )

                offset = len(vertices)
                vertices.extend(block_vertices)

                for indices, face_color in block_faces:
                    faces.append((
                        tuple(index + offset for index in indices),
                        face_color,
                    ))

        x_offset += (len(glyph[0]) + 1.5) * cell_size

    # Перенос центра надписи в начало координат.
    bounds = [
        (
            min(vertex[axis] for vertex in vertices),
            max(vertex[axis] for vertex in vertices),
        )
        for axis in range(3)
    ]

    center = tuple(
        (low + high) / 2
        for low, high in bounds
    )

    vertices = [
        (
            x - center[0],
            y - center[1],
            z - center[2],
        )
        for x, y, z in vertices
    ]

    return vertices, faces


def face_normal(points: list[Vertex]) -> Vertex:
    a, b, c = points[:3]

    u = tuple(b[i] - a[i] for i in range(3))
    v = tuple(c[i] - a[i] for i in range(3))

    nx = u[1] * v[2] - u[2] * v[1]
    ny = u[2] * v[0] - u[0] * v[2]
    nz = u[0] * v[1] - u[1] * v[0]

    length = math.sqrt(nx * nx + ny * ny + nz * nz)

    if length == 0:
        return (0.0, 0.0, 0.0)

    return (
        nx / length,
        ny / length,
        nz / length,
    )


# ============================================================
# Приложение
# ============================================================

class TexturedSurnameApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("3D-фамилия — ТКАЧЕНКО")
        self.root.geometry("1200x760")
        self.root.minsize(900, 560)

        self.vertices, self.faces = build_surname_mesh()

        self.model_width = (
            max(v[0] for v in self.vertices)
            - min(v[0] for v in self.vertices)
        )
        self.model_height = (
            max(v[1] for v in self.vertices)
            - min(v[1] for v in self.vertices)
        )

        self.initial_rotation = multiply_matrices(
            rotation_y(math.radians(-12)),
            rotation_x(math.radians(12)),
        )

        self.rotation = [
            row[:] for row in self.initial_rotation
        ]

        self.zoom = 1.0
        self.shift = [0.0, 0.0, 0.0]
        self.current_matrix = identity_matrix()

        self.last_mouse: tuple[int, int] | None = None
        self.pending_draw: str | None = None

        self.create_interface()
        self.update_matrix()
        self.schedule_draw()

    def create_interface(self) -> None:
        frame = ttk.Frame(self.root, padding=10)
        frame.pack(fill=tk.BOTH, expand=True)

        rotation_controls = ttk.Frame(frame)
        rotation_controls.pack(
            fill=tk.X,
            pady=(0, 6),
        )

        rotation_buttons = [
            ("X −", lambda: self.rotate("x", -15)),
            ("X +", lambda: self.rotate("x", 15)),
            ("Y −", lambda: self.rotate("y", -15)),
            ("Y +", lambda: self.rotate("y", 15)),
            ("Z −", lambda: self.rotate("z", -15)),
            ("Z +", lambda: self.rotate("z", 15)),
        ]

        ttk.Label(
            rotation_controls,
            text="Поворот:",
        ).pack(side=tk.LEFT, padx=(0, 8))

        for text, command in rotation_buttons:
            ttk.Button(
                rotation_controls,
                text=text,
                width=6,
                command=command,
            ).pack(side=tk.LEFT, padx=(0, 5))

        ttk.Button(
            rotation_controls,
            text="Сброс",
            command=self.reset_transform,
        ).pack(side=tk.RIGHT)

        transform_controls = ttk.Frame(frame)
        transform_controls.pack(
            fill=tk.X,
            pady=(0, 8),
        )

        ttk.Label(
            transform_controls,
            text="Масштаб:",
        ).pack(side=tk.LEFT, padx=(0, 8))

        ttk.Button(
            transform_controls,
            text="−",
            width=5,
            command=lambda: self.change_zoom(1 / 1.1),
        ).pack(side=tk.LEFT, padx=(0, 5))

        ttk.Button(
            transform_controls,
            text="+",
            width=5,
            command=lambda: self.change_zoom(1.1),
        ).pack(side=tk.LEFT, padx=(0, 20))

        ttk.Label(
            transform_controls,
            text="Сдвиг:",
        ).pack(side=tk.LEFT, padx=(0, 8))

        move_buttons = [
            ("←", lambda: self.move(-1.5, 0.0)),
            ("→", lambda: self.move(1.5, 0.0)),
            ("↑", lambda: self.move(0.0, 1.5)),
            ("↓", lambda: self.move(0.0, -1.5)),
        ]

        for text, command in move_buttons:
            ttk.Button(
                transform_controls,
                text=text,
                width=5,
                command=command,
            ).pack(side=tk.LEFT, padx=(0, 5))

        self.canvas = tk.Canvas(
            frame,
            bg="#09131d",
            highlightthickness=0,
        )
        self.canvas.pack(
            fill=tk.BOTH,
            expand=True,
        )

        ttk.Label(
            frame,
            text=(
                "ЛКМ + движение — вращение; "
                "колесо мыши — масштаб. "
                "Оси: X — красная, Y — зелёная, Z — синяя."
            ),
        ).pack(
            anchor=tk.W,
            pady=(8, 4),
        )

        self.matrix_var = tk.StringVar()

        ttk.Label(
            frame,
            textvariable=self.matrix_var,
            font=("Courier", 10),
            justify=tk.LEFT,
            padding=8,
            relief=tk.SOLID,
        ).pack(fill=tk.X)

        self.canvas.bind(
            "<Configure>",
            self.schedule_draw,
        )
        self.canvas.bind(
            "<ButtonPress-1>",
            self.mouse_press,
        )
        self.canvas.bind(
            "<B1-Motion>",
            self.mouse_drag,
        )
        self.canvas.bind(
            "<ButtonRelease-1>",
            self.mouse_release,
        )
        self.canvas.bind(
            "<MouseWheel>",
            self.mouse_wheel,
        )

        # Колесо мыши в Linux.
        self.canvas.bind(
            "<Button-4>",
            lambda event: self.change_zoom(1.1),
        )
        self.canvas.bind(
            "<Button-5>",
            lambda event: self.change_zoom(1 / 1.1),
        )

    def update_matrix(self) -> None:
        scale_matrix = scaling(
            self.zoom,
            self.zoom,
            self.zoom,
        )
        move_matrix = translation(*self.shift)

        # M = T · R · S:
        # масштабирование, вращение, перенос.
        self.current_matrix = multiply_matrices(
            move_matrix,
            multiply_matrices(
                self.rotation,
                scale_matrix,
            ),
        )

        self.matrix_var.set(
            "Матрица модели M = T · R · S\n"
            + matrix_to_string(self.current_matrix)
        )

    def schedule_draw(self, event=None) -> None:
        if self.pending_draw is None:
            self.pending_draw = self.root.after_idle(
                self.draw_scene
            )

    def refresh(self) -> None:
        self.update_matrix()
        self.schedule_draw()

    def rotate(self, axis: str, degrees: float) -> None:
        constructors = {
            "x": rotation_x,
            "y": rotation_y,
            "z": rotation_z,
        }

        matrix = constructors[axis](
            math.radians(degrees)
        )

        self.rotation = multiply_matrices(
            matrix,
            self.rotation,
        )
        self.refresh()

    def change_zoom(self, factor: float) -> None:
        self.zoom = max(
            0.15,
            min(5.0, self.zoom * factor),
        )
        self.refresh()

    def move(self, dx: float, dy: float) -> None:
        self.shift[0] += dx
        self.shift[1] += dy
        self.refresh()

    def reset_transform(self) -> None:
        self.rotation = [
            row[:] for row in self.initial_rotation
        ]
        self.zoom = 1.0
        self.shift = [0.0, 0.0, 0.0]
        self.refresh()

    def mouse_press(self, event) -> None:
        self.last_mouse = (event.x, event.y)

    def mouse_release(self, event) -> None:
        self.last_mouse = None

    def mouse_drag(self, event) -> None:
        if self.last_mouse is None:
            return

        previous_x, previous_y = self.last_mouse

        dx = event.x - previous_x
        dy = event.y - previous_y

        self.last_mouse = (event.x, event.y)

        delta = multiply_matrices(
            rotation_y(math.radians(dx * 0.4)),
            rotation_x(math.radians(dy * 0.4)),
        )

        self.rotation = multiply_matrices(
            delta,
            self.rotation,
        )
        self.refresh()

    def mouse_wheel(self, event) -> None:
        if event.delta > 0:
            self.change_zoom(1.1)
        elif event.delta < 0:
            self.change_zoom(1 / 1.1)

    def draw_axes(self) -> None:
        height = self.canvas.winfo_height()

        origin_x = 85
        origin_y = height - 85
        axis_length = 55

        axes = [
            ("X", (1.0, 0.0, 0.0), "#ff6262"),
            ("Y", (0.0, 1.0, 0.0), "#62e889"),
            ("Z", (0.0, 0.0, 1.0), "#69adff"),
        ]

        self.canvas.create_rectangle(
            10,
            height - 170,
            170,
            height - 10,
            fill="#101f2d",
            outline="#294357",
        )

        self.canvas.create_text(
            90,
            height - 155,
            text="Оси модели",
            fill="#dce9f5",
            font=("Arial", 10),
        )

        # Применяется только вращение:
        # индикатор не зависит от переноса и масштаба.
        rotated_axes = [
            (
                label,
                transform_point(direction, self.rotation),
                color,
            )
            for label, direction, color in axes
        ]

        rotated_axes.sort(
            key=lambda axis: axis[1][2],
            reverse=True,
        )

        for label, (x, y, z), color in rotated_axes:
            end_x = origin_x + x * axis_length
            end_y = origin_y - y * axis_length

            projected_length = math.hypot(
                end_x - origin_x,
                end_y - origin_y,
            )

            if projected_length >= 8:
                self.canvas.create_line(
                    origin_x,
                    origin_y,
                    end_x,
                    end_y,
                    fill=color,
                    width=3,
                    arrow=tk.LAST,
                    arrowshape=(10, 12, 4),
                )

                label_x = (
                    end_x
                    + 12 * x * axis_length / projected_length
                )
                label_y = (
                    end_y
                    - 12 * y * axis_length / projected_length
                )
            else:
                # Ось почти перпендикулярна экрану.
                self.canvas.create_oval(
                    end_x - 5,
                    end_y - 5,
                    end_x + 5,
                    end_y + 5,
                    outline=color,
                    fill=color if z < 0 else "#101f2d",
                    width=2,
                )

                label_x = end_x + 13
                label_y = end_y - 13

            self.canvas.create_text(
                label_x,
                label_y,
                text=label,
                fill=color,
                font=("Arial", 11, "bold"),
            )

        self.canvas.create_oval(
            origin_x - 3,
            origin_y - 3,
            origin_x + 3,
            origin_y + 3,
            fill="#ffffff",
            outline="",
        )

    def draw_scene(self) -> None:
        self.pending_draw = None
        self.canvas.delete("all")

        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()

        if width <= 1 or height <= 1:
            return

        transformed = [
            transform_point(
                vertex,
                self.current_matrix,
            )
            for vertex in self.vertices
        ]

        screen_scale = min(
            width * 0.86 / self.model_width,
            height * 0.70 / self.model_height,
        )

        # Ортографическая проекция.
        projected = [
            (
                width / 2 + x * screen_scale,
                height / 2 - y * screen_scale,
            )
            for x, y, z in transformed
        ]

        # Камера находится со стороны отрицательной оси Z.
        # Дальние грани рисуются раньше ближних.
        sorted_faces = sorted(
            self.faces,
            key=lambda face: sum(
                transformed[index][2]
                for index in face[0]
            ) / len(face[0]),
            reverse=True,
        )

        light = (-0.35, 0.45, -0.82)
        light_length = math.sqrt(
            sum(value * value for value in light)
        )
        light = tuple(
            value / light_length
            for value in light
        )

        for indices, base_color in sorted_faces:
            points_3d = [
                transformed[index]
                for index in indices
            ]

            normal = face_normal(points_3d)

            # Отсечение обратных граней.
            if normal[2] >= -1e-9:
                continue

            diffuse = max(
                0.0,
                sum(
                    normal[i] * light[i]
                    for i in range(3)
                ),
            )

            brightness = 0.52 + 0.48 * diffuse
            color = shade_color(
                base_color,
                brightness,
            )

            polygon = [
                coordinate
                for index in indices
                for coordinate in projected[index]
            ]

            self.canvas.create_polygon(
                polygon,
                fill=color,
                outline=shade_color(color, 0.65),
                width=1,
            )

        # Оси рисуются поверх сцены.
        self.draw_axes()


def main() -> None:
    root = tk.Tk()
    TexturedSurnameApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()