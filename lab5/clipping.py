"""Алгоритмы отсечения (вариант 11): Лианг-Барски и Сазерленд-Ходжман."""
EPS = 1e-12


def liang_barsky(x1, y1, x2, y2, xmin, ymin, xmax, ymax):
    """Возвращает видимую часть отрезка (x1,y1,x2,y2) или None."""
    dx, dy = x2 - x1, y2 - y1
    p = (-dx, dx, -dy, dy)
    q = (x1 - xmin, xmax - x1, y1 - ymin, ymax - y1)
    t0, t1 = 0.0, 1.0
    for pk, qk in zip(p, q):
        if abs(pk) < EPS:
            if qk < 0:            # параллелен границе и лежит снаружи
                return None
        else:
            t = qk / pk
            if pk < 0:            # вход
                t0 = max(t0, t)
            else:                 # выход
                t1 = min(t1, t)
            if t0 > t1:
                return None
    return (x1 + t0 * dx, y1 + t0 * dy, x1 + t1 * dx, y1 + t1 * dy)


def _clip_edge(poly, inside, intersect):
    out = []
    n = len(poly)
    for i in range(n):
        cur, prev = poly[i], poly[i - 1]
        if inside(cur):
            if not inside(prev):
                out.append(intersect(prev, cur))
            out.append(cur)
        elif inside(prev):
            out.append(intersect(prev, cur))
    return out


def sutherland_hodgman(poly, xmin, ymin, xmax, ymax):
    """Отсекает выпуклый многоугольник прямоугольным окном."""
    def ix(a, b, x):
        t = (x - a[0]) / (b[0] - a[0]); return (x, a[1] + t * (b[1] - a[1]))
    def iy(a, b, y):
        t = (y - a[1]) / (b[1] - a[1]); return (a[0] + t * (b[0] - a[0]), y)
    edges = [
        (lambda p: p[0] >= xmin, lambda a, b: ix(a, b, xmin)),
        (lambda p: p[0] <= xmax, lambda a, b: ix(a, b, xmax)),
        (lambda p: p[1] >= ymin, lambda a, b: iy(a, b, ymin)),
        (lambda p: p[1] <= ymax, lambda a, b: iy(a, b, ymax)),
    ]
    res = list(poly)
    for inside, inter in edges:
        if not res:
            break
        res = _clip_edge(res, inside, inter)
    return res


def is_convex(poly):
    n = len(poly)
    if n < 3:
        return False
    sign = 0
    for i in range(n):
        a, b, c = poly[i], poly[(i + 1) % n], poly[(i + 2) % n]
        cr = (b[0]-a[0])*(c[1]-b[1]) - (b[1]-a[1])*(c[0]-b[0])
        if abs(cr) < EPS:
            return False
        s = 1 if cr > 0 else -1
        if sign and s != sign:
            return False
        sign = s
    return True


def read_segments(path):
    nums = open(path, encoding="utf-8").read().split()
    n = int(nums[0]); v = list(map(float, nums[1:]))
    if len(v) != 4 * n + 4:
        raise ValueError("Ожидается n, затем n строк по 4 числа и строка окна")
    segs = [tuple(v[4*i:4*i+4]) for i in range(n)]
    return segs, tuple(v[4*n:])


def read_polygon(path):
    nums = open(path, encoding="utf-8").read().split()
    n = int(nums[0]); v = list(map(float, nums[1:]))
    if len(v) != 2 * n + 4:
        raise ValueError("Ожидается n, затем n строк x y и строка окна")
    poly = [(v[2*i], v[2*i+1]) for i in range(n)]
    if not is_convex(poly):
        raise ValueError("Многоугольник не является строго выпуклым")
    return poly, tuple(v[2*n:])
