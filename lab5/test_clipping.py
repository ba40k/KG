import unittest
from clipping import liang_barsky, sutherland_hodgman, is_convex
W = (0, 0, 10, 8)
class T(unittest.TestCase):
    def test_inside(self): self.assertEqual(liang_barsky(1, 1, 4, 3, *W), (1, 1, 4, 3))
    def test_outside(self): self.assertIsNone(liang_barsky(-3, -2, -1, 8, *W))
    def test_cross(self):
        r = liang_barsky(-2, 1, 12, 9, *W)
        for a, b in zip(r, (0, 15/7, 10, 55/7)): self.assertAlmostEqual(a, b)
    def test_corner(self):
        r = liang_barsky(-1, 1, 1, -1, *W)
        for a in r: self.assertAlmostEqual(a, 0)
    def test_poly(self):
        r = sutherland_hodgman([(5, -2), (12, 4), (5, 10), (-2, 4)], *W)
        self.assertEqual(len(r), 8)
    def test_poly_out(self):
        self.assertEqual(sutherland_hodgman([(20, 20), (30, 20), (25, 30)], *W), [])
    def test_convex(self):
        self.assertFalse(is_convex([(0, 0), (4, 0), (1, 1), (0, 4)]))
if __name__ == "__main__": unittest.main()
