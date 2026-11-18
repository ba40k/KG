"""Headless regression tests: python test_algorithms.py."""
import unittest
import main

class RasterTests(unittest.TestCase):
    def test_all_octants_and_degenerate(self):
        for x0 in range(-4,5):
            for y0 in range(-4,5):
                for x1 in range(-4,5):
                    for y1 in range(-4,5):
                        n = max(abs(x1-x0),abs(y1-y0))
                        for f in main.LINE_ALGORITHMS.values():
                            p = f(x0,y0,x1,y1)
                            self.assertEqual(len(p),n+1)
                            self.assertEqual(p[0],(x0,y0))
                            self.assertEqual(p[-1],(x1,y1))
                            self.assertEqual(len(set(p)),len(p))
                            for a,b in zip(p,p[1:]):
                                self.assertEqual(max(abs(a[0]-b[0]),abs(a[1]-b[1])),1)
    def test_known_differences(self):
        self.assertEqual(main.step_by_step_line(0,0,14,1)[7],(7,1))
        self.assertEqual(main.dda_line(0,0,14,1)[7],(7,0))
        self.assertEqual(main.bresenham_line(0,0,14,1)[7],(7,1))
        self.assertEqual(main.step_by_step_line(-3,-2,3,1)[1],(-2,-2))
        self.assertEqual(main.bresenham_line(-3,-2,3,1)[1],(-2,-1))
    def test_trace_matches_implementations(self):
        for case in ((1,1,8,5),(0,0,14,1),(-3,-2,3,1),(4,8,1,-3),(0,0,0,0)):
            rows=main.comparison_rows(*case)
            for j,f in enumerate(main.LINE_ALGORITHMS.values()):
                self.assertEqual([r[4][j] for r in rows],f(*case))
            for r in rows:
                self.assertEqual(tuple(map(main._round_pixel,r[1])),r[4][0])
                self.assertEqual(tuple(map(main._round_pixel,r[2])),r[4][1])
    def test_circle(self):
        for radius in range(31):
            points=main.bresenham_circle(0,0,radius)
            self.assertEqual(len(points),len(set(points)))
            s=set(points)
            for x,y in points:
                self.assertTrue({(x,y),(-x,y),(x,-y),(-x,-y),(y,x),(-y,x),(y,-x),(-y,-x)} <= s)
                self.assertLessEqual(abs((x*x+y*y)**.5-radius),.75)
        self.assertEqual(main.bresenham_circle(2,3,0),[(2,3)])
        self.assertEqual(len(main.bresenham_circle(0,0,5)),28)
        with self.assertRaises(ValueError): main.bresenham_circle(0,0,-1)

if __name__ == '__main__':
    unittest.main()
