# -*- coding: utf-8 -*-
# =============================================================================
#  test_spline.py  -  Tests de verificacion del spline cubico natural
#
#  Ejecutar con:  python test_spline.py
# =============================================================================

import unittest
import numpy as np

from spline_cubico import SplineCubicoNatural, reconstruir_campo_2d

try:
    from scipy.interpolate import CubicSpline
    SCIPY_OK = True
except ImportError:
    SCIPY_OK = False

TOL = 1e-9


class TestSplineCubicoNatural(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.x = np.array([0.0, 0.8, 2.1, 3.0, 4.5, 6.0, 7.2])
        cls.y = np.array([1.0, 2.5, 0.5, 3.0, -1.0, 2.0, 1.5])
        cls.s = SplineCubicoNatural(cls.x, cls.y)

    def test_01_interpola_en_nodos(self):
        for xi, yi in zip(self.x, self.y):
            self.assertAlmostEqual(self.s(xi), yi, delta=1e-8)

    def test_02_condicion_natural(self):
        self.assertAlmostEqual(self.s.M[0], 0.0, delta=1e-12)
        self.assertAlmostEqual(self.s.M[-1], 0.0, delta=1e-12)

    def test_03_continuidad_valor_y_derivada(self):
        eps = 1e-6
        for i in range(1, self.s.n):
            xi = self.x[i]
            izq = self.s(xi - eps)
            der = self.s(xi + eps)
            self.assertAlmostEqual(izq, der, delta=1e-4)
            dizq = self.s.derivada(xi - eps)
            dder = self.s.derivada(xi + eps)
            self.assertAlmostEqual(dizq, dder, delta=1e-3)

    @unittest.skipUnless(SCIPY_OK, "scipy no disponible")
    def test_04_coincide_con_scipy(self):
        cs = CubicSpline(self.x, self.y, bc_type="natural")
        xq = np.linspace(self.x[0], self.x[-1], 200)
        propio = np.array([self.s(v) for v in xq])
        referencia = cs(xq)
        max_err = np.max(np.abs(propio - referencia))
        self.assertLess(max_err, 1e-8, f"Diferencia maxima con scipy: {max_err:.2e}")

    def test_05_reproduce_funcion_suave(self):
        x = np.linspace(0, 10, 9)
        y = np.sin(x) * 0.3 + 0.1 * x
        s = SplineCubicoNatural(x, y)
        xq = np.linspace(0, 10, 500)
        vals = np.array([s(v) for v in xq])
        self.assertTrue(np.all(np.isfinite(vals)))
        self.assertLess(np.max(np.abs(vals)), 10 * np.max(np.abs(y)))

    def test_06_valida_x_no_creciente(self):
        with self.assertRaises(ValueError):
            SplineCubicoNatural([0, 2, 1, 3], [0, 1, 2, 3])

    def test_07_valida_duplicados(self):
        with self.assertRaises(ValueError):
            SplineCubicoNatural([0, 1, 1, 3], [0, 1, 2, 3])

    def test_08_valida_tamanos_distintos(self):
        with self.assertRaises(ValueError):
            SplineCubicoNatural([0, 1, 2, 3], [0, 1, 2])

    def test_09_valida_minimo_puntos(self):
        with self.assertRaises(ValueError):
            SplineCubicoNatural([0, 1], [0, 1])

    def test_10_no_extrapola(self):
        with self.assertRaises(ValueError):
            self.s(self.x[-1] + 1.0)
        with self.assertRaises(ValueError):
            self.s(self.x[0] - 1.0)

    def test_11_evaluacion_vectorizada(self):
        xq = np.linspace(self.x[0], self.x[-1], 37)
        vec = self.s.evaluar(xq)
        esc = np.array([self.s(v) for v in xq])
        self.assertTrue(np.allclose(vec, esc, atol=1e-12))


class TestReconstruccion2D(unittest.TestCase):
    """Tests de reconstruir_campo_2d: reconstruccion separable fila-columna."""

    def test_01_reproduce_funcion_suave_2d(self):
        x_c = np.linspace(0, 20, 6)
        y_c = np.linspace(0, 10, 5)
        Xc, Yc = np.meshgrid(x_c, y_c, indexing="ij")
        campo_c = np.sin(Xc / 5.0) * np.cos(Yc / 4.0) + 0.1 * Xc

        x_f = np.linspace(0, 20, 41)
        y_f = np.linspace(0, 10, 21)
        campo_f = reconstruir_campo_2d(campo_c, x_c, y_c, x_f, y_f)

        Xf, Yf = np.meshgrid(x_f, y_f, indexing="ij")
        exacto = np.sin(Xf / 5.0) * np.cos(Yf / 4.0) + 0.1 * Xf

        err = np.max(np.abs(campo_f - exacto))
        self.assertLess(err, 0.15, f"Error maximo de reconstruccion: {err:.4f}")

    def test_02_interpola_exacto_en_nodos_gruesos(self):
        x_c = np.array([0.0, 2.0, 5.0, 9.0])
        y_c = np.array([0.0, 1.0, 3.0])
        rng = np.random.default_rng(0)
        campo_c = rng.random((4, 3))

        campo_f = reconstruir_campo_2d(campo_c, x_c, y_c, x_c, y_c)
        self.assertTrue(np.allclose(campo_f, campo_c, atol=1e-8))

    def test_03_forma_correcta(self):
        x_c = np.linspace(0, 10, 4)
        y_c = np.linspace(0, 5, 4)
        campo_c = np.zeros((4, 4))
        x_f = np.linspace(0, 10, 30)
        y_f = np.linspace(0, 5, 15)
        campo_f = reconstruir_campo_2d(campo_c, x_c, y_c, x_f, y_f)
        self.assertEqual(campo_f.shape, (30, 15))


class _ResumenResult(unittest.TextTestResult):
    def printErrors(self):
        super().printErrors()
        total = self.testsRun
        fallos = len(self.failures) + len(self.errors)
        ok = total - fallos
        print(f"\n{'='*60}")
        print(f"  Resultado: {ok}/{total} tests pasaron")
        if fallos == 0:
            print("  Spline cubico natural verificado correctamente.")
        else:
            print(f"  {fallos} test(s) fallaron.")
        print(f"{'='*60}")


if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = unittest.TestSuite([
        loader.loadTestsFromTestCase(TestSplineCubicoNatural),
        loader.loadTestsFromTestCase(TestReconstruccion2D),
    ])
    runner = unittest.TextTestRunner(resultclass=_ResumenResult, verbosity=2)
    runner.run(suite)
