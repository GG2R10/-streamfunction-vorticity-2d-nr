# -*- coding: utf-8 -*-
# =============================================================================
#  test_solver.py  —  Tests de verificación del solver ψ-ω
#
#  Ejecutar con:  python test_solver.py
#
#  Tests incluidos:
#    1. Residuo global |F| < tol  (prueba de convergencia principal)
#    2. Condiciones de contorno INLET  (ψ y ω Dirichlet)
#    3. Condiciones de contorno WALL   (ψ Dirichlet)
#    4. Condición de Thom en paredes del canal
#    5. Ecuación de Poisson en nodos FREE
#    6. Condición de Neumann en OUTLET
#    7. Rango físico de ψ  (0 ≤ ψ ≤ ψ_max en fluido)
# =============================================================================

import unittest
import numpy as np

from simulacion import (
    re_continuation, compute_F,
    kind, Nx, Ny, h, V0, psi_max,
    SOLID, INLET, OUTLET, WALL, BC_WALL, FREE,
)

TOL_SOLVER  = 1e-7   # tolerancia de convergencia Newton-Raphson
TOL_BC      = 1e-6   # tolerancia para verificar condiciones de contorno
RE_TEST     = 2    # Reynolds del test


class TestConvergencia(unittest.TestCase):
    """Resuelve el sistema una sola vez y reutiliza en todos los tests."""

    @classmethod
    def setUpClass(cls):
        print(f"\n{'='*60}")
        print(f"  Resolviendo para Re = {RE_TEST} ...")
        print(f"{'='*60}")
        cls.psi, cls.omega = re_continuation(RE_TEST)
        cls.F = compute_F(cls.psi, cls.omega, RE_TEST)
        cls.norm_F = np.linalg.norm(cls.F)
        print(f"\n  |F| final = {cls.norm_F:.4e}   (tol = {TOL_SOLVER:.0e})")

    # ──────────────────────────────────────────────────────────────────────────
    # Test 1 – Convergencia global
    # ──────────────────────────────────────────────────────────────────────────
    def test_01_residuo_global(self):
        """Newton-Raphson debe converger: |F| < TOL_SOLVER."""
        self.assertLess(
            self.norm_F, TOL_SOLVER,
            f"|F| = {self.norm_F:.2e} no alcanzó la tolerancia {TOL_SOLVER:.0e}",
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Test 2 – Condición de entrada (INLET)
    # ──────────────────────────────────────────────────────────────────────────
    def test_02_bc_inlet_psi(self):
        """ψ = V0·j·h en todos los nodos INLET (flujo plug uniforme)."""
        errores = []
        for j in range(Ny + 1):
            if kind[0, j] == INLET:
                esperado = V0 * j * h
                err = abs(self.psi[0, j] - esperado)
                if err > TOL_BC:
                    errores.append(f"  j={j}: ψ={self.psi[0,j]:.6f}  esperado={esperado:.6f}  err={err:.2e}")
        self.assertEqual(len(errores), 0,
            "ψ INLET incorrecto en:\n" + "\n".join(errores))

    def test_03_bc_inlet_omega(self):
        """ω = 0 en todos los nodos INLET."""
        errores = []
        for j in range(Ny + 1):
            if kind[0, j] == INLET:
                err = abs(self.omega[0, j])
                if err > TOL_BC:
                    errores.append(f"  j={j}: ω={self.omega[0,j]:.2e}")
        self.assertEqual(len(errores), 0,
            "ω INLET ≠ 0 en:\n" + "\n".join(errores))

    # ──────────────────────────────────────────────────────────────────────────
    # Test 3 – Condición Dirichlet ψ en paredes del canal (WALL)
    # ──────────────────────────────────────────────────────────────────────────
    def test_04_bc_wall_psi_inferior(self):
        """ψ = 0 en la pared inferior (j=0), nodos WALL."""
        errores = []
        for i in range(1, Nx):
            if kind[i, 0] == WALL:
                err = abs(self.psi[i, 0])
                if err > TOL_BC:
                    errores.append(f"  i={i}: ψ={self.psi[i,0]:.2e}")
        self.assertEqual(len(errores), 0,
            "ψ ≠ 0 en pared inferior:\n" + "\n".join(errores))

    def test_05_bc_wall_psi_superior(self):
        """ψ = ψ_max en la pared superior (j=Ny), nodos WALL."""
        errores = []
        for i in range(1, Nx):
            if kind[i, Ny] == WALL:
                err = abs(self.psi[i, Ny] - psi_max)
                if err > TOL_BC:
                    errores.append(f"  i={i}: ψ={self.psi[i,Ny]:.6f}  esperado={psi_max:.6f}  err={err:.2e}")
        self.assertEqual(len(errores), 0,
            "ψ ≠ ψ_max en pared superior:\n" + "\n".join(errores))

    # ──────────────────────────────────────────────────────────────────────────
    # Test 4 – Condición de Thom en paredes del canal
    #   ω_w = -2·(ψ_inner - ψ_pared) / h²
    # ──────────────────────────────────────────────────────────────────────────
    def test_06_thom_pared_inferior(self):
        """Condición de Thom satisfecha en j=0 (pared inferior)."""
        max_err = 0.0
        for i in range(1, Nx):
            if kind[i, 0] == WALL:
                omega_thom = -2.0 * (self.psi[i, 1] - 0.0) / h**2
                max_err = max(max_err, abs(self.omega[i, 0] - omega_thom))
        self.assertLess(max_err, TOL_BC,
            f"Error máximo Thom pared inferior: {max_err:.2e}")

    def test_07_thom_pared_superior(self):
        """Condición de Thom satisfecha en j=Ny (pared superior)."""
        max_err = 0.0
        for i in range(1, Nx):
            if kind[i, Ny] == WALL:
                omega_thom = -2.0 * (self.psi[i, Ny-1] - psi_max) / h**2
                max_err = max(max_err, abs(self.omega[i, Ny] - omega_thom))
        self.assertLess(max_err, TOL_BC,
            f"Error máximo Thom pared superior: {max_err:.2e}")

    # ──────────────────────────────────────────────────────────────────────────
    # Test 5 – Ecuación de Poisson en nodos FREE
    #   ∇²ψ + ω = 0  →  Σ vecinos - 4ψ + h²·ω ≈ 0
    # ──────────────────────────────────────────────────────────────────────────
    def test_08_poisson_free(self):
        """Residuo de Poisson ≈ 0 en todos los nodos FREE."""
        max_err = 0.0
        for i in range(1, Nx):
            for j in range(1, Ny):
                if kind[i, j] == FREE:
                    res = (self.psi[i+1,j] + self.psi[i-1,j]
                           + self.psi[i,j+1] + self.psi[i,j-1]
                           - 4.0 * self.psi[i,j]
                           + h**2 * self.omega[i,j])
                    max_err = max(max_err, abs(res))
        self.assertLess(max_err, TOL_BC,
            f"Residuo máximo Poisson en FREE: {max_err:.2e}")

    # ──────────────────────────────────────────────────────────────────────────
    # Test 6 – Condición de Neumann en OUTLET
    #   ψ[Nx,j] = ψ[Nx-1,j]  y  ω[Nx,j] = ω[Nx-1,j]
    # ──────────────────────────────────────────────────────────────────────────
    def test_09_neumann_outlet_psi(self):
        """∂ψ/∂x = 0 en la salida (diferencia nula entre columnas Nx y Nx-1)."""
        max_err = 0.0
        for j in range(Ny + 1):
            if kind[Nx, j] == OUTLET:
                max_err = max(max_err, abs(self.psi[Nx, j] - self.psi[Nx-1, j]))
        self.assertLess(max_err, TOL_BC,
            f"Neumann ψ OUTLET no satisfecho, error máx: {max_err:.2e}")

    def test_10_neumann_outlet_omega(self):
        """∂ω/∂x = 0 en la salida."""
        max_err = 0.0
        for j in range(Ny + 1):
            if kind[Nx, j] == OUTLET:
                max_err = max(max_err, abs(self.omega[Nx, j] - self.omega[Nx-1, j]))
        self.assertLess(max_err, TOL_BC,
            f"Neumann ω OUTLET no satisfecho, error máx: {max_err:.2e}")

    # ──────────────────────────────────────────────────────────────────────────
    # Test 7 – Rango físico de ψ en la región fluido
    #   ψ no debe alejarse más de un 5 % de psi_max por encima ni por debajo.
    #   Nota: el principio de máximo no aplica estrictamente en ψ-ω con Thom,
    #   por lo que pequeños sobrepasos son físicamente esperados en esquinas.
    # ──────────────────────────────────────────────────────────────────────────
    def test_11_rango_fisico_psi(self):
        """ψ no debe superar psi_max ni bajar de 0 en más de un 5 % de psi_max."""
        margen = 0.05 * psi_max
        violaciones = []
        for i in range(Nx + 1):
            for j in range(Ny + 1):
                if kind[i, j] == SOLID:
                    continue
                v = self.psi[i, j]
                if v < -margen or v > psi_max + margen:
                    violaciones.append(f"  ({i},{j}) psi={v:.4f}")
        self.assertEqual(len(violaciones), 0,
            f"psi fuera de rango (+/-5%) en {len(violaciones)} nodos:\n"
            + "\n".join(violaciones[:10])
            + ("\n  ..." if len(violaciones) > 10 else ""))


# =============================================================================
# Resumen impreso al final
# =============================================================================
class _ResumenResult(unittest.TextTestResult):
    def printErrors(self):
        super().printErrors()
        total = self.testsRun
        fallos = len(self.failures) + len(self.errors)
        ok = total - fallos
        print(f"\n{'='*60}")
        print(f"  Resultado: {ok}/{total} tests pasaron")
        if fallos == 0:
            print("  ✓ Solver verificado correctamente.")
        else:
            print(f"  ✗ {fallos} test(s) fallaron.")
        print(f"{'='*60}")


if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite  = loader.loadTestsFromTestCase(TestConvergencia)
    runner = unittest.TextTestRunner(resultclass=_ResumenResult, verbosity=2)
    runner.run(suite)
