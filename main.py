# =============================================================================
#  main.py  —  Punto de entrada principal
#  Stream-function / Vorticity 2D Solver
#
#  Orquesta la lógica del solver (simulacion.py) y la visualización
#  (visualizacion.py). Modifica Re_objetivo aquí para explorar otros Reynolds.
# =============================================================================

from simulacion import (re_continuation,
                        kind, Nx, Ny, h, SOLID,
                        B1_ic, B1_fc, B1_jb, B1_jt,
                        B2_ic, B2_fc, B2_jb, B2_jt)
from visualizacion import plot_results

# ── Parámetro principal ───────────────────────────────────────────────────────
Re_objetivo = 10   # ← cambia aquí para explorar otros Reynolds

# ── Resolver ─────────────────────────────────────────────────────────────────
psi_sol, omega_sol = re_continuation(Re_objetivo)

# ── Visualizar ───────────────────────────────────────────────────────────────
plot_results(psi_sol, omega_sol, Re_objetivo,
             kind, Nx, Ny, h, SOLID,
             B1_ic, B1_fc, B1_jb, B1_jt,
             B2_ic, B2_fc, B2_jb, B2_jt)
