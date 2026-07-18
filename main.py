# =============================================================================
#  main.py  —  Punto de entrada principal
#  Stream-function / Vorticity 2D Solver
#
#  Orquesta la lógica del solver (simulacion.py), el refinamiento por splines
#  cúbicos naturales (interpolacion.py) y la visualización (visualizacion.py).
#  Modifica Re_objetivo aquí para explorar otros Reynolds, y FACTOR en
#  simulacion.py para cambiar la reducción de malla.
# =============================================================================

from simulacion import (re_continuation,
                        kind, Nx, Ny, h, SOLID, FREE,
                        FACTOR, NX_F, NY_F, B1_FINO, B2_FINO,
                        B1_ic, B1_fc, B1_jb, B1_jt,
                        B2_ic, B2_fc, B2_jb, B2_jt)
from visualizacion import plot_results
from interpolacion import refinar_campo, kind_malla_fina

# ── Parámetro principal ───────────────────────────────────────────────────────
Re_objetivo = 20   # ← cambia aquí para explorar otros Reynolds

# ── Resolver (en la malla gruesa si FACTOR > 1) ──────────────────────────────
psi_sol, omega_sol = re_continuation(Re_objetivo)

# ── Refinar a la malla original con splines cúbicos naturales ────────────────
if FACTOR > 1:
    print(f"\nRefinando {Nx}x{Ny} -> {NX_F}x{NY_F} "
          f"con splines cúbicos naturales (FACTOR = {FACTOR})...")
    psi_fina   = refinar_campo(psi_sol,   NX_F, NY_F)
    omega_fina = refinar_campo(omega_sol, NX_F, NY_F)
    kind_fina  = kind_malla_fina(NX_F, NY_F, (B1_FINO, B2_FINO), SOLID, FREE)

    plot_results(psi_fina, omega_fina, Re_objetivo,
                 kind_fina, NX_F, NY_F, 1.0, SOLID,
                 *B1_FINO, *B2_FINO)
else:
    plot_results(psi_sol, omega_sol, Re_objetivo,
                 kind, Nx, Ny, h, SOLID,
                 B1_ic, B1_fc, B1_jb, B1_jt,
                 B2_ic, B2_fc, B2_jb, B2_jt)
