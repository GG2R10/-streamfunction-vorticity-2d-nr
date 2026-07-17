# =============================================================================
#  interpolacion.py  —  Refinamiento de malla con splines cúbicos naturales
#  Stream-function / Vorticity 2D Solver
#
#  El solver resuelve en una malla gruesa (FACTOR veces menor, ver
#  simulacion.py); aquí la solución se devuelve a la malla original
#  interpolando con scipy.interpolate.CubicSpline (bc_type="natural",
#  es decir S'' = 0 en los extremos).
# =============================================================================

import numpy as np
from scipy.interpolate import CubicSpline


def refinar_campo(campo, Nx_f, Ny_f):
    """
    Interpola un campo escalar de la malla gruesa a la fina.

    Producto tensorial de splines 1D: primero a lo largo de x (todas las
    filas a la vez con axis=0) y después a lo largo de y. En los nodos
    coincidentes de ambas mallas el valor original se conserva exacto.

    Parámetros
    ----------
    campo      : np.ndarray — campo en la malla gruesa, forma (Nx_c+1, Ny_c+1)
    Nx_f, Ny_f : int        — dimensiones de la malla fina destino

    Retorna
    -------
    np.ndarray de forma (Nx_f+1, Ny_f+1)
    """
    Nx_c, Ny_c = campo.shape[0] - 1, campo.shape[1] - 1

    # Coordenadas normalizadas [0, 1]: ambas mallas cubren el mismo dominio
    x_c, x_f = np.linspace(0, 1, Nx_c + 1), np.linspace(0, 1, Nx_f + 1)
    y_c, y_f = np.linspace(0, 1, Ny_c + 1), np.linspace(0, 1, Ny_f + 1)

    parcial = CubicSpline(x_c, campo, axis=0, bc_type="natural")(x_f)
    return CubicSpline(y_c, parcial, axis=1, bc_type="natural")(y_f)


def kind_malla_fina(Nx_f, Ny_f, bloques, SOLID, FREE):
    """
    Clasificación mínima (SOLID / fluido) en la malla fina, suficiente para
    que plot_results enmascare los obstáculos al graficar.

    bloques : iterable de tuplas (ic, fc, jb, jt) en índices de malla fina.
    """
    kind_f = np.full((Nx_f + 1, Ny_f + 1), FREE, dtype=int)
    for ic, fc, jb, jt in bloques:
        kind_f[ic:fc + 1, jb:jt + 1] = SOLID
    return kind_f
