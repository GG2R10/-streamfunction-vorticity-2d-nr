# =============================================================================
#  visualizacion.py  —  Módulo de visualización
#  Stream-function / Vorticity 2D Solver
#
#  Contiene únicamente la función plot_results, separada de la lógica del
#  solver para mantener el código modular y fácil de mantener.
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def plot_results(psi, omega, Re,
                 kind, Nx, Ny, h, SOLID,
                 B1_ic, B1_fc, B1_jb, B1_jt,
                 B2_ic, B2_fc, B2_jb, B2_jt):
    """
    Produce figura con 3 paneles:
      - Función de corriente (mapa de color + contornos de streamlines)
      - Vorticidad (mapa de color)
      - Velocidades derivadas (ux = ∂ψ/∂y, uy = -∂ψ/∂x) como quiver

    Parámetros
    ----------
    psi, omega : np.ndarray  —  campos solución de forma (Nx+1, Ny+1)
    Re         : float       —  número de Reynolds de la solución
    kind       : np.ndarray  —  clasificación de nodos (Nx+1, Ny+1)
    Nx, Ny     : int         —  dimensiones de la malla
    h          : float       —  espaciado de la malla
    SOLID      : int         —  constante que identifica nodos sólidos
    B1_ic, B1_fc, B1_jb, B1_jt : int  —  índices del Bloque 1
    B2_ic, B2_fc, B2_jb, B2_jt : int  —  índices del Bloque 2
    """
    fluid_mask = (kind != SOLID)

    # Calcular campo de velocidades por diferencias centradas
    ux = np.zeros((Nx + 1, Ny + 1))
    uy = np.zeros((Nx + 1, Ny + 1))
    for i in range(1, Nx):
        for j in range(1, Ny):
            if fluid_mask[i, j]:
                ux[i, j] =  (psi[i, j+1] - psi[i, j-1]) / (2 * h)
                uy[i, j] = -(psi[i+1, j] - psi[i-1, j]) / (2 * h)

    fig, axes = plt.subplots(3, 1, figsize=(18, 12))
    fig.suptitle(f"Flujo en canal con obstáculos  —  Re = {Re:.2f}",
                 fontsize=14, fontweight='bold')

    configs = [
        (psi,   "Función de corriente  ψ", "RdBu_r",  True),
        (omega, "Vorticidad  ω",            "seismic", False),
    ]

    for ax, (field, title, cmap, do_contour) in zip(axes[:2], configs):
        fld = np.ma.array(field.T, mask=~fluid_mask.T)
        im  = ax.imshow(fld, origin="lower", aspect="auto",
                        cmap=cmap, extent=[0, Nx, 0, Ny])
        plt.colorbar(im, ax=ax, fraction=0.015, pad=0.02)

        # Bloques sólidos
        for (ic, fc, jb, jt, lbl) in [
            (B1_ic, B1_fc, B1_jb, B1_jt, "Bloque 1"),
            (B2_ic, B2_fc, B2_jb, B2_jt, "Bloque 2"),
        ]:
            ax.add_patch(mpatches.Rectangle(
                (ic, jb), fc - ic + 1, jt - jb + 1,
                lw=1.5, edgecolor="k", facecolor="dimgray", label=lbl))

        # Contornos de streamlines
        if do_contour:
            X_c = np.arange(Nx + 1)
            Y_c = np.arange(Ny + 1)
            try:
                ax.contour(X_c, Y_c, np.ma.array(psi.T, mask=~fluid_mask.T),
                           levels=25, colors="k", linewidths=0.5, alpha=0.6)
            except Exception:
                pass

        ax.set_title(title, fontsize=11)
        ax.set_xlabel("x  (columnas)")
        ax.set_ylabel("y  (filas)")
        ax.legend(loc="upper right", fontsize=8)

    # Panel 3: campo de velocidades (quiver submuestreado)
    ax   = axes[2]
    step = 5    # submuestreo para no saturar la figura
    xi   = np.arange(0, Nx + 1, step)
    yi   = np.arange(0, Ny + 1, step)
    Xi, Yi = np.meshgrid(xi, yi)
    Ui = np.ma.array(ux[xi[:, None], yi[None, :]].T,
                     mask=~fluid_mask[xi[:, None], yi[None, :]].T)
    Vi = np.ma.array(uy[xi[:, None], yi[None, :]].T,
                     mask=~fluid_mask[xi[:, None], yi[None, :]].T)
    ax.quiver(Xi, Yi, Ui, Vi, scale=30, width=0.002, alpha=0.8)

    for (ic, fc, jb, jt, lbl) in [
        (B1_ic, B1_fc, B1_jb, B1_jt, "Bloque 1"),
        (B2_ic, B2_fc, B2_jb, B2_jt, "Bloque 2"),
    ]:
        ax.add_patch(mpatches.Rectangle(
            (ic, jb), fc - ic + 1, jt - jb + 1,
            lw=1.5, edgecolor="k", facecolor="dimgray", label=lbl))

    ax.set_xlim(0, Nx)
    ax.set_ylim(0, Ny)
    ax.set_title("Campo de velocidades  (ux, uy)", fontsize=11)
    ax.set_xlabel("x  (columnas)")
    ax.set_ylabel("y  (filas)")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_aspect("equal")

    plt.tight_layout()
    fname = f"flujo_Re{Re:.0f}.png"
    plt.savefig(fname, dpi=150)
    plt.show()
    print(f"\nFigura guardada: {fname}")
