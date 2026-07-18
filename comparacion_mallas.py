# -*- coding: utf-8 -*-
# =============================================================================
#  comparacion_mallas.py  —  Verificación de la reconstrucción por spline
#
#  Resuelve el MISMO problema dos veces con el solver del proyecto:
#    - Referencia : malla fina 200x20 (FACTOR = 1)
#    - Reducida   : malla gruesa según el FACTOR configurado en simulacion.py
#  reconstruye la solución gruesa a 200x20 con el spline cúbico natural
#  (interpolacion.py / spline_cubico.py) y compara ambas:
#
#    - Estadísticas de error |reconstruida - referencia| para ψ y ω, en todo
#      el fluido y lejos de los obstáculos (el error pegado a los bloques es
#      en gran parte desajuste GEOMÉTRICO por el redondeo de índices al
#      escalar, no error del spline).
#    - Figura 3x3: (ψ, ω, |v|) x (referencia, reconstruida, error), con la
#      misma escala de color en referencia y reconstruida para que sean
#      comparables, y ω con la normalización ±p98 centrada en 0 de
#      visualizacion.py (sin ella, los picos singulares de las esquinas
#      aplastan el resto del campo hacia el blanco).
#
#  Nota: simulacion.py construye su malla al importarse, con el FACTOR
#  escrito en el archivo. Para tener ambas resoluciones en un mismo proceso,
#  la referencia fina se carga re-ejecutando el módulo con FACTOR = 1.
# =============================================================================

import pathlib
import re as regex
import types

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import TwoSlopeNorm
from scipy.ndimage import binary_dilation

import simulacion as sim_gruesa          # malla según FACTOR del archivo
from interpolacion import refinar_campo

RE_COMPARACION = 10


def cargar_simulacion_con_factor(factor):
    """Re-ejecuta simulacion.py con otro FACTOR, como módulo independiente."""
    src = pathlib.Path(__file__).with_name("simulacion.py").read_text(encoding="utf-8")
    src = regex.sub(r"^FACTOR = \d+", f"FACTOR = {factor}", src, count=1,
                    flags=regex.M)
    mod = types.ModuleType(f"simulacion_factor{factor}")
    exec(compile(src, "simulacion.py", "exec"), mod.__dict__)
    return mod


def velocidad(psi, Nx, Ny, h, fluid_mask):
    ux = np.zeros_like(psi)
    uy = np.zeros_like(psi)
    for i in range(1, Nx):
        for j in range(1, Ny):
            if fluid_mask[i, j]:
                ux[i, j] = (psi[i, j+1] - psi[i, j-1]) / (2*h)
                uy[i, j] = -(psi[i+1, j] - psi[i-1, j]) / (2*h)
    return np.sqrt(ux**2 + uy**2)


def resumen_error(err, mask, nombre):
    p = np.percentile(err[mask], [50, 90, 99, 100])
    print(f"  Error {nombre:<6}: mediana={p[0]:.3e}  p90={p[1]:.3e}  "
          f"p99={p[2]:.3e}  max={p[3]:.3e}")
    return p


def main():
    if sim_gruesa.FACTOR == 1:
        print("FACTOR = 1 en simulacion.py: no hay malla gruesa que comparar.")
        return

    print("=" * 70)
    print(f"  PASO 1 — Referencia fina {sim_gruesa.NX_F}x{sim_gruesa.NY_F} (FACTOR = 1)")
    print("=" * 70)
    sim_fina = cargar_simulacion_con_factor(1)
    psi_ref, om_ref = sim_fina.re_continuation(RE_COMPARACION)

    print("\n" + "=" * 70)
    print(f"  PASO 2 — Malla gruesa {sim_gruesa.Nx}x{sim_gruesa.Ny} "
          f"(FACTOR = {sim_gruesa.FACTOR}) + spline cúbico natural")
    print("=" * 70)
    psi_g, om_g = sim_gruesa.re_continuation(RE_COMPARACION)
    psi_rec = refinar_campo(psi_g, sim_gruesa.NX_F, sim_gruesa.NY_F)
    om_rec = refinar_campo(om_g, sim_gruesa.NX_F, sim_gruesa.NY_F)

    Nx_f, Ny_f = sim_gruesa.NX_F, sim_gruesa.NY_F
    fluid_mask = (sim_fina.kind != sim_fina.SOLID)

    # Franja de ~1 celda gruesa alrededor de los sólidos: ahí el borde del
    # bloque escalado no coincide exactamente con el fino (redondeo de índices)
    solido_dilatado = binary_dilation(~fluid_mask,
                                     iterations=int(sim_gruesa.FACTOR))
    lejos_mask = fluid_mask & ~solido_dilatado

    err_psi = np.abs(psi_rec - psi_ref)
    err_om = np.abs(om_rec - om_ref)

    print("\n" + "=" * 70)
    print("  PASO 3 — Error de la reconstrucción vs la referencia fina")
    print("=" * 70)
    print("  -- en todo el dominio fluido --")
    resumen_error(err_psi, fluid_mask, "psi")
    resumen_error(err_om, fluid_mask, "omega")
    print("  -- lejos de los obstáculos (excluye la franja de redondeo) --")
    resumen_error(err_psi, lejos_mask, "psi")
    resumen_error(err_om, lejos_mask, "omega")

    # ── Figura 3x3 ───────────────────────────────────────────────────────────
    vel_ref = velocidad(psi_ref, Nx_f, Ny_f, sim_fina.h, fluid_mask)
    vel_rec = velocidad(psi_rec, Nx_f, Ny_f, sim_fina.h, fluid_mask)

    om_lim = np.percentile(np.abs(om_ref[fluid_mask]), 98)
    norm_om = TwoSlopeNorm(vmin=-om_lim, vcenter=0.0, vmax=om_lim)
    norm_psi = plt.Normalize(psi_ref[fluid_mask].min(), psi_ref[fluid_mask].max())
    norm_vel = plt.Normalize(0.0, vel_ref[fluid_mask].max())

    filas = [
        ("ψ", psi_ref, psi_rec, err_psi, "RdBu_r", norm_psi),
        ("ω", om_ref, om_rec, err_om, "seismic", norm_om),
        ("|v|", vel_ref, vel_rec, np.abs(vel_rec - vel_ref), "viridis", norm_vel),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(16, 10))
    ext = [-0.5, Nx_f + 0.5, -0.5, Ny_f + 0.5]

    for fila, (nombre, ref, rec, err, cmap, norm) in zip(axes, filas):
        paneles = [
            (ref, f"{nombre}  —  referencia fina (FACTOR = 1)", cmap, norm),
            (rec, f"{nombre}  —  malla gruesa + spline cúbico natural", cmap, norm),
            (err, f"{nombre}  —  error absoluto", "inferno", None),
        ]
        for ax, (campo, titulo, cm, nm) in zip(fila, paneles):
            fld = np.ma.array(campo.T, mask=~fluid_mask.T)
            im = ax.imshow(fld, origin="lower", aspect="auto",
                           cmap=cm, norm=nm, extent=ext)
            plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02,
                         extend="both" if nm is norm_om else "neither")
            for (ic, fc, jb, jt) in (sim_gruesa.B1_FINO, sim_gruesa.B2_FINO):
                ax.add_patch(mpatches.Rectangle(
                    (ic - 0.5, jb - 0.5), fc - ic + 1, jt - jb + 1,
                    lw=1.0, edgecolor="k", facecolor="dimgray"))
            ax.set_title(titulo, fontsize=9)
            ax.set_xlim(ext[0], ext[1])
            ax.set_ylim(ext[2], ext[3])

    fig.suptitle(f"Malla fina vs malla gruesa + spline cúbico natural  —  "
                 f"Re = {RE_COMPARACION}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fname = f"comparacion_mallas_Re{RE_COMPARACION}.png"
    plt.savefig(fname, dpi=140)
    print(f"\n  Figura guardada: {fname}")


if __name__ == "__main__":
    main()
