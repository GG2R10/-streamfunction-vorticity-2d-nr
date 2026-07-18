# -*- coding: utf-8 -*-
# =============================================================================
#  comparacion_metodos_lineales.py  —  Comparación de métodos para el paso
#                                       de Newton  J·Δx = -F
#
#  Objetivo 2 del enunciado (sugerencia): resolver el sistema lineal que
#  aparece en cada iteración de Newton-Raphson con al menos 3 métodos vistos
#  en clase y comparar resultados. Métodos:
#
#    - Directo        : scipy.sparse.linalg.spsolve (LU sparse; el que usa el
#                       solver del proyecto — sirve de referencia exacta)
#    - Jacobi         : x_{k+1} = x_k + D^{-1}(rhs - J x_k)
#    - Gauss-Seidel   : (L+D) x_{k+1} = rhs - U x_k  (barrido triangular)
#    - Gradiente Conjugado (CG): scipy.sparse.linalg.cg. OJO: CG requiere
#                       matriz simétrica definida positiva y J NO lo es
#                       (la convección upwind y las filas de frontera rompen
#                       la simetría); se incluye justamente para mostrar qué
#                       pasa cuando no se cumple la hipótesis del método.
#
#  El sistema usa el solver del proyecto tal cual (simulacion.py, con la
#  malla que dicte FACTOR). Para que la comparación sea significativa, el
#  sistema se congela en un estado NO convergido: la solución convergida
#  para R = 0.5 evaluada con la física de R = 2. Así |F| es sustancial y el
#  paso de Newton es "difícil" de verdad, no un sistema ya casi resuelto.
#
#  Salidas: tabla en consola + tabla_metodos_lineales.csv +
#           convergencia_metodos_lineales.png
# =============================================================================

import time

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import tril
from scipy.sparse.linalg import spsolve, cg, spsolve_triangular

import simulacion as sim

R_PREVIO    = 0.5    # estado congelado: solución convergida a este R...
R_CONGELADO = 2.0    # ...evaluada con la física de este R
TOL         = 1e-8
MAX_ITER    = 3000


def jacobi(J, rhs, x0, tol=TOL, max_iter=MAX_ITER):
    D = J.diagonal()
    x = x0.copy()
    historia = []
    b_norm = np.linalg.norm(rhs) + 1e-30
    for it in range(1, max_iter + 1):
        r = rhs - J @ x
        res = np.linalg.norm(r) / b_norm
        historia.append(res)
        if res < tol:
            return x, historia, it, True
        x = x + r / D
        if not np.all(np.isfinite(x)):
            return x, historia, it, False
    return x, historia, max_iter, False


def gauss_seidel(J, rhs, x0, tol=TOL, max_iter=MAX_ITER):
    L = tril(J, format="csr")          # triangular inferior con diagonal
    U = (J - L).tocsr()
    x = x0.copy()
    historia = []
    b_norm = np.linalg.norm(rhs) + 1e-30
    for it in range(1, max_iter + 1):
        r = rhs - J @ x
        res = np.linalg.norm(r) / b_norm
        historia.append(res)
        if res < tol:
            return x, historia, it, True
        x = spsolve_triangular(L, rhs - U @ x, lower=True)
        if not np.all(np.isfinite(x)):
            return x, historia, it, False
    return x, historia, max_iter, False


def main():
    print("=" * 70)
    print(f"  Estado congelado: solución @ R = {R_PREVIO} evaluada con "
          f"R = {R_CONGELADO}")
    print(f"  Malla: {sim.Nx}x{sim.Ny} (FACTOR = {sim.FACTOR})")
    print("=" * 70)
    psi, omega = sim.re_continuation(R_PREVIO)
    F = sim.compute_F(psi, omega, R_CONGELADO)
    J = sim.compute_J(psi, omega, R_CONGELADO)
    rhs = -F
    print(f"\n  |F| del sistema congelado = {np.linalg.norm(F):.3e}   "
          f"(sistema {J.shape[0]}x{J.shape[0]}, no nulos = {J.nnz})")

    x0 = np.zeros(rhs.size)
    resultados = {}

    t0 = time.perf_counter()
    x_dir = spsolve(J, rhs)
    t_dir = time.perf_counter() - t0
    res_dir = np.linalg.norm(rhs - J @ x_dir) / (np.linalg.norm(rhs) + 1e-30)
    resultados["directo"] = dict(tiempo=t_dir, iters=1, convergio=True,
                                 error_rel=0.0, historia=[res_dir])

    for nombre, metodo in [("jacobi", jacobi), ("gauss_seidel", gauss_seidel)]:
        t0 = time.perf_counter()
        x_m, hist, iters, ok = metodo(J, rhs, x0)
        t_m = time.perf_counter() - t0
        err = np.linalg.norm(x_m - x_dir) / (np.linalg.norm(x_dir) + 1e-30)
        resultados[nombre] = dict(tiempo=t_m, iters=iters, convergio=ok,
                                  error_rel=err, historia=hist)

    hist_cg = []
    b_norm = np.linalg.norm(rhs) + 1e-30
    t0 = time.perf_counter()
    x_cg, info = cg(J, rhs, x0=x0, atol=TOL * b_norm, maxiter=MAX_ITER,
                    callback=lambda xk: hist_cg.append(
                        np.linalg.norm(rhs - J @ xk) / b_norm))
    t_cg = time.perf_counter() - t0
    err_cg = np.linalg.norm(x_cg - x_dir) / (np.linalg.norm(x_dir) + 1e-30)
    resultados["cg"] = dict(tiempo=t_cg, iters=len(hist_cg),
                            convergio=(info == 0), error_rel=err_cg,
                            historia=hist_cg)

    print(f"\n  {'Método':<14}{'Convergió':<12}{'Iters':<10}"
          f"{'Tiempo (s)':<14}{'Error rel. vs directo':<22}")
    print("  " + "-" * 68)
    for nombre in ["directo", "jacobi", "gauss_seidel", "cg"]:
        r = resultados[nombre]
        print(f"  {nombre:<14}{str(r['convergio']):<12}{r['iters']:<10}"
              f"{r['tiempo']:<14.4e}{r['error_rel']:<22.4e}")

    with open("tabla_metodos_lineales.csv", "w", encoding="utf-8") as f:
        f.write("metodo,convergio,iteraciones,tiempo_s,error_relativo\n")
        for nombre in ["directo", "jacobi", "gauss_seidel", "cg"]:
            r = resultados[nombre]
            f.write(f"{nombre},{r['convergio']},{r['iters']},"
                    f"{r['tiempo']},{r['error_rel']}\n")
    print("\n  Tabla guardada en tabla_metodos_lineales.csv")

    # ── Historia de convergencia ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7.5, 5))
    estilos = {"jacobi": ("Jacobi", "-"), "gauss_seidel": ("Gauss-Seidel", "-"),
               "cg": ("Gradiente Conjugado", "--")}
    for nombre, (etiqueta, ls) in estilos.items():
        hist = resultados[nombre]["historia"]
        ax.semilogy(range(1, len(hist) + 1), hist, ls, lw=1.5, label=etiqueta)
    ax.axhline(TOL, color="k", ls=":", lw=1, label=f"tolerancia = {TOL:.0e}")
    ax.set_xlabel("Iteración")
    ax.set_ylabel(r"Residuo relativo  $\|rhs - J\,x_k\| / \|rhs\|$")
    ax.set_title(f"Convergencia de los métodos iterativos  —  "
                 f"J·Δx = -F congelado (malla {sim.Nx}x{sim.Ny})")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which="both")
    plt.tight_layout()
    plt.savefig("convergencia_metodos_lineales.png", dpi=140)
    print("  Figura guardada: convergencia_metodos_lineales.png")


if __name__ == "__main__":
    main()
