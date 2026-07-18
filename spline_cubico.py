# -*- coding: utf-8 -*-
# =============================================================================
#  spline_cubico.py  -  Spline cubico natural (implementacion desde cero)
#
#  Interpolacion de un conjunto de puntos (x_i, y_i), i = 0..n, mediante un
#  spline cubico natural: en cada subintervalo [x_i, x_i+1] se ajusta un
#  polinomio cubico S_i(x), de forma que:
#
#    1. S_i(x_i) = y_i,  S_i(x_i+1) = y_i+1                 (interpolacion)
#    2. S_i'(x_i+1)  = S_i+1'(x_i+1)                        (C1: 1a derivada continua)
#    3. S_i''(x_i+1) = S_i+1''(x_i+1)                       (C2: 2a derivada continua)
#    4. S_0''(x_0) = 0   y   S_{n-1}''(x_n) = 0              (condicion NATURAL)
#
#  Formulacion clasica (Burden & Faires / Kincaid): las incognitas son las
#  segundas derivadas M_i = S''(x_i) en cada nodo. Imponiendo continuidad de
#  la 1a derivada se obtiene un sistema tridiagonal en M_1..M_{n-1}:
#
#    h_{i-1}*M_{i-1} + 2(h_{i-1}+h_i)*M_i + h_i*M_{i+1} = 6*[ (y_{i+1}-y_i)/h_i
#                                                             - (y_i-y_{i-1})/h_{i-1} ]
#
#  con h_i = x_{i+1} - x_i,  y condicion natural M_0 = M_n = 0.
#
#  El sistema tridiagonal se resuelve con el algoritmo de Thomas (O(n), sin
#  necesidad de scipy/numpy.linalg.solve), coherente con el resto del
#  proyecto que resuelve sistemas dispersos "a mano" con matrices en banda.
# =============================================================================

import numpy as np


def _thomas(a, b, c, d):
    """
    Algoritmo de Thomas para sistemas tridiagonales A*x = d.

    a : sub-diagonal   (a[0] no se usa)   - tamano n
    b : diagonal principal                - tamano n
    c : super-diagonal (c[-1] no se usa)  - tamano n
    d : lado derecho                      - tamano n

    Devuelve x (tamano n). Complejidad O(n), estable si la matriz es
    diagonalmente dominante (caso del spline natural).
    """
    n = len(d)
    cp = np.empty(n)
    dp = np.empty(n)
    x = np.empty(n)

    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]
    for i in range(1, n):
        m = b[i] - a[i] * cp[i - 1]
        cp[i] = c[i] / m
        dp[i] = (d[i] - a[i] * dp[i - 1]) / m

    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x


class SplineCubicoNatural:
    """
    Spline cubico natural para interpolar un conjunto de puntos (x, y).

    Uso
    ----
    >>> s = SplineCubicoNatural(x, y)
    >>> y_nuevo = s(x_nuevo)             # evaluacion (escalar o array)
    >>> dy_nuevo = s.derivada(x_nuevo)   # 1a derivada del spline

    Parametros
    ----------
    x : array_like, estrictamente creciente, tamano n+1 (n >= 2)
    y : array_like, mismo tamano que x

    Notas
    -----
    - Condicion NATURAL: M_0 = M_n = 0 (2a derivada nula en los extremos).
      Esto implica que fuera del rango de datos el spline no debe evaluarse
      (se lanza ValueError); interpolacion, no extrapolacion.
    - El sistema para las M_i internas se resuelve con el algoritmo de
      Thomas (tridiagonal, O(n)).
    """

    def __init__(self, x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        if x.shape != y.shape:
            raise ValueError("x e y deben tener la misma forma.")
        if x.ndim != 1:
            raise ValueError("x e y deben ser arreglos 1D.")
        n_pts = x.size
        if n_pts < 3:
            raise ValueError("Se requieren al menos 3 puntos para un spline cubico.")
        if np.any(np.diff(x) <= 0):
            raise ValueError("x debe ser estrictamente creciente (sin duplicados).")

        self.x = x
        self.y = y
        self.n = n_pts - 1          # numero de subintervalos
        self.h = np.diff(x)         # h[i] = x[i+1] - x[i],  i = 0..n-1
        self.M = self._resolver_segundas_derivadas()

    def _resolver_segundas_derivadas(self):
        """Arma y resuelve el sistema tridiagonal para M_1..M_{n-1}."""
        n = self.n
        x, y, h = self.x, self.y, self.h

        M = np.zeros(n + 1)   # M[0] = M[n] = 0 por condicion natural
        if n == 2:
            i = 1
            rhs = 6.0 * ((y[i+1]-y[i]) / h[i] - (y[i]-y[i-1]) / h[i-1])
            M[1] = rhs / (2.0 * (h[i-1] + h[i]))
            return M

        m = n - 1
        a = np.zeros(m)
        b = np.zeros(m)
        c = np.zeros(m)
        d = np.zeros(m)

        for k, i in enumerate(range(1, n)):
            a[k] = h[i - 1]
            b[k] = 2.0 * (h[i - 1] + h[i])
            c[k] = h[i]
            d[k] = 6.0 * ((y[i+1] - y[i]) / h[i] - (y[i] - y[i-1]) / h[i-1])

        M_interior = _thomas(a, b, c, d)
        M[1:n] = M_interior
        return M

    def _localizar(self, xq):
        i = np.searchsorted(self.x, xq, side="right") - 1
        i = np.clip(i, 0, self.n - 1)
        return i

    def _coef_intervalo(self, i):
        """Coeficientes (a,b,c,d) de S_i(x) = a + b(x-x_i) + c(x-x_i)^2 + d(x-x_i)^3."""
        x, y, h, M = self.x, self.y, self.h, self.M
        a = y[i]
        c = M[i] / 2.0
        d = (M[i+1] - M[i]) / (6.0 * h[i])
        b = (y[i+1] - y[i]) / h[i] - h[i] * (2.0*M[i] + M[i+1]) / 6.0
        return a, b, c, d

    def __call__(self, xq):
        return self.evaluar(xq)

    def evaluar(self, xq):
        """Evalua el spline en xq (escalar o array). Solo interpolacion (no extrapola)."""
        xq_arr = np.atleast_1d(np.asarray(xq, dtype=float))
        if np.any(xq_arr < self.x[0] - 1e-12) or np.any(xq_arr > self.x[-1] + 1e-12):
            raise ValueError("xq fuera del rango de interpolacion "
                              f"[{self.x[0]}, {self.x[-1]}].")

        out = np.empty_like(xq_arr)
        idx = self._localizar(xq_arr)
        for k, i in enumerate(idx):
            a, b, c, d = self._coef_intervalo(i)
            dx = xq_arr[k] - self.x[i]
            out[k] = a + b*dx + c*dx**2 + d*dx**3

        return out.item() if np.isscalar(xq) or np.asarray(xq).ndim == 0 else out

    def derivada(self, xq):
        """Evalua S'(x) (primera derivada del spline) en xq."""
        xq_arr = np.atleast_1d(np.asarray(xq, dtype=float))
        if np.any(xq_arr < self.x[0] - 1e-12) or np.any(xq_arr > self.x[-1] + 1e-12):
            raise ValueError("xq fuera del rango de interpolacion "
                              f"[{self.x[0]}, {self.x[-1]}].")

        out = np.empty_like(xq_arr)
        idx = self._localizar(xq_arr)
        for k, i in enumerate(idx):
            a, b, c, d = self._coef_intervalo(i)
            dx = xq_arr[k] - self.x[i]
            out[k] = b + 2.0*c*dx + 3.0*d*dx**2

        return out.item() if np.isscalar(xq) or np.asarray(xq).ndim == 0 else out


# =============================================================================
#  Extension 2D - reconstruccion separable fila-columna
#
#  Para recuperar un campo 2D (p. ej. psi u omega) definido en una malla
#  GRUESA (Nx_c+1, Ny_c+1) a la resolucion original fina (Nx_f+1, Ny_f+1),
#  se aplica el spline cubico natural de forma SEPARABLE (igual que un
#  "bicubico" por splines 1D sucesivos):
#
#    1. Para cada fila j de la malla gruesa, se ajusta un spline en x usando
#       los Nx_c+1 valores de esa fila, y se evalua en los Nx_f+1 puntos finos
#       -> array intermedio de forma (Nx_f+1, Ny_c+1).
#    2. Para cada columna i de ese intermedio, se ajusta un spline en y y se
#       evalua en los Ny_f+1 puntos finos -> campo final (Nx_f+1, Ny_f+1).
#
#  Es el metodo estandar para interpolar un campo 2D con splines 1D cuando
#  la malla es rectangular (tensorial), sin necesitar un spline bidimensional
#  propiamente dicho.
# =============================================================================

def reconstruir_campo_2d(campo_coarse, x_coarse, y_coarse, x_fino, y_fino):
    """
    Reconstruye un campo 2D de una malla gruesa a una malla fina mediante
    spline cubico natural separable (spline en x, luego spline en y).

    Parametros
    ----------
    campo_coarse : ndarray (len(x_coarse), len(y_coarse))
    x_coarse, y_coarse : coordenadas de la malla gruesa (crecientes)
    x_fino, y_fino      : coordenadas donde se quiere el campo reconstruido

    Devuelve
    --------
    campo_fino : ndarray (len(x_fino), len(y_fino))
    """
    nxc, nyc = len(x_coarse), len(y_coarse)
    nxf, nyf = len(x_fino), len(y_fino)

    intermedio = np.empty((nxf, nyc))
    for j in range(nyc):
        s = SplineCubicoNatural(x_coarse, campo_coarse[:, j])
        intermedio[:, j] = s.evaluar(x_fino)

    campo_fino = np.empty((nxf, nyf))
    for i in range(nxf):
        s = SplineCubicoNatural(y_coarse, intermedio[i, :])
        campo_fino[i, :] = s.evaluar(y_fino)

    return campo_fino


# =============================================================================
# Demostracion rapida al ejecutar el modulo directamente
# =============================================================================
if __name__ == "__main__":
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([0.0, 2.5, 2.0, 3.5, 1.0, 2.0])

    s = SplineCubicoNatural(x, y)

    print("Verificacion de interpolacion en los nodos:")
    for xi, yi in zip(x, y):
        print(f"  S({xi}) = {s(xi):.6f}   (esperado {yi})")

    print("\nSegundas derivadas en los extremos (condicion natural, deben ser ~0):")
    print(f"  M[0] = {s.M[0]:.3e}   M[n] = {s.M[-1]:.3e}")
