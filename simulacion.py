# =============================================================================
#  canal_psi_omega_NR.py  —  Stream-function / Vorticity 2D Solver
#  Navier-Stokes 2D incompresible  |  Newton-Raphson + Line Search + Re-continuation
#
#  Arquitectura de nodos:
#    SOLID   → interior del bloque sólido          (ψ=cte, sin PDE)
#    INLET   → columna i=0                         (Dirichlet ψ y ω)
#    OUTLET  → columna i=Nx                        (Neumann ψ y ω, activo en NR)
#    WALL    → bordes j=0 y j=Ny del canal         (Dirichlet ψ, Thom ω, activo)
#    BC_WALL → fluido adyacente a obstáculo sólido (Poisson ψ, Thom ω, activo)
#    FREE    → interior libre                      (PDE completa, activo)
#
#  Correcciones respecto a v5:
#    1. Separación completa de tipos de borde (INLET/OUTLET/WALL/BC_WALL/FREE)
#    2. Thom en paredes del canal (WALL), no solo en obstáculos
#    3. OUTLET como nodo activo con ecuación Neumann explícita en F y J
#    4. Backtracking con copia limpia en cada alpha (bug crítico corregido)
#    5. Detección de divergencia más sensible (factor 10)
#    6. Jacobiana 100% consistente con F para todos los tipos de nodo
# =============================================================================

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve

# =============================================================================
# 0. PARÁMETROS
# =============================================================================
Nx = 200          # columnas: i ∈ [0, Nx]
Ny = 20           # filas:    j ∈ [0, Ny]
h  = 1.0          # espaciado uniforme
V0 = 1.0          # velocidad de referencia

# Bloque 1: esquina superior izquierda (pegado a borde izq. y borde sup.)
B1_ic, B1_fc = 0,  8     # columnas [inclusivo, inclusivo]
B1_jb, B1_jt = 13, 20    # filas    [inclusivo, inclusivo]

# Bloque 2: zona inferior central (pegado al borde inferior)
B2_ic, B2_fc = 90, 98    # columnas
B2_jb, B2_jt = 0,  7     # filas

psi_max = V0 * Ny * h    # ψ en techo y Bloque 1 (misma línea de corriente)

# =============================================================================
# 1. CLASIFICACIÓN DE NODOS
# =============================================================================
#  Tabla de ecuaciones por tipo:
#  ┌─────────┬───────────────────────────────┬──────────────────────────────┐
#  │  Tipo   │  Ecuación ψ                   │  Ecuación ω                  │
#  ├─────────┼───────────────────────────────┼──────────────────────────────┤
#  │  FREE   │  Poisson: ∇²ψ + ω = 0        │  Transporte: ∇²ω - Re/4*J=0 │
#  │  BC_WALL│  Poisson: ∇²ψ + ω = 0        │  Thom: ω + 2(ψ_in-ψ_s)/h²=0│
#  │  WALL   │  Dirichlet: ψ = ψ_pared       │  Thom: ω + 2(ψ_in-ψ_w)/h²=0│
#  │  OUTLET │  Neumann: ψ[Nx]-ψ[Nx-1]=0    │  Neumann: ω[Nx]-ω[Nx-1]=0  │
#  │  INLET  │  Dirichlet: ψ = V0*j*h        │  Dirichlet: ω = 0            │
#  │  SOLID  │  ψ = cte (no PDE)             │  (no PDE)                    │
#  └─────────┴───────────────────────────────┴──────────────────────────────┘

SOLID   = 0
INLET   = 1
OUTLET  = 2
WALL    = 3    # pared del canal (j=0 o j=Ny), activa en NR
BC_WALL = 4    # fluido adyacente a obstáculo sólido, activa en NR
FREE    = 5

kind = np.full((Nx + 1, Ny + 1), FREE, dtype=int)

# --- Sólidos ---
kind[B1_ic:B1_fc + 1, B1_jb:B1_jt + 1] = SOLID
kind[B2_ic:B2_fc + 1, B2_jb:B2_jt + 1] = SOLID

# --- Bordes del dominio (sobre nodos no-sólidos) ---
for i in range(Nx + 1):
    for j in range(Ny + 1):
        if kind[i, j] == SOLID:
            continue
        if i == 0:
            kind[i, j] = INLET
        elif i == Nx:
            kind[i, j] = OUTLET
        elif j == 0 or j == Ny:
            kind[i, j] = WALL    # pared inferior o superior del canal

# --- BC_WALL: nodo FREE adyacente a un SOLID ---
for i in range(1, Nx):
    for j in range(1, Ny):
        if kind[i, j] != FREE:
            continue
        for (ni, nj) in [(i+1,j),(i-1,j),(i,j+1),(i,j-1)]:
            if kind[ni, nj] == SOLID:
                kind[i, j] = BC_WALL
                break

# --- Reporte ---
counts = {name: int(np.sum(kind == val))
          for name, val in [("FREE",FREE),("BC_WALL",BC_WALL),
                             ("WALL",WALL),("OUTLET",OUTLET),
                             ("INLET",INLET),("SOLID",SOLID)]}
print("Clasificación de nodos:")
for name, cnt in counts.items():
    print(f"  {name:8s}: {cnt:5d}")
print(f"  {'TOTAL':8s}: {(Nx+1)*(Ny+1):5d}")

# =============================================================================
# 2. NUMERACIÓN  —  activos: FREE, BC_WALL, WALL, OUTLET
# =============================================================================
ACTIVE = (FREE, BC_WALL, WALL, OUTLET)

node_id = -np.ones((Nx + 1, Ny + 1), dtype=int)
idx = 0
for i in range(Nx + 1):
    for j in range(Ny + 1):
        if kind[i, j] in ACTIVE:
            node_id[i, j] = idx
            idx += 1

N_act = idx
N_dof = 2 * N_act
print(f"\nNodos activos (Newton): {N_act}")
print(f"Grados de libertad:     {N_dof}")

def is_active(i, j):
    return 0 <= i <= Nx and 0 <= j <= Ny and kind[i, j] in ACTIVE

# =============================================================================
# 3. VALORES FIJOS DE CONTORNO  (INLET y SOLID solamente)
# =============================================================================
def psi_fixed(i, j):
    """Retorna el valor fijo de ψ para nodos INLET o SOLID."""
    if kind[i, j] == INLET:
        return V0 * j * h
    if kind[i, j] == SOLID:
        if B1_ic <= i <= B1_fc and B1_jb <= j <= B1_jt:
            return psi_max     # Bloque 1 conectado al techo
        if B2_ic <= i <= B2_fc and B2_jb <= j <= B2_jt:
            return 0.0         # Bloque 2 conectado al piso
    if kind[i, j] == WALL:
        return 0.0 if j == 0 else psi_max
    return 0.0

def apply_bc_inlet_solid(psi, omega):
    """
    Impone SOLO los valores FIJOS (INLET y SOLID).
    WALL, OUTLET, BC_WALL son activos → los actualiza Newton, no esta función.
    """
    # Sólidos
    psi[B1_ic:B1_fc+1, B1_jb:B1_jt+1] = psi_max
    psi[B2_ic:B2_fc+1, B2_jb:B2_jt+1] = 0.0

    # Entrada
    for j in range(Ny + 1):
        if kind[0, j] == INLET:
            psi[0, j]   = V0 * j * h
            omega[0, j] = 0.0

    return psi, omega

# =============================================================================
# 4. PACK / UNPACK
# =============================================================================
def pack(psi, omega):
    x = np.empty(N_dof)
    for i in range(Nx + 1):
        for j in range(Ny + 1):
            if kind[i, j] in ACTIVE:
                k = node_id[i, j]
                x[k]         = psi[i, j]
                x[k + N_act] = omega[i, j]
    return x

def unpack(x, psi, omega):
    for i in range(Nx + 1):
        for j in range(Ny + 1):
            if kind[i, j] in ACTIVE:
                k = node_id[i, j]
                psi[i, j]   = x[k]
                omega[i, j] = x[k + N_act]
    return psi, omega

# =============================================================================
# 5. INFORMACIÓN DE THOM  (robusta para esquinas)
# =============================================================================
def get_thom_info(i, j):
    """
    Para nodo BC_WALL o WALL: lista de (psi_solid, i_inner, j_inner).
    - BC_WALL: busca vecinos SOLID
    - WALL   : busca hacia exterior del dominio (j=-1 o j=Ny+1 → pared)
    Esquinas con varias caras sólidas → lista con múltiples entradas.
    """
    contributions = []

    if kind[i, j] == BC_WALL:
        for (di, dj) in [(1,0),(-1,0),(0,1),(0,-1)]:
            ni, nj = i+di, j+dj
            if not (0 <= ni <= Nx and 0 <= nj <= Ny):
                continue
            if kind[ni, nj] == SOLID:
                psi_s = psi_fixed(ni, nj)
                ii, ij = i-di, j-dj
                if 0 <= ii <= Nx and 0 <= ij <= Ny and kind[ii, ij] != SOLID:
                    contributions.append((psi_s, ii, ij))

    elif kind[i, j] == WALL:
        if j == 0:
            # Pared inferior: ψ_pared = 0, nodo interior en j=1
            contributions.append((0.0, i, 1))
        elif j == Ny:
            # Pared superior: ψ_pared = psi_max, nodo interior en j=Ny-1
            contributions.append((psi_max, i, Ny-1))

    return contributions

# =============================================================================
# 6. RESIDUO  F(ψ, ω)
# =============================================================================
def compute_F(psi, omega, R):
    """
    Construye el vector residuo para todos los nodos activos.

    FREE    : F^ψ = ∇²ψ + ω = 0
              F^ω = ∇²ω - (R/4)*(∂ψ/∂y·∂ω/∂x - ∂ψ/∂x·∂ω/∂y) = 0

    BC_WALL : F^ψ = ∇²ψ + ω = 0   (Poisson, igual que FREE)
              F^ω = ω + (2/h²)·mean(ψ_inner - ψ_solid) = 0   (Thom)

    WALL    : F^ψ = ψ[i,j] - ψ_pared = 0   (Dirichlet)
              F^ω = ω + (2/h²)·(ψ_inner - ψ_pared) = 0   (Thom)

    OUTLET  : F^ψ = ψ[Nx,j] - ψ[Nx-1,j] = 0   (Neumann)
              F^ω = ω[Nx,j] - ω[Nx-1,j] = 0   (Neumann)
    """
    F = np.zeros(N_dof)

    for i in range(Nx + 1):
        for j in range(Ny + 1):
            if kind[i, j] not in ACTIVE:
                continue
            k  = node_id[i, j]
            rw = k + N_act

            # ── OUTLET ───────────────────────────────────────────────────────
            if kind[i, j] == OUTLET:
                F[k]  = psi[i, j]   - psi[i-1, j]
                F[rw] = omega[i, j] - omega[i-1, j]
                continue

            # ── WALL (pared del canal) ────────────────────────────────────────
            if kind[i, j] == WALL:
                psi_w = psi_fixed(i, j)          # 0.0 o psi_max
                F[k]  = psi[i, j] - psi_w        # Dirichlet ψ

                contribs = get_thom_info(i, j)
                if contribs:
                    psi_s, ii, ij = contribs[0]  # solo 1 contribución para WALL
                    F[rw] = omega[i, j] + (2.0/h**2) * (psi[ii, ij] - psi_s)
                else:
                    F[rw] = omega[i, j]
                continue

            # ── FREE y BC_WALL: Poisson para ψ ───────────────────────────────
            F[k] = (psi[i+1,j] + psi[i-1,j] + psi[i,j+1] + psi[i,j-1]
                    - 4.0*psi[i,j] + h**2 * omega[i,j])

            if kind[i, j] == BC_WALL:
                # Thom promediado (esquinas pueden tener >1 cara sólida)
                contribs = get_thom_info(i, j)
                if contribs:
                    thom_sum = sum(psi[ii,ij] - ps
                                   for (ps, ii, ij) in contribs)
                    F[rw] = omega[i,j] + (2.0/h**2) * thom_sum / len(contribs)
                else:
                    F[rw] = omega[i,j]

            else:  # FREE: transporte de vorticidad
                dpsi_y = psi[i, j+1] - psi[i, j-1]
                dpsi_x = psi[i+1, j] - psi[i-1, j]
                dom_x  = omega[i+1, j] - omega[i-1, j]
                dom_y  = omega[i, j+1] - omega[i, j-1]

                F[rw] = (omega[i+1,j] + omega[i-1,j]
                         + omega[i,j+1] + omega[i,j-1]
                         - 4.0*omega[i,j]
                         - (R/4.0)*(dpsi_y*dom_x - dpsi_x*dom_y))
    return F

# =============================================================================
# 7. JACOBIANA  dF/d(ψ,ω)  — 100% consistente con compute_F
# =============================================================================
def compute_J(psi, omega, R):
    """
    Jacobiana analítica sparse.
    Misma lógica que compute_F, derivada término a término.
    """
    J = lil_matrix((N_dof, N_dof))

    for i in range(Nx + 1):
        for j in range(Ny + 1):
            if kind[i, j] not in ACTIVE:
                continue
            k  = node_id[i, j]
            rw = k + N_act

            # ── OUTLET ───────────────────────────────────────────────────────
            # F^ψ = ψ[i,j] - ψ[i-1,j]  →  dF^ψ/dψ[i]=1, dF^ψ/dψ[i-1]=-1
            # F^ω = ω[i,j] - ω[i-1,j]  →  dF^ω/dω[i]=1, dF^ω/dω[i-1]=-1
            if kind[i, j] == OUTLET:
                J[k,  k]  =  1.0
                J[rw, rw] =  1.0
                if is_active(i-1, j):
                    J[k,  node_id[i-1,j]]          = -1.0
                    J[rw, node_id[i-1,j] + N_act]  = -1.0
                continue

            # ── WALL ─────────────────────────────────────────────────────────
            # F^ψ = ψ[i,j] - ψ_w  →  dF^ψ/dψ[i,j] = 1
            # F^ω = ω + (2/h²)*(ψ_inner - ψ_w)
            #        →  dF^ω/dω[i,j]=1,  dF^ω/dψ_inner=(2/h²)
            if kind[i, j] == WALL:
                J[k,  k]  = 1.0
                J[rw, rw] = 1.0
                contribs = get_thom_info(i, j)
                if contribs:
                    psi_s, ii, ij = contribs[0]
                    if is_active(ii, ij):
                        J[rw, node_id[ii,ij]] = 2.0/h**2
                continue

            # ── FREE y BC_WALL: dF^ψ/dψ — Laplaciano ────────────────────────
            J[k, k] = -4.0
            for (ni, nj) in [(i+1,j),(i-1,j),(i,j+1),(i,j-1)]:
                if is_active(ni, nj):
                    J[k, node_id[ni,nj]] += 1.0
            # dF^ψ/dω[i,j] = h²
            J[k, rw] = h**2

            # ── BC_WALL: dF^ω — Thom ─────────────────────────────────────────
            if kind[i, j] == BC_WALL:
                J[rw, rw] = 1.0
                contribs = get_thom_info(i, j)
                if contribs:
                    coeff = (2.0/h**2) / len(contribs)
                    for (ps, ii, ij) in contribs:
                        if is_active(ii, ij):
                            J[rw, node_id[ii,ij]] += coeff
                continue

            # ── FREE: dF^ω — transporte vorticidad ───────────────────────────
            dpsi_y = psi[i, j+1] - psi[i, j-1]
            dpsi_x = psi[i+1, j] - psi[i-1, j]
            dom_x  = omega[i+1, j] - omega[i-1, j]
            dom_y  = omega[i, j+1] - omega[i, j-1]

            # dF^ω/dω[i,j] = -4
            J[rw, rw] = -4.0

            # dF^ω/dω[vecinos] = 1 ± (R/4)*dψ
            neigh_om = {
                (i+1,j): 1.0 - (R/4.0)*dpsi_y,
                (i-1,j): 1.0 + (R/4.0)*dpsi_y,
                (i,j+1): 1.0 + (R/4.0)*dpsi_x,
                (i,j-1): 1.0 - (R/4.0)*dpsi_x,
            }
            for (ni,nj), val in neigh_om.items():
                if is_active(ni, nj):
                    J[rw, node_id[ni,nj] + N_act] += val

            # dF^ω/dψ[vecinos] — términos cruzados
            neigh_psi = {
                (i,j+1): -(R/4.0)*dom_x,
                (i,j-1): +(R/4.0)*dom_x,
                (i+1,j): +(R/4.0)*dom_y,
                (i-1,j): -(R/4.0)*dom_y,
            }
            for (ni,nj), val in neigh_psi.items():
                if is_active(ni, nj):
                    J[rw, node_id[ni,nj]] += val

    return csr_matrix(J)

# =============================================================================
# 8. NEWTON-RAPHSON CON BACKTRACKING LINE SEARCH
# =============================================================================
def newton_raphson(psi, omega, R,
                   tol=1e-7, max_iter=50,
                   alpha_min=1e-12, max_backtracks=40):
    """
    Resuelve F(ψ,ω)=0 para el Reynolds R dado.
    Backtracking con copia limpia en cada intento (bug crítico corregido).
    """
    print(f"\n  Re = {R:.4f}")
    psi, omega = apply_bc_inlet_solid(psi, omega)

    norm_F_prev = np.inf

    for nit in range(1, max_iter + 1):

        F      = compute_F(psi, omega, R)
        norm_F = np.linalg.norm(F)
        print(f"    iter {nit:3d}   |F| = {norm_F:.4e}")

        if norm_F < tol:
            print(f"  ✓ Convergido en {nit-1} iters  (|F| = {norm_F:.2e})")
            return psi, omega, True

        # Detección de divergencia
        if norm_F > 10.0 * norm_F_prev and nit > 2:
            print(f"  ✗ Divergencia detectada (factor > 10)")
            return psi, omega, False
        norm_F_prev = norm_F

        # Resolver sistema lineal
        J = compute_J(psi, omega, R)
        try:
            dx = spsolve(J, -F)
        except Exception as e:
            print(f"  ✗ Fallo en spsolve: {e}")
            return psi, omega, False

        if not np.all(np.isfinite(dx)):
            print(f"  ✗ dx contiene NaN/Inf — Jacobiana singular")
            return psi, omega, False

        # ── Backtracking line search ─────────────────────────────────────────
        # CORRECCIÓN CRÍTICA: copias limpias en CADA intento de alpha
        x_cur      = pack(psi, omega)
        best_norm  = norm_F
        best_alpha = 0.0
        found      = False
        alpha      = 1.0

        for _ in range(max_backtracks):
            # Copia limpia para este alpha (no acumula modificaciones previas)
            p_trial = psi.copy()
            o_trial = omega.copy()
            p_trial, o_trial = unpack(x_cur + alpha * dx, p_trial, o_trial)
            p_trial, o_trial = apply_bc_inlet_solid(p_trial, o_trial)

            norm_try = np.linalg.norm(compute_F(p_trial, o_trial, R))

            if norm_try < best_norm:
                best_norm  = norm_try
                best_alpha = alpha
                found = True
                if norm_try < norm_F:    # mejora suficiente → aceptar
                    break

            alpha *= 0.5
            if alpha < alpha_min:
                break

        if not found:
            print(f"  ✗ Line search agotado (ningún alpha redujo el residuo)")
            return psi, omega, False

        # Aplicar el mejor paso
        psi_new   = psi.copy()
        omega_new = omega.copy()
        psi_new, omega_new = unpack(x_cur + best_alpha * dx, psi_new, omega_new)
        psi, omega = apply_bc_inlet_solid(psi_new, omega_new)

    print(f"  ✗ Máximo de iteraciones ({max_iter}) para Re={R:.4f}")
    return psi, omega, False

# =============================================================================
# 9. CONTINUACIÓN EN REYNOLDS
# =============================================================================
def re_continuation(Re_target, Re_steps=None):
    """
    Arranca desde Re≈0 (flujo lineal) y sube gradualmente hasta Re_target.
    Si un paso falla, intenta subdividirlo automáticamente.
    """
    if Re_steps is None:
        if Re_target <= 1:
            pts = np.linspace(0.01, Re_target, 6)
        elif Re_target <= 5:
            pts = np.concatenate([
                np.linspace(0.01, 1,  7),
                np.linspace(1,    Re_target, 6)[1:]])
        elif Re_target <= 20:
            pts = np.concatenate([
                np.linspace(0.01, 1,  7),
                np.linspace(1,    5,  7)[1:],
                np.linspace(5,    Re_target, 9)[1:]])
        elif Re_target <= 50:
            pts = np.concatenate([
                np.linspace(0.01, 1,  7),
                np.linspace(1,    5,  7)[1:],
                np.linspace(5,   20, 10)[1:],
                np.linspace(20,  Re_target, 9)[1:]])
        else:
            pts = np.concatenate([
                np.linspace(0.01,  1,  7),
                np.linspace(1,     5,  7)[1:],
                np.linspace(5,    20, 10)[1:],
                np.linspace(20,   50, 10)[1:],
                np.linspace(50,   Re_target, 12)[1:]])
        Re_steps = [round(float(r), 6) for r in pts]

    # ── Inicialización: flujo de Couette lineal sin vorticidad ───────────────
    psi   = np.zeros((Nx+1, Ny+1))
    omega = np.zeros((Nx+1, Ny+1))
    for j in range(Ny+1):
        psi[:, j] = V0 * j * h
    psi, omega = apply_bc_inlet_solid(psi, omega)

    print("=" * 65)
    print(f"  Stream-function / Vorticity 2D Solver")
    print(f"  Newton-Raphson + Line Search + Re-continuation")
    print(f"  Re objetivo: {Re_target}   |   Pasos: {len(Re_steps)}")
    print(f"  Secuencia: {[f'{r:.3g}' for r in Re_steps]}")
    print("=" * 65)

    for i_re, Re in enumerate(Re_steps):
        psi, omega, conv = newton_raphson(psi, omega, Re)

        if not conv:
            print(f"\n⚠  Fallo en Re={Re:.4f}. Intentando subdivisión...")
            Re_prev = Re_steps[i_re - 1] if i_re > 0 else 0.0
            sub     = np.linspace(Re_prev, Re, 6)[1:]   # 5 pasos intermedios
            sub_ok  = True
            for Re_sub in sub:
                psi, omega, cv = newton_raphson(psi, omega, Re_sub)
                if not cv:
                    print(f"  ✗ Falló también en Re_sub={Re_sub:.4f}. Deteniendo.")
                    sub_ok = False
                    break
            if not sub_ok:
                print(f"\n  Solución parcial hasta Re≈{Re_prev:.4f}")
                break

    return psi, omega