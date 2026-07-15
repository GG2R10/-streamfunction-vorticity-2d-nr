import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve

# Parámetros
Nx, Ny = 200, 20
h, V0 = 1.0, 1.0

# Bloque 1 — esquina superior izquierda
B1_ic, B1_fc = 0,  8
B1_jb, B1_jt = 13, 20

# Bloque 2 — zona inferior central
B2_ic, B2_fc = 90, 98
B2_jb, B2_jt = 0,  7

psi_max = V0 * Ny * h

# Tipos de nodo
SOLID, INLET, OUTLET, WALL, BC_WALL, FREE = range(6)
kind = np.full((Nx + 1, Ny + 1), FREE, dtype=int)

kind[B1_ic:B1_fc + 1, B1_jb:B1_jt + 1] = SOLID
kind[B2_ic:B2_fc + 1, B2_jb:B2_jt + 1] = SOLID

for i in range(Nx + 1):
    for j in range(Ny + 1):
        if kind[i, j] == SOLID:
            continue
        if i == 0:
            kind[i, j] = INLET
        elif i == Nx:
            kind[i, j] = OUTLET
        elif j == 0 or j == Ny:
            kind[i, j] = WALL

for i in range(1, Nx):
    for j in range(1, Ny):
        if kind[i, j] != FREE:
            continue
        for ni, nj in [(i+1,j),(i-1,j),(i,j+1),(i,j-1)]:
            if kind[ni, nj] == SOLID:
                kind[i, j] = BC_WALL
                break

counts = {name: int(np.sum(kind == val))
          for name, val in [("FREE",FREE),("BC_WALL",BC_WALL),
                             ("WALL",WALL),("OUTLET",OUTLET),
                             ("INLET",INLET),("SOLID",SOLID)]}
print("Clasificación de nodos:")
for name, cnt in counts.items():
    print(f"  {name:8s}: {cnt:5d}")
print(f"  {'TOTAL':8s}: {(Nx+1)*(Ny+1):5d}")

# Numeración de activos: FREE, BC_WALL, WALL, OUTLET
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
print(f"\nNodos activos: {N_act}   Grados de libertad: {N_dof}")

def is_active(i, j):
    return 0 <= i <= Nx and 0 <= j <= Ny and kind[i, j] in ACTIVE

def psi_fixed(i, j):
    if kind[i, j] == INLET:
        return V0 * j * h
    if kind[i, j] == SOLID:
        if B1_ic <= i <= B1_fc and B1_jb <= j <= B1_jt:
            return psi_max
        if B2_ic <= i <= B2_fc and B2_jb <= j <= B2_jt:
            return 0.0
    if kind[i, j] == WALL:
        return 0.0 if j == 0 else psi_max
    return 0.0

def apply_bc_inlet_solid(psi, omega):
    psi[B1_ic:B1_fc+1, B1_jb:B1_jt+1] = psi_max
    psi[B2_ic:B2_fc+1, B2_jb:B2_jt+1] = 0.0
    for j in range(Ny + 1):
        if kind[0, j] == INLET:
            psi[0, j]   = V0 * j * h
            omega[0, j] = 0.0
    return psi, omega

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

def get_thom_info(i, j):
    """Para nodos WALL/BC_WALL: (psi_solid, i_inner, j_inner)."""
    contribs = []
    if kind[i, j] == BC_WALL:
        for di, dj in [(1,0),(-1,0),(0,1),(0,-1)]:
            ni, nj = i+di, j+dj
            if not (0 <= ni <= Nx and 0 <= nj <= Ny):
                continue
            if kind[ni, nj] == SOLID:
                psi_s = psi_fixed(ni, nj)
                ii, ij = i-di, j-dj
                if 0 <= ii <= Nx and 0 <= ij <= Ny and kind[ii, ij] != SOLID:
                    contribs.append((psi_s, ii, ij))
    elif kind[i, j] == WALL:
        if j == 0:
            contribs.append((0.0, i, 1))
        elif j == Ny:
            contribs.append((psi_max, i, Ny-1))
    return contribs

def compute_F(psi, omega, R):
    F = np.zeros(N_dof)
    for i in range(Nx + 1):
        for j in range(Ny + 1):
            if kind[i, j] not in ACTIVE:
                continue
            k  = node_id[i, j]
            rw = k + N_act

            if kind[i, j] == OUTLET:
                F[k]  = psi[i, j]   - psi[i-1, j]
                F[rw] = omega[i, j] - omega[i-1, j]
                continue

            if kind[i, j] == WALL:
                F[k]  = psi[i, j] - psi_fixed(i, j)
                contribs = get_thom_info(i, j)
                if contribs:
                    _, ii, ij = contribs[0]
                    F[rw] = omega[i, j] + (2.0/h**2) * (psi[ii, ij] - psi_fixed(i, j))
                else:
                    F[rw] = omega[i, j]
                continue

            # FREE y BC_WALL: Poisson para ψ
            F[k] = (psi[i+1,j] + psi[i-1,j] + psi[i,j+1] + psi[i,j-1]
                    - 4.0*psi[i,j] + h**2 * omega[i,j])

            if kind[i, j] == BC_WALL:
                contribs = get_thom_info(i, j)
                if contribs:
                    thom_sum = sum(psi[ii,ij] - ps for (ps, ii, ij) in contribs)
                    F[rw] = omega[i,j] + (2.0/h**2) * thom_sum / len(contribs)
                else:
                    F[rw] = omega[i,j]
            else:  # FREE: transporte de vorticidad (convección con UPWIND)
                # Velocidades: u_x = ∂ψ/∂y,  u_y = -∂ψ/∂x
                u_x =  (psi[i, j+1] - psi[i, j-1]) / (2.0*h)
                u_y = -(psi[i+1, j] - psi[i-1, j]) / (2.0*h)
                # Derivadas de ω "río arriba" según el signo de la velocidad:
                # así el coeficiente del vecino nunca cambia de signo (M-matriz)
                # y no aparecen las oscilaciones de las diferencias centradas.
                dwdx = ((omega[i, j] - omega[i-1, j]) if u_x >= 0.0
                        else (omega[i+1, j] - omega[i, j])) / h
                dwdy = ((omega[i, j] - omega[i, j-1]) if u_y >= 0.0
                        else (omega[i, j+1] - omega[i, j])) / h
                conv = R * h**2 * (u_x*dwdx + u_y*dwdy)
                F[rw] = (omega[i+1,j] + omega[i-1,j] + omega[i,j+1] + omega[i,j-1]
                         - 4.0*omega[i,j] - conv)
    return F

def compute_J(psi, omega, R):
    J = lil_matrix((N_dof, N_dof))
    for i in range(Nx + 1):
        for j in range(Ny + 1):
            if kind[i, j] not in ACTIVE:
                continue
            k  = node_id[i, j]
            rw = k + N_act

            if kind[i, j] == OUTLET:
                J[k,  k]  =  1.0
                J[rw, rw] =  1.0
                if is_active(i-1, j):
                    J[k,  node_id[i-1,j]]          = -1.0
                    J[rw, node_id[i-1,j] + N_act]  = -1.0
                continue

            if kind[i, j] == WALL:
                J[k,  k]  = 1.0
                J[rw, rw] = 1.0
                contribs = get_thom_info(i, j)
                if contribs:
                    _, ii, ij = contribs[0]
                    if is_active(ii, ij):
                        J[rw, node_id[ii,ij]] = 2.0/h**2
                continue

            # FREE y BC_WALL: dF^ψ/dψ — Laplaciano
            J[k, k] = -4.0
            for ni, nj in [(i+1,j),(i-1,j),(i,j+1),(i,j-1)]:
                if is_active(ni, nj):
                    J[k, node_id[ni,nj]] += 1.0
            J[k, rw] = h**2

            if kind[i, j] == BC_WALL:
                J[rw, rw] = 1.0
                contribs = get_thom_info(i, j)
                if contribs:
                    coeff = (2.0/h**2) / len(contribs)
                    for _, ii, ij in contribs:
                        if is_active(ii, ij):
                            J[rw, node_id[ii,ij]] += coeff
                continue

            # FREE: dF^ω — transporte de vorticidad (Jacobiana del esquema UPWIND)
            u_x =  (psi[i, j+1] - psi[i, j-1]) / (2.0*h)
            u_y = -(psi[i+1, j] - psi[i-1, j]) / (2.0*h)
            # Selección de nodos "río arriba" (los mismos que usa compute_F).
            # (xp, xm): pareja (+, -) que forma dwdx = (ω[xp] - ω[xm]) / h
            if u_x >= 0.0:
                dwdx = (omega[i, j] - omega[i-1, j]) / h
                xp, xm = (i, j), (i-1, j)
            else:
                dwdx = (omega[i+1, j] - omega[i, j]) / h
                xp, xm = (i+1, j), (i, j)
            if u_y >= 0.0:
                dwdy = (omega[i, j] - omega[i, j-1]) / h
                yp, ym = (i, j), (i, j-1)
            else:
                dwdy = (omega[i, j+1] - omega[i, j]) / h
                yp, ym = (i, j+1), (i, j)

            cx = R * h**2 * u_x
            cy = R * h**2 * u_y

            # dF^ω/dω = Laplaciano  −  ∂conv/∂ω
            J[rw, rw] += -4.0
            for ni, nj in [(i+1,j),(i-1,j),(i,j+1),(i,j-1)]:
                if is_active(ni, nj):
                    J[rw, node_id[ni,nj] + N_act] += 1.0
            # ∂conv/∂ω:  conv = cx·dwdx + cy·dwdy,  dwdx=(ω[xp]-ω[xm])/h
            for (ni, nj), sgn in [(xp, +1.0), (xm, -1.0)]:
                if is_active(ni, nj):
                    J[rw, node_id[ni,nj] + N_act] += -(cx/h)*sgn
            for (ni, nj), sgn in [(yp, +1.0), (ym, -1.0)]:
                if is_active(ni, nj):
                    J[rw, node_id[ni,nj] + N_act] += -(cy/h)*sgn

            # dF^ω/dψ = − ∂conv/∂ψ  (conv depende de ψ vía u_x, u_y)
            for (ni,nj), dconv in {
                (i, j+1): +R*h*dwdx/2.0,
                (i, j-1): -R*h*dwdx/2.0,
                (i+1, j): -R*h*dwdy/2.0,
                (i-1, j): +R*h*dwdy/2.0,
            }.items():
                if is_active(ni, nj):
                    J[rw, node_id[ni,nj]] += -dconv

    return csr_matrix(J)

def newton_raphson(psi, omega, R, tol=1e-7, max_iter=50,
                   alpha_min=1e-12, max_backtracks=40):
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

        if norm_F > 10.0 * norm_F_prev and nit > 2:
            print(f"  ✗ Divergencia detectada (factor > 10)")
            return psi, omega, False
        norm_F_prev = norm_F

        J = compute_J(psi, omega, R)
        try:
            dx = spsolve(J, -F)
        except Exception as e:
            print(f"  ✗ Fallo en spsolve: {e}")
            return psi, omega, False

        if not np.all(np.isfinite(dx)):
            print(f"  ✗ dx contiene NaN/Inf")
            return psi, omega, False

        # Backtracking line search
        x_cur      = pack(psi, omega)
        best_norm  = norm_F
        best_alpha = 0.0
        found      = False
        alpha      = 1.0

        for _ in range(max_backtracks):
            p_trial = psi.copy()
            o_trial = omega.copy()
            p_trial, o_trial = unpack(x_cur + alpha * dx, p_trial, o_trial)
            p_trial, o_trial = apply_bc_inlet_solid(p_trial, o_trial)

            norm_try = np.linalg.norm(compute_F(p_trial, o_trial, R))
            if norm_try < best_norm:
                best_norm  = norm_try
                best_alpha = alpha
                found = True
                if norm_try < norm_F:
                    break
            alpha *= 0.5
            if alpha < alpha_min:
                break

        if not found:
            print(f"  ✗ Line search agotado")
            return psi, omega, False

        psi_new   = psi.copy()
        omega_new = omega.copy()
        psi_new, omega_new = unpack(x_cur + best_alpha * dx, psi_new, omega_new)
        psi, omega = apply_bc_inlet_solid(psi_new, omega_new)

    print(f"  ✗ Máximo de iteraciones ({max_iter}) para Re={R:.4f}")
    return psi, omega, False

def re_continuation(Re_target, Re_steps=None):
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
            print(f"\n⚠  Fallo en Re={Re:.4f}. Subdividiendo...")
            Re_prev = Re_steps[i_re - 1] if i_re > 0 else 0.0
            sub     = np.linspace(Re_prev, Re, 6)[1:]
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
