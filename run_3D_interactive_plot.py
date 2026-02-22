"""
Run 3D elastic wave simulation, save state at intermediate time steps,
and open an interactive 3D plot: velocity magnitude, stress vector,
transparent grey fracture planes, and a time-step slider.
Initial condition: Gaussian stress with peak at domain center.
"""
import numpy as np
from _utility import (
    create_compliance_matrix_from_velocities,
    create_compliance_matrix_isotropic,
    FD_solver_3D_elastic,
    plot_elastic_3D_interactive,
)
from scipy.sparse.linalg import expm_multiply

# Same parameters as main.py
c_p_base = 6000
c_s_base = 3500
rho_base = 2700
S_base = create_compliance_matrix_from_velocities(c_p_base, c_s_base, rho_base)
Nx, Ny, Nz = 8, 8, 8
dx = dy = dz = 0.05
ADD_FRACTURES = True
rho_fracture = 1000.0
K_water = 2.2e9
mu_fracture = 1e-6
lambda_fracture = K_water
fracture1_z = Nz // 2
fracture2_x = Nx // 2

# Grid sizes (same as main.py)
N_main = Nx * Ny * Nz
N_vx = (Nx - 1) * Ny * Nz
N_vy = Nx * (Ny - 1) * Nz
N_vz = Nx * Ny * (Nz - 1)
N_sxy = (Nx - 1) * (Ny - 1) * Nz
N_sxz = (Nx - 1) * Ny * (Nz - 1)
N_syz = Nx * (Ny - 1) * (Nz - 1)
N_vel = N_vx + N_vy + N_vz
N_stress = 3 * N_main + N_sxy + N_sxz + N_syz
N_total = N_vel + N_stress

# Density and compliance (same as main)
rho_model = np.full((Nz, Ny, Nx), rho_base)
fracture_mask = np.zeros((Nz, Ny, Nx), dtype=bool)
if ADD_FRACTURES:
    for i in range(Nz):
        for j in range(Ny):
            for k in range(Nx):
                if abs(i - fracture1_z) <= 0 or abs(k - fracture2_x) <= 0:
                    fracture_mask[i, j, k] = True
                    rho_model[i, j, k] = rho_fracture
rho_model = np.clip(rho_model, 1.0, 3000)

need_varying_S = ADD_FRACTURES
if need_varying_S:
    S_model = np.zeros((6, 6, Nz, Ny, Nx))
    for i in range(Nz):
        for j in range(Ny):
            for k in range(Nx):
                if fracture_mask[i, j, k]:
                    S_model[:, :, i, j, k] = create_compliance_matrix_isotropic(
                        lambda_fracture, mu_fracture
                    )
                else:
                    S_model[:, :, i, j, k] = S_base
    S = S_model
else:
    S = S_base

bcs = {'L': 'DBC', 'R': 'DBC', 'T': 'DBC', 'B': 'DBC', 'F': 'DBC', 'Ba': 'DBC'}
H, A, B, B_sqrt, B_inv, B_inv_sqrt = FD_solver_3D_elastic(
    Nx, Ny, Nz, dx, dy, dz, rho_model, S, bcs
)

# Initial condition: Gaussian with peak at center (0, 0, 0)
x = np.arange(Nx) * dx - (Nx - 1) * dx / 2
y = np.arange(Ny) * dy - (Ny - 1) * dy / 2
z = np.arange(Nz) * dz - (Nz - 1) * dz / 2
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
sigma_g = 2.5 * dx
amplitude = 0.005
gaussian = amplitude * np.exp(-(X**2 + Y**2 + Z**2) / (2 * sigma_g**2))
g_flat = np.ascontiguousarray(gaussian.transpose(2, 1, 0)).flatten()

phi_0 = np.zeros(N_total, dtype=complex)
idx = N_vel
phi_0[idx:idx + N_main] = g_flat
idx += N_main
phi_0[idx:idx + N_main] = g_flat
idx += N_main
phi_0[idx:idx + N_main] = g_flat

# Time stepping
if need_varying_S and ADD_FRACTURES:
    c_p_fracture = np.sqrt((lambda_fracture + 2 * mu_fracture) / rho_fracture)
    c_p_min = min(c_p_fracture, c_p_base)
else:
    c_p_min = c_p_base
dt_max = dx / c_p_min
safety_factor = 0.01 if (need_varying_S and ADD_FRACTURES) else 0.1
dt = safety_factor * dt_max
n_steps = 20
n_substeps_per_step = 100
dt_sub = dt / n_substeps_per_step

# Save a snapshot at every time step so the slider can show step 0, 1, 2, ..., n_steps
save_steps = list(range(0, n_steps + 1))
snapshots = []
times = []

psi_0 = B_sqrt @ phi_0
initial_norm = np.linalg.norm(psi_0)
psi_current = psi_0.copy() / initial_norm
norm_factor = initial_norm

# Initial snapshot (t=0)
phi_t = B_inv_sqrt @ (norm_factor * psi_current)
snapshots.append(np.real(phi_t).copy())
times.append(0.0)

for step in range(1, n_steps + 1):
    for _ in range(n_substeps_per_step):
        psi_current = expm_multiply(-1j * H * dt_sub, psi_current)
    ratio = np.linalg.norm(psi_current)
    if ratio > 1e4 or ratio < 1e-4:
        print("Energy explosion - stopping")
        break
    phi_t = B_inv_sqrt @ (norm_factor * psi_current)
    snapshots.append(np.real(phi_t).copy())
    times.append(step * dt)

print(f"Saved {len(snapshots)} snapshots at t = {times}")

plot_elastic_3D_interactive(
    snapshots, times, Nx, Ny, Nz, dx, dy, dz,
    fracture1_z, fracture2_x,
    subsample=2,
    save_file=None,
)