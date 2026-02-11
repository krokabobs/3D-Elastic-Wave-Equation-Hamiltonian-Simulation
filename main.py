import numpy as np
from _utility import *

c_p_base = 6000  # P-wave velocity (m/s)
c_s_base = 3500  # S-wave velocity (m/s)
rho_base = 2700  # Base density (kg/m³)
S_base = create_compliance_matrix_from_velocities(c_p_base, c_s_base, rho_base)

Nx, Ny, Nz = 8, 8, 8
dx = dy = dz = 0.05

ADD_FRACTURES = True

# Fracture properties (water-filled fractures)
rho_fracture = 1000.0  # Water density (kg/m³)
K_water = 2.2e9  # Water bulk modulus (Pa)
mu_fracture = 1e-6  # Very small shear modulus for numerical stability
lambda_fracture = K_water  # Bulk modulus for λ

# Compute base material Lamé parameters
mu_base = rho_base * c_s_base ** 2
lambda_base = rho_base * c_p_base ** 2 - 2 * mu_base

print(f"\nFracture properties (water-filled):")
print(f"  λ_fracture = {lambda_fracture:.2e} Pa")
print(f"  μ_fracture = {mu_fracture:.2e} Pa")
print(f"  ρ_fracture = {rho_fracture:.1f} kg/m³")
print(f"  Expected c_p = {np.sqrt((lambda_fracture + 2 * mu_fracture) / rho_fracture):.1f} m/s")

# ============================================================================
# CREATE DENSITY MODEL WITH FRACTURES
# ============================================================================
rho_model = np.full((Nz, Ny, Nx), rho_base)
fracture_mask = np.zeros((Nz, Ny, Nx), dtype=bool)

if ADD_FRACTURES:
    # Two perpendicular fracture planes: horizontal (z) and vertical (x)
    fracture1_z = Nz // 2  # Horizontal plane (constant z)
    fracture2_x = Nx // 2  # Vertical plane (constant x)
    fracture_thickness = 0

    print(f"\nAdding 2 perpendicular fracture planes:")
    print(f"  Fracture 1: Horizontal plane at z-index {fracture1_z}")
    print(f"  Fracture 2: Vertical plane at x-index {fracture2_x}")

    fracture_cells = 0
    for i in range(Nz):
        for j in range(Ny):
            for k in range(Nx):
                is_fracture = (abs(i - fracture1_z) <= fracture_thickness or
                               abs(k - fracture2_x) <= fracture_thickness)
                if is_fracture:
                    fracture_mask[i, j, k] = True
                    fracture_cells += 1
                    rho_model[i, j, k] = rho_fracture

    print(
        f"  Total fracture cells: {fracture_cells} out of {Nx * Ny * Nz} ({100 * fracture_cells / (Nx * Ny * Nz):.1f}%)")

# Clip density to reasonable bounds
rho_model = np.clip(rho_model, 1.0, 3000)

print(f"\nDensity Model Statistics:")
print(f"  Min: {np.min(rho_model):.1f} kg/m³, Max: {np.max(rho_model):.1f} kg/m³")
print(f"  Mean: {np.mean(rho_model):.1f} kg/m³, Std: {np.std(rho_model):.1f} kg/m³")

# ============================================================================
# CREATE SPATIALLY VARYING COMPLIANCE MATRIX
# ============================================================================
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

# ============================================================================
# COMPUTE HAMILTONIAN
# ============================================================================

# Grid sizes for staggered grids
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

H, A, B, B_sqrt, B_inv, B_inv_sqrt = FD_solver_3D_elastic(
    Nx, Ny, Nz, dx, dy, dz, rho_model, S
)

# Check A is anti-Hermitian (A† = -A); then H = i B^{-1/2} A B^{-1/2} is Hermitian
A_sum = A + A.T
if hasattr(A_sum, 'data') and len(A_sum.data) > 0:
    A_anti_hermitian_error = np.max(np.abs(A_sum.data))
else:
    A_sum_dense = A_sum.toarray() if hasattr(A_sum, 'toarray') else A_sum
    A_anti_hermitian_error = np.max(np.abs(A_sum_dense)) if A_sum_dense.size > 0 else 0.0

if A_anti_hermitian_error < 1e-10:
    print(f"Matrix A is anti-Hermitian (A† = -A)")
else:
    print(f"WARNING: A may not be anti-Hermitian (max |A + A^T|: {A_anti_hermitian_error:.2e})")

# Verify H is Hermitian (H† = H) for unitary evolution and energy conservation
H_diff = H - H.conj().T
if hasattr(H_diff, 'data') and len(H_diff.data) > 0:
    H_hermitian_error = np.max(np.abs(H_diff.data))
else:
    H_diff_dense = H_diff.toarray() if hasattr(H_diff, 'toarray') else H_diff
    H_hermitian_error = np.max(np.abs(H_diff_dense)) if H_diff_dense.size > 0 else 0.0

if H_hermitian_error < 1e-10:
    print(f"Hamiltonian H is Hermitian (H† = H) - energy will be conserved")
else:
    print(f"WARNING: H may not be Hermitian (max |H - H†|: {H_hermitian_error:.2e})")

# ============================================================================
# TIME EVOLUTION SETUP
# ============================================================================
print(f"\n" + "=" * 60)
print("WAVE EVOLUTION")
print("=" * 60)

phi_0 = np.zeros(N_total, dtype=complex)

x = np.arange(Nx) * dx - (Nx - 1) * dx / 2
y = np.arange(Ny) * dy - (Ny - 1) * dy / 2
z = np.arange(Nz) * dz - (Nz - 1) * dz / 2
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
sigma = 2.5 * dx
amplitude = 0.005
gaussian = amplitude * np.exp(-(X ** 2 + Y ** 2 + Z ** 2) / (2 * sigma ** 2))

# Add compressional stress (equal diagonal components).
# State vector uses (z, y, x) order (x fastest); meshgrid gives (Nx, Ny, Nz), so transpose then flatten.
g_flat = np.ascontiguousarray(gaussian.transpose(2, 1, 0)).flatten()  # (Nz, Ny, Nx) order
idx = N_vel
phi_0[idx:idx + N_main] = g_flat  # σ_xx
idx += N_main
phi_0[idx:idx + N_main] = g_flat  # σ_yy
idx += N_main
phi_0[idx:idx + N_main] = g_flat  # σ_zz

# CFL condition for time step
if need_varying_S and ADD_FRACTURES:
    c_p_fracture = np.sqrt((lambda_fracture + 2 * mu_fracture) / rho_fracture)
    c_p_min = min(c_p_fracture, c_p_base)
else:
    c_p_min = c_p_base

dt_max = dx / c_p_min
safety_factor = 0.01 if need_varying_S and ADD_FRACTURES else 0.1
dt = safety_factor * dt_max
n_steps = 20
t_final = n_steps * dt

# Initial stress scale (for sanity check)
sigma_0_max = np.max(np.abs(phi_0[N_vel:]))
print(f"Initial stress max |σ|: {sigma_0_max:.4e}")
print(f"Time step: {dt:.2e} s ({n_steps} steps, final time: {t_final:.2e} s)")

# ============================================================================
# TIME EVOLUTION (Energy Basis)
# ============================================================================
from scipy.sparse.linalg import expm_multiply

# Transform to energy basis (required for Hamiltonian evolution)
psi_0 = B_sqrt @ phi_0
initial_norm = np.linalg.norm(psi_0)
psi_current = psi_0.copy() / initial_norm
norm_factor = initial_norm

n_substeps_per_step = 100  # many small steps for accurate exp(-iH dt)
dt_sub = dt / n_substeps_per_step
print(f"Using {n_substeps_per_step} sub-steps per main step (dt_sub = {dt_sub:.2e} s)")

# Evolve in energy basis: psi(t+dt) = exp(-i H dt) psi(t).
# We use many sub-steps (dt_sub = dt / n_substeps_per_step) because H = i B_inv_sqrt A B_inv_sqrt
# can have a very large norm when density/compliance vary (e.g. fractures). The matrix exponential
# exp(-i H tau) is accurate only when tau*||H|| is modest; a single step with dt would be unstable
# or inaccurate. Sub-stepping keeps each exp(-i H dt_sub) well-conditioned.
# Refs: Al-Mohy & Higham (2011) SIAM J. Sci. Comput. "Computing the Action of the Matrix
# Exponential"; Higham (2005) SIAM J. Matrix Anal. Appl. "The Scaling and Squaring Method for the Matrix
# Exponential Revisited".
for k in range(n_steps * n_substeps_per_step):
    psi_current = expm_multiply(-1j * H * dt_sub, psi_current)
    if (k + 1) % n_substeps_per_step != 0:
        continue
    step_num = (k + 1) // n_substeps_per_step
    ratio = np.linalg.norm(psi_current)  # should stay ~1 (Hermitian H => unitary evolution)
    if step_num % 5 == 0 or step_num == 1 or step_num == n_steps:
        print(f"  Step {step_num}/{n_steps}: energy ratio = {ratio:.6f}")
    if ratio > 1e4 or ratio < 1e-4:
        print("  Energy explosion detected - stopping simulation")
        break

# Transform back to physical basis
phi_evolved = B_inv_sqrt @ (norm_factor * psi_current)

# ============================================================================
# RESULTS AND VISUALIZATION
# ============================================================================
print(f"\nEvolution complete!")

# Energy conservation check
initial_energy = initial_norm
final_energy = np.linalg.norm(norm_factor * psi_current)
energy_ratio_final = final_energy / initial_energy if initial_energy > 0 else np.nan

print(f"Energy conservation: {energy_ratio_final:.6f}")
if abs(energy_ratio_final - 1.0) > 0.1:
    print(f"  Energy not conserved! (should be ≈ 1.0)")

# Extract fields for visualization
phi_real = np.real(phi_evolved)

idx = 0
v_x = phi_real[idx:idx + N_vx].reshape(Nz, Ny, Nx - 1)
idx += N_vx
v_y = phi_real[idx:idx + N_vy].reshape(Nz, Ny - 1, Nx)
idx += N_vy
v_z = phi_real[idx:idx + N_vz].reshape(Nz - 1, Ny, Nx)
idx += N_vz

σ_xx = phi_real[idx:idx + N_main].reshape(Nz, Ny, Nx)
idx += N_main
σ_yy = phi_real[idx:idx + N_main].reshape(Nz, Ny, Nx)
idx += N_main
σ_zz = phi_real[idx:idx + N_main].reshape(Nz, Ny, Nx)

# Calculate auto-scaling for visualization (because velocity may be much smaller than stress)
v_max = max(np.max(np.abs(v_x)), np.max(np.abs(v_y)), np.max(np.abs(v_z)))
sigma_max = max(np.max(np.abs(σ_xx)), np.max(np.abs(σ_yy)), np.max(np.abs(σ_zz)))

print(f"\nField statistics:")
print(f"  Velocity: max = {v_max:.4e}")
print(f"  Stress: max = {sigma_max:.4e}")

if v_max > 0 and sigma_max > 0:
    target_ratio = 0.25
    scale_stress_auto = 0.5
    scale_v_auto = max(1.0, (sigma_max / v_max) * target_ratio * scale_stress_auto)
    print(f"  Visualization scales: velocity={scale_v_auto:.1f}, stress={scale_stress_auto:.1f}")
else:
    scale_v_auto = 20.0
    scale_stress_auto = 5.0

# Visualize
output_file = "elastic_3D_simulation.png"
plot_elastic_3D(phi_real, Nx, Ny, Nz, dx, dy, dz,
                title=f'3D Elastic Wave (t={t_final:.2e}s) - Staggered Grid',
                save_file=output_file,
                subsample=2,
                scale_v=scale_v_auto,
                scale_stress=scale_stress_auto,
                show_velocity=True,
                show_stress=True)
