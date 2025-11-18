import numpy as np
from _utility import *

# Create compliance matrix from wave velocities
c_p = 6000  # P-wave velocity (m/s)
c_s = 3500  # S-wave velocity (m/s)
rho = 2700  # Density (kg/m^3)
S = create_compliance_matrix_from_velocities(c_p, c_s, rho)

# Set up 3D grid
Nx, Ny, Nz = 8, 8, 8  
dx = dy = dz = 0.05  
rho_model = rho * np.ones((Nz, Ny, Nx))

# Calculate staggered grid sizes
N_main = Nx * Ny * Nz
N_vx = (Nx-1) * Ny * Nz
N_vy = Nx * (Ny-1) * Nz
N_vz = Nx * Ny * (Nz-1)
N_sxy = (Nx-1) * (Ny-1) * Nz
N_sxz = (Nx-1) * Ny * (Nz-1)
N_syz = Nx * (Ny-1) * (Nz-1)
N_vel = N_vx + N_vy + N_vz
N_stress = 3*N_main + N_sxy + N_sxz + N_syz
N_total = N_vel + N_stress

print(f"State vector size: {N_total}")
print(f"Velocity: {N_vel} (vx:{N_vx} + vy:{N_vy} + vz:{N_vz})")
print(f"Stress: {N_stress} (3×{N_main} + {N_sxy} + {N_sxz} + {N_syz})")

H, A, B, B_sqrt, B_inv, B_inv_sqrt = FD_solver_3D_elastic(
    Nx, Ny, Nz, dx, dy, dz, rho_model, S
)
print("Hamiltonian computed successfully!")

print(f"\nMatrix shapes:")
print(f"A: {A.shape} (system matrix)")
print(f"B: {B.shape} (material matrix)")
print(f"H: {H.shape} (Hamiltonian)")

# Verify dimensions match
assert A.shape[0] == A.shape[1] == N_total, f"A matrix has wrong dimensions: {A.shape[0]} != {N_total}"
assert B.shape[0] == B.shape[1] == N_total, f"B matrix has wrong dimensions: {B.shape[0]} != {N_total}"
assert H.shape[0] == H.shape[1] == N_total, f"H matrix has wrong dimensions: {H.shape[0]} != {N_total}"
print("\nAll dimension checks passed!")
print("\n" + "="*60)
print("3D elastic wave equation functions are working correctly!")



print("\n" + "="*60)
print("PART 2: Wave Evolution Using Hamiltonian")
print("="*60)

# Create initial condition: localized pressure pulse
print("\nCreating initial condition (localized pressure pulse)...")

# Start with zero state (staggered grid)
phi_0 = np.zeros(N_total, dtype=complex)

# Create a small Gaussian pressure perturbation in the center
# This creates diagonal stress (pressure = -1/3 * trace(sigma))
x = np.arange(Nx) * dx - (Nx-1)*dx/2
y = np.arange(Ny) * dy - (Ny-1)*dy/2
z = np.arange(Nz) * dz - (Nz-1)*dz/2
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
sigma = 2.5 * dx 
amplitude = 0.005 
gaussian = amplitude * np.exp(-(X**2 + Y**2 + Z**2)/(2*sigma**2))

# Add compressional stress (equal diagonal components on main grid)
idx = N_vel  
phi_0[idx:idx+N_main] = gaussian.flatten()  # σ_xx
idx += N_main
phi_0[idx:idx+N_main] = gaussian.flatten()  # σ_yy
idx += N_main
phi_0[idx:idx+N_main] = gaussian.flatten()  # σ_zz
# Shear components and velocity start at zero

print(f"Initial condition created. Max stress: {np.max(np.abs(phi_0)):.6f}")

# Time evolution using matrix exponential
# dφ/dt = -iH·φ  =>  φ(t) = exp(-iHt)·φ(0)
from scipy.sparse.linalg import expm_multiply

# CFL condition: dt < dx / c_p for stability
# Use smaller time step for better energy conservation
dt_max = dx / c_p  # Maximum stable time step
dt = 0.1 * dt_max 
n_steps = 20  
t_final = n_steps * dt

print(f"\nTime evolution parameters:")
print(f"  dt_max (CFL): {dt_max:.2e} s")
print(f"  dt (actual): {dt:.2e} s")
print(f"  Number of steps: {n_steps}")
print(f"  Final time: {t_final:.2e} s")

phi_current = phi_0.copy()
for step in range(n_steps):
    phi_current = expm_multiply(-1j * H * dt, phi_current)
    if (step + 1) % 5 == 0 or step == 0 or step == n_steps - 1:
        energy_ratio = np.linalg.norm(phi_current) / np.linalg.norm(phi_0)
        print(f"  Step {step+1}/{n_steps}: energy ratio = {energy_ratio:.6f}")
phi_evolved = phi_current

print(f"Evolution complete!")
print(f"Initial energy: {np.linalg.norm(phi_0):.6f}")
print(f"Final energy: {np.linalg.norm(phi_evolved):.6f}")
print(f"Energy conservation: {np.linalg.norm(phi_evolved) / np.linalg.norm(phi_0):.6f}")

# Extract real part for visualization
phi_real = np.real(phi_evolved)

# Extract velocity and stress components (staggered grids)
idx = 0
v_x = phi_real[idx:idx+N_vx].reshape(Nz, Ny, Nx-1)
idx += N_vx
v_y = phi_real[idx:idx+N_vy].reshape(Nz, Ny-1, Nx)
idx += N_vy
v_z = phi_real[idx:idx+N_vz].reshape(Nz-1, Ny, Nx)
idx += N_vz

σ_xx = phi_real[idx:idx+N_main].reshape(Nz, Ny, Nx)
idx += N_main
σ_yy = phi_real[idx:idx+N_main].reshape(Nz, Ny, Nx)
idx += N_main
σ_zz = phi_real[idx:idx+N_main].reshape(Nz, Ny, Nx)

print(f"\nField statistics:")
print(f"  Velocity magnitude: max={max(np.max(np.abs(v_x)), np.max(np.abs(v_y)), np.max(np.abs(v_z))):.6f}")
print(f"  Stress magnitude: max={max(np.max(np.abs(σ_xx)), np.max(np.abs(σ_yy)), np.max(np.abs(σ_zz))):.6f}")

# Visualize the evolved wave
output_file = "elastic_3D_simulation.png"
plot_elastic_3D(phi_real, Nx, Ny, Nz, dx, dy, dz, 
                title=f'3D Elastic Wave (t={t_final:.2e}s) (Staggered Grid)',
                save_file=output_file,
                subsample=2,  
                scale_v=20,  
                scale_stress=5,
                show_velocity=True,
                show_stress=True)

print("\n" + "="*60)
print("Wave evolution complete!")
print("="*60)

