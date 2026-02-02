import numpy as np
from _utility import *

# ===== MATERIAL PROPERTIES =====
# Base material properties
c_p_base = 6000  # P-wave velocity (m/s)
c_s_base = 3500  # S-wave velocity (m/s)
rho_base = 2700  # Base density (kg/m^3)

S_base = create_compliance_matrix_from_velocities(c_p_base, c_s_base, rho_base)

# Set up 3D grid
Nx, Ny, Nz = 8, 8, 8  
dx = dy = dz = 0.05  

# Create coordinate arrays
x = np.arange(Nx) * dx - (Nx-1)*dx/2
y = np.arange(Ny) * dy - (Ny-1)*dy/2
z = np.arange(Nz) * dz - (Nz-1)*dz/2
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

# ===== ADD FRACTURE PLANES =====
# Fractures are represented as intersecting planes with different material properties

ADD_FRACTURES = True  # Set to False to disable fractures

# Compute base material Lamé parameters
mu_base = rho_base * c_s_base**2  # Shear modulus
lambda_base = rho_base * c_p_base**2 - 2*mu_base  # First Lamé parameter

# Initialize fracture properties (will be set based on method)
lambda_fracture = None
mu_fracture = None
rho_fracture = None

# For air-filled fractures: very low stiffness
# For fluid-filled: use fluid properties
# For damaged rock: reduced but not zero stiffness
FRACTURE_LAMBDA_RATIO = 0.01  # Fracture λ = 1% of λ
FRACTURE_MU_RATIO = 0.01      # Fracture μ = 1% of μ
lambda_fracture = FRACTURE_LAMBDA_RATIO * lambda_base
mu_fracture = FRACTURE_MU_RATIO * mu_base
# Density for fractures (e.g., air ≈ 1.2 kg/m³, water ≈ 1000 kg/m³)
rho_fracture = 1.2  # Air-filled fracture (or use 1000 for water-filled)
print(f"\nFracture properties (Lamé parameters directly):")
print(f"  λ_fracture = {lambda_fracture:.2e} Pa ({FRACTURE_LAMBDA_RATIO*100:.1f}% of λ)")
print(f"  μ_fracture = {mu_fracture:.2e} Pa ({FRACTURE_MU_RATIO*100:.1f}% of μ)")
print(f"  ρ_fracture = {rho_fracture:.1f} kg/m³ (air-filled)")

# Initialize density model with base density (homogeneous material)
rho_model = np.full((Nz, Ny, Nx), rho_base)

# Initialize fracture mask
fracture_mask = np.zeros((Nz, Ny, Nx), dtype=bool)

if ADD_FRACTURES:
    # Define fracture planes (intersecting planes)
    fracture1_z = Nz // 2  # Horizontal plane
    fracture2_x = Nx // 2  # Vertical plane (x-direction)
    fracture3_y = Ny // 2  # Vertical plane (y-direction)
    fracture_thickness = 0  # Single cell thick
    
    print(f"\nAdding fracture planes:")
    print(f"  Fracture 1: Horizontal plane at z-index {fracture1_z} ± {fracture_thickness}")
    print(f"  Fracture 2: Vertical plane at x-index {fracture2_x} ± {fracture_thickness}")
    print(f"  Fracture 3: Vertical plane at y-index {fracture3_y} ± {fracture_thickness}")
    
    # Create fracture mask
    fracture_cells = 0
    
    for i in range(Nz):
        for j in range(Ny):
            for k in range(Nx):
                is_fracture = False
                
                if abs(i - fracture1_z) <= fracture_thickness:
                    is_fracture = True
                if abs(k - fracture2_x) <= fracture_thickness:
                    is_fracture = True
                if abs(j - fracture3_y) <= fracture_thickness:
                    is_fracture = True
                
                if is_fracture:
                    fracture_mask[i, j, k] = True
                    fracture_cells += 1
                    # Set fracture density (air-filled)
                    rho_model[i, j, k] = rho_fracture
    
    print(f"  Total fracture cells: {fracture_cells} out of {Nx*Ny*Nz} ({100*fracture_cells/(Nx*Ny*Nz):.1f}%)")

# Clip density to reasonable bounds (allow air density ~1.2 kg/m³)
rho_min, rho_max = 1.0, 3000  
rho_model = np.clip(rho_model, rho_min, rho_max)

# Print density statistics
print(f"\nDensity Model Statistics:")
print(f"  Min density: {np.min(rho_model):.1f} kg/m³")
print(f"  Max density: {np.max(rho_model):.1f} kg/m³")
print(f"  Mean density: {np.mean(rho_model):.1f} kg/m³")
print(f"  Std deviation: {np.std(rho_model):.1f} kg/m³")
print(f"  Variation: {(np.max(rho_model) - np.min(rho_model)) / np.mean(rho_model) * 100:.1f}%")
if ADD_FRACTURES:
    # Note: fracture_mask was already created in the fracture section above
    if np.any(fracture_mask):
        print(f"\n  Fracture regions:")
        print(f"    Fracture density range: {np.min(rho_model[fracture_mask]):.1f} - {np.max(rho_model[fracture_mask]):.1f} kg/m³")
        print(f"    Matrix density range: {np.min(rho_model[~fracture_mask]):.1f} - {np.max(rho_model[~fracture_mask]):.1f} kg/m³")

# Check if we need spatially varying S matrix
need_varying_S = ADD_FRACTURES

if need_varying_S:
    # Create spatially varying compliance matrix (6, 6, Nz, Ny, Nx)
    S_model = np.zeros((6, 6, Nz, Ny, Nx))
    
    print(f"\nCreating spatially varying compliance matrix...")
    
    for i in range(Nz):
        for j in range(Ny):
            for k in range(Nx):
                if ADD_FRACTURES and fracture_mask[i, j, k]:
                    S_model[:, :, i, j, k] = create_compliance_matrix_isotropic(
                        lambda_fracture, mu_fracture
                    )
                else:
                    # Use base material properties
                    S_model[:, :, i, j, k] = S_base
    
    S = S_model
    
    if ADD_FRACTURES:
        print(f"  Fracture regions use Lamé parameters: λ={lambda_fracture:.2e}, μ={mu_fracture:.2e} Pa")
        print(f"  Matrix regions use base properties: λ={lambda_base:.2e}, μ={mu_base:.2e} Pa")
else:
    # Use constant compliance matrix (simpler, faster)
    S = S_base
    print(f"\nUsing constant elastic properties (c_p={c_p_base} m/s, c_s={c_s_base} m/s)")

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

# CFL condition: dt < dx / c_p_min for stability
# Use the MINIMUM P-wave velocity to ensure stability everywhere
# (fractures have lower velocity, so we need smaller time step)
if need_varying_S and ADD_FRACTURES:
    # Compute P-wave velocity in fracture regions
    # c_p = sqrt((λ + 2μ)/ρ)
    c_p_fracture = np.sqrt((lambda_fracture + 2*mu_fracture) / rho_fracture)
    # Use the minimum velocity (fracture) for stability
    c_p_min = min(c_p_fracture, c_p_base)
    print(f"\nVelocity analysis for CFL condition:")
    print(f"  Matrix P-wave velocity: {c_p_base:.1f} m/s")
    print(f"  Fracture P-wave velocity: {c_p_fracture:.1f} m/s")
    print(f"  Using minimum velocity for CFL: {c_p_min:.1f} m/s")
else:
    c_p_min = c_p_base  # Use base velocity when constant
    print(f"\nVelocity analysis for CFL condition:")
    print(f"  Using constant P-wave velocity: {c_p_min:.1f} m/s")
dt_max = dx / c_p_min  # Maximum stable time step (CFL limit)
dt = 0.1 * dt_max  # Use 10% of maximum for safety 
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
                title=f'3D Elastic Wave (t={t_final:.2e}s) - Staggered Grid',
                save_file=output_file,
                subsample=2,  
                scale_v=20,  
                scale_stress=5,
                show_velocity=True,
                show_stress=True)

print("\n" + "="*60)
print("Wave evolution complete!")
print("="*60)

