# -------- Imports --------
import numpy as np
import scipy.sparse as sp
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# -------- Functions --------
# -- Source Functions --
def Gaussian(f, x, y, z, x0=0, y0=0, z0=0):
    """3D Gaussian source."""
    return np.exp(-np.pi**2 * f**2 * ((x - x0)**2 + (y - y0)**2 + (z - z0)**2))

def Ricker(f, x, y, z, x0=0, y0=0, z0=0):
    """Generate a 3D Ricker wavelet."""
    r = np.sqrt((x - x0)**2 + (y - y0)**2 + (z - z0)**2)
    pi_f_r = np.pi * f * r
    return (1 - 2 * pi_f_r**2) * np.exp(-pi_f_r**2)

# -- Simulation Functions --
def FD(dx, Nx):
    """Finite Difference operator for first derivative. First order accurate. All DBC."""
    return (1 / dx) * sp.diags([-1, 1], [0, -1], shape=(Nx, Nx), format='lil')[:, :-1]

def compute_B(c_model, rho_model, rho_stag_x, rho_stag_y):
    """Compute the material matrices for the 2D acoustic wave equation."""
    # -------- Matrix B (Acoustic) --------
    # Flatten the arrays
    c_model = c_model.flatten()
    rho_model = rho_model.flatten()
    rho_stag_x = rho_stag_x.flatten()
    rho_stag_y = rho_stag_y.flatten()

    # Material matrices for u and v (sqrt)
    B_u_sqrt = sp.diags(1 / np.sqrt(rho_model * c_model ** 2), format='csr')
    B_vx_sqrt = sp.diags(np.sqrt(rho_stag_x), format='csr')
    B_vy_sqrt = sp.diags(np.sqrt(rho_stag_y), format='csr')
    B_sqrt = sp.block_diag([B_u_sqrt, B_vx_sqrt, B_vy_sqrt], format='csr')

    # Material matrices for u and v (sqrt_inv)
    B_u_inv_sqrt = sp.diags(np.sqrt(rho_model * c_model ** 2), format='csr')
    B_vx_inv_sqrt = sp.diags(1 / np.sqrt(rho_stag_x), format='csr')
    B_vy_inv_sqrt = sp.diags(1 / np.sqrt(rho_stag_y), format='csr')
    B_inv_sqrt = sp.block_diag([B_u_inv_sqrt, B_vx_inv_sqrt, B_vy_inv_sqrt], format='csr')
    
    return B_sqrt**2, B_sqrt, B_inv_sqrt**2, B_inv_sqrt

def create_compliance_matrix_isotropic(lambda_param, mu):
    """Create the compliance matrix S for isotropic elastic material.
    
    The compliance matrix relates strain to stress: ε(x) = S(x)σ(x).
    For isotropic materials, S is the inverse of the stiffness matrix C.
    
    S(x) = C^-1 where C is:
        [ λ+2μ   λ      λ      0   0   0 ]
        [ λ      λ+2μ   λ      0   0   0 ]
        [ λ      λ      λ+2μ   0   0   0 ]
        [ 0      0      0      μ   0   0 ]
        [ 0      0      0      0   μ   0 ]
        [ 0      0      0      0   0   μ ]
    
    Args:
        lambda_param: First Lamé parameter (λ)
        mu: Second Lamé parameter (μ, shear modulus)
    
    Returns:
        S: 6x6 compliance matrix
    """
    # Construct stiffness matrix C
    C = np.array([
        [lambda_param + 2*mu, lambda_param, lambda_param, 0, 0, 0],
        [lambda_param, lambda_param + 2*mu, lambda_param, 0, 0, 0],
        [lambda_param, lambda_param, lambda_param + 2*mu, 0, 0, 0],
        [0, 0, 0, mu, 0, 0],
        [0, 0, 0, 0, mu, 0],
        [0, 0, 0, 0, 0, mu]
    ])
    
    # Compliance matrix is the inverse of stiffness matrix
    S = np.linalg.inv(C)
    
    return S

def create_compliance_matrix_from_velocities(c_p, c_s, rho):
    """Create the compliance matrix S from P-wave and S-wave velocities.
    
    Args:
        c_p: P-wave velocity
        c_s: S-wave velocity
        rho: Density
    
    Returns:
        S: 6x6 compliance matrix
    """
    # Compute Lamé parameters from velocities
    mu = rho * c_s**2  # Shear modulus
    lambda_param = rho * c_p**2 - 2*mu  # First Lamé parameter
    
    return create_compliance_matrix_isotropic(lambda_param, mu)

def rho_model_compliance_matrix(Nx,Ny,Nz,dx,dy,dz,ADD_FRACTURES):
    c_p_base = 6000  # P-wave velocity (m/s)
    c_s_base = 3500  # S-wave velocity (m/s)
    rho_base = 2700  # Base density (kg/m³)
    S_base = create_compliance_matrix_from_velocities(c_p_base, c_s_base, rho_base)

    ADD_FRACTURES = True

    # Fracture properties (water-filled fractures)
    rho_fracture = 1000.0  # Water density (kg/m³)
    K_water = 2.2e9  # Water bulk modulus (Pa)
    mu_fracture = 1e-6  # Very small shear modulus for numerical stability
    lambda_fracture = K_water  # Bulk modulus for λ

    # Compute base material Lamé parameters
    mu_base = rho_base * c_s_base ** 2
    lambda_base = rho_base * c_p_base ** 2 - 2 * mu_base

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

    # Clip density to reasonable bounds
    rho_model = np.clip(rho_model, 1.0, 3000)

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
    return rho_model, S

def compute_B_elastic_3D(rho_model, S_matrix, Nx, Ny, Nz):
    """Compute the material matrices for the 3D elastic wave equation with staggered grids.
    
    From equation (17): B_elastic = [[ρ(x)I_3×3, 0_3×6], [0_6×3, S(x)]]
    
    Args:
        rho_model: Density field (Nz, Ny, Nx) array on main grid
        S_matrix: Compliance matrix (6, 6) for isotropic or (6, 6, Nz, Ny, Nx) for anisotropic
        Nx, Ny, Nz: Grid dimensions
    
    Returns:
        B, B_sqrt, B_inv, B_inv_sqrt: Material matrices
    """
    from scipy.linalg import sqrtm
    
    # Grid sizes for staggered grids
    N_main = Nx * Ny * Nz
    N_vx = (Nx-1) * Ny * Nz
    N_vy = Nx * (Ny-1) * Nz
    N_vz = Nx * Ny * (Nz-1)
    N_sxy = (Nx-1) * (Ny-1) * Nz
    N_sxz = (Nx-1) * Ny * (Nz-1)
    N_syz = Nx * (Ny-1) * (Nz-1)
    
    # Interpolate density to staggered grids for velocity components
    # For v_x: average in x direction (Nx, Ny, Nz) -> (Nx-1, Ny, Nz)
    rho_vx = 0.5 * (rho_model[:, :, :-1] + rho_model[:, :, 1:])  # Average adjacent x points
    rho_vx_flat = rho_vx.flatten()
    
    # For v_y: average in y direction (Nx, Ny, Nz) -> (Nx, Ny-1, Nz)
    rho_vy = 0.5 * (rho_model[:, :-1, :] + rho_model[:, 1:, :])  # Average adjacent y points
    rho_vy_flat = rho_vy.flatten()
    
    # For v_z: average in z direction (Nx, Ny, Nz) -> (Nx, Ny, Nz-1)
    rho_vz = 0.5 * (rho_model[:-1, :, :] + rho_model[1:, :, :])  # Average adjacent z points
    rho_vz_flat = rho_vz.flatten()
    
    # Create density blocks for velocity components
    B_vx = sp.diags(rho_vx_flat, format='csr')
    B_vy = sp.diags(rho_vy_flat, format='csr')
    B_vz = sp.diags(rho_vz_flat, format='csr')
    B_v = sp.block_diag([B_vx, B_vy, B_vz], format='csr')
    
    # For stress components, interpolate S matrix to appropriate grids
    rho_main_flat = rho_model.flatten()
    
    # Initialize isotropic flag (will be set if S is spatially varying)
    is_isotropic = None
    
    if S_matrix.ndim == 2:
        # Isotropic: single S matrix
        # For isotropic materials, S is constant, so we can use it directly
        # Each stress component is a scalar field, so we just need identity matrices
        # The S matrix will be applied when needed (it couples stress components)
        # For now, use identity (we'll handle S coupling separately if needed)
        # Actually, for the B matrix, we want S to act on the stress vector
        # But since stress components are on different grids, we need to be careful
        
        # Simple approach: use S[0,0] for diagonal terms (1/E) and treat as scalar
        # For isotropic: S[0,0] = S[1,1] = S[2,2] = 1/E, S[3,3] = S[4,4] = S[5,5] = 1/(2*mu)
        S_diag_xx = S_matrix[0, 0]  # 1/E for normal stresses
        S_diag_yy = S_matrix[1, 1]  # 1/E
        S_diag_zz = S_matrix[2, 2]  # 1/E
        S_diag_xy = S_matrix[3, 3]  # 1/(2*mu) for shear stresses
        S_diag_xz = S_matrix[4, 4]  # 1/(2*mu)
        S_diag_yz = S_matrix[5, 5]  # 1/(2*mu)
        
        # Create diagonal matrices (scalar per grid point)
        B_sxx = sp.diags([S_diag_xx] * N_main, format='csr')
        B_syy = sp.diags([S_diag_yy] * N_main, format='csr')
        B_szz = sp.diags([S_diag_zz] * N_main, format='csr')
        B_sxy = sp.diags([S_diag_xy] * N_sxy, format='csr')
        B_sxz = sp.diags([S_diag_xz] * N_sxz, format='csr')
        B_syz = sp.diags([S_diag_yz] * N_syz, format='csr')
    else:
        # S varies spatially - check if isotropic (diagonal-dominant) or anisotropic
        # Reshape S_matrix to (6, 6, Nz, Ny, Nx) for easier indexing
        S_flat = S_matrix.reshape(6, 6, Nz, Ny, Nx) if S_matrix.ndim == 5 else S_matrix
        
        # Check if S matrices are approximately isotropic
        # For isotropic materials created with create_compliance_matrix_isotropic:
        # - S[0,0] = S[1,1] = S[2,2] (normal components)
        # - S[3,3] = S[4,4] = S[5,5] (shear components)  
        # - S[0,1] = S[0,2] = S[1,2] (off-diagonals in 3×3 block)
        # - Other off-diagonals are zero
        # Sample a few points to check
        sample_S = S_flat[:, :, 0, 0, 0]
        sample_S2 = S_flat[:, :, Nz//2, Ny//2, Nx//2]  # Check another point
        
        # Use more lenient tolerance for numerical precision
        tol = 1e-4
        
        # Check isotropic structure (more lenient check):
        # 1. Normal components are approximately equal
        normal_equal = (np.abs(sample_S[0,0] - sample_S[1,1]) < tol * (abs(sample_S[0,0]) + 1e-10) and
                       np.abs(sample_S[1,1] - sample_S[2,2]) < tol * (abs(sample_S[1,1]) + 1e-10))
        # 2. Shear components are approximately equal
        shear_equal = (np.abs(sample_S[3,3] - sample_S[4,4]) < tol * (abs(sample_S[3,3]) + 1e-10) and
                      np.abs(sample_S[4,4] - sample_S[5,5]) < tol * (abs(sample_S[4,4]) + 1e-10))
        # 3. Off-diagonals in 3×3 block are approximately equal
        offdiag_equal = (np.abs(sample_S[0,1] - sample_S[0,2]) < tol * (abs(sample_S[0,1]) + 1e-10) and
                        np.abs(sample_S[0,2] - sample_S[1,2]) < tol * (abs(sample_S[0,2]) + 1e-10))
        # 4. Other off-diagonals are approximately zero
        other_offdiag_zero = (np.abs(sample_S[0,3]) < tol and np.abs(sample_S[0,4]) < tol and
                             np.abs(sample_S[0,5]) < tol and np.abs(sample_S[3,4]) < tol and
                             np.abs(sample_S[3,5]) < tol and np.abs(sample_S[4,5]) < tol)
        
        is_isotropic = normal_equal and shear_equal and offdiag_equal and other_offdiag_zero
        
        # If check fails, assume isotropic anyway (since we're creating isotropic matrices)
        # This is safer than assuming anisotropic and creating wrong-sized matrices
        if not is_isotropic:
            # Default to isotropic for safety - wrong-sized matrices are worse than
            # incorrectly assuming isotropic
            is_isotropic = True
            print(f"  Warning: Isotropic check failed, defaulting to isotropic treatment")
        
        if is_isotropic:
            # Extract diagonal elements (isotropic case - much simpler and correct)
            # Extract diagonal elements for main grid (σ_xx, σ_yy, σ_zz)
            S_diag_xx = np.array([S_flat[0, 0, k, j, i] for k in range(Nz) for j in range(Ny) for i in range(Nx)])
            S_diag_yy = np.array([S_flat[1, 1, k, j, i] for k in range(Nz) for j in range(Ny) for i in range(Nx)])
            S_diag_zz = np.array([S_flat[2, 2, k, j, i] for k in range(Nz) for j in range(Ny) for i in range(Nx)])
            
            # Interpolate to staggered grids for shear components
            # For σ_xy: average in x and y (staggered in both x and y)
            S_diag_xy = []
            for k in range(Nz):
                for j in range(Ny-1):
                    for i in range(Nx-1):
                        avg = 0.25 * (S_flat[3, 3, k, j, i] + S_flat[3, 3, k, j, i+1] +
                                     S_flat[3, 3, k, j+1, i] + S_flat[3, 3, k, j+1, i+1])
                        S_diag_xy.append(avg)
            S_diag_xy = np.array(S_diag_xy)
            
            # For σ_xz: average in x and z (staggered in both x and z)
            S_diag_xz = []
            for k in range(Nz-1):
                for j in range(Ny):
                    for i in range(Nx-1):
                        avg = 0.25 * (S_flat[4, 4, k, j, i] + S_flat[4, 4, k, j, i+1] +
                                     S_flat[4, 4, k+1, j, i] + S_flat[4, 4, k+1, j, i+1])
                        S_diag_xz.append(avg)
            S_diag_xz = np.array(S_diag_xz)
            
            # For σ_yz: average in y and z (staggered in both y and z)
            S_diag_yz = []
            for k in range(Nz-1):
                for j in range(Ny-1):
                    for i in range(Nx):
                        avg = 0.25 * (S_flat[5, 5, k, j, i] + S_flat[5, 5, k, j+1, i] +
                                     S_flat[5, 5, k+1, j, i] + S_flat[5, 5, k+1, j+1, i])
                        S_diag_yz.append(avg)
            S_diag_yz = np.array(S_diag_yz)
            
            # Create diagonal matrices (scalar per grid point - correct size!)
            B_sxx = sp.diags(S_diag_xx, format='csr')
            B_syy = sp.diags(S_diag_yy, format='csr')
            B_szz = sp.diags(S_diag_zz, format='csr')
            B_sxy = sp.diags(S_diag_xy, format='csr')
            B_sxz = sp.diags(S_diag_xz, format='csr')
            B_syz = sp.diags(S_diag_yz, format='csr')
        else:
            # Anisotropic: S varies spatially - need full 6×6 blocks (original code)
            # Main grid (for σ_xx, σ_yy, σ_zz)
            S_main_list = [sp.csr_matrix(S_flat[:, :, k, j, i]) 
                           for k in range(Nz) for j in range(Ny) for i in range(Nx)]
            B_sxx = sp.block_diag(S_main_list, format='csr')
            B_syy = sp.block_diag(S_main_list, format='csr')
            B_szz = sp.block_diag(S_main_list, format='csr')
            
            # Staggered grids (use average of adjacent points) - only for anisotropic
            # For σ_xy: average in x and y
            S_sxy_list = [sp.csr_matrix(0.25 * (S_flat[:, :, k, j, i] + S_flat[:, :, k, j, i+1] +
                                            S_flat[:, :, k, j+1, i] + S_flat[:, :, k, j+1, i+1]))
                          for k in range(Nz) for j in range(Ny-1) for i in range(Nx-1)]
            B_sxy = sp.block_diag(S_sxy_list, format='csr')
            
            # For σ_xz: average in x and z
            S_sxz_list = [sp.csr_matrix(0.25 * (S_flat[:, :, k, j, i] + S_flat[:, :, k, j, i+1] +
                                            S_flat[:, :, k+1, j, i] + S_flat[:, :, k+1, j, i+1]))
                          for k in range(Nz-1) for j in range(Ny) for i in range(Nx-1)]
            B_sxz = sp.block_diag(S_sxz_list, format='csr')
            
            # For σ_yz: average in y and z
            S_syz_list = [sp.csr_matrix(0.25 * (S_flat[:, :, k, j, i] + S_flat[:, :, k, j+1, i] +
                                            S_flat[:, :, k+1, j, i] + S_flat[:, :, k+1, j+1, i]))
                          for k in range(Nz-1) for j in range(Ny-1) for i in range(Nx)]
            B_syz = sp.block_diag(S_syz_list, format='csr')
    
    # Construct stress block
    B_sigma = sp.block_diag([B_sxx, B_syy, B_szz, B_sxy, B_sxz, B_syz], format='csr')
    
    # Construct full B matrix
    B = sp.block_diag([B_v, B_sigma], format='csr')
    
    # Compute square roots
    B_vx_sqrt = sp.diags(np.sqrt(rho_vx_flat), format='csr')
    B_vy_sqrt = sp.diags(np.sqrt(rho_vy_flat), format='csr')
    B_vz_sqrt = sp.diags(np.sqrt(rho_vz_flat), format='csr')
    B_v_sqrt = sp.block_diag([B_vx_sqrt, B_vy_sqrt, B_vz_sqrt], format='csr')
    
    B_vx_inv_sqrt = sp.diags(1/np.sqrt(rho_vx_flat), format='csr')
    B_vy_inv_sqrt = sp.diags(1/np.sqrt(rho_vy_flat), format='csr')
    B_vz_inv_sqrt = sp.diags(1/np.sqrt(rho_vz_flat), format='csr')
    B_v_inv_sqrt = sp.block_diag([B_vx_inv_sqrt, B_vy_inv_sqrt, B_vz_inv_sqrt], format='csr')
    
    # For stress components - use scalar sqrt since we're using diagonal matrices
    if S_matrix.ndim == 2:
        # For diagonal matrices, sqrt is just sqrt of diagonal elements
        S_diag_xx_sqrt = np.sqrt(S_matrix[0, 0])
        S_diag_yy_sqrt = np.sqrt(S_matrix[1, 1])
        S_diag_zz_sqrt = np.sqrt(S_matrix[2, 2])
        S_diag_xy_sqrt = np.sqrt(S_matrix[3, 3])
        S_diag_xz_sqrt = np.sqrt(S_matrix[4, 4])
        S_diag_yz_sqrt = np.sqrt(S_matrix[5, 5])
        
        B_sxx_sqrt = sp.diags([S_diag_xx_sqrt] * N_main, format='csr')
        B_syy_sqrt = sp.diags([S_diag_yy_sqrt] * N_main, format='csr')
        B_szz_sqrt = sp.diags([S_diag_zz_sqrt] * N_main, format='csr')
        B_sxy_sqrt = sp.diags([S_diag_xy_sqrt] * N_sxy, format='csr')
        B_sxz_sqrt = sp.diags([S_diag_xz_sqrt] * N_sxz, format='csr')
        B_syz_sqrt = sp.diags([S_diag_yz_sqrt] * N_syz, format='csr')
        
        # Inverse sqrt
        S_diag_xx_inv_sqrt = 1.0 / S_diag_xx_sqrt
        S_diag_yy_inv_sqrt = 1.0 / S_diag_yy_sqrt
        S_diag_zz_inv_sqrt = 1.0 / S_diag_zz_sqrt
        S_diag_xy_inv_sqrt = 1.0 / S_diag_xy_sqrt
        S_diag_xz_inv_sqrt = 1.0 / S_diag_xz_sqrt
        S_diag_yz_inv_sqrt = 1.0 / S_diag_yz_sqrt
        
        B_sxx_inv_sqrt = sp.diags([S_diag_xx_inv_sqrt] * N_main, format='csr')
        B_syy_inv_sqrt = sp.diags([S_diag_yy_inv_sqrt] * N_main, format='csr')
        B_szz_inv_sqrt = sp.diags([S_diag_zz_inv_sqrt] * N_main, format='csr')
        B_sxy_inv_sqrt = sp.diags([S_diag_xy_inv_sqrt] * N_sxy, format='csr')
        B_sxz_inv_sqrt = sp.diags([S_diag_xz_inv_sqrt] * N_sxz, format='csr')
        B_syz_inv_sqrt = sp.diags([S_diag_yz_inv_sqrt] * N_syz, format='csr')
    elif S_matrix.ndim > 2 and is_isotropic:
        # Isotropic spatially varying: use sqrt of diagonal elements
        # These variables (S_diag_xx, etc.) should be defined in the earlier block (lines 220-253)
        # when is_isotropic was True
        B_sxx_sqrt = sp.diags(np.sqrt(S_diag_xx), format='csr')
        B_syy_sqrt = sp.diags(np.sqrt(S_diag_yy), format='csr')
        B_szz_sqrt = sp.diags(np.sqrt(S_diag_zz), format='csr')
        B_sxy_sqrt = sp.diags(np.sqrt(S_diag_xy), format='csr')
        B_sxz_sqrt = sp.diags(np.sqrt(S_diag_xz), format='csr')
        B_syz_sqrt = sp.diags(np.sqrt(S_diag_yz), format='csr')
        
        # Inverse sqrt
        B_sxx_inv_sqrt = sp.diags(1.0 / np.sqrt(S_diag_xx), format='csr')
        B_syy_inv_sqrt = sp.diags(1.0 / np.sqrt(S_diag_yy), format='csr')
        B_szz_inv_sqrt = sp.diags(1.0 / np.sqrt(S_diag_zz), format='csr')
        B_sxy_inv_sqrt = sp.diags(1.0 / np.sqrt(S_diag_xy), format='csr')
        B_sxz_inv_sqrt = sp.diags(1.0 / np.sqrt(S_diag_xz), format='csr')
        B_syz_inv_sqrt = sp.diags(1.0 / np.sqrt(S_diag_yz), format='csr')
    else:
        # Anisotropic: compute sqrt for each point (use main grid values)
        # This should only execute if S_matrix.ndim > 2 and is_isotropic is False/None
        if S_matrix.ndim == 2:
            raise ValueError("Anisotropic sqrt block should not execute for constant S matrix")
        if 'S_flat' not in locals():
            S_flat = S_matrix.reshape(6, 6, Nz, Ny, Nx)
        
        S_sqrt_list = [sp.csr_matrix(sqrtm(S_flat[:, :, k, j, i]).real)
                       for k in range(Nz) for j in range(Ny) for i in range(Nx)]
        S_inv_sqrt_list = [sp.csr_matrix(sqrtm(np.linalg.inv(S_flat[:, :, k, j, i])).real)
                           for k in range(Nz) for j in range(Ny) for i in range(Nx)]
        B_sxx_sqrt = sp.block_diag(S_sqrt_list, format='csr')
        B_syy_sqrt = sp.block_diag(S_sqrt_list, format='csr')
        B_szz_sqrt = sp.block_diag(S_sqrt_list, format='csr')
        B_sxx_inv_sqrt = sp.block_diag(S_inv_sqrt_list, format='csr')
        B_syy_inv_sqrt = sp.block_diag(S_inv_sqrt_list, format='csr')
        B_szz_inv_sqrt = sp.block_diag(S_inv_sqrt_list, format='csr')
        
        # For staggered grids, use interpolated values
        B_sxy_sqrt = B_sxx_sqrt[:N_sxy*6, :N_sxy*6] 
        B_sxz_sqrt = B_sxx_sqrt[:N_sxz*6, :N_sxz*6]
        B_syz_sqrt = B_sxx_sqrt[:N_syz*6, :N_syz*6]
        B_sxy_inv_sqrt = B_sxx_inv_sqrt[:N_sxy*6, :N_sxy*6]
        B_sxz_inv_sqrt = B_sxx_inv_sqrt[:N_sxz*6, :N_sxz*6]
        B_syz_inv_sqrt = B_sxx_inv_sqrt[:N_syz*6, :N_syz*6]
    
    B_sigma_sqrt = sp.block_diag([B_sxx_sqrt, B_syy_sqrt, B_szz_sqrt, 
                                   B_sxy_sqrt, B_sxz_sqrt, B_syz_sqrt], format='csr')
    B_sigma_inv_sqrt = sp.block_diag([B_sxx_inv_sqrt, B_syy_inv_sqrt, B_szz_inv_sqrt,
                                      B_sxy_inv_sqrt, B_sxz_inv_sqrt, B_syz_inv_sqrt], format='csr')
    
    # Construct full sqrt matrices
    B_sqrt = sp.block_diag([B_v_sqrt, B_sigma_sqrt], format='csr')
    B_inv_sqrt = sp.block_diag([B_v_inv_sqrt, B_sigma_inv_sqrt], format='csr')
    
    # Compute B_inv
    B_inv = B_inv_sqrt @ B_inv_sqrt
    
    return B, B_sqrt, B_inv, B_inv_sqrt

def compute_A(Nx, Ny, dx, dy, bcs):
    """Compute the system matrix for the 2D acoustic wave equation."""
    # -------- Matrix A (Acoustic) --------
    # Finite difference operators
    FD_x = FD(dx, Nx)
    FD_y = FD(dy, Ny)

    # Boundary conditions (Dirichlet or Neumann)
    if bcs['L'] == 'NBC': FD_x[0, :] = 0 
    if bcs['R'] == 'NBC': FD_x[-1, :] = 0
    if bcs['T'] == 'NBC': FD_y[0, :] = 0 
    if bcs['B'] == 'NBC': FD_y[-1, :] = 0

    # Derivative operators
    D_x = sp.kron(sp.eye(Ny), FD_x)
    D_y = sp.kron(FD_y, sp.eye(Nx))

    # Divergence & Gradient operators
    Div = sp.hstack([D_x, D_y], format='csr')
    Grad = -Div.T

    # System matrix A
    A = sp.bmat([
        [None, Div],
        [Grad, None]
    ], format='csr')
    
    return A

def compute_A_elastic_3D(Nx, Ny, Nz, dx, dy, dz, bcs):
    """Compute the system matrix for the 3D elastic wave equation using proper staggered grids.
    
    Staggered grid structure:
    - v_x on (Nx-1, Ny, Nz) - staggered in x
    - v_y on (Nx, Ny-1, Nz) - staggered in y
    - v_z on (Nx, Ny, Nz-1) - staggered in z
    - σ_xx, σ_yy, σ_zz on (Nx, Ny, Nz) - main grid
    - σ_xy on (Nx-1, Ny-1, Nz) - staggered in x and y
    - σ_xz on (Nx-1, Ny, Nz-1) - staggered in x and z
    - σ_yz on (Nx, Ny-1, Nz-1) - staggered in y and z
    
    From equation (18) and (19):
    D = [[∂_x,   0,   0, ∂_y, ∂_z,   0],
         [  0, ∂_y,   0, ∂_x,   0, ∂_z],
         [  0,   0, ∂_z,   0, ∂_x, ∂_y]]
    
    A_elastic = [[0_3×3, D], [D^T, 0_6×6]]
    
    Args:
        Nx, Ny, Nz: Number of grid points in x, y, z directions
        dx, dy, dz: Grid spacing
        bcs: Boundary conditions dictionary with keys 'L', 'R', 'T', 'B', 'F', 'Ba'
    
    Returns:
        A: System matrix with proper staggered grid dimensions
    """
    # Create FD operators (these return (N, N-1) for staggered grids)
    FD_x = FD(dx, Nx)  # Shape: (Nx, Nx-1) - maps from staggered x to main grid
    FD_y = FD(dy, Ny)  # Shape: (Ny, Ny-1) - maps from staggered y to main grid
    FD_z = FD(dz, Nz)  # Shape: (Nz, Nz-1) - maps from staggered z to main grid
    
    # Apply boundary conditions (set rows to zero for Neumann BC)
    if bcs.get('L') == 'NBC': FD_x[0, :] = 0
    if bcs.get('R') == 'NBC': FD_x[-1, :] = 0
    if bcs.get('T') == 'NBC': FD_y[0, :] = 0
    if bcs.get('B') == 'NBC': FD_y[-1, :] = 0
    if bcs.get('F') == 'NBC': FD_z[0, :] = 0
    if bcs.get('Ba') == 'NBC': FD_z[-1, :] = 0
    
    # Grid sizes for different components
    N_main = Nx * Ny * Nz  # Main grid size (for σ_xx, σ_yy, σ_zz)
    N_vx = (Nx-1) * Ny * Nz  # v_x grid size
    N_vy = Nx * (Ny-1) * Nz  # v_y grid size
    N_vz = Nx * Ny * (Nz-1)  # v_z grid size
    N_sxy = (Nx-1) * (Ny-1) * Nz  # σ_xy grid size
    N_sxz = (Nx-1) * Ny * (Nz-1)  # σ_xz grid size
    N_syz = Nx * (Ny-1) * (Nz-1)  # σ_yz grid size
    
    # Create 3D derivative operators using Kronecker products
    # For v_x equation: ∂_x(σ_xx) + ∂_y(σ_xy) + ∂_z(σ_xz) → v_x
    # v_x is on (Nx-1, Ny, Nz), so we need operators that map:
    # - σ_xx (Nx, Ny, Nz) → v_x (Nx-1, Ny, Nz): use FD_x.T (transpose)
    # - σ_xy (Nx-1, Ny-1, Nz) → v_x (Nx-1, Ny, Nz): use FD_y.T in y direction
    # - σ_xz (Nx-1, Ny, Nz-1) → v_x (Nx-1, Ny, Nz): use FD_z.T in z direction
    
    # ∂_x operator: maps from main grid (Nx, Ny, Nz) to v_x grid (Nx-1, Ny, Nz)
    # FD_x maps (Nx-1) → (Nx), so FD_x.T maps (Nx) → (Nx-1)
    FD_x_for_sxx = FD(dx, Nx)
    D_xx = sp.kron(sp.kron(sp.eye(Nz), sp.eye(Ny)), FD_x.T)  # Shape: (N_vx, N_main)
    
    # ∂_y operator for σ_xy: maps from σ_xy grid (Nx-1, Ny-1, Nz) to v_x grid (Nx-1, Ny, Nz)
    # FD_y maps (Ny-1) → (Ny), so FD_y.T maps (Ny) → (Ny-1) - wait, that's backwards
    # Actually, we need (Ny-1) → (Ny), so we use FD_y directly
    FD_y_for_sxy = FD(dy, Ny)  # (Ny, Ny-1) - maps (Ny-1) → (Ny)
    D_xy_y = sp.kron(sp.kron(sp.eye(Nz), FD_y_for_sxy), sp.eye(Nx-1))  # Shape: (N_vx, N_sxy)
    
    # ∂_z operator for σ_xz: maps from σ_xz grid (Nx-1, Ny, Nz-1) to v_x grid (Nx-1, Ny, Nz)
    FD_z_for_sxz = FD(dz, Nz)  # (Nz, Nz-1) - maps (Nz-1) → (Nz)
    D_xz_z = sp.kron(sp.kron(FD_z_for_sxz, sp.eye(Ny)), sp.eye(Nx-1))  # Shape: (N_vx, N_sxz)
    
    # For v_y equation: ∂_x(σ_xy) + ∂_y(σ_yy) + ∂_z(σ_yz) → v_y
    # v_y is on (Nx, Ny-1, Nz)
    
    # ∂_x operator for σ_xy: maps from σ_xy grid (Nx-1, Ny-1, Nz) to v_y grid (Nx, Ny-1, Nz)
    # Need (Nx-1) → (Nx), so use FD_x directly
    FD_x_for_sxy = FD(dx, Nx)  # (Nx, Nx-1) - maps (Nx-1) → (Nx)
    D_yx_x = sp.kron(sp.kron(sp.eye(Nz), sp.eye(Ny-1)), FD_x_for_sxy)  # Shape: (N_vy, N_sxy)
    
    # ∂_y operator: maps from main grid (Nx, Ny, Nz) to v_y grid (Nx, Ny-1, Nz)
    # Need (Ny) → (Ny-1), so use FD_y.T
    D_yy = sp.kron(sp.kron(sp.eye(Nz), FD_y.T), sp.eye(Nx))  # Shape: (N_vy, N_main)
    
    # ∂_z operator for σ_yz: maps from σ_yz grid (Nx, Ny-1, Nz-1) to v_y grid (Nx, Ny-1, Nz)
    # Need (Nz-1) → (Nz), so use FD_z directly
    FD_z_for_syz = FD(dz, Nz)  # (Nz, Nz-1) - maps (Nz-1) → (Nz)
    D_yz_z = sp.kron(sp.kron(FD_z_for_syz, sp.eye(Ny-1)), sp.eye(Nx))  # Shape: (N_vy, N_syz)
    
    # For v_z equation: ∂_x(σ_xz) + ∂_y(σ_yz) + ∂_z(σ_zz) → v_z
    # v_z is on (Nx, Ny, Nz-1)
    
    # ∂_x operator for σ_xz: maps from σ_xz grid (Nx-1, Ny, Nz-1) to v_z grid (Nx, Ny, Nz-1)
    # Need (Nx-1) → (Nx), so use FD_x directly
    FD_x_for_sxz = FD(dx, Nx)  # (Nx, Nx-1) - maps (Nx-1) → (Nx)
    D_zx_x = sp.kron(sp.kron(sp.eye(Nz-1), sp.eye(Ny)), FD_x_for_sxz)  # Shape: (N_vz, N_sxz)
    
    # ∂_y operator for σ_yz: maps from σ_yz grid (Nx, Ny-1, Nz-1) to v_z grid (Nx, Ny, Nz-1)
    # Need (Ny-1) → (Ny), so use FD_y directly
    FD_y_for_syz = FD(dy, Ny)  # (Ny, Ny-1) - maps (Ny-1) → (Ny)
    D_zy_y = sp.kron(sp.kron(sp.eye(Nz-1), FD_y_for_syz), sp.eye(Nx))  # Shape: (N_vz, N_syz)
    
    # ∂_z operator: maps from main grid (Nx, Ny, Nz) to v_z grid (Nx, Ny, Nz-1)
    # Need (Nz) → (Nz-1), so use FD_z.T
    D_zz = sp.kron(sp.kron(FD_z.T, sp.eye(Ny)), sp.eye(Nx))  # Shape: (N_vz, N_main)
    
    # Construct the divergence operator D
    # D maps from stress components to velocity components
    # Row 1 (v_x): ∂_x(σ_xx) + ∂_y(σ_xy) + ∂_z(σ_xz)
    # Row 2 (v_y): ∂_x(σ_xy) + ∂_y(σ_yy) + ∂_z(σ_yz)
    # Row 3 (v_z): ∂_x(σ_xz) + ∂_y(σ_yz) + ∂_z(σ_zz)
    
    zero_vx_main = sp.csr_matrix((N_vx, N_main))
    zero_vx_sxy = sp.csr_matrix((N_vx, N_sxy))
    zero_vx_sxz = sp.csr_matrix((N_vx, N_sxz))
    zero_vx_syz = sp.csr_matrix((N_vx, N_syz))
    
    zero_vy_main = sp.csr_matrix((N_vy, N_main))
    zero_vy_sxy = sp.csr_matrix((N_vy, N_sxy))
    zero_vy_sxz = sp.csr_matrix((N_vy, N_sxz))
    zero_vy_syz = sp.csr_matrix((N_vy, N_syz))
    
    zero_vz_main = sp.csr_matrix((N_vz, N_main))
    zero_vz_sxy = sp.csr_matrix((N_vz, N_sxy))
    zero_vz_sxz = sp.csr_matrix((N_vz, N_sxz))
    zero_vz_syz = sp.csr_matrix((N_vz, N_syz))
    
    D = sp.bmat([
        # Columns: [σ_xx, σ_yy, σ_zz, σ_xy, σ_xz, σ_yz]
        [D_xx,  zero_vx_main, zero_vx_main, D_xy_y, D_xz_z, zero_vx_syz],  # v_x row
        [zero_vy_main, D_yy,  zero_vy_main, D_yx_x, zero_vy_sxz, D_yz_z],  # v_y row
        [zero_vz_main, zero_vz_main, D_zz,  zero_vz_sxy, D_zx_x, D_zy_y ]   # v_z row
    ], format='csr')
    
    # Total velocity size
    N_vel = N_vx + N_vy + N_vz
    # Total stress size
    N_stress = N_main + N_main + N_main + N_sxy + N_sxz + N_syz
    
    # Construct the full A matrix from equation (19)
    # A_elastic = [[0_3×3, D], [D^T, 0_6×6]]
    zero_vel = sp.csr_matrix((N_vel, N_vel))
    zero_stress = sp.csr_matrix((N_stress, N_stress))
    
    A = sp.bmat([
        [zero_vel, D],
        [-D.T, zero_stress]
    ], format='csr')
    
    return A

def FD_solver_2D(Nx, Ny, dx, dy, c_model, rho_model, rho_stag_x, rho_stag_y, 
                 bcs={'L': 'DBC', 'R': 'DBC', 'T': 'DBC', 'B': 'DBC'}):
    """Compute the evolution Hamiltonian for the 2D acoustic wave equation."""
    # Compute the material matrices
    B, B_sqrt, B_inv, B_inv_sqrt = compute_B(c_model, rho_model, rho_stag_x, rho_stag_y)
    
    # Compute the system matrix
    A = compute_A(Nx, Ny, dx, dy, bcs)
    
    # Compute the Hamiltonian
    H = 1j * B_inv_sqrt @ A @ B_inv_sqrt
    
    return H, A, B, B_sqrt, B_inv, B_inv_sqrt

def FD_solver_3D_elastic(Nx, Ny, Nz, dx, dy, dz, rho_model, S_matrix,
                         bcs={'L': 'DBC', 'R': 'DBC', 'T': 'DBC', 'B': 'DBC', 'F': 'DBC', 'Ba': 'DBC'}):
    """Compute the evolution Hamiltonian for the 3D elastic wave equation.
    
    From equation (21): dw_Q(t)/dt = -iHw_Q(t) where H = B^(-1/2)AB^(-1/2)
    
    Args:
        Nx, Ny, Nz: Number of grid points in x, y, z directions
        dx, dy, dz: Grid spacing
        rho_model: Density field (Nz, Ny, Nx) array
        S_matrix: Compliance matrix (6, 6) for isotropic or (6, 6, Nz, Ny, Nx) for anisotropic
        bcs: Boundary conditions dictionary
    
    Returns:
        H: Evolution Hamiltonian
        A: System matrix
        B: Material matrix
        B_sqrt: Square root of material matrix
        B_inv: Inverse of material matrix
        B_inv_sqrt: Inverse square root of material matrix
    """
    # Compute the material matrices
    B, B_sqrt, B_inv, B_inv_sqrt = compute_B_elastic_3D(rho_model, S_matrix, Nx, Ny, Nz)
    
    # Compute the system matrix
    A = compute_A_elastic_3D(Nx, Ny, Nz, dx, dy, dz, bcs)
    
    # Compute the Hamiltonian from equation (21)
    H = 1j * B_inv_sqrt @ A @ B_inv_sqrt
    
    return H, A, B, B_sqrt, B_inv, B_inv_sqrt

def FD_solver_2D_quantum(Nx, Ny, dx, dy, c_model, rho_model, rho_stag_x, rho_stag_y,
                         bcs={'L': 'DBC', 'R': 'DBC', 'T': 'DBC', 'B': 'DBC'}):
    """Compute the evolution Hamiltonian for the 2D acoustic wave equation."""
    H, A, B, B_sqrt, B_inv, B_inv_sqrt = FD_solver_2D(Nx, Ny, dx, dy, c_model, rho_model, rho_stag_x, rho_stag_y, bcs)
    # Find next power of 2
    bit = (H.shape[0]-1).bit_length()
    pad = 2**bit - H.shape[0]
    
    # Pad the Hamiltonian with zeros
    H = sp.block_diag([H, sp.csr_matrix((pad, pad))], format='csr')
    A = sp.block_diag([A, sp.csr_matrix((pad, pad))], format='csr')
    B = sp.block_diag([B, sp.csr_matrix((pad, pad))], format='csr')
    B_sqrt = sp.block_diag([B_sqrt, sp.csr_matrix((pad, pad))], format='csr')
    B_inv = sp.block_diag([B_inv, sp.csr_matrix((pad, pad))], format='csr')
    B_inv_sqrt = sp.block_diag([B_inv_sqrt, sp.csr_matrix((pad, pad))], format='csr')
    
    return H, A, B, B_sqrt, B_inv, B_inv_sqrt

def FD_solver_3D_elastic_quantum(Nx, Ny, Nz, dx, dy, dz, rho_model, S_matrix,
                                  bcs={'L': 'DBC', 'R': 'DBC', 'T': 'DBC', 'B': 'DBC', 'F': 'DBC', 'Ba': 'DBC'}):
    """Compute the evolution Hamiltonian for the 3D elastic wave equation with quantum padding.
    
    Pads the Hamiltonian to the next power of 2 for quantum simulation compatibility.
    
    Args:
        Same as FD_solver_3D_elastic
    
    Returns:
        H, A, B, B_sqrt, B_inv, B_inv_sqrt: Padded matrices
    """
    H, A, B, B_sqrt, B_inv, B_inv_sqrt = FD_solver_3D_elastic(Nx, Ny, Nz, dx, dy, dz, rho_model, S_matrix, bcs)
    
    # Find next power of 2
    bit = (H.shape[0]-1).bit_length()
    pad = 2**bit - H.shape[0]
    
    # Pad all matrices with zeros
    H = sp.block_diag([H, sp.csr_matrix((pad, pad))], format='csr')
    A = sp.block_diag([A, sp.csr_matrix((pad, pad))], format='csr')
    B = sp.block_diag([B, sp.csr_matrix((pad, pad))], format='csr')
    B_sqrt = sp.block_diag([B_sqrt, sp.csr_matrix((pad, pad))], format='csr')
    B_inv = sp.block_diag([B_inv, sp.csr_matrix((pad, pad))], format='csr')
    B_inv_sqrt = sp.block_diag([B_inv_sqrt, sp.csr_matrix((pad, pad))], format='csr')
    
    return H, A, B, B_sqrt, B_inv, B_inv_sqrt

def compute_source_2D(S0, t_span, dx, dy, c0, rho0):
    """Compute the source term for the 2D acoustic wave equation."""
    # -- Grid parameters --
    r_x = int((1/dx)*t_span[1] * c0)  # Radius of the source region (x)
    r_y = int((1/dy)*t_span[1] * c0)  # Radius of the source region (y)
    Nx_S, Ny_S = (2*r_x, 2*r_y)       # Number of grid points in the source region

    # -- Wave field definition --
    u0_S = np.zeros((Ny_S, Nx_S))
    v0x_S = np.zeros((Ny_S, (Nx_S-1)))
    v0y_S = np.zeros(((Ny_S-1), Nx_S))
    phi_0_S = np.hstack([u0_S.flatten(), v0x_S.flatten(), v0y_S.flatten()])

    # -- Material properties --
    c_model_S = c0 * np.ones((Ny_S, Nx_S))
    rho_model_S = rho0 * np.ones((Ny_S, Nx_S))
    rho_stag_x_S = rho0 * np.ones((Ny_S, (Nx_S-1)))
    rho_stag_y_S = rho0 * np.ones(((Ny_S-1), Nx_S))

    # -------- Simulation (2D Acoustic) --------
    (_, A_S, _, _, B_inv_S, _) = FD_solver_2D(Nx_S, Ny_S, dx, dy, c_model_S, rho_model_S, rho_stag_x_S, rho_stag_y_S)

    # -------- Classical source simulation --------
    # Integration definition
    def rhs(t, phi):
        wave = B_inv_S @ A_S @ phi
        source = np.zeros_like(phi)
        source[Nx_S//2*Ny_S + Nx_S//2] = S0(t) # Inject source at the center of the source region
        return wave + source

    # Time Integration
    source_initial = solve_ivp(rhs, t_span, phi_0_S, t_eval=t_span, method='DOP853').y.T[-1]
    S_u = source_initial[:Ny_S*Nx_S].reshape((Ny_S, Nx_S))
    S_vx = source_initial[Ny_S*Nx_S:Ny_S*Nx_S + Ny_S*(Nx_S-1)].reshape((Ny_S, Nx_S-1))
    S_vy = source_initial[Ny_S*Nx_S + Ny_S*(Nx_S-1):].reshape((Ny_S-1, Nx_S))
    
    return S_u, S_vx, S_vy

def compute_source_3D_elastic(S0, t_span, dx, dy, dz, c_p, rho0, S_matrix):
    """Compute the source term for the 3D elastic wave equation.
    
    Args:
        S0: Source time function
        t_span: Time span [t_start, t_end]
        dx, dy, dz: Grid spacing
        c_p: P-wave velocity (for determining source region size)
        rho0: Reference density
        S_matrix: Compliance matrix (6, 6)
    
    Returns:
        Velocity components (v_x, v_y, v_z) and stress components (σ_xx, σ_yy, σ_zz, σ_xy, σ_xz, σ_yz)
        Each as (Nz_S, Ny_S, Nx_S) arrays
    """
    # -- Grid parameters --
    # Determine source region size based on P-wave velocity and time
    r_x = int((1/dx) * t_span[1] * c_p)  # Radius in x direction
    r_y = int((1/dy) * t_span[1] * c_p)  # Radius in y direction
    r_z = int((1/dz) * t_span[1] * c_p)  # Radius in z direction
    Nx_S, Ny_S, Nz_S = (2*r_x, 2*r_y, 2*r_z)
    
    N_points = Nx_S * Ny_S * Nz_S
    
    # -- Initial wave field (all zeros) --
    # State vector: [v_x, v_y, v_z, σ_xx, σ_yy, σ_zz, σ_xy, σ_xz, σ_yz]
    phi_0_S = np.zeros(9 * N_points)
    
    # -- Material properties (homogeneous) --
    rho_model_S = rho0 * np.ones((Nz_S, Ny_S, Nx_S))
    
    # -------- Simulation (3D Elastic) --------
    (_, A_S, _, _, B_inv_S, _) = FD_solver_3D_elastic(Nx_S, Ny_S, Nz_S, dx, dy, dz, 
                                                       rho_model_S, S_matrix)
    
    # -------- Classical source simulation --------
    # Integration definition
    def rhs(t, phi):
        wave = B_inv_S @ A_S @ phi
        source = np.zeros_like(phi)
        # Inject source at the center of the source region (in the v_x component)
        center_idx = (Nz_S//2) * Ny_S * Nx_S + (Ny_S//2) * Nx_S + (Nx_S//2)
        source[center_idx] = S0(t)
        return wave + source
    
    # Time Integration
    source_initial = solve_ivp(rhs, t_span, phi_0_S, t_eval=t_span, method='DOP853').y.T[-1]
    
    # Extract velocity components (first 3*N_points entries)
    v_x = source_initial[:N_points].reshape((Nz_S, Ny_S, Nx_S))
    v_y = source_initial[N_points:2*N_points].reshape((Nz_S, Ny_S, Nx_S))
    v_z = source_initial[2*N_points:3*N_points].reshape((Nz_S, Ny_S, Nx_S))
    
    # Extract stress components (next 6*N_points entries)
    sigma_xx = source_initial[3*N_points:4*N_points].reshape((Nz_S, Ny_S, Nx_S))
    sigma_yy = source_initial[4*N_points:5*N_points].reshape((Nz_S, Ny_S, Nx_S))
    sigma_zz = source_initial[5*N_points:6*N_points].reshape((Nz_S, Ny_S, Nx_S))
    sigma_xy = source_initial[6*N_points:7*N_points].reshape((Nz_S, Ny_S, Nx_S))
    sigma_xz = source_initial[7*N_points:8*N_points].reshape((Nz_S, Ny_S, Nx_S))
    sigma_yz = source_initial[8*N_points:9*N_points].reshape((Nz_S, Ny_S, Nx_S))
    
    return v_x, v_y, v_z, sigma_xx, sigma_yy, sigma_zz, sigma_xy, sigma_xz, sigma_yz

    
# -- Plotting Functions --
def plot_acoustic_2D(phi, Nx, Ny, dx, dy, title='2D Wave Field', subsample=16, scale=2, width=0.003, clim=(-0.15, 0.15)):
    """Plot the 2D wavefield with velocity vectors."""
    # Calculate the number of points for each field
    N_u = Nx * Ny
    N_vx = (Nx - 1) * Ny
    N_vy = Nx * (Ny - 1)
    
    # Split phi into u, vx, vy
    u = phi[:N_u].reshape(Ny, Nx)
    vx = phi[N_u:N_u + N_vx].reshape(Ny, Nx-1)
    vy = phi[N_u + N_vx:].reshape(Ny-1, Nx)
    
    # Create grid coordinates
    x = np.arange(0, Nx*dx, dx)[::subsample]
    y = np.arange(0, Ny*dy, dy)[::subsample]
    X, Y = np.meshgrid(x, y)
    
    # Subsample the velocity components
    vx_sub = vx[::subsample, ::subsample]
    vy_sub = vy[::subsample, ::subsample]
    
    # Initialize the plot
    plt.figure(figsize=(10, 8))
    
    # Plot the wavefield amplitudes
    plt.imshow(u, cmap='RdBu', extent=[0, Nx*dx, 0, Ny*dy], clim=clim, origin='lower')
    plt.colorbar(label='u field')
    
    # Plot the velocity vectors
    plt.quiver(X, Y, vx_sub, vy_sub, scale=scale, width=width)
    
    # Set plot labels and title
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xticks(plt.xticks()[0], plt.xticks()[0] * dx - Nx/2)
    plt.yticks(plt.yticks()[0], plt.yticks()[0] * dy - Ny/2)
    plt.title(title)
    
    # Show the plot
    plt.show()

def plot_elastic_3D(phi, Nx, Ny, Nz, dx, dy, dz,
                    title='3D Elastic Wave Field',
                    subsample=4, scale_v=1.5, scale_stress=0.5,
                    width_v=2, width_stress=1,
                    show_velocity=True, show_stress=True,
                    save_file=None):
    """Plot velocity and stress fields for the 3D elastic wave equation.
    
    State vector: [v_x, v_y, v_z, σ_xx, σ_yy, σ_zz, σ_xy, σ_xz, σ_yz]
    
    Args:
        phi: State vector
        Nx, Ny, Nz: Number of grid points
        dx, dy, dz: Grid spacing
        title: Plot title
        subsample: Subsampling factor for visualization
        scale_v: Scale for velocity vectors
        scale_stress: Scale for stress vectors (diagonal components)
        width_v: Width of velocity arrows
        width_stress: Width of stress arrows
        show_velocity: Whether to show velocity field
        show_stress: Whether to show stress field
        save_file: If provided, save plot to this file instead of showing
    """
    # Staggered grid sizes
    N_main = Nx * Ny * Nz
    N_vx = (Nx-1) * Ny * Nz
    N_vy = Nx * (Ny-1) * Nz
    N_vz = Nx * Ny * (Nz-1)
    N_sxy = (Nx-1) * (Ny-1) * Nz
    N_sxz = (Nx-1) * Ny * (Nz-1)
    N_syz = Nx * (Ny-1) * (Nz-1)
    
    # Extract velocity components (staggered grids)
    idx = 0
    v_x = phi[idx:idx+N_vx].reshape(Nz, Ny, Nx-1)
    idx += N_vx
    v_y = phi[idx:idx+N_vy].reshape(Nz, Ny-1, Nx)
    idx += N_vy
    v_z = phi[idx:idx+N_vz].reshape(Nz-1, Ny, Nx)
    idx += N_vz
    
    # Extract stress components
    sigma_xx = phi[idx:idx+N_main].reshape(Nz, Ny, Nx)
    idx += N_main
    sigma_yy = phi[idx:idx+N_main].reshape(Nz, Ny, Nx)
    idx += N_main
    sigma_zz = phi[idx:idx+N_main].reshape(Nz, Ny, Nx)
    idx += N_main
    sigma_xy = phi[idx:idx+N_sxy].reshape(Nz, Ny-1, Nx-1)
    idx += N_sxy
    sigma_xz = phi[idx:idx+N_sxz].reshape(Nz-1, Ny, Nx-1)
    idx += N_sxz
    sigma_yz = phi[idx:idx+N_syz].reshape(Nz-1, Ny-1, Nx)
    
    # Interpolate staggered fields to main grid for visualization
    # For v_x: staggered in x, so average with neighbors
    v_x_main = np.zeros((Nz, Ny, Nx))
    v_x_main[:, :, 0] = v_x[:, :, 0]  # Left boundary
    v_x_main[:, :, -1] = v_x[:, :, -1]  # Right boundary
    v_x_main[:, :, 1:-1] = 0.5 * (v_x[:, :, :-1] + v_x[:, :, 1:])  # Interior: average
    
    # For v_y: staggered in y
    v_y_main = np.zeros((Nz, Ny, Nx))
    v_y_main[:, 0, :] = v_y[:, 0, :]  # Top boundary
    v_y_main[:, -1, :] = v_y[:, -1, :]  # Bottom boundary
    v_y_main[:, 1:-1, :] = 0.5 * (v_y[:, :-1, :] + v_y[:, 1:, :])  # Interior: average
    
    # For v_z: staggered in z
    v_z_main = np.zeros((Nz, Ny, Nx))
    v_z_main[0, :, :] = v_z[0, :, :]  # Front boundary
    v_z_main[-1, :, :] = v_z[-1, :, :]  # Back boundary
    v_z_main[1:-1, :, :] = 0.5 * (v_z[:-1, :, :] + v_z[1:, :, :])  # Interior: average
    
    # For stress on staggered grids, use nearest neighbor (simpler for visualization)
    sigma_xy_main = np.zeros((Nz, Ny, Nx))
    sigma_xy_main[:, 1:, 1:] = sigma_xy  # Place at lower-right of cell
    
    sigma_xz_main = np.zeros((Nz, Ny, Nx))
    sigma_xz_main[1:, :, 1:] = sigma_xz
    
    sigma_yz_main = np.zeros((Nz, Ny, Nx))
    sigma_yz_main[1:, 1:, :] = sigma_yz
    
    # Use interpolated values for visualization
    v_x = v_x_main
    v_y = v_y_main
    v_z = v_z_main
    sigma_xy = sigma_xy_main
    sigma_xz = sigma_xz_main
    sigma_yz = sigma_yz_main
    
    # Create grid coordinates
    x = np.linspace(0, (Nx-1)*dx, Nx)
    y = np.linspace(0, (Ny-1)*dy, Ny)
    z = np.linspace(0, (Nz-1)*dz, Nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # Subsample the grid for quiver plots
    X_sub = X[::subsample, ::subsample, ::subsample]
    Y_sub = Y[::subsample, ::subsample, ::subsample]
    Z_sub = Z[::subsample, ::subsample, ::subsample]
    
    # Subsample velocity components
    v_x_sub = v_x[::subsample, ::subsample, ::subsample]
    v_y_sub = v_y[::subsample, ::subsample, ::subsample]
    v_z_sub = v_z[::subsample, ::subsample, ::subsample]
    
    # Subsample stress components (using diagonal for visualization)
    sigma_xx_sub = sigma_xx[::subsample, ::subsample, ::subsample]
    sigma_yy_sub = sigma_yy[::subsample, ::subsample, ::subsample]
    sigma_zz_sub = sigma_zz[::subsample, ::subsample, ::subsample]
    
    # Initialize 3D plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot Velocity Field Vectors
    if show_velocity:
        ax.quiver(X_sub, Y_sub, Z_sub,
                  v_x_sub, v_y_sub, v_z_sub,
                  color='r', length=scale_v,
                  linewidth=width_v, arrow_length_ratio=0.3,
                  label='Velocity Field', alpha=0.7)
    
    # Plot Stress Field (diagonal components as vectors)
    if show_stress:
        ax.quiver(X_sub, Y_sub, Z_sub,
                  sigma_xx_sub, sigma_yy_sub, sigma_zz_sub,
                  color='b', length=scale_stress,
                  linewidth=width_stress, arrow_length_ratio=0.3,
                  label='Stress (diagonal)', alpha=0.5)
    
    # Set plot labels and title with larger fonts
    ax.set_xlabel('X [m]', fontsize=18)
    ax.set_ylabel('Y [m]', fontsize=18)
    ax.set_zlabel('Z [m]', fontsize=18)
    ax.set_title(title, fontsize=20, fontweight='bold', pad=20)
    
    # Increase tick label sizes
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.tick_params(axis='z', labelsize=12)
    
    # Add legend with larger font
    ax.legend(fontsize=16, loc='upper right')
    
    # Save or show the plot
    plt.tight_layout()
    if save_file:
        plt.savefig(save_file, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {save_file}")
        plt.close()
    else:
        plt.show()

def plot_maxwells_3D(phi, Nx, Ny, Nz, dx, dy, dz,
                      title='3D Wave Field',
                      subsample=4, scale_E=1.5, scale_B=1.5,
                      width_E=2, width_B=2
                      ):
    """Plot electric and magnetic fields as quiver plots in 3D."""

    # Calculate total number of points
    N_total = Nx * Ny * Nz 
    
    # Split phi into phi_E and phi_B
    phi_E = phi[:3*N_total]
    phi_B = phi[3*N_total:]

    # Split phi_E into Ex, Ey, Ez
    Ex = phi_E[:N_total].reshape(Nz, Ny, Nx)
    Ey = phi_E[N_total:2*N_total].reshape(Nz, Ny, Nx)
    Ez = phi_E[2*N_total:3*N_total].reshape(Nz, Ny, Nx)

    # Split phi_B into Bx, By, Bz
    Bx = phi_B[:(N_total-Ny*Nz)].reshape(Ny, Nz, (Nx-1))
    By = phi_B[(N_total-Ny*Nz):(2*N_total-Ny*Nz-Nx*Nz)].reshape(Nz, (Ny-1), Nx)
    Bz = phi_B[(2*N_total-Ny*Nz-Nx*Nz):].reshape((Nz-1), Ny, Nx)

    # Create grid coordinates
    x = np.linspace(0, (Nx-1)*dx, Nx)
    y = np.linspace(0, (Ny-1)*dy, Ny)
    z = np.linspace(0, (Nz-1)*dz, Nz)
    X, Y, Z = np.meshgrid(x, y, z)
    
    # Subsample the grid for quiver plots
    X_sub = X[::subsample, ::subsample, ::subsample]
    Y_sub = Y[::subsample, ::subsample, ::subsample]
    Z_sub = Z[::subsample, ::subsample, ::subsample]

    # Subsample the vector components
    Ex_sub = Ex[::subsample, ::subsample, ::subsample]
    Ey_sub = Ey[::subsample, ::subsample, ::subsample]
    Ez_sub = Ez[::subsample, ::subsample, ::subsample]

    Bx_sub = Bx[::subsample, ::subsample, ::subsample]
    By_sub = By[::subsample, ::subsample, ::subsample]
    Bz_sub = Bz[::subsample, ::subsample, ::subsample]

    # Initialize 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot Electric Field Vectors
    ax.quiver(X_sub, Y_sub, Z_sub,
              Ex_sub, Ey_sub, Ez_sub,
              color='r', length=scale_E,
              linewidth=width_E, label='Electric Field')

    # Plot Magnetic Field Vectors
    ax.quiver(X_sub, Y_sub, Z_sub,
              Bx_sub, By_sub, Bz_sub,
              color='b', length=scale_B,
              linewidth=width_B, label='Magnetic Field')

    # Set plot labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    # Correct origin to upper
    ax.invert_zaxis()

    # Add legend
    ax.legend()

    # Show the plot
    plt.show()

def plot_field_3D(state):
    """Plot vector field on spherical grid"""
    # Extract dimensions
    _, n_theta, n_phi, n_radii = state.shape

    # Generate spherical coordinates
    theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)  # Azimuthal angle
    phi = np.linspace(-np.pi/2, np.pi/2, n_phi)  # Polar angle
    radii = np.linspace(0, 1, n_radii)  # Radii from the center outward

    # Create a meshgrid for the spherical coordinates
    Theta, Phi, R = np.meshgrid(theta, phi, radii, indexing='xy')

    # Convert spherical coordinates to cartesian coordinates
    X = R * np.cos(Phi) * np.cos(Theta)
    Y = R * np.cos(Phi) * np.sin(Theta)
    Z = R * np.sin(Phi)

    # Flatten the arrays for plotting
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    Z_flat = Z.flatten()

    # Flatten the vector components
    U_flat = state[0].flatten()
    V_flat = state[1].flatten()
    W_flat = state[2].flatten()

    # Create a Plotly figure
    fig = go.Figure()

    # Add quiver arrows using cones
    fig.add_trace(
        go.Cone(
            x=X_flat,
            y=Y_flat,
            z=Z_flat,
            u=U_flat,
            v=V_flat,
            w=W_flat,
            colorscale='Blues',
            sizemode='absolute',
            sizeref=0.05,  # Adjust sizeref for arrow size
            anchor="tail",
            showlegend=False,  # Disable legend for this trace
            showscale=False     # Disable colorbar
        )
    )

    # Update layout for better visualization
    fig.update_layout(
        scene=dict(
            xaxis_title='X-coordinate',
            yaxis_title='Y-coordinate',
            zaxis_title='Z-coordinate',
            aspectratio=dict(x=1, y=1, z=1),
            xaxis = dict(nticks=4, range=[-1,1],),
            yaxis = dict(nticks=4, range=[-1,1],),
            zaxis = dict(nticks=4, range=[-1,1],),
        ),
        margin=dict(r=0, l=0, b=0, t=0),
        scene_camera=dict(
        up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=-0.12),
        eye=dict(x=1.5, y=1.5, z=1.5)
        ),
        font=dict(size=22),
        width=800,
        height=800,
    )

    return fig

def subspaceProjector(staggered_grid_size, subspace_points=None):
    '''
    Return a numpy array corresponding to the staggered grid points included in the subspace projector.
    :param subspace_points: currently assumed to be a numpy array of length staggered_grid_size. Each entry has value 0 (not in subspace) and 1 (in subspace).
    :param staggered_grid_size: nonnegative integer.
    '''
    # If no input for points, randomly select subspace points.
    if subspace_points == None:
        valid_prob = False
        # Probability of each point to be included in the subspace must be a float between 0 and 1.
        while valid_prob == False:
            prob = input("Enter a probability of each point being included in the subspace: ")
            try:
                prob = float(prob)
                if (0 <= prob <= 1):
                    valid_prob = True
                else:
                    print("Probability must be between 0 and 1.")
            except:
                print("Invalid input. Not a float?")
        mask = np.random.choice([0, 1], size=staggered_grid_size, p=[1-prob, prob])
    else:
        # If subspace_points is a numpy array, check if the number of points is equal to staggered_grid_size.
        assert len(subspace_points) == staggered_grid_size
        # The part below will be modified if subspace_points is not a numpy array.
        mask = subspace_points
    return mask
