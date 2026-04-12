# -------- Imports --------
import numpy as np
import scipy.sparse as sp
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from _utility import *
from _fractures import *

#Returns the initial wave field for a Gaussian located at some specified region

def gaussian_IC(Nx,Ny,Nz,dx,dy,dz,xmin,xmax,ymin,ymax,zmin,zmax):
    
    N_main,N_vx,N_vy,N_vz,N_sxy,N_sxz,N_syz,N_vel,N_stress,psi_len = get_grid_size(Nx,Ny,Nz)

    #phi_0 = np.zeros(psi_len, dtype=complex)
    phi_0 = np.zeros(psi_len)

    x = np.linspace(xmin,xmax,Nx)
    y = np.linspace(ymin,ymax,Ny)
    z = np.linspace(zmin,zmax,Nz)
    # Build grid directly in (Nz, Ny, Nx) order
    Z, Y, X = np.meshgrid(z, y, x, indexing='ij')

    sigma = 0.25 * min(xmax - xmin, ymax - ymin, zmax - zmin)
    amplitude = -.5

    # Gaussian centered at (0,0,-zmin)
    gaussian = amplitude * np.exp(-(X**2 + Y**2 + (Z + zmin)**2) / (2 * sigma**2))

    # Flatten directly (already correct memory layout)
    g_flat = gaussian.flatten()

    # Insert stresses
    idx = N_vel
    phi_0[idx:idx + N_main] = g_flat   # σ_xx
    idx += N_main
    phi_0[idx:idx + N_main] = g_flat   # σ_yy
    idx += N_main
    phi_0[idx:idx + N_main] = g_flat   # σ_zz `

    return phi_0

def ricker_IC(Nx,Ny,Nz,dx,dy,dz,xmin,xmax,ymin,ymax,zmin,zmax):

    N_main,N_vx,N_vy,N_vz,N_sxy,N_sxz,N_syz,N_vel,N_stress,psi_len = get_grid_size(Nx,Ny,Nz)


    f0 = 2.0            # Central frequency of the Ricker wavelet
    x0, y0, z0 = 0, 0, 0   # Wavelet center
    x_vx = np.linspace(xmin+dx/2,xmax-dx/2,Nx-1)
    y_vx = np.linspace(ymin,ymax,Ny)
    z_vx = np.linspace(zmin,zmax,Nz)

    X_vx, Y_vx, Z_vx = np.meshgrid(x_vx, y_vx, z_vx, indexing='ij')

    #print(X_vx)

    # Initialize velocity components (all zeros initially)
    v0x = np.zeros((Nx-1, Ny, Nz))
    v0y = np.zeros((Nx, Ny-1, Nz))
    v0z = np.zeros((Nx, Ny, Nz-1))

    # Initialize stress components (all zeros initially)
    sigma_xx = np.zeros((Nx, Ny, Nz))
    sigma_yy = np.zeros((Nx, Ny, Nz))
    sigma_zz = np.zeros((Nx, Ny, Nz))
    sigma_xy = np.zeros((Nx-1, Ny-1, Nz))
    sigma_xz = np.zeros((Nx-1, Ny, Nz-1))
    sigma_yz = np.zeros((Nx, Ny-1, Nz-1))

    # Add a Ricker wavelet source to v_x component similarly to the acoustic case
    # Ricker function takes (f, x, y, z, x0, y0, z0) and returns a scalar or array
    ricker_vx = Ricker(f0, X_vx, Y_vx, Z_vx, x0, y0, z0)
    #print(ricker_vx)
    v0x = np.round(ricker_vx, 20)
    #print(v0x.flatten())
    # Stack the initial conditions in the correct order:
    # [v_x, v_y, v_z, σ_xx, σ_yy, σ_zz, σ_xy, σ_xz, σ_yz]
    phi_0 = np.concatenate([
        v0x.flatten(),      # v_x: (Nx-1)*Ny*Nz
        v0y.flatten(),      # v_y: Nx*(Ny-1)*Nz
        v0z.flatten(),      # v_z: Nx*Ny*(Nz-1)
        sigma_xx.flatten(), # σ_xx: Nx*Ny*Nz
        sigma_yy.flatten(), # σ_yy: Nx*Ny*Nz
        sigma_zz.flatten(), # σ_zz: Nx*Ny*Nz
        sigma_xy.flatten(), # σ_xy: (Nx-1)*(Ny-1)*Nz
        sigma_xz.flatten(), # σ_xz: (Nx-1)*Ny*(Nz-1)
        sigma_yz.flatten()  # σ_yz: Nx*(Ny-1)*(Nz-1)
    ])
    return phi_0

