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
    # Get the exact component lengths from your utility
    (N_main, N_vx, N_vy, N_vz, N_sxy, N_sxz, N_syz, 
     N_vel, N_stress, psi_len) = get_grid_size(Nx, Ny, Nz)

    f0 = 1.0
    x0, y0, z0 = 0, 0, 0.9*zmin #this assumes zmin is negative
    
    # Create the staggered coordinates for v_x: (Nz, Ny, Nx-1)
    x_vx = np.linspace(xmin + dx/2, xmax - dx/2, Nx - 1)
    y_vx = np.linspace(ymin, ymax, Ny)
    z_vx = np.linspace(zmin, zmax, Nz)
    
    # Use 'ij' indexing with the order (Z, Y, X) to match (Nz, Ny, Nx-1)
    Z_vx, Y_vx, X_vx = np.meshgrid(z_vx, y_vx, x_vx, indexing='ij')

    # Compute Ricker on the staggered v_x grid
    ricker_vx = Ricker(f0, X_vx, Y_vx, Z_vx, x0, y0, z0)
    v0x_flat = np.round(ricker_vx, 20).flatten()

    # Initialize the full state vector
    phi_0 = np.zeros(psi_len)

    # Insert v_x at the very beginning
    phi_0[0:N_vx] = v0x_flat
    
    # All other components (v_y, v_z, and all stresses) remain 0
    return phi_0


def ricker_IC_vz(Nx, Ny, Nz, dx, dy, dz, xmin, xmax, ymin, ymax, zmin, zmax):
    # Get the exact component lengths from your utility
    (N_main, N_vx, N_vy, N_vz, N_sxy, N_sxz, N_syz, 
     N_vel, N_stress, psi_len) = get_grid_size(Nx, Ny, Nz)

    f0 = 1.0
    x0, y0, z0 = 0, 0, 0.5*zmin # this assumes zmin is negative
    
    # Create the staggered coordinates for v_z: (Nz-1, Ny, Nx)
    # The half-grid shift is now applied to z_vz
    x_vz = np.linspace(xmin, xmax, Nx)
    y_vz = np.linspace(ymin, ymax, Ny)
    z_vz = np.linspace(zmin + dz/2, zmax - dz/2, Nz - 1)
    
    # Use 'ij' indexing with the order (Z, Y, X) to match (Nz-1, Ny, Nx)
    Z_vz, Y_vz, X_vz = np.meshgrid(z_vz, y_vz, x_vz, indexing='ij')

    # Compute Ricker on the staggered v_z grid, negative since we want velocities pointing up
    ricker_vz = - Ricker(f0, X_vz, Y_vz, Z_vz, x0, y0, z0)
    v0z_flat = np.round(ricker_vz, 20).flatten()

    # Initialize the full state vector
    phi_0 = np.zeros(psi_len)

    # State vector order: [v_x, v_y, v_z, σ_xx, ...]
    # We must skip over v_x and v_y to insert v_z
    start_idx = N_vx + N_vy
    end_idx = start_idx + N_vz

    # Insert v_z into the correct slot
    phi_0[start_idx:end_idx] = v0z_flat
    
    # All other components remain 0
    return phi_0
