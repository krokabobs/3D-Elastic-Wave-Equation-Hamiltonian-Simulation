import numpy as np
import scipy.sparse as sp
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from _utility import *
from _fractures import *
from _initial_conditions import *


'''
functions defined in this file:

plot_time_slider_plotly
get_3d_velocities (helper function)
plot_3d_velocity_plotly
plot_3d_velocity_plotly_log
'''



def plot_time_slider_plotly(k_fixed, sol, B_sqrt, xmin, ymin, zmin, xmax, ymax, zmax, Nx, Ny, Nz, dx, dy, dz):
    # Safety check for k
    k_fixed = max(0, min(k_fixed, Nz - 1))
    
    # Efficiently convert the entire solution matrix to the physical basis in one step
    psi_matrix = B_sqrt @ sol.y
    n_steps = psi_matrix.shape[1]
    
    # Calculate the physical z-coordinate for the title
    z_phys = zmin + k_fixed * dz

    # Track global max values for consistent colorbars across time steps
    global_max = {
        'v_x': 0, 'v_y': 0, 'v_z': 0, 'v_mag': 0,
        'sigma_xx': 0, 'sigma_yy': 0, 'sigma_zz': 0,
        'sigma_xy': 0, 'sigma_xz': 0, 'sigma_yz': 0
    }

    # Grid sizes
    N_vx = (Nx - 1) * Ny * Nz
    N_vy = Nx * (Ny - 1) * Nz
    N_vz = Nx * Ny * (Nz - 1)
    N_main = Nx * Ny * Nz
    N_sxy = (Nx - 1) * (Ny - 1) * Nz
    N_sxz = (Nx - 1) * Ny * (Nz - 1)
    N_syz = Nx * (Ny - 1) * (Nz - 1)

    all_frames_data = []

    # 1. Pre-process and extract all data
    for t in range(n_steps):
        phi = psi_matrix[:, t]
        idx = 0
        
        # --- Unpacking ---
        vx_raw = phi[idx:idx+N_vx].reshape(Nz, Ny, Nx-1); idx += N_vx
        vy_raw = phi[idx:idx+N_vy].reshape(Nz, Ny-1, Nx); idx += N_vy
        vz_raw = phi[idx:idx+N_vz].reshape(Nz-1, Ny, Nx); idx += N_vz
        
        sxx = phi[idx:idx+N_main].reshape(Nz, Ny, Nx); idx += N_main
        syy = phi[idx:idx+N_main].reshape(Nz, Ny, Nx); idx += N_main
        szz = phi[idx:idx+N_main].reshape(Nz, Ny, Nx); idx += N_main
        
        sxy_raw = phi[idx:idx+N_sxy].reshape(Nz, Ny-1, Nx-1); idx += N_sxy
        sxz_raw = phi[idx:idx+N_sxz].reshape(Nz-1, Ny, Nx-1); idx += N_sxz
        syz_raw = phi[idx:idx+N_syz].reshape(Nz-1, Ny-1, Nx)

        # --- Interpolation & Slicing ---
        vx_slice = np.zeros((Ny, Nx))
        vx_slice[:, 0] = vx_raw[k_fixed, :, 0]
        vx_slice[:, -1] = vx_raw[k_fixed, :, -1]
        vx_slice[:, 1:-1] = 0.5 * (vx_raw[k_fixed, :, :-1] + vx_raw[k_fixed, :, 1:])
        
        vy_slice = np.zeros((Ny, Nx))
        vy_slice[0, :] = vy_raw[k_fixed, 0, :]
        vy_slice[-1, :] = vy_raw[k_fixed, -1, :]
        vy_slice[1:-1, :] = 0.5 * (vy_raw[k_fixed, :-1, :] + vy_raw[k_fixed, 1:, :])
        
        if 0 < k_fixed < Nz - 1:
             vz_slice = 0.5 * (vz_raw[k_fixed-1, :, :] + vz_raw[k_fixed, :, :])
        elif k_fixed == 0:
             vz_slice = vz_raw[0, :, :]
        else:
             vz_slice = vz_raw[-1, :, :]
             
        v_mag = np.sqrt(vx_slice**2 + vy_slice**2 + vz_slice**2)
        
        sxx_slice = sxx[k_fixed, :, :]
        syy_slice = syy[k_fixed, :, :]
        szz_slice = szz[k_fixed, :, :]
        
        sxy_slice = np.zeros((Ny, Nx))
        sxy_slice[1:, 1:] = sxy_raw[k_fixed, :, :]
        
        sxz_slice = np.zeros((Ny, Nx))
        if k_fixed < Nz-1: sxz_slice[:, 1:] = sxz_raw[k_fixed, :, :]
            
        syz_slice = np.zeros((Ny, Nx))
        if k_fixed < Nz-1: syz_slice[1:, :] = syz_raw[k_fixed, :, :]

        frame_data = {
            'v_x': vx_slice, 'v_y': vy_slice, 'v_z': vz_slice, 'v_mag': v_mag,
            'sigma_xx': sxx_slice, 'sigma_yy': syy_slice, 'sigma_zz': szz_slice,
            'sigma_xy': sxy_slice, 'sigma_xz': sxz_slice, 'sigma_yz': syz_slice
        }
        all_frames_data.append(frame_data)
        
        # Update global max
        for key in global_max:
            current_max = np.max(np.abs(frame_data[key]))
            if current_max > global_max[key]:
                global_max[key] = current_max

    # 2. Setup Plotly Figure
    layout_keys = [
        ('v_x', 'Velocity X', 'RdBu'), ('v_y', 'Velocity Y', 'RdBu'), ('v_z', 'Velocity Z', 'RdBu'), 
        ('v_mag', 'Velocity Mag', 'Viridis'), ('sigma_xx', 'Normal XX', 'RdBu'),
        ('sigma_yy', 'Normal YY', 'RdBu'), ('sigma_zz', 'Normal ZZ', 'RdBu'),
        ('sigma_xy', 'Shear XY', 'RdBu'), ('sigma_xz', 'Shear XZ', 'RdBu'), ('sigma_yz', 'Shear YZ', 'RdBu')
    ]
    
    titles = [item[1] for item in layout_keys]
    
    cols, rows = 5, 2
    h_space, v_space = 0.07, 0.15
    col_width = (1.0 - (cols - 1) * h_space) / cols
    row_height = (1.0 - (rows - 1) * v_space) / rows

    fig = make_subplots(rows=rows, cols=cols, subplot_titles=titles, 
                        horizontal_spacing=h_space, vertical_spacing=v_space)

    x_vals = np.linspace(xmin, xmax, Nx)
    y_vals = np.linspace(ymin, ymax, Ny)

    # 3. Add base traces (t = 0)
    for idx, (key, title, cmap) in enumerate(layout_keys):
        row = (idx // cols) + 1
        col = (idx % cols) + 1
        
        vmax = global_max[key] if global_max[key] != 0 else 1e-10
        zmin_val = -vmax if cmap == 'RdBu' else 0

        col_idx = col - 1
        row_idx = rows - row 
        
        cb_x = col_idx * (col_width + h_space) + col_width + 0.005
        cb_y = row_idx * (row_height + v_space) + (row_height / 2)

        trace = go.Heatmap(
            z=all_frames_data[0][key], x=x_vals, y=y_vals,
            colorscale=cmap, zmin=zmin_val, zmax=vmax,
            colorbar=dict(thickness=10, len=0.45, x=cb_x, y=cb_y, yanchor='middle', tickfont=dict(size=10)),
            showscale=True
        )
        fig.add_trace(trace, row=row, col=col)

    # 4. Create Animation Frames
    frames = []
    for t in range(n_steps):
        frame_traces = []
        for key, _, _ in layout_keys:
            frame_traces.append(go.Heatmap(z=all_frames_data[t][key]))
        frames.append(go.Frame(data=frame_traces, name=str(t)))

    fig.frames = frames

    # 5. Add UI Controls
    # We now use sol.t to display actual time (in scientific notation) on the slider!
    sliders = [dict(
        steps=[dict(
            method='animate',
            args=[[str(t)], dict(mode='immediate', frame=dict(duration=200, redraw=True), transition=dict(duration=0))],
            label=f't = {sol.t[t]:.2e}s'
        ) for t in range(n_steps)],
        active=0,
        x=0.08, y=-0.15, xanchor='left', yanchor='top'
    )]

    fig.update_layout(
        title=f'Time Evolution at Z-Slice {z_phys:.3f} (k={k_fixed})',
        width=1100, 
        height=550,
        margin=dict(l=20, r=20, t=60, b=20),
        sliders=sliders,
        updatemenus=[dict(
            type='buttons', showactive=False,
            y=-0.15, x=0, xanchor='left', yanchor='top',
            buttons=[
                dict(label='Play', method='animate', args=[None, dict(frame=dict(duration=200, redraw=True), transition=dict(duration=0), fromcurrent=True, mode='immediate')]),
                dict(label='Pause', method='animate', args=[[None], dict(frame=dict(duration=0, redraw=False), mode='immediate', transition=dict(duration=0))])
            ]
        )]
    )

    return fig


def get_3d_velocities(phi, Nx, Ny, Nz):
    """Extracts staggered velocity components and interpolates them to the main grid nodes."""
    N_vx = (Nx - 1) * Ny * Nz
    N_vy = Nx * (Ny - 1) * Nz
    N_vz = Nx * Ny * (Nz - 1)
    
    idx = 0
    # Reshape raw data
    vx_raw = phi[idx:idx+N_vx].reshape(Nz, Ny, Nx-1); idx += N_vx
    vy_raw = phi[idx:idx+N_vy].reshape(Nz, Ny-1, Nx); idx += N_vy
    vz_raw = phi[idx:idx+N_vz].reshape(Nz-1, Ny, Nx); idx += N_vz
    
    # Interpolate v_x
    vx_full = np.zeros((Nz, Ny, Nx))
    vx_full[:, :, 0] = vx_raw[:, :, 0]
    vx_full[:, :, -1] = vx_raw[:, :, -1]
    vx_full[:, :, 1:-1] = 0.5 * (vx_raw[:, :, :-1] + vx_raw[:, :, 1:])
    
    # Interpolate v_y
    vy_full = np.zeros((Nz, Ny, Nx))
    vy_full[:, 0, :] = vy_raw[:, 0, :]
    vy_full[:, -1, :] = vy_raw[:, -1, :]
    vy_full[:, 1:-1, :] = 0.5 * (vy_raw[:, :-1, :] + vy_raw[:, 1:, :])
    
    # Interpolate v_z
    vz_full = np.zeros((Nz, Ny, Nx))
    if Nz > 1:
        vz_full[0, :, :] = vz_raw[0, :, :]
        vz_full[-1, :, :] = vz_raw[-1, :, :]
        vz_full[1:-1, :, :] = 0.5 * (vz_raw[:-1, :, :] + vz_raw[1:, :, :])
        
    return vx_full, vy_full, vz_full



def plot_3d_velocity_plotly(sol, B_sqrt, rho_model, xmin, ymin, zmin, xmax, ymax, zmax, Nx, Ny, Nz, dx, dy, dz):
    # Convert all states at once
    psi_matrix = B_sqrt @ sol.y
    n_steps = psi_matrix.shape[1]

    # Create the coordinate grid
    x_vals = np.linspace(xmin, xmax, Nx)
    y_vals = np.linspace(ymin, ymax, Ny)
    z_vals = np.linspace(zmin, zmax, Nz)
    
    Z_grid, Y_grid, X_grid = np.meshgrid(z_vals, y_vals, x_vals, indexing='ij')
    x_f = X_grid.flatten()
    y_f = Y_grid.flatten()
    z_f = Z_grid.flatten()

    # --- CONTINUOUS-LOOKING SCATTER TRACE ---
    z_idx, y_idx, x_idx = np.where(rho_model < 2700)
    frac_x = xmin + x_idx * dx
    frac_y = ymin + y_idx * dy
    frac_z = zmin + z_idx * dz
    
    fracture_trace = go.Scatter3d(
        x=frac_x, y=frac_y, z=frac_z,
        mode='markers',
        marker=dict(
            size=20,             # Large enough to overlap neighboring cells
            color='gray',        
            symbol='square',     # Squares pack together better into walls
            opacity=0.3,        # Low opacity makes overlapping markers look like a smooth solid
            line=dict(width=0)   # Remove marker borders so they blend seamlessly
        ),
        name='Fracture Geometry',
        showlegend=True,
        hoverinfo='skip'         
    )
    # -----------------------------------------

    # First pass: find global maximum magnitude for consistent color scaling across all time steps
    global_max_mag = 0
    for t in range(n_steps):
        phi = psi_matrix[:, t]
        vx, vy, vz = get_3d_velocities(phi, Nx, Ny, Nz)
        mag = np.sqrt(vx**2 + vy**2 + vz**2)
        global_max_mag = max(global_max_mag, np.max(mag))
        
    if global_max_mag == 0: 
        global_max_mag = 1e-10

    frames = []
    init_u, init_v, init_w = None, None, None

    # Second pass: Build animation frames with TRUE vector lengths
    for t in range(n_steps):
        phi = psi_matrix[:, t]
        vx, vy, vz = get_3d_velocities(phi, Nx, Ny, Nz)
        u = vx.flatten()
        v = vy.flatten()
        w = vz.flatten()

        # Save the first step for the initial plot rendering
        if t == 0:
            init_u, init_v, init_w = u, v, w

        frames.append(go.Frame(
            data=[go.Cone(
                x=x_f, y=y_f, z=z_f,
                u=u, v=v, w=w
            )],
            name=str(t)
        ))

    # Build the base figure cone trace
    cone_trace = go.Cone(
        x=x_f, y=y_f, z=z_f,
        u=init_u, v=init_v, w=init_w,
        colorscale='Viridis',
        cmin=0, cmax=global_max_mag,
        sizemode='scaled',
        sizeref=1, # Adjust this value (e.g., 0.5 or 2) if the cones are too big or too small
        colorbar=dict(title='Velocity Magnitude'),
        name='Velocity',
        showlegend=False # Hide the cones from the legend to keep the UI clean
    )
    
    # Pass both traces to the main figure
    fig = go.Figure(data=[cone_trace, fracture_trace], frames=frames)

    # Configure the slider with actual timestamps
    sliders = [dict(
        steps=[dict(
            method='animate',
            args=[[str(t)], dict(mode='immediate', frame=dict(duration=200, redraw=True), transition=dict(duration=0))],
            label=f'{sol.t[t]:.2e}s'
        ) for t in range(n_steps)],
        active=0,
        x=0.1, y=0, xanchor='left', yanchor='top'
    )]

    fig.update_layout(
        title='3D Velocity Vector Field (Plotly)',
        width=1000,   
        height=800,   
        margin=dict(l=10, r=10, b=10, t=50), 
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            xaxis=dict(range=[xmin, xmax]),
            yaxis=dict(range=[ymin, ymax]),
            zaxis=dict(range=[zmin, zmax]),
            aspectmode='data'
        ),
        sliders=sliders,
        showlegend=True, 
        legend=dict(
            orientation="h",   # Make the legend horizontal
            yanchor="top",
            y=0.95,            # Pushes it just below the title
            xanchor="left",
            x=0.05             # Puts it on the top-left side
        ),
        updatemenus=[dict(
            type='buttons',
            showactive=False,
            y=0, x=0, xanchor='left', yanchor='top',
            buttons=[
                dict(label='Play', method='animate', args=[None, dict(frame=dict(duration=200, redraw=True), transition=dict(duration=0), fromcurrent=True, mode='immediate')]),
                dict(label='Pause', method='animate', args=[[None], dict(frame=dict(duration=0, redraw=False), mode='immediate', transition=dict(duration=0))])
            ]
        )]
    )

    return fig



def plot_3d_velocity_plotly_log(sol, B_sqrt, rho_model, xmin, ymin, zmin, xmax, ymax, zmax, Nx, Ny, Nz, dx, dy, dz, log_boost=1000):
    # Convert all states at once
    psi_matrix = B_sqrt @ sol.y
    n_steps = psi_matrix.shape[1]

    # Create the coordinate grid for the cones
    x_vals = np.linspace(xmin, xmax, Nx)
    y_vals = np.linspace(ymin, ymax, Ny)
    z_vals = np.linspace(zmin, zmax, Nz)
    
    Z_grid, Y_grid, X_grid = np.meshgrid(z_vals, y_vals, x_vals, indexing='ij')
    x_f = X_grid.flatten()
    y_f = Y_grid.flatten()
    z_f = Z_grid.flatten()

    # --- CONTINUOUS-LOOKING SCATTER TRACE ---
    z_idx, y_idx, x_idx = np.where(rho_model < 2700)
    frac_x = xmin + x_idx * dx
    frac_y = ymin + y_idx * dy
    frac_z = zmin + z_idx * dz
    
    fracture_trace = go.Scatter3d(
        x=frac_x, y=frac_y, z=frac_z,
        mode='markers',
        marker=dict(
            size=20,             # Large enough to overlap neighboring cells
            color='gray',        
            symbol='square',     # Squares pack together better into walls
            opacity=0.3,        # Low opacity makes overlapping markers look like a smooth solid
            line=dict(width=0)   # Remove marker borders so they blend seamlessly
        ),
        name='Fracture Geometry',
        showlegend=True,
        hoverinfo='skip'         
    )
    # -----------------------------------------

    # 1. Find global maximum magnitude across all frames
    global_max_mag = 0
    for t in range(n_steps):
        phi = psi_matrix[:, t]
        vx, vy, vz = get_3d_velocities(phi, Nx, Ny, Nz)
        mag = np.sqrt(vx**2 + vy**2 + vz**2)
        global_max_mag = max(global_max_mag, np.max(mag))
        
    if global_max_mag == 0: 
        global_max_mag = 1e-10

    frames = []
    init_u, init_v, init_w = None, None, None

    # 2. Build animation frames with Log-Scaled vectors
    for t in range(n_steps):
        phi = psi_matrix[:, t]
        vx, vy, vz = get_3d_velocities(phi, Nx, Ny, Nz)
        u_raw, v_raw, w_raw = vx.flatten(), vy.flatten(), vz.flatten()
        mag = np.sqrt(u_raw**2 + v_raw**2 + w_raw**2)

        with np.errstate(divide='ignore', invalid='ignore'):
            safe_mag = np.where(mag == 0, 1e-12, mag)
            
            u_dir = u_raw / safe_mag
            v_dir = v_raw / safe_mag
            w_dir = w_raw / safe_mag

            log_mag = np.log10(1 + log_boost * (mag / global_max_mag))

            u_log = u_dir * log_mag
            v_log = v_dir * log_mag
            w_log = w_dir * log_mag
            
            u_log = np.nan_to_num(u_log)
            v_log = np.nan_to_num(v_log)
            w_log = np.nan_to_num(w_log)

        if t == 0:
            init_u, init_v, init_w = u_log, v_log, w_log

        # Notice we ONLY put the Cone trace in the frames. 
        # Plotly will automatically leave the static fracture trace alone.
        frames.append(go.Frame(
            data=[go.Cone(x=x_f, y=y_f, z=z_f, u=u_log, v=v_log, w=w_log)],
            name=str(t)
        ))

    # 3. Create Custom Colorbar Ticks
    max_log_val = np.log10(1 + log_boost)
    tick_vals_log = np.linspace(0, max_log_val, 6)
    tick_vals_true = global_max_mag * ((10**tick_vals_log - 1) / log_boost)
    tick_texts = [f"{val:.2e}" for val in tick_vals_true]

    # 4. Build the base figure (ADDING FRACTURE TRACE HERE)
    cone_trace = go.Cone(
        x=x_f, y=y_f, z=z_f,
        u=init_u, v=init_v, w=init_w,
        colorscale='Viridis',
        cmin=0, cmax=max_log_val,
        sizemode='scaled',
        sizeref=0.5, 
        colorbar=dict(
            title='True Velocity Magnitude',
            tickmode='array',
            tickvals=tick_vals_log,
            ticktext=tick_texts
        ),
        name='Velocity',
        showlegend=False # Hide the cones from the legend to keep the UI clean
    )
    
    # We pass both the moving cone_trace and the static fracture_trace to the base figure
    fig = go.Figure(data=[cone_trace, fracture_trace], frames=frames)

    # 5. Configure UI Layout with actual timestamps
    sliders = [dict(
        steps=[dict(
            method='animate',
            args=[[str(t)], dict(mode='immediate', frame=dict(duration=200, redraw=True), transition=dict(duration=0))],
            label=f'{sol.t[t]:.2e}s'
        ) for t in range(n_steps)],
        active=0,
        x=0.1, y=0, xanchor='left', yanchor='top'
    )]

    fig.update_layout(
        title='3D Velocity Vector Field (Logarithmic Scale)',
        width=1000, 
        height=800,
        margin=dict(l=10, r=10, b=10, t=50),
        scene=dict(
            xaxis_title='X', yaxis_title='Y', zaxis_title='Z',
            xaxis=dict(range=[xmin, xmax]),
            yaxis=dict(range=[ymin, ymax]),
            zaxis=dict(range=[zmin, zmax]),
            aspectmode='data'
        ),
        sliders=sliders,
        showlegend=True, 
        legend=dict(
            orientation="h",   # Make the legend horizontal
            yanchor="top",
            y=0.95,            # Pushes it just below the title
            xanchor="left",
            x=0.05             # Puts it on the top-left side
        ),
        updatemenus=[dict(
            type='buttons', showactive=False, y=0, x=0, xanchor='left', yanchor='top',
            buttons=[
                dict(label='Play', method='animate', args=[None, dict(frame=dict(duration=200, redraw=True), transition=dict(duration=0), fromcurrent=True, mode='immediate')]),
                dict(label='Pause', method='animate', args=[[None], dict(frame=dict(duration=0, redraw=False), mode='immediate', transition=dict(duration=0))])
            ]
        )]
    )

    return fig