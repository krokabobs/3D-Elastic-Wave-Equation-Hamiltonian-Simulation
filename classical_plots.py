"""
Classical 3D elastic wave simulation and plotting.
Combines functionality of main.py, plots.ipynb, and run_elastic_interactive_plot.py.

This version is driven by:
- `_fractures.py` for material parameters and fracture geometry
- `_initial_conditions.py` for initial wavefield (Gaussian / Ricker)

Usage (examples):
  python classical_plots.py --mode static            # save final 3D image only
  python classical_plots.py --mode interactive       # 3D interactive slider (no GIF)
  python classical_plots.py --mode animation         # 3D GIF
  python classical_plots.py --mode all               # 3D image + interactive + GIF
  python classical_plots.py --plots-2d               # also 2D z-slice PNG + GIF + HTML widget (open in browser)
  python classical_plots.py --help                   # show all options
"""
import os
import argparse
import numpy as np
from scipy.sparse.linalg import expm_multiply

import _fractures as fractures
import _initial_conditions as initial_conditions
from _utility import (
    FD_solver_3D_elastic,
    plot_elastic_3D,
    plot_elastic_3D_interactive,
    animate_elastic_3D,
    phi_to_fields_main_grid,
)


# ---------------------------------------------------------------------------
# Run options
# ---------------------------------------------------------------------------
MODE = "interactive"  # "static" | "interactive" | "animation" | "all"
INITIAL_CONDITION = "gaussian"  # "gaussian" | "ricker" (from _initial_conditions.py)
N_STEPS = 20
N_SUBSTEPS_PER_STEP = 100
OUTPUT_IMAGE = "elastic_3D_simulation.png"
OUTPUT_GIF = "elastic_3D_animation.gif"
SUBSAMPLE = 2
MAKE_ANIMATION = True  # when MODE is "animation" or "all"


def plot_2d_fields_zslice(phi_real, Nx, Ny, Nz, k, title=None, save_file=None):
    """
    2D imshow grid for a fixed z-slice (k): v_x, v_y, v_z, |v|,
    sigma_xx, sigma_yy, sigma_zz, sigma_xy, sigma_xz, sigma_yz.
    Mirrors the notebook's plot_all_fields_slice logic, but uses _phi_to_fields_main_grid.
    """
    import matplotlib
    matplotlib.use("Agg" if save_file else "TkAgg")
    import matplotlib.pyplot as plt

    k = int(np.clip(k, 0, Nz - 1))

    vxm, vym, vzm, sxx, syy, szz, sxy, sxz, syz = phi_to_fields_main_grid(
        np.real(phi_real), Nx, Ny, Nz
    )
    vmag = np.sqrt(vxm**2 + vym**2 + vzm**2)

    # fields returned are (Nz, Ny, Nx)
    slices = {
        "v_x": vxm[k, :, :],
        "v_y": vym[k, :, :],
        "v_z": vzm[k, :, :],
        "v_mag": vmag[k, :, :],
        "sigma_xx": sxx[k, :, :],
        "sigma_yy": syy[k, :, :],
        "sigma_zz": szz[k, :, :],
        "sigma_xy": sxy[k, :, :],
        "sigma_xz": sxz[k, :, :],
        "sigma_yz": syz[k, :, :],
    }

    fig, axes = plt.subplots(2, 5, figsize=(22, 8))
    fig.suptitle(title or f"Wavefield Components at Z-Slice k={k}", fontsize=16, fontweight="bold")

    layout = [
        ("v_x", "Velocity X", "RdBu"),
        ("v_y", "Velocity Y", "RdBu"),
        ("v_z", "Velocity Z", "RdBu"),
        ("v_mag", "Velocity Magnitude", "viridis"),
        ("sigma_xx", "Normal Stress XX", "RdBu"),
        ("sigma_yy", "Normal Stress YY", "RdBu"),
        ("sigma_zz", "Normal Stress ZZ", "RdBu"),
        ("sigma_xy", "Shear Stress XY", "RdBu"),
        ("sigma_xz", "Shear Stress XZ", "RdBu"),
        ("sigma_yz", "Shear Stress YZ", "RdBu"),
    ]

    for ax, (key, t, cmap) in zip(axes.flatten(), layout):
        data = slices[key]
        if cmap == "RdBu":
            vmax = float(np.max(np.abs(data)))
            vmax = vmax if vmax > 0 else 1e-10
            im = ax.imshow(data, cmap=cmap, origin="lower", vmin=-vmax, vmax=vmax)
        else:
            im = ax.imshow(data, cmap=cmap, origin="lower")
        ax.set_title(t)
        ax.set_xlabel("X index")
        ax.set_ylabel("Y index")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    if save_file:
        plt.savefig(save_file, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def animate_2d_fields_zslice(history, Nx, Ny, Nz, k, output_file="slice_2d.gif", fps=5):
    """
    Create a GIF animation of the 2D z-slice over time steps.
    Uses global max scaling per component for consistent colorbars, similar to plots.ipynb.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import imageio.v2 as imageio

    k = int(np.clip(k, 0, Nz - 1))
    n_steps = len(history)

    # Precompute frames 
    keys = ["v_x", "v_y", "v_z", "v_mag", "sigma_xx", "sigma_yy", "sigma_zz", "sigma_xy", "sigma_xz", "sigma_yz"]
    global_max = {key: 0.0 for key in keys}
    frames = []
    for phi in history:
        vxm, vym, vzm, sxx, syy, szz, sxy, sxz, syz = phi_to_fields_main_grid(np.real(phi), Nx, Ny, Nz)
        vmag = np.sqrt(vxm**2 + vym**2 + vzm**2)
        frame = {
            "v_x": vxm[k, :, :],
            "v_y": vym[k, :, :],
            "v_z": vzm[k, :, :],
            "v_mag": vmag[k, :, :],
            "sigma_xx": sxx[k, :, :],
            "sigma_yy": syy[k, :, :],
            "sigma_zz": szz[k, :, :],
            "sigma_xy": sxy[k, :, :],
            "sigma_xz": sxz[k, :, :],
            "sigma_yz": syz[k, :, :],
        }
        frames.append(frame)
        for key in keys:
            global_max[key] = max(global_max[key], float(np.max(np.abs(frame[key]))))

    layout = [
        ("v_x", "Velocity X", "RdBu"),
        ("v_y", "Velocity Y", "RdBu"),
        ("v_z", "Velocity Z", "RdBu"),
        ("v_mag", "Velocity Magnitude", "viridis"),
        ("sigma_xx", "Normal Stress XX", "RdBu"),
        ("sigma_yy", "Normal Stress YY", "RdBu"),
        ("sigma_zz", "Normal Stress ZZ", "RdBu"),
        ("sigma_xy", "Shear Stress XY", "RdBu"),
        ("sigma_xz", "Shear Stress XZ", "RdBu"),
        ("sigma_yz", "Shear Stress YZ", "RdBu"),
    ]

    images = []
    for t in range(n_steps):
        fig, axes = plt.subplots(2, 5, figsize=(22, 8))
        fig.suptitle(f"Time Evolution at Z-Slice k={k} (t={t})", fontsize=16, fontweight="bold")
        for ax, (key, title, cmap) in zip(axes.flatten(), layout):
            data = frames[t][key]
            vmax = global_max[key] if global_max[key] > 0 else 1e-10
            if cmap == "RdBu":
                im = ax.imshow(data, cmap=cmap, origin="lower", vmin=-vmax, vmax=vmax)
            else:
                im = ax.imshow(data, cmap=cmap, origin="lower", vmin=0, vmax=vmax)
            ax.set_title(title)
            ax.set_xlabel("X index")
            ax.set_ylabel("Y index")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        images.append(rgba.copy())
        plt.close(fig)

    imageio.mimsave(output_file, images, fps=fps)
    return output_file


def plot_2d_slice_html_widget(history, Nx, Ny, Nz, k, output_file="elastic_2D_slice.html", interval=200):
    """
    Same as the notebook's plot_time_slider: build 2D z-slice frames, create
    FuncAnimation, and save the animation as an HTML widget (to_jshtml()) to a file.
    Open the HTML file in a browser to get the slider + play controls like in Jupyter.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    k = int(np.clip(k, 0, Nz - 1))
    n_steps = len(history)

    keys = ["v_x", "v_y", "v_z", "v_mag", "sigma_xx", "sigma_yy", "sigma_zz", "sigma_xy", "sigma_xz", "sigma_yz"]
    global_max = {key: 0.0 for key in keys}
    frames = []
    for phi in history:
        vxm, vym, vzm, sxx, syy, szz, sxy, sxz, syz = phi_to_fields_main_grid(np.real(phi), Nx, Ny, Nz)
        vmag = np.sqrt(vxm**2 + vym**2 + vzm**2)
        frame = {
            "v_x": vxm[k, :, :], "v_y": vym[k, :, :], "v_z": vzm[k, :, :],
            "v_mag": vmag[k, :, :],
            "sigma_xx": sxx[k, :, :], "sigma_yy": syy[k, :, :], "sigma_zz": szz[k, :, :],
            "sigma_xy": sxy[k, :, :], "sigma_xz": sxz[k, :, :], "sigma_yz": syz[k, :, :],
        }
        frames.append(frame)
        for key in keys:
            global_max[key] = max(global_max[key], float(np.max(np.abs(frame[key]))))

    layout = [
        ("v_x", "Velocity X", "RdBu"), ("v_y", "Velocity Y", "RdBu"), ("v_z", "Velocity Z", "RdBu"),
        ("v_mag", "Velocity Mag", "viridis"), ("sigma_xx", "Normal XX", "RdBu"),
        ("sigma_yy", "Normal YY", "RdBu"), ("sigma_zz", "Normal ZZ", "RdBu"),
        ("sigma_xy", "Shear XY", "RdBu"), ("sigma_xz", "Shear XZ", "RdBu"), ("sigma_yz", "Shear YZ", "RdBu"),
    ]

    fig, axes = plt.subplots(2, 5, figsize=(22, 9))
    title_text = fig.suptitle(f"Time Evolution at Z-Slice k={k} (t=0)", fontsize=16, fontweight="bold")
    images = {}
    for ax, (key, title, cmap) in zip(axes.flatten(), layout):
        data = frames[0][key]
        vmax = global_max[key] if global_max[key] > 0 else 1e-10
        if cmap == "RdBu":
            im = ax.imshow(data, cmap=cmap, origin="lower", vmin=-vmax, vmax=vmax)
        else:
            im = ax.imshow(data, cmap=cmap, origin="lower", vmin=0, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        images[key] = im

    plt.tight_layout()

    def update(t):
        title_text.set_text(f"Time Evolution at Z-Slice k={k} (t={t})")
        for key in images:
            images[key].set_data(frames[t][key])
        return list(images.values())

    anim = FuncAnimation(fig, update, frames=n_steps, interval=interval, blit=False)
    html_str = anim.to_jshtml()
    plt.close(fig)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html_str)
    return output_file


def run_simulation(
    rho_model,
    S,
    Nx,
    Ny,
    Nz,
    dx,
    dy,
    dz,
    xmin,
    xmax,
    ymin,
    ymax,
    zmin,
    zmax,
    n_steps,
    n_substeps_per_step,
    initial_condition="center",
    save_snapshots=False,
    verbose=True,
):
    """
    Run time evolution. Returns final state phi_evolved and optionally list of (snapshots, times).
    """
    N_main, N_vx, N_vy, N_vz, N_sxy, N_sxz, N_syz, N_vel, N_stress, N_total = fractures.get_grid_size(
        Nx, Ny, Nz
    )

    bcs = {"L": "DBC", "R": "DBC", "T": "DBC", "B": "DBC", "F": "DBC", "Ba": "DBC"}
    H, A, B, B_sqrt, B_inv, B_inv_sqrt = FD_solver_3D_elastic(
        Nx, Ny, Nz, dx, dy, dz, rho_model, S, bcs
    )

    if verbose:
        A_sum = A + A.T
        err_a = np.max(np.abs(A_sum.data)) if hasattr(A_sum, "data") and len(A_sum.data) > 0 else (np.max(np.abs(A_sum.toarray())) if hasattr(A_sum, "toarray") else 0)
        H_diff = H - H.conj().T
        err_h = np.max(np.abs(H_diff.data)) if hasattr(H_diff, "data") and len(H_diff.data) > 0 else (np.max(np.abs(H_diff.toarray())) if hasattr(H_diff, "toarray") else 0)
        print("Matrix A anti-Hermitian:" if err_a < 1e-10 else f"WARNING A: {err_a:.2e}")
        print("Hamiltonian H Hermitian:" if err_h < 1e-10 else f"WARNING H: {err_h:.2e}")

    # CFL based on background and fracture P-wave speeds from _fractures
    c_p_fracture = np.sqrt((fractures.lambda_fracture + 2 * fractures.mu_fracture) / fractures.rho_fracture)
    c_p_min = min(c_p_fracture, fractures.c_p_base)
    dt_max = dx / c_p_min
    safety = 0.01
    dt = safety * dt_max
    dt_sub = dt / n_substeps_per_step

    if verbose:
        print(f"Time step: dt={dt:.2e} s, {n_steps} steps, dt_sub={dt_sub:.2e} s")

    # Initial condition from _initial_conditions, using the same physical
    # domain extents as `_fractures.get_grid_parameters()`.
    if initial_condition == "ricker":
        phi_0 = initial_conditions.ricker_IC(
            Nx, Ny, Nz, dx, dy, dz, xmin, xmax, ymin, ymax, zmin, zmax
        )
    else:
        phi_0 = initial_conditions.gaussian_IC(
            Nx, Ny, Nz, dx, dy, dz, xmin, xmax, ymin, ymax, zmin, zmax
        )
    phi_0 = np.asarray(phi_0, dtype=complex)
    psi_0 = B_sqrt @ phi_0
    initial_norm = np.linalg.norm(psi_0)
    psi_current = psi_0.copy() / initial_norm
    norm_factor = initial_norm

    snapshots = []
    times = []
    if save_snapshots:
        phi_t = B_inv_sqrt @ (norm_factor * psi_current)
        snapshots.append(np.real(phi_t).copy())
        times.append(0.0)

    for step in range(1, n_steps + 1):
        for _ in range(n_substeps_per_step):
            psi_current = expm_multiply(-1j * H * dt_sub, psi_current)
        ratio = np.linalg.norm(psi_current)
        if ratio > 1e4 or ratio < 1e-4:
            if verbose:
                print("Energy explosion - stopping")
            break
        if save_snapshots:
            phi_t = B_inv_sqrt @ (norm_factor * psi_current)
            snapshots.append(np.real(phi_t).copy())
            times.append(step * dt)
        if verbose and (step % 5 == 0 or step == 1 or step == n_steps):
            print(f"  Step {step}/{n_steps}: energy ratio = {ratio:.6f}")

    phi_evolved = B_inv_sqrt @ (norm_factor * psi_current)
    if verbose:
        print(f"Evolution complete. Energy ratio final: {np.linalg.norm(norm_factor * psi_current) / initial_norm:.6f}")

    return np.real(phi_evolved), (snapshots, times) if save_snapshots else (None, None)


def main():
    parser = argparse.ArgumentParser(
        description="Classical 3D elastic wave plots"
    )
    parser.add_argument(
        "--mode",
        choices=["static", "interactive", "animation", "all"],
        default=MODE,
        help="Output: static image, interactive window, GIF animation, or all",
    )
    parser.add_argument(
        "--initial",
        choices=["gaussian", "ricker"],
        default=INITIAL_CONDITION,
        help="Initial condition type from _initial_conditions.py",
    )
    parser.add_argument("--steps", type=int, default=N_STEPS, help="Number of time steps")
    parser.add_argument("--substeps", type=int, default=N_SUBSTEPS_PER_STEP, help="Substeps per step")
    parser.add_argument("--output-image", default=OUTPUT_IMAGE, help="Static plot output file")
    parser.add_argument("--output-gif", default=OUTPUT_GIF, help="Animation GIF output file")
    parser.add_argument("--plots-2d", action="store_true", help="Also save a 2D z-slice imshow grid (like plots.ipynb)")
    parser.add_argument("--slice-k", type=int, default=None, help="Z-slice index for 2D plots (default: Nz//2)")
    parser.add_argument("--slice-out", default="elastic_2D_slice.png", help="Output PNG for 2D slice grid")
    parser.add_argument("--slice-gif", default="elastic_2D_slice.gif", help="Output GIF for 2D slice animation (needs snapshots)")
    parser.add_argument("--slice-html", default="elastic_2D_slice.html", help="Output HTML widget for 2D slice (slider+play, like notebook); open in browser")
    parser.add_argument("--fracture-geometry", choices=["horizontal", "vertical", "cross"],
                        default="horizontal",
                        help="Fracture geometry from _fractures: one_horizontal, one_vertical, two_perpendicular")
    parser.add_argument("--subsample", type=int, default=SUBSAMPLE, help="Subsample factor for 3D plots")
    parser.add_argument("--no-verbose", action="store_true", help="Reduce console output")
    args = parser.parse_args()

    verbose = not args.no_verbose
    n_steps = args.steps
    n_substeps = args.substeps
    save_snapshots = args.mode in ("interactive", "animation", "all")

    # Grid and material / fracture model from _fractures
    (
        xmin,
        ymin,
        zmin,
        xmax,
        ymax,
        zmax,
        Nx,
        Ny,
        Nz,
        dx,
        dy,
        dz,
    ) = fractures.get_grid_parameters()
    if args.fracture_geometry == "vertical":
        rho_model, S = fractures.one_vertical_fracture(Nx, Ny, Nz, dx, dy, dz)
    elif args.fracture_geometry == "cross":
        rho_model, S = fractures.two_perpendicular_fractures(Nx, Ny, Nz, dx, dy, dz)
    else:
        rho_model, S = fractures.one_horizontal_fracture(Nx, Ny, Nz, dx, dy, dz)

    # Build fracture plane list for 3D plots from the density model (rho == rho_fracture).
    # Use one plane per orientation: the z-slice and x-slice that contain the most fracture
    # cells (so we get a single cross, not a grid of planes). Also compute the in-plane
    # extents from the fracture mask so that the plotted planes match the physical
    # fracture size (do not reach the domain edges).
    fracture_planes = []
    try:
        rho_frac_val = float(fractures.rho_fracture)
    except Exception:
        rho_frac_val = None
    if rho_frac_val is not None:
        frac_mask = np.isclose(rho_model, rho_frac_val)
        x_phys = np.arange(Nx) * dx - (Nx - 1) * dx / 2
        y_phys = np.arange(Ny) * dy - (Ny - 1) * dy / 2
        z_phys = np.arange(Nz) * dz - (Nz - 1) * dz / 2
        # One z-plane: the z-index with the most fracture cells (main horizontal plane)
        n_frac_per_z = np.sum(frac_mask, axis=(1, 2))
        if args.fracture_geometry in ("horizontal", "cross") and np.any(n_frac_per_z > 0):
            iz = int(np.argmax(n_frac_per_z))
            # Limit plane in x,y to where fractures actually exist
            mask_z = frac_mask[iz, :, :]  # (Ny, Nx)
            ys_nonzero = np.where(np.any(mask_z, axis=1))[0]
            xs_nonzero = np.where(np.any(mask_z, axis=0))[0]
            if ys_nonzero.size > 0 and xs_nonzero.size > 0:
                y_min, y_max = y_phys[ys_nonzero[0]], y_phys[ys_nonzero[-1]]
                x_min, x_max = x_phys[xs_nonzero[0]], x_phys[xs_nonzero[-1]]
                # Entry: (axis, coord, x_min, x_max, y_min, y_max)
                fracture_planes.append(
                    ("z", float(z_phys[iz]), float(x_min), float(x_max), float(y_min), float(y_max))
                )
        # One x-plane: the x-index with the most fracture cells (main vertical plane)
        n_frac_per_x = np.sum(frac_mask, axis=(0, 1))
        if args.fracture_geometry in ("vertical", "cross") and np.any(n_frac_per_x > 0):
            ix = int(np.argmax(n_frac_per_x))
            # Limit plane in y,z to where fractures actually exist
            mask_x = frac_mask[:, :, ix]  # (Nz, Ny)
            zs_nonzero = np.where(np.any(mask_x, axis=1))[0]
            ys_nonzero = np.where(np.any(mask_x, axis=0))[0]
            if zs_nonzero.size > 0 and ys_nonzero.size > 0:
                z_min, z_max = z_phys[zs_nonzero[0]], z_phys[zs_nonzero[-1]]
                y_min, y_max = y_phys[ys_nonzero[0]], y_phys[ys_nonzero[-1]]
                # Entry: (axis, coord, z_min, z_max, y_min, y_max)
                fracture_planes.append(
                    ("x", float(x_phys[ix]), float(z_min), float(z_max), float(y_min), float(y_max))
                )

    N_main, N_vx, N_vy, N_vz, N_sxy, N_sxz, N_syz, N_vel, N_stress, N_total = fractures.get_grid_size(
        Nx, Ny, Nz
    )
    c_p_min = min(
        np.sqrt((fractures.lambda_fracture + 2 * fractures.mu_fracture) / fractures.rho_fracture),
        fractures.c_p_base,
    )
    dt = 0.01 * (dx / c_p_min)
    t_final = n_steps * dt

    phi_real, (snapshots, times) = run_simulation(
        rho_model,
        S,
        Nx,
        Ny,
        Nz,
        dx,
        dy,
        dz,
        xmin,
        xmax,
        ymin,
        ymax,
        zmin,
        zmax,
        n_steps=n_steps,
        n_substeps_per_step=n_substeps,
        initial_condition=args.initial,
        save_snapshots=save_snapshots,
        verbose=verbose,
    )

    fracture1_z, fracture2_x = Nz // 2, Nx // 2

    # 2D slice plots (static PNG; optional GIF if we have snapshots)
    if args.plots_2d:
        k2d = (Nz // 2) if args.slice_k is None else int(args.slice_k)
        plot_2d_fields_zslice(
            phi_real,
            Nx, Ny, Nz,
            k=k2d,
            title=f"Wavefield Components at Z-Slice k={k2d} (t={t_final:.2e}s)",
            save_file=args.slice_out,
        )
        if verbose:
            print(f"Saved 2D slice grid: {args.slice_out}")
        if snapshots:
            animate_2d_fields_zslice(
                snapshots,
                Nx, Ny, Nz,
                k=k2d,
                output_file=args.slice_gif,
                fps=5,
            )
            if verbose:
                print(f"Saved 2D slice animation: {args.slice_gif}")
            # HTML widget (same as notebook's plot_time_slider / to_jshtml): slider + play in browser
            html_path = os.path.join(os.path.dirname(__file__) or ".", args.slice_html)
            plot_2d_slice_html_widget(
                snapshots, Nx, Ny, Nz, k=k2d,
                output_file=html_path,
                interval=200,
            )
            if verbose:
                print(f"Saved 2D slice HTML widget: {html_path} (open in browser for slider/play)")

    # Static plot (single final state)
    if args.mode in ("static", "all"):
        v_max = max(
            np.max(np.abs(phi_real[0:N_vx].reshape(Nz, Ny, Nx - 1))),
            np.max(np.abs(phi_real[N_vx : N_vx + N_vy].reshape(Nz, Ny - 1, Nx))),
            np.max(np.abs(phi_real[N_vx + N_vy : N_vel].reshape(Nz - 1, Ny, Nx))),
        )
        sigma_max = max(
            np.max(np.abs(phi_real[N_vel : N_vel + N_main])),
            np.max(np.abs(phi_real[N_vel + N_main : N_vel + 2 * N_main])),
            np.max(np.abs(phi_real[N_vel + 2 * N_main : N_vel + 3 * N_main])),
        )
        scale_v = max(1.0, (sigma_max / (v_max + 1e-30)) * 0.25 * 0.5) if v_max > 0 else 20.0
        scale_stress = 0.5
        plot_elastic_3D(
            phi_real,
            Nx,
            Ny,
            Nz,
            dx,
            dy,
            dz,
            title=f"3D Elastic Wave (t={t_final:.2e}s) - Staggered Grid",
            save_file=args.output_image,
            subsample=args.subsample,
            scale_v=scale_v,
            scale_stress=scale_stress,
            show_velocity=True,
            show_stress=True,
        )
        if verbose:
            print(f"Saved static plot: {args.output_image}")

    # Interactive plot (slider + play/pause)
    if args.mode in ("interactive", "all") and snapshots:
        if verbose:
            print(f"Opening interactive plot ({len(snapshots)} snapshots)")
        plot_elastic_3D_interactive(
            snapshots,
            times,
            Nx,
            Ny,
            Nz,
            dx,
            dy,
            dz,
            fracture1_z,
            fracture2_x,
            fracture_planes=fracture_planes,
            subsample=args.subsample,
            save_file=None,
        )

    # GIF animation
    if args.mode in ("animation", "all") and snapshots:
        gif_path = os.path.join(os.path.dirname(__file__) or ".", args.output_gif)
        if verbose:
            print(f"Saving animation: {gif_path}")
        animate_elastic_3D(
            snapshots,
            times,
            Nx,
            Ny,
            Nz,
            dx,
            dy,
            dz,
            fracture1_z,
            fracture2_x,
            fracture_planes=fracture_planes,
            subsample=args.subsample,
            output_file=gif_path,
            fps=10,
        )
        if verbose:
            print(f"Saved animation: {gif_path}")


if __name__ == "__main__":
    main()
