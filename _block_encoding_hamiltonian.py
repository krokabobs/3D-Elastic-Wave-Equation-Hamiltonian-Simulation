"""Helpers for analyzing clinic-style elastic Hamiltonians before block encoding."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp

from _utility import (
    create_compliance_matrix_from_velocities,
    create_compliance_matrix_isotropic,
)


@dataclass(frozen=True)
class Elastic3DLayout:
    """
    DOF layout for ``FD_solver_3D_elastic`` state vectors.

    Ordering matches ``plot_elastic_3D`` / ``main.py``:
    ``[v_x | v_y | v_z | sigma_xx | sigma_yy | sigma_zz | sigma_xy | sigma_xz | sigma_yz]``.
    """

    nx: int
    ny: int
    nz: int
    n_vx: int
    n_vy: int
    n_vz: int
    n_main: int
    n_sxy: int
    n_sxz: int
    n_syz: int
    n_vel: int
    n_stress: int
    n_total: int
    vx_slice: slice
    vy_slice: slice
    vz_slice: slice
    sxx_slice: slice
    syy_slice: slice
    szz_slice: slice
    sxy_slice: slice
    sxz_slice: slice
    syz_slice: slice

    @property
    def vel_slice(self) -> slice:
        return slice(0, self.n_vel)

    @property
    def stress_slice(self) -> slice:
        return slice(self.n_vel, self.n_total)

    def slices(self) -> dict[str, slice]:
        return {
            "v_x": self.vx_slice,
            "v_y": self.vy_slice,
            "v_z": self.vz_slice,
            "sigma_xx": self.sxx_slice,
            "sigma_yy": self.syy_slice,
            "sigma_zz": self.szz_slice,
            "sigma_xy": self.sxy_slice,
            "sigma_xz": self.sxz_slice,
            "sigma_yz": self.syz_slice,
        }

    def coarse_slices(self) -> dict[str, slice]:
        return {"velocity": self.vel_slice, "stress": self.stress_slice}


def elastic_3d_layout(nx: int, ny: int, nz: int) -> Elastic3DLayout:
    """Slice indices for the 3D elastic staggered state in ``_utility.FD_solver_3D_elastic``."""
    n_main = nx * ny * nz
    n_vx = (nx - 1) * ny * nz
    n_vy = nx * (ny - 1) * nz
    n_vz = nx * ny * (nz - 1)
    n_sxy = (nx - 1) * (ny - 1) * nz
    n_sxz = (nx - 1) * ny * (nz - 1)
    n_syz = nx * (ny - 1) * (nz - 1)
    n_vel = n_vx + n_vy + n_vz
    n_stress = 3 * n_main + n_sxy + n_sxz + n_syz
    n_total = n_vel + n_stress

    start = 0
    vx_slice = slice(start, start := start + n_vx)
    vy_slice = slice(start, start := start + n_vy)
    vz_slice = slice(start, start := start + n_vz)
    sxx_slice = slice(start, start := start + n_main)
    syy_slice = slice(start, start := start + n_main)
    szz_slice = slice(start, start := start + n_main)
    sxy_slice = slice(start, start := start + n_sxy)
    sxz_slice = slice(start, start := start + n_sxz)
    syz_slice = slice(start, start := start + n_syz)

    return Elastic3DLayout(
        nx=nx,
        ny=ny,
        nz=nz,
        n_vx=n_vx,
        n_vy=n_vy,
        n_vz=n_vz,
        n_main=n_main,
        n_sxy=n_sxy,
        n_sxz=n_sxz,
        n_syz=n_syz,
        n_vel=n_vel,
        n_stress=n_stress,
        n_total=n_total,
        vx_slice=vx_slice,
        vy_slice=vy_slice,
        vz_slice=vz_slice,
        sxx_slice=sxx_slice,
        syy_slice=syy_slice,
        szz_slice=szz_slice,
        sxy_slice=sxy_slice,
        sxz_slice=sxz_slice,
        syz_slice=syz_slice,
    )


def block_boundaries(layout: Elastic3DLayout, *, coarse: bool = False) -> list[int]:
    """Row/column tick positions for spy plots."""
    if coarse:
        return [0, layout.n_vel, layout.n_total]
    ticks = [0]
    for sl in layout.slices().values():
        ticks.append(sl.stop)
    return ticks


def clinic_elastic_materials(
    nx: int,
    ny: int,
    nz: int,
    *,
    add_fractures: bool = True,
    c_p_base: float = 6000.0,
    c_s_base: float = 3500.0,
    rho_base: float = 2700.0,
    rho_fracture: float = 1000.0,
    lambda_fracture: float = 2.2e9,
    mu_fracture: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Clinic-style density and compliance fields (``main.py`` defaults).

    Returns ``(rho_model, S, fracture_mask)`` with shapes
    ``(Nz, Ny, Nx)`` and ``(6, 6, Nz, Ny, Nx)``.
    """
    s_base = create_compliance_matrix_from_velocities(c_p_base, c_s_base, rho_base)
    rho_model = np.full((nz, ny, nx), rho_base, dtype=float)
    fracture_mask = np.zeros((nz, ny, nx), dtype=bool)

    if add_fractures:
        fracture_z = nz // 2
        fracture_x = nx // 2
        for i in range(nz):
            for j in range(ny):
                for k in range(nx):
                    if i == fracture_z or k == fracture_x:
                        fracture_mask[i, j, k] = True
                        rho_model[i, j, k] = rho_fracture

    rho_model = np.clip(rho_model, 1.0, 3000.0)

    if add_fractures:
        s_model = np.zeros((6, 6, nz, ny, nx), dtype=float)
        for i in range(nz):
            for j in range(ny):
                for k in range(nx):
                    if fracture_mask[i, j, k]:
                        s_model[:, :, i, j, k] = create_compliance_matrix_isotropic(
                            lambda_fracture, mu_fracture
                        )
                    else:
                        s_model[:, :, i, j, k] = s_base
        return rho_model, s_model, fracture_mask

    return rho_model, s_base, fracture_mask


def dense_matrix(matrix: sp.spmatrix | np.ndarray) -> np.ndarray:
    """Convert sparse or dense matrix to a dense ``ndarray``."""
    if sp.issparse(matrix):
        return matrix.toarray()
    return np.asarray(matrix)


def hermitian_error(matrix: sp.spmatrix | np.ndarray) -> float:
    """Max entrywise error ``|H - H^dagger|``."""
    dense = dense_matrix(matrix)
    return float(np.max(np.abs(dense - dense.conj().T)))


def anti_hermitian_error(matrix: sp.spmatrix | np.ndarray) -> float:
    """Max entrywise error ``|A + A^T|`` (Euclidean real-part symmetry)."""
    dense = dense_matrix(matrix)
    return float(np.max(np.abs(dense + dense.T)))


def block_slice(matrix: np.ndarray, row_slice: slice, col_slice: slice) -> np.ndarray:
    """Extract a sub-block from a dense matrix."""
    return matrix[row_slice, col_slice]


def summarize_block_coupling(
    matrix: np.ndarray,
    block_slices: dict[str, slice],
    *,
    threshold: float = 1e-12,
) -> dict[tuple[str, str], tuple[int, float]]:
    """Nonzero count and max |entry| for each ordered block pair."""
    summary: dict[tuple[str, str], tuple[int, float]] = {}
    names = list(block_slices)
    for row_name in names:
        row_sl = block_slices[row_name]
        for col_name in names:
            col_sl = block_slices[col_name]
            block = block_slice(matrix, row_sl, col_sl)
            mask = np.abs(block) > threshold
            nnz = int(mask.sum())
            peak = float(np.max(np.abs(block))) if nnz else 0.0
            summary[(row_name, col_name)] = (nnz, peak)
    return summary


@dataclass(frozen=True)
class CoefficientEntry:
    """One nonzero matrix entry with block-section metadata."""

    row: int
    col: int
    row_block: str
    col_block: str
    value: complex
    d_label: int


@dataclass(frozen=True)
class BlockCoefficientLabeling:
    """
    Pechan-style labels for distinct coefficients within each matrix block section.

    ``d = d_ind || d_val`` where ``d_ind`` encodes the ordered block pair
    ``(row_block, col_block)`` and ``d_val`` indexes distinct magnitudes within
    that section.  The same numeric value in different sections receives different
    labels (Appendix E convention).
    """

    matrix_name: str
    layout: Elastic3DLayout
    entries: tuple[CoefficientEntry, ...]
    section_order: tuple[tuple[str, str], ...]
    d_ind_bits: int
    d_val_bits: int
    d_to_value: dict[int, complex]
    value_table: np.ndarray
    d_padded: int

    @property
    def d_prime(self) -> int:
        return len(self.d_to_value)

    @property
    def n_d_qubits(self) -> int:
        return int(np.ceil(np.log2(max(self.d_padded, 2))))


def _index_block(layout: Elastic3DLayout, index: int) -> str:
    for name, sl in layout.slices().items():
        if sl.start <= index < sl.stop:
            return name
    raise ValueError(f"index {index} outside layout range [0, {layout.n_total}).")


def iter_block_entries(
    matrix: sp.spmatrix | np.ndarray,
    layout: Elastic3DLayout,
    *,
    threshold: float = 1e-12,
) -> list[tuple[int, int, str, str, complex]]:
    """Yield ``(row, col, row_block, col_block, value)`` for nonzero entries."""
    if sp.issparse(matrix):
        coo = matrix.tocoo()
        rows, cols, values = coo.row, coo.col, coo.data
    else:
        dense = np.asarray(matrix)
        rows, cols = np.nonzero(np.abs(dense) > threshold)
        values = dense[rows, cols]

    entries: list[tuple[int, int, str, str, complex]] = []
    for row, col, value in zip(rows, cols, values):
        if abs(value) <= threshold:
            continue
        entries.append(
            (
                int(row),
                int(col),
                _index_block(layout, int(row)),
                _index_block(layout, int(col)),
                complex(value),
            )
        )
    return entries


def label_coefficients_by_block(
    matrix: sp.spmatrix | np.ndarray,
    layout: Elastic3DLayout,
    *,
    matrix_name: str = "H",
    threshold: float = 1e-12,
    decimals: int = 12,
) -> BlockCoefficientLabeling:
    """
    Assign Pechan-style labels to nonzero entries grouped by block section.

    Sections are ordered block pairs ``(row_block, col_block)`` that appear in
    the sparsity pattern.  Within each section, distinct rounded values get
    separate ``d_val`` indices.
    """
    raw = iter_block_entries(matrix, layout, threshold=threshold)
    if not raw:
        raise ValueError(f"{matrix_name} has no entries above threshold.")

    section_values: dict[tuple[str, str], set[complex]] = {}
    for _, _, row_block, col_block, value in raw:
        rounded = complex(np.round(value.real, decimals), np.round(value.imag, decimals))
        section_values.setdefault((row_block, col_block), set()).add(rounded)

    section_order = tuple(sorted(section_values))
    section_index = {section: idx for idx, section in enumerate(section_order)}
    d_ind_bits = max(1, int(np.ceil(np.log2(max(len(section_order), 1)))))

    section_value_order: dict[tuple[str, str], dict[complex, int]] = {}
    max_section_values = 0
    for section in section_order:
        ordered = sorted(section_values[section], key=lambda z: (z.real, z.imag))
        section_value_order[section] = {value: idx for idx, value in enumerate(ordered)}
        max_section_values = max(max_section_values, len(ordered))

    d_val_bits = max(1, int(np.ceil(np.log2(max(max_section_values, 1)))))

    d_to_value: dict[int, complex] = {}
    labeled: list[CoefficientEntry] = []
    for row, col, row_block, col_block, value in raw:
        rounded = complex(np.round(value.real, decimals), np.round(value.imag, decimals))
        section = (row_block, col_block)
        d_ind = section_index[section]
        d_val = section_value_order[section][rounded]
        d_label = (d_ind << d_val_bits) | d_val
        d_to_value[d_label] = rounded
        labeled.append(
            CoefficientEntry(
                row=row,
                col=col,
                row_block=row_block,
                col_block=col_block,
                value=rounded,
                d_label=d_label,
            )
        )

    d_padded = 1 << int(np.ceil(np.log2(max(d_to_value) + 1)))
    value_table = np.zeros(d_padded, dtype=complex)
    for d_label, value in d_to_value.items():
        value_table[d_label] = value

    return BlockCoefficientLabeling(
        matrix_name=matrix_name,
        layout=layout,
        entries=tuple(labeled),
        section_order=section_order,
        d_ind_bits=d_ind_bits,
        d_val_bits=d_val_bits,
        d_to_value=d_to_value,
        value_table=value_table,
        d_padded=d_padded,
    )


def summarize_coefficient_labeling(labeling: BlockCoefficientLabeling) -> dict[str, object]:
    """Summary dict for notebook display and oracle budgeting."""
    labels_by_section: dict[tuple[str, str], set[int]] = {}
    for entry in labeling.entries:
        section = (entry.row_block, entry.col_block)
        labels_by_section.setdefault(section, set()).add(entry.d_label)

    by_section: dict[str, list[complex]] = {}
    distinct_counts: dict[str, int] = {}
    for section in labeling.section_order:
        key = f"{section[0]} -> {section[1]}"
        values = sorted(
            {labeling.d_to_value[d_label] for d_label in labels_by_section.get(section, set())},
            key=lambda z: (z.real, z.imag),
        )
        by_section[key] = values
        distinct_counts[key] = len(values)

    return {
        "matrix": labeling.matrix_name,
        "D_prime": labeling.d_prime,
        "D_padded": labeling.d_padded,
        "n_d_qubits": labeling.n_d_qubits,
        "n_sections": len(labeling.section_order),
        "d_ind_bits": labeling.d_ind_bits,
        "d_val_bits": labeling.d_val_bits,
        "distinct_values_per_section": distinct_counts,
        "section_values": by_section,
    }


def diagonal_material_catalog(
    b_inv_sqrt: sp.spmatrix | np.ndarray,
    layout: Elastic3DLayout,
    *,
    threshold: float = 1e-12,
    decimals: int = 12,
) -> pd.DataFrame:
    """
    Distinct diagonal material coefficients in ``B^{-1/2}`` per physical block.

    These are the natural payloads for a separate material-loading oracle before
    coupling blocks are applied.
    """
    diag = dense_matrix(b_inv_sqrt).diagonal()
    rows: list[dict[str, object]] = []
    for block_name, sl in layout.slices().items():
        values = diag[sl]
        uniq = sorted(
            {complex(np.round(v, decimals)) for v in values if abs(v) > threshold},
            key=lambda z: (z.real, z.imag),
        )
        rows.append(
            {
                "block": block_name,
                "n_dof": sl.stop - sl.start,
                "n_distinct": len(uniq),
                "values": uniq,
            }
        )
    return pd.DataFrame(rows)


def coefficient_labeling_table(labeling: BlockCoefficientLabeling) -> pd.DataFrame:
    """Per-section distinct coefficient counts (sorted by section name)."""
    summary = summarize_coefficient_labeling(labeling)["distinct_values_per_section"]
    rows = [
        {"section": section, "n_distinct": count, "matrix": labeling.matrix_name}
        for section, count in sorted(summary.items())
    ]
    return pd.DataFrame(rows)
