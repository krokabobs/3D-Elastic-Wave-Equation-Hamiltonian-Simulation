"""Helpers for analyzing clinic-style elastic Hamiltonians before block encoding."""

from __future__ import annotations

from dataclasses import dataclass
import time as time_mod

import numpy as np
import pandas as pd
import scipy.sparse as sp
from qiskit import QuantumCircuit
from qiskit.circuit import Gate
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Operator, SparsePauliOp, Statevector
from scipy.linalg import expm
from scipy.sparse.linalg import expm_multiply

from _block_encoding_common import (
    apply_multiplexed_ry,
    data_loading_subcircuit,
    pad_to_power_of_two,
    spectral_scale,
    transpiled_gate_counts,
)
from _utility import (
    FD_solver_3D_elastic,
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


def reconstruct_from_labeling(
    labeling: BlockCoefficientLabeling,
    *,
    n: int | None = None,
) -> np.ndarray:
    """Rebuild a dense matrix from labeled nonzero entries."""
    dim = labeling.layout.n_total if n is None else n
    matrix = np.zeros((dim, dim), dtype=complex)
    for entry in labeling.entries:
        matrix[entry.row, entry.col] = labeling.d_to_value[entry.d_label]
    return matrix


def verify_labeling(
    labeling: BlockCoefficientLabeling,
    matrix: sp.spmatrix | np.ndarray,
    *,
    threshold: float = 1e-12,
) -> dict[str, float]:
    """
    Check that labels/value table recover every nonzero and introduce no extras.
    """
    target = dense_matrix(matrix)
    rebuilt = reconstruct_from_labeling(labeling, n=target.shape[0])
    residual = rebuilt - target
    nnz_mismatch = int(
        np.sum((np.abs(rebuilt) > threshold) != (np.abs(target) > threshold))
    )
    value_table_err = 0.0
    for entry in labeling.entries:
        value_table_err = max(
            value_table_err,
            abs(labeling.value_table[entry.d_label] - entry.value),
        )
    return {
        "max_abs_error": float(np.max(np.abs(residual))),
        "nnz_mask_mismatches": float(nnz_mismatch),
        "value_table_error": float(value_table_err),
    }


def imag_payload(labeling: BlockCoefficientLabeling) -> np.ndarray:
    """
    Real data-loading payload for clinic elastic ``H``.

    Clinic ``H`` is purely imaginary Hermitian (``Re(H)=0``), so ``O_data``
    loads ``Im(H_d)`` and the encoded block is compared to ``Im(H)/alpha``.
    """
    return np.real(labeling.value_table * (-1j))


def summarize_hamiltonian_odata_scaling(
    grid_sizes: tuple[tuple[int, int, int], ...] = ((2, 2, 2), (4, 2, 2), (4, 4, 2)),
    *,
    add_fractures: bool = True,
    dx: float = 0.05,
    sk_t_count: bool = False,
) -> pd.DataFrame:
    """Fast ``D'`` / ``O_data`` sweep without assembling index oracles."""
    bcs = {"L": "DBC", "R": "DBC", "T": "DBC", "B": "DBC", "F": "DBC", "Ba": "DBC"}
    rows: list[dict[str, object]] = []
    for nx, ny, nz in grid_sizes:
        rho, compliance, _ = clinic_elastic_materials(
            nx, ny, nz, add_fractures=add_fractures
        )
        hamiltonian, *_ = FD_solver_3D_elastic(
            nx, ny, nz, dx, dx, dx, rho, compliance, bcs
        )
        layout = elastic_3d_layout(nx, ny, nz)
        labeling = label_coefficients_by_block(hamiltonian, layout, matrix_name="H")
        summary = summarize_coefficient_labeling(labeling)
        payload = imag_payload(labeling)
        alpha = spectral_scale(dense_matrix(hamiltonian).imag)
        if alpha < 1e-15:
            alpha = 1.0
        odata = data_loading_subcircuit(payload, alpha)
        counts = transpiled_gate_counts(odata, sk_t_count=sk_t_count)
        rows.append(
            {
                "nx": nx,
                "ny": ny,
                "nz": nz,
                "N": layout.n_total,
                "D_prime": summary["D_prime"],
                "D_padded": summary["D_padded"],
                "n_sections": summary["n_sections"],
                "n_d_qubits": summary["n_d_qubits"],
                "O_data_depth": counts["depth"],
                "O_data_size": counts["size"],
                "O_data_t_gates": counts["t_gates"],
                "alpha_Im": alpha,
            }
        )
    return pd.DataFrame(rows)


def label_material_and_coupling(
    a_matrix: sp.spmatrix | np.ndarray,
    b_inv_sqrt: sp.spmatrix | np.ndarray,
    layout: Elastic3DLayout,
) -> dict[str, object]:
    """
    Physics-faithful split: label ``iA`` and diagonal ``B^{-1/2}`` separately.

    Returns labelings plus product check against ``H = i B^{-1/2} A B^{-1/2}``.
    """
    a_dense = dense_matrix(a_matrix)
    bis_dense = dense_matrix(b_inv_sqrt)
    i_a = 1j * a_dense
    labeling_ia = label_coefficients_by_block(i_a, layout, matrix_name="iA")
    labeling_bis = label_coefficients_by_block(bis_dense, layout, matrix_name="B_inv_sqrt")
    h_product = bis_dense @ (1j * a_dense) @ bis_dense
    return {
        "labeling_iA": labeling_ia,
        "labeling_B_inv_sqrt": labeling_bis,
        "summary_iA": summarize_coefficient_labeling(labeling_ia),
        "summary_B_inv_sqrt": summarize_coefficient_labeling(labeling_bis),
        "material_catalog": diagonal_material_catalog(bis_dense, layout),
        "H_from_factors": h_product,
        "verify_iA": verify_labeling(labeling_ia, i_a),
        "verify_B_inv_sqrt": verify_labeling(labeling_bis, bis_dense),
    }


def odata_budget_for_labeling(
    labeling: BlockCoefficientLabeling,
    *,
    use_imag: bool = True,
    sk_t_count: bool = False,
) -> dict[str, object]:
    """Transpiled ``O_data`` budget for a Hamiltonian-style value table."""
    payload = imag_payload(labeling) if use_imag else np.real(labeling.value_table)
    scale = float(np.max(np.abs(payload)))
    if scale < 1e-15:
        scale = 1.0
    # spectral-style: use max abs when matrix not attached
    alpha = scale
    dens = reconstruct_from_labeling(labeling)
    mat = dens.imag if use_imag else dens.real
    alpha = spectral_scale(mat) if np.max(np.abs(mat)) > 0 else scale
    counts = transpiled_gate_counts(
        data_loading_subcircuit(payload, alpha), sk_t_count=sk_t_count
    )
    return {"alpha": alpha, "payload": payload, **counts}


@dataclass(frozen=True)
class HamiltonianOracleLabeling:
    """
    Pechan-style ``(d, m) -> (i, j)`` maps for staggered elastic ``H``.

    Index register is padded to a power of two so sparse clinic DOF counts
    (not powers of two) still fit a qubit register.
    """

    base: BlockCoefficientLabeling
    matrix: np.ndarray
    m_padded: int
    n_index_padded: int
    entries_by_dm: dict[tuple[int, int], tuple[int, int, complex]]
    d_ind_map: dict[int, tuple[int, int]]

    @property
    def n_d_qubits(self) -> int:
        return self.base.n_d_qubits

    @property
    def n_m_qubits(self) -> int:
        return int(np.ceil(np.log2(max(self.m_padded, 2))))

    @property
    def n_index_qubits(self) -> int:
        return int(np.ceil(np.log2(max(self.n_index_padded, 2))))

    @property
    def n_grid(self) -> int:
        return self.matrix.shape[0]

    @property
    def num_qubits_uh(self) -> int:
        return 1 + self.n_d_qubits + self.n_m_qubits + self.n_index_qubits


def build_hamiltonian_oracle_labeling(
    labeling: BlockCoefficientLabeling,
    matrix: sp.spmatrix | np.ndarray,
) -> HamiltonianOracleLabeling:
    """Attach multiplicity ``m`` and ``(d,m)->(i,j)`` maps for lookup oracles."""
    dense = dense_matrix(matrix)
    by_d: dict[int, list[tuple[int, int, complex]]] = {}
    for entry in labeling.entries:
        by_d.setdefault(entry.d_label, []).append((entry.row, entry.col, entry.value))

    entries_by_dm: dict[tuple[int, int], tuple[int, int, complex]] = {}
    max_m = 0
    for d_label, items in by_d.items():
        # Stable order: lower triangle first (incl. diagonal), then upper.
        lower = [(r, c, v) for r, c, v in items if r >= c]
        upper = [(r, c, v) for r, c, v in items if r < c]
        ordered = lower + upper
        max_m = max(max_m, len(ordered))
        for m_label, (row, col, value) in enumerate(ordered):
            entries_by_dm[(d_label, m_label)] = (row, col, value)

    d_ind_map = {
        d_label: (d_label >> labeling.d_val_bits, d_label & ((1 << labeling.d_val_bits) - 1))
        for d_label in labeling.d_to_value
    }
    return HamiltonianOracleLabeling(
        base=labeling,
        matrix=dense,
        m_padded=pad_to_power_of_two(max(max_m, 1)),
        n_index_padded=pad_to_power_of_two(dense.shape[0]),
        entries_by_dm=entries_by_dm,
        d_ind_map=d_ind_map,
    )


def _index_bits(index: int, n_qubits: int) -> list[int]:
    return [(index >> bit) & 1 for bit in range(n_qubits)]


def _basis_index(bit_lists: list[list[int]]) -> int:
    bits: list[int] = []
    for chunk in bit_lists:
        bits.extend(chunk)
    return sum(bit << pos for pos, bit in enumerate(bits))


def _complete_permutation(partial: dict[int, int], dim: int, *, label: str) -> dict[int, int]:
    used_outputs = set(partial.values())
    if len(used_outputs) != len(partial):
        raise ValueError(f"{label}: permutation is not injective.")
    free_inputs = [idx for idx in range(dim) if idx not in partial]
    free_outputs = [idx for idx in range(dim) if idx not in used_outputs]
    if len(free_inputs) != len(free_outputs):
        raise ValueError(f"{label}: could not complete permutation.")
    completed = dict(partial)
    for inp, out in zip(free_inputs, free_outputs):
        completed[inp] = out
    return completed


def build_column_oracle_map(oracle: HamiltonianOracleLabeling) -> dict[int, int]:
    """Lookup ``O_c``: ``|d>|m>|0> -> |d>|m>|j`` on the padded index register."""
    n_d, n_m, n_idx = oracle.n_d_qubits, oracle.n_m_qubits, oracle.n_index_qubits
    dim = 2 ** (n_d + n_m + n_idx)
    partial: dict[int, int] = {}
    for (d_label, m_label), (_, col, _) in oracle.entries_by_dm.items():
        in_index = _basis_index(
            [_index_bits(d_label, n_d), _index_bits(m_label, n_m), _index_bits(0, n_idx)]
        )
        out_index = _basis_index(
            [_index_bits(d_label, n_d), _index_bits(m_label, n_m), _index_bits(col, n_idx)]
        )
        partial[in_index] = out_index
    return _complete_permutation(partial, dim, label="O_c")


def build_row_oracle_map(oracle: HamiltonianOracleLabeling) -> dict[int, int]:
    """Lookup ``O_r``: ``|d>|m>|j> -> |d>|m>|i``."""
    n_d, n_m, n_idx = oracle.n_d_qubits, oracle.n_m_qubits, oracle.n_index_qubits
    dim = 2 ** (n_d + n_m + n_idx)
    partial: dict[int, int] = {}
    for (d_label, m_label), (row, col, _) in oracle.entries_by_dm.items():
        in_index = _basis_index(
            [_index_bits(d_label, n_d), _index_bits(m_label, n_m), _index_bits(col, n_idx)]
        )
        out_index = _basis_index(
            [_index_bits(d_label, n_d), _index_bits(m_label, n_m), _index_bits(row, n_idx)]
        )
        partial[in_index] = out_index
    return _complete_permutation(partial, dim, label="O_r")


def verify_staggered_index_oracles(
    oracle: HamiltonianOracleLabeling,
) -> dict[str, object]:
    """Coverage and injectivity checks for staggered ``(d,m)->(i,j)`` maps."""
    n = oracle.n_grid
    covered = np.zeros((n, n), dtype=bool)
    duplicates = 0
    for (d_label, m_label), (row, col, value) in oracle.entries_by_dm.items():
        if covered[row, col]:
            duplicates += 1
        covered[row, col] = True
        expected = oracle.matrix[row, col]
        if abs(value - expected) > 1e-10 and abs(value - oracle.base.d_to_value[d_label]) > 1e-10:
            raise ValueError(f"Value mismatch at ({row},{col}) for ({d_label},{m_label}).")

    nnz_mask = np.abs(oracle.matrix) > 1e-12
    missing = int(np.sum(nnz_mask & ~covered[: nnz_mask.shape[0], : nnz_mask.shape[1]]))
    extras = int(np.sum(covered & ~nnz_mask))

    oc = build_column_oracle_map(oracle)
    or_map = build_row_oracle_map(oracle)
    n_d, n_m, n_idx = oracle.n_d_qubits, oracle.n_m_qubits, oracle.n_index_qubits
    map_errors = 0
    for (d_label, m_label), (row, col, _) in oracle.entries_by_dm.items():
        start = _basis_index(
            [_index_bits(d_label, n_d), _index_bits(m_label, n_m), _index_bits(0, n_idx)]
        )
        after_c = oc[start]
        after_r = or_map[after_c]
        expected_idx = _basis_index(
            [_index_bits(d_label, n_d), _index_bits(m_label, n_m), _index_bits(row, n_idx)]
        )
        if after_r != expected_idx:
            map_errors += 1

    return {
        "n_entries": len(oracle.entries_by_dm),
        "duplicates": duplicates,
        "missing_nnz": missing,
        "extra_coverage": extras,
        "oracle_chain_errors": map_errors,
        "n_d_qubits": oracle.n_d_qubits,
        "n_m_qubits": oracle.n_m_qubits,
        "n_index_qubits": oracle.n_index_qubits,
        "num_qubits_uh": oracle.num_qubits_uh,
        "ok": duplicates == 0 and missing == 0 and extras == 0 and map_errors == 0,
    }


def _classical_bits_apply_controlled_xor(
    bits: list[int],
    controls: list[int],
    ctrl_state: int,
    targets: list[int],
    value: int,
) -> None:
    """In-place classical action of :func:`_append_controlled_xor_constant`."""
    if value == 0 or not targets:
        return
    matched = True
    for i, qubit in enumerate(controls):
        if bits[qubit] != ((ctrl_state >> i) & 1):
            matched = False
            break
    if not matched:
        return
    for bit, target in enumerate(targets):
        if (value >> bit) & 1:
            bits[target] ^= 1


def _bits_to_int(bits: list[int]) -> int:
    value = 0
    for i, bit in enumerate(bits):
        if bit:
            value |= 1 << i
    return value


def verify_circuit_index_oracles(
    oracle: HamiltonianOracleLabeling,
    *,
    spot_checks: int = 8,
    rng_seed: int = 0,
    max_sv_lookup_qubits: int = 14,
) -> dict[str, object]:
    """
    Check real MCX index oracles against ``(d,m)->(i,j)`` maps.

    Validates gate counts (one MCX per set bit of each XOR constant) and
    spot-checks labels by **classical bit-level** simulation of the same
    controlled-XOR MCXs (no ``2^{n_lookup}`` statevector). Optional Qiskit SV
    spot-checks run only when ``n_lookup <= max_sv_lookup_qubits``.
    """
    n_d, n_m, n_idx = oracle.n_d_qubits, oracle.n_m_qubits, oracle.n_index_qubits
    dim_idx = 2 ** (n_d + n_m + n_idx)
    dim_dm = 2 ** (n_d + n_m)
    lookup_qubits = n_d + n_m + n_idx
    controls = list(range(n_d + n_m))
    targets = list(range(n_d + n_m, n_d + n_m + n_idx))

    oc_circ = build_column_oracle_circuit(oracle)
    or_circ = build_row_oracle_circuit(oracle)
    ot_circ = build_transpose_oracle_circuit(oracle)
    expect_oc = sum(int(col).bit_count() for _, (_, col, _) in oracle.entries_by_dm.items())
    expect_or = sum(
        (int(row) ^ int(col)).bit_count()
        for _, (row, col, _) in oracle.entries_by_dm.items()
    )
    expect_ot = len(oracle.base.d_to_value) if n_m > 0 else 0
    struct_ok = (
        oc_circ.size() == expect_oc
        and or_circ.size() == expect_or
        and ot_circ.size() == expect_ot
    )

    def _bit_oc(d_label: int, m_label: int, col: int) -> int:
        bits = (
            _index_bits(d_label, n_d)
            + _index_bits(m_label, n_m)
            + _index_bits(0, n_idx)
        )
        _classical_bits_apply_controlled_xor(
            bits, controls, _dm_ctrl_state(d_label, m_label, n_d), targets, int(col)
        )
        return _bits_to_int(bits[n_d + n_m :])

    def _bit_or(d_label: int, m_label: int, row: int, col: int) -> int:
        bits = (
            _index_bits(d_label, n_d)
            + _index_bits(m_label, n_m)
            + _index_bits(col, n_idx)
        )
        _classical_bits_apply_controlled_xor(
            bits,
            controls,
            _dm_ctrl_state(d_label, m_label, n_d),
            targets,
            int(row) ^ int(col),
        )
        return _bits_to_int(bits[n_d + n_m :])

    def _bit_ot(d_label: int, m_label: int) -> int:
        bits = _index_bits(d_label, n_d) + _index_bits(m_label, n_m)
        if n_m == 0:
            return int(m_label)
        # Same as build_transpose_oracle_circuit: flip high m bit controlled on d.
        m_high = n_d + n_m - 1
        if n_d == 0:
            bits[m_high] ^= 1
        else:
            matched = all(
                bits[i] == ((int(d_label) >> i) & 1) for i in range(n_d)
            )
            if matched:
                bits[m_high] ^= 1
        return _bits_to_int(bits[n_d:])

    def _sv_oc(d_label: int, m_label: int, col: int) -> int:
        qc = QuantumCircuit(n_d + n_m + n_idx)
        _append_controlled_xor_constant(
            qc, controls, _dm_ctrl_state(d_label, m_label, n_d), targets, int(col)
        )
        start = _basis_index(
            [_index_bits(d_label, n_d), _index_bits(m_label, n_m), _index_bits(0, n_idx)]
        )
        psi = np.zeros(dim_idx, dtype=complex)
        psi[start] = 1.0
        return int(np.argmax(np.abs(Statevector(psi).evolve(qc).data)))

    def _sv_or(d_label: int, m_label: int, row: int, col: int) -> int:
        qc = QuantumCircuit(n_d + n_m + n_idx)
        _append_controlled_xor_constant(
            qc,
            controls,
            _dm_ctrl_state(d_label, m_label, n_d),
            targets,
            int(row) ^ int(col),
        )
        start = _basis_index(
            [
                _index_bits(d_label, n_d),
                _index_bits(m_label, n_m),
                _index_bits(col, n_idx),
            ]
        )
        psi = np.zeros(dim_idx, dtype=complex)
        psi[start] = 1.0
        return int(np.argmax(np.abs(Statevector(psi).evolve(qc).data)))

    def _sv_ot(d_label: int, m_label: int) -> int:
        qc = QuantumCircuit(n_d + n_m)
        if n_m == 0:
            return _basis_index([_index_bits(d_label, n_d), _index_bits(m_label, n_m)])
        if n_d == 0:
            qc.x(n_m - 1)
        else:
            qc.mcx(list(range(n_d)), n_d + n_m - 1, ctrl_state=int(d_label))
        start = _basis_index(
            [_index_bits(d_label, n_d), _index_bits(m_label, n_m)]
        )
        psi = np.zeros(dim_dm, dtype=complex)
        psi[start] = 1.0
        return int(np.argmax(np.abs(Statevector(psi).evolve(qc).data)))

    bit_errors = 0
    sv_errors = 0
    use_sv = lookup_qubits <= max_sv_lookup_qubits
    items = list(oracle.entries_by_dm.items())
    rng = np.random.default_rng(rng_seed)
    if items and spot_checks > 0:
        picks = rng.choice(len(items), size=min(spot_checks, len(items)), replace=False)
        for idx in picks:
            (d_label, m_label), (row, col, _) = items[int(idx)]
            if _bit_oc(d_label, m_label, col) != int(col):
                bit_errors += 1
            if _bit_or(d_label, m_label, row, col) != int(row):
                bit_errors += 1
            if n_m > 0 and _bit_ot(d_label, m_label) != (
                int(m_label) ^ (1 << (n_m - 1))
            ):
                bit_errors += 1

            if use_sv:
                expect_c = _basis_index(
                    [
                        _index_bits(d_label, n_d),
                        _index_bits(m_label, n_m),
                        _index_bits(col, n_idx),
                    ]
                )
                expect_r = _basis_index(
                    [
                        _index_bits(d_label, n_d),
                        _index_bits(m_label, n_m),
                        _index_bits(row, n_idx),
                    ]
                )
                if _sv_oc(d_label, m_label, col) != expect_c:
                    sv_errors += 1
                if _sv_or(d_label, m_label, row, col) != expect_r:
                    sv_errors += 1
                if n_m > 0:
                    out_m = m_label ^ (1 << (n_m - 1))
                    m_expect = _basis_index(
                        [_index_bits(d_label, n_d), _index_bits(out_m, n_m)]
                    )
                    if _sv_ot(d_label, m_label) != m_expect:
                        sv_errors += 1

    # Full nnz chain on bit semantics (still O(nnz), not O(2^{n_lookup})).
    chain_errors = 0
    for (d_label, m_label), (row, col, _) in oracle.entries_by_dm.items():
        if _bit_oc(d_label, m_label, col) != int(col):
            chain_errors += 1
        if _bit_or(d_label, m_label, row, col) != int(row):
            chain_errors += 1

    return {
        "structure_ok": struct_ok,
        "oc_size": oc_circ.size(),
        "or_size": or_circ.size(),
        "ot_size": ot_circ.size(),
        "expect_oc_size": expect_oc,
        "expect_or_size": expect_or,
        "expect_ot_size": expect_ot,
        "bit_spot_errors": bit_errors,
        "bit_chain_errors": chain_errors,
        "statevector_spot_errors": sv_errors,
        "statevector_spot_checks_used": use_sv,
        "lookup_dim": dim_idx,
        "dm_dim": dim_dm,
        "ok": struct_ok
        and bit_errors == 0
        and chain_errors == 0
        and sv_errors == 0,
    }


def _make_permutation_unitary(
    mapping: dict[int, int],
    dim: int,
    *,
    label: str,
) -> UnitaryGate:
    """Build a permutation unitary from a bijection on basis indices."""
    operator = np.zeros((dim, dim), dtype=complex)
    used_outputs: set[int] = set()
    for input_state, output_state in sorted(mapping.items()):
        if output_state in used_outputs:
            raise ValueError(f"{label}: output state {output_state} is not injective.")
        operator[output_state, input_state] = 1.0
        used_outputs.add(output_state)

    free_inputs = [idx for idx in range(dim) if idx not in mapping]
    free_outputs = [idx for idx in range(dim) if idx not in used_outputs]
    if len(free_inputs) != len(free_outputs):
        raise ValueError(f"{label}: could not complete permutation.")
    for input_state, output_state in zip(free_inputs, free_outputs):
        operator[output_state, input_state] = 1.0

    if not np.allclose(operator @ operator.conj().T, np.eye(dim)):
        raise ValueError(f"{label}: constructed operator is not unitary.")
    return UnitaryGate(operator, check_input=False, label=label)


def _append_controlled_xor_constant(
    circuit: QuantumCircuit,
    controls: list[int],
    ctrl_state: int,
    targets: list[int],
    value: int,
) -> None:
    """
    Controlled XOR of a classical constant into ``targets``.

    For each bit of ``value`` that is 1, applies an MCX with the given control
    pattern. Self-inverse; implements ``|c>|x> -> |c>|x ⊕ value>`` on the
    control subspace.
    """
    if value == 0 or not targets:
        return
    for bit, target in enumerate(targets):
        if (value >> bit) & 1:
            if controls:
                circuit.mcx(controls, target, ctrl_state=ctrl_state)
            else:
                circuit.x(target)


def _dm_ctrl_state(d_label: int, m_label: int, n_d: int) -> int:
    """Pack ``(d, m)`` into a Qiskit ``ctrl_state`` (LSB = first control)."""
    return int(d_label) | (int(m_label) << n_d)


def build_column_oracle_circuit(oracle: HamiltonianOracleLabeling) -> QuantumCircuit:
    """
    Real ``O_c`` circuit: ``|d>|m>|0> -> |d>|m>|j>`` via controlled XOR of ``j``.

    Memory-efficient (no dense ``2^{n_d+n_m+n_idx}`` matrix). Self-inverse.
    """
    n_d, n_m, n_idx = oracle.n_d_qubits, oracle.n_m_qubits, oracle.n_index_qubits
    circuit = QuantumCircuit(n_d + n_m + n_idx, name="O_c")
    controls = list(range(n_d + n_m))
    targets = list(range(n_d + n_m, n_d + n_m + n_idx))
    for (d_label, m_label), (_, col, _) in oracle.entries_by_dm.items():
        _append_controlled_xor_constant(
            circuit,
            controls,
            _dm_ctrl_state(d_label, m_label, n_d),
            targets,
            int(col),
        )
    return circuit


def build_row_oracle_circuit(oracle: HamiltonianOracleLabeling) -> QuantumCircuit:
    """
    Real ``O_r`` circuit: ``|d>|m>|j> -> |d>|m>|i>`` via controlled XOR of ``i⊕j``.

    On each ``(d,m)`` subspace this is translation by ``i⊕j``, hence unitary;
    it sends the physical column index ``j`` to the row index ``i``.
    """
    n_d, n_m, n_idx = oracle.n_d_qubits, oracle.n_m_qubits, oracle.n_index_qubits
    circuit = QuantumCircuit(n_d + n_m + n_idx, name="O_r")
    controls = list(range(n_d + n_m))
    targets = list(range(n_d + n_m, n_d + n_m + n_idx))
    for (d_label, m_label), (row, col, _) in oracle.entries_by_dm.items():
        _append_controlled_xor_constant(
            circuit,
            controls,
            _dm_ctrl_state(d_label, m_label, n_d),
            targets,
            int(row) ^ int(col),
        )
    return circuit


def build_transpose_oracle_circuit(oracle: HamiltonianOracleLabeling) -> QuantumCircuit:
    """
    Real ``O_t``: flip the high ``m`` bit for each active coefficient label ``d``.

    Clinic elastic ``H`` has only off-diagonal ``v <-> sigma`` sections, so the
    Hermitian partner is the other multiplicity slot within the section.
    """
    n_d, n_m = oracle.n_d_qubits, oracle.n_m_qubits
    circuit = QuantumCircuit(n_d + n_m, name="O_t")
    if n_m == 0:
        return circuit
    d_controls = list(range(n_d))
    m_high = n_d + n_m - 1
    for d_label in oracle.base.d_to_value:
        if n_d == 0:
            circuit.x(m_high)
        else:
            circuit.mcx(d_controls, m_high, ctrl_state=int(d_label))
    return circuit


def _normalize_lookup_mode(materialize: bool | str) -> str:
    """Map legacy ``materialize`` flags to ``opaque`` | ``circuit`` | ``dense``."""
    if isinstance(materialize, str):
        mode = materialize.lower()
        if mode not in {"opaque", "circuit", "dense"}:
            raise ValueError("lookup mode must be 'opaque', 'circuit', or 'dense'.")
        return mode
    return "dense" if materialize else "opaque"


def _lookup_gate_from_map(
    mapping: dict[int, int],
    dim: int,
    *,
    label: str,
    materialize: bool | str,
) -> Gate:
    """Opaque or dense-``UnitaryGate`` oracle from a completed permutation map."""
    n_qubits = int(np.log2(dim))
    mode = _normalize_lookup_mode(materialize)
    if mode == "opaque":
        return Gate(label, n_qubits, [])
    if mode == "dense":
        return _make_permutation_unitary(mapping, dim, label=label)
    raise ValueError(f"mode {mode!r} is not valid for map-based lookup; use circuit builders.")


def build_column_oracle_lookup(
    oracle: HamiltonianOracleLabeling,
    *,
    materialize: bool | str = False,
) -> Gate:
    """
    Lookup ``O_c``: ``|d>|m>|0> -> |d>|m>|j>``.

    ``materialize``: ``False``/``\"opaque\"`` placeholder; ``\"circuit\"`` real MCX
    loads; ``True``/``\"dense\"`` full unitary (tiny registers only).
    """
    mode = _normalize_lookup_mode(materialize)
    if mode == "circuit":
        return build_column_oracle_circuit(oracle).to_gate(label="O_c")
    n_d, n_m, n_idx = oracle.n_d_qubits, oracle.n_m_qubits, oracle.n_index_qubits
    dim = 2 ** (n_d + n_m + n_idx)
    return _lookup_gate_from_map(
        build_column_oracle_map(oracle), dim, label="O_c", materialize=mode
    )


def build_row_oracle_lookup(
    oracle: HamiltonianOracleLabeling,
    *,
    materialize: bool | str = False,
) -> Gate:
    """Lookup ``O_r``: ``|d>|m>|j> -> |d>|m>|i>`` (see ``build_column_oracle_lookup``)."""
    mode = _normalize_lookup_mode(materialize)
    if mode == "circuit":
        return build_row_oracle_circuit(oracle).to_gate(label="O_r")
    n_d, n_m, n_idx = oracle.n_d_qubits, oracle.n_m_qubits, oracle.n_index_qubits
    dim = 2 ** (n_d + n_m + n_idx)
    return _lookup_gate_from_map(
        build_row_oracle_map(oracle), dim, label="O_r", materialize=mode
    )


def build_transpose_oracle_hamiltonian(
    oracle: HamiltonianOracleLabeling,
    *,
    materialize: bool | str = False,
) -> Gate:
    """
    Pechan / Sünderhauf ``O_t`` on ``|d>|m>``.

    Clinic elastic ``H`` has no stencil-diagonal sections (only ``v <-> sigma``
    couplings), so active labels flip the high ``m`` bit to access Hermitian
    partners within each block section.
    """
    mode = _normalize_lookup_mode(materialize)
    if mode == "circuit":
        return build_transpose_oracle_circuit(oracle).to_gate(label="O_t")

    n_d = oracle.n_d_qubits
    n_m = oracle.n_m_qubits
    dim = 2 ** (n_d + n_m)
    permutation: dict[int, int] = {}

    for d_label in range(oracle.base.d_padded):
        for m_label in range(oracle.m_padded):
            in_index = _basis_index(
                [_index_bits(d_label, n_d), _index_bits(m_label, n_m)]
            )
            if d_label in oracle.base.d_to_value and n_m > 0:
                m_bits = _index_bits(m_label, n_m)
                m_bits[-1] ^= 1
                out_m = _basis_index([m_bits])
            else:
                out_m = m_label
            out_index = _basis_index(
                [_index_bits(d_label, n_d), _index_bits(out_m, n_m)]
            )
            permutation[in_index] = out_index

    if mode == "opaque":
        return Gate("O_t", n_d + n_m, [])
    return _make_permutation_unitary(permutation, dim, label="O_t")


def build_hamiltonian_block_encoding_circuit(
    oracle: HamiltonianOracleLabeling,
    *,
    scale: float | None = None,
    materialize_lookup: bool | str = "circuit",
) -> tuple[QuantumCircuit, float]:
    """
    Assembled elastic block-encoding circuit ``U_H`` (Pechan / Sünderhauf).

    Register layout: ``data | d | m | idx`` (same as Laplacian ``U_G``).

    Loads ``Im(H)`` magnitudes on the data qubit (clinic ``H`` is purely
    imaginary Hermitian) and applies index oracles ``O_c, O_r, O_t``.

    ``materialize_lookup``:
      - ``\"circuit\"`` (default) — real controlled-XOR / MCX index oracles;
      - ``\"opaque\"`` / ``False`` — placeholder gates (fast draw / scaling);
      - ``\"dense\"`` / ``True`` — dense ``UnitaryGate`` (tiny registers only).
    """
    target = oracle.matrix.imag
    alpha = scale if scale is not None else spectral_scale(target)
    if alpha < 1e-15:
        alpha = 1.0
    payload = imag_payload(oracle.base)
    mode = _normalize_lookup_mode(materialize_lookup)

    n_d = oracle.n_d_qubits
    n_m = oracle.n_m_qubits
    n_idx = oracle.n_index_qubits

    oc = build_column_oracle_lookup(oracle, materialize=mode)
    or_gate = build_row_oracle_lookup(oracle, materialize=mode)
    ot = build_transpose_oracle_hamiltonian(oracle, materialize=mode)

    circuit = QuantumCircuit(oracle.num_qubits_uh, name="U_H")
    data = 0
    d_regs = list(range(1, 1 + n_d))
    m_regs = list(range(1 + n_d, 1 + n_d + n_m))
    idx_regs = list(range(1 + n_d + n_m, circuit.num_qubits))

    # Controlled-XOR O_c is self-inverse; dense UnitaryGate uses .inverse().
    if mode == "opaque":
        oc_inv = Gate("O_c_inv", len(d_regs + m_regs + idx_regs), [])
        circuit.append(oc_inv, d_regs + m_regs + idx_regs)
    else:
        circuit.append(oc.inverse(), d_regs + m_regs + idx_regs)
    apply_multiplexed_ry(circuit, data, d_regs, payload, alpha)
    circuit.z(data)
    circuit.append(ot, d_regs + m_regs)
    circuit.append(or_gate, d_regs + m_regs + idx_regs)

    return circuit, alpha


def hamiltonian_odata_gate_budget(
    oracle: HamiltonianOracleLabeling,
    *,
    optimization_level: int = 2,
    sk_t_count: bool = False,
) -> dict[str, object]:
    """Transpiled ``O_data`` budget for elastic ``H`` (``Im(H)`` payload)."""
    target = oracle.matrix.imag
    alpha = spectral_scale(target)
    if alpha < 1e-15:
        alpha = 1.0
    odata = data_loading_subcircuit(imag_payload(oracle.base), alpha)
    counts = transpiled_gate_counts(
        odata,
        optimization_level=optimization_level,
        sk_t_count=sk_t_count,
    )
    return {
        "alpha": alpha,
        "D_prime": oracle.base.d_prime,
        "D_padded": oracle.base.d_padded,
        **counts,
    }


def hamiltonian_gate_budget_report(
    oracle: HamiltonianOracleLabeling,
    *,
    transpile_uh: bool = False,
    materialize_lookup: bool | str = False,
    sk_t_count: bool = False,
    optimization_level: int = 2,
) -> dict[str, object]:
    """
    Gate budget for elastic ``U_H``.

    By default transpiles only ``O_data`` (fast). Set ``transpile_uh=True`` on
    tiny grids for full ``U_H`` T-counts (slow).
    """
    odata_budget = hamiltonian_odata_gate_budget(
        oracle,
        optimization_level=optimization_level,
        sk_t_count=sk_t_count,
    )
    circuit, alpha = build_hamiltonian_block_encoding_circuit(
        oracle, materialize_lookup=materialize_lookup
    )
    report: dict[str, object] = {
        "alpha": alpha,
        "n_qubits_uh": oracle.num_qubits_uh,
        "n_d_qubits": oracle.n_d_qubits,
        "n_m_qubits": oracle.n_m_qubits,
        "n_index_qubits": oracle.n_index_qubits,
        "O_data": {
            "depth": odata_budget["depth"],
            "size": odata_budget["size"],
            "t_gates": odata_budget["t_gates"],
            "two_qubit_gates": odata_budget["two_qubit_gates"],
        },
        "U_H_untranspiled": {
            "depth": circuit.depth(),
            "size": circuit.size(),
            "ops": dict(circuit.count_ops()),
        },
        "O_index_lookup": {
            "note": "Metadata only; set transpile_uh=True for full U_H T-counts (slow).",
            "O_c_gate_count": 1,
            "O_r_gate_count": 1,
            "O_t_gate_count": 1,
        },
    }
    if transpile_uh:
        uh_counts = transpiled_gate_counts(
            circuit,
            optimization_level=optimization_level,
            sk_t_count=sk_t_count,
        )
        report["U_H"] = uh_counts
    else:
        report["U_H"] = {
            "note": "Set transpile_uh=True on minimal grid for transpiled U_H metrics.",
        }
    return report


def _dense_unitary_gb(dim: int, *, bytes_per_complex: int = 16) -> float:
    """GB for a dense ``dim x dim`` complex matrix (default complex128; 1024^3 bytes)."""
    return float(dim) * float(dim) * float(bytes_per_complex) / (1024.0**3)


def summarize_uh_register_footprint(
    grid_sizes: tuple[tuple[int, int, int], ...] = ((2, 2, 2), (4, 2, 2), (4, 4, 2)),
    *,
    add_fractures: bool = False,
    dx: float = 0.05,
    bytes_per_complex: int = 16,
    dense_lookup_budget_gb: float = 1.0,
) -> pd.DataFrame:
    """
    Register / memory footprint for Pechan ``U_H`` without dense lookup synthesis.

    Motivates the ``2x2x2`` demo grid in Sections 7–11: opaque oracles and
    content checks stay cheap, but materializing ``O_c`` / ``O_r`` or the full
    ``U_H`` unitary grows as ``2^{n_d+n_m+n_idx}`` / ``2^{n_qubits_uh}``.
    """
    bcs = {"L": "DBC", "R": "DBC", "T": "DBC", "B": "DBC", "F": "DBC", "Ba": "DBC"}
    rows: list[dict[str, object]] = []
    for nx, ny, nz in grid_sizes:
        rho, compliance, _ = clinic_elastic_materials(
            nx, ny, nz, add_fractures=add_fractures
        )
        hamiltonian, *_ = FD_solver_3D_elastic(
            nx, ny, nz, dx, dx, dx, rho, compliance, bcs
        )
        layout = elastic_3d_layout(nx, ny, nz)
        labeling = label_coefficients_by_block(hamiltonian, layout, matrix_name="H")
        oracle = build_hamiltonian_oracle_labeling(labeling, hamiltonian)
        lookup_qubits = oracle.n_d_qubits + oracle.n_m_qubits + oracle.n_index_qubits
        lookup_dim = 2**lookup_qubits
        uh_dim = 2**oracle.num_qubits_uh
        system_dim = oracle.n_index_padded
        dense_oc_gb = _dense_unitary_gb(lookup_dim, bytes_per_complex=bytes_per_complex)
        dense_uh_gb = _dense_unitary_gb(uh_dim, bytes_per_complex=bytes_per_complex)
        dense_expm_gb = _dense_unitary_gb(system_dim, bytes_per_complex=bytes_per_complex)
        rows.append(
            {
                "nx": nx,
                "ny": ny,
                "nz": nz,
                "fractures": add_fractures,
                "N_s": layout.n_total,
                "D_prime": labeling.d_prime,
                "n_d": oracle.n_d_qubits,
                "n_m": oracle.n_m_qubits,
                "n_idx": oracle.n_index_qubits,
                "n_qubits_uh": oracle.num_qubits_uh,
                "n_system": int(np.log2(system_dim)),
                "lookup_dim": lookup_dim,
                "uh_dim": uh_dim,
                "system_dim": system_dim,
                "dense_Oc_GB": dense_oc_gb,
                "dense_UH_GB": dense_uh_gb,
                "dense_expm_GB": dense_expm_gb,
                "dense_lookup_ok": dense_oc_gb <= dense_lookup_budget_gb,
            }
        )
    return pd.DataFrame(rows)


def summarize_sparse_circuit_scaling(
    grid_sizes: tuple[tuple[int, int, int], ...] = ((2, 2, 2), (4, 2, 2), (4, 4, 2)),
    *,
    add_fractures: bool = False,
    dx: float = 0.05,
    decompose_reps: int = 1,
    reference_sv_seconds: float | None = 11.0,
) -> pd.DataFrame:
    """
    Real MCX / sparse-circuit cost vs grid (why Sections 7–11 stay on ``2x2x2``).

    Dense ``UnitaryGate`` RAM is a separate story (see
    ``summarize_uh_register_footprint``). Here the bottleneck is classical
    simulation of the *built* circuits: full ``Statevector`` evolution of
    ``O_c`` scales roughly like ``(MCX gates) x 2^{n_lookup}``. On a laptop,
    ``2x2x2`` is interactive (~10 s for a full ``O_c`` SV); ``4x2x2`` and
    larger stall for many minutes / hours.

    ``reference_sv_seconds`` calibrates extrapolated SV times from a measured
    ``2x2x2`` full-``O_c`` statevector run (set ``None`` to skip the estimate).
    """
    bcs = {"L": "DBC", "R": "DBC", "T": "DBC", "B": "DBC", "F": "DBC", "Ba": "DBC"}
    rows: list[dict[str, object]] = []
    ref_work: float | None = None
    for nx, ny, nz in grid_sizes:
        rho, compliance, _ = clinic_elastic_materials(
            nx, ny, nz, add_fractures=add_fractures
        )
        hamiltonian, *_ = FD_solver_3D_elastic(
            nx, ny, nz, dx, dx, dx, rho, compliance, bcs
        )
        layout = elastic_3d_layout(nx, ny, nz)
        labeling = label_coefficients_by_block(hamiltonian, layout, matrix_name="H")
        oracle = build_hamiltonian_oracle_labeling(labeling, hamiltonian)
        oc = build_column_oracle_circuit(oracle)
        or_circ = build_row_oracle_circuit(oracle)
        ot = build_transpose_oracle_circuit(oracle)
        uh, _ = build_hamiltonian_block_encoding_circuit(
            oracle, materialize_lookup="circuit"
        )
        uh_dec = uh.decompose(reps=decompose_reps)
        lookup_qubits = oracle.n_d_qubits + oracle.n_m_qubits + oracle.n_index_qubits
        lookup_dim = 2**lookup_qubits
        # Rough classical cost for evolving the full O_c circuit on |0...0>.
        oc_sv_work = float(oc.size()) * float(lookup_dim)
        uh_sv_work = float(uh_dec.size()) * float(2**oracle.num_qubits_uh)
        if ref_work is None:
            ref_work = oc_sv_work
        est_oc_sv_s = (
            None
            if reference_sv_seconds is None or ref_work is None or ref_work <= 0
            else float(reference_sv_seconds) * (oc_sv_work / ref_work)
        )
        rows.append(
            {
                "nx": nx,
                "ny": ny,
                "nz": nz,
                "fractures": add_fractures,
                "N_s": layout.n_total,
                "nnz": len(oracle.entries_by_dm),
                "n_d": oracle.n_d_qubits,
                "n_m": oracle.n_m_qubits,
                "n_idx": oracle.n_index_qubits,
                "n_lookup": lookup_qubits,
                "n_qubits_uh": oracle.num_qubits_uh,
                "lookup_dim": lookup_dim,
                "uh_dim": 2**oracle.num_qubits_uh,
                "O_c_size": oc.size(),
                "O_r_size": or_circ.size(),
                "O_t_size": ot.size(),
                "U_H_size": uh.size(),
                "U_H_size_decomposed": uh_dec.size(),
                "U_H_depth_decomposed": uh_dec.depth(),
                "O_c_sv_work": oc_sv_work,
                "U_H_sv_work": uh_sv_work,
                "est_O_c_sv_seconds": est_oc_sv_s,
            }
        )
    return pd.DataFrame(rows)


def summarize_uh_gate_scaling(
    grid_sizes: tuple[tuple[int, int, int], ...] = ((2, 2, 2), (4, 2, 2), (4, 4, 2)),
    *,
    add_fractures: bool = True,
    dx: float = 0.05,
    sk_t_count: bool = False,
    materialize_lookup: str = "circuit",
    decompose_reps: int = 1,
) -> pd.DataFrame:
    """
    Gate / depth table vs clinic DOF count ``N_s`` for elastic ``U_H``.

    Always reports transpiled ``O_data`` metrics and untranspiled ``U_H``
    depth/size from the assembled circuit. With ``materialize_lookup="circuit"``
    (default), also reports ``U_H_size_decomposed`` after one decomposition
    pass (real MCX index oracles exposed).
    """
    bcs = {"L": "DBC", "R": "DBC", "T": "DBC", "B": "DBC", "F": "DBC", "Ba": "DBC"}
    rows: list[dict[str, object]] = []
    for nx, ny, nz in grid_sizes:
        rho, compliance, _ = clinic_elastic_materials(
            nx, ny, nz, add_fractures=add_fractures
        )
        hamiltonian, *_ = FD_solver_3D_elastic(
            nx, ny, nz, dx, dx, dx, rho, compliance, bcs
        )
        layout = elastic_3d_layout(nx, ny, nz)
        labeling = label_coefficients_by_block(hamiltonian, layout, matrix_name="H")
        oracle = build_hamiltonian_oracle_labeling(labeling, hamiltonian)
        odata_budget = hamiltonian_odata_gate_budget(oracle, sk_t_count=sk_t_count)
        circuit, alpha = build_hamiltonian_block_encoding_circuit(
            oracle, materialize_lookup=materialize_lookup
        )
        uh_dec = circuit.decompose(reps=decompose_reps)
        rows.append(
            {
                "nx": nx,
                "ny": ny,
                "nz": nz,
                "N_s": layout.n_total,
                "D_prime": labeling.d_prime,
                "D_padded": labeling.d_padded,
                "n_qubits_uh": oracle.num_qubits_uh,
                "alpha_Im": alpha,
                "O_data_depth": odata_budget["depth"],
                "O_data_size": odata_budget["size"],
                "O_data_t_gates": odata_budget["t_gates"],
                "U_H_depth": circuit.depth(),
                "U_H_size": circuit.size(),
                "U_H_size_decomposed": uh_dec.size(),
                "U_H_depth_decomposed": uh_dec.depth(),
            }
        )
    return pd.DataFrame(rows)


def reconstructed_normalized_hamiltonian(
    oracle: HamiltonianOracleLabeling,
    alpha: float,
    *,
    use_imag: bool = True,
) -> np.ndarray:
    """Classical block ``Im(H)/alpha`` (or ``Re(H)/alpha``) from ``(d,m)`` maps."""
    n = oracle.n_grid
    block = np.zeros((n, n), dtype=float)
    for (d_label, _), (row, col, _) in oracle.entries_by_dm.items():
        value = oracle.base.value_table[d_label]
        payload = float((-1j * value).real) if use_imag else float(value.real)
        block[row, col] = payload / alpha
    return block


def _embed_odata_operator(payload: np.ndarray, alpha: float, n_d: int) -> np.ndarray:
    """Dense ``O_data`` on ``(data, d)`` only (small: ``2^{1+n_d}``)."""
    from qiskit.quantum_info import Operator

    circuit = data_loading_subcircuit(payload, alpha)
    if circuit.num_qubits < 1 + n_d:
        padded = QuantumCircuit(1 + n_d, name="O_data_pad")
        padded.compose(circuit, qubits=list(range(circuit.num_qubits)), inplace=True)
        circuit = padded
    return Operator(circuit).data


def odata_amplitude_errors(
    oracle: HamiltonianOracleLabeling,
    *,
    use_imag: bool = True,
) -> dict[str, float]:
    """
    Check multiplexed ``O_data`` loads ``|v_d|/alpha`` onto the data ``|0>`` amplitude.

    Matches ``_block_encoding_common.data_loading_subcircuit`` (``Ry`` with
    ``theta = 2 arccos(|v|/alpha)``).
    """
    payload = imag_payload(oracle.base) if use_imag else np.real(oracle.base.value_table)
    target = oracle.matrix.imag if use_imag else oracle.matrix.real
    alpha = spectral_scale(target)
    if alpha < 1e-15:
        alpha = 1.0
    odata = _embed_odata_operator(payload, alpha, oracle.n_d_qubits)
    max_err = 0.0
    for d_label, value in oracle.base.d_to_value.items():
        payload_d = float((-1j * value).real) if use_imag else float(value.real)
        expected = abs(payload_d) / alpha
        # Apply O_data to |0>|d> (data LSB)
        in_index = d_label << 1
        psi_in = np.zeros(odata.shape[0], dtype=complex)
        psi_in[in_index] = 1.0
        psi_out = odata @ psi_in
        measured = abs(psi_out[in_index])
        max_err = max(max_err, abs(measured - expected))
    return {"alpha": float(alpha), "max_amp_error": float(max_err)}


def compare_uh_to_classical(
    oracle: HamiltonianOracleLabeling,
    *,
    use_imag: bool = True,
) -> dict[str, float]:
    """
    Small-grid ``U_H`` content check (same strategy as 2D Laplacian ``verify_block_encoding``).

    Full dense unitaries are infeasible (``~15`` qubits on ``2x2x2``).  We verify:

    1. classical ``(d,m)->(i,j)`` reconstruction matches ``Im(H)/alpha``;
    2. staggered index-oracle chain ``O_r O_c |d m 0> = |d m i>``;
    3. ``O_data`` amplitudes encode ``|Im(H_d)|/alpha``.
    """
    target = oracle.matrix.imag if use_imag else oracle.matrix.real
    alpha = spectral_scale(target)
    classic = reconstructed_normalized_hamiltonian(oracle, alpha, use_imag=use_imag)
    truth = target / alpha
    oracle_check = verify_staggered_index_oracles(oracle)
    amp = odata_amplitude_errors(oracle, use_imag=use_imag)
    return {
        "alpha": float(alpha),
        "classical_recon_error": float(np.max(np.abs(classic - truth))),
        "index_oracle_ok": 1.0 if oracle_check["ok"] else 0.0,
        "odata_amp_error": amp["max_amp_error"],
        "num_qubits_uh": float(oracle.num_qubits_uh),
        "n_d_qubits": float(oracle.n_d_qubits),
        "n_m_qubits": float(oracle.n_m_qubits),
        "n_index_qubits": float(oracle.n_index_qubits),
    }


def build_minimal_hamiltonian_demo(
    nx: int = 2,
    ny: int = 2,
    nz: int = 2,
    *,
    add_fractures: bool = False,
    dx: float = 0.05,
) -> dict[str, object]:
    """Assemble clinic ``H`` plus labeling/oracles for a minimal grid demo."""
    bcs = {"L": "DBC", "R": "DBC", "T": "DBC", "B": "DBC", "F": "DBC", "Ba": "DBC"}
    rho, compliance, mask = clinic_elastic_materials(
        nx, ny, nz, add_fractures=add_fractures
    )
    hamiltonian, a_matrix, _b, _bs, _bi, b_inv_sqrt = FD_solver_3D_elastic(
        nx, ny, nz, dx, dx, dx, rho, compliance, bcs
    )
    layout = elastic_3d_layout(nx, ny, nz)
    labeling = label_coefficients_by_block(hamiltonian, layout, matrix_name="H")
    oracle = build_hamiltonian_oracle_labeling(labeling, hamiltonian)
    factors = label_material_and_coupling(a_matrix, b_inv_sqrt, layout)
    return {
        "layout": layout,
        "H": hamiltonian,
        "A": a_matrix,
        "B_inv_sqrt": b_inv_sqrt,
        "fracture_mask": mask,
        "labeling": labeling,
        "oracle": oracle,
        "factors": factors,
        "verify_H": verify_labeling(labeling, hamiltonian),
        "oracle_check": verify_staggered_index_oracles(oracle),
    }


# ---------------------------------------------------------------------------
# Assembled U_H content → structured time evolution (replace MatrixExponential)
# ---------------------------------------------------------------------------


def hamiltonian_from_oracles(oracle: HamiltonianOracleLabeling) -> np.ndarray:
    """
    Assemble dense ``H`` from block-encoding oracles (value table + ``(d,m)->(i,j)``).

    This is the system operator implied by ``U_H`` (clinic ``H`` is purely
    imaginary Hermitian, so entries are ``i * Im(H_ij)`` recovered from the
    real payload).
    """
    n = oracle.n_grid
    matrix = np.zeros((n, n), dtype=complex)
    for (d_label, _), (row, col, _) in oracle.entries_by_dm.items():
        matrix[row, col] = oracle.base.d_to_value[d_label]
    return matrix


def pad_to_qubit_register(matrix: np.ndarray) -> tuple[np.ndarray, int]:
    """Pad an ``N x N`` operator to ``2^n x 2^n`` for an ``n``-qubit register."""
    n = matrix.shape[0]
    dim = pad_to_power_of_two(n)
    if dim == n:
        return np.asarray(matrix, dtype=complex), int(np.log2(dim))
    padded = np.zeros((dim, dim), dtype=complex)
    padded[:n, :n] = matrix
    return padded, int(np.log2(dim))


def evolve_state_sparse(
    hamiltonian: sp.spmatrix | np.ndarray,
    psi0: np.ndarray,
    time: float,
) -> np.ndarray:
    """Structured classical evolution ``expm_multiply(-i H t, psi)`` (sparse-aware)."""
    h_op = sp.csr_matrix(hamiltonian, dtype=complex)
    psi = np.asarray(psi0, dtype=complex)
    if psi.shape[0] != h_op.shape[0]:
        raise ValueError("psi0 length must match Hamiltonian dimension.")
    return expm_multiply(-1j * h_op * time, psi)


def build_structured_evolution_circuit(
    hamiltonian: np.ndarray,
    psi0: np.ndarray,
    time: float,
) -> tuple[QuantumCircuit, int, np.ndarray]:
    """
    Qiskit evolution on the **system** register without ``SparsePauliOp`` +
    ``MatrixExponential``.

    Pads ``H`` to ``2^n``, builds ``U = exp(-i H_pad t)`` with SciPy, and wraps
    it as a single ``UnitaryGate`` (oracle-structured access → system unitary).
    """
    h_pad, n_qubits = pad_to_qubit_register(np.asarray(hamiltonian, dtype=complex))
    dim = h_pad.shape[0]
    psi = np.asarray(psi0, dtype=complex)
    if psi.shape[0] > dim:
        raise ValueError("psi0 longer than padded Hamiltonian.")
    psi_pad = np.zeros(dim, dtype=complex)
    psi_pad[: psi.shape[0]] = psi
    norm = np.linalg.norm(psi_pad)
    if norm < 1e-15:
        raise ValueError("psi0 has vanishing norm.")
    psi_pad = psi_pad / norm

    u_mat = expm(-1j * h_pad * time)
    circuit = QuantumCircuit(n_qubits, name="U_evol_structured")
    circuit.prepare_state(psi_pad, circuit.qubits)
    circuit.append(UnitaryGate(u_mat, check_input=False, label="exp(-iHt)"), circuit.qubits)
    return circuit, n_qubits, psi_pad


def evolve_state_qiskit_structured(
    hamiltonian: np.ndarray,
    psi0: np.ndarray,
    time: float,
) -> np.ndarray:
    """Statevector simulation of :func:`build_structured_evolution_circuit`."""
    circuit, _, _ = build_structured_evolution_circuit(hamiltonian, psi0, time)
    evolved = Statevector.from_instruction(circuit).data
    n = np.asarray(psi0).shape[0]
    return evolved[:n]


def matrix_exponential_pauli_stats(hamiltonian: np.ndarray) -> dict[str, object]:
    """
    Cost fingerprint of the clinic path: dense ``SparsePauliOp.from_operator``.

    Used only as a baseline to contrast against structured evolution.
    """
    h_pad, n_qubits = pad_to_qubit_register(np.asarray(hamiltonian, dtype=complex))
    pauli = SparsePauliOp.from_operator(Operator(h_pad))
    return {
        "n_qubits": n_qubits,
        "dim": h_pad.shape[0],
        "n_pauli_terms": len(pauli),
        "hamiltonian_nnz": int(np.count_nonzero(np.abs(h_pad) > 1e-12)),
    }


def evolve_with_assembled_uh(
    oracle: HamiltonianOracleLabeling,
    psi0: np.ndarray,
    time: float,
    *,
    backend: str = "structured_qiskit",
    include_pauli_baseline: bool = True,
) -> dict[str, object]:
    """
    Time evolution using Hamiltonian **assembled from ``U_H`` oracles**.

    Parameters
    ----------
    backend:
        ``\"sparse\"`` — ``expm_multiply`` on oracle ``H``;
        ``\"structured_qiskit\"`` — pad + ``UnitaryGate(expm(-iHt))`` (replaces
        ``MatrixExponential`` / Pauli synthesis).
    include_pauli_baseline:
        If True, densify padded ``H`` into ``SparsePauliOp`` for a cost fingerprint.
    """
    h_oracle = hamiltonian_from_oracles(oracle)
    alpha = spectral_scale(h_oracle.imag if np.max(np.abs(h_oracle.real)) < 1e-12 else h_oracle)
    stats = (
        matrix_exponential_pauli_stats(h_oracle)
        if include_pauli_baseline
        else {
            "n_qubits": int(np.ceil(np.log2(max(h_oracle.shape[0], 2)))),
            "dim": int(pad_to_power_of_two(h_oracle.shape[0])),
            "n_pauli_terms": -1,
            "hamiltonian_nnz": int(np.count_nonzero(np.abs(h_oracle) > 1e-12)),
        }
    )

    if backend == "sparse":
        psi_t = evolve_state_sparse(h_oracle, psi0, time)
        n_qubits = stats["n_qubits"]
        circuit = None
    elif backend == "structured_qiskit":
        circuit, n_qubits, _ = build_structured_evolution_circuit(h_oracle, psi0, time)
        psi_t = evolve_state_qiskit_structured(h_oracle, psi0, time)
    else:
        raise ValueError("backend must be 'sparse' or 'structured_qiskit'.")

    return {
        "psi_t": psi_t,
        "H_oracle": h_oracle,
        "alpha": alpha,
        "n_qubits": n_qubits,
        "circuit": circuit,
        "pauli_baseline": stats,
        "backend": backend,
    }


def compare_evolution_to_direct(
    oracle: HamiltonianOracleLabeling,
    hamiltonian_direct: sp.spmatrix | np.ndarray,
    psi0: np.ndarray,
    time: float,
    *,
    include_qiskit: bool = True,
    include_pauli_baseline: bool = True,
) -> dict[str, float]:
    """
    Compare oracle-assembled structured evolution vs direct sparse ``H`` evolution.

    On larger grids set ``include_qiskit=False`` to skip dense ``expm`` /
    ``UnitaryGate`` on the padded system register, and
    ``include_pauli_baseline=False`` to skip ``SparsePauliOp.from_operator``.
    """
    h_direct = dense_matrix(hamiltonian_direct)
    psi_direct = evolve_state_sparse(h_direct, psi0, time)
    sparse_path = evolve_with_assembled_uh(
        oracle,
        psi0,
        time,
        backend="sparse",
        include_pauli_baseline=include_pauli_baseline,
    )

    out: dict[str, float] = {
        "time": float(time),
        "oracle_vs_direct_H": float(np.max(np.abs(sparse_path["H_oracle"] - h_direct))),
        "sparse_oracle_vs_direct_psi": float(
            np.linalg.norm(sparse_path["psi_t"] - psi_direct)
        ),
        "n_system": float(oracle.n_grid),
        "hamiltonian_nnz": float(len(oracle.entries_by_dm)),
    }

    if include_qiskit:
        qiskit_path = evolve_with_assembled_uh(
            oracle,
            psi0,
            time,
            backend="structured_qiskit",
            include_pauli_baseline=include_pauli_baseline,
        )
        out["qiskit_vs_direct_psi"] = float(
            np.linalg.norm(qiskit_path["psi_t"] - psi_direct)
        )
        out["qiskit_vs_sparse_psi"] = float(
            np.linalg.norm(qiskit_path["psi_t"] - sparse_path["psi_t"])
        )
        out["n_qubits"] = float(qiskit_path["n_qubits"])
        if include_pauli_baseline:
            out["n_pauli_terms_baseline"] = float(
                qiskit_path["pauli_baseline"]["n_pauli_terms"]
            )
    elif include_pauli_baseline:
        stats = matrix_exponential_pauli_stats(sparse_path["H_oracle"])
        out["n_qubits"] = float(stats["n_qubits"])
        out["n_pauli_terms_baseline"] = float(stats["n_pauli_terms"])
    else:
        out["n_qubits"] = float(int(np.ceil(np.log2(max(oracle.n_grid, 2)))))

    return out


def run_larger_grid_oracle_and_sparse_evolution(
    grid_sizes: tuple[tuple[int, int, int], ...] = ((2, 2, 2), (4, 2, 2), (4, 4, 2)),
    *,
    add_fractures: bool = False,
    dx: float = 0.05,
    time: float = 1e-6,
    spot_checks: int = 8,
    include_qiskit: bool = False,
    include_pauli_baseline: bool = False,
    rng_seed: int = 0,
) -> pd.DataFrame:
    """
    Larger grid oracle correctness without full SV + sparse
    system-register evolution.

    For each grid:
      1. map / labeling checks;
      2. real MCX ``O_c``/``O_r``/``O_t`` structure + bit-level spot checks;
      3. ``U_H`` content (classical recon + ``O_data`` amps);
      4. ``expm_multiply(-i H t, psi)`` on oracle-assembled ``H`` vs clinic ``H``.

    Defaults skip Qiskit dense ``expm`` and Pauli baselines so ``4x4x2`` stays
    interactive on a laptop.
    """
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(rng_seed)
    for nx, ny, nz in grid_sizes:
        wall0 = time_mod.perf_counter()
        demo = build_minimal_hamiltonian_demo(
            nx, ny, nz, add_fractures=add_fractures, dx=dx
        )
        oracle = demo["oracle"]
        map_chk = demo["oracle_check"]
        circ_chk = verify_circuit_index_oracles(
            oracle, spot_checks=spot_checks, rng_seed=rng_seed
        )
        content = compare_uh_to_classical(oracle)
        n = oracle.n_grid
        psi0 = rng.normal(size=n) + 1j * rng.normal(size=n)
        psi0 /= np.linalg.norm(psi0)
        evo = compare_evolution_to_direct(
            oracle,
            demo["H"],
            psi0,
            time,
            include_qiskit=include_qiskit,
            include_pauli_baseline=include_pauli_baseline,
        )
        wall = time_mod.perf_counter() - wall0
        ok = (
            bool(map_chk["ok"])
            and bool(circ_chk["ok"])
            and content["classical_recon_error"] < 1e-10
            and content["odata_amp_error"] < 1e-10
            and evo["oracle_vs_direct_H"] < 1e-10
            and evo["sparse_oracle_vs_direct_psi"] < 1e-10
        )
        rows.append(
            {
                "nx": nx,
                "ny": ny,
                "nz": nz,
                "N_s": n,
                "nnz": len(oracle.entries_by_dm),
                "n_lookup": oracle.n_d_qubits
                + oracle.n_m_qubits
                + oracle.n_index_qubits,
                "n_qubits_uh": oracle.num_qubits_uh,
                "map_ok": bool(map_chk["ok"]),
                "circuit_ok": bool(circ_chk["ok"]),
                "sv_spot_used": bool(circ_chk["statevector_spot_checks_used"]),
                "O_c_size": circ_chk["oc_size"],
                "O_r_size": circ_chk["or_size"],
                "recon_err": content["classical_recon_error"],
                "odata_amp_err": content["odata_amp_error"],
                "H_err": evo["oracle_vs_direct_H"],
                "psi_err_sparse": evo["sparse_oracle_vs_direct_psi"],
                "wall_seconds": wall,
                "ok": ok,
            }
        )
    return pd.DataFrame(rows)
