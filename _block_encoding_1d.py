"""1D Poisson / Laplacian block encodings: periodic, Dirichlet, heterogeneous, clinic proxy.

This module implements structured block encodings for finite-difference operators on a
1D grid with ``N = 2^n`` points.  The homogeneous periodic Laplacian is encoded as a
linear combination of unitaries (LCU):

    L = 2 I - S_- - S_+,

where ``S_±`` are cyclic (modular) shift unitaries on the index register.  Heterogeneous
Poisson operators use Pechan-style relabeling ``d = d_ind || d_val`` together with the
shared data-loading oracle in ``_block_encoding_common``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Gate
from qiskit.circuit.library import MCXGate, StatePreparation, UnitaryGate
from qiskit.quantum_info import Operator

from _block_encoding_common import (
    data_loading_subcircuit,
    pad_to_power_of_two,
    spectral_scale,
    transpiled_gate_counts,
)


def laplacian_1d_periodic(n_grid: int) -> np.ndarray:
    """
    Discrete negative Laplacian ``(-Delta)`` with periodic BC and grid spacing ``h = 1``.

    Stencil on grid indices ``j in {0, ..., N-1}`` (with wrap-around):

        (L u)_j = 2 u_j - u_{j-1} - u_{j+1}.

    Matrix form: ``L = 2 I - S_- - S_+`` with cyclic shifts ``S_±`` (see
    ``permutation_unitary``).
    """
    lap = np.zeros((n_grid, n_grid), dtype=float)
    for j in range(n_grid):
        lap[j, j] = 2.0
        lap[j, (j - 1) % n_grid] = -1.0
        lap[j, (j + 1) % n_grid] = -1.0
    return lap


def laplacian_1d_dirichlet(n_grid: int) -> np.ndarray:
    """
    Discrete negative Laplacian ``(-Delta)`` with homogeneous Dirichlet BC.

    Interior stencil matches the periodic case; boundary rows have one neighbor only,
    so ``L_{0,0} = L_{N-1,N-1} = 1`` and ``L_{j,j} = 2`` for interior ``j``.
    """
    lap = np.zeros((n_grid, n_grid), dtype=float)
    for j in range(n_grid):
        diag = 0.0
        if j > 0:
            lap[j, j - 1] = -1.0
            diag += 1.0
        if j < n_grid - 1:
            lap[j, j + 1] = -1.0
            diag += 1.0
        lap[j, j] = diag
    return lap


def _log2_grid_qubits(n_grid: int) -> int:
    n_qubits = int(np.ceil(np.log2(n_grid)))
    if n_grid != 2**n_qubits:
        raise ValueError(f"n_grid must be a power of 2 (got {n_grid}).")
    return n_qubits


def permutation_unitary(n_qubits: int, n_grid: int, delta: int) -> np.ndarray:
    """
    Cyclic shift unitary on the active grid (reference dense implementation).

    For basis states ``|j>`` with ``0 <= j < n_grid``,

        S_delta |j> = |(j + delta) mod n_grid>,

    and padding states ``n_grid <= j < 2^n`` are left unchanged.  ``delta = +1``
    gives ``S_+`` (right shift); ``delta = -1`` gives ``S_-`` (left shift).
    """
    dim = 2**n_qubits
    unitary = np.zeros((dim, dim), dtype=complex)
    # Active grid: cyclic permutation on |0>,...,|N-1>.
    for j in range(n_grid):
        unitary[(j + delta) % n_grid, j] = 1.0
    # Padding: identity on unused basis states |N>,...,|2^n-1>.
    for j in range(n_grid, dim):
        unitary[j, j] = 1.0
    return unitary


def increment_gate(n_qubits: int, *, label: str = "INC") -> Gate:
    """
    Modular binary increment on ``n_qubits`` (implements ``S_+`` when ``n_grid = 2^n``).

    Acts as ``|j> -> |(j+1) mod 2^n>`` on the computational basis.  Built from
    ``O(n)`` controlled-X gates (Sunderhauf Sec. 3.2, circulant remark).
    """
    circuit = QuantumCircuit(n_qubits, name=label)
    # Flip qubit t when all lower bits are 1.
    for target in range(n_qubits):
        controls = list(range(target))
        if controls:
            circuit.mcx(controls, target)
        else:
            circuit.x(target)
    return circuit.to_gate(label=label)


def decrement_gate(n_qubits: int, *, label: str = "DEC") -> Gate:
    """Modular binary decrement ``S_- = S_+^dagger`` on ``n_qubits``."""
    circuit = QuantumCircuit(n_qubits, name=label)
    inc = QuantumCircuit(n_qubits)
    for target in range(n_qubits):
        controls = list(range(target))
        if controls:
            inc.mcx(controls, target)
        else:
            inc.x(target)
    circuit.append(inc.to_gate(label="INC").inverse(), circuit.qubits)
    return circuit.to_gate(label=label)


def shift_gate(
    n_qubits: int,
    n_grid: int,
    delta: int,
    *,
    use_adders: bool,
) -> Gate | np.ndarray:
    """
    Shift by ``delta`` on the active grid.

    When ``use_adders=True`` and ``n_grid = 2^n``, returns structured ``increment_gate``
    / ``decrement_gate`` for ``delta = pm 1``.  Otherwise returns the dense
    ``permutation_unitary`` matrix.
    """
    if use_adders and n_grid == 2**n_qubits and delta in (-1, 1):
        # Structured path: O(log N) gates, no N x N matrix synthesis.
        if delta == 1:
            return increment_gate(n_qubits)
        return decrement_gate(n_qubits)
    # Fallback: explicit 2^n x 2^n permutation (clnic reference / small-n only).
    return permutation_unitary(n_qubits, n_grid, delta)


def build_org_dirichlet_circuit(
    n_qubits: int,
    n_grid: int,
    delta: int,
) -> QuantumCircuit:
    """
    QLS paper-style out-of-bounds oracle ``O_rg`` (Appendix E).

    Ancilla ``|del>`` is flipped when a shift by ``delta`` would leave the physical
    domain ``[0, n_grid - 1]``: for ``delta = +1`` when ``j = n_grid - 1``, for
    ``delta = -1`` when ``j = 0``.  Used inside Dirichlet block encodings to mark
    invalid index updates.
    """
    if delta not in (-1, 1):
        raise ValueError("Dirichlet O_rg prototype supports delta = ±1 only.")

    circuit = QuantumCircuit(n_qubits + 1, name="O_rg")
    index = list(range(n_qubits))
    del_qubit = n_qubits

    if delta == 1:
        control_state = format(n_grid - 1, f"0{n_qubits}b")
    else:
        control_state = "0" * n_qubits

    circuit.append(MCXGate(n_qubits, ctrl_state=control_state), index + [del_qubit])
    return circuit


def dense_block_encoding(matrix: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Standard dilated unitary block encoding (small-``n`` reference only).

    Returns ``(U, alpha)`` with ``alpha = ||A||_2`` and top-left block of ``U``
    equal to ``A / alpha``.  Not scalable; use structured LCU / oracle encodings
    for large grids.
    """
    alpha = spectral_scale(matrix)
    normalized = matrix / alpha
    n_grid = matrix.shape[0]
    complement = np.eye(n_grid) - normalized @ normalized.T
    eigvals, eigvecs = np.linalg.eigh(complement)
    eigvals = np.maximum(eigvals, 0.0)
    sqrt_complement = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
    block = np.block(
        [
            [normalized, sqrt_complement],
            [sqrt_complement, -normalized],
        ]
    )
    return block, alpha


def lcu_prepare_amplitudes(coefficients: list[float]) -> tuple[np.ndarray, float]:
    """
    PREPARE amplitudes for an LCU with real coefficients ``c_k``.

    The LCU scale is the **1-norm** ``beta = sum_k |c_k|`` (not the spectral norm).
    Amplitudes ``a_k = sqrt(|c_k| / beta)`` satisfy ``sum_k |a_k|^2 = 1``, so
    ``PREPARE |0> = sum_k a_k |k>`` is a valid quantum state.  Signs of negative
    ``c_k`` are absorbed into SELECT (see ``_append_shift_term``).

    After the PREPARE--SELECT--PREPARE dagger sandwich, the system register sees
    ``(sum_k c_k U_k) / beta``.
    """
    # 1-norm: smallest beta such that |c_k|/beta are valid squared amplitudes.
    beta = sum(abs(c) for c in coefficients)
    n_terms = len(coefficients)
    # Pad ancilla to the next power of two (unused basis states stay amplitude 0).
    n_term_qubits = int(np.ceil(np.log2(n_terms)))
    dim = 2**n_term_qubits
    amplitudes = np.zeros(dim, dtype=float)
    for idx, coeff in enumerate(coefficients):
        # |a_k|^2 = |c_k| / beta  =>  a_k = sqrt(|c_k| / beta)
        amplitudes[idx] = np.sqrt(abs(coeff) / beta)
    # Guard against floating-point drift; StatePreparation expects unit norm.
    amplitudes /= np.linalg.norm(amplitudes)
    return amplitudes, beta


def _append_shift_term(
    select: QuantumCircuit,
    term_qubits: list,
    index_qubits: list,
    term_index: int,
    n_term_qubits: int,
    shift: Gate | np.ndarray,
    *,
    negative: bool,
) -> None:
    """Add one controlled term ``|term_index><term_index| ⊗ tilde{U}`` to SELECT."""
    # Ancilla control pattern for basis state |term_index>.
    control_state = format(term_index, f"0{n_term_qubits}b")
    if isinstance(shift, Gate):
        matrix = Operator(shift).data
    else:
        matrix = shift
    # Negative LCU coefficient c_k < 0  =>  apply -U_k instead of absorbing sign in PREPARE.
    if negative:
        matrix = -matrix
    select.append(
        UnitaryGate(matrix, check_input=False).control(
            n_term_qubits, ctrl_state=control_state
        ),
        term_qubits + index_qubits,
    )


def build_laplacian_block_encoding_circuit(
    n_grid: int,
    *,
    use_adders: bool = True,
    boundary: str = "periodic",
) -> tuple[QuantumCircuit, np.ndarray, float]:
    """
    Block encoding of ``L / alpha`` via PREPARE + SELECT + PREPARE dagger.

    Periodic Laplacian decomposition:

        L = c_0 U_0 + c_1 U_1 + c_2 U_2,
        c = (2, -1, -1),  U = (I, S_-, S_+).

    LCU scale ``alpha = beta = sum_k |c_k| = 4``.  Circuit:

        U_L = PREPARE^dag SELECT PREPARE,

    with ``SELECT = sum_k |k><k| ⊗ tilde{U}_k`` and ``tilde{U}_k = sign(c_k) U_k``.
    The top-left block on the index register (ancilla in ``|0>``) equals ``L / alpha``.

    Parameters
    ----------
    use_adders
        If True, implement ``S_±`` as modular increment/decrement gates.
    boundary
        Only ``"periodic"`` is supported in this builder.
    """
    if boundary != "periodic":
        raise ValueError("Homogeneous LCU builder supports periodic BC only.")

    n_qubits = _log2_grid_qubits(n_grid)
    identity = np.eye(2**n_qubits, dtype=complex)
    # S_- and S_+: either structured adders or dense permutation matrices.
    shift_down = shift_gate(n_qubits, n_grid, -1, use_adders=use_adders)
    shift_up = shift_gate(n_qubits, n_grid, 1, use_adders=use_adders)

    # L = 2*I + (-1)*S_- + (-1)*S_+  =>  beta = |2|+|-1|+|-1| = 4.
    coefficients = [2.0, -1.0, -1.0]
    prepare_amplitudes, beta = lcu_prepare_amplitudes(coefficients)

    n_term_qubits = int(np.ceil(np.log2(len(coefficients))))
    circuit = QuantumCircuit(n_term_qubits + n_qubits, name="U_L_periodic")
    term_qubits = circuit.qubits[:n_term_qubits]
    index_qubits = circuit.qubits[n_term_qubits:]

    # PREPARE: load sqrt(|c_k|/beta) on the term ancilla.
    circuit.append(StatePreparation(prepare_amplitudes), term_qubits)
    select = QuantumCircuit(n_term_qubits + n_qubits, name="SELECT")
    # Term k=0: +2 * I  (coefficient positive).
    _append_shift_term(
        select,
        list(term_qubits),
        list(index_qubits),
        0,
        n_term_qubits,
        identity,
        negative=False,
    )
    # Term k=1: -1 * S_-  (sign via negative=True).
    _append_shift_term(
        select,
        list(term_qubits),
        list(index_qubits),
        1,
        n_term_qubits,
        shift_down,
        negative=True,
    )
    # Term k=2: -1 * S_+.
    _append_shift_term(
        select,
        list(term_qubits),
        list(index_qubits),
        2,
        n_term_qubits,
        shift_up,
        negative=True,
    )
    circuit.compose(select, circuit.qubits, inplace=True)
    # PREPARE dagger: uncompute ancilla so the block appears on |0...0> term states.
    circuit.append(StatePreparation(prepare_amplitudes).inverse(), term_qubits)
    # beta is the LCU 1-norm scale; returned as alpha for verify_block_encoding.
    return circuit, laplacian_1d_periodic(n_grid), beta


def _basis_index(bit_lists: list[list[int]]) -> int:
    bits: list[int] = []
    for chunk in bit_lists:
        bits.extend(chunk)
    return sum(bit << pos for pos, bit in enumerate(bits))


def _index_bits(index: int, n_qubits: int) -> list[int]:
    return [(index >> bit) & 1 for bit in range(n_qubits)]


def extracted_block(
    unitary: np.ndarray,
    n_grid: int,
    n_term_qubits: int,
    n_index_qubits: int,
) -> np.ndarray:
    """Top-left ``n_grid x n_grid`` block of an LCU unitary (ancilla term register in ``|0>``)."""
    block = np.zeros((n_grid, n_grid), dtype=complex)
    dim = 2 ** (n_term_qubits + n_index_qubits)
    term_zero = [0] * n_term_qubits
    for col in range(n_grid):
        # Input: |0>_ancilla |col>_index
        input_vector = np.zeros(dim, dtype=complex)
        index_bits = _index_bits(col, n_index_qubits)
        input_vector[_basis_index([term_zero, index_bits])] = 1.0
        output_vector = unitary @ input_vector
        # Read system components still in |0>_ancilla.
        for row in range(n_grid):
            row_bits = _index_bits(row, n_index_qubits)
            block[row, col] = output_vector[_basis_index([term_zero, row_bits])]
    return block


def verify_block_encoding(
    circuit: QuantumCircuit,
    laplacian: np.ndarray,
    scale: float,
    *,
    atol: float = 1e-9,
) -> float:
    """
    Max entrywise error ``|| extracted_block(U) - A/alpha ||_max``.

    Simulates the full LCU unitary (feasible only for small ``n_grid``).
    """
    n_grid = laplacian.shape[0]
    n_index_qubits = _log2_grid_qubits(n_grid)
    n_term_qubits = int(np.ceil(np.log2(3)))
    unitary = Operator(circuit).data
    block = extracted_block(unitary, n_grid, n_term_qubits, n_index_qubits)
    return float(np.max(np.abs(block - laplacian / scale)))


# --- Pechan relabeling (1D) ---


def section_index_1d(row: int, col: int) -> int:
    """Pechan section label: 0 = diagonal, 1 = right neighbor, 2 = left neighbor."""
    if row == col:
        return 0
    if col == row + 1:
        return 1
    if col == row - 1:
        return 2
    raise ValueError(f"({row}, {col}) is not a nearest-neighbor pair.")


@dataclass(frozen=True)
class PechanLabeling1D:
    """Relabeled sparse matrix data for 1D Pechan-style block encoding."""

    n_grid: int
    matrix: np.ndarray
    d_padded: int
    value_table: np.ndarray
    d_ind_map: dict[int, tuple[int, int]]
    d_to_value: dict[int, float]
    entries: list[tuple[int, int, int, int, float]]

    @property
    def n_index_qubits(self) -> int:
        return _log2_grid_qubits(self.n_grid)

    @property
    def n_d_qubits(self) -> int:
        return int(np.log2(self.d_padded))


def permeability_field_1d_two_material(
    n_grid: int,
    *,
    k_rock: float = 1.0,
    k_fracture: float = 5.0,
    fracture_start: int | None = None,
) -> np.ndarray:
    """Piecewise-constant permeability: rock on ``[0, fracture_start)``, fracture elsewhere."""
    field = np.full(n_grid, k_rock, dtype=float)
    if fracture_start is None:
        fracture_start = n_grid // 2
    field[fracture_start:] = k_fracture
    return field


def poisson_matrix_1d(
    permeability: np.ndarray,
    *,
    periodic: bool = False,
    dx: float = 1.0,
) -> np.ndarray:
    """
    Variable-coefficient 1D Poisson operator ``-d/dx (k(x) d/dx)`` (FD stencil).

    Off-diagonal conductance at edge ``(j, j+1)`` is
    ``-0.5 (k_j + k_{j+1}) / dx``; rows sum to zero.
    """
    n_grid = len(permeability)
    matrix = np.zeros((n_grid, n_grid), dtype=float)
    for j in range(n_grid):
        neighbors: list[tuple[int, float]] = []
        if periodic:
            offsets = (-1, 1)
        else:
            offsets = tuple(offset for offset in (-1, 1) if 0 <= j + offset < n_grid)
        for offset in offsets:
            neighbor = (j + offset) % n_grid if periodic else j + offset
            conductance = -0.5 * (permeability[j] + permeability[neighbor]) / dx
            matrix[j, neighbor] = conductance
            neighbors.append((neighbor, conductance))
        matrix[j, j] = -sum(value for _, value in neighbors)
    return matrix


def pechan_relabel_1d(matrix: np.ndarray) -> PechanLabeling1D:
    """
    Relabel nonzero entries as ``d = d_ind || d_val`` (Pechan et al.).

    ``d_ind`` encodes the stencil section (diag / right / left); ``d_val`` indexes
    distinct magnitudes within each section.  Returns padded value table for ``O_data``.
    """
    n_grid = matrix.shape[0]
    section_values: dict[int, set[float]] = {0: set(), 1: set(), 2: set()}
    raw_entries: list[tuple[int, int, int, float]] = []

    for row in range(n_grid):
        for col in range(n_grid):
            value = float(matrix[row, col])
            if np.isclose(value, 0.0):
                continue
            section = section_index_1d(row, col)
            rounded = float(np.round(value, 12))
            section_values[section].add(rounded)
            raw_entries.append((row, col, section, rounded))

    d_val_bits = max(
        1, int(np.ceil(np.log2(max(len(section_values[s]) for s in (0, 1, 2)))))
    )
    section_value_order = {
        section: {value: idx for idx, value in enumerate(sorted(section_values[section]))}
        for section in (0, 1, 2)
    }

    d_to_value: dict[int, float] = {}
    d_ind_map: dict[int, tuple[int, int]] = {}
    entries: list[tuple[int, int, int, int, float]] = []

    for row, col, section, value in raw_entries:
        d_val = section_value_order[section][value]
        d_label = (section << d_val_bits) | d_val
        d_to_value[d_label] = value
        d_ind_map[d_label] = (section, d_val)
        entries.append((row, col, section, d_label, value))

    d_padded = pad_to_power_of_two(max(d_to_value) + 1)
    value_table = np.zeros(d_padded, dtype=float)
    for d_label, value in d_to_value.items():
        value_table[d_label] = value

    return PechanLabeling1D(
        n_grid=n_grid,
        matrix=matrix,
        d_padded=d_padded,
        value_table=value_table,
        d_ind_map=d_ind_map,
        d_to_value=d_to_value,
        entries=entries,
    )


def summarize_labeling_1d(labeling: PechanLabeling1D) -> dict[str, object]:
    """Human-readable counts: ``D'``, padded ``D``, section sizes, and ``alpha``."""
    section_names = {0: "diagonal", 1: "right", 2: "left"}
    by_section: dict[str, list[float]] = {section_names[s]: [] for s in (0, 1, 2)}
    seen: set[tuple[int, int]] = set()
    for d_label, value in labeling.d_to_value.items():
        section, d_val = labeling.d_ind_map[d_label]
        key = (section, d_val)
        if key in seen:
            continue
        seen.add(key)
        by_section[section_names[section]].append(value)
    return {
        "N": labeling.n_grid,
        "D_prime": len(labeling.d_to_value),
        "D_padded": labeling.d_padded,
        "D_init_sections": {k: len(v) for k, v in by_section.items()},
        "spectral_norm": spectral_scale(labeling.matrix),
        "value_table": labeling.value_table.copy(),
    }


def gate_budget_1d(labeling: PechanLabeling1D) -> dict[str, object]:
    """Transpiled gate counts for ``O_data`` at scale ``alpha = ||A||_2``."""
    alpha = spectral_scale(labeling.matrix)
    odata = data_loading_subcircuit(labeling.value_table, alpha)
    return {
        "alpha": alpha,
        "D_prime": len(labeling.d_to_value),
        "D_padded": labeling.d_padded,
        "O_data": transpiled_gate_counts(odata),
    }


def clinic_elastic_stiffness_1d(
    n_grid: int,
    *,
    rho_base: float = 2700.0,
    rho_fracture: float = 1000.0,
    lambda_base: float = 2.16e10,
    mu_base: float = 8.1e9,
    lambda_fracture: float = 2.2e9,
    mu_fracture: float = 1e-6,
    fracture_start: int | None = None,
    dx: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    1D clinic proxy: block-encode ``-d/dx (E(x) d/dx)`` with ``E = lambda + 2 mu``.

    Uses rock/fracture moduli from the clinic defaults.  Full 3D elastic
    Hamiltonian assembly remains future work.
    """
    e_rock = lambda_base + 2.0 * mu_base
    e_fracture = lambda_fracture + 2.0 * mu_fracture
    stiffness = permeability_field_1d_two_material(
        n_grid,
        k_rock=e_rock,
        k_fracture=e_fracture,
        fracture_start=fracture_start,
    )
    matrix = poisson_matrix_1d(stiffness, periodic=False, dx=dx)
    return matrix, stiffness


def compare_shift_implementations(n_grid: int) -> dict[str, int]:
    """
    Transpiled two-qubit gate counts: dense ``UnitaryGate`` shifts vs modular adders.

    Both implement the same ``S_±``; this compares **circuit structure** cost.
    At small ``N`` the transpiler may fold them to similar counts — the adder path
    is the scalable one for large grids.
    """
    dense, _, _ = build_laplacian_block_encoding_circuit(n_grid, use_adders=False)
    adder, _, _ = build_laplacian_block_encoding_circuit(n_grid, use_adders=True)
    return {
        "dense_two_qubit_gates": transpiled_gate_counts(dense)["two_qubit_gates"],
        "adder_two_qubit_gates": transpiled_gate_counts(adder)["two_qubit_gates"],
    }


def data_loading_subcircuit(values: np.ndarray, scale: float) -> QuantumCircuit:
    """
    Multiplexed ``R_y`` data-loading oracle ``O_data`` (Sunderhauf; QLS paper).

    Prepares ``|d>|0> -> |d>( cos(theta_d)|0> + sin(theta_d)|1> )`` with
    ``theta_d = arccos(|values[d]| / scale)``, so the ``|1>`` amplitude on the
    data qubit encodes ``values[d] / scale``.  Requires ``|values[d]| <= scale``.
    """
    if len(values) == 0:
        raise ValueError("values must be non-empty")
    if not np.all(np.abs(values) <= scale + 1e-12):
        raise ValueError("All |values| must be <= scale.")

    n_value_qubits = int(np.ceil(np.log2(len(values))))
    angles = 2.0 * np.arccos(np.clip(np.abs(values) / scale, 0.0, 1.0))

    circuit = QuantumCircuit(1 + n_value_qubits, name="O_data")
    data = 0
    value_regs = list(range(1, 1 + n_value_qubits))

    circuit.ry(float(angles[0]), data)
    if len(values) == 2 and n_value_qubits == 1:
        circuit.cry(float(angles[1] - angles[0]), value_regs[0], data)
        return circuit

    for value_index in range(1, len(values)):
        control_state = format(value_index, f"0{n_value_qubits}b")
        delta = float(angles[value_index] - angles[0])
        cry_step = QuantumCircuit(1)
        cry_step.ry(delta, 0)
        circuit.append(
            cry_step.to_gate(label=f"RY({value_index})").control(
                n_value_qubits, ctrl_state=control_state
            ),
            value_regs + [data],
        )

    return circuit