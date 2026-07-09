"""2D heterogeneous Poisson / Laplacian block encoding with Pechan relabeling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Gate
from qiskit.circuit.library import MCXGate, UnitaryGate
from qiskit.quantum_info import Operator

from _block_encoding_1d import decrement_gate, increment_gate
from _block_encoding_common import (
    apply_multiplexed_ry,
    data_loading_subcircuit,
    log2_dim,
    pad_to_power_of_two,
    power_of_two,
    spectral_scale,
    transpiled_gate_counts,
)


@dataclass(frozen=True)
class PechanLabeling:
    """Pechan-style relabeling d = d_ind || d_val for a symmetric sparse matrix."""

    nx: int
    ny: int
    matrix: np.ndarray
    d_ind_bits: int
    d_val_bits: int
    d_padded: int
    m_padded: int
    s_padded: int
    value_table: np.ndarray
    d_ind_map: dict[int, tuple[int, int]]
    d_to_value: dict[int, float]
    entries: list[tuple[int, int, int, int, float]]

    @property
    def n_grid(self) -> int:
        return self.nx * self.ny

    @property
    def n_index_qubits(self) -> int:
        return log2_dim(pad_to_power_of_two(self.n_grid))

    @property
    def n_d_qubits(self) -> int:
        return log2_dim(self.d_padded)

    @property
    def n_m_qubits(self) -> int:
        return log2_dim(self.m_padded)

    @property
    def d_prime(self) -> int:
        return len(self.d_to_value)


def section_index_2d(row: int, col: int, nx: int) -> int:
    """Stencil section: 00 diagonal, 01 x-neighbor, 10 y-neighbor."""
    row_a, row_b = divmod(row, nx)
    col_a, col_b = divmod(col, nx)
    if row == col:
        return 0
    if row_a == col_a:
        return 1
    if row_b == col_b:
        return 2
    raise ValueError(f"({row}, {col}) is not a nearest-neighbor pair on the grid.")


def permeability_field_two_material(
    nx: int,
    ny: int,
    *,
    k_rock: float = 1.0,
    k_fracture: float = 5.0,
    fracture_axis: str = "x",
    fracture_index: int | None = None,
) -> np.ndarray:
    """Piecewise-constant permeability with a single planar fracture."""
    power_of_two(nx, name="nx")
    power_of_two(ny, name="ny")
    field = np.full((ny, nx), k_rock, dtype=float)
    if fracture_index is None:
        fracture_index = nx // 2
    if fracture_axis == "x":
        field[:, fracture_index:] = k_fracture
    elif fracture_axis == "y":
        field[fracture_index:, :] = k_fracture
    else:
        raise ValueError("fracture_axis must be 'x' or 'y'.")
    return field


def poisson_matrix_2d_periodic(
    permeability: np.ndarray,
    *,
    dx: float = 1.0,
    dy: float = 1.0,
) -> np.ndarray:
    """
    Symmetric negative Laplacian -div(k grad u) on a periodic Nx x Ny grid.

    Face conductance between neighboring cells p and q uses the arithmetic mean
    0.5 * (k_p + k_q). The diagonal is the negative row sum.
    """
    ny, nx = permeability.shape
    n_grid = nx * ny
    matrix = np.zeros((n_grid, n_grid), dtype=float)

    for row_b in range(ny):
        for row_a in range(nx):
            row = row_a + row_b * nx
            neighbors = (
                ((row_a + 1) % nx, row_b, 1),
                ((row_a - 1) % nx, row_b, -1),
                (row_a, (row_b + 1) % ny, nx),
                (row_a, (row_b - 1) % ny, -nx),
            )
            diagonal = 0.0
            for col_a, col_b, _ in neighbors:
                col = col_a + col_b * nx
                conductance = -0.5 * (
                    permeability[row_b, row_a] + permeability[col_b, col_a]
                ) / (dx * dy)
                matrix[row, col] = conductance
                diagonal -= conductance
            matrix[row, row] = diagonal
    return matrix


def poisson_matrix_2d_dirichlet(
    permeability: np.ndarray,
    *,
    dx: float = 1.0,
    dy: float = 1.0,
) -> np.ndarray:
    """
    Symmetric negative Laplacian with homogeneous Dirichlet BC on the box boundary.

    Same face conductance as the periodic operator; boundary rows omit neighbors
    outside the domain (clinic-style box grid).
    """
    ny, nx = permeability.shape
    n_grid = nx * ny
    matrix = np.zeros((n_grid, n_grid), dtype=float)

    for row_b in range(ny):
        for row_a in range(nx):
            row = row_a + row_b * nx
            neighbors: list[tuple[int, int]] = []
            if row_a < nx - 1:
                neighbors.append((row_a + 1, row_b))
            if row_a > 0:
                neighbors.append((row_a - 1, row_b))
            if row_b < ny - 1:
                neighbors.append((row_a, row_b + 1))
            if row_b > 0:
                neighbors.append((row_a, row_b - 1))

            diagonal = 0.0
            for col_a, col_b in neighbors:
                col = col_a + col_b * nx
                conductance = -0.5 * (
                    permeability[row_b, row_a] + permeability[col_b, col_a]
                ) / (dx * dy)
                matrix[row, col] = conductance
                diagonal -= conductance
            matrix[row, row] = diagonal
    return matrix


def pechan_relabel(matrix: np.ndarray, nx: int, ny: int) -> PechanLabeling:
    """
    Assign Pechan labels d = d_ind || d_val to every nonzero entry.

    Distinct numeric values in different stencil sections receive different d
    labels even when the magnitude coincides.
    """
    power_of_two(nx, name="nx")
    power_of_two(ny, name="ny")
    n_grid = nx * ny
    if matrix.shape != (n_grid, n_grid):
        raise ValueError("matrix shape must be (nx*ny, nx*ny).")

    section_values: dict[int, set[float]] = {0: set(), 1: set(), 2: set()}
    raw_entries: list[tuple[int, int, int, float]] = []

    for row in range(n_grid):
        for col in range(n_grid):
            value = float(matrix[row, col])
            if np.isclose(value, 0.0):
                continue
            section = section_index_2d(row, col, nx)
            rounded = float(np.round(value, 12))
            section_values[section].add(rounded)
            raw_entries.append((row, col, section, rounded))

    d_ind_bits = 2
    max_section_values = max(len(section_values[s]) for s in (0, 1, 2))
    d_val_bits = int(np.ceil(np.log2(max(max_section_values, 1))))
    if d_val_bits == 0:
        d_val_bits = 1

    section_value_order: dict[int, dict[float, int]] = {}
    for section in (0, 1, 2):
        ordered = sorted(section_values[section])
        section_value_order[section] = {value: idx for idx, value in enumerate(ordered)}

    d_to_value: dict[int, float] = {}
    d_ind_map: dict[int, tuple[int, int]] = {}
    entries: list[tuple[int, int, int, int, float]] = []

    for row, col, section, value in raw_entries:
        d_val = section_value_order[section][value]
        d_label = (section << d_val_bits) | d_val
        d_to_value[d_label] = value
        d_ind_map[d_label] = (section, d_val)
        entries.append((row, col, section, d_label, value))

    d_prime = len(d_to_value)
    d_padded = pad_to_power_of_two(max(d_to_value) + 1)

    multiplicity: dict[int, int] = {}
    for _, _, _, d_label, _ in entries:
        multiplicity[d_label] = multiplicity.get(d_label, 0) + 1
    m_padded = pad_to_power_of_two(max(multiplicity.values()))

    s_natural = 5  # 2D 5-point stencil
    s_padded = pad_to_power_of_two(max(s_natural, d_padded))

    value_table = np.zeros(d_padded, dtype=float)
    for d_label, value in d_to_value.items():
        value_table[d_label] = value

    return PechanLabeling(
        nx=nx,
        ny=ny,
        matrix=matrix,
        d_ind_bits=d_ind_bits,
        d_val_bits=d_val_bits,
        d_padded=d_padded,
        m_padded=m_padded,
        s_padded=s_padded,
        value_table=value_table,
        d_ind_map=d_ind_map,
        d_to_value=d_to_value,
        entries=entries,
    )


def _entries_by_dm(labeling: PechanLabeling) -> dict[tuple[int, int], tuple[int, int, float]]:
    """Map (d, m) to (row, col, value) using lower-triangle then upper-triangle ordering."""
    by_d: dict[int, list[tuple[int, int, float]]] = {}
    for row, col, _, d_label, value in labeling.entries:
        by_d.setdefault(d_label, []).append((row, col, value))

    mapping: dict[tuple[int, int], tuple[int, int, float]] = {}
    for d_label, items in by_d.items():
        lower = [(row, col, value) for row, col, value in items if row >= col]
        upper = [(row, col, value) for row, col, value in items if row < col]
        ordered = lower + upper
        for m_label, (row, col, value) in enumerate(ordered):
            mapping[(d_label, m_label)] = (row, col, value)
    return mapping


def _index_bits(index: int, n_qubits: int) -> list[int]:
    return [(index >> bit) & 1 for bit in range(n_qubits)]


def _basis_index(bit_lists: list[list[int]]) -> int:
    bits: list[int] = []
    for chunk in bit_lists:
        bits.extend(chunk)
    return sum(bit << pos for pos, bit in enumerate(bits))


def _make_permutation_unitary(
    mapping: dict[int, int],
    dim: int,
    *,
    label: str,
) -> UnitaryGate:
    """Build a permutation unitary from a partial bijection on basis indices."""
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


def build_column_oracle_lookup(labeling: PechanLabeling) -> UnitaryGate:
    """Lookup O_c: |d>|m>|0> -> |d>|m>|j> for small grids."""
    n_d = labeling.n_d_qubits
    n_m = labeling.n_m_qubits
    n_idx = labeling.n_index_qubits
    dim = 2 ** (n_d + n_m + n_idx)
    mapping_entries = _entries_by_dm(labeling)

    permutation: dict[int, int] = {}
    for (d_label, m_label), (_, col, _) in mapping_entries.items():
        in_index = _basis_index(
            [_index_bits(d_label, n_d), _index_bits(m_label, n_m), _index_bits(0, n_idx)]
        )
        out_index = _basis_index(
            [_index_bits(d_label, n_d), _index_bits(m_label, n_m), _index_bits(col, n_idx)]
        )
        permutation[in_index] = out_index

    return _make_permutation_unitary(permutation, dim, label="O_c")


def build_row_oracle_lookup(labeling: PechanLabeling) -> UnitaryGate:
    """Lookup O_r: |d>|m>|j> -> |d>|m>|i>."""
    n_d = labeling.n_d_qubits
    n_m = labeling.n_m_qubits
    n_idx = labeling.n_index_qubits
    dim = 2 ** (n_d + n_m + n_idx)
    mapping_entries = _entries_by_dm(labeling)

    permutation: dict[int, int] = {}
    for (d_label, m_label), (row, col, _) in mapping_entries.items():
        in_index = _basis_index(
            [_index_bits(d_label, n_d), _index_bits(m_label, n_m), _index_bits(col, n_idx)]
        )
        out_index = _basis_index(
            [_index_bits(d_label, n_d), _index_bits(m_label, n_m), _index_bits(row, n_idx)]
        )
        permutation[in_index] = out_index

    return _make_permutation_unitary(permutation, dim, label="O_r")


def build_transpose_oracle_pechan(labeling: PechanLabeling) -> UnitaryGate:
    """
    Pechan / Sünderhauf O_t on |d>|m>.

    d_ind = 00 (diagonal): identity on m; off-diagonal sections flip m_hi.
    """
    n_d = labeling.n_d_qubits
    n_m = labeling.n_m_qubits
    dim = 2 ** (n_d + n_m)
    permutation: dict[int, int] = {}

    for d_label in range(labeling.d_padded):
        if d_label not in labeling.d_ind_map:
            continue
        section, _ = labeling.d_ind_map[d_label]
        for m_label in range(labeling.m_padded):
            in_index = _basis_index(
                [_index_bits(d_label, n_d), _index_bits(m_label, n_m)]
            )
            if section == 0:
                out_m = m_label
            else:
                m_bits = _index_bits(m_label, n_m)
                m_bits[-1] ^= 1
                out_m = _basis_index([m_bits])
            out_index = _basis_index(
                [_index_bits(d_label, n_d), _index_bits(out_m, n_m)]
            )
            permutation[in_index] = out_index

    return _make_permutation_unitary(permutation, dim, label="O_t")


def _grid_qubit_split(nx: int, ny: int) -> tuple[int, int, int]:
    """Return (n_x_qubits, n_y_qubits, n_index_qubits) for power-of-two nx, ny."""
    power_of_two(nx, name="nx")
    power_of_two(ny, name="ny")
    n_x = int(np.log2(nx))
    n_y = int(np.log2(ny))
    return n_x, n_y, n_x + n_y


def build_org_dirichlet_2d_circuit(
    nx: int,
    ny: int,
    *,
    delta_x: int = 0,
    delta_y: int = 0,
) -> QuantumCircuit:
    """
    Out-of-bounds oracle ``O_rg`` on a 2D row-major index ``|x>|y>``.

    Ancilla ``|del>`` is set when a shift by ``(delta_x, delta_y)`` would leave the
    physical box ``[0, nx-1] x [0, ny-1]``.  Exactly one of ``delta_x``, ``delta_y``
    must be ``±1``.
    """
    n_x, n_y, n_idx = _grid_qubit_split(nx, ny)
    shifts = (delta_x, delta_y)
    if shifts not in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        raise ValueError("Exactly one of delta_x, delta_y must be ±1.")

    circuit = QuantumCircuit(n_idx + 1, name="O_rg_2d")
    x_regs = list(range(n_x))
    y_regs = list(range(n_x, n_idx))
    del_qubit = n_idx

    if delta_x == 1:
        ctrl = _register_control_state(nx - 1, n_x)
        circuit.append(MCXGate(n_x, ctrl_state=ctrl), x_regs + [del_qubit])
    elif delta_x == -1:
        circuit.append(MCXGate(n_x, ctrl_state="0" * n_x), x_regs + [del_qubit])
    elif delta_y == 1:
        ctrl = _register_control_state(ny - 1, n_y)
        circuit.append(MCXGate(n_y, ctrl_state=ctrl), y_regs + [del_qubit])
    elif delta_y == -1:
        circuit.append(MCXGate(n_y, ctrl_state="0" * n_y), y_regs + [del_qubit])
    return circuit


def _constant_add_gate(n_qubits: int, amount: int, *, label: str | None = None) -> Gate:
    """Modular addition of a non-negative constant on ``n_qubits`` (ripple carry)."""
    if amount < 0:
        raise ValueError("amount must be non-negative.")
    mod = 2**n_qubits
    amount %= mod
    if amount == 0:
        return QuantumCircuit(n_qubits, name=label or "ADD_0").to_gate(label=label or "ADD_0")

    circuit = QuantumCircuit(n_qubits, name=label or f"ADD_{amount}")
    inc = increment_gate(n_qubits)
    for _ in range(amount):
        circuit.append(inc, circuit.qubits)
    return circuit.to_gate(label=label or f"ADD_{amount}")


def _append_controlled_constant_add(
    circuit: QuantumCircuit,
    target_qubits: list[int],
    amount: int,
    control_qubits: list[int],
    control_state: str,
) -> None:
    """Apply ``|idx> -> |(idx + amount) mod 2^n>`` when controls match ``control_state``."""
    if amount == 0:
        return
    add_gate = _constant_add_gate(len(target_qubits), amount)
    circuit.append(
        add_gate.control(len(control_qubits), ctrl_state=control_state),
        control_qubits + target_qubits,
    )


def _register_control_state(value: int, n_bits: int) -> str:
    """Qiskit ``ctrl_state`` string for ``n_bits`` with qubit ``0`` as LSB."""
    bits = [(value >> bit) & 1 for bit in range(n_bits)]
    return "".join(str(bit) for bit in reversed(bits))


def _joint_dm_control_state(d_label: int, m_label: int, n_d: int, n_m: int) -> str:
    """``ctrl_state`` for controls ``d_regs + m_regs`` (each register LSB-first)."""
    bits: list[str] = []
    for value, n_bits in ((d_label, n_d), (m_label, n_m)):
        for bit in range(n_bits):
            bits.append(str((value >> bit) & 1))
    return "".join(reversed(bits))


def build_column_oracle_arithmetic(labeling: PechanLabeling) -> Gate:
    """
    Arithmetic ``O_c``: controlled modular adds load column index ``j`` from ``|0>``.

    Replaces the dense lookup permutation with Sünderhauf-style ripple-carry adds.
    """
    n_d = labeling.n_d_qubits
    n_m = labeling.n_m_qubits
    n_idx = labeling.n_index_qubits
    d_regs = list(range(n_d))
    m_regs = list(range(n_d, n_d + n_m))
    idx_regs = list(range(n_d + n_m, n_d + n_m + n_idx))

    circuit = QuantumCircuit(n_d + n_m + n_idx, name="O_c_adder")
    for (d_label, m_label), (_, col, _) in _entries_by_dm(labeling).items():
        ctrl_state = _joint_dm_control_state(d_label, m_label, n_d, n_m)
        _append_controlled_constant_add(
            circuit,
            idx_regs,
            col,
            d_regs + m_regs,
            ctrl_state,
        )
    return circuit.to_gate(label="O_c_adder")


def build_row_oracle_arithmetic(labeling: PechanLabeling) -> Gate:
    """
    Arithmetic ``O_r``: add ``(row - col)`` to the index register holding ``j``.

    Composes with ``O_c`` so ``|i>`` is reached from ``|0>`` via column then row offset.
    """
    n_d = labeling.n_d_qubits
    n_m = labeling.n_m_qubits
    n_idx = labeling.n_index_qubits
    n_grid_padded = 2**n_idx
    d_regs = list(range(n_d))
    m_regs = list(range(n_d, n_d + n_m))
    idx_regs = list(range(n_d + n_m, n_d + n_m + n_idx))

    circuit = QuantumCircuit(n_d + n_m + n_idx, name="O_r_adder")
    for (d_label, m_label), (row, col, _) in _entries_by_dm(labeling).items():
        delta = (row - col) % n_grid_padded
        ctrl_state = _joint_dm_control_state(d_label, m_label, n_d, n_m)
        _append_controlled_constant_add(
            circuit,
            idx_regs,
            delta,
            d_regs + m_regs,
            ctrl_state,
        )
    return circuit.to_gate(label="O_r_adder")


def build_transpose_oracle_arithmetic(labeling: PechanLabeling) -> Gate:
    """Pechan ``O_t`` via controlled flip of the high ``m`` bit for off-diagonal sections."""
    n_d = labeling.n_d_qubits
    n_m = labeling.n_m_qubits
    d_regs = list(range(n_d))
    m_msb = n_d + n_m - 1

    circuit = QuantumCircuit(n_d + n_m, name="O_t_adder")
    for d_label, (section, _) in labeling.d_ind_map.items():
        if section == 0:
            continue
        ctrl_state = _register_control_state(d_label, n_d)
        circuit.append(MCXGate(n_d, ctrl_state=ctrl_state), d_regs + [m_msb])
    return circuit.to_gate(label="O_t_adder")


def shift_gate_2d_x(nx: int, ny: int, delta: int, *, use_adders: bool = True) -> Gate:
    """Shift row-major index by ``delta`` in the x direction (lower ``log2(nx)`` qubits)."""
    if delta not in (-1, 1):
        raise ValueError("delta must be ±1.")
    n_x, _, n_idx = _grid_qubit_split(nx, ny)
    if use_adders:
        sub = increment_gate(n_x, label="INC_x") if delta == 1 else decrement_gate(n_x, label="DEC_x")
        circuit = QuantumCircuit(n_idx, name=f"S_x_{delta:+d}")
        circuit.append(sub, list(range(n_x)))
        return circuit.to_gate(label=f"S_x_{delta:+d}")

    dense = np.eye(2**n_idx, dtype=complex)
    for y in range(ny):
        for x in range(nx):
            idx = x + y * nx
            if delta == 1:
                new_x = (x + 1) % nx
            else:
                new_x = (x - 1) % nx
            new_idx = new_x + y * nx
            dense[new_idx, idx] = 1.0
            dense[idx, idx] = 0.0
    for idx in range(nx * ny, 2**n_idx):
        dense[idx, idx] = 1.0
    return UnitaryGate(dense, check_input=False, label=f"S_x_{delta:+d}")


def shift_gate_2d_y(nx: int, ny: int, delta: int, *, use_adders: bool = True) -> Gate:
    """Shift row-major index by ``delta`` in the y direction (add ``± nx``)."""
    if delta not in (-1, 1):
        raise ValueError("delta must be ±1.")
    _, _, n_idx = _grid_qubit_split(nx, ny)
    offset = nx if delta == 1 else (2**n_idx - nx)
    if use_adders:
        return _constant_add_gate(n_idx, offset, label=f"S_y_{delta:+d}")
    dense = np.eye(2**n_idx, dtype=complex)
    for y in range(ny):
        for x in range(nx):
            idx = x + y * nx
            new_y = (y + delta) % ny
            new_idx = x + new_y * nx
            dense[new_idx, idx] = 1.0
            dense[idx, idx] = 0.0
    for idx in range(nx * ny, 2**n_idx):
        dense[idx, idx] = 1.0
    return UnitaryGate(dense, check_input=False, label=f"S_y_{delta:+d}")


def _index_oracle_builders(use_adders: bool):
    if use_adders:
        return (
            build_column_oracle_arithmetic,
            build_row_oracle_arithmetic,
            build_transpose_oracle_arithmetic,
        )
    return (
        build_column_oracle_lookup,
        build_row_oracle_lookup,
        build_transpose_oracle_pechan,
    )


def verify_index_oracle_equivalence(
    labeling: PechanLabeling,
    *,
    atol: float = 1e-9,
) -> dict[str, float]:
    """Compare lookup vs arithmetic oracles on the Pechan (d, m) basis states they act on."""
    n_d = labeling.n_d_qubits
    n_m = labeling.n_m_qubits
    n_idx = labeling.n_index_qubits
    mapping = _entries_by_dm(labeling)

    oc_lookup = Operator(build_column_oracle_lookup(labeling)).data
    oc_adder = Operator(build_column_oracle_arithmetic(labeling)).data
    or_lookup = Operator(build_row_oracle_lookup(labeling)).data
    or_adder = Operator(build_row_oracle_arithmetic(labeling)).data
    ot_lookup = Operator(build_transpose_oracle_pechan(labeling)).data
    ot_adder = Operator(build_transpose_oracle_arithmetic(labeling)).data

    def basis_vector(*bit_groups: list[int]) -> np.ndarray:
        dim = 2 ** sum(len(group) for group in bit_groups)
        vector = np.zeros(dim, dtype=complex)
        vector[_basis_index(list(bit_groups))] = 1.0
        return vector

    errors: dict[str, float] = {"O_c": 0.0, "O_r": 0.0, "O_t": 0.0}

    for (d_label, m_label), (row, col, _) in mapping.items():
        d_bits = _index_bits(d_label, n_d)
        m_bits = _index_bits(m_label, n_m)

        in_c = basis_vector(d_bits, m_bits, _index_bits(0, n_idx))
        errors["O_c"] = max(
            errors["O_c"],
            float(np.linalg.norm(oc_lookup @ in_c - oc_adder @ in_c)),
        )

        in_r = basis_vector(d_bits, m_bits, _index_bits(col, n_idx))
        errors["O_r"] = max(
            errors["O_r"],
            float(np.linalg.norm(or_lookup @ in_r - or_adder @ in_r)),
        )

        in_t = basis_vector(d_bits, m_bits)
        errors["O_t"] = max(
            errors["O_t"],
            float(np.linalg.norm(ot_lookup @ in_t - ot_adder @ in_t)),
        )

    for name, err in errors.items():
        if err > atol:
            raise ValueError(f"{name} lookup vs adder mismatch: {err:.3e}")
    return errors


def compare_index_oracle_implementations(
    labeling: PechanLabeling,
    *,
    optimization_level: int = 2,
) -> dict[str, int]:
    """Transpiled T-gate counts for lookup vs arithmetic ``O_c``, ``O_r``, ``O_t``."""
    counts = transpiled_gate_counts
    report: dict[str, int] = {}

    for prefix, lookup_fn, adder_fn in (
        ("O_c", build_column_oracle_lookup, build_column_oracle_arithmetic),
        ("O_r", build_row_oracle_lookup, build_row_oracle_arithmetic),
        ("O_t", build_transpose_oracle_pechan, build_transpose_oracle_arithmetic),
    ):
        n_qubits = (
            labeling.n_d_qubits + labeling.n_m_qubits + labeling.n_index_qubits
            if prefix != "O_t"
            else labeling.n_d_qubits + labeling.n_m_qubits
        )
        lookup_circ = QuantumCircuit(n_qubits)
        lookup_circ.append(lookup_fn(labeling), lookup_circ.qubits)
        adder_circ = QuantumCircuit(n_qubits)
        adder_circ.append(adder_fn(labeling), adder_circ.qubits)
        report[f"{prefix}_lookup_t_gates"] = counts(
            lookup_circ, optimization_level=optimization_level
        )["t_gates"]
        report[f"{prefix}_adder_t_gates"] = counts(
            adder_circ, optimization_level=optimization_level
        )["t_gates"]
    return report


def _embed_on_qubits(
    operator: np.ndarray,
    qubit_positions: list[int],
    num_qubits: int,
) -> np.ndarray:
    """Embed a 2^k operator on ordered qubit_positions of a num_qubits system."""
    dim = 2**num_qubits
    full = np.zeros((dim, dim), dtype=complex)
    k = len(qubit_positions)

    for input_state in range(dim):
        bits = [(input_state >> q) & 1 for q in range(num_qubits)]
        sub_in = sum(bits[q] << pos for pos, q in enumerate(qubit_positions))
        for sub_out in range(2**k):
            out_state = input_state
            for pos, q in enumerate(qubit_positions):
                out_state = (out_state & ~(1 << q)) | (((sub_out >> pos) & 1) << q)
            full[out_state, input_state] = operator[sub_out, sub_in]
    return full


def _data_loading_matrix(values: np.ndarray, scale: float, n_d: int) -> np.ndarray:
    """O_data on one data qubit controlled by an n_d-qubit label register."""
    dim = 2 ** (1 + n_d)
    operator = np.zeros((dim, dim), dtype=complex)
    for d_label, value in enumerate(values):
        if np.isclose(value, 0.0):
            continue
        angle = 2.0 * np.arccos(np.clip(abs(value) / scale, 0.0, 1.0))
        cos_a = np.cos(angle / 2.0)
        sin_a = np.sin(angle / 2.0)
        idx = d_label << 1
        operator[idx, idx] = cos_a
        operator[idx + 1, idx] = sin_a if value >= 0 else -sin_a
        operator[idx + 1, idx + 1] = 1.0
    for state in range(dim):
        if np.isclose(np.linalg.norm(operator[:, state]), 0.0):
            operator[state, state] = 1.0
    return operator


def reconstructed_normalized_matrix(labeling: PechanLabeling, alpha: float) -> np.ndarray:
    """
    Reassemble G/alpha from the (d, m) -> (i, j) lookup and value table.

    Valid when each nonzero matrix entry has a unique (d, m) label.
    """
    n_grid = labeling.n_grid
    block = np.zeros((n_grid, n_grid), dtype=float)
    for (d_label, m_label), (row, col, _) in _entries_by_dm(labeling).items():
        value = labeling.value_table[d_label]
        if block[row, col] != 0.0:
            raise ValueError(
                f"Duplicate coverage at ({row}, {col}) for label ({d_label}, {m_label})."
            )
        block[row, col] = value / alpha
    return block


def build_block_encoding_unitary_matrix(
    labeling: PechanLabeling,
    *,
    scale: float | None = None,
    use_adders: bool = False,
) -> tuple[np.ndarray, float]:
    """
    Build U by applying the oracle product to each computational basis state.

    Intended for small register sizes only (default demo grids).
    """
    alpha = scale if scale is not None else spectral_scale(labeling.matrix)
    n_d = labeling.n_d_qubits
    n_m = labeling.n_m_qubits
    n_idx = labeling.n_index_qubits
    num_qubits = 1 + n_d + n_m + n_idx
    dim = 2**num_qubits

    oc_builder, or_builder, ot_builder = _index_oracle_builders(use_adders)
    oc = Operator(oc_builder(labeling)).data
    or_matrix = Operator(or_builder(labeling)).data
    ot = Operator(ot_builder(labeling)).data
    odata = _data_loading_matrix(labeling.value_table, alpha, n_d)
    z = np.array([[1, 0], [0, -1]], dtype=complex)

    oc_full = _embed_on_qubits(oc, list(range(1, 1 + n_d + n_m + n_idx)), num_qubits)
    ot_full = _embed_on_qubits(ot, list(range(1, 1 + n_d + n_m)), num_qubits)
    or_full = _embed_on_qubits(or_matrix, list(range(1, 1 + n_d + n_m + n_idx)), num_qubits)
    odata_full = _embed_on_qubits(odata, list(range(0, 1 + n_d)), num_qubits)
    z_full = _embed_on_qubits(z, [0], num_qubits)

    step = oc_full.conj().T @ z_full @ odata_full @ z_full @ ot_full @ or_full
    unitary = np.zeros((dim, dim), dtype=complex)
    for col in range(dim):
        basis = np.zeros(dim, dtype=complex)
        basis[col] = 1.0
        unitary[:, col] = step @ basis
    return unitary, alpha


def build_symmetric_block_encoding_circuit(
    labeling: PechanLabeling,
    *,
    scale: float | None = None,
    use_adders: bool = False,
) -> tuple[QuantumCircuit, float]:
    """Qiskit circuit mirroring build_block_encoding_unitary_matrix."""
    alpha = scale if scale is not None else spectral_scale(labeling.matrix)
    n_d = labeling.n_d_qubits
    n_m = labeling.n_m_qubits
    n_idx = labeling.n_index_qubits

    oc_builder, or_builder, ot_builder = _index_oracle_builders(use_adders)
    oc = oc_builder(labeling)
    or_gate = or_builder(labeling)
    ot = ot_builder(labeling)

    circuit = QuantumCircuit(1 + n_d + n_m + n_idx, name="U_G")
    data = 0
    d_regs = list(range(1, 1 + n_d))
    m_regs = list(range(1 + n_d, 1 + n_d + n_m))
    idx_regs = list(range(1 + n_d + n_m, circuit.num_qubits))

    circuit.append(oc.inverse(), d_regs + m_regs + idx_regs)
    apply_multiplexed_ry(circuit, data, d_regs, labeling.value_table, alpha)
    circuit.z(data)
    circuit.append(ot, d_regs + m_regs)
    circuit.append(or_gate, d_regs + m_regs + idx_regs)

    return circuit, alpha


def extracted_block_from_unitary(
    unitary: np.ndarray,
    labeling: PechanLabeling,
) -> np.ndarray:
    """Read out (⟨0|_data ⟨k|) U (|0⟩_data |j⟩)."""
    n_d = labeling.n_d_qubits
    n_m = labeling.n_m_qubits
    n_idx = labeling.n_index_qubits
    n_grid = labeling.n_grid
    dim = unitary.shape[0]

    block = np.zeros((n_grid, n_grid), dtype=complex)
    for col in range(n_grid):
        in_vector = np.zeros(dim, dtype=complex)
        in_bits = [0] + [0] * n_d + [0] * n_m + _index_bits(col, n_idx)
        in_vector[_basis_index([in_bits])] = 1.0
        out_vector = unitary @ in_vector
        for row in range(n_grid):
            out_bits = [0] + [0] * n_d + [0] * n_m + _index_bits(row, n_idx)
            block[row, col] = out_vector[_basis_index([out_bits])]
    return block


def verify_block_encoding_matrix(
    labeling: PechanLabeling,
    *,
    scale: float | None = None,
    use_full_unitary: bool = False,
) -> tuple[float, float]:
    """Return (max error, scale alpha) against labeling.matrix / alpha."""
    alpha = scale if scale is not None else spectral_scale(labeling.matrix)
    target = labeling.matrix / alpha

    if use_full_unitary and labeling.n_grid <= 4:
        unitary, alpha = build_block_encoding_unitary_matrix(labeling, scale=alpha)
        block = extracted_block_from_unitary(unitary, labeling)
    else:
        block = reconstructed_normalized_matrix(labeling, alpha)

    return float(np.max(np.abs(block - target))), alpha


def verify_pechan_labeling(labeling: PechanLabeling) -> float:
    """Max error between matrix entries and their Pechan label values."""
    max_error = 0.0
    for row, col, _, d_label, value in labeling.entries:
        max_error = max(max_error, abs(labeling.matrix[row, col] - value))
        if labeling.value_table[d_label] != value:
            raise ValueError(f"Value table mismatch at d={d_label}.")
    return max_error


def verify_block_encoding(
    circuit: QuantumCircuit,
    labeling: PechanLabeling,
    scale: float,
    *,
    atol: float = 1e-9,
) -> float:
    """Verify via explicit unitary assembly (avoids simulating huge circuits)."""
    del circuit, atol
    error, _ = verify_block_encoding_matrix(labeling, scale=scale)
    return error


def odata_gate_budget(
    labeling: PechanLabeling,
    *,
    optimization_level: int = 2,
    sk_t_count: bool = True,
) -> dict[str, object]:
    """Transpiled gate counts for ``O_data`` only (no index oracles or full ``U_G``)."""
    alpha = spectral_scale(labeling.matrix)
    odata = data_loading_subcircuit(labeling.value_table, alpha)
    return {
        "alpha": alpha,
        "D_prime": labeling.d_prime,
        "D_padded": labeling.d_padded,
        "O_data": transpiled_gate_counts(
            odata,
            optimization_level=optimization_level,
            sk_t_count=sk_t_count,
        ),
    }


def summarize_odata_scaling(
    grid_sizes: tuple[int, ...] | list[int],
    *,
    k_rock: float = 1.0,
    k_fracture: float = 5.0,
    sk_t_count: bool = False,
    optimization_level: int = 2,
) -> list[dict[str, object]]:
    """
    Fast ``D`` / ``O_data`` scaling sweep without assembling lookup ``U_G``.

    By default reports transpiled depth and size (no Solovay--Kitaev T-count).
    Set ``sk_t_count=True`` for fault-tolerant-style ``O_data`` T-gates (slower).
    """
    rows: list[dict[str, object]] = []
    for nx in grid_sizes:
        labeling, alpha = build_two_material_labeling(
            nx, nx, k_rock=k_rock, k_fracture=k_fracture
        )
        summary = summarize_labeling(labeling)
        odata_budget = odata_gate_budget(
            labeling,
            optimization_level=optimization_level,
            sk_t_count=sk_t_count,
        )
        odata = odata_budget["O_data"]
        row: dict[str, object] = {
            "nx": nx,
            "N": nx * nx,
            "D_prime": summary["D_prime"],
            "D_padded": summary["D_padded"],
            "O_data_depth": odata["depth"],
            "O_data_size": odata["size"],
            "alpha": alpha,
        }
        if sk_t_count:
            row["O_data_T_gates"] = odata["t_gates"]
        rows.append(row)
    return rows


def summarize_labeling(labeling: PechanLabeling) -> dict[str, object]:
    section_names = {0: "diagonal", 1: "x-neighbor", 2: "y-neighbor"}
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
        "nx": labeling.nx,
        "ny": labeling.ny,
        "N": labeling.n_grid,
        "D_init_sections": {k: len(v) for k, v in by_section.items()},
        "D_prime": labeling.d_prime,
        "D_padded": labeling.d_padded,
        "M_padded": labeling.m_padded,
        "S_padded": labeling.s_padded,
        "value_table": labeling.value_table.copy(),
        "spectral_norm": spectral_scale(labeling.matrix),
    }


def gate_budget_report(
    labeling: PechanLabeling,
    *,
    optimization_level: int = 2,
    transpile_index_oracles: bool = False,
    compare_oracles: bool = False,
    transpile_full: bool = False,
    use_adders: bool = False,
) -> dict[str, object]:
    """
    Gate budgets with fast defaults.

    By default only ``O_data`` is transpiled (T-count). Index oracles return lightweight
    metadata unless ``transpile_index_oracles=True``. Lookup-vs-adder comparison and
    full ``U_G`` transpilation are opt-in via ``compare_oracles`` and ``transpile_full``.
    """
    alpha = spectral_scale(labeling.matrix)
    odata = data_loading_subcircuit(labeling.value_table, alpha)
    odata_counts = transpiled_gate_counts(odata, optimization_level=optimization_level)

    n_index_qubits = (
        labeling.n_d_qubits + labeling.n_m_qubits + labeling.n_index_qubits
    )
    index_dim = 2**n_index_qubits

    index_key = "O_index_adder" if use_adders else "O_index_lookup"
    report: dict[str, object] = {
        "alpha": alpha,
        "D_prime": labeling.d_prime,
        "D_padded": labeling.d_padded,
        "O_data": odata_counts,
    }

    if transpile_index_oracles:
        oc_builder, or_builder, ot_builder = _index_oracle_builders(use_adders)
        oc = oc_builder(labeling)
        or_gate = or_builder(labeling)
        ot = ot_builder(labeling)

        oc_circuit = QuantumCircuit(n_index_qubits)
        oc_circuit.append(oc, oc_circuit.qubits)
        or_circuit = QuantumCircuit(n_index_qubits)
        or_circuit.append(or_gate, or_circuit.qubits)
        ot_circuit = QuantumCircuit(labeling.n_d_qubits + labeling.n_m_qubits)
        ot_circuit.append(ot, ot_circuit.qubits)

        report[index_key] = {
            "O_c": transpiled_gate_counts(oc_circuit, optimization_level=optimization_level),
            "O_r": transpiled_gate_counts(or_circuit, optimization_level=optimization_level),
            "O_t": transpiled_gate_counts(ot_circuit, optimization_level=optimization_level),
            "note": (
                "Transpiled index oracles (slow; dense lookup or arithmetic adders)."
            ),
        }
    else:
        report[index_key] = {
            "note": (
                "Metadata only; set transpile_index_oracles=True for T-counts "
                "(slow on large grids)."
            ),
            "num_qubits": n_index_qubits,
            "unitary_dimension": index_dim,
            "O_c_gate_count": 1,
            "O_r_gate_count": 1,
            "O_t_gate_count": 1,
        }

    if compare_oracles:
        report["oracle_comparison"] = compare_index_oracle_implementations(
            labeling, optimization_level=optimization_level
        )

    if transpile_full and labeling.n_grid <= 4:
        full_circuit, _ = build_symmetric_block_encoding_circuit(
            labeling, scale=alpha, use_adders=use_adders
        )
        report["U_full"] = transpiled_gate_counts(
            full_circuit, optimization_level=optimization_level
        )
    else:
        report["U_full"] = {
            "note": (
                "Set transpile_full=True on n_grid<=4 for full U_G T-counts (slow)."
            ),
            "O_data_t_gates": odata_counts["t_gates"],
        }

    return report


def clinic_elastic_stiffness_2d(
    nx: int,
    ny: int,
    *,
    lambda_base: float = 2.16e10,
    mu_base: float = 8.1e9,
    lambda_fracture: float = 2.2e9,
    mu_fracture: float = 1e-6,
    fracture_axis: str = "x",
    fracture_index: int | None = None,
    dx: float = 1.0,
    dy: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    2D clinic proxy: block-encode ``-div(E grad)`` with ``E = lambda + 2 mu``.

    Uses the same rock/fracture mask as ``permeability_field_two_material`` with
    clinic Lamé parameters mapped to scalar stiffness ``E``.
    """
    e_rock = lambda_base + 2.0 * mu_base
    e_fracture = lambda_fracture + 2.0 * mu_fracture
    stiffness = permeability_field_two_material(
        nx,
        ny,
        k_rock=e_rock,
        k_fracture=e_fracture,
        fracture_axis=fracture_axis,
        fracture_index=fracture_index,
    )
    matrix = poisson_matrix_2d_dirichlet(stiffness, dx=dx, dy=dy)
    return matrix, stiffness


def build_dirichlet_demo(
    nx: int = 4,
    ny: int = 4,
    *,
    k_rock: float = 1.0,
    k_fracture: float = 5.0,
) -> tuple[PechanLabeling, QuantumCircuit, float]:
    """Two-material Dirichlet Poisson demo (clinic-style box BC)."""
    permeability = permeability_field_two_material(
        nx, ny, k_rock=k_rock, k_fracture=k_fracture, fracture_axis="x"
    )
    matrix = poisson_matrix_2d_dirichlet(permeability)
    labeling = pechan_relabel(matrix, nx, ny)
    circuit, alpha = build_symmetric_block_encoding_circuit(labeling)
    return labeling, circuit, alpha


def build_two_material_labeling(
    nx: int = 4,
    ny: int = 4,
    *,
    k_rock: float = 1.0,
    k_fracture: float = 5.0,
) -> tuple[PechanLabeling, float]:
    """Pechan labeling + ``alpha`` without building lookup ``U_G`` (fast for sweeps)."""
    permeability = permeability_field_two_material(
        nx, ny, k_rock=k_rock, k_fracture=k_fracture, fracture_axis="x"
    )
    matrix = poisson_matrix_2d_periodic(permeability)
    labeling = pechan_relabel(matrix, nx, ny)
    alpha = spectral_scale(labeling.matrix)
    return labeling, alpha


def build_two_material_demo(
    nx: int = 4,
    ny: int = 4,
    *,
    k_rock: float = 1.0,
    k_fracture: float = 5.0,
) -> tuple[PechanLabeling, QuantumCircuit, float]:
    labeling, alpha = build_two_material_labeling(
        nx, ny, k_rock=k_rock, k_fracture=k_fracture
    )
    circuit, _ = build_symmetric_block_encoding_circuit(labeling, scale=alpha)
    return labeling, circuit, alpha
