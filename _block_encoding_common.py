"""Shared helpers for structured block-encoding prototypes.

Utilities used by the 1D and 2D block-encoding notebooks:

* padding and dimension checks for power-of-two registers;
* spectral normalization ``alpha = ||A||_2`` for block encodings;
* transpiled gate budgets;
* multiplexed ``R_y`` data-loading oracle ``O_data`` (Sunderhauf; Pechan).
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Operator
from qiskit.synthesis import SolovayKitaevDecomposition

_SK_DECOMPOSER = SolovayKitaevDecomposition()
_ROTATION_GATE_NAMES = frozenset({"u", "u1", "u2", "u3", "p"})


def power_of_two(value: int, *, name: str = "value") -> int:
    """Raise if ``value`` is not a positive power of two; otherwise return it unchanged."""
    if value <= 0 or value & (value - 1) != 0:
        raise ValueError(f"{name} must be a positive power of two (got {value}).")
    return value


def log2_dim(value: int) -> int:
    """``log2(value)`` for ``value`` a power of two."""
    power_of_two(value)
    return int(np.log2(value))


def pad_to_power_of_two(count: int) -> int:
    """Smallest power of two ``>= max(count, 1)`` (register padding for oracles)."""
    return 1 << int(np.ceil(np.log2(max(count, 1))))


def spectral_scale(matrix: np.ndarray) -> float:
    """Spectral norm ``alpha = ||A||_2`` used to normalize block encodings ``A/alpha``."""
    return float(np.linalg.norm(matrix, ord=2))


def count_t_gates(
    circuit: QuantumCircuit,
    *,
    optimization_level: int = 3,
    sk_recursion_degree: int = 2,
    transpiled: QuantumCircuit | None = None,
) -> int:
    """
    Fault-tolerant-style T count after transpilation.

    Counts explicit ``t`` / ``tdg`` gates, then approximates each single-qubit
    rotation (``u``, ``u1``, ``u2``, ``u3``, ``p``) with Solovay--Kitaev.
    """
    if transpiled is None:
        transpiled = transpile(circuit, optimization_level=optimization_level)
    ops = dict(transpiled.count_ops())
    t_gates = ops.get("t", 0) + ops.get("tdg", 0)
    for instruction in transpiled.data:
        name = instruction.operation.name
        if name not in _ROTATION_GATE_NAMES:
            continue
        matrix = Operator(instruction.operation).data
        if matrix.shape != (2, 2):
            continue
        decomp = _SK_DECOMPOSER.run(matrix, sk_recursion_degree)
        decomp_ops = decomp.count_ops()
        t_gates += decomp_ops.get("t", 0) + decomp_ops.get("tdg", 0)
    return int(t_gates)


def transpiled_gate_counts(
    circuit: QuantumCircuit,
    *,
    optimization_level: int = 3,
    sk_recursion_degree: int = 2,
    sk_t_count: bool = True,
) -> dict[str, int | dict[str, int]]:
    """Depth, size, and T-gate count after Qiskit transpilation (+ optional SK on rotations)."""
    transpiled = transpile(circuit, optimization_level=optimization_level)
    ops = dict(transpiled.count_ops())
    two_qubit_names = {"cx", "cz", "cy", "ch", "swap", "ecr"}
    two_qubit_gates = sum(ops.get(name, 0) for name in two_qubit_names)
    if sk_t_count:
        t_gates = count_t_gates(
            circuit,
            optimization_level=optimization_level,
            sk_recursion_degree=sk_recursion_degree,
            transpiled=transpiled,
        )
    else:
        t_gates = int(ops.get("t", 0) + ops.get("tdg", 0))
    return {
        "depth": transpiled.depth(),
        "size": transpiled.size(),
        "t_gates": t_gates,
        "two_qubit_gates": two_qubit_gates,
        "ops": ops,
    }


def data_loading_subcircuit(values: np.ndarray, scale: float) -> QuantumCircuit:
    """
    Multiplexed ``R_y`` data-loading oracle ``O_data`` (Sunderhauf; Pechan).

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


def apply_multiplexed_ry(
    circuit: QuantumCircuit,
    data_qubit: int,
    value_qubits: list[int],
    values: np.ndarray,
    scale: float,
) -> None:
    """Compose ``data_loading_subcircuit`` in-place on an existing register layout."""
    circuit.compose(
        data_loading_subcircuit(values, scale),
        [data_qubit] + value_qubits,
        inplace=True,
    )
