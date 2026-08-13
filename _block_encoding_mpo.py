"""MPO representation of H and Nibbi–Mendl block encoding.

Implements the construction of Martina Nibbi & Christian B. Mendl,
*Block encoding of matrix product operators*, Phys. Rev. A 110, 042427 (2024):

* sequential SVD MPO factorization of a ``2^L × 2^L`` Hamiltonian;
* unitary dilation of each reshaped MPO tensor;
* quantum circuit on ``L`` physical + ``L`` dilation + ``D`` bond qubits;
* postselected extraction / Aer or Statevector verification.

Clinic elastic ``H`` is padded to the next qubit register and treated as an
operator on ``L = ceil(log2 N_s)`` sites (snake / bit ordering). Bond dimension
may be truncated for laptop demos.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Operator, Statevector

from _block_encoding_common import pad_to_power_of_two


@dataclass(frozen=True)
class MPO:
    """Open-boundary MPO cores of shape ``(chi_l, 2, 2, chi_r)`` each."""

    cores: tuple[np.ndarray, ...]
    """Site tensors ``A[ℓ]`` with indices ``(left, out, in, right)``."""

    @property
    def n_sites(self) -> int:
        return len(self.cores)

    @property
    def max_bond(self) -> int:
        bonds = [1]
        for core in self.cores:
            bonds.append(int(core.shape[0]))
            bonds.append(int(core.shape[3]))
        return int(max(bonds))

    def bond_dims(self) -> list[int]:
        if not self.cores:
            return [1]
        dims = [int(self.cores[0].shape[0])]
        for core in self.cores:
            dims.append(int(core.shape[3]))
        return dims


def _bits_to_index(bits: list[int]) -> int:
    """Little-endian bit list → integer (bits[0] = LSB)."""
    idx = 0
    for k, bit in enumerate(bits):
        idx |= (int(bit) & 1) << k
    return idx


def _index_to_bits(index: int, n_bits: int) -> list[int]:
    return [(int(index) >> k) & 1 for k in range(n_bits)]


def matrix_to_mpo(
    matrix: np.ndarray,
    *,
    max_bond: int | None = None,
    cutoff: float = 1e-12,
) -> MPO:
    """
    Exact / truncated MPO via sequential SVD of a ``2^L × 2^L`` matrix.

    Physical wires are qubits in little-endian Qiskit order (site 0 = LSB).
    """
    h = np.asarray(matrix, dtype=complex)
    if h.ndim != 2 or h.shape[0] != h.shape[1]:
        raise ValueError("matrix must be square")
    dim = h.shape[0]
    if dim & (dim - 1) != 0:
        raise ValueError("matrix dimension must be a power of two")
    n_sites = int(np.log2(dim))
    if n_sites < 1:
        raise ValueError("need at least one qubit")

    # Site-major tensor (out_0,in_0,...,out_{L-1},in_{L-1}) with site0 = LSB of H.
    tensor = np.zeros((2,) * (2 * n_sites), dtype=complex)
    for row in range(dim):
        for col in range(dim):
            outs = _index_to_bits(row, n_sites)
            inns = _index_to_bits(col, n_sites)
            coord: list[int] = []
            for site in range(n_sites):
                coord.append(outs[site])
                coord.append(inns[site])
            tensor[tuple(coord)] = h[row, col]

    cores: list[np.ndarray] = []
    chi_l = 1
    rest = tensor.reshape(4, -1)

    for site in range(n_sites - 1):
        u, s, vh = np.linalg.svd(rest, full_matrices=False)
        keep = max(int(np.sum(s > cutoff)), 1)
        if max_bond is not None:
            keep = min(keep, int(max_bond))
        u, s, vh = u[:, :keep], s[:keep], vh[:keep, :]
        chi_r = keep
        cores.append(u.reshape(chi_l, 2, 2, chi_r))
        contracted = np.diag(s) @ vh
        if site == n_sites - 2:
            cores.append(contracted.reshape(chi_r, 2, 2, 1))
            break
        chi_l = chi_r
        rest = contracted.reshape(chi_l * 4, -1)
    else:
        cores.append(tensor.reshape(1, 2, 2, 1))

    return MPO(cores=tuple(np.asarray(c, dtype=complex) for c in cores))


def mpo_to_matrix(mpo: MPO) -> np.ndarray:
    """Contract an MPO back to a dense ``2^L × 2^L`` matrix (site0 = LSB)."""
    cores = list(mpo.cores)
    n_sites = len(cores)
    dim = 1 << n_sites
    out = np.zeros((dim, dim), dtype=complex)

    # Enumerate is fine for demo sizes; keeps LSB packing explicit.
    def contract_amp(outs: list[int], inns: list[int]) -> complex:
        # boundary left bond = 0
        # accumulate vectors over bond
        vec = np.array([1.0 + 0.0j])  # chi_l = 1
        for site, core in enumerate(cores):
            # core[bl, so, si, br]; vec over bl
            chi_l, _, _, chi_r = core.shape
            new_vec = np.zeros(chi_r, dtype=complex)
            for bl in range(chi_l):
                if bl >= vec.shape[0]:
                    break
                for br in range(chi_r):
                    new_vec[br] += vec[bl] * core[bl, outs[site], inns[site], br]
            vec = new_vec
        return complex(vec[0]) if vec.size else 0.0

    for row in range(dim):
        for col in range(dim):
            out[row, col] = contract_amp(
                _index_to_bits(row, n_sites), _index_to_bits(col, n_sites)
            )
    return out


def mpo_reconstruction_error(matrix: np.ndarray, mpo: MPO) -> float:
    """``‖H - H_MPO‖_max``."""
    return float(np.max(np.abs(np.asarray(matrix) - mpo_to_matrix(mpo))))


def _pad_bond_matrix(core: np.ndarray, bond_dim: int) -> np.ndarray:
    """
    Reshape core ``(chi_l, 2, 2, chi_r)`` to a matrix on ``phys ⊗ bond``.

    Rows / cols ordered as Qiskit little-endian on ``[phys, bond_qubits…]``
    with bond register size ``bond_dim`` (power of two ≥ max(chi_l, chi_r)).
    Sequential L→R update on a shared bond register:
    ``M |α_{ℓ-1}⟩|s_in⟩ = ∑ A_{α_{ℓ-1}, s_out, s_in, α_ℓ} |α_ℓ⟩|s_out⟩``
    with virtual indices zero-padded into ``0..bond_dim-1``.
    """
    chi_l, p_out, p_in, chi_r = core.shape
    if p_out != 2 or p_in != 2:
        raise ValueError("physical dimension must be 2")
    if bond_dim < max(chi_l, chi_r) or bond_dim & (bond_dim - 1) != 0:
        raise ValueError("bond_dim must be a power of two covering chi_l/chi_r")

    dim = 2 * bond_dim
    mat = np.zeros((dim, dim), dtype=complex)
    # |b⟩|s⟩ ↔ s + 2*b (phys = LSB); input b=left, output b=right
    for bl in range(chi_l):
        for so in range(2):
            for si in range(2):
                for br in range(chi_r):
                    row = so + 2 * br
                    col = si + 2 * bl
                    mat[row, col] = core[bl, so, si, br]
    return mat


def unitary_dilate(matrix: np.ndarray, norm: float | None = None) -> tuple[np.ndarray, float]:
    """
    Unitary dilation of square ``M`` (Nibbi–Mendl / Ref. [9] App. D).

    Returns ``(U, N)`` with upper-left block **exactly** ``M/N`` (no QR phase
    flips). Completes the isometry ``W = [M/N; B]`` with an ONB of its
    orthogonal complement.
    """
    m = np.asarray(matrix, dtype=complex)
    if m.ndim != 2 or m.shape[0] != m.shape[1]:
        raise ValueError("matrix must be square")
    dim = m.shape[0]
    spectral = float(np.linalg.norm(m, ord=2))
    nrm = float(spectral if norm is None else norm)
    if nrm < spectral - 1e-9:
        raise ValueError(f"norm {nrm} < spectral norm {spectral}")
    if nrm <= 0:
        nrm = 1.0

    _u, s, vh = np.linalg.svd(m, full_matrices=True)
    scale = np.sqrt(np.maximum(1.0 - (s * s) / (nrm * nrm), 0.0))
    # B = sqrt(I - Σ²/N²) V†  (paper eq. after (7))
    b = scale[:, None] * vh
    w = np.vstack([m / nrm, b])  # (2 dim) × dim, orthonormal columns
    # Complete to a 2dim × 2dim unitary with first columns exactly W.
    # Null space of W† spans the orthogonal complement.
    # Use SVD of W†: last dim right-singular vectors.
    _uw, _sw, vhw = np.linalg.svd(w.conj().T, full_matrices=True)
    # vhw shape (2dim, 2dim); rows corresponding to zero singular values
    # of W† are an ONB of ker(W†). W† has rank dim, so last dim rows.
    complement = vhw[dim:, :].conj().T  # (2dim, dim)
    u_full = np.hstack([w, complement])
    # Fix possible det=-1 / reflection by flipping one complement column if needed
    if abs(np.linalg.det(u_full) - 1) > abs(np.linalg.det(u_full) + 1):
        u_full[:, -1] *= -1
    return u_full.astype(complex), nrm


@dataclass(frozen=True)
class MPOBlockEncoding:
    """Nibbi–Mendl block encoding of an MPO."""

    mpo: MPO
    bond_dim: int
    d_bond_qubits: int
    site_unitaries: tuple[np.ndarray, ...]
    site_norms: tuple[float, ...]
    n_mpo: float
    circuit: QuantumCircuit
    """Qubit order: ``[dil_0..dil_{L-1}, phys_0..phys_{L-1}, bond_0..]``."""


def build_mpo_block_encoding(
    mpo: MPO,
    *,
    bond_dim: int | None = None,
) -> MPOBlockEncoding:
    """
    Build the Fig. 2 block-encoding circuit for ``mpo``.

    Boundary vectors are the unit left/right bonds embedded as ``|0…0⟩`` on
    the bond register (exact for open SVD MPOs with ``χ_0 = χ_L = 1``).
    Postselect all dilation qubits and all bond qubits in ``|0⟩``.
    """
    chi = mpo.max_bond
    if bond_dim is None:
        bond_dim = pad_to_power_of_two(chi)
    d_bond = int(np.log2(bond_dim))
    n_sites = mpo.n_sites

    unitaries: list[np.ndarray] = []
    norms: list[float] = []
    for core in mpo.cores:
        m_mat = _pad_bond_matrix(core, bond_dim)
        u_site, n_site = unitary_dilate(m_mat)
        unitaries.append(u_site)
        norms.append(n_site)
    n_mpo = float(np.prod(norms)) if norms else 1.0

    # Qubit layout: dilations | physical | bond
    n_qubits = n_sites + n_sites + d_bond
    circuit = QuantumCircuit(n_qubits, name="U_MPO")
    dil = list(range(n_sites))
    phys = list(range(n_sites, 2 * n_sites))
    bond = list(range(2 * n_sites, 2 * n_sites + d_bond))

    for ell, u_site in enumerate(unitaries):
        gate = UnitaryGate(u_site, check_input=False, label=f"UA{ell}")
        # Numpy dilation uses dil as MSB; Qiskit LSB-first → [phys, bond…, dil]
        wires = [phys[ell], *bond, dil[ell]]
        circuit.append(gate, wires)

    return MPOBlockEncoding(
        mpo=mpo,
        bond_dim=bond_dim,
        d_bond_qubits=d_bond,
        site_unitaries=tuple(unitaries),
        site_norms=tuple(norms),
        n_mpo=n_mpo,
        circuit=circuit,
    )


def _postselect_amplitude(
    state: Statevector,
    *,
    n_sites: int,
    d_bond: int,
    phys_bits: int,
) -> complex:
    """
    Amplitude of ``|0…0⟩_dil |phys_bits⟩_phys |0…0⟩_bond`` in Qiskit little-endian.

    Qubit order: dil_0..dil_{L-1}, phys_0..phys_{L-1}, bond_….
    """
    # Full basis index: bit i is qubit i
    idx = 0
    for q in range(n_sites):
        if (phys_bits >> q) & 1:
            idx |= 1 << (n_sites + q)
    # dilations and bond already 0
    _ = d_bond
    return complex(state.data[idx])


def apply_mpo_block_encoding_to_state(
    be: MPOBlockEncoding,
    psi: np.ndarray,
    *,
    backend: str = "statevector",
) -> tuple[np.ndarray, float]:
    """
    Apply ``U_MPO`` to ``|0⟩_dil|ψ⟩_phys|0⟩_bond`` and postselect ancillas to 0.

    Returns ``(ψ_out_unnormalized, success_probability)``. ``ψ_out / √p`` is the
    normalized postselected state; the encoded action is
    ``ψ_out ≈ (H / N_MPO) ψ`` (unnormalized postselection).
    """
    psi = np.asarray(psi, dtype=complex).reshape(-1)
    n_sites = be.mpo.n_sites
    dim = 1 << n_sites
    if psi.shape[0] != dim:
        raise ValueError(f"psi length must be {dim}")

    # Statevector on all qubits: dil=0, phys=psi, bond=0
    full = np.zeros(1 << be.circuit.num_qubits, dtype=complex)
    for s, amp in enumerate(psi):
        if abs(amp) == 0:
            continue
        idx = 0
        for q in range(n_sites):
            if (s >> q) & 1:
                idx |= 1 << (n_sites + q)
        full[idx] = amp

    if backend == "aer":
        try:
            from qiskit_aer import AerSimulator
        except ImportError as exc:
            raise ImportError(
                "qiskit-aer is not installed. Use backend='statevector' or "
                "pip install qiskit-aer."
            ) from exc
        sim = AerSimulator(method="statevector")
        qc = be.circuit.copy()
        qc.save_statevector()
        # Initialize via circuit initialize is awkward for custom amp; evolve SV
        evolved = Statevector(full).evolve(be.circuit)
        data = evolved.data
    elif backend == "statevector":
        data = Statevector(full).evolve(be.circuit).data
    else:
        raise ValueError("backend must be 'statevector' or 'aer'")

    out = np.zeros(dim, dtype=complex)
    # Sum probability on postselected subspace and collect phys amplitudes
    # Postselect: all dil=0 and all bond=0; phys free
    n_qubits = be.circuit.num_qubits
    d_bond = be.d_bond_qubits
    mask_ancilla = 0
    for q in range(n_sites):  # dilations
        mask_ancilla |= 1 << q
    for q in range(d_bond):
        mask_ancilla |= 1 << (2 * n_sites + q)

    success = 0.0
    for idx, amp in enumerate(data):
        if idx & mask_ancilla:
            continue
        # phys bits from qubits n_sites .. 2n_sites-1
        phys = 0
        for q in range(n_sites):
            if (idx >> (n_sites + q)) & 1:
                phys |= 1 << q
        out[phys] += amp
        success += float(abs(amp) ** 2)
    return out, success


def encoded_action_error(
    be: MPOBlockEncoding,
    hamiltonian: np.ndarray,
    *,
    n_states: int = 4,
    rng_seed: int = 0,
    backend: str = "statevector",
) -> dict[str, float]:
    """
    Compare postselected ``U_MPO`` action to ``H / N_MPO`` on random states.
    """
    h = np.asarray(hamiltonian, dtype=complex)
    dim = h.shape[0]
    rng = np.random.default_rng(rng_seed)
    errs = []
    ps = []
    for _ in range(n_states):
        psi = rng.normal(size=dim) + 1j * rng.normal(size=dim)
        psi /= np.linalg.norm(psi)
        out, p = apply_mpo_block_encoding_to_state(be, psi, backend=backend)
        target = (h @ psi) / be.n_mpo
        errs.append(float(np.linalg.norm(out - target)))
        ps.append(p)
    return {
        "mean_action_error": float(np.mean(errs)),
        "max_action_error": float(np.max(errs)),
        "mean_success_prob": float(np.mean(ps)),
        "n_mpo": float(be.n_mpo),
        "n_qubits": float(be.circuit.num_qubits),
        "max_bond": float(be.mpo.max_bond),
        "bond_dim_padded": float(be.bond_dim),
    }


def extract_encoded_block_operator(be: MPOBlockEncoding) -> np.ndarray:
    """
    Dense postselected block ``⟨0_dil 0_bond| U |0_dil 0_bond⟩`` (small qubit count).

    Expensive: builds the full unitary. Prefer :func:`encoded_action_error` for
    ``L ≳ 4``.
    """
    if be.circuit.num_qubits > 12:
        raise ValueError("extract_encoded_block_operator only for ≤12 qubits")
    full_u = Operator(be.circuit).data
    n_sites = be.mpo.n_sites
    d_bond = be.d_bond_qubits
    dim = 1 << n_sites
    block = np.zeros((dim, dim), dtype=complex)

    def full_index(phys: int) -> int:
        idx = 0
        for q in range(n_sites):
            if (phys >> q) & 1:
                idx |= 1 << (n_sites + q)
        return idx

    for col in range(dim):
        for row in range(dim):
            block[row, col] = full_u[full_index(row), full_index(col)]
    _ = d_bond
    return block


def clinic_h_to_mpo_block_encoding(
    hamiltonian: np.ndarray,
    *,
    max_bond: int | None = 8,
    cutoff: float = 1e-12,
) -> tuple[MPOBlockEncoding, np.ndarray, dict[str, float]]:
    """
    Pad clinic ``H`` to ``2^n``, build (truncated) MPO, and block-encode it.

    Returns ``(be, H_padded, stats)`` with MPO reconstruction + BE action errors.
    """
    from _block_encoding_hamiltonian import pad_to_qubit_register

    h_pad, _n = pad_to_qubit_register(np.asarray(hamiltonian, dtype=complex))
    mpo = matrix_to_mpo(h_pad, max_bond=max_bond, cutoff=cutoff)
    recon = mpo_reconstruction_error(h_pad, mpo)
    be = build_mpo_block_encoding(mpo)
    # Compare BE to the *truncated* MPO operator (fair), and report recon to H
    h_mpo = mpo_to_matrix(mpo)
    stats = encoded_action_error(be, h_mpo, n_states=3, backend="statevector")
    stats["mpo_recon_error_vs_H"] = recon
    stats["mpo_max_bond"] = float(mpo.max_bond)
    stats["n_sites"] = float(mpo.n_sites)
    return be, h_pad, stats
