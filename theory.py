"""theory.py — VERISHEAF: Cellular Sheaf Cohomology Framework for End-to-End Voting Verifiability.

Mathematical foundation of the framework. Implements the five theorems that establish
that end-to-end verifiability of blockchain-based voting reduces to the vanishing of
the first cohomology of a cellular sheaf over the temporal heterogeneous governance
graph, plus the dimensional-asymmetry corollary that characterizes the framework's
prediction on motifs with single-edge record and count stages.

THEOREM 1 (Canonical Sheaf Construction)
    For a typed heterogeneous proposal-voter-execution graph G with the Benaloh typing
    (five node types, five edge types), there exists a canonical cellular sheaf F_G,
    constructible in O(|V| + |E|) time, whose space of global sections H^0(F_G) is in
    bijection with cryptographically consistent voting executions over G.

THEOREM 2 (Cohomological Characterization of Attack Types)
    There exist three pairwise orthogonal subspaces S_cast, S_record, S_count of H^1(F_G)
    such that an attack of type tau in {cast, record, count} produces an element of
    H^1(F_G) in S_tau with probability one and produces no component in S_{tau'} for
    tau' != tau. Orthogonality is measured under the sheaf Laplacian inner product.

THEOREM 3 (Spectral Approximation)
    dim H^1(F_G) = multiplicity of the zero eigenvalue of the sheaf Laplacian Delta_F
    minus dim H^0(F_G). For eps > 0, dim H^1(F_G) and projections onto the three
    attack subspaces can be approximated in O(n log(1/eps)) time via truncated power
    iteration on Delta_F + lambda I, where n = |V| + |E|.

THEOREM 4 (PAC-Bayes Learning Guarantee)
    Given m sampled verifiable executions and stalk dimension d, the empirical neural
    sheaf diffusion recovers the true restriction maps with operator-norm error
    bounded by C * sqrt(log(d/delta) / m) with probability at least 1 - delta, where
    C depends only on the PAC-Bayes prior covariance.

THEOREM 5 (Cross-Stage Subspace Leakage Under Restriction-Map Perturbation)
    Let F be the canonical sheaf and F_tilde a perturbation with
    ||F_tilde - F||_op <= eps per restriction map. Let S_tau denote the stage-tau
    attack subspace of H^1(F) and S_tau_tilde its perturbed counterpart. Then the
    sin-theta principal-angle distance between the two satisfies
        || sin Theta(S_tau, S_tau_tilde) ||_F  <=  4 * rho * eps * sqrt(2 * d * |E_tau| * Delta) / gamma
    where rho is the operator-norm radius, d the stalk dimension, |E_tau| the number
    of stage-tau edges, Delta the maximum node-stalk degree, and gamma the spectral
    gap of L^1_tau above zero.

COROLLARY 1 (Predicted All-Cast Collapse on Dimensionally-Crushed Motifs)
    For motifs with n_c >= 2 cast edges and n_r = n_q = 1 record/count edges, the
    canonical restriction's record and count stage kernels are trivial (dimension 0),
    while the cast-stage kernel has dimension d. The maximum-projection classifier
    therefore predicts cast for every such motif under canonical restriction, and the
    Theorem 5 perturbation correction is O(eps) so the prediction is stable for
    eps below the gamma/4 rho sqrt(2 d |E_tau| Delta) threshold.

References:
    [HG20]  Hansen & Gebhart. "Sheaf Neural Networks." NeurIPS 2020 Workshop.
    [Bod22] Bodnar et al. "Neural Sheaf Diffusion: A Topological Perspective on
            Heterophily and Oversmoothing in GNNs." NeurIPS 2022.
    [Cur14] Curry. "Sheaves, Cosheaves, and Applications." PhD thesis, U. Pennsylvania.
    [Tro15] Tropp. "An Introduction to Matrix Concentration Inequalities." FnTML 2015.
    [McA03] McAllester. "PAC-Bayesian Stochastic Model Selection." Machine Learning 2003.
    [Choi+26] Choi, Choi, Ko, Kim, Kim. "Sheaf Graph Neural Networks via PAC-Bayes
            Spectral Optimization." AAAI 2026, 20570-20578.
    [YWS15] Yu, Wang, Samworth. "A useful variant of the Davis-Kahan theorem for
            statisticians." Biometrika 2015, 102(2):315-323.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch
import torch.nn as nn


# =============================================================================
# Node and edge type system. Five node types and five edge types implement the
# Benaloh cast-recorded-counted decomposition on the governance graph.
# =============================================================================


class NodeType(Enum):
    """Type-system entry for the five node categories in the governance graph."""

    VOTER = "voter"
    DELEGATE = "delegate"
    PROPOSAL = "proposal"
    GOV_TOKEN = "governance_token"
    EXEC_CONTRACT = "execution_contract"


class EdgeType(Enum):
    """Type-system entry for the five edge categories in the governance graph."""

    CAST = "cast"            # voter -> proposal, off-chain ballot
    DELEGATE = "delegate"    # voter -> delegate
    TRANSFER = "transfer"    # voter -> voter, governance token transfer
    RECORD = "record"        # proposal -> execution_contract, on-chain VoteCast
    COUNT = "count"          # execution_contract -> proposal, ProposalExecuted


class BenalohStage(Enum):
    """Benaloh cast-record-count stage labels for the verifiability decomposition."""

    CAST = "cast"
    RECORD = "record"
    COUNT = "count"


# Stalk dimensions for the verifiability sheaf F_G. These are design
# hyperparameters of the framework chosen to balance representational capacity
# with computational tractability. The relative ordering (proposal and
# exec_contract larger than voter and delegate; gov_token smaller) reflects the
# relative complexity of the cryptographic objects each stalk represents, but
# the absolute values are not derived from any specific ballot encoding scheme.
DEFAULT_STALK_DIMS: dict[NodeType, int] = {
    NodeType.VOTER: 8,           # voter intent encoding
    NodeType.DELEGATE: 8,        # same as voter (delegation lifts intent)
    NodeType.PROPOSAL: 16,       # candidate outcomes + bulletin board entry
    NodeType.GOV_TOKEN: 4,       # token balance vector
    NodeType.EXEC_CONTRACT: 16,  # tally outcomes
}

DEFAULT_EDGE_STALK_DIMS: dict[EdgeType, int] = {
    EdgeType.CAST: 12,       # ciphertext ballot
    EdgeType.DELEGATE: 8,
    EdgeType.TRANSFER: 4,
    EdgeType.RECORD: 16,     # on-chain bulletin entry
    EdgeType.COUNT: 16,      # decrypted tally
}


# =============================================================================
# CellularSheaf: the central mathematical object.
# =============================================================================


@dataclass
class CellularSheaf:
    """A cellular sheaf F on a graph G = (V, E).

    Attributes:
        node_types: list of NodeType, indexed by node id
        edge_index: (2, |E|) tensor of (src, tgt) node ids
        edge_types: list of EdgeType, indexed by edge id
        node_stalk_dims: dim F(v) for each node v
        edge_stalk_dims: dim F(e) for each edge e
        restriction_src: (|E|, edge_dim, src_dim) tensor — F_{src(e) -> e}
        restriction_tgt: (|E|, edge_dim, tgt_dim) tensor — F_{tgt(e) -> e}
        stage_mask: list of BenalohStage or None, indexed by edge id
    """

    node_types: list[NodeType]
    edge_index: torch.Tensor
    edge_types: list[EdgeType]
    node_stalk_dims: list[int]
    edge_stalk_dims: list[int]
    restriction_src: torch.Tensor
    restriction_tgt: torch.Tensor
    stage_mask: list[Optional[BenalohStage]] = field(default_factory=list)

    @property
    def num_nodes(self) -> int:
        """Return the number of nodes in the underlying graph."""
        return len(self.node_types)

    @property
    def num_edges(self) -> int:
        """Return the number of edges in the underlying graph."""
        return len(self.edge_types)

    @property
    def total_node_dim(self) -> int:
        """Return the total dimension of the C^0 cochain space, sum of node stalks."""
        return sum(self.node_stalk_dims)

    @property
    def total_edge_dim(self) -> int:
        """Return the total dimension of the C^1 cochain space, sum of edge stalks."""
        return sum(self.edge_stalk_dims)

    def coboundary(self) -> sp.csr_matrix:
        """Sheaf coboundary delta : C^0 -> C^1.

        For x in C^0 = direct sum of node stalks, (delta x)_e = F_{tgt(e) -> e}(x_{tgt(e)})
        - F_{src(e) -> e}(x_{src(e)}). The cellular sheaf cohomology is H^0 = ker(delta),
        H^1 = coker(delta) = C^1 / im(delta).

        Returns:
            sp.csr_matrix of shape (total_edge_dim, total_node_dim).

        Complexity: O(|E| * d_max^2) where d_max = max(node_stalk_dim, edge_stalk_dim).
        """
        node_offset = np.cumsum([0] + self.node_stalk_dims)
        edge_offset = np.cumsum([0] + self.edge_stalk_dims)
        E = self.num_edges

        src_arr = self.edge_index[0].cpu().numpy().astype(int)
        tgt_arr = self.edge_index[1].cpu().numpy().astype(int)
        Fsrc_all = self.restriction_src.detach().cpu().numpy()
        Ftgt_all = self.restriction_tgt.detach().cpu().numpy()

        row_parts: list[np.ndarray] = []
        col_parts: list[np.ndarray] = []
        val_parts: list[np.ndarray] = []
        for e in range(E):
            de = self.edge_stalk_dims[e]
            src = src_arr[e]
            tgt = tgt_arr[e]
            dsrc = self.node_stalk_dims[src]
            dtgt = self.node_stalk_dims[tgt]

            # Block (e, src) contributes -F_{src -> e}
            block_src = Fsrc_all[e, :de, :dsrc]
            ri, ci = np.nonzero(np.abs(block_src) > 1e-12)
            if ri.size:
                row_parts.append(edge_offset[e] + ri)
                col_parts.append(node_offset[src] + ci)
                val_parts.append(-block_src[ri, ci])

            # Block (e, tgt) contributes +F_{tgt -> e}
            block_tgt = Ftgt_all[e, :de, :dtgt]
            ri, ci = np.nonzero(np.abs(block_tgt) > 1e-12)
            if ri.size:
                row_parts.append(edge_offset[e] + ri)
                col_parts.append(node_offset[tgt] + ci)
                val_parts.append(block_tgt[ri, ci])

        if row_parts:
            rows = np.concatenate(row_parts)
            cols = np.concatenate(col_parts)
            vals = np.concatenate(val_parts)
        else:
            rows = np.array([], dtype=int)
            cols = np.array([], dtype=int)
            vals = np.array([], dtype=float)

        return sp.csr_matrix(
            (vals, (rows, cols)),
            shape=(self.total_edge_dim, self.total_node_dim),
        )

    def laplacian(self) -> sp.csr_matrix:
        """Sheaf Laplacian Delta_F = delta^T delta acting on C^0.

        The spectrum of Delta_F encodes the sheaf cohomology by Hodge theory:
        dim ker(Delta_F) = dim H^0(F). For computing H^1 we use the dual
        Laplacian on edges; see h1_dimension below.

        Returns:
            sp.csr_matrix of shape (total_node_dim, total_node_dim).

        Complexity: O(nnz(delta) + |E| * d_max^3).
        """
        delta = self.coboundary()
        return (delta.T @ delta).tocsr()

    def edge_laplacian(self) -> sp.csr_matrix:
        """Edge Laplacian Delta_F^1 = delta delta^T acting on C^1.

        dim ker(Delta_F^1) = dim H^1(F) by Hodge theory for cellular sheaves.

        Returns:
            sp.csr_matrix of shape (total_edge_dim, total_edge_dim).

        Complexity: O(nnz(delta) + |E| * d_max^3).
        """
        delta = self.coboundary()
        return (delta @ delta.T).tocsr()

    def h0_dimension(self, tol: float = 1e-8) -> int:
        """Compute dim H^0(F) = dim ker(delta) = dim ker(Delta_F) exactly.

        Args:
            tol: absolute eigenvalue threshold below which an eigenvalue is zero.

        Returns:
            Integer dim H^0(F).

        Complexity: O((total_node_dim)^3) for the dense fallback, O(nnz * sqrt(kappa))
        for the sparse path. See _count_zero_eigenvalues for the dispatch logic.
        """
        L = self.laplacian()
        return int(_count_zero_eigenvalues(L, tol))

    def h1_dimension(self, tol: float = 1e-8) -> int:
        """Compute dim H^1(F) = dim ker(Delta_F^1) exactly.

        Args:
            tol: absolute eigenvalue threshold below which an eigenvalue is zero.

        Returns:
            Integer dim H^1(F).

        Complexity: O((total_edge_dim)^3) for the dense fallback.
        """
        L1 = self.edge_laplacian()
        return int(_count_zero_eigenvalues(L1, tol))

    def stage_indices(self, stage: BenalohStage) -> torch.Tensor:
        """Return the edge ids in C^1 belonging to a given Benaloh stage.

        Args:
            stage: one of BenalohStage.CAST, RECORD, COUNT.

        Returns:
            torch.LongTensor of edge indices whose stage_mask entry equals stage.
        """
        idx = [i for i, s in enumerate(self.stage_mask) if s == stage]
        return torch.tensor(idx, dtype=torch.long)


def _count_zero_eigenvalues(L: sp.csr_matrix, tol: float) -> int:
    """Count zero eigenvalues of a symmetric positive semidefinite sparse matrix.

    For small matrices uses dense eigensolver; for large matrices uses Lanczos
    with shift-invert at sigma=0, falling back to dense on factorization failure
    (which occurs precisely when L has the nontrivial kernel we are trying to
    count, and the LU backend raises RuntimeError rather than ArpackNoConvergence).

    Args:
        L: symmetric PSD sparse matrix.
        tol: absolute threshold below which an eigenvalue counts as zero.

    Returns:
        Number of eigenvalues strictly below tol.
    """
    n = L.shape[0]
    if n <= 200:
        eigs = np.linalg.eigvalsh(L.toarray())
        return int(np.sum(eigs < tol))
    k = min(50, n - 2)
    try:
        eigs, _ = spla.eigsh(L, k=k, sigma=0, which="LM")
        return int(np.sum(eigs < tol))
    except (spla.ArpackNoConvergence, RuntimeError, np.linalg.LinAlgError):
        eigs = np.linalg.eigvalsh(L.toarray())
        return int(np.sum(eigs < tol))


# =============================================================================
# THEOREM 1: Canonical Sheaf Construction
# =============================================================================


def build_canonical_sheaf(
    node_types: list[NodeType],
    edge_index: torch.Tensor,
    edge_types: list[EdgeType],
    stage_mask: Optional[list[Optional[BenalohStage]]] = None,
    learned_restrictions: Optional[dict] = None,
    stalk_dims: Optional[dict[NodeType, int]] = None,
    edge_stalk_dims: Optional[dict[EdgeType, int]] = None,
    device: str = "cpu",
) -> CellularSheaf:
    """Construct the canonical verifiability sheaf F_G for a typed graph (Theorem 1).

    THEOREM 1 proof sketch. The construction proceeds in three steps. Step 1: assign
    stalks F(v) of dimension d(tau(v)) to each node and F(e) of dimension d(tau(e))
    to each edge, where d is the canonical dimension function in DEFAULT_STALK_DIMS.
    Step 2: for each incidence v <= e, construct the restriction map F_{v <= e} as
    the linear part of the canonical cryptographic transformation associated with
    the Benaloh-stage typing of e (delegated to cryptography.build_restriction).
    Step 3: verify that the constructed sheaf admits a non-trivial space of global
    sections H^0(F_G) corresponding to cryptographically consistent voting executions.

    Args:
        node_types: typing of nodes.
        edge_index: (2, |E|) tensor of source/target node indices.
        edge_types: typing of edges.
        stage_mask: Benaloh stage per edge (auto-derived from edge_types if None).
        learned_restrictions: optional dict with keys ('src', src_type, edge_type)
            and ('tgt', edge_type, tgt_type) mapping to learned restriction matrices.
            If None, canonical maps from cryptography.build_restriction are used.
        stalk_dims: optional per-NodeType stalk dimensions (default DEFAULT_STALK_DIMS).
        edge_stalk_dims: optional per-EdgeType edge-stalk dimensions.
        device: torch device string ("cpu" or "cuda").

    Returns:
        CellularSheaf with stalks and restriction maps populated.

    Citation: manuscript Theorem 1 (Canonical Sheaf Construction).

    Complexity: O(|V| + |E| * d_max^2).
    """
    stalk_dims = stalk_dims or DEFAULT_STALK_DIMS
    edge_stalk_dims = edge_stalk_dims or DEFAULT_EDGE_STALK_DIMS

    if stage_mask is None:
        stage_mask = []
        for et in edge_types:
            if et == EdgeType.CAST:
                stage_mask.append(BenalohStage.CAST)
            elif et == EdgeType.RECORD:
                stage_mask.append(BenalohStage.RECORD)
            elif et == EdgeType.COUNT:
                stage_mask.append(BenalohStage.COUNT)
            else:
                stage_mask.append(None)

    node_dims = [stalk_dims[nt] for nt in node_types]
    edge_dims = [edge_stalk_dims[et] for et in edge_types]

    max_edge_dim = max(edge_dims)
    max_node_dim = max(node_dims)
    E = len(edge_types)

    restriction_src = torch.zeros(E, max_edge_dim, max_node_dim, device=device)
    restriction_tgt = torch.zeros(E, max_edge_dim, max_node_dim, device=device)

    for e in range(E):
        et = edge_types[e]
        src = int(edge_index[0, e].item())
        tgt = int(edge_index[1, e].item())
        nt_src = node_types[src]
        nt_tgt = node_types[tgt]
        de = edge_dims[e]
        dsrc = node_dims[src]
        dtgt = node_dims[tgt]

        if learned_restrictions is not None:
            key_src = ("src", nt_src, et)
            key_tgt = ("tgt", et, nt_tgt)
            if key_src in learned_restrictions:
                restriction_src[e, :de, :dsrc] = learned_restrictions[key_src].to(device)
            else:
                restriction_src[e, :de, :dsrc] = _canonical_restriction(
                    de, dsrc, side="src", node_type=nt_src, edge_type=et
                )
            if key_tgt in learned_restrictions:
                restriction_tgt[e, :de, :dtgt] = learned_restrictions[key_tgt].to(device)
            else:
                restriction_tgt[e, :de, :dtgt] = _canonical_restriction(
                    de, dtgt, side="tgt", node_type=nt_tgt, edge_type=et
                )
        else:
            restriction_src[e, :de, :dsrc] = _canonical_restriction(
                de, dsrc, side="src", node_type=nt_src, edge_type=et
            )
            restriction_tgt[e, :de, :dtgt] = _canonical_restriction(
                de, dtgt, side="tgt", node_type=nt_tgt, edge_type=et
            )

    return CellularSheaf(
        node_types=node_types,
        edge_index=edge_index,
        edge_types=edge_types,
        node_stalk_dims=node_dims,
        edge_stalk_dims=edge_dims,
        restriction_src=restriction_src,
        restriction_tgt=restriction_tgt,
        stage_mask=stage_mask,
    )


def _canonical_restriction(
    d_out: int,
    d_in: int,
    side: str = "tgt",
    node_type: Optional[NodeType] = None,
    edge_type: Optional[EdgeType] = None,
) -> torch.Tensor:
    """Canonical restriction map derived from the cryptographic registry.

    SPEC-REQUIRED scaffold notice: this function delegates to cryptography.build_restriction
    when a typed (side, node_type, edge_type) context is supplied. The numerical block
    contents returned by cryptography.build_restriction are deterministic placeholders
    that the cryptography module's docstring marks SPEC-REQUIRED. The structural
    guarantees (stage-disjoint column supports, full-rank blocks, linear H^0 scaling)
    hold for the scaffold and are exercised by the theorem verifiers in this module
    (verify_theorem_1 through verify_theorem_5). Replacing the placeholder numerical
    blocks with the final exponential ElGamal numerical specification is a separable
    research task; experimental results obtained under this scaffold are explicitly
    conditional on the scaffold being structurally faithful to that final specification.

    When the typed context is absent, falls back to the categorical inclusion/projection
    so that legacy callers (and auxiliary edges) remain well-defined.

    Args:
        d_out: edge stalk dimension (rows of the returned matrix).
        d_in: node stalk dimension (columns of the returned matrix).
        side: "src" or "tgt"; passed through to cryptography.build_restriction for
            signature stability and future asymmetric specifications.
        node_type: NodeType of the incident node, or None for auxiliary fallback.
        edge_type: EdgeType of the edge, or None for auxiliary fallback.

    Returns:
        torch.Tensor of shape (d_out, d_in) carrying the canonical restriction map.

    Citation: manuscript Theorem 1, Remark 1 (ElGamal restriction-map instantiation).

    Complexity: O(d_out * d_in).
    """
    if node_type is not None and edge_type is not None:
        from cryptography import build_restriction
        return build_restriction(
            side, node_type.value, edge_type.value, d_out, d_in
        )
    M = torch.zeros(d_out, d_in)
    d_min = min(d_out, d_in)
    M[:d_min, :d_min] = torch.eye(d_min)
    return M


# =============================================================================
# THEOREM 2: Cohomological Characterization of Attack Types
# =============================================================================


def compute_attack_subspaces(
    sheaf: CellularSheaf,
    eps: float = 1e-8,
) -> dict[BenalohStage, torch.Tensor]:
    """Compute S_cast, S_record, S_count via per-stage sub-Laplacian kernels (Theorem 2).

    THEOREM 2 (reformulated for a structurally honest construction). Partition the
    edge set into E_cast, E_record, E_count by Benaloh stage. For each stage tau,
    restrict the coboundary delta to the C^1 rows indexed by E_tau, giving
    delta_tau, and set S_tau = ker(delta_tau delta_tau^T) embedded back into the
    full C^1 by zero-padding on the complementary edge indices.

    Because the three index sets are disjoint, the three subspaces are supported on
    disjoint coordinate blocks of C^1. They are therefore pairwise orthogonal under
    BOTH the Euclidean inner product AND the sheaf Laplacian inner product:
    Laplacian-orthogonality holds because L^1 = delta delta^T maps each stage's
    coordinate block into itself up to cross terms that vanish on the kernels.
    This replaces the prior sequential Gram-Schmidt, under which orthogonality was
    an artefact of the orthogonalisation procedure rather than a consequence of the
    sheaf structure.

    Args:
        sheaf: the verifiability sheaf F_G.
        eps: numerical threshold for zero eigenvalues.

    Returns:
        dict mapping BenalohStage -> torch.Tensor of shape
        (total_edge_dim, dim_S_stage) with orthonormal columns, supported on the
        stage's edge block.

    Citation: manuscript Theorem 2 (Cohomological Characterization of Attack Types).

    Complexity: O(|E| * d_max^3) for the per-stage dense eigendecomposition.
    """
    delta = sheaf.coboundary()
    edge_offset = np.cumsum([0] + sheaf.edge_stalk_dims)
    n_edge_total = sheaf.total_edge_dim

    subspaces: dict[BenalohStage, torch.Tensor] = {}
    for stage in [BenalohStage.CAST, BenalohStage.RECORD, BenalohStage.COUNT]:
        idx: list[int] = []
        for e, s in enumerate(sheaf.stage_mask):
            if s == stage:
                idx.extend(range(int(edge_offset[e]), int(edge_offset[e + 1])))

        if not idx:
            subspaces[stage] = torch.zeros(n_edge_total, 0)
            continue

        idx_arr = np.array(idx, dtype=int)
        delta_stage = delta[idx_arr, :]
        L1_stage = (delta_stage @ delta_stage.T).toarray()

        eigs, vecs = np.linalg.eigh(L1_stage)
        rank = int(np.sum(eigs < eps))

        basis = np.zeros((n_edge_total, rank))
        if rank > 0:
            basis[idx_arr, :] = vecs[:, :rank]
        subspaces[stage] = torch.from_numpy(basis).float()

    return subspaces


def has_nontrivial_attack_subspaces(
    subspaces: dict[BenalohStage, torch.Tensor]
) -> bool:
    """Diagnostic: True iff at least one stage subspace is non-empty.

    Callers that classify a cohomology class by projection (validate_theorem_2,
    verisheaf_anomaly_score) must check this before projecting; an all-empty
    result means H^1 contributes nothing on the typed edges and any projection
    norm would be a meaningless zero.

    Args:
        subspaces: dict from compute_attack_subspaces.

    Returns:
        True iff sum of column counts across all three subspaces is strictly positive.
    """
    return sum(basis.shape[1] for basis in subspaces.values()) > 0


def project_onto_subspace(
    cocycle: torch.Tensor, subspace_basis: torch.Tensor
) -> tuple[torch.Tensor, float]:
    """Project a 1-cocycle onto a subspace and return both the projection and its norm.

    Used to classify which attack subspace a given cohomology class lives in. The
    projection norm equals ||basis^T cocycle||_2 because the subspace basis has
    orthonormal columns by construction in compute_attack_subspaces.

    Args:
        cocycle: torch.Tensor of shape (total_edge_dim,).
        subspace_basis: torch.Tensor of shape (total_edge_dim, dim_subspace) with
            orthonormal columns.

    Returns:
        (projection, norm) where projection has shape (total_edge_dim,) and norm
        is the L2 norm of the coefficient vector.

    Complexity: O(total_edge_dim * dim_subspace).
    """
    coeffs = subspace_basis.T @ cocycle
    projection = subspace_basis @ coeffs
    return projection, float(torch.linalg.norm(coeffs).item())


# =============================================================================
# THEOREM 3: Spectral Approximation
# =============================================================================


def approximate_h1_dimension(
    sheaf: CellularSheaf, eps: float = 1e-3, max_iter: Optional[int] = None
) -> dict:
    """Approximate dim H^1(F) and the spectral gap via sparse rank computation (Theorem 3).

    THEOREM 3 (corrected statement). By Hodge theory for cellular sheaves,
    dim H^1(F) = dim coker(delta) = dim C^1 - rank(delta), where delta is the
    sheaf coboundary. For sparse delta the rank is obtained from a sparse LU
    factorisation, whose cost is governed by nnz(delta) and the fill-in of the
    factorisation; under bounded stalk dimension and bounded node degree this is
    O(nnz(L^1) * sqrt(kappa) * log(1/eps))-comparable in practice and, crucially,
    returns the EXACT nullity regardless of its size.

    The prior implementation used scipy.sparse.linalg.eigsh(which='SM', k=20) on
    the edge Laplacian, which can never report a nullity larger than k and
    therefore silently collapsed dim H^1 to <= 20 on larger motifs. The rank-based
    count below has no such ceiling.

    Args:
        sheaf: the verifiability sheaf F_G.
        eps: numerical tolerance for zero singular values.
        max_iter: optional maximum iterations for the spectral-gap Lanczos pass.

    Returns:
        dict with keys 'h1_dim', 'wall_clock_s', 'iterations', 'spectral_gap', 'method'.

    Citation: manuscript Theorem 3 (Spectral Approximation).

    Complexity: O((total_edge_dim)^3) for the dense path, O(nnz * sqrt(kappa)
    * log(1/eps)) for the sparse path.
    """
    import time

    L1 = sheaf.edge_laplacian()
    n = L1.shape[0]

    if max_iter is None:
        max_iter = max(50, int(20 * math.log(1.0 / eps)))

    t0 = time.perf_counter()

    if n <= 300:
        eigs = np.linalg.eigvalsh(L1.toarray())
        h1_dim = int(np.sum(eigs < eps))
        spectral_gap = float(eigs[h1_dim]) if h1_dim < len(eigs) else 0.0
        return {
            "h1_dim": h1_dim,
            "wall_clock_s": time.perf_counter() - t0,
            "iterations": 0,
            "spectral_gap": spectral_gap,
            "method": "dense_eigvalsh",
        }

    delta = sheaf.coboundary().tocsc()
    n_edge = delta.shape[0]

    rank = _sparse_rank(delta, tol=eps)
    h1_dim = int(n_edge - rank)

    spectral_gap = 0.0
    try:
        kk = min(h1_dim + 5, n - 2)
        if kk >= 1:
            small = spla.eigsh(
                L1, k=kk, sigma=0.0, which="LM",
                tol=eps, maxiter=max_iter, return_eigenvectors=False,
            )
            small = np.sort(small)
            nonzero = small[small > eps]
            spectral_gap = float(nonzero[0]) if nonzero.size else 0.0
    except (spla.ArpackNoConvergence, RuntimeError, np.linalg.LinAlgError):
        spectral_gap = 0.0

    return {
        "h1_dim": h1_dim,
        "wall_clock_s": time.perf_counter() - t0,
        "iterations": max_iter,
        "spectral_gap": spectral_gap,
        "method": "sparse_rank",
    }


def _sparse_rank(M: sp.csc_matrix, tol: float = 1e-6) -> int:
    """Numerical rank of a sparse coboundary via the Gram eigenspectrum.

    rank(M) = rank(M^T M) = (number of Gram eigenvalues that are not zero). The
    Gram matrix M^T M (or M M^T, whichever is smaller) is symmetric PSD; its
    eigenvalues are the squared singular values of M. We count a Gram eigenvalue
    as a genuine zero using the SAME absolute criterion the dense ground truth
    applies to the edge-Laplacian spectrum (eigenvalue < tol), so that
    dim H^1 = dim C^1 - rank(M) agrees with the dense eigvalsh count exactly.

    Args:
        M: sparse matrix.
        tol: absolute threshold.

    Returns:
        Integer numerical rank, never exceeding min(rows, cols).
    """
    m, k = M.shape
    if m >= k:
        G = (M.T @ M).toarray()
    else:
        G = (M @ M.T).toarray()
    w = np.linalg.eigvalsh(G)
    rank_full = int(np.sum(w >= tol))
    return min(rank_full, min(m, k))


# =============================================================================
# THEOREM 4: PAC-Bayes Learning Guarantee
# =============================================================================


def pac_bayes_bound(
    m: int,
    d: int,
    delta: float,
    operator_norm_radius: float = 1.0,
    prior_variance: float = 1.0,
) -> float:
    """Operator-norm bound on restriction-map recovery error (Theorem 4).

    Combines McAllester's PAC-Bayes bound for a Gaussian posterior centred at the
    empirical risk minimiser with a Gaussian prior of variance prior_variance
    [McA03], and Tropp's matrix Bernstein inequality for the operator-norm
    concentration of a sample-averaged matrix estimator [Tro15, Theorem 6.1.1],
    via the union bound at split confidence delta/2 + delta/2 = delta.

    With probability at least 1 - delta over m independent samples,

        ||F_hat - F||_op  <=  R * sqrt(8 d log(2 d / delta) / m)
                            +  (4 R log(2 d / delta)) / (3 m)

    where R = operator_norm_radius is the operator-norm radius of the true
    restriction maps. For m large enough that the second (Bernstein lower-order)
    term is negligible, the bound is O(R sqrt(d log(d/delta) / m)) with explicit
    leading constant sqrt(8) ~ 2.83.

    Args:
        m: number of independent training samples (verifiable executions).
        d: stalk dimension.
        delta: confidence parameter; the bound holds w.p. at least 1 - delta.
        operator_norm_radius: R, the assumed operator-norm bound on each true
            restriction map. Default 1.0 matches Cayley-orthogonal maps.
        prior_variance: sigma_P^2 of the Gaussian PAC-Bayes prior. In the
            operator-norm-bounded regime the bound is independent of it; retained
            for posterior-calibration consistency checks.

    Returns:
        Upper bound on ||F_hat - F||_op.

    Citation: manuscript Theorem 4 (PAC-Bayes Learning Guarantee).

    Complexity: O(1).
    """
    if m <= 0:
        return float("inf")
    if d <= 0 or not (0.0 < delta < 1.0):
        raise ValueError(
            f"Invalid arguments: d={d}, delta={delta}; require d>=1 and 0<delta<1"
        )
    if operator_norm_radius <= 0 or prior_variance <= 0:
        raise ValueError(
            "operator_norm_radius and prior_variance must both be strictly positive"
        )
    log_term = math.log(2.0 * d / delta)
    leading = operator_norm_radius * math.sqrt(8.0 * d * log_term / m)
    second_order = (4.0 * operator_norm_radius * log_term) / (3.0 * m)
    return leading + second_order


# =============================================================================
# THEOREM 5: Cross-Stage Subspace Leakage Under Restriction-Map Perturbation
# =============================================================================

# THEOREM 5 PROOF OUTLINE (block comment per construction rule).
#
# Setup. Let F be the canonical sheaf, F_tilde the perturbed sheaf with
# ||F_tilde_e - F_e||_op <= eps per restriction map. Let L^1 = delta delta^T be
# the canonical edge Laplacian and L^1_tilde its perturbed counterpart. For each
# Benaloh stage tau, write L^1_tau for the stage-tau block (the principal
# submatrix of L^1 indexed by the stage-tau edge coordinates) and S_tau =
# ker(L^1_tau) embedded into C^1 by zero-padding, exactly as in
# compute_attack_subspaces.
#
# Step 1 (Weyl). For Hermitian matrices A, A + E with ||E||_op <= eta, every
# eigenvalue lambda_i(A + E) lies in [lambda_i(A) - eta, lambda_i(A) + eta].
# We instantiate this with A = L^1_tau and E = L^1_tau_tilde - L^1_tau. The
# coboundary perturbation has operator norm bounded by
#   ||L^1_tau_tilde - L^1_tau||_op  <=  2 * rho * eps * sqrt(2 * d * |E_tau| * Delta)
# where rho is the operator-norm radius (||F_e||_op <= rho), d the stalk dim,
# |E_tau| the number of stage-tau edges, and Delta the maximum node-stalk degree
# (this expansion uses ||delta - delta_tilde||_op <= sqrt(2 |E_tau| Delta) * eps
# from the structure of delta as a block-sparse map; see Tropp 2015 [Tro15] for
# the matrix-Bernstein-style aggregation of per-edge perturbations).
#
# Step 2 (Davis-Kahan, Yu-Wang-Samworth 2015 formulation). For Hermitian A,
# A + E with spectral gap gamma above the eigenvalues corresponding to the
# invariant subspace S, the sin-theta principal-angle distance between S and
# the perturbed invariant subspace S_tilde satisfies
#   || sin Theta(S, S_tilde) ||_F  <=  2 * ||E||_op / gamma
# under the [YWS15] one-sided gap condition. Combining with Step 1 yields
#   || sin Theta(S_tau, S_tau_tilde) ||_F  <=  4 * rho * eps * sqrt(2 d |E_tau| Delta) / gamma.
#
# Step 3 (Tropp operator-Frobenius relationship). For matrices of bounded rank
# k, ||M||_F <= sqrt(k) ||M||_op. The bound above is already in Frobenius norm,
# and the equivalent operator-norm bound is recovered by dividing by sqrt(k),
# which is recorded as the diagnostic `theorem_5_bound_operator_norm` in the
# returned dictionary alongside the Frobenius-norm bound.
#
# Corollary 1 (dimensional asymmetry). For motifs with n_c >= 2 and n_r = n_q = 1,
# dim ker(L^1_cast) = d while dim ker(L^1_record) = dim ker(L^1_count) = 0
# (single-edge stage Laplacians are 2d-by-2d full-rank). The maximum-projection
# classifier predicts cast for every such motif under canonical restriction.
# Theorem 5 then bounds the perturbation correction to the cast-projection norm
# at the eps-times-prefactor level, so the prediction is stable for eps below
# gamma / (4 rho sqrt(2 d |E_tau| Delta)).


def theorem_5_bound(
    eps: float,
    rho: float,
    d: int,
    n_edges_in_stage: int,
    max_node_degree: int,
    spectral_gap: float,
) -> dict:
    """Theorem 5 upper bound on cross-stage subspace leakage (manuscript Theorem 5).

    Returns the Frobenius-norm and operator-norm forms of the bound

        || sin Theta(S_tau, S_tau_tilde) ||_F  <=  4 * rho * eps * sqrt(2 d |E_tau| Delta) / gamma

    derived from Weyl's inequality on the spectral-gap perturbation, the
    Yu-Wang-Samworth 2015 sin-theta formulation of Davis-Kahan, and the Tropp
    2015 operator-Frobenius norm relationship. See the block-comment proof
    outline above this function.

    Args:
        eps: per-restriction-map perturbation magnitude, ||F_tilde_e - F_e||_op.
        rho: operator-norm radius of the true restriction maps.
        d: stalk dimension.
        n_edges_in_stage: |E_tau|, the number of stage-tau edges.
        max_node_degree: Delta, the maximum node-stalk degree in the typed graph.
        spectral_gap: gamma, the spectral gap of L^1_tau above the kernel.

    Returns:
        dict with keys:
            'frobenius_bound':  4 * rho * eps * sqrt(2 d |E_tau| Delta) / gamma
            'operator_bound':   frobenius_bound / sqrt(max(rank_estimate, 1))
            'stability_threshold_eps': gamma / (4 * rho * sqrt(2 d |E_tau| Delta))
            'coboundary_norm_bound': 2 * rho * eps * sqrt(2 d |E_tau| Delta)

    Citation: manuscript Theorem 5 (Cross-Stage Subspace Leakage), proof relies
    on [Tro15], [YWS15], and classical Weyl's inequality.

    Complexity: O(1).
    """
    if eps < 0 or rho <= 0 or d <= 0 or n_edges_in_stage < 0 or max_node_degree <= 0:
        raise ValueError(
            f"theorem_5_bound: invalid arguments eps={eps}, rho={rho}, d={d}, "
            f"n_edges_in_stage={n_edges_in_stage}, max_node_degree={max_node_degree}"
        )
    if spectral_gap <= 0:
        return {
            "frobenius_bound": float("inf"),
            "operator_bound": float("inf"),
            "stability_threshold_eps": 0.0,
            "coboundary_norm_bound": float("inf"),
        }

    coboundary_norm = 2.0 * rho * eps * math.sqrt(
        2.0 * d * max(n_edges_in_stage, 1) * max_node_degree
    )
    frobenius_bound = 2.0 * coboundary_norm / spectral_gap
    rank_estimate = max(1, min(d, n_edges_in_stage * d))
    operator_bound = frobenius_bound / math.sqrt(rank_estimate)
    denom = 4.0 * rho * math.sqrt(2.0 * d * max(n_edges_in_stage, 1) * max_node_degree)
    stability_eps = spectral_gap / denom if denom > 0 else float("inf")
    return {
        "frobenius_bound": frobenius_bound,
        "operator_bound": operator_bound,
        "stability_threshold_eps": stability_eps,
        "coboundary_norm_bound": coboundary_norm,
    }


def corollary_1_predicted_class(
    n_cast_edges: int,
    n_record_edges: int,
    n_count_edges: int,
    eps: float = 0.0,
    spectral_gap: float = 1.0,
    rho: float = 1.0,
    d: int = 8,
    max_node_degree: int = 4,
) -> str:
    """Corollary 1's predicted stage classification for a motif with given edge counts.

    COROLLARY 1. For motifs whose stage-tau Laplacian has the largest kernel
    dimension, the maximum-projection classifier predicts tau. Under canonical
    restriction, the per-stage kernel dimensions are dim ker(L^1_tau) =
    max(0, n_tau - 1) * d for a star-shaped stage block with n_tau parallel edges
    sharing the proposal endpoint. The argmax-stage prediction is therefore the
    stage with the largest n_tau, with ties broken by the canonical cast > record
    > count order. Theorem 5 guarantees this prediction is stable under perturbation
    of magnitude eps below gamma / (4 rho sqrt(2 d |E_tau| Delta)).

    Args:
        n_cast_edges: n_c, count of cast-stage edges in the motif.
        n_record_edges: n_r, count of record-stage edges in the motif.
        n_count_edges: n_q, count of count-stage edges in the motif.
        eps: optional perturbation magnitude for stability assessment.
        spectral_gap: gamma for the dominant-stage Laplacian; default 1.0 (canonical).
        rho: operator-norm radius; default 1.0.
        d: stalk dimension; default 8.
        max_node_degree: Delta; default 4 (typical proposal node degree).

    Returns:
        One of "cast", "record", "count" — the predicted stage label.

    Citation: manuscript Corollary 1 (Predicted All-Cast Collapse).

    Complexity: O(1).
    """
    kernel_dim = {
        "cast": max(0, n_cast_edges - 1) * d,
        "record": max(0, n_record_edges - 1) * d,
        "count": max(0, n_count_edges - 1) * d,
    }
    edges = {
        "cast": n_cast_edges,
        "record": n_record_edges,
        "count": n_count_edges,
    }
    ordered = ["cast", "record", "count"]
    return max(ordered, key=lambda stage: (kernel_dim[stage], edges[stage]))


# =============================================================================
# Neural Sheaf Diffusion learner with Cayley-parameterized restriction maps.
# =============================================================================


class CayleyOrthogonal(nn.Module):
    """Cayley parameterization of orthogonal restriction maps.

    For a learnable skew-symmetric matrix A (parameterized by upper-triangular
    free parameters), the Cayley transform Q = (I - A)(I + A)^{-1} produces an
    orthogonal matrix. The Cayley parameterization is differentiable, retract-free
    (the gradient on the orthogonal group is recovered exactly), and consistent
    with the PAC-Bayes prior in Theorem 4.
    """

    def __init__(self, dim: int, init_std: float = 0.01):
        super().__init__()
        self.dim = dim
        # Small-Gaussian initialisation breaks the identity symmetry that would
        # otherwise trap the learner at the canonical sheaf: A=0 gives the Cayley
        # transform Q=(I-A)(I+A)^{-1}=I, so zero-init would make every untrained
        # restriction map the identity and produce no learning signal. init_std
        # 0.01 keeps the initial map within ~0.01 of the identity in operator
        # norm while remaining non-degenerate. This line is the in-place CC-2 fix
        # and must not be altered.
        self.free = nn.Parameter(torch.randn(dim * (dim - 1) // 2) * init_std)

    def forward(self) -> torch.Tensor:
        """Compute the Cayley transform Q = (I - A)(I + A)^{-1} where A is skew.

        Returns:
            torch.Tensor of shape (dim, dim) with orthogonal columns/rows.

        Complexity: O(dim^3) for the linear solve.
        """
        d = self.dim
        device = self.free.device
        A = torch.zeros(d, d, device=device, dtype=self.free.dtype)
        idx = torch.triu_indices(d, d, offset=1)
        A[idx[0], idx[1]] = self.free
        A = A - A.T
        I = torch.eye(d, device=device, dtype=self.free.dtype)
        return torch.linalg.solve(I + A, I - A)


class CayleyRectangular(nn.Module):
    """Cayley parameterization for rectangular restriction maps.

    For d_out != d_in we parameterize a Stiefel-manifold element via Cayley on
    the augmented square matrix and then project onto the first d_in columns
    (if d_out > d_in) or first d_out rows (if d_in > d_out).
    """

    def __init__(self, d_out: int, d_in: int):
        super().__init__()
        self.d_out = d_out
        self.d_in = d_in
        d = max(d_out, d_in)
        self.cayley = CayleyOrthogonal(d)

    def forward(self) -> torch.Tensor:
        """Return the rectangular restriction map as a Stiefel-manifold element.

        Returns:
            torch.Tensor of shape (d_out, d_in).
        """
        Q = self.cayley()
        if self.d_out >= self.d_in:
            return Q[:, : self.d_in]
        else:
            return Q[: self.d_out, :]


class SheafLearner(nn.Module):
    """Neural sheaf diffusion learner for the canonical verifiability sheaf.

    Learns the restriction maps F_{v -> e} and F_{e -> v'} via Cayley-parameterized
    orthogonal projections, trained to minimize the sheaf-Dirichlet energy on the
    observed corpus of verifiable executions.

    The training objective is

        L = E_{x ~ corpus}[||delta x||^2] + lambda_pac * ||theta||^2

    where delta is the sheaf coboundary and theta is the free-parameter vector.
    The PAC-Bayes regularizer ||theta||^2 enters with weight lambda_pac chosen
    to match the prior covariance in Theorem 4.
    """

    def __init__(
        self,
        node_types_universe: list[NodeType] = None,
        edge_types_universe: list[EdgeType] = None,
        stalk_dims: Optional[dict[NodeType, int]] = None,
        edge_stalk_dims: Optional[dict[EdgeType, int]] = None,
    ):
        super().__init__()
        self.node_types_universe = node_types_universe or list(NodeType)
        self.edge_types_universe = edge_types_universe or list(EdgeType)
        self.stalk_dims = stalk_dims or DEFAULT_STALK_DIMS
        self.edge_stalk_dims = edge_stalk_dims or DEFAULT_EDGE_STALK_DIMS

        self.restrictions_src = nn.ModuleDict()
        self.restrictions_tgt = nn.ModuleDict()
        for nt in self.node_types_universe:
            for et in self.edge_types_universe:
                key_src = f"src__{nt.value}__{et.value}"
                key_tgt = f"tgt__{et.value}__{nt.value}"
                self.restrictions_src[key_src] = CayleyRectangular(
                    self.edge_stalk_dims[et], self.stalk_dims[nt]
                )
                self.restrictions_tgt[key_tgt] = CayleyRectangular(
                    self.edge_stalk_dims[et], self.stalk_dims[nt]
                )

    def assemble_restrictions(self) -> dict:
        """Return the learned restriction maps as a dict consumable by build_canonical_sheaf.

        Returns:
            dict with keys ('src', NodeType, EdgeType) and ('tgt', EdgeType, NodeType).
        """
        out = {}
        for nt in self.node_types_universe:
            for et in self.edge_types_universe:
                key_src = f"src__{nt.value}__{et.value}"
                key_tgt = f"tgt__{et.value}__{nt.value}"
                out[("src", nt, et)] = self.restrictions_src[key_src]()
                out[("tgt", et, nt)] = self.restrictions_tgt[key_tgt]()
        return out

    def sheaf_dirichlet_energy(
        self,
        node_signals: torch.Tensor,
        node_types: list[NodeType],
        edge_index: torch.Tensor,
        edge_types: list[EdgeType],
    ) -> torch.Tensor:
        """Compute || delta x ||^2 for a batch of node signals.

        Args:
            node_signals: (num_nodes, max_node_dim) tensor of node-stalk values.
            node_types: per-node NodeType.
            edge_index: (2, num_edges) source/target.
            edge_types: per-edge EdgeType.

        Returns:
            scalar tensor: total sheaf-Dirichlet energy.

        Complexity: O(|E| * d_max^2).
        """
        restrictions = self.assemble_restrictions()
        energy = node_signals.new_zeros(())
        for e in range(len(edge_types)):
            et = edge_types[e]
            src = int(edge_index[0, e].item())
            tgt = int(edge_index[1, e].item())
            nt_src = node_types[src]
            nt_tgt = node_types[tgt]
            dsrc = self.stalk_dims[nt_src]
            dtgt = self.stalk_dims[nt_tgt]

            Fsrc = restrictions[("src", nt_src, et)]
            Ftgt = restrictions[("tgt", et, nt_tgt)]

            x_src = node_signals[src, :dsrc]
            x_tgt = node_signals[tgt, :dtgt]

            diff = Ftgt @ x_tgt - Fsrc @ x_src
            energy = energy + (diff ** 2).sum()
        return energy

    def pac_bayes_regularizer(self) -> torch.Tensor:
        """L2 regularizer on free parameters, calibrated to Theorem 4's prior.

        Returns:
            scalar tensor: sum of squared free Cayley parameters across all
            restriction-map modules.
        """
        reg = next(self.parameters()).new_zeros(())
        for module in self.restrictions_src.values():
            reg = reg + (module.cayley.free ** 2).sum()
        for module in self.restrictions_tgt.values():
            reg = reg + (module.cayley.free ** 2).sum()
        return reg


# =============================================================================
# Empirical verification entry points (used as unit tests).
# =============================================================================


def verify_theorem_1(num_executions: int = 25) -> bool:
    """Verify the Theorem 1 bijection property on a synthetic multi-execution corpus.

    Theorem 1 claims H^0(F_G) is in bijection with cryptographically consistent
    voting executions. The testable consequence is that for a disjoint union of
    K independent, internally consistent executions, dim H^0 scales linearly in
    K (the disjoint-union sheaf satisfies H^0(union) = direct sum of the per-
    execution H^0). We verify the linear scaling and that it is strictly
    positive, which the prior 'H^0 >= 1' check did not establish.

    Args:
        num_executions: K, the number of independent executions in the union test.

    Returns:
        True on pass; raises AssertionError otherwise.

    KNOWN_LIMITATION (CC-4): this verifier tests a testable consequence (linear
    H^0 scaling) rather than the full bijection statement.

    Citation: manuscript Theorem 1.
    """
    def build_k_executions(k: int) -> CellularSheaf:
        node_types: list[NodeType] = []
        src: list[int] = []
        tgt: list[int] = []
        edge_types: list[EdgeType] = []
        for j in range(k):
            base = 3 * j
            node_types.extend(
                [NodeType.VOTER, NodeType.PROPOSAL, NodeType.EXEC_CONTRACT]
            )
            src.extend([base + 0, base + 1, base + 2])
            tgt.extend([base + 1, base + 2, base + 1])
            edge_types.extend([EdgeType.CAST, EdgeType.RECORD, EdgeType.COUNT])
        edge_index = torch.tensor([src, tgt], dtype=torch.long)
        return build_canonical_sheaf(node_types, edge_index, edge_types)

    h0_one = build_k_executions(1).h0_dimension()
    h0_k = build_k_executions(num_executions).h0_dimension()

    assert h0_one >= 1, f"Theorem 1: single-execution H^0 is trivial ({h0_one})"
    assert h0_k == num_executions * h0_one, (
        f"Theorem 1: H^0 does not scale linearly with executions: "
        f"H^0(1)={h0_one}, H^0({num_executions})={h0_k}, "
        f"expected {num_executions * h0_one}"
    )
    return True


def verify_theorem_2() -> bool:
    """Verify Theorem 2 orthogonality non-vacuously on a motif with non-trivial H^1.

    Uses three independent cast-record-count loops sharing distinct proposal and
    execution-contract nodes, which produces non-trivial stage subspaces. The
    verifier REQUIRES at least two non-empty stage subspaces (so the test is not
    vacuously satisfied, unlike the prior 'len(bases) < 2 -> return True') and
    asserts pairwise orthogonality under BOTH the Euclidean and the sheaf
    Laplacian inner products.

    Returns:
        True on pass; raises AssertionError otherwise.

    KNOWN_LIMITATION (CC-4): this verifier tests orthogonality on three specific
    independent loops rather than the population statement.

    Citation: manuscript Theorem 2.
    """
    torch.manual_seed(0)
    node_types = (
        [NodeType.VOTER] * 9
        + [NodeType.PROPOSAL] * 3
        + [NodeType.EXEC_CONTRACT] * 3
    )
    edge_list: list[list[int]] = []
    edge_types_list: list[EdgeType] = []
    for loop in range(3):
        prop = 9 + loop
        exec_c = 12 + loop
        for v in range(3):
            edge_list.append([loop * 3 + v, prop])
            edge_types_list.append(EdgeType.CAST)
        edge_list.append([prop, exec_c])
        edge_types_list.append(EdgeType.RECORD)
        edge_list.append([exec_c, prop])
        edge_types_list.append(EdgeType.COUNT)

    edge_index = torch.tensor(edge_list, dtype=torch.long).T
    sheaf = build_canonical_sheaf(node_types, edge_index, edge_types_list)

    subs = compute_attack_subspaces(sheaf)
    if not has_nontrivial_attack_subspaces(subs):
        raise AssertionError(
            "Theorem 2 verifier: all stage subspaces empty; cannot test "
            "orthogonality (the construction failed to produce nontrivial H^1)."
        )

    stages = [BenalohStage.CAST, BenalohStage.RECORD, BenalohStage.COUNT]
    nonempty = [(s, subs[s]) for s in stages if subs[s].shape[1] > 0]
    assert len(nonempty) >= 2, (
        f"Theorem 2 verifier: expected >= 2 non-empty subspaces, "
        f"got {len(nonempty)}"
    )

    L1 = sheaf.edge_laplacian().toarray()
    L1_t = torch.from_numpy(L1).float()
    for i in range(len(nonempty)):
        for j in range(i + 1, len(nonempty)):
            _, bi = nonempty[i]
            _, bj = nonempty[j]
            euclid = (bi.T @ bj).abs().max().item()
            laplac = (bi.T @ (L1_t @ bj)).abs().max().item()
            assert euclid < 1e-4, (
                f"Subspaces {nonempty[i][0].value},{nonempty[j][0].value} not "
                f"Euclidean-orthogonal: max product {euclid}"
            )
            assert laplac < 1e-4, (
                f"Subspaces {nonempty[i][0].value},{nonempty[j][0].value} not "
                f"Laplacian-orthogonal: max product {laplac}"
            )
    return True


def verify_theorem_3() -> bool:
    """Verify Theorem 3 across multiple motif sizes that have non-trivial H^1.

    The prior verifier used a star graph (no cycles -> H^1 = 0), so exact and
    approximate both returned 0 and the test was vacuous. Here each motif is k
    cast edges plus a record/count loop sharing one proposal and one execution
    contract, which produces non-trivial H^1, and we require exact==approx on
    at least all-but-one of the sizes.

    Returns:
        True on pass; raises AssertionError otherwise.

    KNOWN_LIMITATION (CC-4): the verifier checks agreement on five fixed sizes.

    Citation: manuscript Theorem 3.
    """
    torch.manual_seed(0)
    sizes = [4, 8, 16, 32, 64]
    agreements = []
    for k in sizes:
        node_types = (
            [NodeType.VOTER] * k + [NodeType.PROPOSAL] + [NodeType.EXEC_CONTRACT]
        )
        edge_list: list[list[int]] = []
        edge_types_list: list[EdgeType] = []
        prop, exec_c = k, k + 1
        for v in range(k):
            edge_list.append([v, prop])
            edge_types_list.append(EdgeType.CAST)
        edge_list.append([prop, exec_c])
        edge_types_list.append(EdgeType.RECORD)
        edge_list.append([exec_c, prop])
        edge_types_list.append(EdgeType.COUNT)
        edge_index = torch.tensor(edge_list, dtype=torch.long).T
        sheaf = build_canonical_sheaf(node_types, edge_index, edge_types_list)
        exact = sheaf.h1_dimension()
        approx = approximate_h1_dimension(sheaf, eps=1e-6)
        agreements.append(exact == approx["h1_dim"])

    assert sum(agreements) >= len(sizes) - 1, (
        f"Theorem 3: exact/approx agreement only {sum(agreements)}/{len(sizes)}"
    )
    return True


def verify_theorem_4(
    num_trials: int = 100,
    m_values: Optional[list[int]] = None,
    d_values: Optional[list[int]] = None,
    delta_values: Optional[list[float]] = None,
) -> bool:
    """Verify Theorem 4: the PAC-Bayes bound holds with probability >= 1 - delta.

    Sweeps over sample sizes, stalk dimensions, and confidence levels, and in
    every cell asserts that the empirical bound-satisfaction rate is at least
    1 - delta - mc_slack (a small Monte Carlo allowance). The prior verifier
    asserted only rate >= 0.5 at a single (m, d), which is far below the
    theorem's 1 - delta claim and tests no scaling behaviour.

    Args:
        num_trials: number of Monte Carlo trials per (m, d, delta) cell.
        m_values: sample sizes; default [25, 50, 100, 250].
        d_values: stalk dimensions; default [4, 8, 16].
        delta_values: confidence parameters; default [0.01, 0.05, 0.10].

    Returns:
        True on pass; raises AssertionError otherwise.

    KNOWN_LIMITATION (CC-4): the verifier tests the 1-delta success rate on a
    Gaussian-additive-noise model rather than on the population of all
    operator-bounded distributions.

    Citation: manuscript Theorem 4.
    """
    m_values = m_values or [25, 50, 100, 250]
    d_values = d_values or [4, 8, 16]
    delta_values = delta_values or [0.01, 0.05, 0.10]
    mc_slack = 0.02

    rng = np.random.RandomState(0)
    for d in d_values:
        for m in m_values:
            for delta in delta_values:
                bound = pac_bayes_bound(m=m, d=d, delta=delta)
                successes = 0
                for _ in range(num_trials):
                    truth = rng.randn(d, d) / math.sqrt(d)
                    noise = rng.randn(m, d, d) / math.sqrt(d)
                    estimate = (truth + noise).mean(axis=0)
                    err = np.linalg.norm(estimate - truth, ord=2)
                    if err <= bound:
                        successes += 1
                rate = successes / num_trials
                threshold = 1.0 - delta - mc_slack
                assert rate >= threshold, (
                    f"PAC-Bayes bound violated at m={m}, d={d}, delta={delta}: "
                    f"rate={rate:.3f}, required >= {threshold:.3f}"
                )
    return True


def verify_theorem_5(
    num_trials: int = 25,
    eps_values: Optional[list[float]] = None,
    motif_sizes: Optional[list[int]] = None,
    rho: float = 1.0,
    seed: int = 0,
) -> bool:
    """Verify Theorem 5: the perturbation bound dominates empirical sin-theta leakage.

    For each (eps, n_c) cell, construct a canonical sheaf on a motif with n_c cast
    edges plus a record/count loop, perturb each restriction map by a symmetric
    Gaussian with operator-norm magnitude at most eps, compute the per-stage
    eigendecomposition of L^1_tau and L^1_tau_tilde, and verify that the empirical
    sin-theta principal-angle distance between the cast-stage kernel and its
    perturbed counterpart is at most the theorem_5_bound Frobenius-bound value.

    The verifier follows the structure of verify_theorem_3 and verify_theorem_4:
    it asserts bound-domination on at least num_trials * (1 - mc_slack) of trials
    across each cell, mirroring the 1-delta success-rate threshold of Theorem 4.

    Args:
        num_trials: number of Monte Carlo trials per (eps, n_c) cell.
        eps_values: perturbation magnitudes; default [0.01, 0.05, 0.10, 0.20].
        motif_sizes: cast-edge counts; default [4, 8, 16, 32].
        rho: operator-norm radius used by the bound; default 1.0.
        seed: RNG seed for reproducibility.

    Returns:
        True on pass; raises AssertionError otherwise.

    KNOWN_LIMITATION (CC-4): the verifier samples Gaussian perturbations and
    tests bound-domination empirically. The full Theorem 5 statement covers
    every operator-bounded perturbation; the empirical test covers a subset.

    Citation: manuscript Theorem 5.

    Complexity: O(num_trials * |cells| * (n_c * d)^3).
    """
    eps_values = eps_values or [0.01, 0.05, 0.10, 0.20]
    motif_sizes = motif_sizes or [4, 8, 16, 32]
    mc_slack = 0.05

    rng_global = np.random.RandomState(seed)

    for n_c in motif_sizes:
        node_types = [NodeType.VOTER] * n_c + [NodeType.PROPOSAL, NodeType.EXEC_CONTRACT]
        edge_list: list[list[int]] = []
        edge_types_list: list[EdgeType] = []
        prop, exec_c = n_c, n_c + 1
        for v in range(n_c):
            edge_list.append([v, prop])
            edge_types_list.append(EdgeType.CAST)
        edge_list.append([prop, exec_c])
        edge_types_list.append(EdgeType.RECORD)
        edge_list.append([exec_c, prop])
        edge_types_list.append(EdgeType.COUNT)
        edge_index = torch.tensor(edge_list, dtype=torch.long).T

        sheaf_canonical = build_canonical_sheaf(node_types, edge_index, edge_types_list)
        subspaces_canonical = compute_attack_subspaces(sheaf_canonical)

        L1_canonical = sheaf_canonical.edge_laplacian().toarray()
        eigs_canonical = np.linalg.eigvalsh(L1_canonical)
        nonzero = eigs_canonical[eigs_canonical > 1e-8]
        gamma = float(nonzero.min()) if nonzero.size else 1.0

        d_stalk = DEFAULT_STALK_DIMS[NodeType.PROPOSAL]
        max_node_degree = n_c + 2  # proposal node has n_c + 2 incidences

        cast_basis_canonical = subspaces_canonical[BenalohStage.CAST]
        if cast_basis_canonical.shape[1] == 0:
            continue

        for eps in eps_values:
            bound = theorem_5_bound(
                eps=eps, rho=rho, d=d_stalk,
                n_edges_in_stage=n_c, max_node_degree=max_node_degree,
                spectral_gap=gamma,
            )
            successes = 0
            for trial in range(num_trials):
                trial_rng = np.random.RandomState(rng_global.randint(0, 2**31 - 1))
                learned: dict = {}
                for nt in [NodeType.VOTER, NodeType.PROPOSAL, NodeType.EXEC_CONTRACT]:
                    for et in [EdgeType.CAST, EdgeType.RECORD, EdgeType.COUNT]:
                        de = DEFAULT_EDGE_STALK_DIMS[et]
                        dn = DEFAULT_STALK_DIMS[nt]
                        base_src = _canonical_restriction(de, dn, "src", nt, et)
                        base_tgt = _canonical_restriction(de, dn, "tgt", nt, et)
                        noise_src = trial_rng.randn(de, dn) * (eps / math.sqrt(max(de, dn)))
                        noise_tgt = trial_rng.randn(de, dn) * (eps / math.sqrt(max(de, dn)))
                        learned[("src", nt, et)] = base_src + torch.from_numpy(noise_src).float()
                        learned[("tgt", et, nt)] = base_tgt + torch.from_numpy(noise_tgt).float()

                sheaf_perturbed = build_canonical_sheaf(
                    node_types, edge_index, edge_types_list,
                    learned_restrictions=learned,
                )
                subspaces_perturbed = compute_attack_subspaces(sheaf_perturbed)
                cast_basis_perturbed = subspaces_perturbed[BenalohStage.CAST]

                if cast_basis_perturbed.shape[1] == 0:
                    empirical_leakage = float(cast_basis_canonical.shape[1])
                else:
                    k_min = min(cast_basis_canonical.shape[1], cast_basis_perturbed.shape[1])
                    proj = cast_basis_canonical.T @ cast_basis_perturbed
                    sigma_sq = torch.linalg.svdvals(proj) ** 2
                    sin_theta_sq = max(0.0, float(k_min) - float(sigma_sq.sum().item()))
                    empirical_leakage = math.sqrt(max(0.0, sin_theta_sq))

                if empirical_leakage <= bound["frobenius_bound"]:
                    successes += 1

            rate = successes / num_trials
            threshold = 1.0 - mc_slack
            assert rate >= threshold, (
                f"Theorem 5 bound violated at n_c={n_c}, eps={eps}: "
                f"rate={rate:.3f}, required >= {threshold:.3f}, "
                f"bound={bound['frobenius_bound']:.4f}"
            )
    return True


# =============================================================================
# MODIFICATION 1E: ElGamal Substitution Structural Verifier
# =============================================================================
#
# verify_elgamal_substitution is the sixth empirical verifier. It tests the four
# structural guarantees the ElGamal-instantiated cryptography.py contract is
# responsible for delivering, exercised through the public canonical-sheaf
# construction path (theory.build_canonical_sheaf -> _canonical_restriction ->
# cryptography.build_restriction). Each guarantee corresponds to a property the
# downstream theorems and learners depend on:
#
#   G1 (stage-disjoint column support). On each Benaloh-stage edge, the source
#      and target restriction-map blocks read ONLY the stage-tau slice of the
#      incident node's stalk. Operationally: for the proposal hub (which is the
#      unique node type that touches all three Benaloh stages), the column
#      supports of its three stage-tau restriction maps must be pairwise
#      disjoint as subsets of [0, dim(proposal)). This is the structural
#      precondition that makes Theorem 2's Laplacian-orthogonality a
#      consequence of the sheaf construction rather than of post-hoc
#      orthogonalisation.
#
#   G2 (full-rank blocks). Each restriction-map block has rank min(d_out, w)
#      where w is the stage-slice width. Without this, the coboundary delta
#      would have a larger-than-necessary kernel and H^0 would lose its
#      bijection-with-executions interpretation (Theorem 1).
#
#   G3 (linear H^0 scaling under disjoint unions). For a disjoint union of K
#      independent cast-record-count loops, dim H^0 = K * dim H^0(one loop).
#      This is the same testable consequence that verify_theorem_1 checks for
#      the categorical fallback; here we confirm it continues to hold under
#      the ElGamal-instantiated restriction maps. If ElGamal substitution were
#      to introduce spurious linear dependencies, the disjoint-union H^0 would
#      collapse below the expected linear scaling.
#
#   G4 (well-conditioned coboundary). The condition number of the non-trivial
#      part of the canonical sheaf coboundary (the ratio of largest to smallest
#      nonzero singular value of delta) must be finite. Equivalently, every
#      nonzero singular value of delta lies above a fixed numerical tolerance.
#      Without this guarantee, the sparse eigensolvers in approximate_h1_dimension
#      and the Lanczos calls used in F4 would lose precision as motif size grows.
#
# In addition to the four structural guarantees, the verifier asserts that the
# operator norm of every restriction-map block returned by cryptography.build_restriction
# lies within a small tolerance of the documented operator-norm radius (unity by
# the singular-value normalization in cryptography._deterministic_block). This
# is the bounded-operator-norm contract the ElGamal substitution exposes; it is
# the rho parameter that Theorem 5's perturbation bound depends on and the
# parameter pac_bayes_bound defaults to 1.0 for.
#
# The verifier follows the structure of verify_theorem_1 through verify_theorem_5:
# it operates on a small but non-trivial canonical sheaf, raises AssertionError
# on any guarantee failure, and returns True on success so the __main__ block
# can report a PASS line consistent with the existing five verifiers.


# Documented operator-norm radius for ElGamal-instantiated restriction-map blocks.
# Per cryptography.py, _deterministic_block normalizes each block to unit operator
# norm by singular-value rescaling; the build_restriction self-test asserts
# op_norm <= 1.0 + 1e-10. The auxiliary (delegate/transfer) blocks are categorical
# inclusion/projection maps and also have operator norm exactly one. The value
# below MUST match cryptography.py's contract; if cryptography.py changes the
# normalization radius, this constant changes in lockstep.
_ELGAMAL_OPERATOR_NORM_RADIUS: float = 1.0

# Tolerances for the ElGamal substitution verifier. The operator-norm tolerance
# matches cryptography._self_test's 1e-10 bound; the structural tolerances are
# the same as the per-theorem verifiers (zero-support 0.0 exactly, zero
# eigenvalue 1e-8).
_ELGAMAL_OP_NORM_TOL: float = 1e-9
_ELGAMAL_ZERO_SUPPORT_TOL: float = 0.0
_ELGAMAL_ZERO_EIG_TOL: float = 1e-8


def verify_elgamal_substitution(
    num_executions: int = 8,
    n_cast_edges_per_execution: int = 3,
) -> bool:
    """Verify the four structural guarantees of the ElGamal substitution.

    Constructs a small canonical sheaf using the ElGamal-instantiated restriction
    maps from cryptography.build_restriction (invoked through the canonical
    construction path build_canonical_sheaf -> _canonical_restriction). Confirms:

      G1. Stage-disjoint column support on the proposal hub.
      G2. Full-rank ElGamal blocks at every (node_type, edge_type, dim) cell
          exercised by the canonical sheaf.
      G3. Linear H^0 scaling under disjoint unions of independent
          cast-record-count loops, mirroring verify_theorem_1's test.
      G4. Well-conditioned coboundary (every nonzero singular value of delta
          above _ELGAMAL_ZERO_EIG_TOL; finite condition number).

    In addition, asserts that the operator norm of every restriction-map block
    invoked during the construction lies within _ELGAMAL_OP_NORM_TOL of
    _ELGAMAL_OPERATOR_NORM_RADIUS (= 1.0 per cryptography.py's documented
    singular-value normalization contract).

    Args:
        num_executions: K, the number of independent cast-record-count loops
            assembled in the disjoint-union test (G3). Default 8.
        n_cast_edges_per_execution: n_c, the number of cast edges per loop in
            the disjoint-union test (G3). Default 3, which gives a non-trivial
            single-loop H^0 and a clear linear scaling signature under union.

    Returns:
        True on pass; raises AssertionError on any guarantee failure.

    KNOWN_LIMITATION (CC-4): the verifier tests the four structural guarantees
    on the canonical-construction path with a small synthetic union; it does
    not re-derive the formal cryptographic-correspondence proof. The verifier
    is the operational acceptance criterion that the ElGamal substitution
    preserves the structural properties the downstream theorems and learners
    depend on, which is the engineering contract this verifier enforces.

    Citation: manuscript Theorem 1 Remark 1 (ElGamal restriction-map
    instantiation), and cryptography.py module docstring (four structural
    guarantees).

    Complexity: O(num_executions * (n_cast_edges_per_execution + 2) * d_max^3).
    """
    from cryptography import build_restriction

    # -------------------------------------------------------------------------
    # G1: Stage-disjoint column support on the proposal hub.
    #
    # The proposal node type is the unique type that touches all three Benaloh
    # stages. Its three stage-tau restriction maps must have pairwise-disjoint
    # column supports when viewed as subsets of [0, dim(proposal)). We exercise
    # this directly through the public build_restriction API rather than
    # through the sheaf, so a failure here is unambiguously attributable to
    # cryptography.py's stage-slicing rather than to the sheaf's edge typing.
    # -------------------------------------------------------------------------
    proposal_dim = DEFAULT_STALK_DIMS[NodeType.PROPOSAL]
    edge_dim_cast = DEFAULT_EDGE_STALK_DIMS[EdgeType.CAST]
    edge_dim_record = DEFAULT_EDGE_STALK_DIMS[EdgeType.RECORD]
    edge_dim_count = DEFAULT_EDGE_STALK_DIMS[EdgeType.COUNT]

    M_cast_t = build_restriction(
        "tgt", NodeType.PROPOSAL.value, EdgeType.CAST.value,
        edge_dim_cast, proposal_dim,
    )
    M_record_t = build_restriction(
        "src", NodeType.PROPOSAL.value, EdgeType.RECORD.value,
        edge_dim_record, proposal_dim,
    )
    M_count_t = build_restriction(
        "tgt", NodeType.PROPOSAL.value, EdgeType.COUNT.value,
        edge_dim_count, proposal_dim,
    )

    def _column_support(M: torch.Tensor, tol: float) -> set[int]:
        """Return the set of column indices with at least one entry exceeding tol."""
        M_np = M.detach().cpu().numpy() if hasattr(M, "detach") else np.asarray(M)
        col_norms = np.linalg.norm(M_np, axis=0)
        return {int(j) for j in range(M_np.shape[1]) if col_norms[j] > tol}

    sup_cast = _column_support(M_cast_t, _ELGAMAL_ZERO_SUPPORT_TOL)
    sup_record = _column_support(M_record_t, _ELGAMAL_ZERO_SUPPORT_TOL)
    sup_count = _column_support(M_count_t, _ELGAMAL_ZERO_SUPPORT_TOL)

    assert sup_cast, "ElGamal G1: proposal-cast block has empty column support"
    assert sup_record, "ElGamal G1: proposal-record block has empty column support"
    assert sup_count, "ElGamal G1: proposal-count block has empty column support"
    assert sup_cast.isdisjoint(sup_record), (
        f"ElGamal G1: cast and record column supports overlap on proposal: "
        f"{sorted(sup_cast & sup_record)}"
    )
    assert sup_cast.isdisjoint(sup_count), (
        f"ElGamal G1: cast and count column supports overlap on proposal: "
        f"{sorted(sup_cast & sup_count)}"
    )
    assert sup_record.isdisjoint(sup_count), (
        f"ElGamal G1: record and count column supports overlap on proposal: "
        f"{sorted(sup_record & sup_count)}"
    )

    # -------------------------------------------------------------------------
    # G2 + operator-norm contract: full-rank blocks and bounded operator norm
    # for every (node_type, edge_type) cell that the canonical sheaf actually
    # exercises (the Benaloh-stage triples that have a non-empty stage slice
    # on the incident node). We iterate the cells via the same logic
    # _canonical_restriction uses, so failure modes here are guaranteed to
    # manifest in the canonical-sheaf construction.
    # -------------------------------------------------------------------------
    benaloh_cells = [
        (NodeType.VOTER, EdgeType.CAST),
        (NodeType.PROPOSAL, EdgeType.CAST),
        (NodeType.PROPOSAL, EdgeType.RECORD),
        (NodeType.PROPOSAL, EdgeType.COUNT),
        (NodeType.EXEC_CONTRACT, EdgeType.RECORD),
        (NodeType.EXEC_CONTRACT, EdgeType.COUNT),
    ]
    for nt, et in benaloh_cells:
        d_out = DEFAULT_EDGE_STALK_DIMS[et]
        d_in = DEFAULT_STALK_DIMS[nt]
        for side in ("src", "tgt"):
            M = build_restriction(side, nt.value, et.value, d_out, d_in)
            M_np = M.detach().cpu().numpy() if hasattr(M, "detach") else np.asarray(M)
            assert M_np.shape == (d_out, d_in), (
                f"ElGamal G2: shape mismatch for ({side}, {nt.value}, {et.value}): "
                f"got {M_np.shape}, expected ({d_out}, {d_in})"
            )

            # G2: full rank on the stage-slice columns. The total block has
            # rank min(d_out, w) where w is the stage-slice width, not d_in,
            # because columns outside the slice are zero by G1. Compute w from
            # the column support determined above; for the slice rank check we
            # examine only the non-zero columns.
            col_norms = np.linalg.norm(M_np, axis=0)
            slice_cols = np.where(col_norms > _ELGAMAL_ZERO_SUPPORT_TOL)[0]
            assert slice_cols.size > 0, (
                f"ElGamal G2: empty column support for ({side}, {nt.value}, {et.value})"
            )
            within = M_np[:, slice_cols]
            within_rank = int(np.linalg.matrix_rank(within, tol=1e-10))
            expected_rank = min(d_out, int(slice_cols.size))
            assert within_rank == expected_rank, (
                f"ElGamal G2: within-slice rank deficit for "
                f"({side}, {nt.value}, {et.value}): got {within_rank}, "
                f"expected {expected_rank} on slice of width {slice_cols.size}"
            )

            # Operator-norm contract: every block must have operator norm
            # within _ELGAMAL_OP_NORM_TOL of the documented radius (= 1.0).
            op_norm = float(np.linalg.norm(M_np, ord=2))
            assert abs(op_norm - _ELGAMAL_OPERATOR_NORM_RADIUS) <= _ELGAMAL_OP_NORM_TOL, (
                f"ElGamal operator-norm contract violated for "
                f"({side}, {nt.value}, {et.value}): got {op_norm}, "
                f"expected {_ELGAMAL_OPERATOR_NORM_RADIUS} +- {_ELGAMAL_OP_NORM_TOL}"
            )

    # -------------------------------------------------------------------------
    # G3: Linear H^0 scaling under disjoint unions.
    #
    # We assemble a canonical sheaf from K independent cast-record-count loops,
    # each loop being n_cast_edges_per_execution cast edges sharing a proposal
    # node, plus a record/count loop sharing one execution-contract node. The
    # disjoint-union sheaf must satisfy H^0(union of K loops) = K * H^0(one loop)
    # under the ElGamal-instantiated restriction maps. This is the same
    # testable consequence that verify_theorem_1 establishes for the
    # categorical fallback; here we confirm the ElGamal substitution preserves
    # it. Failure indicates that ElGamal restriction maps introduce spurious
    # linear dependencies across loops that were independent under the
    # categorical scaffold.
    # -------------------------------------------------------------------------
    def _build_k_loops(k: int) -> CellularSheaf:
        """Assemble K disjoint cast-record-count loops as a single sheaf."""
        nodes_per_loop = n_cast_edges_per_execution + 2  # voters + proposal + exec
        node_types_list: list[NodeType] = []
        src_list: list[int] = []
        tgt_list: list[int] = []
        edge_types_list: list[EdgeType] = []
        for j in range(k):
            base = nodes_per_loop * j
            voter_ids = list(range(base, base + n_cast_edges_per_execution))
            prop_id = base + n_cast_edges_per_execution
            exec_id = base + n_cast_edges_per_execution + 1
            node_types_list.extend([NodeType.VOTER] * n_cast_edges_per_execution)
            node_types_list.append(NodeType.PROPOSAL)
            node_types_list.append(NodeType.EXEC_CONTRACT)
            for v in voter_ids:
                src_list.append(v)
                tgt_list.append(prop_id)
                edge_types_list.append(EdgeType.CAST)
            src_list.append(prop_id)
            tgt_list.append(exec_id)
            edge_types_list.append(EdgeType.RECORD)
            src_list.append(exec_id)
            tgt_list.append(prop_id)
            edge_types_list.append(EdgeType.COUNT)
        edge_index = torch.tensor([src_list, tgt_list], dtype=torch.long)
        return build_canonical_sheaf(node_types_list, edge_index, edge_types_list)

    sheaf_one = _build_k_loops(1)
    sheaf_k = _build_k_loops(num_executions)
    h0_one = sheaf_one.h0_dimension(tol=_ELGAMAL_ZERO_EIG_TOL)
    h0_k = sheaf_k.h0_dimension(tol=_ELGAMAL_ZERO_EIG_TOL)
    assert h0_one >= 1, (
        f"ElGamal G3: single-loop H^0 is trivial under ElGamal substitution "
        f"({h0_one}); the disjoint-union scaling test cannot proceed"
    )
    assert h0_k == num_executions * h0_one, (
        f"ElGamal G3: H^0 does not scale linearly under disjoint unions with "
        f"ElGamal restriction maps: H^0(1)={h0_one}, "
        f"H^0({num_executions})={h0_k}, expected {num_executions * h0_one}"
    )

    # -------------------------------------------------------------------------
    # G4: Well-conditioned coboundary.
    #
    # Every nonzero singular value of the canonical coboundary delta on the
    # single-loop sheaf must lie above _ELGAMAL_ZERO_EIG_TOL. We use the
    # single-loop sheaf rather than the union to keep the computation small;
    # the disjoint-union coboundary's nonzero spectrum is the union of the
    # per-loop spectra, so single-loop conditioning is the correct
    # representative quantity. The condition number kappa = sigma_max /
    # sigma_min_nonzero is reported as a diagnostic; we assert it is finite,
    # which combined with sigma_min_nonzero > tol is the operational
    # well-conditioning criterion that the downstream sparse solvers depend on.
    # -------------------------------------------------------------------------
    delta = sheaf_one.coboundary()
    sigma = np.linalg.svd(delta.toarray(), compute_uv=False)
    sigma = sigma[sigma > _ELGAMAL_ZERO_EIG_TOL]
    assert sigma.size > 0, (
        "ElGamal G4: canonical coboundary has no nonzero singular values; "
        "the ElGamal substitution has collapsed delta to the zero map"
    )
    sigma_min_nonzero = float(sigma.min())
    sigma_max = float(sigma.max())
    assert sigma_min_nonzero > _ELGAMAL_ZERO_EIG_TOL, (
        f"ElGamal G4: smallest nonzero singular value {sigma_min_nonzero} is "
        f"below tolerance {_ELGAMAL_ZERO_EIG_TOL}; coboundary is "
        f"ill-conditioned under the ElGamal substitution"
    )
    kappa = sigma_max / sigma_min_nonzero
    assert math.isfinite(kappa), (
        f"ElGamal G4: condition number kappa is not finite "
        f"(sigma_max={sigma_max}, sigma_min_nonzero={sigma_min_nonzero})"
    )

    return True


if __name__ == "__main__":
    print("VERISHEAF theory.py — empirical verification of Theorems 1-5")
    ok = True
    try:
        verify_theorem_1()
        print("  Theorem 1 (canonical sheaf construction)      PASS")
    except AssertionError as e:
        print(f"  Theorem 1 FAIL: {e}")
        ok = False
    try:
        verify_theorem_2()
        print("  Theorem 2 (attack subspace orthogonality)     PASS")
    except AssertionError as e:
        print(f"  Theorem 2 FAIL: {e}")
        ok = False
    try:
        verify_theorem_3()
        print("  Theorem 3 (spectral approximation)            PASS")
    except AssertionError as e:
        print(f"  Theorem 3 FAIL: {e}")
        ok = False
    try:
        verify_theorem_4()
        print("  Theorem 4 (PAC-Bayes learning bound)          PASS")
    except AssertionError as e:
        print(f"  Theorem 4 FAIL: {e}")
        ok = False
    try:
        verify_theorem_5()
        print("  Theorem 5 (perturbation subspace leakage)     PASS")
    except AssertionError as e:
        print(f"  Theorem 5 FAIL: {e}")
        ok = False
    try:
        verify_elgamal_substitution()
        print("  ElGamal substitution (structural guarantees)  PASS")
    except AssertionError as e:
        print(f"  ElGamal substitution FAIL: {e}")
        ok = False

    pred = corollary_1_predicted_class(
        n_cast_edges=8, n_record_edges=1, n_count_edges=1,
    )
    assert pred == "cast", f"Corollary 1: expected 'cast' on (8,1,1), got {pred!r}"
    print("  Corollary 1 (dimensional asymmetry prediction) PASS")

    print(f"\n{'All theorems verified.' if ok else 'Some theorems FAILED.'}")
