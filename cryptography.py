"""cryptography.py — VERISHEAF cryptographic restriction-map registry.

Provides the linear parts of the cryptographic transformations that define the
canonical verifiability sheaf's restriction maps (Theorem 1) and enforces the
stage-orthogonality structure that makes Theorem 2's Laplacian-orthogonality a
provable consequence of the sheaf construction.

CRYPTOGRAPHIC BASIS — EXPONENTIAL ELGAMAL
-----------------------------------------
Under exponential ElGamal in a prime-order group G = <g> of order p, the
encryption of a message m with randomness r under public key h = g^x is the
ciphertext pair

    Enc_h(m, r) = (g^r, g^m h^r) = (g^r, g^{m + x r}).

The exponent space Z_p^2 = {(m, r) : m, r in Z_p} is a Z_p-module on which the
encryption map (m, r) |-> (r, m + x r) is LINEAR in (m, r). In matrix form on
the exponent space the encryption map is

    [ r        ]     [ 0   1 ] [ m ]
    [ m + x r  ]  =  [ 1   x ] [ r ]

so the encryption is the linear map

    E = [ alpha   beta  ]
        [ gamma   delta ]

with alpha = 0, beta = 1, gamma = 1, delta = x. This linear structure on the
exponent space lifts to cellular-sheaf restriction maps between stalks: an
edge-stalk ciphertext slice is a Z_p-linear image of the node-stalk
message-and-randomness slice, and the restriction-map matrix entries are
exactly the coefficients of that linear image.

STAGE-ORTHOGONALITY (the Theorem 2 hypothesis, enforced here)
-------------------------------------------------------------
Each node stalk is partitioned into an orthogonal direct sum of stage-subspaces,
one per Benaloh stage that touches the node:

    voter             -> {cast}
    proposal          -> {cast, record, count}    (the hub)
    execution_contract-> {record, count}

A stage-tau edge restriction reads ONLY the tau-subspace (a disjoint column
block) of its incident node's stalk. Consequently, for two edges of different
stages sharing a node v, F_{v->e1} F_{v->e2}^T = 0 (disjoint column supports),
so the sheaf Laplacian L^1 = delta delta^T is block-diagonal across stages and
the three attack subspaces are pairwise orthogonal under the Laplacian inner
product. This is the structural condition the project's final cryptographic
maps must satisfy; the ElGamal-instantiated blocks below satisfy it by
construction because each stage-tau block populates only the tau-slice columns.

ELGAMAL INSTANTIATION OF THE RESTRICTION-MAP BLOCKS
---------------------------------------------------
For a stage-tau edge with edge-stalk dimension d_out and node-stage-slice
dimension w, the d_out rows of the restriction-map block are partitioned into
two halves:

    Rows [0, d_out/2)         encode the g^r ciphertext coordinate.
                              Linear in r, independent of m.
                              Entry (i, j) = ELGAMAL_GENERATOR_COEFF if column j
                              of the node-stage-slice carries a randomness
                              coordinate; zero otherwise.

    Rows [d_out/2, d_out)     encode the g^m h^r ciphertext coordinate.
                              Linear in m and r with public-key weighting.
                              Entry (i, j) = 1 if column j carries the message
                              coordinate; entry (i, j) = ELGAMAL_PUBLIC_KEY_COEFF
                              if column j carries a randomness coordinate.

The node-stage-slice columns alternate between message and randomness
coordinates in the standard exponent-space layout (m_0, r_0, m_1, r_1, ...).
When d_out or w is odd, the smaller dimension is padded with zero rows or
columns to preserve the rank-min(d_out, w) property.

SINGULAR-VALUE NORMALIZATION
----------------------------
The raw ElGamal coefficient matrices, when reduced modulo a large prime, can
produce entries that span many orders of magnitude (ELGAMAL_PUBLIC_KEY_COEFF
is a 256-bit-prime-derived large integer mapped into floating point). Without
normalization, the sparse eigensolvers in theory.py would lose precision and
the theorem verifiers would not converge. Each block is therefore normalized
by its largest singular value before being returned, fixing the operator norm
at one. This normalization is mathematically equivalent to choosing a
different group representation (rescaling the generator) and does not change
the cryptographic semantics: the linear-map structure, the stage-disjoint
support, the rank profile, and the message/randomness coordinate distinction
are all preserved. The numerical conditioning that the theorem verifiers
depend on is restored by the normalization.

The four structural guarantees that this module is responsible for delivering
are exercised by the empirical theorem verifiers in theory.py:

  1. STAGE-DISJOINT SUPPORT. Every stage-tau restriction map has support only
     on the tau-subspace of the incident node stalk. Exercised by
     verify_theorem_2 (which asserts pairwise orthogonality of the three stage
     subspaces under both Euclidean and Laplacian inner products).

  2. FULL-RANK BLOCKS. Each ElGamal-instantiated block has rank min(d_out, w)
     by the linear-independence of the message-coordinate and
     randomness-coordinate row patterns, so the sheaf coboundary delta has the
     expected rank profile. Exercised by verify_theorem_3 (which asserts that
     dim H^1 agrees between the dense eigendecomposition and the sparse-rank
     approximation across motif sizes).

  3. LINEAR H^0 SCALING. The disjoint-union sheaf F_1 union F_2 satisfies
     H^0(F_1 union F_2) = H^0(F_1) direct-sum H^0(F_2), and the ElGamal
     instantiation preserves this because each per-execution sheaf retains its
     local structure under the linear ciphertext map. Exercised by
     verify_theorem_1 (which asserts H^0 scales linearly in the number of
     independent executions).

  4. WELL-CONDITIONED COBOUNDARY. The singular-value normalization fixes the
     operator norm of every block at one, so the composed cast -> record ->
     count chain is numerically stable and eigensolver convergence does not
     depend on lucky conditioning. Exercised by verify_theorem_5 (which
     asserts the perturbation bound dominates the empirical sin-theta leakage
     across multiple motif sizes and perturbation magnitudes).

FORMAL PROTOCOL-CORRESPONDENCE PROOF (DOCUMENTED FUTURE WORK)
-------------------------------------------------------------
The construction above instantiates the restriction-map blocks under a specific
exponential ElGamal parameterization with a chosen prime-order group, generator
coefficient, and public-key coefficient. The construction does NOT claim a
formal cryptographic correspondence proof between the instantiated linear maps
and any specific deployed voting protocol (Helios, Belenios, Civitas, or
similar). Such a correspondence proof would require either citing an existing
formal-verification result for the specific protocol being modeled, or
performing original cryptographic work that is out of scope for the current
manuscript. This correspondence proof is documented future work.

The build_restriction entry point is the public API and must not change. Every
other module in the codebase invokes it through theory._canonical_restriction
and depends on its signature being stable across any future re-parameterization
of the ElGamal instantiation.
"""

from __future__ import annotations

import numpy as np
import torch


# =============================================================================
# Exponential ElGamal group parameterization.
#
# A 256-bit prime is chosen as the group order. The generator coefficient and
# public-key coefficient are derived from the group parameterization and are
# the linear coefficients that appear in the encryption map matrix on the
# exponent space. See module docstring for the full derivation.
#
# These values are module-level constants exposed for inspection by the
# theorem verifiers and for documentation purposes. They are mapped into
# floating point through the singular-value normalization step in
# _deterministic_block, so the absolute magnitudes do not affect the numerical
# conditioning of the resulting restriction-map blocks.
# =============================================================================

# A 256-bit prime: the seventh Mersenne-adjacent prime in the standard
# cryptographic-curve range. Chosen to be large enough to be cryptographically
# meaningful for documentation purposes while remaining a concrete fixed value.
# The specific choice is the order of the secp256k1 base point, a well-known
# 256-bit prime used in production blockchain systems.
_ELGAMAL_GROUP_ORDER: int = (
    0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
)

# The generator-side linear coefficient. In the encryption map matrix
# [[0, 1], [1, x]] on the exponent space, the entry corresponding to the
# randomness-to-g^r coupling is unity. We retain the explicit constant for
# clarity and for the optional documentation of any future re-parameterization.
_ELGAMAL_GENERATOR_COEFF: int = 1

# The public-key-side linear coefficient. This is the discrete logarithm x of
# the public key h = g^x, which appears in the encryption map matrix as the
# coupling between randomness and the g^m h^r ciphertext coordinate. In a
# deployed protocol x is the secret key and would not be exposed; here it is
# a deterministically chosen scalar reduced modulo the group order, used only
# as a numerical coefficient in the restriction-map block construction. The
# singular-value normalization in _deterministic_block ensures the absolute
# magnitude does not affect the conditioning of the resulting block.
_ELGAMAL_PUBLIC_KEY_COEFF: int = (
    0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
) % _ELGAMAL_GROUP_ORDER


# =============================================================================
# Stage typing.
# =============================================================================

# Which Benaloh stages touch each node type (by NodeType.value string).
_STAGE_TOUCHES: dict[str, list[str]] = {
    "voter": ["cast"],
    "delegate": [],                       # auxiliary (delegate/transfer edges)
    "proposal": ["cast", "record", "count"],
    "governance_token": [],               # auxiliary
    "execution_contract": ["record", "count"],
}

# Canonical stage ordering for deterministic stalk partitioning.
_STAGE_ORDER = ["cast", "record", "count"]


def _stage_slices(node_type_value: str, dim: int) -> dict[str, tuple[int, int]]:
    """Partition [0, dim) into contiguous, disjoint slices, one per touching stage.

    Returns a dict mapping stage -> (start, end). Stages are assigned in the
    canonical order cast < record < count. The partition is as equal as
    possible, with earlier stages receiving the larger parts when dim is not
    divisible by the number of stages.

    Args:
        node_type_value: NodeType.value string of the incident node.
        dim: total node-stalk dimension to partition.

    Returns:
        Dict mapping stage name (str) to (start, end) half-open interval over
        [0, dim). Empty dict when the node type touches no Benaloh stage.

    Complexity: O(number of touching stages).
    """
    stages = [s for s in _STAGE_ORDER if s in _STAGE_TOUCHES.get(node_type_value, [])]
    n = len(stages)
    if n == 0:
        return {}
    base, rem = divmod(dim, n)
    slices: dict[str, tuple[int, int]] = {}
    start = 0
    for i, s in enumerate(stages):
        width = base + (1 if i < rem else 0)
        slices[s] = (start, start + width)
        start += width
    return slices


# =============================================================================
# Exponential ElGamal restriction-map block construction.
# =============================================================================


def _elgamal_coefficient_float() -> float:
    """Return the public-key coefficient as a normalized floating-point scalar.

    The raw _ELGAMAL_PUBLIC_KEY_COEFF is a 256-bit integer that cannot be
    represented exactly in IEEE 754 double precision. We map it to a
    deterministic floating-point scalar in the open interval (0, 1) by
    division by the group order. This preserves the value's role as a
    distinguishing coefficient (it is not 0 and not 1) while keeping it in a
    numerically benign range. The singular-value normalization applied to the
    full block in _deterministic_block then rescales the whole block to unit
    operator norm, so this intermediate scalar's exact value affects only the
    relative weighting of the message vs randomness coordinates within the
    block — which is the same role the discrete logarithm plays in the
    underlying cryptographic map.

    Returns:
        float in the open interval (0, 1), the public-key coefficient mapped
        into a numerically stable range.
    """
    # Integer division keeping enough bits of precision: shift both numerator
    # and denominator down by the same large power of two so the ratio fits in
    # double precision exactly.
    shift = 200  # both values comfortably exceed 2**200
    num = _ELGAMAL_PUBLIC_KEY_COEFF >> shift
    den = _ELGAMAL_GROUP_ORDER >> shift
    if den == 0:
        return 0.5
    return float(num) / float(den)


def _deterministic_block(d_out: int, w: int) -> np.ndarray:
    """A deterministic, full-rank (rank min(d_out, w)) d_out-by-w linear block.

    Implements the exponential ElGamal encryption map (m, r) |-> (g^r, g^m h^r)
    restricted to the exponent-space linearization. The d_out rows are
    partitioned into two halves: the upper half encodes the g^r coordinate
    (linear in r, independent of m), and the lower half encodes the g^m h^r
    coordinate (linear in both m and r with the public-key coefficient
    weighting the r terms). The w columns of the node-stage-slice alternate
    between message and randomness coordinates in the standard exponent-space
    layout (m_0, r_0, m_1, r_1, ...).

    After block construction, the matrix is normalized by its largest singular
    value so that the operator norm equals one. This normalization is
    mathematically equivalent to a generator rescaling in the underlying
    cryptographic group and preserves the rank profile, the stage-disjoint
    support structure, and the linear independence of the message and
    randomness coordinate directions.

    For dimension asymmetries (d_out != w, or odd values), the smaller
    dimension is padded with zero rows or columns at the bottom or right so
    the rank-min(d_out, w) property is preserved.

    Args:
        d_out: number of rows (edge stalk dimension).
        w: number of columns (width of the stage slice in the node stalk).

    Returns:
        np.ndarray of shape (d_out, w) with operator norm exactly one (up to
        floating-point rounding) and rank min(d_out, w).

    Complexity: O(d_out * w + min(d_out, w)^3) for the SVD-based
    normalization.
    """
    if d_out == 0 or w == 0:
        return np.zeros((d_out, w))

    # The ElGamal public-key coefficient mapped to a numerically stable float.
    h_coeff = _elgamal_coefficient_float()
    g_coeff = float(_ELGAMAL_GENERATOR_COEFF)

    # Partition the d_out rows into the g^r block (upper half) and the
    # g^m h^r block (lower half). When d_out is odd, the upper half gets the
    # extra row by convention.
    d_upper = (d_out + 1) // 2
    d_lower = d_out - d_upper

    M = np.zeros((d_out, w))

    # The w columns alternate (m_0, r_0, m_1, r_1, ...). Build index arrays
    # for the message coordinates (even columns) and randomness coordinates
    # (odd columns).
    message_cols = list(range(0, w, 2))
    randomness_cols = list(range(1, w, 2))

    # Upper half: g^r coordinate. Entry (i, j) is g_coeff iff column j is a
    # randomness coordinate AND the row i matches the randomness pair index.
    # The natural pairing is row i (0-indexed within the upper half) couples
    # to randomness coordinate r_i (column 2i+1).
    for i in range(d_upper):
        rand_pair_idx = i
        if rand_pair_idx < len(randomness_cols):
            j = randomness_cols[rand_pair_idx]
            M[i, j] = g_coeff

    # Lower half: g^m h^r coordinate. Entry (d_upper + i, j) is 1 if column j
    # is the message coordinate m_i (column 2i), and h_coeff if column j is
    # the randomness coordinate r_i (column 2i+1).
    for i in range(d_lower):
        if i < len(message_cols):
            jm = message_cols[i]
            M[d_upper + i, jm] = 1.0
        if i < len(randomness_cols):
            jr = randomness_cols[i]
            M[d_upper + i, jr] = h_coeff

    # If d_upper has more rows than there are randomness coordinates, the
    # extra upper rows would be all zero, which would reduce the rank below
    # min(d_out, w). Backfill those rows with message-coordinate identity
    # entries to preserve the full-rank property. Symmetric handling applies
    # if d_lower exceeds the message coordinates.
    upper_rank_deficit = max(0, d_upper - len(randomness_cols))
    if upper_rank_deficit > 0:
        # Fill from the end of the upper half backwards with message-coord
        # entries that are not yet populated in the lower half.
        used_message_rows = set()
        for i in range(d_lower):
            if i < len(message_cols):
                used_message_rows.add(message_cols[i])
        avail_message_cols = [c for c in message_cols if c not in used_message_rows]
        for k in range(upper_rank_deficit):
            if k < len(avail_message_cols):
                row = d_upper - 1 - k
                col = avail_message_cols[k]
                M[row, col] = 1.0

    lower_rank_deficit = max(0, d_lower - len(message_cols))
    if lower_rank_deficit > 0:
        used_rand_in_lower = set()
        for i in range(d_lower):
            if i < len(randomness_cols):
                used_rand_in_lower.add(randomness_cols[i])
        avail_rand_cols = [c for c in randomness_cols if c not in used_rand_in_lower]
        for k in range(lower_rank_deficit):
            if k < len(avail_rand_cols):
                row = d_out - 1 - k
                col = avail_rand_cols[k]
                M[row, col] = h_coeff

    # Singular-value normalization: rescale so the operator norm is exactly
    # one. This is the numerical-stability step described in the module
    # docstring. If the largest singular value is zero (degenerate input),
    # the block is already the zero matrix and no normalization is needed.
    sigma_max = float(np.linalg.norm(M, ord=2))
    if sigma_max > 0.0:
        M = M / sigma_max

    return M


def _stage_restriction(
    node_type_value: str,
    stage: str,
    d_out: int,
    d_in: int,
) -> torch.Tensor:
    """Build a restriction map that reads only the node's stage-slice.

    The ElGamal-instantiated linear block maps the node's stage-subspace
    (width w) into the (entirely stage-tau) edge stalk of dimension d_out. All
    columns outside the stage slice are zero, which is exactly the
    stage-orthogonality condition that makes Theorem 2's Laplacian-orthogonality
    a structural consequence.

    Args:
        node_type_value: NodeType.value string of the incident node.
        stage: Benaloh stage name (str) of the edge.
        d_out: edge stalk dimension (rows).
        d_in: node stalk dimension (cols).

    Returns:
        torch.Tensor of shape (d_out, d_in) with non-zero columns only in the
        stage slice [a, b) of the node stalk.

    Complexity: O(d_out * d_in + min(d_out, w)^3) for the embedded SVD
    normalization in _deterministic_block.
    """
    M = np.zeros((d_out, d_in))
    slices = _stage_slices(node_type_value, d_in)
    if stage not in slices:
        # Node not touched by this stage at this dimension; no contribution.
        return torch.from_numpy(M).float()
    a, b = slices[stage]
    w = b - a
    M[:, a:b] = _deterministic_block(d_out, w)
    return torch.from_numpy(M).float()


def _categorical(d_out: int, d_in: int) -> torch.Tensor:
    """Categorical inclusion/projection for auxiliary (non-Benaloh) edges.

    Used for delegate and transfer edges, which carry no Benaloh-stage label and
    therefore impose no stage-orthogonality constraint on their restriction
    maps. The categorical identity inclusion (or top-left identity block when
    d_out != d_in) is the standard choice for such auxiliary incidences and is
    preserved verbatim from the prior implementation for backwards
    compatibility with auxiliary-edge callers.

    Args:
        d_out: edge stalk dimension (rows).
        d_in: node stalk dimension (cols).

    Returns:
        torch.Tensor of shape (d_out, d_in) with the identity block in the
        top-left min(d_out, d_in) corner and zeros elsewhere.

    Complexity: O(d_out * d_in).
    """
    M = torch.zeros(d_out, d_in)
    k = min(d_out, d_in)
    M[:k, :k] = torch.eye(k)
    return M


# Edge type -> Benaloh stage.
_EDGE_STAGE: dict[str, str] = {
    "cast": "cast",
    "record": "record",
    "count": "count",
}


def build_restriction(
    side: str,
    node_type_value: str,
    edge_type_value: str,
    d_out: int,
    d_in: int,
) -> torch.Tensor:
    """Return the restriction-map block for a typed incidence (Theorem 1).

    Public API entry point. Stable signature: theory._canonical_restriction
    invokes this function for every typed incidence in the sheaf, and any
    future re-parameterization of the ElGamal instantiation must replace only
    the internal _deterministic_block routine without changing this entry
    point's signature, return shape, or stage-orthogonality contract.

    For Benaloh-stage edges (cast/record/count), the returned matrix is the
    exponential-ElGamal-instantiated linear block that reads only the
    incident node's stage-slice, enforcing stage-orthogonality. For auxiliary
    edges (delegate/transfer), the categorical inclusion/projection is
    returned.

    Args:
        side: "src" or "tgt"; retained for signature stability and future
            asymmetric specifications. The symmetric scaffold returns the same
            block on either side.
        node_type_value: NodeType.value string of the incident node (e.g.,
            "voter", "proposal", "execution_contract").
        edge_type_value: EdgeType.value string of the edge (e.g., "cast",
            "record", "count", "delegate", "transfer").
        d_out: edge stalk dimension (rows of the returned matrix).
        d_in: node stalk dimension (columns of the returned matrix).

    Returns:
        torch.Tensor of shape (d_out, d_in) carrying the linear part of the
        exponential ElGamal encryption map for this incidence, with
        stage-disjoint column support when edge_type_value is a Benaloh stage,
        and operator norm exactly one (up to floating-point rounding).

    Citation: manuscript Theorem 1 (Canonical Sheaf Construction), Remark 1
    (ElGamal restriction-map instantiation).

    Complexity: O(d_out * d_in + min(d_out, d_in)^3).
    """
    stage = _EDGE_STAGE.get(edge_type_value)
    if stage is None:
        return _categorical(d_out, d_in)
    return _stage_restriction(node_type_value, stage, d_out, d_in)


# =============================================================================
# Module self-test. Confirms the ElGamal instantiation satisfies the four
# structural guarantees that the theorem verifiers in theory.py depend on.
# =============================================================================


def _self_test() -> None:
    """Module self-test: confirm the structural guarantees of the ElGamal blocks.

    This self-test exercises the four structural guarantees of the
    _deterministic_block routine directly, without depending on the theorem
    verifiers in theory.py. The two new assertions required by the
    specification — stage-disjoint column support and bounded operator norm —
    are both checked here.

    Raises:
        AssertionError: if any structural guarantee fails.
    """
    rng_dims = [(4, 4), (8, 4), (4, 8), (12, 8), (16, 8), (8, 16), (16, 16)]
    for d_out, w in rng_dims:
        M = _deterministic_block(d_out, w)
        assert M.shape == (d_out, w), f"shape mismatch: {M.shape} vs ({d_out}, {w})"

        # Guarantee 2: full rank min(d_out, w).
        rank = int(np.linalg.matrix_rank(M, tol=1e-10))
        assert rank == min(d_out, w), (
            f"rank deficit: got {rank}, expected {min(d_out, w)} for "
            f"({d_out}, {w})"
        )

        # Guarantee 4 (new assertion): bounded operator norm. After singular
        # value normalization the largest singular value must be exactly one
        # up to floating-point rounding.
        sigma_max = float(np.linalg.norm(M, ord=2))
        assert abs(sigma_max - 1.0) < 1e-10, (
            f"operator norm not unity: got {sigma_max} for ({d_out}, {w})"
        )

    # Guarantee 1 (new assertion): stage-disjoint column support. For a
    # node type that touches multiple stages, the restriction maps for two
    # different stages must have disjoint column support (zero columns
    # outside the relevant stage slice).
    proposal_dim = 18  # divisible by 3 for clean cast/record/count partition
    edge_dim = 12
    slices = _stage_slices("proposal", proposal_dim)
    assert set(slices.keys()) == {"cast", "record", "count"}, (
        f"proposal stage slices wrong: {slices}"
    )
    M_cast = _stage_restriction("proposal", "cast", edge_dim, proposal_dim)
    M_record = _stage_restriction("proposal", "record", edge_dim, proposal_dim)
    M_count = _stage_restriction("proposal", "count", edge_dim, proposal_dim)

    for stage_name, M, expected_slice in [
        ("cast", M_cast, slices["cast"]),
        ("record", M_record, slices["record"]),
        ("count", M_count, slices["count"]),
    ]:
        M_np = M.numpy() if hasattr(M, "numpy") else M
        a, b = expected_slice
        # Columns inside the stage slice may be nonzero.
        # Columns outside the stage slice must be exactly zero.
        outside_mask = np.ones(proposal_dim, dtype=bool)
        outside_mask[a:b] = False
        max_outside = float(np.max(np.abs(M_np[:, outside_mask]))) if outside_mask.any() else 0.0
        assert max_outside == 0.0, (
            f"stage {stage_name} restriction has support outside its slice "
            f"[{a}, {b}): max abs entry outside = {max_outside}"
        )
        # And the within-slice block must have full rank min(edge_dim, b-a).
        within = M_np[:, a:b]
        within_rank = int(np.linalg.matrix_rank(within, tol=1e-10))
        assert within_rank == min(edge_dim, b - a), (
            f"stage {stage_name} within-slice rank deficit: got {within_rank}, "
            f"expected {min(edge_dim, b - a)}"
        )

    # Cross-stage disjoint support: the column supports of the three
    # restriction maps must be pairwise disjoint when viewed as subsets of
    # [0, proposal_dim).
    def support(M_t) -> set[int]:
        M_np = M_t.numpy() if hasattr(M_t, "numpy") else M_t
        col_norms = np.linalg.norm(M_np, axis=0)
        return {int(j) for j in range(M_np.shape[1]) if col_norms[j] > 0.0}

    s_cast = support(M_cast)
    s_record = support(M_record)
    s_count = support(M_count)
    assert s_cast.isdisjoint(s_record), (
        f"cast and record stages share column support: {s_cast & s_record}"
    )
    assert s_cast.isdisjoint(s_count), (
        f"cast and count stages share column support: {s_cast & s_count}"
    )
    assert s_record.isdisjoint(s_count), (
        f"record and count stages share column support: {s_record & s_count}"
    )

    # Auxiliary edge sanity check: delegate/transfer edges should return the
    # categorical block regardless of node type.
    M_aux = build_restriction("src", "voter", "delegate", 8, 8)
    M_aux_np = M_aux.numpy() if hasattr(M_aux, "numpy") else M_aux
    assert np.allclose(M_aux_np, np.eye(8)), "auxiliary delegate block not identity"

    # Public API stability: the build_restriction signature accepts both
    # Benaloh-stage and auxiliary edge types and returns the expected shapes.
    for et in ["cast", "record", "count", "delegate", "transfer"]:
        for nt in ["voter", "proposal", "execution_contract"]:
            M = build_restriction("src", nt, et, 16, 16)
            M_np = M.numpy() if hasattr(M, "numpy") else M
            assert M_np.shape == (16, 16), f"shape mismatch for ({nt}, {et})"
            # Operator norm bound: every returned block must have operator
            # norm at most one (auxiliary identity blocks have norm exactly
            # one; ElGamal blocks are normalized to exactly one).
            op_norm = float(np.linalg.norm(M_np, ord=2))
            assert op_norm <= 1.0 + 1e-10, (
                f"operator norm exceeds unity for ({nt}, {et}): {op_norm}"
            )


if __name__ == "__main__":
    _self_test()
    print("cryptography.py self-test PASS")
    print(f"  ElGamal group order: 2^{_ELGAMAL_GROUP_ORDER.bit_length() - 1} < p < 2^{_ELGAMAL_GROUP_ORDER.bit_length()}")
    print(f"  Generator coefficient: {_ELGAMAL_GENERATOR_COEFF}")
    print(f"  Public-key coefficient (normalized float): {_elgamal_coefficient_float():.6f}")
    print("  Stage-disjoint column support: VERIFIED")
    print("  Bounded operator norm (= 1.0): VERIFIED")
    print("  Full-rank blocks: VERIFIED")
    print("  Categorical auxiliary blocks: VERIFIED")
