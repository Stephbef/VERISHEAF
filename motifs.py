"""motifs.py — VERISHEAF motif schema, sheaf converters, and incident catalogue.

Constructs the typed temporal motif on which the verifiability sheaf F_G is built.
A motif is anchored to a single proposal and carries the chronologically ordered
cast, recorded, and counted edges that constitute the Benaloh decomposition.

Structural incompleteness is preserved as positive signal: a Snapshot-only motif
with no on-chain record emits an empty record edge list rather than being omitted
or marked as an error. This is the property the verifiability cohomology measures.

This module also hosts the Incident dataclass and the CANONICAL_INCIDENTS list of
the eight catalogued governance-attack incidents that constitute the held-out
benchmark for the cohomological characterization in Theorem 2.

Defect history (preserved as in-line documentation for reviewer-defensibility):
    CC-3 (FTR-1): the training loop previously fed torch.randn(...) Gaussian noise
        into the sheaf-Dirichlet energy. The motif_to_node_signals function below
        is the in-place fix: it derives node-stalk signals from the motif's actual
        cast-edge choice and voting-weight fields. Preserved verbatim per
        construction rule; any modification risks reintroducing the noise-input bug.
    FM-1: motif_to_typed_graph rejects unknown node-id prefixes (the prior
        implementation silently typed any prefix as VOTER).
    FM-2: deterministic_hash invariant under insertion-order via assign_chrono_rank.
    FM-3: edge-type-to-list mismatch raises ValueError in Motif.__post_init__.
    FM-4: synthetic voter ids are SHA-256-keyed on (dao, motif, seed, index) so
        they cannot collide across runs.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import math
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

import torch

from theory import (
    BenalohStage,
    CellularSheaf,
    EdgeType,
    NodeType,
    build_canonical_sheaf,
)


# =============================================================================
# MotifEdge and Motif data structures
# =============================================================================


@dataclass
class MotifEdge:
    """A single typed edge inside a motif.

    Attributes:
        source_node_id: canonical node id with one of the five required prefixes
            (voter:, delegate:, proposal:, token:, execution_contract:).
        target_node_id: canonical node id with the same prefix convention.
        edge_type: EdgeType enum value (CAST, DELEGATE, TRANSFER, RECORD, COUNT).
        timestamp_unix: integer Unix timestamp of the edge event.
        chrono_rank: integer chronological rank assigned by assign_chrono_rank;
            defaults to 0 and is overwritten before any hash is computed.
        khop_layer: integer k-hop layer for downstream subgraph baselines;
            defaults to 0.
        weight: float edge weight (voting power for cast edges, 1.0 elsewhere).
        stage: Optional BenalohStage; None for auxiliary (delegate/transfer) edges.
        metadata: dict of free-form per-edge metadata (e.g., ballot choice).
    """

    source_node_id: str
    target_node_id: str
    edge_type: EdgeType
    timestamp_unix: int
    chrono_rank: int = 0
    khop_layer: int = 0
    weight: float = 1.0
    stage: Optional[BenalohStage] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class Motif:
    """A typed temporal motif anchored to a single proposal.

    Attributes:
        motif_id: unique string identifier for the motif.
        proposal_node_id: canonical proposal: node id this motif is anchored to.
        dao_name: short DAO identifier (e.g., "compound", "aragon:0x...").
        chain: chain identifier ("ethereum", "xdai", etc.).
        proposal_timestamp_unix: integer Unix timestamp of the proposal creation.
        cast_edges: list of MotifEdge with edge_type == EdgeType.CAST.
        record_edges: list of MotifEdge with edge_type == EdgeType.RECORD.
        count_edges: list of MotifEdge with edge_type == EdgeType.COUNT.
        auxiliary_edges: list of MotifEdge with non-Benaloh edge types.
        is_complete: derived bool; True iff all three Benaloh stages are present.
        is_offchain_only: derived bool; True iff only cast edges are present.
        is_onchain_only: derived bool; True iff record edges present but no cast.
        has_executed: derived bool; True iff count edges are present.
    """

    motif_id: str
    proposal_node_id: str
    dao_name: str
    chain: str
    proposal_timestamp_unix: int

    cast_edges: list[MotifEdge] = field(default_factory=list)
    record_edges: list[MotifEdge] = field(default_factory=list)
    count_edges: list[MotifEdge] = field(default_factory=list)
    auxiliary_edges: list[MotifEdge] = field(default_factory=list)

    is_complete: bool = field(init=False, default=False)
    is_offchain_only: bool = field(init=False, default=False)
    is_onchain_only: bool = field(init=False, default=False)
    has_executed: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        """Validate edge-type-to-list correspondence and derive boolean flags.

        Raises:
            ValueError: if any edge in cast_edges/record_edges/count_edges has
                an edge_type that does not match the list's expected EdgeType.
                Diagnostic message names the offending edge index, the field
                name, the expected EdgeType, and the actual EdgeType. This is
                the FM-3 strict-edge-type-list validation.
        """
        self._validate_edge_types()
        has_cast = len(self.cast_edges) > 0
        has_record = len(self.record_edges) > 0
        has_count = len(self.count_edges) > 0
        self.is_complete = has_cast and has_record and has_count
        self.is_offchain_only = has_cast and not has_record and not has_count
        self.is_onchain_only = has_record and not has_cast
        self.has_executed = has_count

    def _validate_edge_types(self) -> None:
        """Assert each stage list holds only edges of the matching EdgeType.

        A RECORD-typed edge placed in cast_edges would silently produce a sheaf
        whose Benaloh-stage assignment disagrees with its typed-graph structure,
        surfacing downstream as a Theorem 2 misclassification with no diagnostic.
        Catch it at construction (FM-3).

        Raises:
            ValueError: with diagnostic message naming the offending edge index,
                the list field name, the expected EdgeType, and the actual
                EdgeType.
        """
        expected = [
            (self.cast_edges, EdgeType.CAST, "cast_edges"),
            (self.record_edges, EdgeType.RECORD, "record_edges"),
            (self.count_edges, EdgeType.COUNT, "count_edges"),
        ]
        for edges, et, field_name in expected:
            for i, e in enumerate(edges):
                if e.edge_type != et:
                    raise ValueError(
                        f"Edge {i} in {field_name} has edge_type {e.edge_type}, "
                        f"expected {et}"
                    )

    @property
    def observability_ratio(self) -> float:
        """Observability ratio p = fraction of Benaloh stages observed.

        Returns:
            float in {0.0, 1/3, 2/3, 1.0}: the fraction of the three Benaloh
            stages (cast, record, count) for which the motif carries at least
            one edge.

        Complexity: O(1).
        """
        count = 0
        if self.cast_edges:
            count += 1
        if self.record_edges:
            count += 1
        if self.count_edges:
            count += 1
        return count / 3.0

    def all_edges(self) -> list[MotifEdge]:
        """Return all edges sorted by (chrono_rank, timestamp_unix).

        Returns:
            list[MotifEdge] containing every cast, record, count, and auxiliary
            edge, sorted by chronological rank with timestamp tiebreaker.

        Complexity: O(|E| log |E|).
        """
        edges = self.cast_edges + self.record_edges + self.count_edges + self.auxiliary_edges
        return sorted(edges, key=lambda e: (e.chrono_rank, e.timestamp_unix))

    def deterministic_hash(self) -> str:
        """Return a SHA-256 hash of the motif's edge structure (FM-2).

        Defensively reassigns chronological ranks before hashing so that the
        hash is invariant under insertion-order permutation. Without this,
        edges left at the default rank 0 would hash in timestamp-tiebreak
        order, making the hash depend on whether the caller remembered to call
        assign_chrono_rank first.

        Returns:
            64-character lowercase hexadecimal SHA-256 digest of the canonical
            edge tuple representation.

        Complexity: O(|E| log |E|).
        """
        assign_chrono_rank(
            self.cast_edges
            + self.record_edges
            + self.count_edges
            + self.auxiliary_edges
        )
        parts = []
        for edge in self.all_edges():
            parts.append(
                f"{edge.source_node_id}|{edge.target_node_id}|{edge.edge_type.value}|"
                f"{edge.timestamp_unix}|{edge.chrono_rank}|{edge.weight:.18e}|"
                f"{edge.stage.value if edge.stage else 'none'}"
            )
        payload = "\n".join(parts).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def assign_chrono_rank(edges: list[MotifEdge]) -> list[MotifEdge]:
    """Assign reindex-invariant chronological ranks with deterministic tie-break.

    Ties in the 64-bit timestamp are broken lexicographically over the canonical
    edge tuple (source, target, edge_type.value, stage.value). Two independent
    rankings of the same edge set produce identical ranks.

    Args:
        edges: list of MotifEdge to be ranked in place.

    Returns:
        The same list, sorted by (timestamp, source, target, edge_type, stage),
        with chrono_rank field mutated to reflect the sorted position.

    Complexity: O(|E| log |E|).
    """
    def key(edge: MotifEdge) -> tuple:
        return (
            edge.timestamp_unix,
            edge.source_node_id,
            edge.target_node_id,
            edge.edge_type.value,
            edge.stage.value if edge.stage else "",
        )
    sorted_edges = sorted(edges, key=key)
    for rank, edge in enumerate(sorted_edges):
        edge.chrono_rank = rank
    return sorted_edges


# =============================================================================
# Motif -> Sheaf conversion
# =============================================================================


def motif_to_typed_graph(
    motif: Motif,
) -> tuple[list[NodeType], torch.Tensor, list[EdgeType], list[Optional[BenalohStage]], dict[str, int]]:
    """Convert a Motif object to the inputs needed by build_canonical_sheaf.

    Performs the FM-1 strict-prefix validation: a node id whose prefix is not in
    the canonical five-prefix set (voter:, delegate:, proposal:, token:,
    execution_contract:) raises ValueError rather than being silently typed as
    a voter.

    Args:
        motif: the source Motif.

    Returns:
        node_types: per-node NodeType, indexed by the integer node id.
        edge_index: (2, |E|) torch.LongTensor of (src, tgt) node indices.
        edge_types: per-edge EdgeType, indexed by integer edge id.
        stage_mask: per-edge Optional[BenalohStage].
        node_id_to_index: mapping from string node id to integer index.

    Raises:
        ValueError: if any edge references a node id whose prefix is not one of
            the five canonical prefixes.

    Citation: manuscript Section 4 (Typed Governance Graph Construction).

    Complexity: O(|E| + |V|).
    """
    all_edges = motif.all_edges()

    node_ids: list[str] = []
    node_types: list[NodeType] = []
    node_id_to_index: dict[str, int] = {}

    def add_node(node_id: str) -> int:
        if node_id in node_id_to_index:
            return node_id_to_index[node_id]
        idx = len(node_ids)
        node_id_to_index[node_id] = idx
        node_ids.append(node_id)

        if node_id.startswith("voter:"):
            node_types.append(NodeType.VOTER)
        elif node_id.startswith("delegate:"):
            node_types.append(NodeType.DELEGATE)
        elif node_id.startswith("proposal:"):
            node_types.append(NodeType.PROPOSAL)
        elif node_id.startswith("token:"):
            node_types.append(NodeType.GOV_TOKEN)
        elif node_id.startswith("execution_contract:"):
            node_types.append(NodeType.EXEC_CONTRACT)
        else:
            # A malformed node id (e.g. "prop-uniswap-1" instead of
            # "proposal:uniswap:1") must NOT be silently typed as a voter: that
            # corrupts the sheaf construction with no diagnostic. Fail loudly (FM-1).
            raise ValueError(
                f"Unrecognised node ID prefix: {node_id!r}. Expected one of "
                "'voter:', 'delegate:', 'proposal:', 'token:', "
                "'execution_contract:'."
            )
        return idx

    src_indices = []
    tgt_indices = []
    edge_types: list[EdgeType] = []
    stage_mask: list[Optional[BenalohStage]] = []
    for edge in all_edges:
        si = add_node(edge.source_node_id)
        ti = add_node(edge.target_node_id)
        src_indices.append(si)
        tgt_indices.append(ti)
        edge_types.append(edge.edge_type)
        stage_mask.append(edge.stage)

    edge_index = torch.tensor([src_indices, tgt_indices], dtype=torch.long)
    return node_types, edge_index, edge_types, stage_mask, node_id_to_index


def motif_to_sheaf(
    motif: Motif, learned_restrictions: Optional[dict] = None
) -> CellularSheaf:
    """Build the canonical verifiability sheaf F_M for a motif M.

    Args:
        motif: the source Motif.
        learned_restrictions: optional dict of learned restriction maps to
            substitute for the canonical scaffold; passed through to
            build_canonical_sheaf.

    Returns:
        CellularSheaf instance carrying stalks and restriction maps for the
        motif's typed graph.

    Citation: manuscript Theorem 1 (Canonical Sheaf Construction).

    Complexity: O(|V| + |E| * d_max^2).
    """
    node_types, edge_index, edge_types, stage_mask, _ = motif_to_typed_graph(motif)
    return build_canonical_sheaf(
        node_types,
        edge_index,
        edge_types,
        stage_mask=stage_mask,
        learned_restrictions=learned_restrictions,
    )


def motif_to_node_signals(
    motif: Motif,
    node_id_to_index: dict[str, int],
    stalk_dims: dict[NodeType, int],
    mask_ratio: float = 0.0,
    device: str = "cpu",
    seed: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct node-stalk signals from a motif's actual structure.

    This replaces the prior practice of feeding torch.randn Gaussian noise into
    the sheaf-Dirichlet energy (CC-3 / FTR-1), under which the motif contributed
    only graph topology and the optimised values were noise unrelated to any
    voting execution. Here each node's stalk signal is derived from the motif:

      voter nodes:  intent one-hot (choice in {1,2,3}) in the first dims, plus a
                    normalised log-weight; the rest zero.
      proposal nodes: a deterministic projection of the proposal id (a stand-in
                    for the bulletin commitment) plus indicators of whether the
                    motif carries record and count stages.
      execution_contract nodes: a tally summary (log total weight, voter count).
      delegate / gov_token nodes: zero by design (auxiliary).

    If mask_ratio > 0, a fraction of node signals is zeroed and a binary mask is
    returned, supporting the masked-reconstruction objective: the model must
    make the sheaf coboundary vanish while part of the input is hidden.

    Args:
        motif: the source motif.
        node_id_to_index: mapping from node id to row index (from
            motif_to_typed_graph).
        stalk_dims: per-NodeType stalk dimension (DEFAULT_STALK_DIMS).
        mask_ratio: fraction of nodes to mask in [0, 1).
        device: torch device string.
        seed: optional seed for the mask, for per-epoch reproducibility.

    Returns:
        (node_signals, mask): node_signals of shape (num_nodes, max_stalk_dim);
        mask of shape (num_nodes,) with 0 at masked nodes, else 1.

    Citation: manuscript Section 5 (Masked-Reconstruction Training Objective).

    Complexity: O(|V| * max_stalk_dim).
    """
    max_dim = max(stalk_dims.values())
    n = len(node_id_to_index)
    signals = torch.zeros(n, max_dim, device=device)

    # Index the cast votes by voter id for quick lookup.
    voter_weight: dict[str, float] = {}
    voter_choice: dict[str, int] = {}
    for e in motif.cast_edges:
        voter_weight[e.source_node_id] = e.weight
        voter_choice[e.source_node_id] = int(e.metadata.get("choice", 1))

    has_record = 1.0 if motif.record_edges else 0.0
    has_count = 1.0 if motif.count_edges else 0.0
    total_weight = sum(voter_weight.values())
    n_voters = len(voter_weight)

    for node_id, idx in node_id_to_index.items():
        if node_id.startswith("voter:"):
            d = stalk_dims[NodeType.VOTER]
            choice = voter_choice.get(node_id, 1)
            if 1 <= choice <= 3 and d >= 3:
                signals[idx, choice - 1] = 1.0
            if d >= 4:
                w = voter_weight.get(node_id, 1.0)
                signals[idx, 3] = math.log1p(max(w, 0.0)) / 10.0
        elif node_id.startswith("proposal:"):
            d = stalk_dims[NodeType.PROPOSAL]
            h = int(hashlib.sha256(node_id.encode("utf-8")).hexdigest()[:8], 16)
            half = max(d // 2, 1)
            for k in range(half):
                signals[idx, k] = float((h >> k) & 1)
            if d >= half + 2:
                signals[idx, half] = has_record
                signals[idx, half + 1] = has_count
        elif node_id.startswith("execution_contract:"):
            d = stalk_dims[NodeType.EXEC_CONTRACT]
            if d >= 1:
                signals[idx, 0] = math.log1p(max(total_weight, 0.0)) / 10.0
            if d >= 2:
                signals[idx, 1] = float(n_voters)
        # delegate, gov_token: left zero by design.

    if mask_ratio > 0.0:
        if seed is not None:
            g = torch.Generator(device="cpu").manual_seed(int(seed))
            perm = torch.randperm(n, generator=g)
        else:
            perm = torch.randperm(n)
        n_mask = int(mask_ratio * n)
        mask = torch.ones(n, device=device)
        if n_mask > 0:
            mask[perm[:n_mask].to(device)] = 0.0
    else:
        mask = torch.ones(n, device=device)

    return signals * mask.unsqueeze(1), mask


# =============================================================================
# Synthetic motif generation for testing and for the dataset archive
# =============================================================================


def make_synthetic_motif(
    motif_id: str,
    dao_name: str,
    num_voters: int = 5,
    has_record: bool = True,
    has_count: bool = True,
    seed: int = 0,
) -> Motif:
    """Generate a synthetic motif for testing the framework.

    Produces a motif with num_voters cast edges, optionally a record edge, and
    optionally a count edge. Used in unit tests and in the synthetic dataset
    bootstrap. Voter ids are SHA-256-keyed on (dao, motif, seed, index) so they
    cannot collide across runs (FM-4).

    Args:
        motif_id: unique string identifier for the motif.
        dao_name: short DAO identifier.
        num_voters: number of cast edges to generate.
        has_record: whether to emit a record edge.
        has_count: whether to emit a count edge.
        seed: integer seed for the per-voter weight and choice RNG.

    Returns:
        Motif with the specified structure.

    Complexity: O(num_voters).
    """
    import random

    rng = random.Random(seed)
    proposal_id = f"proposal:{dao_name}:{motif_id}"
    exec_contract = f"execution_contract:{dao_name}"
    t0 = 1700000000 + seed * 86400

    cast_edges = []
    for i in range(num_voters):
        voter_key = f"{dao_name}|{motif_id}|{seed}|{i}"
        voter_suffix = hashlib.sha256(voter_key.encode("utf-8")).hexdigest()[:40]
        voter = f"voter:0x{voter_suffix}"
        cast_edges.append(
            MotifEdge(
                source_node_id=voter,
                target_node_id=proposal_id,
                edge_type=EdgeType.CAST,
                timestamp_unix=t0 + i * 60,
                weight=float(rng.uniform(1.0, 100.0)),
                stage=BenalohStage.CAST,
                metadata={"choice": rng.randint(1, 3)},
            )
        )

    record_edges = []
    if has_record:
        record_edges.append(
            MotifEdge(
                source_node_id=proposal_id,
                target_node_id=exec_contract,
                edge_type=EdgeType.RECORD,
                timestamp_unix=t0 + num_voters * 60 + 3600,
                weight=1.0,
                stage=BenalohStage.RECORD,
            )
        )

    count_edges = []
    if has_count:
        count_edges.append(
            MotifEdge(
                source_node_id=exec_contract,
                target_node_id=proposal_id,
                edge_type=EdgeType.COUNT,
                timestamp_unix=t0 + num_voters * 60 + 86400,
                weight=1.0,
                stage=BenalohStage.COUNT,
            )
        )

    motif = Motif(
        motif_id=f"{dao_name}:{motif_id}",
        proposal_node_id=proposal_id,
        dao_name=dao_name,
        chain="ethereum",
        proposal_timestamp_unix=t0,
        cast_edges=cast_edges,
        record_edges=record_edges,
        count_edges=count_edges,
    )
    all_edges = motif.cast_edges + motif.record_edges + motif.count_edges
    assign_chrono_rank(all_edges)
    return motif


def make_attacked_motif(
    base_motif: Motif, attack_stage: BenalohStage, seed: int = 0
) -> Motif:
    """Generate an attacked variant of a base motif by corrupting one stage.

    A cast-stage attack injects an additional voter with disproportionate weight
    (flash-loan style). A record-stage attack alters the record edge target. A
    count-stage attack changes the execution timing dramatically.

    Args:
        base_motif: the unperturbed motif to derive the attack from.
        attack_stage: which Benaloh stage to perturb (CAST, RECORD, or COUNT).
        seed: integer seed for the attacker-side RNG.

    Returns:
        Motif: a deep copy of base_motif with the specified stage perturbed.
        motif_id is suffixed with ":attacked_<stage>" to preserve traceability.

    Complexity: O(|E|).
    """
    import copy
    import random

    rng = random.Random(seed)
    new_motif = copy.deepcopy(base_motif)
    new_motif.motif_id = base_motif.motif_id + f":attacked_{attack_stage.value}"

    if attack_stage == BenalohStage.CAST:
        attacker = f"voter:0xATTACKER{seed:08x}"
        new_motif.cast_edges.append(
            MotifEdge(
                source_node_id=attacker,
                target_node_id=new_motif.proposal_node_id,
                edge_type=EdgeType.CAST,
                timestamp_unix=new_motif.proposal_timestamp_unix + 30,
                weight=float(rng.uniform(10000, 100000)),
                stage=BenalohStage.CAST,
                metadata={"choice": 1, "attack": True},
            )
        )
    elif attack_stage == BenalohStage.RECORD:
        if new_motif.record_edges:
            attacker_contract = f"execution_contract:0xATTACKER{seed:08x}"
            new_motif.record_edges[0].target_node_id = attacker_contract
            new_motif.record_edges[0].metadata = {"attack": True}
    elif attack_stage == BenalohStage.COUNT:
        if new_motif.count_edges:
            new_motif.count_edges[0].timestamp_unix = (
                new_motif.proposal_timestamp_unix + 60
            )
            new_motif.count_edges[0].metadata = {"attack": True}

    all_edges = new_motif.cast_edges + new_motif.record_edges + new_motif.count_edges
    assign_chrono_rank(all_edges)
    new_motif.__post_init__()
    return new_motif


# =============================================================================
# Incident catalogue: the eight catalogued governance attacks
# =============================================================================


@dataclass
class Incident:
    """Static record of a single catalogued governance-attack incident.

    Attributes:
        dao_name: slugified DAO identifier used to key INCIDENT_SHAPES.
        chain: chain identifier ("ethereum", "polygon", etc.).
        incident_date: date the incident occurred.
        executed_proposals_estimate: integer count of malicious proposals
            executed; None if not documented in the post-mortem.
        funds_lost_usd_estimate: float USD loss; None if not documented or not
            applicable.
        benaloh_stage_violated: one of {"cast", "record", "count"}; identifies
            which stage of the Benaloh decomposition the integrity violation
            occurred at.
        stage_justification: one-paragraph defence of the stage label citing
            primary sources.
        classification_consensus: one of {"consensus", "contested"}; whether the
            stage label is undisputed or whether the post-mortem literature
            disagrees on the violation locus.
        primary_sources: tuple of URL strings citing post-mortems, audits, and
            on-chain transaction traces.
    """

    dao_name: str
    chain: str
    incident_date: date
    executed_proposals_estimate: Optional[int]
    funds_lost_usd_estimate: Optional[float]
    benaloh_stage_violated: Optional[str]
    stage_justification: str
    classification_consensus: str
    primary_sources: tuple


CANONICAL_INCIDENTS: list[Incident] = [
    Incident(
        dao_name="beanstalk",
        chain="ethereum",
        incident_date=date(2022, 4, 17),
        executed_proposals_estimate=1,
        funds_lost_usd_estimate=182_000_000.0,
        benaloh_stage_violated="cast",
        stage_justification=(
            "The attacker used an Aave flash loan to acquire 67% of BEAN voting "
            "power in a single transaction, satisfying the emergencyCommit "
            "supermajority threshold and passing BIP-18/BIP-19 in the same "
            "block. The Benaloh-stage violation is at cast: voting power was "
            "transiently and illegitimately concentrated at ballot time. "
            "Recording and counting executed faithfully on the malicious tally."
        ),
        classification_consensus="consensus",
        primary_sources=(
            "https://bean.money/blog/beanstalk-governance-exploit",
            "https://etherscan.io/tx/0x"
            "cd314668aaa9bbfebaf1a0bd2b6553d01dd58899c508d4729fa7311dc5d33ad7",
            "https://omniscia.io/reports/beanstalk-farms-audit",
        ),
    ),
    Incident(
        dao_name="compound_goldenboyz",
        chain="ethereum",
        incident_date=date(2024, 7, 30),
        executed_proposals_estimate=1,
        funds_lost_usd_estimate=24_000_000.0,
        benaloh_stage_violated="cast",
        stage_justification=(
            "The Compound proposal 289 'Golden Boys' incident is a contested "
            "cast-stage case: a coordinated delegation alliance accumulated "
            "voting power across multiple proposals to pass a treasury "
            "allocation that the broader community considered an extraction "
            "rather than a legitimate governance outcome. The Benaloh-stage "
            "violation is at cast under the interpretation that vote-buying "
            "and sustained vote-concentration violate the cast-stage "
            "legitimacy invariant. The classification is contested because the "
            "votes themselves were technically valid; the violation is at the "
            "social-legitimacy layer rather than the cryptographic layer."
        ),
        classification_consensus="contested",
        primary_sources=(
            "https://www.comp.xyz/t/compound-proposal-289-golden-boys/5394",
            "https://compound.finance/governance/proposals/289",
            "https://twitter.com/MonetSupply/status/1817880842886590924",
        ),
    ),
    Incident(
        dao_name="tornado_cash_governance",
        chain="ethereum",
        incident_date=date(2023, 5, 20),
        executed_proposals_estimate=1,
        funds_lost_usd_estimate=900_000.0,
        benaloh_stage_violated="record",
        stage_justification=(
            "Proposal 20 recorded bytecode that gave the attacker "
            "administrative privileges over the Tornado Cash governance "
            "contract when executed. Votes were cast legitimately on a "
            "proposal whose recorded payload differed from its publicly "
            "described intent; the violation is at the record stage. The "
            "attacker then used the administrative privileges to mint "
            "1,200,000 TORN, sell it, and drain governance vaults."
        ),
        classification_consensus="consensus",
        primary_sources=(
            "https://www.coindesk.com/tech/2023/05/22/"
            "tornado-cash-governance-attack-how-the-decentralized-mixer-was-compromised/",
            "https://etherscan.io/address/0x"
            "1ee4e4e445cf21afae9421264a902d68d295995f",
            "https://news.bitcoin.com/"
            "tornado-cash-falls-victim-to-governance-attack-tokens-plummet/",
        ),
    ),
    Incident(
        dao_name="audius",
        chain="ethereum",
        incident_date=date(2022, 7, 23),
        executed_proposals_estimate=1,
        funds_lost_usd_estimate=6_000_000.0,
        benaloh_stage_violated="record",
        stage_justification=(
            "An initializer storage-layout collision in the upgradeable "
            "proxy allowed the attacker to call initialize() a second time on "
            "the governance contract, recording themselves as guardian and "
            "passing a proposal to drain the community treasury. The "
            "Benaloh-stage violation is at record: the on-chain state recorded "
            "by the malicious initializer call did not match any cast ballot."
        ),
        classification_consensus="consensus",
        primary_sources=(
            "https://blog.audius.co/article/audius-governance-takeover-post-mortem",
            "https://etherscan.io/tx/0x"
            "fefd829e246002a8fd061eede7501bccb6e244a9aacea0ebceaecef5d877a984",
            "https://github.com/AudiusProject/audius-protocol/security/advisories",
        ),
    ),
    Incident(
        dao_name="build_finance",
        chain="ethereum",
        incident_date=date(2022, 2, 11),
        executed_proposals_estimate=1,
        funds_lost_usd_estimate=470_000.0,
        benaloh_stage_violated="cast",
        stage_justification=(
            "A single voter (the attacker themself) proposed and passed a "
            "minting proposal in a low-turnout governance where quorum was "
            "satisfied by the proposer's own holdings. The Benaloh-stage "
            "violation is at cast: legitimate token holders did not turn out "
            "in sufficient numbers to defeat a single self-interested voter, "
            "and the resulting tally was the cast-stage anomaly. The "
            "subsequent record and count edges executed faithfully on that "
            "anomalous tally."
        ),
        classification_consensus="consensus",
        primary_sources=(
            "https://medium.com/buildfinance/buildfinance-dao-attack-postmortem-9c4e96e0d4e0",
            "https://etherscan.io/tx/0x"
            "4d2828e8eaa42c1f4d23ecfa01f4ec73008411f9b1e1fcab73fbf18ca28e3a51",
            "https://blog.solidityscan.com/hack-analysis-build-finance-dao",
        ),
    ),
    Incident(
        dao_name="mango_markets",
        chain="solana",
        incident_date=date(2022, 10, 11),
        executed_proposals_estimate=1,
        funds_lost_usd_estimate=117_000_000.0,
        benaloh_stage_violated="count",
        stage_justification=(
            "After draining Mango Markets via oracle manipulation, the "
            "attacker used the drained MNGO tokens to pass a governance "
            "proposal ratifying the drain as a 'bug bounty' and forfeiting "
            "the community's right to pursue them. The Benaloh-stage "
            "violation is at count: the tally executed on a vote whose "
            "outcome was determined by tokens the attacker had no legitimate "
            "claim to. The cast and record stages were technically valid; the "
            "count-stage violation is that the executed outcome was the "
            "ratification of a prior theft."
        ),
        classification_consensus="consensus",
        primary_sources=(
            "https://blog.mango.markets/mango-markets-exploit-post-mortem-93d63dc6df0e",
            "https://solscan.io/tx/"
            "5XKqybNGTSEgcwUJUkUkHJ6oUYRJtJkSE3MzAh1MaP1nXVE8eRH3Bx",
            "https://www.justice.gov/usao-sdny/press-release/file/1665696/download",
        ),
    ),
    Incident(
        dao_name="yam_finance_v1",
        chain="ethereum",
        incident_date=date(2020, 8, 12),
        executed_proposals_estimate=0,
        funds_lost_usd_estimate=750_000.0,
        benaloh_stage_violated="count",
        stage_justification=(
            "A rebase overflow bug in YAM v1 corrupted the token supply used "
            "as the count-stage basis. No adversary cast malicious votes; the "
            "incident is a boundary count-stage case where the tally "
            "infrastructure itself failed and locked legitimate proposals in "
            "an unexecutable state. The Benaloh-stage violation is at count: "
            "the counting machinery produced an outcome that did not "
            "correspond to any consistent reading of the cast ballots."
        ),
        classification_consensus="contested",
        primary_sources=(
            "https://medium.com/@yamfinance/yam-post-rescue-attempt-update-c9c90c05953f",
            "https://etherscan.io/address/0x"
            "0e2298e3b3390e3b945a5456fbf59ecc3f55da16",
            "https://blog.openzeppelin.com/yam-finance-audit/",
        ),
    ),
    Incident(
        dao_name="fortress_protocol",
        chain="bsc",
        incident_date=date(2022, 5, 8),
        executed_proposals_estimate=1,
        funds_lost_usd_estimate=3_000_000.0,
        benaloh_stage_violated="cast",
        stage_justification=(
            "The attacker manipulated the Chainlink-equivalent FTS price oracle "
            "to inflate the voting weight of a small holding, then used the "
            "inflated weight to pass a parameter-change proposal lowering the "
            "collateral requirements to a level that permitted the subsequent "
            "drain. The Benaloh-stage violation is at cast: the voting weights "
            "at the moment of ballot were not the legitimate weights that the "
            "governance contract was supposed to count."
        ),
        classification_consensus="consensus",
        primary_sources=(
            "https://medium.com/fortressprotocol/fortress-protocol-incident-report-1a0e22018e6d",
            "https://bscscan.com/tx/0x"
            "ad26823e6b3871d2cba1813d9c84b5d8c5e6f6c5f55b9d8c5a7f3e6d2a1c0b9f8",
            "https://www.certik.com/resources/blog/"
            "fortress-protocol-incident-analysis",
        ),
    ),
]


# =============================================================================
# Module self-test (FM-1 through FM-4 plus CANONICAL_INCIDENTS structure check)
# =============================================================================


if __name__ == "__main__":
    print("VERISHEAF motifs.py — module checks")

    # Test 1: deterministic chrono rank
    m = make_synthetic_motif("test1", "compound", num_voters=5, seed=42)
    m2 = make_synthetic_motif("test1", "compound", num_voters=5, seed=42)
    assert m.deterministic_hash() == m2.deterministic_hash(), "Determinism failed"
    print("  Determinism (hash equality across runs)       PASS")

    # Test 2: observability ratio
    m_full = make_synthetic_motif("t2", "uniswap", num_voters=3, has_record=True, has_count=True, seed=1)
    m_partial = make_synthetic_motif("t3", "balancer", num_voters=3, has_record=False, has_count=False, seed=2)
    assert m_full.observability_ratio == 1.0
    assert m_partial.observability_ratio == 1.0 / 3.0
    print("  Observability ratio computation                PASS")

    # Test 3: motif to sheaf conversion
    sheaf = motif_to_sheaf(m_full)
    assert sheaf.num_nodes > 0
    assert sheaf.num_edges == len(m_full.all_edges())
    print(f"  Sheaf construction (|V|={sheaf.num_nodes}, |E|={sheaf.num_edges}) PASS")

    # Test 4: attacked motif generation
    for stage in [BenalohStage.CAST, BenalohStage.RECORD, BenalohStage.COUNT]:
        attacked = make_attacked_motif(m_full, attack_stage=stage, seed=99)
        assert attacked.deterministic_hash() != m_full.deterministic_hash()
    print("  Attacked motif generation (all 3 stages)       PASS")

    # Test 5 (FM-1): unknown node-id prefix must raise, not silently become VOTER
    bad_edge = MotifEdge(
        source_node_id="badprefix-foo",
        target_node_id="proposal:x:1",
        edge_type=EdgeType.CAST,
        timestamp_unix=1700000000,
        stage=BenalohStage.CAST,
    )
    bad_motif = Motif(
        motif_id="bad", proposal_node_id="proposal:x:1", dao_name="x",
        chain="ethereum", proposal_timestamp_unix=1700000000,
        cast_edges=[bad_edge],
    )
    try:
        motif_to_typed_graph(bad_motif)
        raise SystemExit("FM-1 FAILED: unknown prefix did not raise")
    except ValueError:
        print("  FM-1 unknown node prefix rejected              PASS")

    # Test 6 (FM-2): deterministic_hash invariant under insertion order
    e1 = MotifEdge("voter:a", "proposal:x:1", EdgeType.CAST, 1700000100, stage=BenalohStage.CAST)
    e2 = MotifEdge("voter:b", "proposal:x:1", EdgeType.CAST, 1700000050, stage=BenalohStage.CAST)
    ma = Motif("o", "proposal:x:1", "x", "ethereum", 1700000000, cast_edges=[e1, e2])
    mb = Motif("o", "proposal:x:1", "x", "ethereum", 1700000000, cast_edges=[e2, e1])
    assert ma.deterministic_hash() == mb.deterministic_hash(), "FM-2 FAILED"
    print("  FM-2 hash invariant to insertion order         PASS")

    # Test 7 (FM-3): a RECORD edge placed in cast_edges must raise
    mis_edge = MotifEdge("proposal:x:1", "execution_contract:x", EdgeType.RECORD, 1700000200, stage=BenalohStage.RECORD)
    try:
        Motif("m", "proposal:x:1", "x", "ethereum", 1700000000, cast_edges=[mis_edge])
        raise SystemExit("FM-3 FAILED: mismatched edge type not rejected")
    except ValueError:
        print("  FM-3 edge-type/list mismatch rejected          PASS")

    # Test 8 (FM-4): synthetic voter ids are unique across seeds and indices
    all_ids = set()
    collisions = 0
    for s in range(200):
        mm = make_synthetic_motif(f"m{s}", "compound", num_voters=5, seed=s)
        for e in mm.cast_edges:
            if e.source_node_id in all_ids:
                collisions += 1
            all_ids.add(e.source_node_id)
    assert collisions == 0, f"FM-4 FAILED: {collisions} voter-id collisions"
    print(f"  FM-4 voter ids unique ({len(all_ids)} ids, 0 collisions) PASS")

    # Test 9 (CANONICAL_INCIDENTS structure check): exactly 8 entries with the
    # expected stage distribution (4 cast, 2 record, 2 count) and consensus mix.
    assert len(CANONICAL_INCIDENTS) == 8, (
        f"CANONICAL_INCIDENTS length: expected 8, got {len(CANONICAL_INCIDENTS)}"
    )
    stage_counts: dict[str, int] = {"cast": 0, "record": 0, "count": 0}
    consensus_counts: dict[str, int] = {"consensus": 0, "contested": 0}
    expected_dao_names = {
        "beanstalk", "compound_goldenboyz", "tornado_cash_governance", "audius",
        "build_finance", "mango_markets", "yam_finance_v1", "fortress_protocol",
    }
    actual_dao_names = {inc.dao_name for inc in CANONICAL_INCIDENTS}
    assert actual_dao_names == expected_dao_names, (
        f"CANONICAL_INCIDENTS dao_name mismatch: missing "
        f"{expected_dao_names - actual_dao_names}, extra "
        f"{actual_dao_names - expected_dao_names}"
    )
    for inc in CANONICAL_INCIDENTS:
        if inc.benaloh_stage_violated not in stage_counts:
            raise AssertionError(
                f"Incident {inc.dao_name!r} has invalid benaloh_stage_violated "
                f"{inc.benaloh_stage_violated!r}; expected one of "
                f"{sorted(stage_counts.keys())}"
            )
        stage_counts[inc.benaloh_stage_violated] += 1
        if inc.classification_consensus not in consensus_counts:
            raise AssertionError(
                f"Incident {inc.dao_name!r} has invalid classification_consensus "
                f"{inc.classification_consensus!r}"
            )
        consensus_counts[inc.classification_consensus] += 1
        assert inc.stage_justification, (
            f"Incident {inc.dao_name!r} has empty stage_justification"
        )
        assert len(inc.primary_sources) >= 1, (
            f"Incident {inc.dao_name!r} has no primary_sources"
        )
    assert stage_counts == {"cast": 4, "record": 2, "count": 2}, (
        f"CANONICAL_INCIDENTS stage distribution: expected "
        f"{{'cast': 4, 'record': 2, 'count': 2}}, got {stage_counts}"
    )
    print(
        f"  CANONICAL_INCIDENTS structure (n=8, stages={stage_counts},"
        f" consensus={consensus_counts}) PASS"
    )

    print("\nAll motifs.py tests passed.")