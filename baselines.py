"""baselines.py — VERISHEAF baseline adapters for DCL-GFD, KnowGraph, NSD, and ShadowEyes.

Consolidates four published graph-based fraud-detection baselines into a single
module with a shared homogeneous-graph adapter, a shared GPU-aware training
loop, and a shared public entry-point signature pattern. All adapters consume
VERISHEAF's typed Motif input format and produce both per-motif binary anomaly
scores (for the binary anomaly detection task) and three-way Benaloh-stage
predictions (for the headline three-way task).

PUBLIC ENTRY POINTS
-------------------
    build_dcl_gfd_scorer(training_motifs, training_labels, training_stages,
                         seed, epochs) -> (binary_scorer, stage_predictor)
    build_knowgraph_scorer(training_motifs, training_labels, training_stages,
                           seed, epochs) -> (binary_scorer, stage_predictor)
    build_nsd_scorer(training_motifs, training_labels, training_stages,
                     seed, epochs, parameterization) -> (binary_scorer, stage_predictor)
    build_shadow_eyes_scorer(training_motifs, training_labels, training_stages,
                             seed, epochs) -> (binary_scorer, stage_predictor)

All four functions return a tuple of two callables:
    binary_scorer(motif: Motif) -> float in [0, 1]   higher = more anomalous
    stage_predictor(motif: Motif) -> str in {"cast", "record", "count"}

The signatures match the existing evaluation.py call sites. All four functions
invoke set_global_determinism(seed) at entry so the adapters are byte-
deterministic given the seed.

BASELINE 1: DCL-GFD
-------------------
Faithful reimplementation of Yu et al. "Dynamic Neighborhood Modeling via
Node-Subgraph Contrastive Learning for Graph-Based Fraud Detection" (AAAI 2025,
pp. 13115-13123).

Published hyperparameters [Yu+25]: Adam lr 0.01, weight decay 5e-3, hidden 64,
RWR k=8, restart r=0.5, propagation layers 1.

BASELINE 2: KnowGraph
---------------------
Faithful reimplementation of Zhou et al. "KnowGraph: Knowledge-Enabled
Anomaly Detection via Logical Reasoning on Graph Data" (ACM CCS 2024).

Published hyperparameters [Zhou+24]: Adam lr 1e-3, weight decay 1e-5, hidden 64,
layers 2, dropout 0.2.

BASELINE 3: NSD (Neural Sheaf Diffusion)
----------------------------------------
Faithful reimplementation of Bodnar et al. "Neural Sheaf Diffusion: A
Topological Perspective on Heterophily and Oversmoothing in GNNs"
(NeurIPS 2022). Per-edge restriction maps are learned via a parametric
function Phi of node features (Equation 6 of [Bod22]); three published
restriction-map parameterizations (diagonal, orthogonal via Householder
reflections, general) are supported.

Published hyperparameters [Bod22] Table 2: hidden channels {8,16,32,64},
stalk dim 1-5, layers 2-8, lr 0.01, ELU, Adam, patience 100-200.

BASELINE 4: ShadowEyes
----------------------
Faithful reimplementation of Che et al. "Across-Platform Detection of
Malicious Cryptocurrency Accounts via Interaction Feature Learning"
(IEEE TIFS 2025, vol. 20, pp. 4783-4798). Three-component pipeline:
TxGraph subgraph with HybridSub embedding, contrastive ResNet-50 backbone,
downstream MLP classifier.

Published hyperparameters [Che+25]: Adam lr 1e-4, weight decay 1e-7,
Top-K K=4, ResNet-50 first conv (1->64, kernel 3, stride 1, padding 1),
projection head 2048->512->128, MLP 2048->1024->512->2.

ADAPTATION TO VERISHEAF MOTIFS
------------------------------
All four adapters convert a Motif to a homogeneous attributed graph via the
shared _motif_to_graph helper. For the three-way task, three independent
per-stage binary models are trained (stage-routing reduction); the same
reduction is applied uniformly across baselines so the comparison measures
the value of VERISHEAF's stage-orthogonal structure rather than the value of
having a stage-aware architecture.

References:
    [Yu+25] Yu et al. AAAI 2025, pp. 13115-13123.
    [Zhou+24] Zhou et al. ACM CCS 2024.
    [Bod22] Bodnar et al. NeurIPS 2022.
    [Che+25] Che et al. IEEE TIFS 2025, vol. 20, pp. 4783-4798.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW

from motifs import Motif
from theory import BenalohStage, EdgeType
from training import set_global_determinism


# =============================================================================
# Shared device management
# =============================================================================


def _resolve_device() -> str:
    """Return the torch device string respecting CUDA_VISIBLE_DEVICES.

    Returns:
        "cuda" if torch.cuda.is_available() and at least one device is visible,
        otherwise "cpu". The CUDA_VISIBLE_DEVICES environment variable is
        respected automatically by PyTorch; this function does not override it.

    Complexity: O(1).
    """
    if torch.cuda.is_available() and torch.cuda.device_count() >= 1:
        return "cuda"
    return "cpu"


# =============================================================================
# Shared homogeneous-graph adapter
# =============================================================================


_FEATURE_DIM_DEFAULT = 16


@dataclass
class _MotifGraph:
    """Internal homogeneous-graph representation of a Motif.

    Attributes:
        x: (num_nodes, feature_dim) torch.FloatTensor of per-node features.
        edge_index: (2, num_edges) torch.LongTensor of (src, tgt) node indices.
        proposal_idx: integer index of the proposal node within the node list.
        node_ids: per-node string identifier (the canonical Motif node id).
        edge_stages: per-edge BenalohStage label (None for auxiliary edges).
    """

    x: torch.Tensor
    edge_index: torch.Tensor
    proposal_idx: int
    node_ids: list[str]
    edge_stages: list[Optional[BenalohStage]]


def _motif_to_graph(motif: Motif, feature_dim: int = _FEATURE_DIM_DEFAULT) -> _MotifGraph:
    """Convert a Motif to a homogeneous attributed graph for the baselines.

    Each unique node id becomes a graph node; each cast/record/count edge
    becomes a graph edge (auxiliary edges are also included for graph
    connectivity). Per-node features are derived from per-node edge
    statistics (in/out degree, total weights, timestamps, stage participation
    counts, type indicators).

    Args:
        motif: source Motif.
        feature_dim: dimension of the per-node feature vector. Default 16.

    Returns:
        _MotifGraph instance.

    Complexity: O(|V| + |E|).
    """
    if feature_dim < 16:
        raise ValueError(
            f"_motif_to_graph: feature_dim must be >= 16, got {feature_dim}"
        )

    node_ids: list[str] = []
    node_to_idx: dict[str, int] = {}

    def add_node(nid: str) -> int:
        if nid in node_to_idx:
            return node_to_idx[nid]
        idx = len(node_ids)
        node_to_idx[nid] = idx
        node_ids.append(nid)
        return idx

    proposal_idx = add_node(motif.proposal_node_id)

    edge_src: list[int] = []
    edge_tgt: list[int] = []
    edge_stages: list[Optional[BenalohStage]] = []

    all_edges = (
        list(motif.cast_edges)
        + list(motif.record_edges)
        + list(motif.count_edges)
        + list(motif.auxiliary_edges)
    )
    for e in all_edges:
        si = add_node(e.source_node_id)
        ti = add_node(e.target_node_id)
        edge_src.append(si)
        edge_tgt.append(ti)
        edge_stages.append(e.stage)

    n = len(node_ids)
    x = torch.zeros(n, feature_dim, dtype=torch.float32)

    in_count = np.zeros(n, dtype=np.float64)
    out_count = np.zeros(n, dtype=np.float64)
    in_weight = np.zeros(n, dtype=np.float64)
    out_weight = np.zeros(n, dtype=np.float64)
    ts_sum = np.zeros(n, dtype=np.float64)
    ts_count = np.zeros(n, dtype=np.float64)
    cast_inc = np.zeros(n, dtype=np.float64)
    record_inc = np.zeros(n, dtype=np.float64)
    count_inc = np.zeros(n, dtype=np.float64)

    for e in all_edges:
        si = node_to_idx[e.source_node_id]
        ti = node_to_idx[e.target_node_id]
        out_count[si] += 1.0
        in_count[ti] += 1.0
        out_weight[si] += float(e.weight)
        in_weight[ti] += float(e.weight)
        ts_sum[si] += float(e.timestamp_unix)
        ts_sum[ti] += float(e.timestamp_unix)
        ts_count[si] += 1.0
        ts_count[ti] += 1.0
        if e.stage == BenalohStage.CAST:
            cast_inc[si] += 1.0
            cast_inc[ti] += 1.0
        elif e.stage == BenalohStage.RECORD:
            record_inc[si] += 1.0
            record_inc[ti] += 1.0
        elif e.stage == BenalohStage.COUNT:
            count_inc[si] += 1.0
            count_inc[ti] += 1.0

    for i in range(n):
        denom_ts = max(ts_count[i], 1.0)
        x[i, 0] = float(in_count[i])
        x[i, 1] = float(out_count[i])
        x[i, 2] = float(in_weight[i])
        x[i, 3] = float(out_weight[i])
        x[i, 4] = float(ts_sum[i] / denom_ts) / 1.0e10
        x[i, 5] = float(cast_inc[i])
        x[i, 6] = float(record_inc[i])
        x[i, 7] = float(count_inc[i])
        nid = node_ids[i]
        x[i, 8] = 1.0 if nid.startswith("voter:") else 0.0
        x[i, 9] = 1.0 if nid.startswith("proposal:") else 0.0
        x[i, 10] = 1.0 if nid.startswith("execution_contract:") else 0.0
        x[i, 11] = 1.0 if nid.startswith("delegate:") else 0.0
        x[i, 12] = 1.0 if nid.startswith("token:") else 0.0
        x[i, 13] = float(in_weight[i] / max(in_count[i], 1.0))
        x[i, 14] = float(out_weight[i] / max(out_count[i], 1.0))
        x[i, 15] = float(np.log1p(in_count[i] + out_count[i]))

    if edge_src:
        edge_index = torch.tensor([edge_src, edge_tgt], dtype=torch.long)
    else:
        edge_index = torch.zeros(2, 0, dtype=torch.long)

    return _MotifGraph(
        x=x,
        edge_index=edge_index,
        proposal_idx=proposal_idx,
        node_ids=node_ids,
        edge_stages=edge_stages,
    )


def _build_adjacency_lists(edge_index: torch.Tensor, n: int) -> list[list[int]]:
    """Build undirected adjacency lists from edge_index for RWR sampling.

    Args:
        edge_index: (2, |E|) torch.LongTensor.
        n: total node count.

    Returns:
        list of length n; entry i is the list of neighbour indices of node i.

    Complexity: O(|E|).
    """
    adj: list[list[int]] = [[] for _ in range(n)]
    if edge_index.shape[1] == 0:
        return adj
    src = edge_index[0].tolist()
    tgt = edge_index[1].tolist()
    for s, t in zip(src, tgt):
        adj[s].append(t)
        adj[t].append(s)
    return adj


def _gcn_propagate(
    x: torch.Tensor, edge_index: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    """One symmetric-normalised GCN propagation step.

    Implements H_out = D^{-1/2} (A + I) D^{-1/2} H_in W with self-loops added.

    Args:
        x: (n, d_in) node features.
        edge_index: (2, |E|) undirected edge indices.
        weight: (d_in, d_out) linear weight tensor.

    Returns:
        (n, d_out) propagated features.

    Complexity: O((|E| + n) * d_out).
    """
    n = x.shape[0]
    device = x.device
    if edge_index.shape[1] == 0:
        src = torch.arange(n, device=device)
        tgt = torch.arange(n, device=device)
    else:
        src = torch.cat([edge_index[0], edge_index[1], torch.arange(n, device=device)])
        tgt = torch.cat([edge_index[1], edge_index[0], torch.arange(n, device=device)])
    deg = torch.zeros(n, device=device)
    deg.scatter_add_(0, src, torch.ones_like(src, dtype=torch.float32))
    deg_inv_sqrt = (deg + 1e-8).pow(-0.5)
    norm = deg_inv_sqrt[src] * deg_inv_sqrt[tgt]
    msgs = x[src] * norm.unsqueeze(1)
    out = torch.zeros(n, x.shape[1], device=device)
    out.index_add_(0, tgt, msgs)
    return out @ weight


# =============================================================================
# DCL-GFD namespace
# =============================================================================


class _dcl_gfd:
    """Namespace for DCL-GFD-specific components (Yu et al. AAAI 2025)."""

    LR: float = 0.01
    WEIGHT_DECAY: float = 5.0e-3
    HIDDEN: int = 64
    RWR_K: int = 8
    RWR_R: float = 0.5
    N_PROP_LAYERS: int = 1
    CL_LAMBDA: float = 0.5
    DROPOUT: float = 0.0

    @staticmethod
    def sample_rwr_subgraph(
        target: int, adj: list[list[int]], k: int, r: float, rng: np.random.RandomState
    ) -> list[int]:
        """Sample an RWR subgraph rooted at `target` per [Yu+25] §3.2."""
        visited: set[int] = {target}
        current = target
        for _ in range(k):
            if rng.rand() < r or not adj[current]:
                current = target
            else:
                current = adj[current][rng.randint(0, len(adj[current]))]
            visited.add(current)
        return sorted(visited)


class _DCLGFDModel(nn.Module):
    """Faithful DCL-GFD architecture per [Yu+25]."""

    def __init__(self, in_dim: int, hidden: int):
        super().__init__()
        self.in_dim = in_dim
        self.hidden = hidden
        self.gcn_w1 = nn.Parameter(torch.empty(in_dim, hidden))
        self.gcn_w2 = nn.Parameter(torch.empty(hidden, hidden))
        nn.init.xavier_uniform_(self.gcn_w1)
        nn.init.xavier_uniform_(self.gcn_w2)
        self.target_ffn = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        self.W_be = nn.Parameter(torch.empty(in_dim, hidden))
        self.W_fr = nn.Parameter(torch.empty(in_dim, hidden))
        nn.init.xavier_uniform_(self.W_be)
        nn.init.xavier_uniform_(self.W_fr)
        self.head = nn.Sequential(
            nn.Linear(in_dim + hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def encode_subgraph(self, x_sub: torch.Tensor, sub_edge_index: torch.Tensor) -> torch.Tensor:
        h = F.relu(_gcn_propagate(x_sub, sub_edge_index, self.gcn_w1))
        h = _gcn_propagate(h, sub_edge_index, self.gcn_w2)
        return h.mean(dim=0)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n = x.shape[0]
        device = x.device
        with torch.no_grad():
            alpha_prior = torch.full((n, 1), 0.5, device=device)
        h_be = _gcn_propagate(x, edge_index, self.W_be)
        h_fr = _gcn_propagate(x, edge_index, self.W_fr)
        alpha_hat = (1.0 + alpha_prior) / 2.0
        h_neigh = alpha_hat * h_be + (1.0 - alpha_hat) * h_fr
        ego_concat = torch.cat([x, h_neigh], dim=1)
        logits = self.head(ego_concat).squeeze(-1)
        return logits, h_neigh


def _train_dcl_gfd(
    training_motifs: list[Motif],
    training_labels: list[int],
    device: str,
    seed: int,
    epochs: int,
) -> _DCLGFDModel:
    """Train a _DCLGFDModel on the provided labelled motif corpus."""
    set_global_determinism(seed)
    if len(training_motifs) != len(training_labels):
        raise ValueError(
            f"_train_dcl_gfd: training_motifs ({len(training_motifs)}) and "
            f"training_labels ({len(training_labels)}) length mismatch"
        )
    if not training_motifs:
        raise ValueError("_train_dcl_gfd: empty training corpus")

    model = _DCLGFDModel(in_dim=_FEATURE_DIM_DEFAULT, hidden=_dcl_gfd.HIDDEN).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=_dcl_gfd.LR,
        weight_decay=_dcl_gfd.WEIGHT_DECAY,
    )
    rng = np.random.RandomState(seed)

    pos = sum(1 for lbl in training_labels if lbl == 1)
    neg = len(training_labels) - pos
    pos_weight = torch.tensor(
        [max(neg, 1) / max(pos, 1)], device=device, dtype=torch.float32
    )

    for epoch in range(epochs):
        model.train()
        perm = rng.permutation(len(training_motifs))
        epoch_loss = 0.0
        for idx in perm:
            motif = training_motifs[int(idx)]
            label = float(training_labels[int(idx)])
            g = _motif_to_graph(motif, feature_dim=_FEATURE_DIM_DEFAULT)
            x = g.x.to(device)
            ei = g.edge_index.to(device)

            optimizer.zero_grad()
            logits, _h_neigh = model(x, ei)

            target_logit = logits[g.proposal_idx].unsqueeze(0)
            target_label = torch.tensor([label], device=device, dtype=torch.float32)
            l_fd = F.binary_cross_entropy_with_logits(
                target_logit, target_label, pos_weight=pos_weight
            )

            adj = _build_adjacency_lists(g.edge_index, x.shape[0])
            sub_nodes = _dcl_gfd.sample_rwr_subgraph(
                g.proposal_idx, adj,
                k=_dcl_gfd.RWR_K, r=_dcl_gfd.RWR_R, rng=rng,
            )
            if len(sub_nodes) > 1:
                sub_idx_tensor = torch.tensor(sub_nodes, device=device, dtype=torch.long)
                x_sub = x[sub_idx_tensor].clone()
                rel_target = sub_nodes.index(g.proposal_idx)
                x_sub[rel_target] = 0.0
                node_remap = {n: i for i, n in enumerate(sub_nodes)}
                sub_src, sub_tgt = [], []
                src_list = g.edge_index[0].tolist()
                tgt_list = g.edge_index[1].tolist()
                for s, t in zip(src_list, tgt_list):
                    if s in node_remap and t in node_remap:
                        sub_src.append(node_remap[s])
                        sub_tgt.append(node_remap[t])
                if sub_src:
                    sub_ei = torch.tensor([sub_src, sub_tgt], device=device, dtype=torch.long)
                else:
                    sub_ei = torch.zeros(2, 0, device=device, dtype=torch.long)
                sub_readout = model.encode_subgraph(x_sub, sub_ei)
                target_proj = model.target_ffn(x[g.proposal_idx])
                sim = F.cosine_similarity(
                    target_proj.unsqueeze(0), sub_readout.unsqueeze(0)
                ).squeeze()
                l_cl = -((1.0 - label) * sim - label * sim)
            else:
                l_cl = torch.zeros((), device=device)

            loss = l_fd + _dcl_gfd.CL_LAMBDA * l_cl
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())

        if epoch % max(1, epochs // 5) == 0 or epoch == epochs - 1:
            print(
                f"    [DCL-GFD seed {seed}] epoch {epoch:3d} | "
                f"loss {epoch_loss / max(len(perm), 1):.4e}",
                flush=True,
            )

    return model


def _score_motif_dcl_gfd(model: _DCLGFDModel, motif: Motif, device: str) -> float:
    """Score a motif: sigmoid of the proposal-node logit."""
    model.eval()
    with torch.no_grad():
        g = _motif_to_graph(motif, feature_dim=_FEATURE_DIM_DEFAULT)
        logits, _ = model(g.x.to(device), g.edge_index.to(device))
        return float(torch.sigmoid(logits[g.proposal_idx]).item())


def build_dcl_gfd_scorer(
    training_motifs: list[Motif],
    training_labels: list[int],
    training_stages: Optional[list[Optional[BenalohStage]]] = None,
    seed: int = 42,
    epochs: int = 30,
) -> tuple[Callable[[Motif], float], Callable[[Motif], str]]:
    """Build (binary_scorer, stage_predictor) DCL-GFD adapter callables.

    Citation: manuscript Section 7 (Baseline Adapters), [Yu+25].
    """
    set_global_determinism(seed)
    device = _resolve_device()
    binary_model = _train_dcl_gfd(
        training_motifs, training_labels,
        device=device, seed=seed, epochs=epochs,
    )

    stage_models: dict[str, _DCLGFDModel] = {}
    if training_stages is not None:
        for stage_name in ("cast", "record", "count"):
            per_stage_labels: list[int] = []
            for lbl, stg in zip(training_labels, training_stages):
                if lbl == 0:
                    per_stage_labels.append(0)
                elif stg is not None and stg.value == stage_name:
                    per_stage_labels.append(1)
                else:
                    per_stage_labels.append(0)
            if sum(per_stage_labels) > 0:
                stage_seed = seed + abs(hash(stage_name)) % 1000
                stage_models[stage_name] = _train_dcl_gfd(
                    training_motifs, per_stage_labels,
                    device=device, seed=stage_seed, epochs=epochs,
                )

    def binary_scorer(motif: Motif) -> float:
        return _score_motif_dcl_gfd(binary_model, motif, device=device)

    def stage_predictor(motif: Motif) -> str:
        if not stage_models:
            return "cast"
        scores = {
            s: _score_motif_dcl_gfd(m, motif, device=device)
            for s, m in stage_models.items()
        }
        return max(scores, key=lambda s: scores[s])

    return binary_scorer, stage_predictor


# =============================================================================
# KnowGraph namespace
# =============================================================================


class _knowgraph:
    """Namespace for KnowGraph-specific components ([Zhou+24] ACM CCS 2024)."""

    LR: float = 1.0e-3
    WEIGHT_DECAY: float = 1.0e-5
    HIDDEN: int = 64
    N_LAYERS: int = 2
    DROPOUT: float = 0.2
    N_RELATIONS: int = 5

    EDGE_TYPE_TO_REL: dict[str, int] = {
        EdgeType.CAST.value: 0,
        EdgeType.DELEGATE.value: 1,
        EdgeType.TRANSFER.value: 2,
        EdgeType.RECORD.value: 3,
        EdgeType.COUNT.value: 4,
    }


class _KnowledgeGCN(nn.Module):
    """Relational message-passing model per [Zhou+24] Section 4."""

    def __init__(self, in_dim: int, hidden: int, n_layers: int, n_relations: int, dropout: float):
        super().__init__()
        self.n_layers = n_layers
        self.n_relations = n_relations
        self.dropout = dropout
        self.weights = nn.ParameterList()
        prev_dim = in_dim
        for _ in range(n_layers):
            w = nn.Parameter(torch.empty(n_relations, prev_dim, hidden))
            nn.init.xavier_uniform_(w)
            self.weights.append(w)
            prev_dim = hidden
        self.self_w = nn.ParameterList()
        prev_dim = in_dim
        for _ in range(n_layers):
            sw = nn.Parameter(torch.empty(prev_dim, hidden))
            nn.init.xavier_uniform_(sw)
            self.self_w.append(sw)
            prev_dim = hidden
        self.head = nn.Linear(hidden, 1)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_rels: torch.Tensor,
    ) -> torch.Tensor:
        n = x.shape[0]
        device = x.device
        h = x
        for layer in range(self.n_layers):
            new_h = h @ self.self_w[layer]
            if edge_index.shape[1] > 0:
                for r in range(self.n_relations):
                    mask = edge_rels == r
                    if mask.any():
                        ei_r = edge_index[:, mask]
                        src_r = torch.cat([ei_r[0], ei_r[1]])
                        tgt_r = torch.cat([ei_r[1], ei_r[0]])
                        deg = torch.zeros(n, device=device)
                        deg.scatter_add_(0, src_r, torch.ones_like(src_r, dtype=torch.float32))
                        deg_inv_sqrt = (deg + 1e-8).pow(-0.5)
                        norm = deg_inv_sqrt[src_r] * deg_inv_sqrt[tgt_r]
                        msgs = h[src_r] * norm.unsqueeze(1)
                        prop = torch.zeros(n, h.shape[1], device=device)
                        prop.index_add_(0, tgt_r, msgs)
                        new_h = new_h + prop @ self.weights[layer][r]
            new_h = F.relu(new_h)
            new_h = F.dropout(new_h, p=self.dropout, training=self.training)
            h = new_h
        return self.head(h).squeeze(-1)


def _motif_to_relational_graph(
    motif: Motif, feature_dim: int = _FEATURE_DIM_DEFAULT
) -> tuple[_MotifGraph, torch.Tensor]:
    """Convert a Motif into the (graph, edge_rels) pair the KnowledgeGCN needs."""
    g = _motif_to_graph(motif, feature_dim=feature_dim)
    rels: list[int] = []
    all_edges = (
        list(motif.cast_edges)
        + list(motif.record_edges)
        + list(motif.count_edges)
        + list(motif.auxiliary_edges)
    )
    for e in all_edges:
        rels.append(_knowgraph.EDGE_TYPE_TO_REL.get(e.edge_type.value, 0))
    edge_rels = torch.tensor(rels, dtype=torch.long)
    return g, edge_rels


def _train_knowgraph(
    training_motifs: list[Motif],
    training_labels: list[int],
    device: str,
    seed: int,
    epochs: int,
) -> _KnowledgeGCN:
    """Train a _KnowledgeGCN on the provided labelled motif corpus."""
    set_global_determinism(seed)
    if len(training_motifs) != len(training_labels):
        raise ValueError(
            f"_train_knowgraph: training_motifs ({len(training_motifs)}) and "
            f"training_labels ({len(training_labels)}) length mismatch"
        )
    if not training_motifs:
        raise ValueError("_train_knowgraph: empty training corpus")

    model = _KnowledgeGCN(
        in_dim=_FEATURE_DIM_DEFAULT,
        hidden=_knowgraph.HIDDEN,
        n_layers=_knowgraph.N_LAYERS,
        n_relations=_knowgraph.N_RELATIONS,
        dropout=_knowgraph.DROPOUT,
    ).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=_knowgraph.LR,
        weight_decay=_knowgraph.WEIGHT_DECAY,
    )
    rng = np.random.RandomState(seed)

    pos = sum(1 for lbl in training_labels if lbl == 1)
    neg = len(training_labels) - pos
    pos_weight = torch.tensor(
        [max(neg, 1) / max(pos, 1)], device=device, dtype=torch.float32
    )

    for epoch in range(epochs):
        model.train()
        perm = rng.permutation(len(training_motifs))
        epoch_loss = 0.0
        for idx in perm:
            motif = training_motifs[int(idx)]
            label = float(training_labels[int(idx)])
            g, edge_rels = _motif_to_relational_graph(motif, feature_dim=_FEATURE_DIM_DEFAULT)
            x = g.x.to(device)
            ei = g.edge_index.to(device)
            er = edge_rels.to(device)

            optimizer.zero_grad()
            logits = model(x, ei, er)
            target_logit = logits[g.proposal_idx].unsqueeze(0)
            target_label = torch.tensor([label], device=device, dtype=torch.float32)
            loss = F.binary_cross_entropy_with_logits(
                target_logit, target_label, pos_weight=pos_weight
            )
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())

        if epoch % max(1, epochs // 5) == 0 or epoch == epochs - 1:
            print(
                f"    [KnowGraph seed {seed}] epoch {epoch:3d} | "
                f"loss {epoch_loss / max(len(perm), 1):.4e}",
                flush=True,
            )

    return model


def _score_motif_knowgraph(model: _KnowledgeGCN, motif: Motif, device: str) -> float:
    """Score a motif under the trained KnowledgeGCN."""
    model.eval()
    with torch.no_grad():
        g, edge_rels = _motif_to_relational_graph(motif, feature_dim=_FEATURE_DIM_DEFAULT)
        logits = model(g.x.to(device), g.edge_index.to(device), edge_rels.to(device))
        return float(torch.sigmoid(logits[g.proposal_idx]).item())


def build_knowgraph_scorer(
    training_motifs: list[Motif],
    training_labels: list[int],
    training_stages: Optional[list[Optional[BenalohStage]]] = None,
    seed: int = 42,
    epochs: int = 30,
) -> tuple[Callable[[Motif], float], Callable[[Motif], str]]:
    """Build (binary_scorer, stage_predictor) KnowGraph adapter callables.

    Citation: manuscript Section 7 (Baseline Adapters), [Zhou+24].
    """
    set_global_determinism(seed)
    device = _resolve_device()
    binary_model = _train_knowgraph(
        training_motifs, training_labels,
        device=device, seed=seed, epochs=epochs,
    )

    stage_models: dict[str, _KnowledgeGCN] = {}
    if training_stages is not None:
        for stage_name in ("cast", "record", "count"):
            per_stage_labels: list[int] = []
            for lbl, stg in zip(training_labels, training_stages):
                if lbl == 0:
                    per_stage_labels.append(0)
                elif stg is not None and stg.value == stage_name:
                    per_stage_labels.append(1)
                else:
                    per_stage_labels.append(0)
            if sum(per_stage_labels) > 0:
                stage_seed = seed + abs(hash(stage_name)) % 1000
                stage_models[stage_name] = _train_knowgraph(
                    training_motifs, per_stage_labels,
                    device=device, seed=stage_seed, epochs=epochs,
                )

    def binary_scorer(motif: Motif) -> float:
        return _score_motif_knowgraph(binary_model, motif, device=device)

    def stage_predictor(motif: Motif) -> str:
        if not stage_models:
            return "cast"
        scores = {
            s: _score_motif_knowgraph(m, motif, device=device)
            for s, m in stage_models.items()
        }
        return max(scores, key=lambda s: scores[s])

    return binary_scorer, stage_predictor


# =============================================================================
# NSD (Neural Sheaf Diffusion) namespace
# =============================================================================


class _nsd:
    """Namespace for NSD-specific components ([Bod22] NeurIPS 2022).

    Published hyperparameter grid [Bod22] Table 2:
        Hidden channels:    {8, 16, 32, 64}      -> default 32 (mid-grid)
        Stalk dimension d:  1 - 5                -> default 3 (mid-grid)
        Layers:             2 - 8                -> default 3 (small-graph bias)
        Learning rate:      0.01                 -> default 0.01
        Weight decay:       log-uniform          -> default 5e-4
        Activation:         ELU                  -> ELU
        Optimizer:          Adam                 -> Adam (not AdamW)
        Patience:           100 - 200            -> default 200
        Input dropout:      Uniform [0, 0.9]     -> default 0.3
        Layer dropout:      Uniform [0, 0.9]     -> default 0.3

    The default configuration corresponds to the mid-grid hyperparameters of
    the published search range. The build_nsd_scorer signature accepts the
    parameterization argument which selects among the three published
    restriction-map parameterizations.
    """

    LR: float = 0.01
    WEIGHT_DECAY: float = 5.0e-4
    HIDDEN: int = 32
    STALK_DIM: int = 3
    N_LAYERS: int = 3
    INPUT_DROPOUT: float = 0.3
    LAYER_DROPOUT: float = 0.3
    PATIENCE: int = 200
    PARAMETERIZATION_DEFAULT: str = "orthogonal"
    SUPPORTED_PARAMETERIZATIONS: tuple = ("diagonal", "orthogonal", "general")


class _NSDSheafLearner(nn.Module):
    """Parametric Phi function that learns per-edge restriction maps [Bod22] §5.

    Implements Phi(x_v, x_u) = tanh(V [x_v || x_u]) followed by a reshape into
    a d-by-d restriction map. Three parameterizations are supported:

        DIAGONAL    Phi outputs d scalars, becomes a d-by-d diagonal matrix.
        ORTHOGONAL  Phi outputs d Householder reflection vectors of dimension
                    d; the d-by-d restriction map is the product of d
                    Householder reflections, giving an element of O(d).
        GENERAL     Phi outputs d * d scalars, becomes a general d-by-d
                    matrix.

    tanh activation matches the bounded-output requirement of [Bod22] §5 for
    well-conditioned Laplacian normalization.
    """

    def __init__(self, in_dim: int, stalk_dim: int, parameterization: str):
        super().__init__()
        self.in_dim = in_dim
        self.stalk_dim = stalk_dim
        self.parameterization = parameterization

        if parameterization == "diagonal":
            out_dim = stalk_dim
        elif parameterization == "orthogonal":
            out_dim = stalk_dim * stalk_dim
        elif parameterization == "general":
            out_dim = stalk_dim * stalk_dim
        else:
            raise ValueError(
                f"_NSDSheafLearner: parameterization {parameterization!r} not in "
                f"{_nsd.SUPPORTED_PARAMETERIZATIONS}"
            )

        self.linear = nn.Linear(2 * in_dim, out_dim, bias=False)
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, x_v: torch.Tensor, x_u: torch.Tensor) -> torch.Tensor:
        """Compute the restriction map F_{v -> e} from concatenated node features.

        Args:
            x_v: (E, in_dim) features of the v-endpoint of each edge.
            x_u: (E, in_dim) features of the u-endpoint of each edge.

        Returns:
            (E, stalk_dim, stalk_dim) per-edge restriction-map tensor.

        Complexity: O(E * (in_dim + stalk_dim^2)) for diagonal and general;
        O(E * (in_dim + stalk_dim^3)) for orthogonal due to the iterative
        Householder product.
        """
        concat = torch.cat([x_v, x_u], dim=-1)
        raw = torch.tanh(self.linear(concat))

        if self.parameterization == "diagonal":
            return torch.diag_embed(raw)
        elif self.parameterization == "orthogonal":
            return self._householder_orthogonal(raw)
        else:
            return raw.view(*raw.shape[:-1], self.stalk_dim, self.stalk_dim)

    def _householder_orthogonal(self, params: torch.Tensor) -> torch.Tensor:
        """Build an orthogonal d-by-d matrix from d Householder reflections.

        Each Householder reflection H_i = I - 2 v_i v_i^T / (v_i^T v_i) is
        orthogonal; the product of d such reflections spans O(d) generically.
        Per [Bod22] §5 we use this construction because it is differentiable
        and produces strictly orthogonal matrices without an explicit
        projection step.

        Args:
            params: (..., d * d) tensor of d Householder vector entries.

        Returns:
            (..., d, d) orthogonal matrix tensor.

        Complexity: O(prod(...) * d^3).
        """
        d = self.stalk_dim
        batch_shape = params.shape[:-1]
        vectors = params.view(*batch_shape, d, d)

        device = params.device
        dtype = params.dtype
        I = torch.eye(d, device=device, dtype=dtype)
        Q = I.expand(*batch_shape, d, d).clone()

        for i in range(d):
            v = vectors[..., i, :]
            v_norm_sq = (v * v).sum(dim=-1, keepdim=True).clamp(min=1e-8)
            Qv = torch.matmul(Q, v.unsqueeze(-1)).squeeze(-1)
            scale = (2.0 / v_norm_sq).unsqueeze(-1)
            Q = Q - scale * Qv.unsqueeze(-1) * v.unsqueeze(-2)

        return Q


class _NSDLayer(nn.Module):
    """One discrete sheaf diffusion layer per [Bod22] Equation 6.

    Implements the residual update

        X_{t+1} = (1 + epsilon) X_t - sigma( Delta_F (I_n ⊗ W1) X_t W2 )

    where W1 is the stalk-dimension transform, W2 is the channel transform,
    epsilon is a learnable per-stalk-dimension scaling parameter, and Delta_F
    is the augmented-normalised sheaf Laplacian computed externally and
    supplied as the second forward-pass argument.
    """

    def __init__(self, stalk_dim: int, channels: int):
        super().__init__()
        self.stalk_dim = stalk_dim
        self.channels = channels
        self.W1 = nn.Parameter(torch.empty(stalk_dim, stalk_dim))
        self.W2 = nn.Parameter(torch.empty(channels, channels))
        self.epsilon = nn.Parameter(torch.zeros(stalk_dim))
        nn.init.xavier_uniform_(self.W1)
        nn.init.xavier_uniform_(self.W2)

    def forward(self, x: torch.Tensor, laplacian: torch.Tensor) -> torch.Tensor:
        """Apply one residual sheaf diffusion step.

        Args:
            x: (n, d, f) tensor of stalk-dimensional node features per channel.
            laplacian: (n * d, n * d) augmented-normalised sheaf Laplacian.

        Returns:
            (n, d, f) updated node features.

        Complexity: O(n^2 * d^2 * f) for the dense Laplacian matvec, plus
        O(n * d * f * (d + f)) for the W1, W2 multiplications.
        """
        n, d, f = x.shape
        x_w1 = torch.einsum("ij,njf->nif", self.W1, x)
        x_w1w2 = torch.einsum("ndf,fg->ndg", x_w1, self.W2)
        x_flat = x_w1w2.reshape(n * d, f)
        diffused_flat = laplacian @ x_flat
        diffused = diffused_flat.reshape(n, d, f)
        diffused = F.elu(diffused)
        epsilon_b = self.epsilon.view(1, d, 1)
        return (1.0 + epsilon_b) * x - diffused


class _NSDModel(nn.Module):
    """Faithful Neural Sheaf Diffusion model per [Bod22] Equation 6.

    Architecture pipeline:
      (1) Input projection: per-node features -> (stalk_dim * channels)
          followed by reshape to (n, stalk_dim, channels).
      (2) For each layer t:
          (2a) Compute per-edge restriction maps via _NSDSheafLearner from the
               current node-feature tensor.
          (2b) Construct the augmented-normalised sheaf Laplacian Delta_F(t).
          (2c) Apply one _NSDLayer to produce the next layer's node features.
          (2d) Apply layer dropout.
      (3) Output head: read out the proposal-node embedding (stalk_dim *
          channels) and pass through a linear classifier to produce the
          per-motif logit.
    """

    def __init__(
        self,
        in_dim: int,
        stalk_dim: int,
        n_layers: int,
        channels: int,
        parameterization: str,
        input_dropout: float,
        layer_dropout: float,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.stalk_dim = stalk_dim
        self.n_layers = n_layers
        self.channels = channels
        self.input_dropout = input_dropout
        self.layer_dropout = layer_dropout
        self.parameterization = parameterization

        self.input_proj = nn.Linear(in_dim, stalk_dim * channels)
        self.sheaf_learners = nn.ModuleList([
            _NSDSheafLearner(stalk_dim * channels, stalk_dim, parameterization)
            for _ in range(n_layers)
        ])
        self.layers = nn.ModuleList([
            _NSDLayer(stalk_dim, channels) for _ in range(n_layers)
        ])
        self.head = nn.Linear(stalk_dim * channels, 1)

    def _build_normalised_laplacian(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        sheaf_learner: _NSDSheafLearner,
    ) -> torch.Tensor:
        """Construct Delta_F = (D + I)^{-1/2} L_F (D + I)^{-1/2}.

        The sheaf coboundary is delta : C^0 -> C^1 with per-edge action
        (delta x)_e = F_{tgt -> e} x_tgt - F_{src -> e} x_src. The sheaf
        Laplacian L_F = delta^T delta has block structure

            L_F[v, v] = sum over incident edges of F_{v -> e}^T F_{v -> e}
            L_F[v, u] = - F_{v -> e}^T F_{u -> e}   for edge e = (v, u)

        The augmented normalisation D_tilde = D + I_{nd} is taken on the block
        diagonal of L_F, with the block-wise inverse square root computed via
        eigendecomposition of each d-by-d positive-definite block.

        Args:
            node_features: (n, stalk_dim * channels) flat node feature tensor.
            edge_index: (2, |E|) edge indices.
            sheaf_learner: the per-layer _NSDSheafLearner module.

        Returns:
            (n * d, n * d) dense augmented-normalised sheaf Laplacian.

        Complexity: O((n + |E|) * d^2 + n * d^3) for the per-node block
        eigendecomposition.
        """
        n = node_features.shape[0]
        d = self.stalk_dim
        device = node_features.device
        dtype = node_features.dtype

        L = torch.zeros(n * d, n * d, device=device, dtype=dtype)

        if edge_index.shape[1] > 0:
            src = edge_index[0]
            tgt = edge_index[1]
            x_src = node_features[src]
            x_tgt = node_features[tgt]
            F_src = sheaf_learner(x_src, x_tgt)
            F_tgt = sheaf_learner(x_tgt, x_src)

            for e in range(edge_index.shape[1]):
                s = int(src[e].item())
                t = int(tgt[e].item())
                Fs = F_src[e]
                Ft = F_tgt[e]
                L[s * d:(s + 1) * d, s * d:(s + 1) * d] = (
                    L[s * d:(s + 1) * d, s * d:(s + 1) * d] + Fs.T @ Fs
                )
                L[t * d:(t + 1) * d, t * d:(t + 1) * d] = (
                    L[t * d:(t + 1) * d, t * d:(t + 1) * d] + Ft.T @ Ft
                )
                L[s * d:(s + 1) * d, t * d:(t + 1) * d] = (
                    L[s * d:(s + 1) * d, t * d:(t + 1) * d] - Fs.T @ Ft
                )
                L[t * d:(t + 1) * d, s * d:(s + 1) * d] = (
                    L[t * d:(t + 1) * d, s * d:(s + 1) * d] - Ft.T @ Fs
                )

        I_d = torch.eye(d, device=device, dtype=dtype)
        D_inv_sqrt = torch.zeros_like(L)
        for v in range(n):
            block = L[v * d:(v + 1) * d, v * d:(v + 1) * d] + I_d
            eigvals, eigvecs = torch.linalg.eigh(block)
            eigvals_inv_sqrt = eigvals.clamp(min=1e-8).pow(-0.5)
            block_inv_sqrt = eigvecs @ torch.diag(eigvals_inv_sqrt) @ eigvecs.T
            D_inv_sqrt[v * d:(v + 1) * d, v * d:(v + 1) * d] = block_inv_sqrt

        return D_inv_sqrt @ L @ D_inv_sqrt

    def forward(self, x_in: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Return per-node logits.

        Args:
            x_in: (n, in_dim) raw node features.
            edge_index: (2, |E|) edge indices.

        Returns:
            (n,) per-node logits.

        Complexity: O(n_layers * (n^2 * d^2 * channels + n * d^3)).
        """
        n = x_in.shape[0]
        x_in = F.dropout(x_in, p=self.input_dropout, training=self.training)
        x = self.input_proj(x_in)
        x = x.view(n, self.stalk_dim, self.channels)

        for t in range(self.n_layers):
            x_flat = x.view(n, self.stalk_dim * self.channels)
            laplacian = self._build_normalised_laplacian(
                x_flat, edge_index, self.sheaf_learners[t]
            )
            x = self.layers[t](x, laplacian)
            x = F.dropout(x, p=self.layer_dropout, training=self.training)

        x_out = x.view(n, self.stalk_dim * self.channels)
        return self.head(x_out).squeeze(-1)


def _train_nsd(
    training_motifs: list[Motif],
    training_labels: list[int],
    device: str,
    seed: int,
    epochs: int,
    parameterization: str,
) -> _NSDModel:
    """Train an _NSDModel on the provided labelled motif corpus.

    Uses the published Adam optimiser with lr=0.01 and weight decay 5e-4
    (mid-range of the published log-uniform search). Loss is weighted binary
    cross-entropy on the proposal-node logit. Early stopping monitors the
    best-loss epoch with patience PATIENCE.

    Args:
        training_motifs: list of Motif.
        training_labels: per-motif binary label (0 normal, 1 attacked).
        device: torch device string.
        seed: integer seed for set_global_determinism.
        epochs: number of training epochs.
        parameterization: one of {"diagonal", "orthogonal", "general"}.

    Returns:
        Trained _NSDModel.

    Complexity: O(epochs * |motifs| * (n^2 * d^2 * channels + n * d^3)).
    """
    set_global_determinism(seed)
    if len(training_motifs) != len(training_labels):
        raise ValueError(
            f"_train_nsd: training_motifs ({len(training_motifs)}) and "
            f"training_labels ({len(training_labels)}) length mismatch"
        )
    if not training_motifs:
        raise ValueError("_train_nsd: empty training corpus")
    if parameterization not in _nsd.SUPPORTED_PARAMETERIZATIONS:
        raise ValueError(
            f"_train_nsd: parameterization {parameterization!r} not in "
            f"{_nsd.SUPPORTED_PARAMETERIZATIONS}"
        )

    model = _NSDModel(
        in_dim=_FEATURE_DIM_DEFAULT,
        stalk_dim=_nsd.STALK_DIM,
        n_layers=_nsd.N_LAYERS,
        channels=_nsd.HIDDEN,
        parameterization=parameterization,
        input_dropout=_nsd.INPUT_DROPOUT,
        layer_dropout=_nsd.LAYER_DROPOUT,
    ).to(device)
    optimizer = Adam(
        model.parameters(),
        lr=_nsd.LR,
        weight_decay=_nsd.WEIGHT_DECAY,
    )
    rng = np.random.RandomState(seed)

    pos = sum(1 for lbl in training_labels if lbl == 1)
    neg = len(training_labels) - pos
    pos_weight = torch.tensor(
        [max(neg, 1) / max(pos, 1)], device=device, dtype=torch.float32
    )

    best_loss = float("inf")
    best_state: Optional[dict] = None
    epochs_without_improvement = 0

    for epoch in range(epochs):
        model.train()
        perm = rng.permutation(len(training_motifs))
        epoch_loss = 0.0
        for idx in perm:
            motif = training_motifs[int(idx)]
            label = float(training_labels[int(idx)])
            g = _motif_to_graph(motif, feature_dim=_FEATURE_DIM_DEFAULT)
            x = g.x.to(device)
            ei = g.edge_index.to(device)

            optimizer.zero_grad()
            logits = model(x, ei)
            target_logit = logits[g.proposal_idx].unsqueeze(0)
            target_label = torch.tensor([label], device=device, dtype=torch.float32)
            loss = F.binary_cross_entropy_with_logits(
                target_logit, target_label, pos_weight=pos_weight
            )
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())

        mean_loss = epoch_loss / max(len(perm), 1)

        if mean_loss < best_loss - 1e-6:
            best_loss = mean_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= _nsd.PATIENCE:
                break

        if epoch % max(1, epochs // 5) == 0 or epoch == epochs - 1:
            print(
                f"    [NSD-{parameterization} seed {seed}] epoch {epoch:3d} | "
                f"loss {mean_loss:.4e}",
                flush=True,
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


def _score_motif_nsd(model: _NSDModel, motif: Motif, device: str) -> float:
    """Score a motif: sigmoid of the proposal-node logit under the NSD model.

    Args:
        model: trained _NSDModel.
        motif: source Motif.
        device: torch device string.

    Returns:
        Float in [0, 1]; higher = more anomalous.
    """
    model.eval()
    with torch.no_grad():
        g = _motif_to_graph(motif, feature_dim=_FEATURE_DIM_DEFAULT)
        logits = model(g.x.to(device), g.edge_index.to(device))
        return float(torch.sigmoid(logits[g.proposal_idx]).item())


def build_nsd_scorer(
    training_motifs: list[Motif],
    training_labels: list[int],
    training_stages: Optional[list[Optional[BenalohStage]]] = None,
    seed: int = 42,
    epochs: int = 30,
    parameterization: str = _nsd.PARAMETERIZATION_DEFAULT,
) -> tuple[Callable[[Motif], float], Callable[[Motif], str]]:
    """Build (binary_scorer, stage_predictor) NSD adapter callables.

    Same signature pattern as build_dcl_gfd_scorer and build_knowgraph_scorer
    with the addition of the parameterization argument, which selects among
    the three published restriction-map parameterizations
    {"diagonal", "orthogonal", "general"}.

    For the three-way stage classification task, three independent per-stage
    binary NSD models are trained (the same stage-routing reduction applied
    to DCL-GFD and KnowGraph). This ensures that the NSD comparison measures
    the value of VERISHEAF's stage-orthogonal structure rather than the value
    of having a stage-aware architecture.

    Args:
        training_motifs: list of Motif.
        training_labels: per-motif binary label (0 normal, 1 attacked).
        training_stages: optional per-motif BenalohStage (None for normal).
        seed: integer seed.
        epochs: number of training epochs.
        parameterization: one of {"diagonal", "orthogonal", "general"};
            default _nsd.PARAMETERIZATION_DEFAULT (= "orthogonal").

    Returns:
        (binary_scorer, stage_predictor) callable pair.

    Citation: manuscript Section 7 (Baseline Adapters), [Bod22].
    """
    set_global_determinism(seed)
    if parameterization not in _nsd.SUPPORTED_PARAMETERIZATIONS:
        raise ValueError(
            f"build_nsd_scorer: parameterization {parameterization!r} not in "
            f"{_nsd.SUPPORTED_PARAMETERIZATIONS}"
        )
    device = _resolve_device()
    binary_model = _train_nsd(
        training_motifs, training_labels,
        device=device, seed=seed, epochs=epochs,
        parameterization=parameterization,
    )

    stage_models: dict[str, _NSDModel] = {}
    if training_stages is not None:
        for stage_name in ("cast", "record", "count"):
            per_stage_labels: list[int] = []
            for lbl, stg in zip(training_labels, training_stages):
                if lbl == 0:
                    per_stage_labels.append(0)
                elif stg is not None and stg.value == stage_name:
                    per_stage_labels.append(1)
                else:
                    per_stage_labels.append(0)
            if sum(per_stage_labels) > 0:
                stage_seed = seed + abs(hash(stage_name)) % 1000
                stage_models[stage_name] = _train_nsd(
                    training_motifs, per_stage_labels,
                    device=device, seed=stage_seed, epochs=epochs,
                    parameterization=parameterization,
                )

    def binary_scorer(motif: Motif) -> float:
        return _score_motif_nsd(binary_model, motif, device=device)

    def stage_predictor(motif: Motif) -> str:
        if not stage_models:
            return "cast"
        scores = {
            s: _score_motif_nsd(m, motif, device=device)
            for s, m in stage_models.items()
        }
        return max(scores, key=lambda s: scores[s])

    return binary_scorer, stage_predictor


# =============================================================================
# ShadowEyes namespace
# =============================================================================


class _shadow_eyes:
    """Namespace for ShadowEyes-specific components ([Che+25] IEEE TIFS 2025).

    Published hyperparameters from [Che+25] Section VI.A (Experimental Settings):
        LR:                    1e-4
        WEIGHT_DECAY:          1e-7
        TOP_K_NEIGHBOURS:      4
        AUGMENT_PROBABILITY:   0.5  (the published "certain probability P"
                                     for time delay and amount split)
        AUGMENT_DELAY_MAX:     600  (random delay in seconds)
        GAT_HIDDEN:            64   (HybridSub GAT hidden dimension)
        UNIVERSAL_FEAT_DIM:    43   (per-account universal features, Table XII)
        RESNET_FIRST_OUT:      64   (first conv layer output channels)
        RESNET_FINAL_DIM:      2048 (ResNet-50 backbone output dimension)
        PROJ_HEAD_HIDDEN:      512  (first projection-head layer)
        PROJ_HEAD_OUT:         128  (contrastive embedding dimension)
        MLP_HIDDEN_1:          1024 (downstream MLP first hidden layer)
        MLP_HIDDEN_2:          512  (downstream MLP second hidden layer)
        CONTRASTIVE_TEMP:      0.5  (NT-Xent temperature)
        RESNET_BLOCKS:         (2, 2, 2, 2)  (compact ResNet-50 variant
                                     tractable on small motif corpora)
    """

    LR: float = 1.0e-4
    WEIGHT_DECAY: float = 1.0e-7
    TOP_K_NEIGHBOURS: int = 4
    AUGMENT_PROBABILITY: float = 0.5
    AUGMENT_DELAY_MAX: float = 600.0
    GAT_HIDDEN: int = 64
    UNIVERSAL_FEAT_DIM: int = 43
    RESNET_FIRST_OUT: int = 64
    RESNET_FINAL_DIM: int = 2048
    PROJ_HEAD_HIDDEN: int = 512
    PROJ_HEAD_OUT: int = 128
    MLP_HIDDEN_1: int = 1024
    MLP_HIDDEN_2: int = 512
    CONTRASTIVE_TEMP: float = 0.5
    RESNET_BLOCKS: tuple = (2, 2, 2, 2)
    CONTRASTIVE_PRETRAIN_EPOCH_FRACTION: float = 0.5


def _extract_universal_features(motif: Motif) -> np.ndarray:
    """Extract the 43 across-platform universal features from a Motif.

    Implements the feature set described in [Che+25] Table XII, partitioned
    into three groups: time-dimension (15 features), amount-dimension
    (20 features), and frequency-dimension (8 features). Features are
    extracted from the proposal node's perspective, treating its incident
    edges (in and out) as the analogue of the published per-account
    transaction history.

    The 43 features are returned in the canonical ordering documented in the
    paper's Table XII (feature IDs 1 through 43):

      Time (1-15): T_p, T_v, RT_p, T_v^n, T_v_min/max/avg, DT_v, STDT_v,
                   T_f, T_fmin/max/avg, DT_f, T_fstd.
      Amount (16-35): A_in/out, AA, DA, RA, A_in/out^max/min, DA_in/out,
                      RDA_in/out, STDA_in/out/all, RMA, RAA.
      Frequency (36-43): F_in/out/all, RF (F_in/F_all), RF (F_out/F_all),
                         RF (F_in/F_out), DF, RDF.

    Args:
        motif: source Motif.

    Returns:
        np.ndarray of shape (43,) containing the universal features. Missing
        statistics (e.g., zero-denominator ratios) are replaced with 0.0
        following the published convention.

    Complexity: O(|E|).
    """
    proposal_id = motif.proposal_node_id
    all_edges = (
        list(motif.cast_edges)
        + list(motif.record_edges)
        + list(motif.count_edges)
        + list(motif.auxiliary_edges)
    )

    incident_in: list = []
    incident_out: list = []
    all_incident_timestamps: list[float] = []
    all_incident_weights: list[float] = []

    for e in all_edges:
        ts = float(e.timestamp_unix)
        w = float(e.weight)
        if e.target_node_id == proposal_id:
            incident_in.append(e)
            all_incident_timestamps.append(ts)
            all_incident_weights.append(w)
        if e.source_node_id == proposal_id:
            incident_out.append(e)
            all_incident_timestamps.append(ts)
            all_incident_weights.append(w)

    F_in = float(len(incident_in))
    F_out = float(len(incident_out))
    F_all = F_in + F_out

    in_weights = np.array([float(e.weight) for e in incident_in], dtype=np.float64)
    out_weights = np.array([float(e.weight) for e in incident_out], dtype=np.float64)
    all_weights = np.array(all_incident_weights, dtype=np.float64)
    all_ts = np.array(all_incident_timestamps, dtype=np.float64)

    feats = np.zeros(43, dtype=np.float64)

    # ----- Time dimension features (IDs 1 - 15). -----
    if all_ts.size > 0:
        T_p = float(all_ts.max() - all_ts.min())
        T_v = T_p
        feats[0] = T_p
        feats[1] = T_v
        feats[2] = (T_p / T_v) if T_v > 0 else 0.0
        feats[3] = (F_all / T_v) if T_v > 0 else 0.0
        feats[4] = float(all_ts.min())
        feats[5] = float(all_ts.max())
        feats[6] = float(all_ts.mean())
        feats[7] = float(all_ts.max() - all_ts.min())
        feats[8] = float(all_ts.std()) if all_ts.size > 1 else 0.0

        sorted_ts = np.sort(all_ts)
        if sorted_ts.size > 1:
            diffs = np.diff(sorted_ts)
            feats[9] = float(diffs.mean())
            feats[10] = float(diffs.min())
            feats[11] = float(diffs.max())
            feats[12] = float(diffs.mean())
            feats[13] = float(diffs.max() - diffs.min())
            feats[14] = float(diffs.std()) if diffs.size > 1 else 0.0

    # ----- Amount dimension features (IDs 16 - 35). -----
    A_in = float(in_weights.sum()) if in_weights.size > 0 else 0.0
    A_out = float(out_weights.sum()) if out_weights.size > 0 else 0.0
    AA = A_in + A_out
    DA = abs(A_in - A_out)
    RA = (A_in / A_out) if A_out > 0 else 0.0
    feats[15] = A_in
    feats[16] = A_out
    feats[17] = AA
    feats[18] = DA
    feats[19] = RA

    A_in_max = float(in_weights.max()) if in_weights.size > 0 else 0.0
    A_in_min = float(in_weights.min()) if in_weights.size > 0 else 0.0
    A_out_max = float(out_weights.max()) if out_weights.size > 0 else 0.0
    A_out_min = float(out_weights.min()) if out_weights.size > 0 else 0.0
    feats[20] = A_in_max
    feats[21] = A_in_min
    feats[22] = A_out_max
    feats[23] = A_out_min

    DA_in = A_in_max - A_in_min
    DA_out = A_out_max - A_out_min
    feats[24] = DA_in
    feats[25] = DA_out
    feats[26] = (DA_in / A_in) if A_in > 0 else 0.0
    feats[27] = (DA_out / A_out) if A_out > 0 else 0.0

    feats[28] = float(in_weights.std()) if in_weights.size > 1 else 0.0
    feats[29] = float(out_weights.std()) if out_weights.size > 1 else 0.0
    feats[30] = float(all_weights.std()) if all_weights.size > 1 else 0.0

    if all_weights.size > 0:
        nz_min = max(float(all_weights.min()), 1e-9)
        feats[31] = float(all_weights.max()) / nz_min
        feats[32] = float(all_weights.max()) / nz_min
    feats[33] = (A_out / AA) if AA > 0 else 0.0
    feats[34] = (A_in / AA) if AA > 0 else 0.0

    # ----- Frequency dimension features (IDs 36 - 43). -----
    feats[35] = F_in
    feats[36] = F_out
    feats[37] = F_all
    feats[38] = (F_in / F_all) if F_all > 0 else 0.0
    feats[39] = (F_out / F_all) if F_all > 0 else 0.0
    feats[40] = (F_in / F_out) if F_out > 0 else 0.0
    feats[41] = abs(F_in - F_out)
    feats[42] = (abs(F_in - F_out) / F_all) if F_all > 0 else 0.0

    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    return feats.astype(np.float32)


def _sample_topk_subgraph(
    motif: Motif, K: int = 4
) -> tuple[list[str], list[tuple[int, int, float]]]:
    """Sample a Top-K subgraph around the proposal node per [Che+25] Algorithm 1.

    Performs a BFS from the proposal node, then restricts the visited
    neighbours to the Top-K by average transaction amount per edge. Returns
    the subgraph node list (proposal first) and the per-edge tuples restricted
    to this subgraph.

    Args:
        motif: source Motif.
        K: number of top neighbours to retain.

    Returns:
        (node_ids, edges) where node_ids has the proposal at index 0 and
        edges is a list of (src_idx, tgt_idx, weight) tuples indexed into
        node_ids.

    Complexity: O(|E| + |V| log |V|).
    """
    proposal_id = motif.proposal_node_id
    all_edges = (
        list(motif.cast_edges)
        + list(motif.record_edges)
        + list(motif.count_edges)
        + list(motif.auxiliary_edges)
    )

    adj: dict[str, set] = {proposal_id: set()}
    edge_weights: dict[tuple[str, str], list[float]] = {}
    for e in all_edges:
        s = e.source_node_id
        t = e.target_node_id
        adj.setdefault(s, set()).add(t)
        adj.setdefault(t, set()).add(s)
        key = (s, t) if s <= t else (t, s)
        edge_weights.setdefault(key, []).append(float(e.weight))

    visited = {proposal_id}
    queue = [proposal_id]
    neighbours: list[str] = []
    while queue:
        u = queue.pop(0)
        for v in adj.get(u, set()):
            if v not in visited:
                visited.add(v)
                if v != proposal_id:
                    neighbours.append(v)
                queue.append(v)

    def neighbour_score(n: str) -> float:
        key = (proposal_id, n) if proposal_id <= n else (n, proposal_id)
        weights = edge_weights.get(key, [])
        if not weights:
            return 0.0
        return float(np.mean(weights))

    neighbours_sorted = sorted(neighbours, key=neighbour_score, reverse=True)
    top_k = neighbours_sorted[:K]

    sub_nodes = [proposal_id] + top_k
    sub_node_idx = {n: i for i, n in enumerate(sub_nodes)}
    sub_edges: list[tuple[int, int, float]] = []
    for e in all_edges:
        if e.source_node_id in sub_node_idx and e.target_node_id in sub_node_idx:
            sub_edges.append(
                (
                    sub_node_idx[e.source_node_id],
                    sub_node_idx[e.target_node_id],
                    float(e.weight),
                )
            )

    return sub_nodes, sub_edges


def _augment_motif(motif: Motif, rng: np.random.RandomState) -> Motif:
    """Generate an augmented Motif view per [Che+25] Equations 7 and 8.

    Applies two independent augmentations each with probability
    AUGMENT_PROBABILITY: time random delay (Equation 7) and transaction
    amount split (Equation 8). Implementation handles both mutable and
    frozen Motif dataclasses by making a shallow copy of the edge lists
    and replacing per-edge fields where the dataclass permits assignment.

    Args:
        motif: source Motif.
        rng: NumPy RandomState controlling the augmentation randomness.

    Returns:
        Augmented Motif view exposing the same attribute names as the input.

    Complexity: O(|E|).
    """
    import copy

    try:
        aug = copy.copy(motif)
    except (TypeError, ValueError):
        return motif

    if rng.rand() < _shadow_eyes.AUGMENT_PROBABILITY:
        delay = float(rng.uniform(0.0, _shadow_eyes.AUGMENT_DELAY_MAX))

        def shift_edges(edges):
            out = []
            for e in edges:
                e_copy = copy.copy(e)
                try:
                    e_copy.timestamp_unix = float(e.timestamp_unix) + delay
                except (AttributeError, TypeError):
                    pass
                out.append(e_copy)
            return out

        try:
            aug.cast_edges = shift_edges(motif.cast_edges)
            aug.record_edges = shift_edges(motif.record_edges)
            aug.count_edges = shift_edges(motif.count_edges)
            aug.auxiliary_edges = shift_edges(motif.auxiliary_edges)
        except (AttributeError, TypeError):
            pass

    if rng.rand() < _shadow_eyes.AUGMENT_PROBABILITY:

        def split_edges(edges):
            out = []
            for e in edges:
                e_copy = copy.copy(e)
                try:
                    e_copy.weight = float(e.weight) * 0.5
                except (AttributeError, TypeError):
                    pass
                out.append(e_copy)
            return out

        try:
            aug.cast_edges = split_edges(aug.cast_edges)
            aug.record_edges = split_edges(aug.record_edges)
            aug.count_edges = split_edges(aug.count_edges)
            aug.auxiliary_edges = split_edges(aug.auxiliary_edges)
        except (AttributeError, TypeError):
            pass

    return aug


class _GATLayer(nn.Module):
    """Single-head Graph Attention Network layer per [Che+25] Equations 3-5.

    Computes per-edge attention coefficients alpha_uv via a LeakyReLU-activated
    linear transform of concatenated source/target features, normalises with
    softmax over each node's incoming edges, and aggregates neighbour features
    by attention-weighted sum.
    """

    def __init__(self, in_dim: int, out_dim: int, negative_slope: float = 0.2):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.negative_slope = negative_slope
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a = nn.Linear(2 * out_dim, 1, bias=False)
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a.weight)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Apply one GAT layer.

        Args:
            x: (n, in_dim) node features.
            edge_index: (2, |E|) edge indices (directed; self-loops added).

        Returns:
            (n, out_dim) updated node features.

        Complexity: O(|E| * out_dim).
        """
        n = x.shape[0]
        device = x.device
        h = self.W(x)

        if edge_index.shape[1] == 0:
            src = torch.arange(n, device=device)
            tgt = torch.arange(n, device=device)
        else:
            src = torch.cat([edge_index[0], edge_index[1], torch.arange(n, device=device)])
            tgt = torch.cat([edge_index[1], edge_index[0], torch.arange(n, device=device)])

        h_concat = torch.cat([h[src], h[tgt]], dim=-1)
        e_logits = F.leaky_relu(self.a(h_concat), negative_slope=self.negative_slope).squeeze(-1)

        max_logits = torch.full((n,), float("-inf"), device=device)
        max_logits.scatter_reduce_(0, tgt, e_logits, reduce="amax", include_self=True)
        max_logits[max_logits == float("-inf")] = 0.0
        e_shifted = e_logits - max_logits[tgt]
        e_exp = torch.exp(e_shifted)
        sum_exp = torch.zeros(n, device=device)
        sum_exp.scatter_add_(0, tgt, e_exp)
        alpha = e_exp / sum_exp[tgt].clamp(min=1e-12)

        out = torch.zeros(n, self.out_dim, device=device)
        out.index_add_(0, tgt, alpha.unsqueeze(-1) * h[src])
        return out


class _ShadowEyesResNet(nn.Module):
    """Compact 1D ResNet-50 variant backbone per [Che+25] Section V.C.

    Implements the published "ResNet-50 backbone by reconfiguring its first
    convolutional layer to process single-channel input with 64 output
    channels, using a kernel size of 3, stride of 1, and padding of 1"
    architecture, followed by ResNet-50-style bottleneck blocks with the
    channel progression 64 -> 256 -> 512 -> 1024 -> 2048 and a global
    adaptive pool producing the 2048-dimensional feature embedding consumed
    by the projection head and the downstream MLP.
    """

    class _BottleneckBlock(nn.Module):
        """Standard ResNet-50 bottleneck block: 1x1 -> 3x3 -> 1x1 + skip."""

        expansion = 4

        def __init__(self, in_channels: int, out_channels: int, stride: int):
            super().__init__()
            bottleneck_channels = out_channels // self.expansion
            self.conv1 = nn.Conv1d(in_channels, bottleneck_channels, kernel_size=1, bias=False)
            self.bn1 = nn.BatchNorm1d(bottleneck_channels)
            self.conv2 = nn.Conv1d(
                bottleneck_channels, bottleneck_channels,
                kernel_size=3, stride=stride, padding=1, bias=False,
            )
            self.bn2 = nn.BatchNorm1d(bottleneck_channels)
            self.conv3 = nn.Conv1d(bottleneck_channels, out_channels, kernel_size=1, bias=False)
            self.bn3 = nn.BatchNorm1d(out_channels)
            if stride != 1 or in_channels != out_channels:
                self.downsample = nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm1d(out_channels),
                )
            else:
                self.downsample = None

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            identity = x
            out = F.relu(self.bn1(self.conv1(x)))
            out = F.relu(self.bn2(self.conv2(out)))
            out = self.bn3(self.conv3(out))
            if self.downsample is not None:
                identity = self.downsample(x)
            return F.relu(out + identity)

    def __init__(
        self,
        input_length: int,
        first_out_channels: int = _shadow_eyes.RESNET_FIRST_OUT,
        final_dim: int = _shadow_eyes.RESNET_FINAL_DIM,
        blocks_per_stage: tuple = _shadow_eyes.RESNET_BLOCKS,
    ):
        super().__init__()
        self.input_length = input_length
        self.final_dim = final_dim

        self.first_conv = nn.Conv1d(
            in_channels=1,
            out_channels=first_out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.first_bn = nn.BatchNorm1d(first_out_channels)

        stage_out_channels = [256, 512, 1024, final_dim]
        stage_strides = [1, 2, 2, 2]
        self.stages = nn.ModuleList()
        in_ch = first_out_channels
        for stage_idx, (out_ch, stride, n_blocks) in enumerate(
            zip(stage_out_channels, stage_strides, blocks_per_stage)
        ):
            blocks = []
            for block_idx in range(n_blocks):
                blocks.append(
                    _ShadowEyesResNet._BottleneckBlock(
                        in_channels=in_ch if block_idx == 0 else out_ch,
                        out_channels=out_ch,
                        stride=stride if block_idx == 0 else 1,
                    )
                )
                in_ch = out_ch
            self.stages.append(nn.Sequential(*blocks))

        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a (B, input_length) feature batch into (B, final_dim).

        Args:
            x: (B, input_length) feature tensor.

        Returns:
            (B, final_dim) backbone embedding.

        Complexity: O(B * input_length * final_dim).
        """
        h = x.unsqueeze(1)
        h = F.relu(self.first_bn(self.first_conv(h)))
        for stage in self.stages:
            h = stage(h)
        h = self.pool(h).squeeze(-1)
        return h


class _ShadowEyesEncoder(nn.Module):
    """Full ShadowEyes encoder per [Che+25] Section V.

    Pipeline:
      (1) HybridSub: GAT structural embedding of the Top-K subgraph
          concatenated with the proposal node's 43 universal features.
      (2) ResNet-50 backbone applied to the HybridSub feature vector.
      (3) Two-layer projection head (2048 -> 512 -> 128) for contrastive
          learning.

    The forward signature returns the backbone embedding and the projection-
    head output as a pair; the contrastive pretraining loss operates on the
    projection-head output, while the downstream MLP classifier operates on
    the backbone embedding (with the projection head discarded after
    pretraining, per the standard SimCLR-family convention).
    """

    def __init__(
        self,
        graph_in_dim: int = _FEATURE_DIM_DEFAULT,
        gat_hidden: int = _shadow_eyes.GAT_HIDDEN,
        universal_dim: int = _shadow_eyes.UNIVERSAL_FEAT_DIM,
        resnet_final_dim: int = _shadow_eyes.RESNET_FINAL_DIM,
        proj_hidden: int = _shadow_eyes.PROJ_HEAD_HIDDEN,
        proj_out: int = _shadow_eyes.PROJ_HEAD_OUT,
    ):
        super().__init__()
        self.graph_in_dim = graph_in_dim
        self.gat_hidden = gat_hidden
        self.universal_dim = universal_dim
        self.resnet_final_dim = resnet_final_dim
        self.gat = _GATLayer(in_dim=graph_in_dim, out_dim=gat_hidden)
        hybrid_dim = gat_hidden + universal_dim
        self.resnet = _ShadowEyesResNet(input_length=hybrid_dim, final_dim=resnet_final_dim)
        self.projection_head = nn.Sequential(
            nn.Linear(resnet_final_dim, proj_hidden, bias=False),
            nn.BatchNorm1d(proj_hidden),
            nn.ReLU(),
            nn.Linear(proj_hidden, proj_out),
        )

    def encode_motif(
        self, motif: Motif, device: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a single motif through the full pipeline.

        Args:
            motif: source Motif.
            device: torch device string.

        Returns:
            (backbone_embedding, projection_embedding) where backbone has
            shape (1, resnet_final_dim) and projection has shape (1, proj_out).

        Complexity: O(|E| + K^2 + input_length * resnet_final_dim).
        """
        sub_nodes, sub_edges = _sample_topk_subgraph(
            motif, K=_shadow_eyes.TOP_K_NEIGHBOURS
        )
        n_sub = len(sub_nodes)
        x_sub = torch.zeros(n_sub, self.graph_in_dim, device=device, dtype=torch.float32)
        full_graph = _motif_to_graph(motif, feature_dim=self.graph_in_dim)
        node_to_full = {nid: i for i, nid in enumerate(full_graph.node_ids)}
        for i, nid in enumerate(sub_nodes):
            if nid in node_to_full:
                x_sub[i] = full_graph.x[node_to_full[nid]].to(device)
        if sub_edges:
            ei_sub = torch.tensor(
                [[e[0] for e in sub_edges], [e[1] for e in sub_edges]],
                device=device, dtype=torch.long,
            )
        else:
            ei_sub = torch.zeros(2, 0, device=device, dtype=torch.long)
        h_struct = self.gat(x_sub, ei_sub)
        h_proposal = h_struct[0]

        univ_feats = _extract_universal_features(motif)
        univ_tensor = torch.from_numpy(univ_feats).to(device)
        hybrid = torch.cat([h_proposal, univ_tensor], dim=-1).unsqueeze(0)

        backbone_emb = self.resnet(hybrid)
        proj_emb = self.projection_head(backbone_emb)
        return backbone_emb, proj_emb


class _ShadowEyesClassifier(nn.Module):
    """Downstream MLP classifier per [Che+25] Section VI.A.

    Architecture: 2048 -> 1024 -> 512 -> 2 with ReLU activations. Trained on
    the labelled subset after the encoder has been pretrained via contrastive
    learning.
    """

    def __init__(
        self,
        in_dim: int = _shadow_eyes.RESNET_FINAL_DIM,
        hidden_1: int = _shadow_eyes.MLP_HIDDEN_1,
        hidden_2: int = _shadow_eyes.MLP_HIDDEN_2,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_1),
            nn.ReLU(),
            nn.Linear(hidden_1, hidden_2),
            nn.ReLU(),
            nn.Linear(hidden_2, 2),
        )

    def forward(self, backbone_emb: torch.Tensor) -> torch.Tensor:
        """Return (B, 2) class logits."""
        return self.mlp(backbone_emb)


def _nt_xent_loss(
    z1: torch.Tensor, z2: torch.Tensor, temperature: float
) -> torch.Tensor:
    """NT-Xent contrastive loss per [Che+25] Equations 10 and 11.

    Computes sim(s_i, s_j) = (s_i^T s_j) / (||s_i|| ||s_j||) for normalised
    projection-head embeddings and the InfoNCE-style cross-entropy loss

        L = - log( exp(sim(s_i, s'_i) / tau) / sum_k exp(sim(s_i, s_k) / tau) )

    averaged over the batch.

    Args:
        z1: (B, D) projection embeddings of view 1.
        z2: (B, D) projection embeddings of view 2.
        temperature: NT-Xent temperature parameter tau.

    Returns:
        Scalar loss tensor.

    Complexity: O(B^2 * D).
    """
    z1n = F.normalize(z1, p=2, dim=-1)
    z2n = F.normalize(z2, p=2, dim=-1)
    B = z1n.shape[0]
    device = z1n.device

    z = torch.cat([z1n, z2n], dim=0)
    sim = torch.matmul(z, z.T) / temperature
    mask_self = torch.eye(2 * B, dtype=torch.bool, device=device)
    sim = sim.masked_fill(mask_self, float("-inf"))

    targets = torch.cat([
        torch.arange(B, 2 * B, device=device),
        torch.arange(0, B, device=device),
    ])
    return F.cross_entropy(sim, targets)


def _train_shadow_eyes(
    training_motifs: list[Motif],
    training_labels: list[int],
    device: str,
    seed: int,
    epochs: int,
) -> tuple[_ShadowEyesEncoder, _ShadowEyesClassifier]:
    """Train ShadowEyes (contrastive pretraining + supervised fine-tuning).

    The training splits the supplied epoch budget into a contrastive
    pretraining phase (the first CONTRASTIVE_PRETRAIN_EPOCH_FRACTION of the
    budget) and a supervised fine-tuning phase (the remainder), per the
    two-stage pipeline of [Che+25] Section V.

    During pretraining the encoder learns from augmented view pairs via the
    NT-Xent contrastive loss. During fine-tuning the encoder is frozen and
    the downstream MLP classifier is trained with weighted cross-entropy on
    the proposal-node binary labels.

    Args:
        training_motifs: list of Motif.
        training_labels: per-motif binary label (0 normal, 1 attacked).
        device: torch device string.
        seed: integer seed for set_global_determinism.
        epochs: total number of training epochs (split between pretraining
            and fine-tuning).

    Returns:
        (encoder, classifier) trained ShadowEyes components.

    Complexity: O(epochs * |motifs| * resnet_cost).
    """
    set_global_determinism(seed)
    if len(training_motifs) != len(training_labels):
        raise ValueError(
            f"_train_shadow_eyes: training_motifs ({len(training_motifs)}) "
            f"and training_labels ({len(training_labels)}) length mismatch"
        )
    if not training_motifs:
        raise ValueError("_train_shadow_eyes: empty training corpus")

    encoder = _ShadowEyesEncoder().to(device)
    classifier = _ShadowEyesClassifier().to(device)

    pretrain_optimizer = Adam(
        encoder.parameters(),
        lr=_shadow_eyes.LR,
        weight_decay=_shadow_eyes.WEIGHT_DECAY,
    )
    finetune_optimizer = Adam(
        classifier.parameters(),
        lr=_shadow_eyes.LR,
        weight_decay=_shadow_eyes.WEIGHT_DECAY,
    )

    rng = np.random.RandomState(seed)

    pretrain_epochs = max(1, int(epochs * _shadow_eyes.CONTRASTIVE_PRETRAIN_EPOCH_FRACTION))
    finetune_epochs = max(1, epochs - pretrain_epochs)

    # ----- Phase 1: contrastive pretraining via NT-Xent on augmented pairs. -----
    encoder.train()
    for epoch in range(pretrain_epochs):
        perm = rng.permutation(len(training_motifs))
        epoch_loss = 0.0
        batch_size = max(2, min(8, len(training_motifs)))
        n_batches = 0
        for batch_start in range(0, len(perm), batch_size):
            batch_idx = perm[batch_start:batch_start + batch_size]
            if len(batch_idx) < 2:
                continue
            view1_proj: list[torch.Tensor] = []
            view2_proj: list[torch.Tensor] = []
            for idx in batch_idx:
                motif = training_motifs[int(idx)]
                aug1 = _augment_motif(motif, rng)
                aug2 = _augment_motif(motif, rng)
                _, p1 = encoder.encode_motif(aug1, device=device)
                _, p2 = encoder.encode_motif(aug2, device=device)
                view1_proj.append(p1)
                view2_proj.append(p2)
            z1 = torch.cat(view1_proj, dim=0)
            z2 = torch.cat(view2_proj, dim=0)
            pretrain_optimizer.zero_grad()
            loss = _nt_xent_loss(z1, z2, temperature=_shadow_eyes.CONTRASTIVE_TEMP)
            loss.backward()
            pretrain_optimizer.step()
            epoch_loss += float(loss.item())
            n_batches += 1

        if epoch % max(1, pretrain_epochs // 5) == 0 or epoch == pretrain_epochs - 1:
            print(
                f"    [ShadowEyes-pretrain seed {seed}] epoch {epoch:3d} | "
                f"loss {epoch_loss / max(n_batches, 1):.4e}",
                flush=True,
            )

    # ----- Phase 2: supervised fine-tuning with the encoder frozen. -----
    for param in encoder.parameters():
        param.requires_grad = False
    encoder.eval()

    pos = sum(1 for lbl in training_labels if lbl == 1)
    neg = len(training_labels) - pos
    class_weight = torch.tensor(
        [1.0, max(neg, 1) / max(pos, 1)], device=device, dtype=torch.float32
    )

    for epoch in range(finetune_epochs):
        classifier.train()
        perm = rng.permutation(len(training_motifs))
        epoch_loss = 0.0
        for idx in perm:
            motif = training_motifs[int(idx)]
            label = int(training_labels[int(idx)])
            with torch.no_grad():
                backbone_emb, _ = encoder.encode_motif(motif, device=device)
            finetune_optimizer.zero_grad()
            logits = classifier(backbone_emb)
            target = torch.tensor([label], device=device, dtype=torch.long)
            loss = F.cross_entropy(logits, target, weight=class_weight)
            loss.backward()
            finetune_optimizer.step()
            epoch_loss += float(loss.item())

        if epoch % max(1, finetune_epochs // 5) == 0 or epoch == finetune_epochs - 1:
            print(
                f"    [ShadowEyes-finetune seed {seed}] epoch {epoch:3d} | "
                f"loss {epoch_loss / max(len(perm), 1):.4e}",
                flush=True,
            )

    return encoder, classifier


def _score_motif_shadow_eyes(
    encoder: _ShadowEyesEncoder,
    classifier: _ShadowEyesClassifier,
    motif: Motif,
    device: str,
) -> float:
    """Score a motif: softmax probability of the malicious class.

    Args:
        encoder: trained _ShadowEyesEncoder (frozen).
        classifier: trained _ShadowEyesClassifier.
        motif: source Motif.
        device: torch device string.

    Returns:
        Float in [0, 1]; higher = more anomalous.
    """
    encoder.eval()
    classifier.eval()
    with torch.no_grad():
        backbone_emb, _ = encoder.encode_motif(motif, device=device)
        logits = classifier(backbone_emb)
        probs = F.softmax(logits, dim=-1)
        return float(probs[0, 1].item())


def build_shadow_eyes_scorer(
    training_motifs: list[Motif],
    training_labels: list[int],
    training_stages: Optional[list[Optional[BenalohStage]]] = None,
    seed: int = 42,
    epochs: int = 30,
) -> tuple[Callable[[Motif], float], Callable[[Motif], str]]:
    """Build (binary_scorer, stage_predictor) ShadowEyes adapter callables.

    Same signature pattern as build_dcl_gfd_scorer, build_knowgraph_scorer,
    and build_nsd_scorer. Trains a binary (encoder, classifier) pair on the
    full labelled corpus and, when training_stages is provided, trains three
    independent per-stage binary (encoder, classifier) pairs for the
    three-way classification reduction.

    Args:
        training_motifs: list of Motif.
        training_labels: per-motif binary label (0 normal, 1 attacked).
        training_stages: optional per-motif BenalohStage (None for normal).
        seed: integer seed; set_global_determinism is called inside the
            adapter's internal training loop.
        epochs: total training epochs (split between contrastive pretraining
            and supervised fine-tuning).

    Returns:
        (binary_scorer, stage_predictor) callable pair.

    Citation: manuscript Section 7 (Baseline Adapters), [Che+25].

    Complexity: O(epochs * (|training_motifs| + 3 * |attacked_motifs|) *
    resnet_cost).
    """
    set_global_determinism(seed)
    device = _resolve_device()
    binary_encoder, binary_classifier = _train_shadow_eyes(
        training_motifs, training_labels,
        device=device, seed=seed, epochs=epochs,
    )

    stage_models: dict[str, tuple[_ShadowEyesEncoder, _ShadowEyesClassifier]] = {}
    if training_stages is not None:
        for stage_name in ("cast", "record", "count"):
            per_stage_labels: list[int] = []
            for lbl, stg in zip(training_labels, training_stages):
                if lbl == 0:
                    per_stage_labels.append(0)
                elif stg is not None and stg.value == stage_name:
                    per_stage_labels.append(1)
                else:
                    per_stage_labels.append(0)
            if sum(per_stage_labels) > 0:
                stage_seed = seed + abs(hash(stage_name)) % 1000
                stage_models[stage_name] = _train_shadow_eyes(
                    training_motifs, per_stage_labels,
                    device=device, seed=stage_seed, epochs=epochs,
                )

    def binary_scorer(motif: Motif) -> float:
        return _score_motif_shadow_eyes(
            binary_encoder, binary_classifier, motif, device=device
        )

    def stage_predictor(motif: Motif) -> str:
        if not stage_models:
            return "cast"
        scores = {
            s: _score_motif_shadow_eyes(enc, clf, motif, device=device)
            for s, (enc, clf) in stage_models.items()
        }
        return max(scores, key=lambda s: scores[s])

    return binary_scorer, stage_predictor


# =============================================================================
# Module self-test
# =============================================================================


if __name__ == "__main__":
    print("VERISHEAF baselines.py — module checks")

    from motifs import make_attacked_motif, make_synthetic_motif

    # Test 1: _motif_to_graph produces the expected shape and indexes the proposal.
    m = make_synthetic_motif(
        "t1", "compound", num_voters=4, has_record=True, has_count=True, seed=1
    )
    g = _motif_to_graph(m, feature_dim=_FEATURE_DIM_DEFAULT)
    assert g.x.shape[1] == _FEATURE_DIM_DEFAULT
    assert g.x.shape[0] == len(g.node_ids)
    assert g.node_ids[g.proposal_idx] == m.proposal_node_id
    assert g.edge_index.shape[0] == 2
    assert g.x[g.proposal_idx, 9] == 1.0
    print(
        f"  _motif_to_graph (|V|={g.x.shape[0]}, |E|={g.edge_index.shape[1]}, "
        f"feature_dim={g.x.shape[1]}) PASS"
    )

    # Test 2: _motif_to_relational_graph attaches per-edge relation ids.
    rg, rels = _motif_to_relational_graph(m, feature_dim=_FEATURE_DIM_DEFAULT)
    assert rels.shape[0] == rg.edge_index.shape[1]
    cast_rel = _knowgraph.EDGE_TYPE_TO_REL["cast"]
    assert (rels == cast_rel).sum().item() >= len(m.cast_edges)
    print("  _motif_to_relational_graph relation tagging    PASS")

    # Test 3: build a small training corpus shared by all four baselines.
    normal = [
        make_synthetic_motif(
            f"n{i}", "compound", num_voters=4,
            has_record=True, has_count=True, seed=i,
        )
        for i in range(6)
    ]
    attacked: list[tuple[Motif, BenalohStage]] = []
    for stage in [BenalohStage.CAST, BenalohStage.RECORD, BenalohStage.COUNT]:
        for i in range(2):
            base = normal[i % len(normal)]
            attacked.append((make_attacked_motif(base, stage, seed=100 + i), stage))

    training_motifs: list[Motif] = list(normal) + [m for m, _ in attacked]
    training_labels: list[int] = [0] * len(normal) + [1] * len(attacked)
    training_stages: list[Optional[BenalohStage]] = (
        [None] * len(normal) + [s for _, s in attacked]
    )

    # Test 4: DCL-GFD scorer trains and returns sane callables.
    binary_dcl, stage_dcl = build_dcl_gfd_scorer(
        training_motifs, training_labels, training_stages,
        seed=42, epochs=3,
    )
    sc = binary_dcl(normal[0])
    assert 0.0 <= sc <= 1.0
    sp = stage_dcl(attacked[0][0])
    assert sp in {"cast", "record", "count"}
    print(f"  DCL-GFD binary score={sc:.3f}, stage={sp}   PASS")

    # Test 5: KnowGraph scorer trains and returns sane callables.
    binary_kg, stage_kg = build_knowgraph_scorer(
        training_motifs, training_labels, training_stages,
        seed=42, epochs=3,
    )
    sc = binary_kg(normal[0])
    assert 0.0 <= sc <= 1.0
    sp = stage_kg(attacked[0][0])
    assert sp in {"cast", "record", "count"}
    print(f"  KnowGraph binary score={sc:.3f}, stage={sp}   PASS")

    # Test 6: NSD scorer trains and returns sane callables (all three params).
    for param_choice in _nsd.SUPPORTED_PARAMETERIZATIONS:
        binary_nsd, stage_nsd = build_nsd_scorer(
            training_motifs, training_labels, training_stages,
            seed=42, epochs=3, parameterization=param_choice,
        )
        sc = binary_nsd(normal[0])
        assert 0.0 <= sc <= 1.0, (
            f"NSD-{param_choice}: binary score {sc} not in [0, 1]"
        )
        sp = stage_nsd(attacked[0][0])
        assert sp in {"cast", "record", "count"}, (
            f"NSD-{param_choice}: stage {sp!r} not in valid set"
        )
        print(f"  NSD-{param_choice} binary score={sc:.3f}, stage={sp}   PASS")

    # Test 7: ShadowEyes scorer trains and returns sane callables.
    binary_se, stage_se = build_shadow_eyes_scorer(
        training_motifs, training_labels, training_stages,
        seed=42, epochs=4,
    )
    sc = binary_se(normal[0])
    assert 0.0 <= sc <= 1.0, f"ShadowEyes: binary score {sc} not in [0, 1]"
    sp = stage_se(attacked[0][0])
    assert sp in {"cast", "record", "count"}, (
        f"ShadowEyes: stage {sp!r} not in valid set"
    )
    print(f"  ShadowEyes binary score={sc:.3f}, stage={sp}   PASS")

    # Test 8: identical seed reproduces identical DCL-GFD binary scores.
    binary_dcl_2, _ = build_dcl_gfd_scorer(
        training_motifs, training_labels, training_stages,
        seed=42, epochs=3,
    )
    sc1 = binary_dcl(normal[0])
    sc2 = binary_dcl_2(normal[0])
    assert abs(sc1 - sc2) < 1e-3, (
        f"DCL-GFD determinism check: |{sc1} - {sc2}| = {abs(sc1 - sc2)}"
    )
    print(
        f"  DCL-GFD determinism (|sc1-sc2|={abs(sc1 - sc2):.2e})   PASS"
    )

    # Test 9: 43 universal features extracted from a motif have correct shape.
    univ = _extract_universal_features(normal[0])
    assert univ.shape == (43,), (
        f"_extract_universal_features: shape {univ.shape} != (43,)"
    )
    assert np.all(np.isfinite(univ)), (
        f"_extract_universal_features: non-finite entries {univ}"
    )
    print(f"  _extract_universal_features (43-dim, all finite)   PASS")

    # Test 10: Top-K subgraph sampler with K=4 yields a subgraph rooted at proposal.
    sub_nodes, sub_edges = _sample_topk_subgraph(normal[0], K=4)
    assert sub_nodes[0] == normal[0].proposal_node_id, (
        f"_sample_topk_subgraph: root {sub_nodes[0]!r} != proposal"
    )
    assert len(sub_nodes) <= 5, (
        f"_sample_topk_subgraph: |sub_nodes| {len(sub_nodes)} > 1 + K=5"
    )
    print(
        f"  _sample_topk_subgraph (|sub_nodes|={len(sub_nodes)}, "
        f"|sub_edges|={len(sub_edges)})   PASS"
    )

    # Test 11: NT-Xent loss is differentiable and finite on a tiny batch.
    z1_test = torch.randn(4, 16, requires_grad=True)
    z2_test = torch.randn(4, 16, requires_grad=True)
    loss_test = _nt_xent_loss(z1_test, z2_test, temperature=0.5)
    assert torch.isfinite(loss_test), "_nt_xent_loss: produced non-finite value"
    loss_test.backward()
    assert z1_test.grad is not None, "_nt_xent_loss: did not produce gradient"
    print(f"  _nt_xent_loss (loss={float(loss_test):.4e}, gradients present)   PASS")

    # Test 12: length-mismatch raises in all four trainers.
    try:
        _train_dcl_gfd([normal[0]], [0, 1], device="cpu", seed=0, epochs=1)
        raise SystemExit("DCL-GFD length-mismatch did not raise")
    except ValueError:
        pass
    try:
        _train_knowgraph([normal[0]], [0, 1], device="cpu", seed=0, epochs=1)
        raise SystemExit("KnowGraph length-mismatch did not raise")
    except ValueError:
        pass
    try:
        _train_nsd([normal[0]], [0, 1], device="cpu", seed=0, epochs=1,
                   parameterization="diagonal")
        raise SystemExit("NSD length-mismatch did not raise")
    except ValueError:
        pass
    try:
        _train_shadow_eyes([normal[0]], [0, 1], device="cpu", seed=0, epochs=1)
        raise SystemExit("ShadowEyes length-mismatch did not raise")
    except ValueError:
        pass
    print(
        "  Length-mismatch raises (DCL-GFD, KnowGraph, NSD, ShadowEyes)   PASS"
    )

    # Test 13: bad NSD parameterization raises.
    try:
        build_nsd_scorer(
            training_motifs, training_labels, training_stages,
            seed=0, epochs=1, parameterization="unknown_param",
        )
        raise SystemExit("Bad NSD parameterization did not raise")
    except ValueError:
        pass
    print("  Bad NSD parameterization raises   PASS")

    print("\nAll baselines.py tests passed.")
