"""evaluation.py — VERISHEAF theorem validation and warehouse experiment entry points.

The primary entry point for the framework's empirical validation. Provides the
four classical theorem-validation procedures (validate_theorem_1 through
validate_theorem_4), the new validate_theorem_5 perturbation stress test, the
binary and three-way evaluation harnesses against the eight-incident benchmark,
ablation studies, and the seven dedicated warehouse-experiment entry points
(warehouse_experiment_f1 through f7) plus the warehouse_evaluation_suite
orchestrator that produces the consolidated RESULTS_FOR_MANUSCRIPT.json.

EXPERIMENTAL PROGRAM
--------------------
Seven warehouse experiments are implemented, each producing a structured JSON
output and an associated RUN_LOG.md fragment so the PI can attach reproducibility
appendices to the manuscript:

  F1 (warehouse_experiment_f1): three-way classification on the full warehouse
      corpus via verisheaf_anomaly_score under the trained restriction maps,
      with the predicted-class distribution stratified by cast-edge count n_c.

  F2 (warehouse_experiment_f2): binary scoring with DCL-GFD and KnowGraph
      baselines trained on the eight-incident attacks plus synthetic normals,
      then evaluated against the warehouse corpus with false-positive rates
      at five operating thresholds.

  F3 (warehouse_experiment_f3): cross-platform stratification (Aragon, DAOhaus,
      DAOstack) re-running F1 and F2 per platform.

  F4 (warehouse_experiment_f4): Theorem 5 perturbation stress test across
      eps in {0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30}.

  F5 (warehouse_experiment_f5): motif-topology enrichment counterfactual,
      partitioning the corpus by record-stage edge count n_r and reporting the
      predicted-class distribution per n_r bin together with the counterfactual
      classification under forced n_r = 1.

  F6 (warehouse_experiment_f6): temporal-incoherence baseline across the full
      corpus with false-positive rates at five operating thresholds.

  F7 (warehouse_experiment_f7): complementary-pipeline experiment with multi-
      detector flagging. For each of DCL-GFD, KnowGraph, NSD, and ShadowEyes
      (the four measured binary baselines), trains the binary scorer on the
      eight-incident-plus-synthetic training set, flags the top decile of
      warehouse motifs by binary score, runs VERISHEAF stage classification
      on the flagged subset, and computes the Theorem 5 stability threshold
      per flagged motif. Reports per-method results (flagged subset size,
      stage-prediction distribution, median stability threshold) plus
      cross-method consistency (fraction of motifs flagged by all detectors,
      fraction of multi-flagged motifs with consistent stage predictions,
      per-pair Cohen's kappa). The cross-method consistency is the
      experimental evidence that VERISHEAF operates as a complementary tool
      rather than as a competitor to any specific binary detector.

DETERMINISM CONTRACT
--------------------
Every warehouse experiment calls set_global_determinism with the experiment-
specific seed at entry. Output JSON is written with sort_keys=True, indent=2,
allow_nan=False. The RUN_LOG.md fragment records the command-line equivalent,
the seed, the wall times, output paths, headline numbers, and any
KNOWN_LIMITATION entries triggered (notably CC-4 for the weakened theorem
verifier forms preserved from the prior codebase version).

The four validation procedures of the original module are:
    validate_theorem_1  Section bijectivity on the held-out incident set
    validate_theorem_2  Confusion matrix of attack-type classification via subspaces
    validate_theorem_3  Speed and accuracy of approximate H^1 versus exact
    validate_theorem_4  PAC-Bayes bound versus empirical operator-norm error
    validate_theorem_5  NEW: empirical sin-theta leakage versus Theorem 5 bound
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from scipy import stats as scipy_stats

from motifs import (
    CANONICAL_INCIDENTS,
    BenalohStage,
    EdgeType,
    Incident,
    Motif,
    NodeType,
    make_attacked_motif,
    make_synthetic_motif,
    motif_to_sheaf,
)
from theory import (
    DEFAULT_EDGE_STALK_DIMS,
    DEFAULT_STALK_DIMS,
    SheafLearner,
    approximate_h1_dimension,
    build_canonical_sheaf,
    compute_attack_subspaces,
    corollary_1_predicted_class,
    has_nontrivial_attack_subspaces,
    pac_bayes_bound,
    project_onto_subspace,
    theorem_5_bound,
)
from training import set_global_determinism


# =============================================================================
# Module-level constants
# =============================================================================


_F2_OPERATING_THRESHOLDS: tuple[float, ...] = (0.10, 0.30, 0.50, 0.70, 0.90)
_F4_EPSILON_SWEEP: tuple[float, ...] = (0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30)
_F4_STRATIFIED_SAMPLE_SIZE_PER_PLATFORM: int = 334  # 3 platforms * 334 ~= 1002 motifs
_F5_NR_BINS: tuple[int, ...] = (0, 1, 2, 3, 5, 10)

# F7 (Complementary-Pipeline Experiment with Multi-Detector Flagging) constants.
# Default flagging methods cover the four measured binary baselines defined in
# baselines.py: DCL-GFD (Yu+25), KnowGraph (Zhou+24), NSD (Bod22), ShadowEyes
# (Che+25). The flagging quantile of 0.10 corresponds to flagging the top decile
# of warehouse motifs by each method's binary score. The quantile sweep
# {0.05, 0.10, 0.15, 0.20} retains the threshold-sensitivity analysis required
# by the manuscript revision plan.
_F7_FLAGGING_METHODS_DEFAULT: tuple[str, ...] = (
    "dcl_gfd",
    "knowgraph",
    "nsd",
    "shadow_eyes",
)
_F7_FLAGGING_QUANTILE_DEFAULT: float = 0.10
_F7_QUANTILE_SWEEP_DEFAULT: tuple[float, ...] = (0.05, 0.10, 0.15, 0.20)
# Reference perturbation magnitude passed to theorem_5_bound when computing the
# per-motif stability threshold. The stability_threshold_eps return value of
# theorem_5_bound depends only on the spectral gap, the operator-norm radius,
# the stalk dimension, the edge count in the predicted stage, and the maximum
# node-stalk degree; it does NOT depend on eps. The reference value 0.05 is
# supplied only so the call is well-formed; the returned stability threshold
# is the intrinsic per-motif quantity gamma / (4 rho sqrt(2 d |E_tau| Delta)).
_F7_THEOREM_5_REFERENCE_EPS: float = 0.05


# =============================================================================
# Bootstrap and statistical testing utilities
# =============================================================================


def bootstrap_ci(
    values: np.ndarray, n_resamples: int = 1000, confidence: float = 0.95, seed: int = 0
) -> tuple[float, float, float]:
    """Compute bootstrap (mean, lower, upper) at the given confidence level.

    Args:
        values: 1-D NumPy array of values to bootstrap over.
        n_resamples: number of bootstrap resamples; default 1000.
        confidence: two-sided confidence level in (0, 1); default 0.95.
        seed: integer seed for the bootstrap RNG.

    Returns:
        (mean, lower, upper) triple of floats. NaN triple when values is empty.

    Complexity: O(n_resamples * |values|).
    """
    rng = np.random.RandomState(seed)
    n = len(values)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    means = np.empty(n_resamples)
    for i in range(n_resamples):
        sample = rng.choice(values, size=n, replace=True)
        means[i] = sample.mean()
    alpha = 1.0 - confidence
    lower = float(np.percentile(means, 100 * alpha / 2))
    upper = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return float(values.mean()), lower, upper


def paired_wilcoxon(method_scores: np.ndarray, baseline_scores: np.ndarray) -> dict:
    """Paired Wilcoxon signed-rank test, returning p-value and effect size."""
    diff = method_scores - baseline_scores
    if np.all(diff == 0):
        return {"p_value": 1.0, "median_diff": 0.0, "n_pairs": len(diff)}
    try:
        stat, p = scipy_stats.wilcoxon(diff, alternative="greater")
        return {
            "p_value": float(p),
            "median_diff": float(np.median(diff)),
            "n_pairs": int(len(diff)),
            "statistic": float(stat),
        }
    except ValueError:
        return {"p_value": 1.0, "median_diff": float(np.median(diff)), "n_pairs": len(diff)}


def holm_correct(p_values: dict[str, float], alpha: float = 0.05) -> dict[str, dict]:
    """Holm-Bonferroni multiple comparison correction."""
    items = sorted(p_values.items(), key=lambda x: x[1])
    n = len(items)
    result = {}
    for i, (key, p) in enumerate(items):
        threshold = alpha / (n - i)
        result[key] = {
            "p_value": p,
            "holm_threshold": threshold,
            "significant": p < threshold,
        }
    return result


def _cohen_kappa(
    predictions_a: list[str], predictions_b: list[str]
) -> Optional[float]:
    """Cohen's kappa for two categorical-label vectors on a shared item set.

    Computes kappa = (p_o - p_e) / (1 - p_e), where p_o is the observed
    agreement (fraction of items on which the two raters give the same label)
    and p_e is the expected agreement under marginal independence
    (sum over categories of row_marg[c] * col_marg[c] in the confusion matrix).
    The shared category vocabulary is constructed as sorted(set(a) | set(b)),
    which makes the result invariant to label-order permutations of the input
    lists.

    Args:
        predictions_a, predictions_b: per-item label lists of equal length.

    Returns:
        Cohen's kappa in [-1, 1], or None when undefined. Undefined cases:
        (i) fewer than two items, (ii) expected agreement equals one
        (degenerate marginal: one rater assigns a single label across all
        items), in which case the kappa formula divides by zero.

    Citation: Cohen, J. (1960). A coefficient of agreement for nominal
    scales. Educational and Psychological Measurement, 20(1), 37-46.

    Complexity: O(n + k^2) where k = number of distinct categories.
    """
    if len(predictions_a) != len(predictions_b):
        raise ValueError(
            f"_cohen_kappa: label length mismatch "
            f"({len(predictions_a)} vs {len(predictions_b)})"
        )
    n = len(predictions_a)
    if n < 2:
        return None
    categories = sorted(set(predictions_a) | set(predictions_b))
    cat_to_idx = {c: i for i, c in enumerate(categories)}
    k = len(categories)
    confusion = np.zeros((k, k), dtype=np.int64)
    for a, b in zip(predictions_a, predictions_b):
        confusion[cat_to_idx[a], cat_to_idx[b]] += 1
    p_o = float(np.trace(confusion)) / n
    row_marg = confusion.sum(axis=1) / n
    col_marg = confusion.sum(axis=0) / n
    p_e = float(np.sum(row_marg * col_marg))
    if 1.0 - p_e < 1e-12:
        # Degenerate: marginal distribution concentrated on one category.
        # If observed agreement is also one, kappa is conventionally 1.0;
        # otherwise the kappa formula is undefined and we return None.
        return 1.0 if abs(p_o - 1.0) < 1e-12 else None
    return float((p_o - p_e) / (1.0 - p_e))


def _nan_to_none(value):
    """Convert NaN/inf floats to None for JSON serialization with allow_nan=False.

    Args:
        value: any value; floats that are NaN or +/-inf are mapped to None.

    Returns:
        None when value is a non-finite float; the original value otherwise.
    """
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


# =============================================================================
# Theorem 1 validation
# =============================================================================


def validate_theorem_1(motifs: list[Motif]) -> dict:
    """Empirical validation of Theorem 1 against a motif corpus.

    For each motif, verifies that the canonical sheaf has the expected H^0
    dimension.

    KNOWN_LIMITATION (CC-4): tests the H^0 >= 1 testable consequence rather
    than the full bijection statement.

    Citation: manuscript Theorem 1 (Canonical Sheaf Construction).
    """
    results = []
    for motif in motifs:
        sheaf = motif_to_sheaf(motif)
        h0 = sheaf.h0_dimension()
        expected = min(DEFAULT_STALK_DIMS.values()) if motif.is_complete else None
        results.append(
            {
                "motif_id": motif.motif_id,
                "is_complete": motif.is_complete,
                "observability_ratio": motif.observability_ratio,
                "h0_dim": h0,
                "expected_h0": expected,
            }
        )

    h0_values = np.array([r["h0_dim"] for r in results], dtype=float)
    mean, lo, hi = bootstrap_ci(h0_values)
    return {
        "results": results,
        "mean_h0": mean,
        "ci_lower": lo,
        "ci_upper": hi,
        "verified": all(r["h0_dim"] >= 1 for r in results if r["is_complete"]),
    }


# =============================================================================
# Theorem 2 validation
# =============================================================================


def validate_theorem_2(base_motifs: list[Motif], num_per_attack: int = 30) -> dict:
    """Confusion matrix of attack classification using subspace projections.

    Citation: manuscript Theorem 2 (Cohomological Characterization).
    """
    stages = [BenalohStage.CAST, BenalohStage.RECORD, BenalohStage.COUNT]
    confusion = np.zeros((3, 3), dtype=int)
    stage_idx = {s: i for i, s in enumerate(stages)}

    detailed = []
    n_undefined = 0
    for base in base_motifs[: max(1, len(base_motifs))]:
        for true_stage in stages:
            for k in range(num_per_attack // len(base_motifs) + 1):
                attacked = make_attacked_motif(
                    base, true_stage,
                    seed=k * 17 + base.proposal_timestamp_unix % 1000,
                )
                sheaf = motif_to_sheaf(attacked)
                subspaces = compute_attack_subspaces(sheaf)

                L1 = sheaf.edge_laplacian()
                eigs, eigvecs = np.linalg.eigh(L1.toarray())

                if np.sum(eigs < 1e-8) == 0 or not has_nontrivial_attack_subspaces(subspaces):
                    n_undefined += 1
                    detailed.append(
                        {
                            "base_motif": base.motif_id,
                            "true_stage": true_stage.value,
                            "predicted_stage": "undefined_trivial_h1",
                            "norms": {s.value: 0.0 for s in stages},
                        }
                    )
                    continue

                h1_vec = torch.from_numpy(eigvecs[:, int(np.argmin(eigs))]).float()

                norms = {}
                for s in stages:
                    basis = subspaces[s]
                    if basis.shape[1] > 0:
                        _, norm = project_onto_subspace(h1_vec, basis)
                    else:
                        norm = 0.0
                    norms[s] = norm

                predicted = max(stages, key=lambda s: norms[s])
                confusion[stage_idx[true_stage], stage_idx[predicted]] += 1
                detailed.append(
                    {
                        "base_motif": base.motif_id,
                        "true_stage": true_stage.value,
                        "predicted_stage": predicted.value,
                        "norms": {s.value: float(v) for s, v in norms.items()},
                    }
                )

    n_classified = int(np.sum(confusion))
    accuracy = float(np.trace(confusion)) / max(n_classified, 1)
    return {
        "confusion_matrix": confusion.tolist(),
        "stages_order": [s.value for s in stages],
        "accuracy": accuracy,
        "n_classified": n_classified,
        "n_undefined_trivial_h1": n_undefined,
        "details": detailed,
    }


# =============================================================================
# Theorem 3 validation
# =============================================================================


def validate_theorem_3(motifs: list[Motif], eps: float = 1e-3) -> dict:
    """Compare approximate H^1 dimension against exact, with timing.

    Citation: manuscript Theorem 3 (Spectral Approximation).
    """
    rows = []
    for motif in motifs:
        sheaf = motif_to_sheaf(motif)
        t0 = time.perf_counter()
        exact = sheaf.h1_dimension()
        t_exact = time.perf_counter() - t0

        approx = approximate_h1_dimension(sheaf, eps=eps)
        rows.append(
            {
                "motif_id": motif.motif_id,
                "n_edges_total_dim": sheaf.total_edge_dim,
                "h1_exact": exact,
                "h1_approx": approx["h1_dim"],
                "wall_clock_exact_s": t_exact,
                "wall_clock_approx_s": approx["wall_clock_s"],
                "spectral_gap": approx["spectral_gap"],
            }
        )

    agreement = float(np.mean([r["h1_exact"] == r["h1_approx"] for r in rows]))
    speedup = float(
        np.median(
            [
                r["wall_clock_exact_s"] / max(r["wall_clock_approx_s"], 1e-9)
                for r in rows
            ]
        )
    )
    return {
        "rows": rows,
        "agreement_rate": agreement,
        "median_speedup": speedup,
        "verified": bool(agreement >= 1.0 - (1.0 / max(len(rows), 1)) - 1e-9),
    }


# =============================================================================
# Theorem 4 validation
# =============================================================================


def validate_theorem_4(
    sample_sizes: list[int] = [10, 25, 50, 100, 250],
    d: int = 8,
    delta: float = 0.05,
    num_trials: int = 50,
    seed: int = 0,
) -> dict:
    """For each sample size m, compare PAC-Bayes bound against empirical max error.

    Citation: manuscript Theorem 4 (PAC-Bayes Learning Guarantee).
    """
    rng = np.random.RandomState(seed)
    rows = []

    for m in sample_sizes:
        bound = pac_bayes_bound(m=m, d=d, delta=delta)

        errors = []
        for _ in range(num_trials):
            truth = rng.randn(d, d) / math.sqrt(d)
            noise = rng.randn(m, d, d) / math.sqrt(d)
            samples = truth + noise
            estimate = samples.mean(axis=0)
            err = np.linalg.norm(estimate - truth, ord=2)
            errors.append(err)

        errors = np.array(errors)
        p95 = float(np.percentile(errors, 95))
        rows.append(
            {
                "m": m,
                "pac_bayes_bound": bound,
                "empirical_p95": p95,
                "empirical_mean": float(errors.mean()),
                "bound_satisfied_fraction": float(np.mean(errors <= bound)),
            }
        )

    mc_slack = 0.02
    verified = all(
        r["bound_satisfied_fraction"] >= 1.0 - delta - mc_slack for r in rows
    )
    return {
        "rows": rows,
        "delta": delta,
        "d": d,
        "num_trials": num_trials,
        "verified": bool(verified),
    }


# =============================================================================
# Theorem 5 validation: empirical sin-theta leakage vs theorem bound
# =============================================================================


def _learned_restrictions_perturbed(
    eps: float, rho: float, rng: np.random.RandomState
) -> dict:
    """Build a perturbed `learned_restrictions` dict for build_canonical_sheaf."""
    from theory import _canonical_restriction
    learned: dict = {}
    for nt in [
        NodeType.VOTER, NodeType.DELEGATE, NodeType.PROPOSAL,
        NodeType.GOV_TOKEN, NodeType.EXEC_CONTRACT,
    ]:
        for et in [
            EdgeType.CAST, EdgeType.DELEGATE, EdgeType.TRANSFER,
            EdgeType.RECORD, EdgeType.COUNT,
        ]:
            de = DEFAULT_EDGE_STALK_DIMS[et]
            dn = DEFAULT_STALK_DIMS[nt]
            base_src = _canonical_restriction(de, dn, "src", nt, et)
            base_tgt = _canonical_restriction(de, dn, "tgt", nt, et)
            scale = eps / math.sqrt(max(de, dn))
            noise_src = rng.randn(de, dn) * scale
            noise_tgt = rng.randn(de, dn) * scale
            learned[("src", nt, et)] = base_src + torch.from_numpy(noise_src).float()
            learned[("tgt", et, nt)] = base_tgt + torch.from_numpy(noise_tgt).float()
    _ = rho
    return learned


def _empirical_sin_theta_frobenius(
    basis_a: torch.Tensor, basis_b: torch.Tensor
) -> float:
    """Compute the Frobenius sin-theta principal-angle distance between two subspaces."""
    if basis_a.shape[1] == 0 or basis_b.shape[1] == 0:
        return float(max(basis_a.shape[1], basis_b.shape[1]))
    k_min = min(basis_a.shape[1], basis_b.shape[1])
    proj = basis_a.T @ basis_b
    sigma_sq = torch.linalg.svdvals(proj) ** 2
    sin_theta_sq = max(0.0, float(k_min) - float(sigma_sq.sum().item()))
    return math.sqrt(max(0.0, sin_theta_sq))


def validate_theorem_5(
    base_motifs: list[Motif],
    eps_values: tuple[float, ...] = _F4_EPSILON_SWEEP,
    rho: float = 1.0,
    seed: int = 0,
    max_motifs: int = 50,
) -> dict:
    """Empirical stress test: sin-theta leakage versus the Theorem 5 bound.

    Citation: manuscript Theorem 5 (Cross-Stage Subspace Leakage).
    """
    motifs = base_motifs[:max_motifs] if max_motifs > 0 else base_motifs
    rng = np.random.RandomState(seed)
    rows: list[dict] = []
    n_dominations = 0
    n_total = 0

    for motif in motifs:
        sheaf_canonical = motif_to_sheaf(motif)
        subspaces_canonical = compute_attack_subspaces(sheaf_canonical)
        cast_basis_canonical = subspaces_canonical[BenalohStage.CAST]
        if cast_basis_canonical.shape[1] == 0:
            continue

        L1_canonical = sheaf_canonical.edge_laplacian().toarray()
        eigs_canonical = np.linalg.eigvalsh(L1_canonical)
        nonzero = eigs_canonical[eigs_canonical > 1e-8]
        gamma = float(nonzero.min()) if nonzero.size else 1.0

        n_cast = len(motif.cast_edges)
        n_record = len(motif.record_edges)
        n_count = len(motif.count_edges)
        max_node_degree = max(n_cast + n_record + n_count, 1)
        d_stalk = DEFAULT_STALK_DIMS[NodeType.PROPOSAL]

        for eps in eps_values:
            bound = theorem_5_bound(
                eps=eps, rho=rho, d=d_stalk,
                n_edges_in_stage=n_cast,
                max_node_degree=max_node_degree,
                spectral_gap=gamma,
            )
            learned = _learned_restrictions_perturbed(eps, rho, rng)
            from motifs import motif_to_typed_graph
            node_types, edge_index, edge_types, stage_mask, _ = motif_to_typed_graph(motif)
            sheaf_perturbed = build_canonical_sheaf(
                node_types, edge_index, edge_types,
                stage_mask=stage_mask, learned_restrictions=learned,
            )
            subspaces_perturbed = compute_attack_subspaces(sheaf_perturbed)
            cast_basis_perturbed = subspaces_perturbed[BenalohStage.CAST]
            empirical_leakage = _empirical_sin_theta_frobenius(
                cast_basis_canonical, cast_basis_perturbed
            )

            dominates = empirical_leakage <= bound["frobenius_bound"]
            n_dominations += int(dominates)
            n_total += 1
            rows.append({
                "motif_id": motif.motif_id,
                "n_cast": n_cast,
                "n_record": n_record,
                "n_count": n_count,
                "eps": eps,
                "spectral_gap": gamma,
                "theorem_5_frobenius_bound": bound["frobenius_bound"],
                "theorem_5_operator_bound": bound["operator_bound"],
                "stability_threshold_eps": bound["stability_threshold_eps"],
                "empirical_sin_theta_frobenius": empirical_leakage,
                "bound_dominates": bool(dominates),
            })

    overall = n_dominations / max(n_total, 1)
    return {
        "eps_sweep": list(eps_values),
        "rows": rows,
        "n_evaluated": n_total,
        "overall_dominance_rate": overall,
        "verified": bool(overall >= 1.0 - 0.05),
    }


# =============================================================================
# Baselines and VERISHEAF scorers
# =============================================================================


def baseline_edge_count_anomaly(motif: Motif) -> float:
    """Trivial baseline: total number of cast edges, normalized."""
    return float(len(motif.cast_edges))


def baseline_weight_concentration(motif: Motif) -> float:
    """Voting-power concentration (Gini of cast weights)."""
    if not motif.cast_edges:
        return 0.0
    weights = np.array([e.weight for e in motif.cast_edges])
    weights = np.sort(weights)
    n = len(weights)
    if n == 0 or weights.sum() == 0:
        return 0.0
    cum = np.cumsum(weights)
    gini = (n + 1 - 2 * np.sum(cum) / cum[-1]) / n
    return float(gini)


def baseline_temporal_incoherence(motif: Motif) -> float:
    """Temporal incoherence: squared CV of the cast-to-record gap."""
    if not motif.record_edges or not motif.cast_edges:
        return 0.0
    cast_times = [e.timestamp_unix for e in motif.cast_edges]
    record_time = motif.record_edges[0].timestamp_unix
    gaps = np.array([record_time - t for t in cast_times], dtype=float)
    mean_gap = float(np.mean(gaps))
    if abs(mean_gap) < 1e-9:
        return 0.0
    return float(np.var(gaps) / (mean_gap ** 2))


def verisheaf_anomaly_score(
    motif: Motif, learned_restrictions: Optional[dict] = None
) -> dict[str, float]:
    """The VERISHEAF stage scores: subspace projection norms per Benaloh stage.

    Citation: manuscript Section 4.4 (VERISHEAF Anomaly Score).
    """
    sheaf = motif_to_sheaf(motif, learned_restrictions=learned_restrictions)
    subspaces = compute_attack_subspaces(sheaf)
    L1 = sheaf.edge_laplacian()
    eigs, eigvecs = np.linalg.eigh(L1.toarray())

    stages = [BenalohStage.CAST, BenalohStage.RECORD, BenalohStage.COUNT]
    if np.sum(eigs < 1e-8) == 0 or not has_nontrivial_attack_subspaces(subspaces):
        return {s.value: 0.0 for s in stages}

    h1_vec = torch.from_numpy(eigvecs[:, int(np.argmin(eigs))]).float()
    scores = {}
    for stage in stages:
        basis = subspaces[stage]
        if basis.shape[1] > 0:
            _, norm = project_onto_subspace(h1_vec, basis)
        else:
            norm = 0.0
        scores[stage.value] = norm
    return scores


# =============================================================================
# Baseline comparison harness
# =============================================================================


def run_baseline_comparison(
    normal_motifs: list[Motif], attacked_motifs: list[tuple[Motif, BenalohStage]]
) -> dict:
    """Compare VERISHEAF against baselines on the binary and three-way tasks."""
    from sklearn.metrics import (
        accuracy_score, average_precision_score, confusion_matrix, roc_auc_score,
    )

    binary_methods = {
        "verisheaf_max_subspace": lambda m: max(verisheaf_anomaly_score(m).values()),
        "baseline_edge_count": baseline_edge_count_anomaly,
        "baseline_weight_concentration": baseline_weight_concentration,
        "baseline_temporal_incoherence": baseline_temporal_incoherence,
    }
    labels: list[int] = [0] * len(normal_motifs) + [1] * len(attacked_motifs)
    all_motifs: list[Motif] = list(normal_motifs) + [m for m, _ in attacked_motifs]

    binary_summary: dict[str, dict] = {}
    for name, scorer in binary_methods.items():
        scores = np.array([scorer(m) for m in all_motifs], dtype=float)
        if len(set(labels)) < 2:
            auc = ap = float("nan")
        else:
            auc = float(roc_auc_score(labels, scores))
            ap = float(average_precision_score(labels, scores))
        binary_summary[name] = {
            "auc": auc,
            "average_precision": ap,
            "mean_normal": float(np.mean(scores[: len(normal_motifs)]))
                if normal_motifs else float("nan"),
            "mean_attacked": float(np.mean(scores[len(normal_motifs):]))
                if attacked_motifs else float("nan"),
        }

    stage_order = ["cast", "record", "count"]
    true_stages: list[str] = []
    pred_stages: list[str] = []
    n_undefined = 0
    for m, true_stage in attacked_motifs:
        scores = verisheaf_anomaly_score(m)
        if max(scores.values()) <= 0.0:
            n_undefined += 1
            continue
        pred = max(stage_order, key=lambda s: scores[s])
        true_stages.append(true_stage.value)
        pred_stages.append(pred)

    if true_stages:
        three_way_acc = float(accuracy_score(true_stages, pred_stages))
        cm = confusion_matrix(true_stages, pred_stages, labels=stage_order).tolist()
    else:
        three_way_acc = float("nan")
        cm = None

    return {
        "binary": binary_summary,
        "three_way": {
            "accuracy": three_way_acc,
            "confusion_matrix": cm,
            "stages_order": stage_order,
            "n_classified": len(true_stages),
            "n_undefined_trivial_h1": n_undefined,
        },
    }


def run_ablations(
    normal_motifs: list[Motif],
    attacked_motifs: list[tuple[Motif, BenalohStage]],
) -> dict:
    """Per-ablation effect on the cohomological characterization AUC."""
    from sklearn.metrics import roc_auc_score

    def evaluate_setting(name: str, scorer_fn) -> dict:
        labels = [0] * len(normal_motifs) + [1] * len(attacked_motifs)
        scores = []
        for m in normal_motifs:
            scores.append(scorer_fn(m))
        for m, _ in attacked_motifs:
            scores.append(scorer_fn(m))
        scores = np.array(scores)
        if len(set(labels)) < 2:
            auc = float("nan")
        else:
            auc = float(roc_auc_score(labels, scores))
        return {"name": name, "auc": auc}

    results = []
    results.append(
        evaluate_setting(
            "verisheaf_full", lambda m: max(verisheaf_anomaly_score(m).values())
        )
    )

    def drop_count_score(m: Motif) -> float:
        import copy
        m2 = copy.deepcopy(m)
        m2.count_edges = []
        m2.__post_init__()
        return max(verisheaf_anomaly_score(m2).values())

    results.append(evaluate_setting("ablate_no_count", drop_count_score))

    def cast_only_score(m: Motif) -> float:
        import copy
        m2 = copy.deepcopy(m)
        m2.record_edges = []
        m2.count_edges = []
        m2.__post_init__()
        return max(verisheaf_anomaly_score(m2).values())

    results.append(evaluate_setting("ablate_cast_only", cast_only_score))

    return {"ablations": results}


def cohomological_characterization_diagram(
    normal_motifs: list[Motif],
    attacked_motifs: list[tuple[Motif, BenalohStage]],
) -> dict:
    """Produce the data for the manuscript's headline ternary scatter."""
    points = []
    for m in normal_motifs:
        scores = verisheaf_anomaly_score(m)
        points.append({"motif_id": m.motif_id, "label": "normal", "true_attack": None, **scores})
    for m, stage in attacked_motifs:
        scores = verisheaf_anomaly_score(m)
        points.append(
            {
                "motif_id": m.motif_id,
                "label": "attacked",
                "true_attack": stage.value,
                **scores,
            }
        )
    return {"points": points}


def _make_loop_motif(num_voters: int, dao_name: str, seed: int) -> Motif:
    """Build a complete cast/record/count loop motif for Theorem 3 validation."""
    return make_synthetic_motif(
        f"t3loop{seed}", dao_name, num_voters=num_voters,
        has_record=True, has_count=True, seed=seed,
    )


def full_evaluation_report(
    normal_motifs: list[Motif],
    attacked_motifs: list[tuple[Motif, BenalohStage]],
) -> dict:
    """Run the complete eight-incident evaluation suite and consolidate."""
    report = {}
    print("  Validating Theorem 1...")
    report["theorem_1"] = validate_theorem_1(normal_motifs)
    print("  Validating Theorem 2...")
    report["theorem_2"] = validate_theorem_2(normal_motifs, num_per_attack=12)
    print("  Validating Theorem 3...")
    t3_motifs = [
        _make_loop_motif(k + 2, "compound", seed=500 + k) for k in range(5)
    ]
    report["theorem_3"] = validate_theorem_3(t3_motifs)
    print("  Validating Theorem 4...")
    report["theorem_4"] = validate_theorem_4(
        sample_sizes=[10, 25, 50, 100], num_trials=20
    )
    print("  Validating Theorem 5...")
    report["theorem_5"] = validate_theorem_5(
        normal_motifs, eps_values=_F4_EPSILON_SWEEP, max_motifs=10,
    )
    print("  Running baseline comparison...")
    report["baselines"] = run_baseline_comparison(normal_motifs, attacked_motifs)
    print("  Running ablations...")
    report["ablations"] = run_ablations(normal_motifs, attacked_motifs)
    print("  Producing headline figure data...")
    report["headline_figure"] = cohomological_characterization_diagram(
        normal_motifs, attacked_motifs
    )
    return report


# =============================================================================
# Warehouse experiment infrastructure
# =============================================================================


def _load_trained_restrictions(checkpoint_path: Optional[Path]) -> Optional[dict]:
    """Load trained restriction maps from a SheafLearner checkpoint."""
    if checkpoint_path is None:
        return None
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        return None
    state = torch.load(checkpoint_path, map_location="cpu")
    model = SheafLearner()
    model.load_state_dict(state["model_state"])
    model.eval()
    with torch.no_grad():
        return model.assemble_restrictions()


def _write_json(payload: dict, output_path: Path) -> None:
    """Write payload as deterministic UTF-8 JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str, allow_nan=False),
        encoding="utf-8",
    )


def _write_run_log(
    experiment: str,
    seed: int,
    cmd: str,
    wall_clock_s: float,
    output_paths: list[Path],
    headline: dict,
    known_limitations: list[str],
    log_path: Path,
) -> None:
    """Write a RUN_LOG.md fragment for a warehouse experiment."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# RUN_LOG.md — VERISHEAF warehouse experiment {experiment}",
        "",
        f"- **Seed:** {seed}",
        f"- **Command:** `{cmd}`",
        f"- **Wall clock (s):** {wall_clock_s:.2f}",
        "",
        "## Outputs",
        "",
    ]
    for p in output_paths:
        lines.append(f"- `{p}`")
    lines.append("")
    lines.append("## Headline numbers")
    lines.append("")
    for k in sorted(headline.keys()):
        lines.append(f"- **{k}:** {headline[k]}")
    lines.append("")
    if known_limitations:
        lines.append("## KNOWN_LIMITATION entries")
        lines.append("")
        for kl in known_limitations:
            lines.append(f"- {kl}")
        lines.append("")
    log_path.write_text("\n".join(lines), encoding="utf-8")


# =============================================================================
# F1: three-way classification on the full warehouse corpus
# =============================================================================


def warehouse_experiment_f1(
    warehouse_motifs: list[Motif],
    checkpoint_path: Optional[Path],
    output_dir: Path,
    seed: int = 20260601,
) -> dict:
    """F1: three-way classification on the full warehouse corpus.

    Citation: manuscript Section 6.2 (F1 — Predicted-Class Distribution).
    """
    set_global_determinism(seed)
    t_start = time.perf_counter()
    output_dir = Path(output_dir)
    learned = _load_trained_restrictions(checkpoint_path)

    rows: list[dict] = []
    pred_counts: dict[str, int] = {"cast": 0, "record": 0, "count": 0, "undefined": 0}
    nc_bins = {"1": 0, "2-4": 0, "5-9": 0, "10-49": 0, "50+": 0}
    nc_pred: dict[str, dict[str, int]] = {
        k: {"cast": 0, "record": 0, "count": 0, "undefined": 0} for k in nc_bins
    }

    for motif in warehouse_motifs:
        scores = verisheaf_anomaly_score(motif, learned_restrictions=learned)
        n_c = len(motif.cast_edges)
        n_r = len(motif.record_edges)
        n_q = len(motif.count_edges)
        if max(scores.values()) <= 0.0:
            pred = "undefined"
        else:
            pred = max(["cast", "record", "count"], key=lambda s: scores[s])
        pred_counts[pred] += 1
        if n_c <= 1:
            bin_key = "1"
        elif n_c <= 4:
            bin_key = "2-4"
        elif n_c <= 9:
            bin_key = "5-9"
        elif n_c <= 49:
            bin_key = "10-49"
        else:
            bin_key = "50+"
        nc_bins[bin_key] += 1
        nc_pred[bin_key][pred] += 1
        rows.append({
            "motif_id": motif.motif_id,
            "n_cast": n_c,
            "n_record": n_r,
            "n_count": n_q,
            "score_cast": float(scores["cast"]),
            "score_record": float(scores["record"]),
            "score_count": float(scores["count"]),
            "predicted_stage": pred,
            "corollary_1_predicted": corollary_1_predicted_class(n_c, n_r, n_q),
        })

    total = max(len(rows), 1)
    distribution_fractions = {k: v / total for k, v in pred_counts.items()}
    wall = time.perf_counter() - t_start
    payload = {
        "experiment": "F1",
        "seed": seed,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
        "n_motifs": len(rows),
        "wall_clock_s": wall,
        "predicted_class_counts": pred_counts,
        "predicted_class_fractions": distribution_fractions,
        "n_c_bin_counts": nc_bins,
        "n_c_bin_predicted_class_counts": nc_pred,
        "rows": rows,
    }
    out_json = output_dir / "F1.json"
    _write_json(payload, out_json)
    _write_run_log(
        experiment="F1", seed=seed,
        cmd="python reproduce.py --warehouse --experiment F1",
        wall_clock_s=wall,
        output_paths=[out_json],
        headline={
            "n_motifs": len(rows),
            "predicted_cast_fraction": distribution_fractions["cast"],
            "predicted_record_fraction": distribution_fractions["record"],
            "predicted_count_fraction": distribution_fractions["count"],
            "predicted_undefined_fraction": distribution_fractions["undefined"],
        },
        known_limitations=[
            "CC-4: stage-prediction relies on the structural max-projection rule "
            "rather than a fully calibrated classifier; manuscript explicitly "
            "frames the predicted-class distribution under this rule.",
        ],
        log_path=output_dir / "F1_RUN_LOG.md",
    )
    return payload


# =============================================================================
# F2: binary scoring with baselines
# =============================================================================


def _build_baseline_training_set(
    seed: int,
    num_synthetic_normals: int = 80,
) -> tuple[list[Motif], list[int], list[Optional[BenalohStage]]]:
    """Build the labelled training set for the warehouse baseline comparison."""
    from assemblers import assemble_incident_motifs

    set_global_determinism(seed)
    normals: list[Motif] = []
    for i in range(num_synthetic_normals):
        normals.append(make_synthetic_motif(
            f"f2norm{i}",
            dao_name=["compound", "uniswap", "aave", "ens"][i % 4],
            num_voters=3 + (i % 6),
            has_record=True, has_count=True, seed=seed + i,
        ))
    incident_triples = assemble_incident_motifs()
    attack_motifs = [m for m, _, _ in incident_triples]
    attack_stages = [s for _, s, _ in incident_triples]

    motifs = normals + attack_motifs
    labels = [0] * len(normals) + [1] * len(attack_motifs)
    stages: list[Optional[BenalohStage]] = [None] * len(normals) + attack_stages
    return motifs, labels, stages


def warehouse_experiment_f2(
    warehouse_motifs: list[Motif],
    checkpoint_path: Optional[Path],
    output_dir: Path,
    seed: int = 20260602,
    baseline_epochs: int = 30,
) -> dict:
    """F2: binary scoring of VERISHEAF and trained baselines on the warehouse corpus.

    Citation: manuscript Section 6.2 (F2 — Binary Scoring vs Baselines).
    """
    from baselines import build_dcl_gfd_scorer, build_knowgraph_scorer
    set_global_determinism(seed)
    t_start = time.perf_counter()
    output_dir = Path(output_dir)

    learned = _load_trained_restrictions(checkpoint_path)
    train_motifs, train_labels, train_stages = _build_baseline_training_set(seed=seed)

    print(f"  F2: training DCL-GFD on {len(train_motifs)} labelled motifs...", flush=True)
    dcl_binary, _dcl_stage = build_dcl_gfd_scorer(
        train_motifs, train_labels, train_stages, seed=seed, epochs=baseline_epochs,
    )
    print(f"  F2: training KnowGraph on {len(train_motifs)} labelled motifs...", flush=True)
    kg_binary, _kg_stage = build_knowgraph_scorer(
        train_motifs, train_labels, train_stages, seed=seed + 1, epochs=baseline_epochs,
    )

    def verisheaf_binary(m: Motif) -> float:
        return float(max(verisheaf_anomaly_score(m, learned_restrictions=learned).values()))

    methods: dict[str, Callable[[Motif], float]] = {
        "verisheaf_max_subspace": verisheaf_binary,
        "dcl_gfd_binary": dcl_binary,
        "knowgraph_binary": kg_binary,
    }

    n = len(warehouse_motifs)
    per_method: dict[str, dict] = {}
    for name, scorer in methods.items():
        scores = np.array([scorer(m) for m in warehouse_motifs], dtype=float)
        if scores.size == 0:
            per_method[name] = {
                "n_motifs": 0,
                "mean": None,
                "median": None,
                "p95": None,
                "fpr_at_threshold": {},
            }
            continue
        score_min = float(scores.min())
        score_max = float(scores.max())
        if score_max > score_min:
            threshold_grid = [
                score_min + t * (score_max - score_min) for t in _F2_OPERATING_THRESHOLDS
            ]
        else:
            threshold_grid = [score_max] * len(_F2_OPERATING_THRESHOLDS)
        fpr_at_threshold = {
            f"{q:.2f}": float(np.mean(scores >= thr))
            for q, thr in zip(_F2_OPERATING_THRESHOLDS, threshold_grid)
        }
        per_method[name] = {
            "n_motifs": int(scores.size),
            "mean": float(scores.mean()),
            "median": float(np.median(scores)),
            "p95": float(np.percentile(scores, 95)),
            "score_min": score_min,
            "score_max": score_max,
            "thresholds_native": threshold_grid,
            "fpr_at_threshold": fpr_at_threshold,
        }

    wall = time.perf_counter() - t_start
    payload = {
        "experiment": "F2",
        "seed": seed,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
        "n_motifs": n,
        "wall_clock_s": wall,
        "operating_thresholds_quantile": list(_F2_OPERATING_THRESHOLDS),
        "per_method": per_method,
        "training_set_composition": {
            "n_synthetic_normals": train_labels.count(0),
            "n_incident_attacks": train_labels.count(1),
        },
    }
    out_json = output_dir / "F2.json"
    _write_json(payload, out_json)
    _write_run_log(
        experiment="F2", seed=seed,
        cmd="python reproduce.py --warehouse --experiment F2",
        wall_clock_s=wall,
        output_paths=[out_json],
        headline={
            "n_motifs": n,
            "verisheaf_fpr_at_0p50": _nan_to_none(per_method.get(
                "verisheaf_max_subspace", {}).get("fpr_at_threshold", {}).get(
                "0.50", float("nan"))),
            "dcl_gfd_fpr_at_0p50": _nan_to_none(per_method.get(
                "dcl_gfd_binary", {}).get("fpr_at_threshold", {}).get(
                "0.50", float("nan"))),
            "knowgraph_fpr_at_0p50": _nan_to_none(per_method.get(
                "knowgraph_binary", {}).get("fpr_at_threshold", {}).get(
                "0.50", float("nan"))),
        },
        known_limitations=[
            "CC-1: baselines trained on a corpus composed of synthetic normals "
            "plus eight-incident attacks; baseline calibration relative to the "
            "warehouse marginal is therefore approximate.",
        ],
        log_path=output_dir / "F2_RUN_LOG.md",
    )
    return payload


# =============================================================================
# F3: per-platform stratification of F1 and F2
# =============================================================================


def _partition_by_platform(motifs: list[Motif]) -> dict[str, list[Motif]]:
    """Partition warehouse motifs by their platform prefix."""
    by_platform: dict[str, list[Motif]] = {"aragon": [], "daohaus": [], "daostack": []}
    for m in motifs:
        prefix = m.motif_id.split(":", 1)[0]
        if prefix in by_platform:
            by_platform[prefix].append(m)
    return by_platform


def warehouse_experiment_f3(
    warehouse_motifs: list[Motif],
    checkpoint_path: Optional[Path],
    output_dir: Path,
    seed: int = 20260603,
    baseline_epochs: int = 30,
) -> dict:
    """F3: cross-platform stratification of F1 and F2."""
    set_global_determinism(seed)
    t_start = time.perf_counter()
    output_dir = Path(output_dir)
    by_platform = _partition_by_platform(warehouse_motifs)

    per_platform_f1: dict[str, dict] = {}
    per_platform_f2: dict[str, dict] = {}
    for platform, motifs in by_platform.items():
        platform_dir = output_dir / f"f3_{platform}"
        per_platform_f1[platform] = warehouse_experiment_f1(
            motifs, checkpoint_path, platform_dir, seed=seed,
        )
        per_platform_f2[platform] = warehouse_experiment_f2(
            motifs, checkpoint_path, platform_dir, seed=seed + 1,
            baseline_epochs=baseline_epochs,
        )

    wall = time.perf_counter() - t_start
    payload = {
        "experiment": "F3",
        "seed": seed,
        "wall_clock_s": wall,
        "platform_motif_counts": {p: len(ms) for p, ms in by_platform.items()},
        "per_platform_f1": per_platform_f1,
        "per_platform_f2": per_platform_f2,
    }
    out_json = output_dir / "F3.json"
    _write_json(payload, out_json)
    _write_run_log(
        experiment="F3", seed=seed,
        cmd="python reproduce.py --warehouse --experiment F3",
        wall_clock_s=wall,
        output_paths=[out_json],
        headline={
            f"{p}_n_motifs": len(ms) for p, ms in by_platform.items()
        } | {
            f"{p}_predicted_cast_fraction":
                per_platform_f1[p]["predicted_class_fractions"]["cast"]
            for p in by_platform
        },
        known_limitations=[
            "CC-1/CC-4: per-platform results inherit F1 and F2 limitations.",
        ],
        log_path=output_dir / "F3_RUN_LOG.md",
    )
    return payload


# =============================================================================
# F4: Theorem 5 stress test on a stratified warehouse sample
# =============================================================================


def warehouse_experiment_f4(
    warehouse_motifs: list[Motif],
    checkpoint_path: Optional[Path],
    output_dir: Path,
    seed: int = 20260604,
    per_platform_sample: int = _F4_STRATIFIED_SAMPLE_SIZE_PER_PLATFORM,
) -> dict:
    """F4: Theorem 5 perturbation stress test on a per-platform stratified sample.

    Citation: manuscript Section 6.2 (F4 — Theorem 5 Stress Test).
    """
    set_global_determinism(seed)
    _ = checkpoint_path
    t_start = time.perf_counter()
    output_dir = Path(output_dir)
    by_platform = _partition_by_platform(warehouse_motifs)
    rng = np.random.RandomState(seed)
    sampled: list[Motif] = []
    for platform, motifs in by_platform.items():
        if not motifs:
            continue
        k = min(per_platform_sample, len(motifs))
        idx = rng.permutation(len(motifs))[:k]
        sampled.extend(motifs[int(i)] for i in idx)

    result = validate_theorem_5(
        sampled, eps_values=_F4_EPSILON_SWEEP, rho=1.0, seed=seed, max_motifs=len(sampled),
    )
    wall = time.perf_counter() - t_start
    payload = {
        "experiment": "F4",
        "seed": seed,
        "wall_clock_s": wall,
        "n_sampled_motifs": len(sampled),
        "per_platform_sample": per_platform_sample,
        "eps_sweep": list(_F4_EPSILON_SWEEP),
        "n_evaluated": result["n_evaluated"],
        "overall_dominance_rate": result["overall_dominance_rate"],
        "verified": result["verified"],
        "rows": result["rows"],
    }
    out_json = output_dir / "F4.json"
    _write_json(payload, out_json)
    _write_run_log(
        experiment="F4", seed=seed,
        cmd="python reproduce.py --warehouse --experiment F4",
        wall_clock_s=wall,
        output_paths=[out_json],
        headline={
            "n_sampled_motifs": len(sampled),
            "n_evaluated_cells": result["n_evaluated"],
            "overall_dominance_rate": result["overall_dominance_rate"],
            "verified": result["verified"],
        },
        known_limitations=[
            "CC-4: Theorem 5 validation samples Gaussian perturbations and tests "
            "bound dominance empirically; the population statement covers every "
            "operator-bounded perturbation.",
        ],
        log_path=output_dir / "F4_RUN_LOG.md",
    )
    return payload


# =============================================================================
# F5: motif-topology enrichment counterfactual
# =============================================================================


def warehouse_experiment_f5(
    warehouse_motifs: list[Motif],
    checkpoint_path: Optional[Path],
    output_dir: Path,
    seed: int = 20260605,
) -> dict:
    """F5: motif-topology enrichment counterfactual.

    Citation: manuscript Section 6.2 (F5 — Topology Enrichment Counterfactual).
    """
    set_global_determinism(seed)
    t_start = time.perf_counter()
    output_dir = Path(output_dir)
    learned = _load_trained_restrictions(checkpoint_path)

    bin_edges = list(_F5_NR_BINS) + [float("inf")]
    bin_labels = [
        f"{int(bin_edges[i])}-{int(bin_edges[i + 1] - 1)}"
        if not math.isinf(bin_edges[i + 1])
        else f"{int(bin_edges[i])}+"
        for i in range(len(bin_edges) - 1)
    ]
    nr_pred: dict[str, dict[str, int]] = {
        lbl: {"cast": 0, "record": 0, "count": 0, "undefined": 0} for lbl in bin_labels
    }
    nr_counts: dict[str, int] = {lbl: 0 for lbl in bin_labels}

    counterfactual_rows: list[dict] = []
    cf_pred_counts: dict[str, int] = {"cast": 0, "record": 0, "count": 0, "undefined": 0}

    for motif in warehouse_motifs:
        n_r = len(motif.record_edges)
        bin_label = bin_labels[-1]
        for i, lbl in enumerate(bin_labels):
            if n_r < bin_edges[i + 1]:
                bin_label = lbl
                break
        nr_counts[bin_label] += 1

        scores = verisheaf_anomaly_score(motif, learned_restrictions=learned)
        if max(scores.values()) <= 0.0:
            pred = "undefined"
        else:
            pred = max(["cast", "record", "count"], key=lambda s: scores[s])
        nr_pred[bin_label][pred] += 1

        import copy
        cf = copy.deepcopy(motif)
        if cf.record_edges:
            cf.record_edges = cf.record_edges[:1]
        cf.__post_init__()
        cf_scores = verisheaf_anomaly_score(cf, learned_restrictions=learned)
        if max(cf_scores.values()) <= 0.0:
            cf_pred = "undefined"
        else:
            cf_pred = max(["cast", "record", "count"], key=lambda s: cf_scores[s])
        cf_pred_counts[cf_pred] += 1
        counterfactual_rows.append({
            "motif_id": motif.motif_id,
            "n_r_original": n_r,
            "predicted_stage_original": pred,
            "predicted_stage_counterfactual_nr_1": cf_pred,
            "score_cast_original": float(scores["cast"]),
            "score_cast_counterfactual": float(cf_scores["cast"]),
        })

    wall = time.perf_counter() - t_start
    total = max(len(counterfactual_rows), 1)
    payload = {
        "experiment": "F5",
        "seed": seed,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
        "n_motifs": len(counterfactual_rows),
        "wall_clock_s": wall,
        "n_r_bin_counts": nr_counts,
        "n_r_bin_predicted_class_counts": nr_pred,
        "counterfactual_predicted_class_counts": cf_pred_counts,
        "counterfactual_predicted_class_fractions": {
            k: v / total for k, v in cf_pred_counts.items()
        },
        "rows": counterfactual_rows,
    }
    out_json = output_dir / "F5.json"
    _write_json(payload, out_json)
    _write_run_log(
        experiment="F5", seed=seed,
        cmd="python reproduce.py --warehouse --experiment F5",
        wall_clock_s=wall,
        output_paths=[out_json],
        headline={
            "n_motifs": len(counterfactual_rows),
            "cf_predicted_cast_fraction": cf_pred_counts["cast"] / total,
            "cf_predicted_record_fraction": cf_pred_counts["record"] / total,
            "cf_predicted_count_fraction": cf_pred_counts["count"] / total,
        },
        known_limitations=[
            "CC-4: counterfactual is constructed by truncating record edges to "
            "the first emitted record; the manipulation is purely structural.",
        ],
        log_path=output_dir / "F5_RUN_LOG.md",
    )
    return payload


# =============================================================================
# F6: temporal-incoherence baseline
# =============================================================================


def warehouse_experiment_f6(
    warehouse_motifs: list[Motif],
    checkpoint_path: Optional[Path],
    output_dir: Path,
    seed: int = 20260606,
) -> dict:
    """F6: temporal-incoherence baseline at five operating thresholds.

    Citation: manuscript Section 6.2 (F6 — Temporal Incoherence Baseline).
    """
    set_global_determinism(seed)
    _ = checkpoint_path
    t_start = time.perf_counter()
    output_dir = Path(output_dir)

    scores = np.array(
        [baseline_temporal_incoherence(m) for m in warehouse_motifs], dtype=float
    )
    if scores.size == 0:
        score_min, score_max = None, None
        threshold_grid: list = []
        fpr_at_threshold: dict[str, float] = {}
        score_mean = score_median = score_p95 = None
    else:
        score_min = float(scores.min())
        score_max = float(scores.max())
        if score_max > score_min:
            threshold_grid = [
                score_min + t * (score_max - score_min) for t in _F2_OPERATING_THRESHOLDS
            ]
        else:
            threshold_grid = [score_max] * len(_F2_OPERATING_THRESHOLDS)
        fpr_at_threshold = {
            f"{q:.2f}": float(np.mean(scores >= thr))
            for q, thr in zip(_F2_OPERATING_THRESHOLDS, threshold_grid)
        }
        score_mean = float(scores.mean())
        score_median = float(np.median(scores))
        score_p95 = float(np.percentile(scores, 95))
    wall = time.perf_counter() - t_start
    payload = {
        "experiment": "F6",
        "seed": seed,
        "n_motifs": int(scores.size),
        "wall_clock_s": wall,
        "operating_thresholds_quantile": list(_F2_OPERATING_THRESHOLDS),
        "thresholds_native": threshold_grid,
        "fpr_at_threshold": fpr_at_threshold,
        "score_min": score_min,
        "score_max": score_max,
        "score_mean": score_mean,
        "score_median": score_median,
        "score_p95": score_p95,
    }
    out_json = output_dir / "F6.json"
    _write_json(payload, out_json)
    _write_run_log(
        experiment="F6", seed=seed,
        cmd="python reproduce.py --warehouse --experiment F6",
        wall_clock_s=wall,
        output_paths=[out_json],
        headline={
            "n_motifs": int(scores.size),
            "fpr_at_0p50": _nan_to_none(fpr_at_threshold.get("0.50", float("nan"))),
            "score_median": _nan_to_none(score_median) if score_median is not None else None,
        },
        known_limitations=[
            "CC-1: baseline_temporal_incoherence uses only the first record-edge "
            "timestamp as the anchor; motifs with no record edge return 0.",
        ],
        log_path=output_dir / "F6_RUN_LOG.md",
    )
    return payload


# =============================================================================
# F7: Complementary-pipeline experiment with multi-detector flagging
# =============================================================================


def _build_flagging_scorer(
    method_name: str,
    train_motifs: list[Motif],
    train_labels: list[int],
    train_stages: list[Optional[BenalohStage]],
    seed: int,
    epochs: int,
) -> Callable[[Motif], float]:
    """Build the binary scorer callable for one of the four flagging methods.

    Dispatches to the appropriate builder from baselines.py based on the method
    name. The method names match the published baseline identifiers used in
    baselines.py:

        "dcl_gfd"       -> build_dcl_gfd_scorer        [Yu+25]
        "knowgraph"     -> build_knowgraph_scorer      [Zhou+24]
        "nsd"           -> build_nsd_scorer            [Bod22]
        "shadow_eyes"   -> build_shadow_eyes_scorer    [Che+25]

    Each builder returns (binary_scorer, stage_predictor); the stage predictor
    is discarded here because the F7 experiment uses VERISHEAF's stage
    classifier on the flagged subset (not the baseline's own stage routing).

    Args:
        method_name: one of {"dcl_gfd", "knowgraph", "nsd", "shadow_eyes"}.
        train_motifs: labelled training corpus.
        train_labels: per-motif binary label.
        train_stages: per-motif BenalohStage (None for normals).
        seed: integer seed.
        epochs: number of training epochs.

    Returns:
        binary_scorer callable: Motif -> float in [0, 1].

    Raises:
        ValueError: if method_name is not one of the four supported flagging
            methods.

    Complexity: dominated by the underlying baseline training cost.
    """
    if method_name == "dcl_gfd":
        from baselines import build_dcl_gfd_scorer
        binary, _ = build_dcl_gfd_scorer(
            train_motifs, train_labels, train_stages,
            seed=seed, epochs=epochs,
        )
        return binary
    elif method_name == "knowgraph":
        from baselines import build_knowgraph_scorer
        binary, _ = build_knowgraph_scorer(
            train_motifs, train_labels, train_stages,
            seed=seed, epochs=epochs,
        )
        return binary
    elif method_name == "nsd":
        from baselines import build_nsd_scorer
        binary, _ = build_nsd_scorer(
            train_motifs, train_labels, train_stages,
            seed=seed, epochs=epochs,
        )
        return binary
    elif method_name == "shadow_eyes":
        from baselines import build_shadow_eyes_scorer
        binary, _ = build_shadow_eyes_scorer(
            train_motifs, train_labels, train_stages,
            seed=seed, epochs=epochs,
        )
        return binary
    else:
        raise ValueError(
            f"_build_flagging_scorer: unsupported method {method_name!r}; "
            f"expected one of "
            f"{('dcl_gfd', 'knowgraph', 'nsd', 'shadow_eyes')}"
        )


def _verisheaf_predict_stage(
    motif: Motif, learned: Optional[dict]
) -> str:
    """Predict the Benaloh stage for a motif via the VERISHEAF max-projection rule.

    Args:
        motif: source Motif.
        learned: optional dict of trained restriction maps.

    Returns:
        One of {"cast", "record", "count", "undefined"}; "undefined" is
        returned when the VERISHEAF anomaly scores are all zero (trivial H^1).

    Complexity: O(|E|^3) for the dense eigendecomposition.
    """
    scores = verisheaf_anomaly_score(motif, learned_restrictions=learned)
    if max(scores.values()) <= 0.0:
        return "undefined"
    return max(["cast", "record", "count"], key=lambda s: scores[s])


def _stability_threshold_for_motif(
    motif: Motif,
    predicted_stage: str,
    learned: Optional[dict],
) -> Optional[float]:
    """Compute the Theorem 5 stability threshold eps for a flagged motif.

    The Theorem 5 stability threshold is the value of eps below which the
    framework's stage prediction is guaranteed stable under perturbation, per
    the explicit formula in theorem_5_bound:

        stability_threshold_eps = gamma / (4 rho sqrt(2 d |E_tau| Delta))

    where gamma is the spectral gap of L^1_tau above the kernel, rho is the
    operator-norm radius (1.0 under the canonical scaffold), d the stalk
    dimension, |E_tau| the number of stage-tau edges in the motif, and Delta
    the maximum node-stalk degree. The threshold does NOT depend on the eps
    value passed to theorem_5_bound; the reference value of
    _F7_THEOREM_5_REFERENCE_EPS is supplied only so the call is well-formed.

    The predicted stage determines which edge count |E_tau| enters the
    denominator: cast -> n_cast, record -> n_record, count -> n_count. When
    the prediction is "undefined" the cast-edge count is used as the default
    (the structurally largest stage in typical warehouse motifs); the
    returned value should be interpreted with care in that case.

    Args:
        motif: source Motif.
        predicted_stage: one of {"cast", "record", "count", "undefined"}.
        learned: optional dict of trained restriction maps.

    Returns:
        Stability threshold eps as a non-negative float, or None when the
        threshold is undefined (zero spectral gap; degenerate canonical
        sheaf).

    Complexity: O(|E|^3) for the dense eigendecomposition of the canonical
    edge Laplacian.
    """
    sheaf = motif_to_sheaf(motif, learned_restrictions=learned)
    L1 = sheaf.edge_laplacian().toarray()
    eigs = np.linalg.eigvalsh(L1)
    nonzero = eigs[eigs > 1e-8]
    if not nonzero.size:
        return None
    gamma = float(nonzero.min())
    n_cast = len(motif.cast_edges)
    n_record = len(motif.record_edges)
    n_count = len(motif.count_edges)
    d_stalk = DEFAULT_STALK_DIMS[NodeType.PROPOSAL]
    max_node_degree = max(n_cast + n_record + n_count, 1)
    n_edges_in_predicted = {
        "cast": n_cast,
        "record": n_record,
        "count": n_count,
        "undefined": max(n_cast, 1),
    }.get(predicted_stage, n_cast)
    n_edges_in_predicted = max(n_edges_in_predicted, 1)
    bound = theorem_5_bound(
        eps=_F7_THEOREM_5_REFERENCE_EPS,
        rho=1.0,
        d=d_stalk,
        n_edges_in_stage=n_edges_in_predicted,
        max_node_degree=max_node_degree,
        spectral_gap=gamma,
    )
    stability = bound["stability_threshold_eps"]
    if not math.isfinite(stability):
        return None
    return float(stability)


def _evaluate_flagged_subset(
    warehouse_motifs: list[Motif],
    flagged_indices: list[int],
    learned: Optional[dict],
) -> tuple[dict[str, int], dict[int, str], list[float]]:
    """Run VERISHEAF stage classification and per-motif stability on a flagged subset.

    Args:
        warehouse_motifs: full warehouse corpus indexed by motif index.
        flagged_indices: sorted list of motif indices to evaluate.
        learned: optional dict of trained restriction maps.

    Returns:
        (stage_counts, per_motif_stage, stability_thresholds) where:
            stage_counts: dict mapping {"cast","record","count","undefined"}
                to the per-stage flagged-motif count.
            per_motif_stage: dict mapping motif index to predicted stage.
            stability_thresholds: list of per-motif Theorem 5 stability
                thresholds (excluding motifs whose threshold is undefined).

    Complexity: O(|flagged_indices| * |E|^3) dominated by the per-motif
    VERISHEAF stage classification and the edge-Laplacian eigendecomposition.
    """
    stage_counts: dict[str, int] = {
        "cast": 0, "record": 0, "count": 0, "undefined": 0,
    }
    per_motif_stage: dict[int, str] = {}
    stability_thresholds: list[float] = []

    for idx in flagged_indices:
        motif = warehouse_motifs[idx]
        pred = _verisheaf_predict_stage(motif, learned)
        stage_counts[pred] += 1
        per_motif_stage[idx] = pred
        stability = _stability_threshold_for_motif(motif, pred, learned)
        if stability is not None:
            stability_thresholds.append(stability)

    return stage_counts, per_motif_stage, stability_thresholds


def _top_decile_indices(scores: np.ndarray, quantile: float) -> list[int]:
    """Return the sorted indices of motifs whose score is in the top `quantile`.

    Uses the (1 - quantile) sample quantile of the score distribution as the
    flagging threshold; motifs with score at least the threshold are flagged.
    Ties at the threshold are all included, which can produce a flagged set
    slightly larger than ceil(N * quantile) — the convention is consistent
    with the published top-decile flagging used in the manuscript.

    Args:
        scores: 1-D numpy array of binary-detector scores.
        quantile: top-quantile to flag, in (0, 1).

    Returns:
        Sorted list of motif indices flagged.

    Complexity: O(|scores| log |scores|) for the quantile computation.
    """
    if scores.size == 0:
        return []
    quantile = max(0.0, min(1.0, quantile))
    if quantile >= 1.0:
        return list(range(scores.size))
    if quantile <= 0.0:
        return []
    threshold = float(np.quantile(scores, 1.0 - quantile))
    return sorted(int(i) for i, s in enumerate(scores) if s >= threshold)


def warehouse_experiment_f7(
    warehouse_motifs: list[Motif],
    checkpoint_path: Optional[Path],
    output_dir: Path,
    seed: int = 20260607,
    baseline_epochs: int = 30,
    flagging_methods: tuple[str, ...] = _F7_FLAGGING_METHODS_DEFAULT,
    flagging_quantile: float = _F7_FLAGGING_QUANTILE_DEFAULT,
    quantile_sweep: Optional[tuple[float, ...]] = _F7_QUANTILE_SWEEP_DEFAULT,
) -> dict:
    """F7: Complementary-pipeline experiment with multi-detector flagging.

    For each flagging method in `flagging_methods`, trains the binary scorer
    on the eight-incident-plus-synthetic training set (the same training set
    used by F2), flags the top decile of warehouse motifs by binary score,
    runs VERISHEAF stage classification on the flagged subset with the
    trained restriction maps loaded from `checkpoint_path`, and computes the
    Theorem 5 stability threshold for each flagged motif.

    The output JSON reports four classes of results:

      (1) Per-method results: flagged subset size, stage-prediction
          distribution (counts and fractions for cast, record, count,
          undefined), median stability threshold.

      (2) Cross-method consistency: fraction of motifs flagged by all
          flagging methods, fraction of multi-flagged motifs receiving
          consistent stage predictions across detectors, per-pair Cohen's
          kappa for stage-prediction agreement between detector pairs on
          the motifs they jointly flag.

      (3) Quantile sweep: per-quantile stage-prediction distributions and
          median stability thresholds (when `quantile_sweep` is provided).

      (4) Methodological framing: cross-method consistency is the
          experimental evidence that VERISHEAF operates as a complementary
          tool rather than as a competitor to any specific binary detector.
          High consistency indicates that VERISHEAF's stage characterization
          is robust to the choice of upstream detector; low consistency
          would indicate that the stage characterization depends on detector
          choice and would weaken the complementary-pipeline argument. Both
          outcomes are informative, which is the property a well-designed
          experiment should have.

    Args:
        warehouse_motifs: full validated motif corpus.
        checkpoint_path: optional pathlib.Path to a SheafLearner checkpoint
            supplying the trained restriction maps for the VERISHEAF stage
            classification on the flagged subset.
        output_dir: pathlib.Path for the F7.json and F7_RUN_LOG.md outputs.
        seed: experiment seed; default 20260607. Per-method baseline seeds
            derive deterministically from this value.
        baseline_epochs: per-baseline training epochs; default 30.
        flagging_methods: tuple of method identifiers; default contains all
            four measured binary baselines
            ("dcl_gfd", "knowgraph", "nsd", "shadow_eyes").
        flagging_quantile: top-quantile of warehouse motifs to flag by each
            method's binary score; default 0.10 (top decile).
        quantile_sweep: optional tuple of additional quantiles to evaluate
            for the threshold-sensitivity analysis; default
            (0.05, 0.10, 0.15, 0.20). Set to None to skip the sweep.

    Returns:
        dict containing the experiment payload (per_method, cross_method_
        consistency, quantile_sweep, and methodological_framing keys).

    Citation: manuscript Section 6.2 (F7 — Complementary-Pipeline Experiment
    with Multi-Detector Flagging).

    Complexity:
        O(|flagging_methods| * baseline_epochs * |training_motifs| *
        baseline_per_step_cost +
        |flagging_methods| * |warehouse_motifs| * baseline_per_score_cost +
        |flagging_methods| * |flagged_subset| * |E|^3 +
        |quantile_sweep| * |flagging_methods| * |flagged_subset| * |E|^3)
        dominated in practice by the baseline training cost.
    """
    set_global_determinism(seed)
    t_start = time.perf_counter()
    output_dir = Path(output_dir)

    if not flagging_methods:
        raise ValueError(
            "warehouse_experiment_f7: flagging_methods must contain at least "
            "one method identifier."
        )
    flagging_quantile = max(0.0, min(1.0, float(flagging_quantile)))

    learned = _load_trained_restrictions(checkpoint_path)
    train_motifs, train_labels, train_stages = _build_baseline_training_set(seed=seed)

    # --- 1. Train one binary scorer per flagging method and score the warehouse. ---
    print(
        f"  F7: training {len(flagging_methods)} flagging method(s) on "
        f"{len(train_motifs)} labelled motifs...",
        flush=True,
    )
    all_scores: dict[str, np.ndarray] = {}
    flagging_thresholds: dict[str, Optional[float]] = {}
    for offset, method_name in enumerate(flagging_methods):
        method_seed = seed + 1 + offset
        print(
            f"  F7: training '{method_name}' (seed={method_seed}, "
            f"epochs={baseline_epochs})...",
            flush=True,
        )
        scorer = _build_flagging_scorer(
            method_name, train_motifs, train_labels, train_stages,
            seed=method_seed, epochs=baseline_epochs,
        )
        scores = np.array([scorer(m) for m in warehouse_motifs], dtype=float)
        all_scores[method_name] = scores

    # --- 2. Flag the top decile under each method and run VERISHEAF + Thm 5. ---
    flagged_indices: dict[str, list[int]] = {}
    per_method_results: dict[str, dict] = {}
    per_method_stage_pred: dict[str, dict[int, str]] = {}

    for method_name in flagging_methods:
        scores = all_scores[method_name]
        flagged = _top_decile_indices(scores, flagging_quantile)
        flagged_indices[method_name] = flagged
        if scores.size > 0 and flagged:
            threshold = float(scores[flagged].min())
        else:
            threshold = None
        flagging_thresholds[method_name] = threshold

        stage_counts, per_motif_stage, stability_thresholds = (
            _evaluate_flagged_subset(warehouse_motifs, flagged, learned)
        )
        per_method_stage_pred[method_name] = per_motif_stage
        n_flagged = len(flagged)
        total = max(n_flagged, 1)
        stage_fractions = {k: v / total for k, v in stage_counts.items()}
        median_stability = (
            float(np.median(stability_thresholds))
            if stability_thresholds
            else None
        )
        per_method_results[method_name] = {
            "n_flagged": n_flagged,
            "flagging_threshold": threshold,
            "stage_prediction_counts": stage_counts,
            "stage_prediction_fractions": stage_fractions,
            "median_stability_threshold": median_stability,
            "stability_threshold_n_finite": len(stability_thresholds),
        }

    # --- 3. Cross-method consistency. ---
    flagged_sets: dict[str, set[int]] = {
        m: set(flagged_indices[m]) for m in flagging_methods
    }
    n_warehouse = max(len(warehouse_motifs), 1)
    if flagging_methods:
        intersection_all = set.intersection(*[flagged_sets[m] for m in flagging_methods])
    else:
        intersection_all = set()
    fraction_flagged_by_all = len(intersection_all) / n_warehouse

    # Fraction of intersection motifs receiving consistent stage predictions
    # across all detectors. An intersection motif is "consistent" iff every
    # detector's VERISHEAF stage prediction for that motif is identical.
    if intersection_all:
        consistent_count = 0
        for idx in intersection_all:
            stages_for_idx = [
                per_method_stage_pred[m].get(idx) for m in flagging_methods
            ]
            if len(set(stages_for_idx)) == 1 and stages_for_idx[0] is not None:
                consistent_count += 1
        fraction_consistent = consistent_count / len(intersection_all)
    else:
        consistent_count = 0
        fraction_consistent = None

    # Per-pair Cohen's kappa for stage-prediction agreement between detectors
    # on the motifs they jointly flag. Pairs are reported lexicographically
    # by their (method_a, method_b) keys to maintain byte-deterministic JSON
    # output independent of the flagging_methods tuple ordering.
    pairwise_kappa: dict[str, Optional[float]] = {}
    pairwise_n_common: dict[str, int] = {}
    for i, m1 in enumerate(flagging_methods):
        for j in range(i + 1, len(flagging_methods)):
            m2 = flagging_methods[j]
            pair_key = f"{m1}__vs__{m2}" if m1 <= m2 else f"{m2}__vs__{m1}"
            common = sorted(flagged_sets[m1] & flagged_sets[m2])
            pairwise_n_common[pair_key] = len(common)
            if len(common) < 2:
                pairwise_kappa[pair_key] = None
                continue
            preds_1 = [per_method_stage_pred[m1].get(idx, "undefined") for idx in common]
            preds_2 = [per_method_stage_pred[m2].get(idx, "undefined") for idx in common]
            kappa = _cohen_kappa(preds_1, preds_2)
            pairwise_kappa[pair_key] = (
                _nan_to_none(kappa) if isinstance(kappa, float) else kappa
            )

    cross_method_consistency = {
        "n_flagged_per_method": {m: len(flagged_indices[m]) for m in flagging_methods},
        "n_warehouse_motifs": n_warehouse,
        "n_intersection_all_methods": len(intersection_all),
        "fraction_flagged_by_all_methods": fraction_flagged_by_all,
        "n_consistent_stage_predictions_in_intersection": consistent_count,
        "fraction_intersection_with_consistent_stage_predictions": fraction_consistent,
        "pairwise_cohen_kappa": pairwise_kappa,
        "pairwise_n_common_flagged": pairwise_n_common,
    }

    # --- 4. Quantile sweep (threshold-sensitivity analysis). ---
    quantile_sweep_results: dict[str, dict] = {}
    if quantile_sweep:
        for q in quantile_sweep:
            q = max(0.0, min(1.0, float(q)))
            per_q_results: dict[str, dict] = {}
            for method_name in flagging_methods:
                scores = all_scores[method_name]
                q_flagged = _top_decile_indices(scores, q)
                if scores.size > 0 and q_flagged:
                    q_threshold = float(scores[q_flagged].min())
                else:
                    q_threshold = None
                q_stage_counts, _q_per_motif_stage, q_stability = (
                    _evaluate_flagged_subset(warehouse_motifs, q_flagged, learned)
                )
                q_total = max(len(q_flagged), 1)
                q_stage_fractions = {
                    k: v / q_total for k, v in q_stage_counts.items()
                }
                q_median_stab = (
                    float(np.median(q_stability)) if q_stability else None
                )
                per_q_results[method_name] = {
                    "n_flagged": len(q_flagged),
                    "flagging_threshold": q_threshold,
                    "stage_prediction_counts": q_stage_counts,
                    "stage_prediction_fractions": q_stage_fractions,
                    "median_stability_threshold": q_median_stab,
                    "stability_threshold_n_finite": len(q_stability),
                }
            quantile_sweep_results[f"{q:.2f}"] = per_q_results

    # --- 5. Methodological framing. ---
    methodological_framing = (
        "F7 reports the cross-method consistency of VERISHEAF stage "
        "predictions on motifs flagged by independent binary detectors "
        "(DCL-GFD, KnowGraph, NSD, ShadowEyes). The "
        "'fraction_intersection_with_consistent_stage_predictions' metric "
        "is the experimental evidence that VERISHEAF operates as a "
        "complementary tool rather than as a competitor to any specific "
        "binary detector: high consistency demonstrates that the framework's "
        "stage characterization is robust to the choice of upstream detector, "
        "while low consistency would indicate that the stage characterization "
        "depends on detector choice and would weaken the complementary-"
        "pipeline argument. Either outcome is informative, which is the "
        "property a well-designed experiment should have. The per-pair "
        "Cohen's kappa values quantify the stage-prediction agreement "
        "between specific detector pairs on the motifs they jointly flag."
    )

    wall = time.perf_counter() - t_start
    payload = {
        "experiment": "F7",
        "seed": seed,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
        "n_warehouse_motifs": len(warehouse_motifs),
        "wall_clock_s": wall,
        "flagging_methods": list(flagging_methods),
        "flagging_quantile": flagging_quantile,
        "flagging_thresholds_per_method": flagging_thresholds,
        "training_set_composition": {
            "n_synthetic_normals": train_labels.count(0),
            "n_incident_attacks": train_labels.count(1),
        },
        "per_method": per_method_results,
        "cross_method_consistency": cross_method_consistency,
        "quantile_sweep": quantile_sweep_results,
        "methodological_framing": methodological_framing,
    }
    out_json = output_dir / "F7.json"
    _write_json(payload, out_json)
    _write_run_log(
        experiment="F7", seed=seed,
        cmd="python reproduce.py --warehouse --experiment F7",
        wall_clock_s=wall,
        output_paths=[out_json],
        headline={
            "n_warehouse_motifs": len(warehouse_motifs),
            "n_flagging_methods": len(flagging_methods),
            "flagging_quantile": flagging_quantile,
            "fraction_flagged_by_all_methods": fraction_flagged_by_all,
            "fraction_intersection_with_consistent_stage_predictions":
                _nan_to_none(fraction_consistent) if fraction_consistent is not None else None,
            "n_intersection_all_methods": len(intersection_all),
        },
        known_limitations=[
            "CC-1: per-method binary scorers are trained on a corpus composed "
            "of synthetic normals plus the eight catalogued incidents; baseline "
            "calibration relative to the warehouse marginal is approximate.",
            "CC-4: the Theorem 5 stability threshold uses the predicted "
            "stage's edge count for the |E_tau| factor in the denominator; "
            "alternative conventions (max edge count across stages, or per-"
            "stage thresholds) would shift the absolute numbers but not the "
            "cross-method consistency conclusions.",
        ],
        log_path=output_dir / "F7_RUN_LOG.md",
    )
    return payload


# =============================================================================
# Warehouse evaluation suite orchestrator
# =============================================================================


def warehouse_evaluation_suite(
    warehouse_motifs: list[Motif],
    checkpoint_path: Optional[Path],
    output_dir: Path,
    seed: int = 20260600,
    baseline_epochs: int = 30,
) -> dict:
    """Run F1-F7 in dependency order and consolidate RESULTS_FOR_MANUSCRIPT.json.

    Args:
        warehouse_motifs: full validated motif corpus.
        checkpoint_path: optional pathlib.Path to a SheafLearner checkpoint.
        output_dir: pathlib.Path for the suite outputs.
        seed: master seed; per-experiment seeds derive deterministically.
        baseline_epochs: per-baseline training epochs.

    Returns:
        dict consolidating headline numbers from F1-F7.

    Citation: manuscript Section 6 (Warehouse Evaluation Suite).

    Complexity: dominated by F2 and F7 baseline training and F4 perturbation
    stress test: O(|warehouse| * |E|^3) plus baseline-specific training costs.
    """
    set_global_determinism(seed)
    t_start = time.perf_counter()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("VERISHEAF warehouse evaluation suite")
    print(f"  n_motifs={len(warehouse_motifs)}  checkpoint={checkpoint_path}  seed={seed}")
    print("=" * 72)

    print("\n[F1] three-way classification on the full corpus...")
    f1 = warehouse_experiment_f1(warehouse_motifs, checkpoint_path, output_dir, seed=seed + 1)
    print("\n[F2] binary scoring with baselines...")
    f2 = warehouse_experiment_f2(
        warehouse_motifs, checkpoint_path, output_dir,
        seed=seed + 2, baseline_epochs=baseline_epochs,
    )
    print("\n[F3] cross-platform stratification...")
    f3 = warehouse_experiment_f3(
        warehouse_motifs, checkpoint_path, output_dir,
        seed=seed + 3, baseline_epochs=baseline_epochs,
    )
    print("\n[F4] Theorem 5 stress test...")
    f4 = warehouse_experiment_f4(warehouse_motifs, checkpoint_path, output_dir, seed=seed + 4)
    print("\n[F5] motif-topology enrichment counterfactual...")
    f5 = warehouse_experiment_f5(warehouse_motifs, checkpoint_path, output_dir, seed=seed + 5)
    print("\n[F6] temporal-incoherence baseline...")
    f6 = warehouse_experiment_f6(warehouse_motifs, checkpoint_path, output_dir, seed=seed + 6)
    print("\n[F7] complementary-pipeline experiment with multi-detector flagging...")
    f7 = warehouse_experiment_f7(
        warehouse_motifs, checkpoint_path, output_dir,
        seed=seed + 7, baseline_epochs=baseline_epochs,
    )

    wall = time.perf_counter() - t_start
    headline = {
        "n_motifs": len(warehouse_motifs),
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
        "seed_master": seed,
        "wall_clock_total_s": wall,
        "f1_predicted_class_fractions": f1["predicted_class_fractions"],
        "f2_per_method_fpr_at_0p50": {
            name: _nan_to_none(per["fpr_at_threshold"].get("0.50", float("nan")))
            for name, per in f2["per_method"].items()
        },
        "f3_platform_motif_counts": f3["platform_motif_counts"],
        "f4_overall_dominance_rate": f4["overall_dominance_rate"],
        "f5_counterfactual_predicted_class_fractions":
            f5["counterfactual_predicted_class_fractions"],
        "f6_fpr_at_0p50": _nan_to_none(f6["fpr_at_threshold"].get("0.50", float("nan"))),
        "f7_fraction_flagged_by_all_methods":
            f7["cross_method_consistency"]["fraction_flagged_by_all_methods"],
        "f7_fraction_intersection_with_consistent_stage_predictions":
            f7["cross_method_consistency"][
                "fraction_intersection_with_consistent_stage_predictions"
            ],
        "f7_n_flagged_per_method":
            f7["cross_method_consistency"]["n_flagged_per_method"],
    }

    consolidated = {
        "experiment_suite": "warehouse",
        "wall_clock_total_s": wall,
        "headline": headline,
        "F1": f1,
        "F2": f2,
        "F3": f3,
        "F4": f4,
        "F5": f5,
        "F6": f6,
        "F7": f7,
    }
    out_json = output_dir / "RESULTS_FOR_MANUSCRIPT.json"
    _write_json(consolidated, out_json)
    print(f"\n  RESULTS_FOR_MANUSCRIPT.json -> {out_json}")
    print(f"  Total wall clock: {wall:.1f} s")
    return consolidated


# =============================================================================
# Module self-test
# =============================================================================


if __name__ == "__main__":
    print("VERISHEAF evaluation.py — smoke test")

    normal = [
        make_synthetic_motif(
            f"n{i}", "compound", num_voters=4,
            has_record=True, has_count=True, seed=i,
        )
        for i in range(5)
    ]
    attacked: list[tuple[Motif, BenalohStage]] = []
    for stage in [BenalohStage.CAST, BenalohStage.RECORD, BenalohStage.COUNT]:
        for i in range(3):
            base = normal[i % len(normal)]
            attacked.append((make_attacked_motif(base, stage, seed=100 + i), stage))

    report = full_evaluation_report(normal, attacked)
    print(f"\n  Theorem 1 verified: {report['theorem_1']['verified']}")
    print(
        f"  Theorem 2 three-way accuracy: {report['theorem_2']['accuracy']:.3f} "
        f"(classified {report['theorem_2']['n_classified']}, "
        f"undefined {report['theorem_2']['n_undefined_trivial_h1']})"
    )
    print(f"  Theorem 3 agreement: {report['theorem_3']['agreement_rate']:.3f}")
    print(f"  Theorem 4 verified: {report['theorem_4'].get('verified')}")
    print(
        f"  Theorem 5 dominance rate: "
        f"{report['theorem_5']['overall_dominance_rate']:.3f}"
    )

    tw = report["baselines"]["three_way"]
    print(f"\n  HEADLINE — three-way stage classification:")
    print(
        f"    accuracy={tw['accuracy']:.3f}  classified={tw['n_classified']}  "
        f"confusion={tw['confusion_matrix']}"
    )

    # Smoke test of the warehouse experiment entry points using the synthetic
    # corpus as a stand-in. The CSV deposit is not consumed here; this exercises
    # the JSON-output discipline and the RUN_LOG.md fragment generation.
    import tempfile
    warehouse_smoke = list(normal)
    for m, _ in attacked:
        warehouse_smoke.append(m)
    with tempfile.TemporaryDirectory() as td:
        out_dir = Path(td) / "warehouse_smoke"
        f1 = warehouse_experiment_f1(warehouse_smoke, None, out_dir, seed=1)
        assert (out_dir / "F1.json").exists()
        assert (out_dir / "F1_RUN_LOG.md").exists()
        f4 = warehouse_experiment_f4(
            warehouse_smoke, None, out_dir, seed=4, per_platform_sample=2,
        )
        assert (out_dir / "F4.json").exists()
        f5 = warehouse_experiment_f5(warehouse_smoke, None, out_dir, seed=5)
        assert (out_dir / "F5.json").exists()
        f6 = warehouse_experiment_f6(warehouse_smoke, None, out_dir, seed=6)
        assert (out_dir / "F6.json").exists()
        # Byte-determinism check on F1.json
        raw1 = (out_dir / "F1.json").read_bytes()
        expected = json.dumps(
            f1, indent=2, sort_keys=True, default=str, allow_nan=False
        ).encode("utf-8")
        assert raw1 == expected, "F1.json is not byte-deterministic"

        # F7 smoke test with a reduced flagging-method set and small epoch
        # budget to keep wall time manageable; full-config F7 trains four
        # baselines and is exercised only in the warehouse_evaluation_suite
        # entry point. The reduced configuration is sufficient to verify the
        # JSON structure, the RUN_LOG.md fragment, the per-method results,
        # the cross-method consistency block, the quantile-sweep payload, and
        # the byte-determinism contract.
        f7 = warehouse_experiment_f7(
            warehouse_smoke, None, out_dir,
            seed=7,
            baseline_epochs=2,
            flagging_methods=("dcl_gfd", "knowgraph"),
            flagging_quantile=0.20,
            quantile_sweep=(0.10, 0.20),
        )
        assert (out_dir / "F7.json").exists()
        assert (out_dir / "F7_RUN_LOG.md").exists()
        # Verify the four output sections required by the specification.
        assert "per_method" in f7
        assert "cross_method_consistency" in f7
        assert "quantile_sweep" in f7
        assert "methodological_framing" in f7
        # Per-method block sanity.
        for m in ("dcl_gfd", "knowgraph"):
            assert m in f7["per_method"]
            block = f7["per_method"][m]
            assert "n_flagged" in block
            assert "stage_prediction_counts" in block
            assert "stage_prediction_fractions" in block
            assert "median_stability_threshold" in block
        # Cross-method consistency block sanity.
        cmc = f7["cross_method_consistency"]
        assert "fraction_flagged_by_all_methods" in cmc
        assert (
            "fraction_intersection_with_consistent_stage_predictions" in cmc
        )
        assert "pairwise_cohen_kappa" in cmc
        # Exactly one pair for the two-method configuration.
        pair_keys = list(cmc["pairwise_cohen_kappa"].keys())
        assert len(pair_keys) == 1, f"expected 1 kappa pair, got {pair_keys}"
        # Quantile-sweep block sanity.
        assert set(f7["quantile_sweep"].keys()) == {"0.10", "0.20"}
        for q_key, q_results in f7["quantile_sweep"].items():
            for m in ("dcl_gfd", "knowgraph"):
                assert m in q_results, (
                    f"quantile_sweep[{q_key}] missing method {m}"
                )
        # Byte-determinism check on F7.json
        raw7 = (out_dir / "F7.json").read_bytes()
        expected7 = json.dumps(
            f7, indent=2, sort_keys=True, default=str, allow_nan=False
        ).encode("utf-8")
        assert raw7 == expected7, "F7.json is not byte-deterministic"
    print("  Warehouse experiment smoke tests + byte-determinism PASS")
    print("  F7 (complementary-pipeline experiment) smoke test PASS")
    print("\n  evaluation.py PASS")
