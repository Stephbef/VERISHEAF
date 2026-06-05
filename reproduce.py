"""reproduce.py — VERISHEAF end-to-end reproduction orchestrator.

Single command-line entry point for the entire experimental program. Four
subcommands are exposed via argparse subparsers, each producing a structured
output directory under 04_outputs/ with byte-deterministic JSON and a
RUN_LOG.md fragment per experiment.

SUBCOMMANDS
-----------
    --quick      Smoke-test against a small synthetic corpus. Completes in
                 under thirty seconds. Verifies that the framework's wiring
                 is correct on the operator's machine. Writes
                 04_outputs/quick/metrics.json and a summary.md.

    --incident   The eight-incident benchmark experiment. Builds the eight
                 incident motifs via assemblers.assemble_incident_motifs,
                 trains the SheafLearner across the configured seed set
                 (single-GPU by default; multi-GPU via multi_seed_training
                 when CUDA is available), runs the full evaluation report,
                 and writes 04_outputs/incident/metrics.json plus a
                 summary.md.

    --warehouse  The principal experimental program. Assembles the warehouse
                 motif corpus from the 18072773 deposit, trains the
                 SheafLearner on the corpus (multi-seed multi-GPU when
                 hardware permits), runs F1 through F7 in dependency order,
                 and writes the consolidated 04_outputs/warehouse/
                 RESULTS_FOR_MANUSCRIPT.json together with one RUN_LOG.md
                 per experiment under 04_outputs/run_logs/.

                 The F7 complementary-pipeline experiment trains four binary
                 scorers (DCL-GFD, KnowGraph, NSD, ShadowEyes) on the
                 labelled training set, flags the top decile of warehouse
                 motifs per method, and runs VERISHEAF stage characterization
                 together with Theorem 5 stability-threshold computation on
                 each flagged subset. The cross-method consistency of stage
                 predictions on motifs flagged by multiple detectors is the
                 principal headline number, surfaced in the consolidated
                 RESULTS_FOR_MANUSCRIPT.json under the headline.f7_* keys.

    --verify     Sanity-check the corpus by loading the warehouse manifest
                 and printing a one-glance summary of file metadata,
                 per-platform motif counts, validation failures, and the
                 eight-incident cross-reference.

CONSTRUCTION RULES
------------------
Every subcommand calls set_global_determinism with the global seed before any
stochastic operation. The PYTHONHASHSEED and CUBLAS_WORKSPACE_CONFIG
environment variables are checked and, when not set externally, are set by
set_global_determinism so the program is reproducible bit-for-bit when the
operator launches Python with the documented preamble. Wall-time tracking is
per-experiment and per-stage. Errors in any individual experiment are caught,
logged in stage_errors[], and do not abort the orchestrator; partial results
are still written.

F2, F3, and F7 all train the same four measured baselines (DCL-GFD, KnowGraph,
NSD, ShadowEyes) on the labelled training set. The baseline_epochs parameter
is routed to all three experiments through the orchestrator's per-experiment
dispatch table, so a single --baseline-epochs override propagates consistently
across the binary-scoring, per-platform stratification, and complementary-
pipeline experiments. The F4, F5, and F6 experiments do not train baselines
and are dispatched without the baseline_epochs argument.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from assemblers import (
    WarehouseAssembler,
    WarehouseConfig,
    assemble_incident_motifs,
    write_warehouse_manifest,
)
from evaluation import (
    bootstrap_ci,
    full_evaluation_report,
    holm_correct,
    paired_wilcoxon,
    warehouse_evaluation_suite,
    warehouse_experiment_f1,
    warehouse_experiment_f2,
    warehouse_experiment_f3,
    warehouse_experiment_f4,
    warehouse_experiment_f5,
    warehouse_experiment_f6,
    warehouse_experiment_f7,
)
from motifs import (
    BenalohStage,
    Motif,
    NodeType,
    make_attacked_motif,
    make_synthetic_motif,
)
from theory import (
    DEFAULT_STALK_DIMS,
    pac_bayes_bound,
    verify_elgamal_substitution,
    verify_theorem_1,
    verify_theorem_2,
    verify_theorem_3,
    verify_theorem_4,
    verify_theorem_5,
)
from training import (
    TrainingConfig,
    multi_seed_training,
    set_global_determinism,
)


# =============================================================================
# Per-experiment dispatch tables. These constants define which experiments take
# the baseline_epochs argument (F2, F3, F7 — the three experiments that train
# binary scorers from the four measured baselines) and which do not (F1, F4,
# F5, F6 — the pure-VERISHEAF experiments).
# =============================================================================

# Experiments that train one or more binary baseline scorers and therefore
# accept baseline_epochs. F2 trains four; F3 inherits F2's training per
# platform; F7 trains four (one per flagging method).
_EXPERIMENTS_WITH_BASELINE_EPOCHS: frozenset[str] = frozenset({"F2", "F3", "F7"})


# =============================================================================
# Global configuration
# =============================================================================


@dataclass
class GlobalConfig:
    """Single source of truth for orchestrator-wide settings.

    Attributes:
        global_seed: master seed; per-experiment seeds derive from it.
        outputs_root: pathlib.Path for all generated artefacts.
        run_logs_dir: pathlib.Path for per-experiment RUN_LOG.md fragments.
        warehouse_data_path: pathlib.Path to the 18072773 deposit root.
        warehouse_motifs_cache: pathlib.Path for the validated motif corpus
            cache (assembled once per --warehouse run).
        warehouse_manifest_path: pathlib.Path for the warehouse manifest.
        incident_outputs_dir: pathlib.Path for --incident outputs.
        warehouse_outputs_dir: pathlib.Path for --warehouse outputs.
        quick_outputs_dir: pathlib.Path for --quick outputs.
        training_config: TrainingConfig used by --incident and --warehouse.
        baseline_epochs: per-baseline training epochs for F2, F3, and F7.
        bootstrap_resamples: bootstrap iterations.
        confidence_level: bootstrap confidence level.
        statistical_alpha: family-wise error rate for Holm-Bonferroni.
    """

    global_seed: int = 20260601
    outputs_root: Path = Path("04_outputs")
    run_logs_dir: Path = Path("04_outputs/run_logs")
    warehouse_data_path: Path = Path("02_data/18072773")
    warehouse_motifs_cache: Path = Path("04_outputs/warehouse/motifs_cache.json")
    warehouse_manifest_path: Path = Path("04_outputs/warehouse/manifest.json")
    incident_outputs_dir: Path = Path("04_outputs/incident")
    warehouse_outputs_dir: Path = Path("04_outputs/warehouse")
    quick_outputs_dir: Path = Path("04_outputs/quick")

    training_config: TrainingConfig = field(default_factory=TrainingConfig)
    baseline_epochs: int = 30

    bootstrap_resamples: int = 1000
    confidence_level: float = 0.95
    statistical_alpha: float = 0.05


# =============================================================================
# Environment preamble
# =============================================================================


def _check_environment_preamble() -> dict[str, str]:
    """Verify reproducibility-critical environment variables and report them.

    PYTHONHASHSEED and CUBLAS_WORKSPACE_CONFIG must be set BEFORE Python
    launches for full byte-equal reproducibility. set_global_determinism mutates
    them when not set, but the mutation has no effect on modules already
    imported by the interpreter. The orchestrator reports the observed values
    so the operator knows whether to relaunch with the documented preamble.

    Returns:
        dict mapping variable name to its observed value (or 'unset').

    Complexity: O(1).
    """
    return {
        "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED", "unset"),
        "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG", "unset"),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
    }


def _hardware_report() -> dict:
    """Return a snapshot of the hardware context the orchestrator observes.

    Returns:
        dict with CUDA availability, device count, and per-device names.

    Complexity: O(device count).
    """
    cuda_available = torch.cuda.is_available()
    n_devices = torch.cuda.device_count() if cuda_available else 0
    devices = []
    if cuda_available:
        for i in range(n_devices):
            devices.append({
                "index": i,
                "name": torch.cuda.get_device_name(i),
                "total_memory_bytes": int(
                    torch.cuda.get_device_properties(i).total_memory
                ),
            })
    return {
        "cuda_available": cuda_available,
        "n_cuda_devices": n_devices,
        "devices": devices,
        "torch_version": torch.__version__,
    }


# =============================================================================
# Stage tracking
# =============================================================================


@dataclass
class _StageRecord:
    """Per-stage execution record for the orchestrator.

    Attributes:
        name: stage identifier (e.g., 'quick.theorem_verification').
        status: 'ok' or 'error'.
        wall_clock_s: stage wall-clock duration in seconds.
        error: optional error string (empty on ok).
        output_paths: list of pathlib.Path produced by the stage.
        headline: dict of headline numbers for the stage.
    """

    name: str
    status: str = "ok"
    wall_clock_s: float = 0.0
    error: str = ""
    output_paths: list[str] = field(default_factory=list)
    headline: dict = field(default_factory=dict)


def _run_stage(
    name: str,
    fn,
    *args,
    stage_records: list[_StageRecord],
    **kwargs,
) -> Optional[object]:
    """Execute a stage callable, capturing wall time and any error.

    Args:
        name: stage identifier.
        fn: callable to execute.
        *args: positional arguments forwarded to fn.
        stage_records: list of _StageRecord; the new record is appended.
        **kwargs: keyword arguments forwarded to fn.

    Returns:
        Whatever fn returns on success, or None on error.

    Complexity: O(fn).
    """
    record = _StageRecord(name=name)
    stage_records.append(record)
    t_start = time.perf_counter()
    try:
        result = fn(*args, **kwargs)
        record.wall_clock_s = time.perf_counter() - t_start
        return result
    except Exception as exc:
        record.wall_clock_s = time.perf_counter() - t_start
        record.status = "error"
        record.error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        print(f"  [ERROR] stage {name}: {exc}", file=sys.stderr, flush=True)
        return None


def _write_stage_log(
    stage_records: list[_StageRecord],
    output_path: Path,
    mode: str,
    global_seed: int,
) -> None:
    """Write a consolidated stages.json capturing every stage's wall time and status.

    Args:
        stage_records: list of _StageRecord.
        output_path: pathlib.Path destination.
        mode: subcommand name ('quick', 'incident', 'warehouse', 'verify').
        global_seed: master seed.

    Complexity: O(|stage_records|).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "mode": mode,
        "global_seed": global_seed,
        "env": _check_environment_preamble(),
        "hardware": _hardware_report(),
        "stages": [
            {
                "name": r.name,
                "status": r.status,
                "wall_clock_s": r.wall_clock_s,
                "error": r.error,
                "output_paths": r.output_paths,
                "headline": r.headline,
            }
            for r in stage_records
        ],
        "wall_clock_total_s": sum(r.wall_clock_s for r in stage_records),
        "n_errors": sum(1 for r in stage_records if r.status == "error"),
    }
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str, allow_nan=False),
        encoding="utf-8",
    )


# =============================================================================
# --quick subcommand
# =============================================================================


def _build_quick_corpus(
    config: GlobalConfig,
) -> tuple[list[Motif], list[tuple[Motif, BenalohStage]]]:
    """Build the small synthetic corpus for the --quick smoke test.

    Args:
        config: GlobalConfig.

    Returns:
        (normal_motifs, attacked_motifs) tuple.

    Complexity: O(20 motifs).
    """
    set_global_determinism(config.global_seed)
    normals = [
        make_synthetic_motif(
            f"qn{i}", "compound", num_voters=4,
            has_record=True, has_count=True, seed=config.global_seed + i,
        )
        for i in range(8)
    ]
    attacked: list[tuple[Motif, BenalohStage]] = []
    for stage in [BenalohStage.CAST, BenalohStage.RECORD, BenalohStage.COUNT]:
        for i in range(2):
            base = normals[i % len(normals)]
            attacked.append((
                make_attacked_motif(base, stage, seed=config.global_seed + 1000 + i),
                stage,
            ))
    return normals, attacked


def run_quick(config: GlobalConfig) -> int:
    """Execute the --quick smoke test.

    Verifies the framework's wiring against a small synthetic corpus, runs the
    six structural verifiers (the five theorem verifiers plus the new
    verify_elgamal_substitution from Modification 1E), runs the full evaluation
    report on the synthetic motifs, and writes a metrics.json plus summary.md.

    Args:
        config: GlobalConfig.

    Returns:
        0 on success, 1 if any stage errored.

    Complexity: O(20 motifs * |E|^3); under thirty seconds on commodity hardware.
    """
    set_global_determinism(config.global_seed)
    config.quick_outputs_dir.mkdir(parents=True, exist_ok=True)
    stage_records: list[_StageRecord] = []
    t_start = time.perf_counter()

    print("=" * 72)
    print("VERISHEAF --quick: smoke-test against a small synthetic corpus")
    print(f"  global_seed={config.global_seed}")
    print(f"  outputs={config.quick_outputs_dir}")
    print("=" * 72)

    print("\n[1/3] Theorem verifiers + ElGamal substitution check...")
    theorem_checks: dict[str, bool] = {}
    for name, fn in [
        ("theorem_1", verify_theorem_1),
        ("theorem_2", verify_theorem_2),
        ("theorem_3", verify_theorem_3),
        ("theorem_4", lambda: verify_theorem_4(num_trials=20)),
        ("theorem_5", lambda: verify_theorem_5(num_trials=5, motif_sizes=[4, 8])),
        ("elgamal_substitution", verify_elgamal_substitution),
    ]:
        rec = _run_stage(f"quick.{name}", fn, stage_records=stage_records)
        ok = bool(rec) if rec is not None else False
        theorem_checks[name] = ok
        print(f"    {name}: {'PASS' if ok else 'FAIL'}")

    print("\n[2/3] Corpus + evaluation...")
    normals, attacked = _run_stage(
        "quick.corpus",
        _build_quick_corpus, config,
        stage_records=stage_records,
    ) or ([], [])
    report = _run_stage(
        "quick.evaluation",
        full_evaluation_report, normals, attacked,
        stage_records=stage_records,
    )

    if report is None:
        report = {
            "theorem_1": {"verified": False},
            "theorem_2": {"accuracy": float("nan"), "confusion_matrix": [],
                          "n_classified": 0, "n_undefined_trivial_h1": 0},
            "theorem_3": {"agreement_rate": float("nan"), "verified": False},
            "theorem_4": {"verified": False},
            "theorem_5": {"overall_dominance_rate": float("nan"), "verified": False},
            "baselines": {"binary": {}, "three_way": {"accuracy": float("nan"),
                                                       "confusion_matrix": None,
                                                       "stages_order": [],
                                                       "n_classified": 0,
                                                       "n_undefined_trivial_h1": 0}},
            "ablations": {"ablations": []},
            "headline_figure": {"points": []},
        }

    print("\n[3/3] Emitting metrics.json + summary.md...")
    metrics_path = config.quick_outputs_dir / "metrics.json"
    summary_path = config.quick_outputs_dir / "summary.md"
    metrics = {
        "mode": "quick",
        "global_seed": config.global_seed,
        "env": _check_environment_preamble(),
        "hardware": _hardware_report(),
        "wall_clock_total_s": time.perf_counter() - t_start,
        "theorem_checks": theorem_checks,
        "evaluation": report,
        "n_normal": len(normals),
        "n_attacked": len(attacked),
    }
    metrics_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True, default=str, allow_nan=False),
        encoding="utf-8",
    )

    lines = [
        "# VERISHEAF --quick summary",
        "",
        f"- Global seed: `{config.global_seed}`",
        f"- Total wall clock: {metrics['wall_clock_total_s']:.2f} s",
        "",
        "## Theorem verifiers + ElGamal substitution",
        "",
    ]
    for k, v in theorem_checks.items():
        lines.append(f"- **{k}:** {'PASS' if v else 'FAIL'}")
    lines.append("")
    lines.append("## Evaluation headline")
    lines.append("")
    lines.append(
        f"- Theorem 1 verified on corpus: {report['theorem_1'].get('verified')}"
    )
    lines.append(
        f"- Theorem 2 three-way accuracy: "
        f"{report['theorem_2'].get('accuracy', float('nan')):.3f}"
    )
    lines.append(
        f"- Theorem 3 exact/approx agreement: "
        f"{report['theorem_3'].get('agreement_rate', float('nan')):.3f}"
    )
    lines.append(
        f"- Theorem 4 verified: {report['theorem_4'].get('verified')}"
    )
    lines.append(
        f"- Theorem 5 dominance rate: "
        f"{report['theorem_5'].get('overall_dominance_rate', float('nan')):.3f}"
    )
    lines.append("")
    summary_path.write_text("\n".join(lines), encoding="utf-8")

    _write_stage_log(
        stage_records, config.quick_outputs_dir / "stages.json",
        mode="quick", global_seed=config.global_seed,
    )
    print(f"\n  metrics.json -> {metrics_path}")
    print(f"  summary.md   -> {summary_path}")
    print(f"  total wall clock: {metrics['wall_clock_total_s']:.2f} s")
    n_errors = sum(1 for r in stage_records if r.status == "error")
    return 0 if n_errors == 0 else 1


# =============================================================================
# --incident subcommand
# =============================================================================


def run_incident(config: GlobalConfig) -> int:
    """Execute the eight-incident benchmark experiment.

    Builds the eight catalogued incident motifs, augments with synthetic
    normals to give the SheafLearner a non-trivial training corpus, runs
    multi-seed training, executes the full evaluation report on the
    eight-incident benchmark, and writes metrics.json + summary.md.

    Args:
        config: GlobalConfig.

    Returns:
        0 on success, 1 if any stage errored.

    Complexity: O(num_epochs * |corpus| * |E|^3).
    """
    set_global_determinism(config.global_seed)
    config.incident_outputs_dir.mkdir(parents=True, exist_ok=True)
    stage_records: list[_StageRecord] = []
    t_start = time.perf_counter()

    print("=" * 72)
    print("VERISHEAF --incident: eight-incident benchmark experiment")
    print(f"  global_seed={config.global_seed}")
    print(f"  outputs={config.incident_outputs_dir}")
    print("=" * 72)

    print("\n[1/5] Theorem verifiers + ElGamal substitution check...")
    theorem_checks = {
        "theorem_1": _run_stage(
            "incident.theorem_1", verify_theorem_1, stage_records=stage_records,
        ) or False,
        "theorem_2": _run_stage(
            "incident.theorem_2", verify_theorem_2, stage_records=stage_records,
        ) or False,
        "theorem_3": _run_stage(
            "incident.theorem_3", verify_theorem_3, stage_records=stage_records,
        ) or False,
        "theorem_4": _run_stage(
            "incident.theorem_4", lambda: verify_theorem_4(num_trials=50),
            stage_records=stage_records,
        ) or False,
        "theorem_5": _run_stage(
            "incident.theorem_5", lambda: verify_theorem_5(num_trials=10),
            stage_records=stage_records,
        ) or False,
        "elgamal_substitution": _run_stage(
            "incident.elgamal_substitution", verify_elgamal_substitution,
            stage_records=stage_records,
        ) or False,
    }
    for k, v in theorem_checks.items():
        print(f"    {k}: {'PASS' if v else 'FAIL'}")

    print("\n[2/5] Building eight-incident corpus and synthetic normals...")
    triples = _run_stage(
        "incident.assemble", assemble_incident_motifs,
        stage_records=stage_records,
    ) or []
    attacked: list[tuple[Motif, BenalohStage]] = [(m, s) for m, s, _ in triples]
    set_global_determinism(config.global_seed)
    normals: list[Motif] = [
        make_synthetic_motif(
            f"inc_norm{i}",
            dao_name=["compound", "uniswap", "aave", "ens",
                      "arbitrum", "optimism"][i % 6],
            num_voters=3 + (i % 6),
            has_record=True, has_count=True,
            seed=config.global_seed + 100 + i,
        )
        for i in range(80)
    ]
    print(f"    n_normal={len(normals)}  n_attacked={len(attacked)}")

    print("\n[3/5] Multi-seed training on (normals + attacked) corpus...")
    training_reports = _run_stage(
        "incident.training",
        multi_seed_training,
        normals + [m for m, _ in attacked],
        config.training_config,
        config.incident_outputs_dir / "checkpoints",
        stage_records=stage_records,
    ) or []
    if training_reports:
        print(f"    trained {len(training_reports)} seeds")

    print("\n[4/5] Full evaluation report...")
    report = _run_stage(
        "incident.evaluation",
        full_evaluation_report, normals, attacked,
        stage_records=stage_records,
    )
    if report is None:
        report = {
            "theorem_1": {"verified": False, "mean_h0": float("nan"),
                          "ci_lower": float("nan"), "ci_upper": float("nan")},
            "theorem_2": {"accuracy": float("nan"), "confusion_matrix": [],
                          "n_classified": 0, "n_undefined_trivial_h1": 0},
            "theorem_3": {"agreement_rate": float("nan"),
                          "median_speedup": float("nan"),
                          "rows": [], "verified": False},
            "theorem_4": {"rows": [], "verified": False},
            "theorem_5": {"overall_dominance_rate": float("nan"),
                          "verified": False, "rows": []},
            "baselines": {"binary": {},
                          "three_way": {"accuracy": float("nan"),
                                        "confusion_matrix": None,
                                        "stages_order": [],
                                        "n_classified": 0,
                                        "n_undefined_trivial_h1": 0}},
            "ablations": {"ablations": []},
            "headline_figure": {"points": []},
        }

    print("\n[5/5] Statistical analysis + emission...")
    binary_summary = report["baselines"].get("binary", {})
    baseline_auc_values = {name: vals["auc"] for name, vals in binary_summary.items()}
    verisheaf_auc = baseline_auc_values.get("verisheaf_max_subspace", float("nan"))
    other_baselines = {
        n: v for n, v in baseline_auc_values.items() if n != "verisheaf_max_subspace"
    }
    strongest = (
        max(other_baselines.items(), key=lambda kv: kv[1])
        if other_baselines else (None, None)
    )

    from evaluation import (
        baseline_edge_count_anomaly,
        baseline_temporal_incoherence,
        baseline_weight_concentration,
        verisheaf_anomaly_score,
    )
    wilcoxon_p_values: dict[str, float] = {}
    holm_results: dict = {}
    if normals and attacked:
        all_motifs_for_stat = list(normals) + [m for m, _ in attacked]
        try:
            verisheaf_scores = np.array(
                [max(verisheaf_anomaly_score(m).values()) for m in all_motifs_for_stat]
            )
            baseline_scorers = {
                "baseline_edge_count": baseline_edge_count_anomaly,
                "baseline_weight_concentration": baseline_weight_concentration,
                "baseline_temporal_incoherence": baseline_temporal_incoherence,
            }
            for name, scorer in baseline_scorers.items():
                bs = np.array([scorer(m) for m in all_motifs_for_stat])
                stat_result = paired_wilcoxon(verisheaf_scores, bs)
                wilcoxon_p_values[name] = float(stat_result.get("p_value", float("nan")))
            holm_results = holm_correct(wilcoxon_p_values, alpha=config.statistical_alpha)
        except Exception as exc:
            print(f"    statistical analysis warning: {exc}", flush=True)

    metrics_path = config.incident_outputs_dir / "metrics.json"
    summary_path = config.incident_outputs_dir / "summary.md"
    metrics = {
        "mode": "incident",
        "global_seed": config.global_seed,
        "env": _check_environment_preamble(),
        "hardware": _hardware_report(),
        "wall_clock_total_s": time.perf_counter() - t_start,
        "theorem_checks": theorem_checks,
        "training_reports": training_reports,
        "evaluation": report,
        "wilcoxon_p_values": wilcoxon_p_values,
        "holm_corrected": holm_results,
        "strongest_baseline": {"name": strongest[0], "auc": strongest[1]},
        "verisheaf_auc": verisheaf_auc,
        "n_normal": len(normals),
        "n_attacked": len(attacked),
        "n_incidents": len(triples),
    }
    metrics_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True, default=str, allow_nan=True),
        encoding="utf-8",
    )

    lines = [
        "# VERISHEAF --incident summary",
        "",
        f"- Global seed: `{config.global_seed}`",
        f"- Total wall clock: {metrics['wall_clock_total_s']:.1f} s",
        f"- Incidents: {len(triples)}",
        f"- Synthetic normals: {len(normals)}",
        "",
        "## Theorem verifiers + ElGamal substitution",
        "",
    ]
    for k, v in theorem_checks.items():
        lines.append(f"- **{k}:** {'PASS' if v else 'FAIL'}")
    lines.append("")
    lines.append("## Evaluation headline")
    lines.append("")
    lines.append(
        f"- Theorem 2 three-way accuracy: "
        f"{report['theorem_2'].get('accuracy', float('nan')):.3f} "
        f"(classified {report['theorem_2'].get('n_classified', 0)})"
    )
    lines.append(
        f"- Theorem 5 dominance rate: "
        f"{report['theorem_5'].get('overall_dominance_rate', float('nan')):.3f}"
    )
    lines.append("")
    lines.append("## Binary anomaly AUC")
    lines.append("")
    for name, vals in binary_summary.items():
        lines.append(
            f"- **{name}:** AUC={vals.get('auc', float('nan')):.3f}, "
            f"AP={vals.get('average_precision', float('nan')):.3f}"
        )
    lines.append("")
    if holm_results:
        lines.append(
            f"## Statistical significance (Holm-Bonferroni at "
            f"alpha={config.statistical_alpha:.2f})"
        )
        lines.append("")
        for name, r in holm_results.items():
            sig = "significant" if r.get("significant") else "not significant"
            lines.append(
                f"- **{name}:** p={r.get('p_value', float('nan')):.4f} ({sig})"
            )
        lines.append("")
    summary_path.write_text("\n".join(lines), encoding="utf-8")

    _write_stage_log(
        stage_records, config.incident_outputs_dir / "stages.json",
        mode="incident", global_seed=config.global_seed,
    )
    print(f"\n  metrics.json -> {metrics_path}")
    print(f"  summary.md   -> {summary_path}")
    print(f"  total wall clock: {metrics['wall_clock_total_s']:.1f} s")
    n_errors = sum(1 for r in stage_records if r.status == "error")
    return 0 if n_errors == 0 else 1


# =============================================================================
# --warehouse subcommand
# =============================================================================


def _assemble_warehouse_corpus(
    config: GlobalConfig,
) -> tuple[list[Motif], dict]:
    """Run the WarehouseAssembler against the 18072773 deposit.

    Args:
        config: GlobalConfig.

    Returns:
        (motifs, manifest) tuple. The manifest is written to
        config.warehouse_manifest_path.

    Complexity: O(sum of CSV file sizes + |motifs| * |E|).
    """
    config.warehouse_outputs_dir.mkdir(parents=True, exist_ok=True)
    assembler = WarehouseAssembler(WarehouseConfig())
    motifs, manifest = assembler.assemble(config.warehouse_data_path)
    write_warehouse_manifest(manifest, config.warehouse_manifest_path)
    return motifs, manifest


def _train_on_warehouse(
    config: GlobalConfig, motifs: list[Motif]
) -> list[dict]:
    """Run multi-seed training on the warehouse corpus.

    Args:
        config: GlobalConfig.
        motifs: validated warehouse motif corpus.

    Returns:
        list of per-seed training reports from multi_seed_training.

    Complexity: O(num_seeds * num_epochs * |motifs| * |E|^3).
    """
    checkpoint_dir = config.warehouse_outputs_dir / "checkpoints"
    reports = multi_seed_training(
        motifs, config.training_config, output_dir=checkpoint_dir,
    )
    return reports


def _best_warehouse_checkpoint(reports: list[dict]) -> Optional[Path]:
    """Return the path of the checkpoint with the lowest best_val_loss.

    Args:
        reports: list of per-seed training reports.

    Returns:
        pathlib.Path of the best checkpoint, or None when no report is present.

    Complexity: O(|reports|).
    """
    if not reports:
        return None
    best = min(reports, key=lambda r: r.get("best_val_loss", float("inf")))
    cp = best.get("checkpoint_path")
    return Path(cp) if cp else None


def run_warehouse(config: GlobalConfig) -> int:
    """Execute the principal warehouse experimental program.

    Assembles the warehouse motif corpus from the 18072773 deposit, trains the
    SheafLearner across the configured seed set with multi-GPU dispatch, then
    runs F1 through F7 in dependency order and writes the consolidated
    RESULTS_FOR_MANUSCRIPT.json. Per-experiment RUN_LOG.md fragments are
    written under 04_outputs/run_logs/.

    F2, F3, and F7 receive the baseline_epochs parameter because they train
    one or more binary baseline scorers from the four measured baselines
    (DCL-GFD, KnowGraph, NSD, ShadowEyes). F1, F4, F5, and F6 do not. The
    routing is driven by the _EXPERIMENTS_WITH_BASELINE_EPOCHS frozenset
    declared at module top, so adding a new baseline-training experiment in
    the future requires only adding its identifier to that set rather than
    editing the dispatch loop.

    Args:
        config: GlobalConfig.

    Returns:
        0 on success, 1 if any stage errored.

    Complexity: dominated by training: O(num_seeds * num_epochs * |motifs| * |E|^3).
    """
    set_global_determinism(config.global_seed)
    config.warehouse_outputs_dir.mkdir(parents=True, exist_ok=True)
    config.run_logs_dir.mkdir(parents=True, exist_ok=True)
    stage_records: list[_StageRecord] = []
    t_start = time.perf_counter()

    print("=" * 72)
    print("VERISHEAF --warehouse: principal experimental program")
    print(f"  global_seed={config.global_seed}")
    print(f"  deposit={config.warehouse_data_path}")
    print(f"  outputs={config.warehouse_outputs_dir}")
    print(f"  run logs={config.run_logs_dir}")
    print(f"  hardware={_hardware_report()}")
    print("=" * 72)

    print("\n[1/4] Warehouse motif assembly...")
    assembly_result = _run_stage(
        "warehouse.assembly",
        _assemble_warehouse_corpus, config,
        stage_records=stage_records,
    )
    if assembly_result is None:
        print(
            "  Warehouse assembly failed; cannot continue. Inspect stages.json "
            "for the diagnostic and verify deposit 18072773 is unpacked at "
            f"{config.warehouse_data_path}.",
            file=sys.stderr,
        )
        _write_stage_log(
            stage_records, config.warehouse_outputs_dir / "stages.json",
            mode="warehouse", global_seed=config.global_seed,
        )
        return 1
    motifs, manifest = assembly_result
    stage_records[-1].headline = {
        "n_motifs": len(motifs),
        "n_aragon": manifest["validated_motif_counts"]["aragon"],
        "n_daohaus": manifest["validated_motif_counts"]["daohaus"],
        "n_daostack": manifest["validated_motif_counts"]["daostack"],
    }
    stage_records[-1].output_paths = [str(config.warehouse_manifest_path)]
    print(f"    n_motifs={len(motifs)}  manifest={config.warehouse_manifest_path}")

    print("\n[2/4] Multi-seed training on the warehouse corpus...")
    training_reports = _run_stage(
        "warehouse.training",
        _train_on_warehouse, config, motifs,
        stage_records=stage_records,
    ) or []
    if training_reports:
        best_ckpt = _best_warehouse_checkpoint(training_reports)
        stage_records[-1].headline = {
            "n_seeds_trained": len(training_reports),
            "best_val_loss": min(
                r.get("best_val_loss", float("inf")) for r in training_reports
            ),
            "best_checkpoint_path": str(best_ckpt) if best_ckpt else None,
        }
        print(f"    best checkpoint: {best_ckpt}")
    else:
        best_ckpt = None
        print("    training reports empty; experiments will run under the canonical scaffold")

    # -------------------------------------------------------------------------
    # MODIFICATION 1F: F7 added to the dispatch table at seed offset
    # config.global_seed + 7, alongside F1 through F6. The dispatch loop
    # consults _EXPERIMENTS_WITH_BASELINE_EPOCHS to decide whether to forward
    # the baseline_epochs argument, so F2, F3, and F7 receive it while F1,
    # F4, F5, and F6 do not. The same dispatch loop also extracts F7's
    # cross-method consistency keys into the per-stage headline so they
    # appear in stages.json, then later in the consolidated
    # RESULTS_FOR_MANUSCRIPT.json under the headline.f7_* keys.
    # -------------------------------------------------------------------------
    print("\n[3/4] Warehouse experiments F1 through F7...")
    f_results: dict[str, dict] = {}
    experiment_dispatch: list[tuple[str, callable, int]] = [
        ("F1", warehouse_experiment_f1, config.global_seed + 1),
        ("F2", warehouse_experiment_f2, config.global_seed + 2),
        ("F3", warehouse_experiment_f3, config.global_seed + 3),
        ("F4", warehouse_experiment_f4, config.global_seed + 4),
        ("F5", warehouse_experiment_f5, config.global_seed + 5),
        ("F6", warehouse_experiment_f6, config.global_seed + 6),
        ("F7", warehouse_experiment_f7, config.global_seed + 7),
    ]
    for fname, fn, fseed in experiment_dispatch:
        print(f"\n  -> {fname} ...")
        if fname in _EXPERIMENTS_WITH_BASELINE_EPOCHS:
            out = _run_stage(
                f"warehouse.{fname}", fn,
                motifs, best_ckpt, config.warehouse_outputs_dir,
                stage_records=stage_records,
                seed=fseed, baseline_epochs=config.baseline_epochs,
            )
        else:
            out = _run_stage(
                f"warehouse.{fname}", fn,
                motifs, best_ckpt, config.warehouse_outputs_dir,
                stage_records=stage_records,
                seed=fseed,
            )
        if out is None:
            f_results[fname] = {"error": "stage_errored"}
            continue
        f_results[fname] = out
        run_log_src = config.warehouse_outputs_dir / f"{fname}_RUN_LOG.md"
        run_log_dst = config.run_logs_dir / f"{fname}_RUN_LOG.md"
        if run_log_src.exists():
            run_log_dst.parent.mkdir(parents=True, exist_ok=True)
            run_log_dst.write_bytes(run_log_src.read_bytes())
            stage_records[-1].output_paths.append(str(run_log_dst))
        # The headline subset is the per-experiment-known set of top-level
        # keys; F7 contributes cross-method consistency keys that are nested
        # under cross_method_consistency, surfaced explicitly below.
        stage_records[-1].headline = {
            k: v for k, v in out.items()
            if k in {
                "n_motifs", "n_evaluated", "overall_dominance_rate",
                "predicted_class_fractions", "counterfactual_predicted_class_fractions",
                "fpr_at_threshold", "platform_motif_counts",
            }
        }
        if fname == "F7" and isinstance(out.get("cross_method_consistency"), dict):
            cmc = out["cross_method_consistency"]
            stage_records[-1].headline.update({
                "f7_n_flagged_per_method": cmc.get("n_flagged_per_method"),
                "f7_fraction_flagged_by_all_methods":
                    cmc.get("fraction_flagged_by_all_methods"),
                "f7_fraction_intersection_with_consistent_stage_predictions":
                    cmc.get("fraction_intersection_with_consistent_stage_predictions"),
            })

    # -------------------------------------------------------------------------
    # MODIFICATION 1F (continued): consolidated RESULTS_FOR_MANUSCRIPT.json
    # gains an F7 entry alongside F1 through F6, and the headline subdict
    # gains F7 entries with the per-method flagged subset sizes, the
    # cross-method consistency metrics, and the per-method median stability
    # thresholds. Mirroring the structure emitted by
    # evaluation.warehouse_evaluation_suite ensures the orchestrator's
    # output and the in-evaluation suite's output share the same headline
    # schema, which is the property the manuscript's results table consumes.
    # -------------------------------------------------------------------------
    print("\n[4/4] Consolidating RESULTS_FOR_MANUSCRIPT.json...")
    consolidated_path = config.warehouse_outputs_dir / "RESULTS_FOR_MANUSCRIPT.json"
    f7 = f_results.get("F7", {})
    f7_cmc = f7.get("cross_method_consistency", {}) if isinstance(f7, dict) else {}
    f7_per_method = f7.get("per_method", {}) if isinstance(f7, dict) else {}
    consolidated = {
        "experiment_suite": "warehouse",
        "global_seed": config.global_seed,
        "env": _check_environment_preamble(),
        "hardware": _hardware_report(),
        "n_motifs_validated": len(motifs),
        "warehouse_manifest_path": str(config.warehouse_manifest_path),
        "best_checkpoint_path": str(best_ckpt) if best_ckpt else None,
        "training_reports": training_reports,
        "wall_clock_total_s": time.perf_counter() - t_start,
        "F1": f_results.get("F1", {}),
        "F2": f_results.get("F2", {}),
        "F3": f_results.get("F3", {}),
        "F4": f_results.get("F4", {}),
        "F5": f_results.get("F5", {}),
        "F6": f_results.get("F6", {}),
        "F7": f_results.get("F7", {}),
        "headline": {
            "n_motifs": len(motifs),
            "f1_predicted_class_fractions":
                f_results.get("F1", {}).get("predicted_class_fractions", {}),
            "f2_per_method_fpr_at_0p50": {
                name: per.get("fpr_at_threshold", {}).get("0.50", float("nan"))
                for name, per in f_results.get("F2", {}).get("per_method", {}).items()
            },
            "f3_platform_motif_counts":
                f_results.get("F3", {}).get("platform_motif_counts", {}),
            "f4_overall_dominance_rate":
                f_results.get("F4", {}).get("overall_dominance_rate", float("nan")),
            "f5_counterfactual_predicted_class_fractions":
                f_results.get("F5", {}).get("counterfactual_predicted_class_fractions", {}),
            "f6_fpr_at_0p50":
                f_results.get("F6", {}).get("fpr_at_threshold", {}).get("0.50", float("nan")),
            # F7 headline numbers: per-method flagged subset sizes, dominant
            # predicted stage fractions, median stability thresholds, and
            # cross-method consistency metrics. These are the four families
            # of headline numbers Modification 1F's acceptance criterion
            # enumerates; they appear here in the same byte-deterministic
            # JSON order under sort_keys=True.
            "f7_n_flagged_per_method":
                f7_cmc.get("n_flagged_per_method", {}),
            "f7_fraction_flagged_by_all_methods":
                f7_cmc.get("fraction_flagged_by_all_methods", float("nan")),
            "f7_fraction_intersection_with_consistent_stage_predictions":
                f7_cmc.get("fraction_intersection_with_consistent_stage_predictions",
                           float("nan")),
            "f7_per_method_dominant_predicted_stage_fraction": {
                method: max(
                    per.get("stage_prediction_fractions", {}).values(), default=0.0,
                )
                for method, per in f7_per_method.items()
            },
            "f7_per_method_median_stability_threshold": {
                method: per.get("median_stability_threshold")
                for method, per in f7_per_method.items()
            },
        },
    }
    consolidated_path.write_text(
        json.dumps(consolidated, indent=2, sort_keys=True, default=str, allow_nan=False),
        encoding="utf-8",
    )

    _write_stage_log(
        stage_records, config.warehouse_outputs_dir / "stages.json",
        mode="warehouse", global_seed=config.global_seed,
    )
    print(f"\n  RESULTS_FOR_MANUSCRIPT.json -> {consolidated_path}")
    print(f"  total wall clock: {consolidated['wall_clock_total_s']:.1f} s")
    n_errors = sum(1 for r in stage_records if r.status == "error")
    return 0 if n_errors == 0 else 1


# =============================================================================
# --verify subcommand
# =============================================================================


def run_verify(config: GlobalConfig) -> int:
    """Load the warehouse manifest and print the one-glance corpus summary.

    Args:
        config: GlobalConfig.

    Returns:
        0 if the manifest is present and well-formed, 1 otherwise.

    Complexity: O(|manifest|).
    """
    print("=" * 72)
    print("VERISHEAF --verify: corpus sanity check")
    print(f"  manifest={config.warehouse_manifest_path}")
    print("=" * 72)
    path = Path(config.warehouse_manifest_path)
    if not path.is_file():
        print(
            f"MISSING manifest: {path}. Run --warehouse first to produce it.",
            file=sys.stderr,
        )
        return 1
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"MALFORMED manifest: {exc}", file=sys.stderr)
        return 1

    print("\nDataset provenance:")
    print(f"  source:  {manifest.get('dataset_source')}")
    print(f"  version: {manifest.get('dataset_version')}")
    print(f"  extracted: {manifest.get('extraction_date')}")
    print(f"  ethereum anchor block: {manifest.get('ethereum_anchor_block')}")
    print(f"  xdai anchor block:     {manifest.get('xdai_anchor_block')}")

    print("\nFile metadata:")
    fm = manifest.get("file_metadata", {})
    for rel_path in sorted(fm.keys()):
        m = fm[rel_path]
        print(
            f"  {rel_path:32s}  n_rows={m.get('n_rows'):>7,d}  "
            f"sha256={m.get('sha256', '')[:16]}..."
        )

    print("\nPer-platform motif counts:")
    raw = manifest.get("raw_proposal_counts", {})
    val = manifest.get("validated_motif_counts", {})
    exp = manifest.get("expected_motif_counts", {})
    for platform in ("aragon", "daohaus", "daostack"):
        print(
            f"  {platform:10s}  raw={raw.get(platform, 0):>6,d}  "
            f"validated={val.get(platform, 0):>6,d}  "
            f"expected={exp.get(platform, 0):>6,d}"
        )
    print(f"  TOTAL    validated={manifest.get('total_validated_motifs', 0):>6,d}")

    print("\nValidation failures (per platform, per criterion):")
    failures = manifest.get("validation_failures", {})
    for platform in ("aragon", "daohaus", "daostack"):
        pf = failures.get(platform, {})
        print(
            f"  {platform:10s}  n_c<2={pf.get('n_c_lt_2', 0):>5,d}  "
            f"cast_weight_nonpositive={pf.get('cast_weight_nonpositive', 0):>5,d}  "
            f"n_r<1={pf.get('n_r_lt_1', 0):>5,d}  "
            f"n_q!=1={pf.get('n_q_ne_1', 0):>5,d}  "
            f"total_excluded={pf.get('total_excluded', 0):>5,d}"
        )

    print("\nEight-incident cross-reference:")
    xr = manifest.get("incident_cross_reference", {})
    print(
        f"  n_motifs={xr.get('n_motifs', 0):>6,d}  "
        f"n_matches={xr.get('n_matches', 0)}  "
        f"expected={xr.get('expected_matches', 0)}"
    )

    print("\nCORPUS OK." if val else "\nCORPUS MISSING.")
    return 0


# =============================================================================
# CLI dispatch
# =============================================================================


def _build_parser() -> argparse.ArgumentParser:
    """Construct the argparse CLI dispatch table for the four subcommands.

    Returns:
        argparse.ArgumentParser with --quick, --incident, --warehouse, --verify.

    Complexity: O(1).
    """
    parser = argparse.ArgumentParser(
        prog="reproduce.py",
        description="VERISHEAF end-to-end reproduction orchestrator.",
    )
    parser.add_argument(
        "--global-seed", type=int, default=20260601,
        help="Master seed; per-experiment seeds derive from it deterministically.",
    )
    parser.add_argument(
        "--outputs-root", type=Path, default=Path("04_outputs"),
        help="Root directory for all generated artefacts.",
    )
    parser.add_argument(
        "--warehouse-data", type=Path, default=Path("02_data/18072773"),
        help="Path to the unpacked 18072773 deposit root.",
    )
    parser.add_argument(
        "--baseline-epochs", type=int, default=30,
        help=(
            "Per-baseline training epochs for the four measured baselines "
            "(DCL-GFD, KnowGraph, NSD, ShadowEyes). Applied uniformly in F2, "
            "F3, and F7."
        ),
    )

    sub = parser.add_subparsers(dest="mode", required=True, metavar="MODE")
    sub.add_parser("quick", help="Smoke-test against a small synthetic corpus.")
    sub.add_parser("incident", help="Eight-incident benchmark experiment.")
    sub.add_parser("warehouse", help="Principal warehouse experimental program.")
    sub.add_parser("verify", help="Sanity-check the warehouse manifest.")
    return parser


def _config_from_args(args: argparse.Namespace) -> GlobalConfig:
    """Build a GlobalConfig from the parsed CLI arguments.

    Args:
        args: argparse.Namespace from _build_parser().parse_args().

    Returns:
        GlobalConfig with derived path attributes set under outputs_root.

    Complexity: O(1).
    """
    root = Path(args.outputs_root)
    cfg = GlobalConfig(
        global_seed=int(args.global_seed),
        outputs_root=root,
        run_logs_dir=root / "run_logs",
        warehouse_data_path=Path(args.warehouse_data),
        warehouse_motifs_cache=root / "warehouse" / "motifs_cache.json",
        warehouse_manifest_path=root / "warehouse" / "manifest.json",
        incident_outputs_dir=root / "incident",
        warehouse_outputs_dir=root / "warehouse",
        quick_outputs_dir=root / "quick",
        baseline_epochs=int(args.baseline_epochs),
    )
    cfg.training_config.checkpoint_dir = cfg.outputs_root / "checkpoints"
    return cfg


def main(argv: Optional[list[str]] = None) -> int:
    """Dispatch the four subcommands.

    Args:
        argv: optional argv override (used by the self-test).

    Returns:
        process exit code: 0 on success, 1 on any stage error.

    Complexity: dominated by the dispatched subcommand.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)
    config = _config_from_args(args)
    set_global_determinism(config.global_seed)

    print(
        f"VERISHEAF reproduce.py mode={args.mode} seed={config.global_seed}",
        flush=True,
    )
    env = _check_environment_preamble()
    if env["PYTHONHASHSEED"] == "unset" or env["CUBLAS_WORKSPACE_CONFIG"] == "unset":
        print(
            "  WARNING: PYTHONHASHSEED and/or CUBLAS_WORKSPACE_CONFIG were not "
            "set before Python launched; full byte-equal reproducibility requires "
            "setting them in the environment per the README preamble.",
            flush=True,
        )

    if args.mode == "quick":
        return run_quick(config)
    if args.mode == "incident":
        return run_incident(config)
    if args.mode == "warehouse":
        return run_warehouse(config)
    if args.mode == "verify":
        return run_verify(config)
    parser.error(f"Unknown mode {args.mode!r}")
    return 2


# =============================================================================
# Module self-test
# =============================================================================


if __name__ == "__main__":
    sys.exit(main())
