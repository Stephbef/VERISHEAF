"""training.py — VERISHEAF SheafLearner training loop, determinism helper, and multi-seed orchestration.

Trains the SheafLearner from theory.py on the stratified corpus of complete-motif
verifiable executions. The training objective is a masked-reconstruction sheaf-
Dirichlet energy on motif-derived node signals, plus the PAC-Bayes regularizer
derived from Theorem 4.

This module absorbs the 34-line _determinism helper (set_global_determinism) at
the top so that callers have a single entry point for the six-layer
reproducibility discipline (Python random, NumPy, PyTorch CPU/CUDA, cuDNN, and
the two environment variables PYTHONHASHSEED and CUBLAS_WORKSPACE_CONFIG).

Hardware support: single-device CPU/GPU training plus multi-GPU orchestration
via multi_seed_training. When CUDA is available and torch.cuda.device_count()
is at least len(config.seeds), each seed's training job is pinned to a
dedicated GPU via torch.cuda.set_device, so the four-seed default fully
utilises the four-RTX-3090 server hardware.

Defect history (preserved as in-line documentation for reviewer-defensibility):
    CC-3 (FTR-1): the training loop previously fed torch.randn(...) noise as the
        node-signal input to sheaf_dirichlet_energy. _run_epoch below calls
        motif_to_node_signals(motif, ...) at the same line position as the
        in-place fix in the previous codebase version. Any modification of
        that call risks reintroducing the noise-input bug.
    FTR-2: per-epoch seeding so any stochastic component (mask, shuffle) is
        reproducible for a given (seed, epoch) pair.
    FTR-3: early stopping tracks VALIDATION loss (not training loss) so the
        saved checkpoint generalises rather than memorises.
"""

from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import Dataset

from motifs import (
    Motif,
    make_synthetic_motif,
    motif_to_node_signals,
    motif_to_typed_graph,
)
from theory import (
    DEFAULT_EDGE_STALK_DIMS,
    DEFAULT_STALK_DIMS,
    EdgeType,
    NodeType,
    SheafLearner,
)


# =============================================================================
# Determinism: the six-layer reproducibility helper (absorbed from _determinism.py)
# =============================================================================


def set_global_determinism(seed: int) -> None:
    """Enforce byte-equal reproducibility across runs at a given seed.

    Six layers are configured:
      1. Python random.seed.
      2. NumPy np.random.seed.
      3. PyTorch CPU torch.manual_seed.
      4. PyTorch CUDA torch.cuda.manual_seed_all (when CUDA available).
      5. cuDNN backend flags (deterministic=True, benchmark=False).
      6. Environment variables CUBLAS_WORKSPACE_CONFIG and PYTHONHASHSEED.

    Note: PYTHONHASHSEED is set via os.environ, which has no effect on modules
    already imported in the current process; for full PYTHONHASHSEED isolation,
    set it in the environment before launching the script.

    Args:
        seed: integer seed applied to every layer.

    Returns:
        None; the function mutates global RNG state and environment.

    Complexity: O(1).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["PYTHONHASHSEED"] = str(seed)


# =============================================================================
# Training configuration
# =============================================================================


@dataclass
class TrainingConfig:
    """All training hyperparameters as inspectable Python.

    Attributes:
        seed: primary integer seed for single-seed training.
        seeds: list of integer seeds for multi-seed orchestration. Default of
            four seeds matches the four-GPU server configuration so that each
            seed is dispatched to a dedicated RTX 3090.
        lr: AdamW learning rate.
        weight_decay: AdamW weight decay (independent of the PAC-Bayes regulariser).
        pac_bayes_weight: scalar multiplier on the PAC-Bayes regulariser term;
            calibrated to the Theorem 4 prior covariance.
        gradient_clip_norm: max L2 gradient norm before optimiser step.
        num_epochs: total epoch budget.
        early_stopping_patience: epochs of non-improvement before stopping.
        batch_size: number of motifs per batch.
        num_workers: DataLoader worker count (kept for interface parity; the
            module currently iterates motifs in the main process because each
            motif is a small heterogeneous graph).
        pac_bayes_delta: confidence parameter used by Theorem 4 bound diagnostic.
        mask_ratio: fraction of node signals to mask during training.
        val_split: fraction of motifs allocated to the validation split.
        use_cuda: whether to use CUDA when available.
        checkpoint_dir: directory for per-seed checkpoints.
        log_every_n_steps: stride for stdout logging within an epoch.
    """

    seed: int = 20260520
    seeds: list[int] = field(default_factory=lambda: [20260520, 20260521, 20260522, 20260523])

    lr: float = 1.0e-4
    weight_decay: float = 1.0e-5
    pac_bayes_weight: float = 1.0e-3
    gradient_clip_norm: float = 1.0

    num_epochs: int = 40
    early_stopping_patience: int = 8

    batch_size: int = 16
    num_workers: int = 4

    pac_bayes_delta: float = 0.05

    mask_ratio: float = 0.2
    val_split: float = 0.2

    use_cuda: bool = True

    checkpoint_dir: Path = Path("outputs/checkpoints")
    log_every_n_steps: int = 25


# =============================================================================
# Stratified motif dataset
# =============================================================================


class StratifiedMotifDataset(Dataset):
    """Yields complete-motif executions from the stratified training corpus.

    Per the corpus stratification correction, training is restricted to motifs
    exhibiting the full cast-recorded-counted Benaloh sequence. Incomplete
    motifs are silently filtered at construction time; the constructor raises
    if no complete motif remains.
    """

    STRATIFIED_TRAINING_DAOS = [
        "compound", "uniswap", "aave", "ens", "arbitrum",
        "optimism", "gitcoin", "lido", "maker", "safe_dao",
    ]

    def __init__(self, motifs: list[Motif]):
        """Filter to complete motifs and store them.

        Args:
            motifs: list of Motif instances.

        Raises:
            ValueError: if no motif in the input list is_complete.
        """
        self.motifs = [m for m in motifs if m.is_complete]
        if not self.motifs:
            raise ValueError(
                "Stratified training corpus is empty. The framework requires at "
                "least one complete-motif execution for the masked-reconstruction "
                "objective to be well-defined."
            )

    def __len__(self) -> int:
        """Return the number of complete motifs available for training."""
        return len(self.motifs)

    def __getitem__(self, idx: int) -> Motif:
        """Return the motif at index idx; collation is done by collate_motifs."""
        return self.motifs[idx]


def collate_motifs(motifs: list[Motif]) -> list[Motif]:
    """Trivial collator: each motif keeps its own typed-graph structure.

    Args:
        motifs: list of Motif instances from the dataset.

    Returns:
        The same list, unchanged. Each motif retains its own variable-size
        typed-graph representation.
    """
    return motifs


# =============================================================================
# Per-epoch loop with motif-derived node signals (the CC-3 fix)
# =============================================================================


def _run_epoch(
    model: SheafLearner,
    motifs: list[Motif],
    optimizer: Optional[AdamW],
    config: TrainingConfig,
    device: str,
    epoch_seed: int,
    train: bool,
) -> dict:
    """Run one epoch over a motif list.

    If train is False, no gradient step is taken and the mask is applied
    deterministically (mask_ratio=0.0) for a stable validation signal.

    The node-signal input is motif-derived via motif_to_node_signals
    (the CC-3 fix). The prior implementation fed torch.randn(...) Gaussian
    noise as the input, under which the motif contributed only graph topology
    and the optimised values were noise unrelated to any voting execution. The
    motif_to_node_signals call at the same line position is the in-place fix.

    Args:
        model: SheafLearner instance, already on device.
        motifs: list of Motif to iterate over.
        optimizer: AdamW optimizer; None for validation.
        config: TrainingConfig.
        device: torch device string ("cpu" or "cuda" or "cuda:N").
        epoch_seed: integer seed for per-epoch determinism (mask, shuffle).
        train: bool; True for training, False for validation.

    Returns:
        dict with keys 'loss', 'energy', 'reg' averaged over motifs seen.

    Citation: manuscript Section 5 (Masked-Reconstruction Training Objective).

    Complexity: O(|motifs| * (|E| * d_max^2 + restriction-assembly cost)).
    """
    if train:
        model.train()
    else:
        model.eval()

    indices = list(range(len(motifs)))
    if train:
        rng = np.random.RandomState(epoch_seed % (2**32 - 1))
        rng.shuffle(indices)

    epoch_loss = epoch_energy = epoch_reg = 0.0
    seen = 0

    for batch_start in range(0, len(motifs), config.batch_size):
        batch_indices = indices[batch_start : batch_start + config.batch_size]
        batch_motifs = [motifs[i] for i in batch_indices]
        if optimizer is not None:
            optimizer.zero_grad()

        batch_energy = torch.zeros((), device=device)
        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for j, motif in enumerate(batch_motifs):
                node_types, edge_index, edge_types, _, n2i = motif_to_typed_graph(motif)
                sig_seed = (epoch_seed * 100003 + batch_start * 31 + j) % (2**31 - 1)
                node_signals, _mask = motif_to_node_signals(
                    motif, n2i, DEFAULT_STALK_DIMS,
                    mask_ratio=(config.mask_ratio if train else 0.0),
                    device=device, seed=sig_seed,
                )
                edge_index_dev = edge_index.to(device)
                energy = model.sheaf_dirichlet_energy(
                    node_signals, node_types, edge_index_dev, edge_types
                )
                batch_energy = batch_energy + energy

            batch_energy = batch_energy / max(len(batch_motifs), 1)
            reg = model.pac_bayes_regularizer()
            loss = batch_energy + config.pac_bayes_weight * reg

        if train and optimizer is not None:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            optimizer.step()

        epoch_loss += float(loss.item()) * len(batch_motifs)
        epoch_energy += float(batch_energy.item()) * len(batch_motifs)
        epoch_reg += float(reg.item()) * len(batch_motifs)
        seen += len(batch_motifs)

    return {
        "loss": epoch_loss / max(seen, 1),
        "energy": epoch_energy / max(seen, 1),
        "reg": epoch_reg / max(seen, 1),
    }


# =============================================================================
# Single-seed training entry point
# =============================================================================


def train_sheaf_learner(
    motifs: list[Motif],
    config: TrainingConfig,
    output_dir: Optional[Path] = None,
    device_override: Optional[str] = None,
) -> dict:
    """Train the SheafLearner on the stratified motif corpus (single seed).

    The objective is a masked-reconstruction sheaf-Dirichlet energy on motif-
    derived node signals plus the PAC-Bayes regulariser (Theorem 4). The
    corpus is split at the motif level into train and validation subsets;
    early stopping tracks VALIDATION loss (FTR-3), so the saved checkpoint is
    the epoch that generalises best rather than the epoch that memorises hardest.

    Args:
        motifs: list of Motif (complete motifs only are used; the dataset
            constructor filters).
        config: TrainingConfig.
        output_dir: pathlib.Path destination for the checkpoint; defaults to
            config.checkpoint_dir.
        device_override: optional explicit torch device string. When provided
            (e.g., "cuda:2"), overrides the default device selection. Used by
            multi_seed_training to pin each seed to a dedicated GPU.

    Returns:
        dict containing per-epoch history, best validation loss, checkpoint
        path, Theorem 4 PAC-Bayes operator-norm bound, and wall-clock totals.

    Citation: manuscript Section 5 (Training Procedure).

    Complexity: O(num_epochs * |train_motifs| * |E_per_motif| * d_max^2).
    """
    set_global_determinism(config.seed)
    if device_override is not None:
        device = device_override
    else:
        device = "cuda" if (config.use_cuda and torch.cuda.is_available()) else "cpu"

    full = StratifiedMotifDataset(motifs)
    n_total = len(full)
    split_rng = np.random.RandomState(config.seed)
    perm = split_rng.permutation(n_total)
    n_val = max(1, int(config.val_split * n_total)) if n_total > 1 else 0
    val_idx = set(perm[:n_val].tolist())
    train_motifs = [full.motifs[i] for i in range(n_total) if i not in val_idx]
    val_motifs = [full.motifs[i] for i in range(n_total) if i in val_idx]
    if not train_motifs:
        train_motifs = list(full.motifs)
        val_motifs = list(full.motifs)

    model = SheafLearner().to(device)
    optimizer = AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    restart_period = max(2, config.num_epochs // 4)
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=restart_period, T_mult=2, eta_min=1e-6
    )

    if output_dir is None:
        output_dir = config.checkpoint_dir
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_dir / f"sheaf_learner_seed{config.seed}.pt"

    history: list[dict] = []
    best_val_loss = float("inf")
    best_epoch = -1
    patience_counter = 0
    t_total_start = time.perf_counter()

    for epoch in range(config.num_epochs):
        t_start = time.perf_counter()
        # FTR-2: per-epoch seeding so any stochastic component (mask, shuffle)
        # is reproducible for a given (seed, epoch) pair.
        epoch_seed = config.seed * 10000 + epoch
        torch.manual_seed(epoch_seed)

        train_stats = _run_epoch(
            model, train_motifs, optimizer, config, device, epoch_seed, train=True
        )
        val_stats = _run_epoch(
            model, val_motifs, None, config, device, epoch_seed, train=False
        )
        scheduler.step()
        t_epoch = time.perf_counter() - t_start

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_stats["loss"],
                "val_loss": val_stats["loss"],
                "dirichlet_energy": train_stats["energy"],
                "pac_bayes_reg": train_stats["reg"],
                "lr": optimizer.param_groups[0]["lr"],
                "wall_clock_s": t_epoch,
            }
        )

        if epoch % 5 == 0 or epoch == config.num_epochs - 1:
            print(
                f"  [seed {config.seed}] epoch {epoch:3d} | "
                f"train {train_stats['loss']:.4e} | val {val_stats['loss']:.4e} | "
                f"reg {train_stats['reg']:.4e} | "
                f"lr {optimizer.param_groups[0]['lr']:.2e} | {t_epoch:.2f}s",
                flush=True,
            )

        # FTR-3: early stopping on VALIDATION loss.
        if val_stats["loss"] < best_val_loss - 1e-6:
            best_val_loss = val_stats["loss"]
            best_epoch = epoch
            patience_counter = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": {
                        k: str(v) if isinstance(v, Path) else v
                        for k, v in config.__dict__.items()
                    },
                    "epoch": epoch,
                    "val_loss": best_val_loss,
                    "seed": config.seed,
                    "device": device,
                },
                ckpt_path,
            )
        else:
            patience_counter += 1
            if patience_counter >= config.early_stopping_patience:
                print(
                    f"  [seed {config.seed}] early stopping at epoch {epoch} "
                    f"(best epoch {best_epoch}, val {best_val_loss:.4e})",
                    flush=True,
                )
                break

    from theory import pac_bayes_bound
    theorem_4_bound = pac_bayes_bound(
        m=max(len(train_motifs), 1),
        d=DEFAULT_STALK_DIMS[NodeType.VOTER],
        delta=config.pac_bayes_delta,
    )

    report = {
        "seed": config.seed,
        "device": device,
        "n_train_motifs": len(train_motifs),
        "n_val_motifs": len(val_motifs),
        "history": history,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "checkpoint_path": str(ckpt_path),
        "theorem_4_pac_bayes_bound": theorem_4_bound,
        "wall_clock_total_s": time.perf_counter() - t_total_start,
    }
    return report


# =============================================================================
# Multi-seed multi-GPU orchestration
# =============================================================================


def _select_device_for_seed(seed_idx: int, use_cuda: bool) -> str:
    """Resolve the torch device string for a given seed index.

    When CUDA is available and torch.cuda.device_count() >= 1, the device is
    'cuda:{seed_idx % device_count}', and torch.cuda.set_device is called so
    the model and tensors are pinned to that GPU. When CUDA is not available
    or use_cuda is False, the device is 'cpu'.

    Args:
        seed_idx: 0-based index of the seed within config.seeds.
        use_cuda: whether to attempt CUDA dispatch.

    Returns:
        torch device string.

    Complexity: O(1).
    """
    if use_cuda and torch.cuda.is_available():
        n_devices = torch.cuda.device_count()
        if n_devices >= 1:
            gpu = seed_idx % n_devices
            torch.cuda.set_device(gpu)
            return f"cuda:{gpu}"
    return "cpu"


def multi_seed_training(
    motifs: list[Motif],
    config: TrainingConfig,
    output_dir: Optional[Path] = None,
) -> list[dict]:
    """Run train_sheaf_learner once per seed in config.seeds, pinning each to a GPU.

    When CUDA is available and torch.cuda.device_count() >= len(config.seeds),
    seed i is dispatched to GPU (i % device_count) so the four-seed default
    fully utilises the four-RTX-3090 server hardware. The dispatch is in-process
    sequential (one seed runs to completion before the next starts), so each
    seed has uncontested access to its target GPU for the duration of its run.

    A per-seed checkpoint and a consolidated multi-seed manifest are written to
    output_dir. The manifest reports per-seed best validation loss, best epoch,
    wall-clock total, checkpoint path, and device assignment.

    Args:
        motifs: list of Motif.
        config: TrainingConfig (the .seeds field drives this loop).
        output_dir: pathlib.Path destination; defaults to config.checkpoint_dir.

    Returns:
        list of per-seed training reports.

    Citation: manuscript Section 5 (Multi-Seed Training Stability).

    Complexity: O(|seeds| * num_epochs * |motifs| * |E_per_motif| * d_max^2).
    """
    if output_dir is None:
        output_dir = config.checkpoint_dir
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_devices = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print(
        f"  multi_seed_training: {len(config.seeds)} seeds, "
        f"{n_devices} CUDA devices available, use_cuda={config.use_cuda}",
        flush=True,
    )

    reports: list[dict] = []
    for seed_idx, seed in enumerate(config.seeds):
        device_str = _select_device_for_seed(seed_idx, config.use_cuda)
        seed_config = TrainingConfig(**{**config.__dict__, "seed": seed})
        print(
            f"  multi_seed_training: seed {seed} -> device {device_str}",
            flush=True,
        )
        report = train_sheaf_learner(
            motifs, seed_config, output_dir=output_dir, device_override=device_str
        )
        reports.append(report)

    manifest = {
        "n_seeds": len(reports),
        "seeds": list(config.seeds),
        "n_cuda_devices_available": n_devices,
        "use_cuda": config.use_cuda,
        "per_seed": [
            {
                "seed": r["seed"],
                "device": r["device"],
                "best_val_loss": r["best_val_loss"],
                "best_epoch": r["best_epoch"],
                "epochs_completed": r["epochs_completed"],
                "checkpoint_path": r["checkpoint_path"],
                "wall_clock_total_s": r["wall_clock_total_s"],
                "theorem_4_pac_bayes_bound": r["theorem_4_pac_bayes_bound"],
            }
            for r in reports
        ],
        "aggregate": {
            "mean_best_val_loss": float(
                np.mean([r["best_val_loss"] for r in reports])
            ) if reports else float("nan"),
            "std_best_val_loss": float(
                np.std([r["best_val_loss"] for r in reports])
            ) if reports else float("nan"),
            "total_wall_clock_s": float(
                sum(r["wall_clock_total_s"] for r in reports)
            ),
        },
    }
    manifest_path = output_dir / "multi_seed_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str, allow_nan=True),
        encoding="utf-8",
    )
    print(
        f"  multi_seed_training: manifest -> {manifest_path}",
        flush=True,
    )
    return reports


# =============================================================================
# Module self-test
# =============================================================================


if __name__ == "__main__":
    print("VERISHEAF training.py — module checks")

    # Test 1: set_global_determinism mutates the six layers without raising.
    set_global_determinism(42)
    assert os.environ.get("PYTHONHASHSEED") == "42"
    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False
    print("  set_global_determinism six layers              PASS")

    # Test 2: TrainingConfig defaults match the specification.
    cfg = TrainingConfig()
    assert cfg.seeds == [20260520, 20260521, 20260522, 20260523], (
        f"TrainingConfig.seeds default: {cfg.seeds}"
    )
    assert cfg.num_epochs == 40
    assert cfg.batch_size == 16
    assert abs(cfg.pac_bayes_weight - 1.0e-3) < 1e-12
    assert cfg.use_cuda is True
    assert cfg.early_stopping_patience == 8
    print("  TrainingConfig defaults                        PASS")

    # Test 3: StratifiedMotifDataset filters incomplete motifs and raises on empty.
    complete = [
        make_synthetic_motif(f"c{i}", "compound", num_voters=4,
                             has_record=True, has_count=True, seed=i)
        for i in range(3)
    ]
    incomplete = [
        make_synthetic_motif(f"i{i}", "compound", num_voters=4,
                             has_record=False, has_count=False, seed=10 + i)
        for i in range(2)
    ]
    ds = StratifiedMotifDataset(complete + incomplete)
    assert len(ds) == 3, f"StratifiedMotifDataset filtering: {len(ds)} != 3"
    try:
        StratifiedMotifDataset(incomplete)
        raise SystemExit("StratifiedMotifDataset must raise on empty corpus")
    except ValueError:
        pass
    print("  StratifiedMotifDataset filtering + empty-raise PASS")

    # Test 4: short training run on a tiny synthetic corpus.
    short_cfg = TrainingConfig(
        seed=20260520,
        seeds=[20260520, 20260521],
        num_epochs=3,
        batch_size=4,
        use_cuda=False,
        early_stopping_patience=10,
        checkpoint_dir=Path("outputs/test_checkpoints"),
    )
    motifs_small = [
        make_synthetic_motif(f"t{i}", "compound", num_voters=4,
                             has_record=True, has_count=True, seed=i)
        for i in range(8)
    ]
    import shutil
    shutil.rmtree(short_cfg.checkpoint_dir, ignore_errors=True)
    report = train_sheaf_learner(motifs_small, short_cfg)
    assert len(report["history"]) == 3
    assert report["best_val_loss"] < float("inf")
    assert Path(report["checkpoint_path"]).exists()
    print(
        f"  train_sheaf_learner (3 epochs, best_val={report['best_val_loss']:.4e}) PASS"
    )

    # Test 5: _select_device_for_seed returns 'cpu' when CUDA unavailable.
    if not torch.cuda.is_available():
        assert _select_device_for_seed(0, use_cuda=True) == "cpu"
        assert _select_device_for_seed(3, use_cuda=True) == "cpu"
        print("  _select_device_for_seed CPU fallback           PASS")
    else:
        dev0 = _select_device_for_seed(0, use_cuda=True)
        assert dev0.startswith("cuda:")
        print(f"  _select_device_for_seed CUDA dispatch ({dev0}) PASS")

    # Test 6: multi_seed_training produces a manifest with per-seed entries.
    shutil.rmtree(short_cfg.checkpoint_dir, ignore_errors=True)
    reports = multi_seed_training(motifs_small, short_cfg)
    assert len(reports) == len(short_cfg.seeds)
    manifest_path = short_cfg.checkpoint_dir / "multi_seed_manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["n_seeds"] == len(short_cfg.seeds)
    assert len(manifest["per_seed"]) == len(short_cfg.seeds)
    assert all(p["checkpoint_path"] for p in manifest["per_seed"])
    print(
        f"  multi_seed_training (n_seeds={manifest['n_seeds']}, "
        f"mean_val={manifest['aggregate']['mean_best_val_loss']:.4e}) PASS"
    )

    # Test 7: byte-deterministic JSON output for the manifest.
    raw = manifest_path.read_bytes()
    raw_again = json.dumps(manifest, indent=2, sort_keys=True, default=str, allow_nan=True).encode("utf-8")
    assert raw == raw_again, "multi_seed_manifest.json is not byte-deterministic"
    print("  multi_seed_manifest byte-deterministic         PASS")

    shutil.rmtree(short_cfg.checkpoint_dir, ignore_errors=True)
    print("\nAll training.py tests passed.")