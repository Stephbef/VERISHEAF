# VERISHEAF

**Cellular Sheaf Cohomology for End-to-End Verifiability of Blockchain-Based Voting Governance**

VERISHEAF is a framework for stage-attributed anomaly detection on typed temporal governance graphs. It reduces end-to-end verifiability of DAO governance executions to the vanishing of the first cohomology $H^1(F_G) = 0$ of a cellular sheaf whose restriction maps are instantiated from exponential ElGamal encryption. The three Benaloh stages — cast, recorded, counted — correspond to pairwise-orthogonal subspaces of $H^1(F_G)$ by construction, and Theorem 5 supplies an explicit per-motif stability threshold below which the stage attribution is provably stable under restriction-map perturbation.

The framework operates as a complementary stage-attribution layer atop existing binary anomaly detectors (DCL-GFD, KnowGraph, NSD, ShadowEyes), supplying the cryptographic-stage classification that binary detectors lack.

---

## Repository Structure

```
verisheaf/
├── theory.py            # Core mathematical framework: five theorems, corollary,
│                        # sheaf construction, Laplacian, SheafLearner neural module
├── cryptography.py      # Exponential ElGamal restriction-map registry with
│                        # stage-disjoint column support enforcement
├── motifs.py            # Typed temporal motif schema, sheaf converters,
│                        # eight-incident catalogue, synthetic motif generators
├── assemblers.py        # Warehouse ingestion (DAO Analyzer deposit) and
│                        # eight-incident benchmark motif reconstruction
├── training.py          # SheafLearner training loop, six-layer determinism
│                        # helper, multi-seed multi-GPU orchestration
├── baselines.py         # Four baseline adapters: DCL-GFD, KnowGraph, NSD,
│                        # ShadowEyes with shared stage-routing reduction
├── evaluation.py        # Seven warehouse experiments (F1–F7), theorem validators,
│                        # bootstrap CI, Holm–Bonferroni correction
├── reproduce.py         # CLI orchestrator: --quick, --incident, --warehouse, --verify
├── requirements.txt     # Python dependencies with pinned versions
├── README.md            # This file
└── 02_data/
    └── 18072773/        # Unpacked DAO Analyzer Zenodo deposit (see Data Setup below)
        ├── aragon/
        │   ├── organizations.csv
        │   ├── votes.csv
        │   └── casts.csv
        ├── daohaus/
        │   ├── moloches.csv
        │   ├── proposals.csv
        │   └── votes.csv
        ├── daostack/
        │   ├── daos.csv
        │   ├── proposals.csv
        │   └── votes.csv
        ├── version.txt
        └── update_date.txt
```

All generated artefacts are written to `04_outputs/` under mode-specific subdirectories. The orchestrator never mutates source data.

---

## Data Setup

The warehouse experiments consume the **DAO Analyzer** dataset hosted on Zenodo.

**Citation:**
> J. Arroyo, D. Davó, and Y. Faqir-Rhazoui, "DAO Analyzer dataset," *Zenodo*, version 2025-12-28, Dec. 2025, doi: [10.5281/zenodo.18072773](https://doi.org/10.5281/zenodo.18072773).

**Download and extraction:**

```bash
# Download the deposit archive from Zenodo
wget https://zenodo.org/records/18072773/files/archive.zip -O archive.zip

# Create the expected directory and extract
mkdir -p 02_data/18072773
unzip archive.zip -d 02_data/18072773
```

After extraction the directory `02_data/18072773/` must contain the three platform subdirectories (`aragon/`, `daohaus/`, `daostack/`) and the version metadata files. The `WarehouseAssembler` verifies row counts at 5% tolerance (WC-1), SHA-256 hashes every consumed CSV for the byte-deterministic manifest, and enforces four structural validation criteria (WC-2) before any motif enters the analysed corpus.

**Expected deposit metadata:** version `1.5.10`, extraction date `2025-12-28`, Ethereum mainnet anchor block `24,107,981`, xDai anchor block `43,868,880`.

---

## Installation

**Prerequisites:** Python 3.10 or later, pip, and (optionally) one or more CUDA-capable GPUs. The manuscript results were produced on four NVIDIA RTX 3090 GPUs; the framework runs on CPU when CUDA is unavailable, though wall-clock times will be substantially longer.

```bash
# Clone the repository
git clone <repository-url>
cd verisheaf

# Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate    # Linux/macOS
# .venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

**PyTorch with CUDA:** If you have NVIDIA GPUs, install the CUDA-enabled PyTorch build appropriate to your driver version. The `requirements.txt` installs the CPU-only build by default. For CUDA 12.x:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Consult [pytorch.org/get-started](https://pytorch.org/get-started/locally/) for other CUDA versions.

---

## Reproducibility

Full byte-equal reproducibility requires setting two environment variables **before** the Python interpreter launches, because `PYTHONHASHSEED` affects module import order and `CUBLAS_WORKSPACE_CONFIG` controls cuBLAS workspace allocation:

```bash
export PYTHONHASHSEED=20260601
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

The `set_global_determinism` function in `training.py` enforces six layers of determinism (Python `random`, NumPy, PyTorch CPU, PyTorch CUDA, cuDNN backend flags, and the two environment variables), but the environment variables must be set externally for modules already imported at launch time.

---

## Usage

The entire experimental program is driven by `reproduce.py`, which exposes four subcommands through argparse.

### Quick Smoke Test

Validates the framework's wiring against a small synthetic corpus. Completes in under thirty seconds on commodity hardware. Runs the five theorem verifiers, the ElGamal substitution check, and a full evaluation report on twenty synthetic motifs.

```bash
python reproduce.py quick
```

**Output:** `04_outputs/quick/metrics.json` and `04_outputs/quick/summary.md`.

### Eight-Incident Benchmark

Reconstructs the eight catalogued governance-attack incidents (Beanstalk, Compound Golden Boys, Tornado Cash Governance, Audius, Build Finance, Mango Markets, Yam Finance v1, Fortress Protocol) as typed temporal motifs with stage-targeted perturbations, trains the SheafLearner, and runs the full evaluation pipeline.

```bash
python reproduce.py incident
```

**Output:** `04_outputs/incident/metrics.json` and `04_outputs/incident/summary.md`.

### Principal Warehouse Experiment

The full experimental programme. Assembles 17,401 validated governance motifs from the DAO Analyzer deposit across Aragon, DAOhaus, and DAOstack, trains the SheafLearner (multi-seed, multi-GPU when hardware permits), and runs all seven warehouse experiments in dependency order:

| Experiment | Description |
|:--|:--|
| **F1** | Three-way stage classification on the full corpus |
| **F2** | Binary scoring with DCL-GFD and KnowGraph baselines |
| **F3** | Cross-platform stratification (Aragon, DAOhaus, DAOstack) |
| **F4** | Theorem 5 perturbation stress test across seven ε magnitudes (principal empirical contribution) |
| **F5** | Stage-attribution accuracy on the eight catalogued incidents |
| **F6** | Temporal-incoherence baseline |
| **F7** | Complementary-pipeline experiment with four-detector flagging |

```bash
# Full run (approximately 6h 28m on 4× RTX 3090)
python reproduce.py warehouse

# Override data path or baseline training epochs
python reproduce.py --warehouse-data /path/to/18072773 --baseline-epochs 50 warehouse
```

**Output:** `04_outputs/warehouse/RESULTS_FOR_MANUSCRIPT.json` (byte-deterministic, `sort_keys=True`, `allow_nan=False`) plus per-experiment run-log fragments under `04_outputs/run_logs/`.

### Corpus Verification

Loads the warehouse manifest and prints a one-glance summary of file metadata, per-platform motif counts, validation failures, and the eight-incident cross-reference. Run this after `--warehouse` to verify the corpus integrity.

```bash
python reproduce.py verify
```

### Global Options

All subcommands accept the following options:

| Option | Default | Description |
|:--|:--|:--|
| `--global-seed` | `20260601` | Master seed; per-experiment seeds derive deterministically |
| `--outputs-root` | `04_outputs` | Root directory for all generated artefacts |
| `--warehouse-data` | `02_data/18072773` | Path to the unpacked Zenodo deposit |
| `--baseline-epochs` | `30` | Per-baseline training epochs for F2, F3, and F7 |

---

## Module Reference

### `theory.py` (1,870 lines)

Core mathematical framework implementing the five theorems and Corollary 1. Contains the `CellularSheaf` data structure, sheaf Laplacian construction, coboundary operator, `SheafLearner` neural module (a `torch.nn.Module` that learns per-edge restriction maps), and the empirical verifiers `verify_theorem_1` through `verify_theorem_5`. The canonical stalk dimensions are: voter/delegate 8, proposal/execution-contract 16, governance-token 4; edge stalks are cast 12, record/count 16, delegate 8, transfer 4.

### `cryptography.py` (639 lines)

Exponential ElGamal restriction-map registry. Enforces stage-disjoint column support by constructing each stage-τ restriction-map block to read only the τ-slice of the incident node's stalk. The two-row structure encodes the $g^r$ ciphertext coordinate (rows 0 to $d_{out}/2$) and the $g^m h^r$ coordinate (rows $d_{out}/2$ to $d_{out}$), with singular-value normalisation fixing the operator-norm radius at unity.

### `motifs.py` (996 lines)

Typed temporal motif schema. Defines `MotifEdge`, `Motif`, `Incident`, and the `CANONICAL_INCIDENTS` list of eight governance-attack incidents. Provides `motif_to_typed_graph` (converter from Motif to the graph structure consumed by the sheaf), `motif_to_node_signals` (the CC-3 in-place fix replacing the prior Gaussian-noise input bug), `make_synthetic_motif`, `make_attacked_motif`, and the deterministic `assign_chrono_rank` that makes motif hashes insertion-order-invariant.

### `assemblers.py` (1,531 lines)

Two converter pathways. Pathway 1 (`assemble_incident_motifs`) reconstructs the eight-incident benchmark. Pathway 2 (`WarehouseAssembler`) consumes nine CSVs from the Zenodo deposit, enforces four validation criteria (WC-1 through WC-4), and emits a SHA-256-hashed manifest alongside the validated motif corpus.

### `training.py` (712 lines)

SheafLearner training loop with `AdamW` optimiser, cosine-annealing-with-warm-restarts schedule, masked-reconstruction sheaf-Dirichlet energy objective augmented by the Theorem 4 PAC-Bayes regulariser, and early stopping on validation loss. The `multi_seed_training` function dispatches four seeds to four GPUs sequentially. The `set_global_determinism` helper enforces six-layer reproducibility.

### `baselines.py` (2,415 lines)

Faithful reimplementations of four published graph-based anomaly detectors: DCL-GFD (AAAI 2025), KnowGraph (CCS 2024), NSD (NeurIPS 2022), and ShadowEyes (TIFS 2025). All adapters convert VERISHEAF motifs to homogeneous attributed graphs and apply stage-routing reduction for the three-way classification task. Each `build_*_scorer` function returns a `(binary_scorer, stage_predictor)` tuple.

### `evaluation.py` (2,309 lines)

Seven warehouse experiment entry points (`warehouse_experiment_f1` through `warehouse_experiment_f7`), the `warehouse_evaluation_suite` orchestrator, the `full_evaluation_report` harness for the incident benchmark, and statistical utilities including `bootstrap_ci`, `holm_correct`, and `paired_wilcoxon`.

### `reproduce.py` (1,271 lines)

CLI orchestrator. Four subcommands (`quick`, `incident`, `warehouse`, `verify`) with per-stage error capture, wall-time tracking, and structured JSON output. Partial results are preserved if individual experiments error; the orchestrator does not abort on stage failures.

---

## Key Results (Manuscript Headline Numbers)

| Metric | Value |
|:--|:--|
| Warehouse corpus size | 17,401 validated motifs (Aragon 5,914 / DAOhaus 9,113 / DAOstack 2,374) |
| F1 cast-prediction fraction | 0.98972 (empirical realisation of Corollary 1) |
| F3 cross-platform agreement | Within 0.003 absolute across three platforms |
| F4 Theorem 5 dominance rate | 0.991 across 7,014 perturbation cells (exceeds 0.95 threshold by 4.1 pp) |
| F5 incident attribution accuracy | 8/8 = 1.000 (95% CI [0.632, 1.000]) |
| F7 four-detector intersection | 1,287 motifs (7.40% of warehouse) jointly flagged |
| Wall-clock budget | 6h 28m 41s on 4× RTX 3090 |

---

## Citation

The warehouse corpus is derived from:

```bibtex
@misc{arroyo2025daoanalyzer,
  author    = {Arroyo, Javier and Dav{\'o}, David and Faqir-Rhazoui, Youssef},
  title     = {{DAO} Analyzer dataset},
  year      = {2025},
  publisher = {Zenodo},
  version   = {2025-12-28},
  doi       = {10.5281/zenodo.18072773}
}
```

---

## License

This repository is released for academic research purposes. See `LICENSE` for terms.

---

## Troubleshooting

**`WarehouseSchemaError` on CSV row counts:** Verify that the Zenodo deposit was extracted completely. The assembler expects version `1.5.10` with nine specific CSVs. Re-download if row counts diverge by more than 5%.

**CUDA out of memory:** Reduce `--baseline-epochs` or run on CPU by setting `CUDA_VISIBLE_DEVICES=""`. The SheafLearner's memory footprint is modest (stalk dimension ≤ 16); the baselines (particularly the ShadowEyes ResNet-50 backbone) are the primary GPU consumers.

**Non-deterministic results across runs:** Ensure `PYTHONHASHSEED` and `CUBLAS_WORKSPACE_CONFIG` are set in the shell environment **before** launching `python reproduce.py`. Setting them inside the process (which `set_global_determinism` does as a fallback) has no effect on already-imported modules.

**Missing `scipy` or `torch` after install:** The `requirements.txt` pins CPU-only PyTorch. For GPU support, install the CUDA-enabled build first and then install the remaining dependencies.
