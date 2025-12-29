# Deviation Spectroscopy: Invariant Metric Discovery and Validation

This repository contains the full experimental and analysis code accompanying a study on **invariant metric discovery in dynamical systems**. The project introduces and validates a flux-based method for identifying system-invariant quadratic forms from data, even under forcing variation, partial observation, nonlinearity, and noise.

The codebase is organized as a reproducible scientific software project, with clearly separated system definitions, identification routines, experimental runners, and figure-generation scripts used in the accompanying manuscript.

## Citation

If you use this code or the associated methodology in your research, please cite the accompanying dataset/software record:

**APA:**
Riaz, A. (2025). *Deviation Spectroscopy (v1.0)* Zenodo. https://doi.org/10.5281/zenodo.18079313

**IEEE:**
[1] A. Riaz, "Deviation Spectroscopy". Zenodo, Dec. 27, 2025. doi: 10.5281/zenodo.18079313.

**BibTeX:**
```bibtex
@software{Riaz2025DeviationSpectroscopy,
  author       = {Riaz, Asaad},
  title        = {Deviation Spectroscopy},
  version      = {v1.0},
  publisher    = {Zenodo},
  year         = 2025,
  doi          = {10.5281/zenodo.18079313},
  url          = {[https://doi.org/10.5281/zenodo.18079313](https://doi.org/10.5281/zenodo.18079313)}
}
```

## Scientific Objective

The central goal of this project is to demonstrate that a physically meaningful invariant metric can be:

- Discovered directly from data via flux-matching,
- Preserved across changes in forcing parameters,
- Robust to partial observation and observation noise,
- Sensitive to true structural changes (regime switches),
- Superior to common baselines (Lyapunov- and covariance-based metrics) in invariance tests.

## Repository Structure

- `src/deviation_spectroscopy/` — Core library code (systems, identification, metrics, flux matching)
- `src/deviation_spectroscopy/experiments/` — Reproducible experiment runners
- `src/deviation_spectroscopy/scripts/` — Figure and table generation scripts
- `results/` — Cached experiment outputs and manuscript figures

## Environment and Setup

This project is implemented in Python and follows standard scientific computing practices.

### Requirements
- Python ≥ 3.10
- NumPy
- SciPy
- Matplotlib
- Pandas
- Jinja

### Installation

```bash
python -m venv venv
source venv/bin/activate
pip install -e .
```

## Flux-Based Metric Identification (Core Method)

This project implements a **flux-matching framework** for identifying an invariant quadratic metric

    E_H(z) = 0.5 · zᵀ H z

directly from observed trajectories of linear or reconstructed dynamical systems.

Unlike classical Lyapunov approaches, the metric **H is not prescribed**.  
Instead, it is inferred by enforcing **energy balance closure** over finite time windows.

All discovery, invariance, robustness, and real-data results in this repository rely on this method.

---

### Energy Balance Formulation

Consider a continuous-time system

    ż = A z + B u

For a quadratic energy E_H(z), its time derivative satisfies

    dE_H/dt = zᵀ H B u − zᵀ S(H) z

where the dissipation matrix is defined as

    S(H) = −0.5 · (Aᵀ H + H A)

---

### Windowed Energy Balance

Over a finite time window, the method enforces the approximate balance

    ΔE_H ≈ ∫ zᵀ H B u dt − ∫ zᵀ S(H) z dt

This balance is evaluated numerically over sliding windows along the trajectory.

---

### Windowed Residual

For each window k, the residual is defined as

    r_H[k] = ΔE_H[k] − I_{Π,H}[k] + I_{Σ,H}[k]

where:
- ΔE_H is the net energy change over the window
- I_{Π,H} is injected power
- I_{Σ,H} is dissipated energy

Residuals are aggregated using an RMS ratio

    ||r|| / ||ΔE|| = sqrt(mean(r²)) / max(sqrt(mean(ΔE²)), ε)

This quantity serves as a **dimensionless physical validity metric** throughout the paper.

---

### Optimization Problem

The inferred metric H* is obtained by solving

    minimize_H   Σ_k r_H[k]²
    subject to   H ≻ 0
                 S(H) ≽ 0
                 trace(H) = 1

---

### Key Design Choices

- **SPD parameterization**  
  H is represented as H = L Lᵀ with lower-triangular L

- **Exact trace normalization**  
  Enforces trace(H) = 1 to remove scale ambiguity

- **PSD constraint on dissipation**  
  S(H) ≽ 0 enforced via eigenvalue penalty

- **Windowed residual objective**  
  Improves robustness to noise and finite sampling

---

### Implementation

The flux-matching algorithm is implemented in:

    src/deviation_spectroscopy/flux/
        ├── flux_match.py
        └── residuals.py

#### `fit_H_flux_matching(...)`

Defined in `flux_match.py`, this routine:
- optimizes H using **L-BFGS-B**
- enforces H ≻ 0, trace(H) = 1, and S(H) ≽ 0
- returns a `FluxMatchResult` containing:
  - inferred metric H
  - dissipation matrix S(H)
  - objective value
  - eigenvalue diagnostics
  - convergence status

This function is used uniformly across:
- discovery tests
- invariance benchmarks
- partial-observation experiments
- negative controls
- real-world power grid data

#### Residual Computation

Implemented in `residuals.py`:
- `windowed_integrals(...)`: computes ΔE_H, injected and dissipated fluxes
- `windowed_residual(...)`: constructs r_H[k]
- `residual_ratio(...)`: aggregates residuals into a scalar error metric

---

### Conceptual Role

The flux-matching formulation:
- does **not** assume a Lyapunov function
- separates **geometric invariance** from **physical closure**
- remains well-posed under forcing variation, partial observation, reconstruction error, and noise

It forms the **foundational primitive** on which all higher-level experiments and figures in this repository are built.

## Core Mathematical Primitives

The `core/` module provides the **mathematical foundation** shared across all experiments,
including baselines, linear algebra utilities, physical metrics, and windowing operators.
These components are intentionally minimal, deterministic, and independently testable.

---

### Baseline Metrics (`core/baselines.py`)

Two standard reference metrics are implemented for comparison against the learned flux-based metric.

**Lyapunov Baseline**

Solves the continuous-time Lyapunov equation

    Aᵀ H + H A = −I

This baseline:
- requires A to be Hurwitz
- produces a symmetric positive definite (SPD) matrix
- is trace-normalized for scale invariance

Used to contrast **prescribed Lyapunov metrics** against data-driven inference.

**Covariance Baseline**

Constructs a metric from inverse state covariance

    P = ⟨z zᵀ⟩
    H ∝ P⁻¹

Includes ridge regularization, SPD projection, and trace normalization.
This baseline reflects common statistical and identification-based approaches.

---

### Linear Algebra Utilities (`core/linalg.py`)

Low-level helpers used throughout the project:
- `sym(A)`: symmetric projection
- `project_spd(A)`: eigenvalue-based SPD projection
- `normalize_trace(H)`: enforces trace(H) = constant
- `is_spd`, `is_psd`: numerical definiteness checks
- `fro_norm(A)`: Frobenius norm

These utilities ensure **numerical safety and scale consistency** across optimization,
metrics, and diagnostics.

---

### Physical Metrics (`core/metrics.py`)

Scalar quantities used for invariance and robustness evaluation.

**Thermal Ratio (TΔ)**

    TΔ = ⟨E_H⟩ / ⟨Σ_H⟩

Computed as a time-average ratio of quadratic energy to dissipation.
Used extensively in invariance, drift, and regime-comparison tests.

**Dissipation Drift**

    ||S_A − S_B||_F / ||S_A||_F

Measures relative deviation between dissipation matrices inferred
under different conditions.

---

### Windowing Operators (`core/windows.py`)

Provides deterministic sliding-window machinery used across:
- flux residual computation
- invariance benchmarks
- noise robustness tests

Includes:
- index window generation with stride control
- Hann, Tukey, and boxcar tapers
- windowed numerical integration with integral preservation

Tapers are normalized so that integrals of constant signals are preserved exactly.

---

### Design Philosophy

The `core/` layer is:
- **model-agnostic**
- **numerically conservative**
- **free of experiment-specific assumptions**

All higher-level discovery, invariance, and real-data results depend on these primitives,
ensuring interpretability and reproducibility.

## Simulation Infrastructure

The `sim/` module provides reproducible numerical tools for generating
forced continuous-time dynamics, observation noise, and stochastic inputs.
All simulations in this repository rely on these routines.

---

### Forcing Functions (`sim/forcing.py`)

Standardized input signals used across invariance, robustness, and discovery tests:

- `zero_force(t)`  
  No external input.

- `sine_force(t, amplitude, frequency, phase)`  
  Single-frequency sinusoidal forcing used in baseline invariance experiments.

- `multi_sine_force(t, amplitudes, frequencies, phases)`  
  Superposition of sinusoidal components for richer excitation.

- `band_limited_noise(t, rms, f_low, f_high)`  
  Gaussian noise filtered in the frequency domain and RMS-normalized.
  Used to test robustness under stochastic forcing.

- `scale_amplitude(u, scale)`  
  Pure amplitude scaling operator used in forcing-invariance tests.

All forcing functions are deterministic given fixed parameters and random seeds.

---

### Continuous-Time Integration (`sim/integrate.py`)

Deterministic simulation of linear time-invariant (LTI) systems:

    ż = A z + B u(t)

Implemented using a fixed-step **fourth-order Runge–Kutta (RK4)** scheme
on a uniform time grid.

Key features:
- exact handling of multi-input forcing
- shape-safe input validation
- linear interpolation of inputs at RK mid-steps

Returns a `SimResult` object containing time, state trajectory, and input.

---

### Stochastic Noise Processes (`sim/ou_noise.py`)

Implements a one-dimensional **Ornstein–Uhlenbeck (OU)** process:

    dx = −(1/τ) x dt + σ dW

using Euler–Maruyama discretization.

Used to:
- test robustness under colored noise
- validate partial-observation and noise-invariance claims
- construct negative controls

---

### Design Role

The simulation layer is:
- numerically conservative
- fully deterministic under fixed seeds
- independent of the flux-matching method itself

It serves solely as controlled input generation and does not encode
any invariance assumptions.

## Dynamical Systems

The `systems/` module defines the continuous-time linear systems used throughout
the synthetic experiments, discovery tests, invariance benchmarks, and controls.

All systems conform to the standard state-space form:

    ż = A z + B u

These models are intentionally low-dimensional, physically interpretable, and
structurally transparent, allowing clear separation between system physics and
metric inference.

---

### Base Interface (`systems/base.py`)

Defines a minimal abstract interface for linear systems:

- state dimension
- system matrices `(A, B)`
- optional process noise matrix
- optional descriptive metadata

This interface enables consistent use across simulation, identification,
and evaluation pipelines.

---

### Coupled Oscillator System (`systems/coupled_oscillator.py`)

A pair of coupled damped oscillators with known physical energy.

State vector:

    z = [x₁, v₁, x₂, v₂]

Dynamics correspond to two masses coupled by springs and dampers, with external
forcing applied to the second oscillator.

Crucially, this system includes a **ground-truth quadratic energy metric**:

    E(z) = ½ zᵀ H_true z

This allows direct comparison between:
- inferred metrics from flux matching
- classical baselines
- known physical energy structure

The coupled oscillator is the primary system used for:
- discovery demonstrations
- partial-observation experiments
- noise robustness tests

---

### Two-Mass Spring–Damper System (`systems/two_mass.py`)

A canonical mechanical benchmark with physically meaningful parameters:

- two masses
- springs and dampers
- external force applied to mass 1

State vector:

    z = [x₁, v₁, x₂, v₂]

Unlike the coupled oscillator, this model **does not expose a ground-truth energy metric**.
It is used to test whether the flux-based method can infer a consistent metric
purely from observed dynamics and forcing.

Metadata is recorded for reproducibility and parameter sweeps.

---

### Design Rationale

The systems in this repository are chosen to be:
- linear but nontrivial
- physically interpretable
- suitable for partial observation
- sensitive to forcing variation
- compatible with analytic baselines

They provide a controlled setting in which metric inference, invariance,
and robustness claims can be meaningfully evaluated.

## State Reconstruction and System Identification

The `id/` module implements the **state reconstruction and linear system identification**
pipeline used when the full system state is not directly observed.

This is essential for:
- partial-observation experiments,
- noise-robust invariance tests,
- discovery under realistic measurement constraints.

All identification steps are **decoupled** from the flux-based metric inference,
ensuring that invariance results do not depend on privileged access to the true state.

---

### Linear System Identification (`identify_ab.py`)

Provides ridge-regularized estimation of system matrices:

    ż ≈ A z + B u

from time-series data using central-difference derivatives and least squares.

Key features:
- ridge regularization for numerical stability,
- condition-number diagnostics,
- residual norm reporting.

This module is used when the **true state is available** (e.g., synthetic benchmarks).

Public API:
- `identify_ab(...)` → returns `(Â, B̂)` and diagnostics.

---

### Identification from Reconstructed State (`identify_from_reconstruction.py`)

Extends system identification to **reconstructed latent states**.

Key robustness features:
- moving-average smoothing prior to differentiation,
- regularized regression to suppress noise amplification,
- identical API to full-state identification.

This ensures that:
- system identification does not overfit reconstruction artifacts,
- flux-based metric inference remains well-posed under noise.

Outputs:
- identified `(Â, B̂)`
- normalized residual error

---

### Delay-Embedding State Reconstruction (`state_reconstruct.py`)

Implements a **deterministic subspace reconstruction** based on delay embeddings
and singular value decomposition.

This reconstruction:
- does *not* assume a model,
- does *not* require stochastic filtering,
- produces a fixed-dimensional latent state usable across datasets.

#### SubspaceProjector

Core reconstruction class with guarantees:

- output dimension = specified `order` (always),
- same fitted basis reusable across conditions,
- stable under moderate observation noise.

Pipeline:
1. optional moving-average smoothing of observations,
2. delay embedding with fixed horizon,
3. SVD-based subspace extraction,
4. optional √Σ normalization to equalize modal energy.

The projector is **fit once** on a base condition and reused across all others,
preventing information leakage in invariance tests.

---

### Design Rationale

The reconstruction and identification pipeline is designed to:

- preserve **geometric structure** under partial observation,
- avoid stochastic filtering assumptions,
- cleanly separate:
  - observation → reconstruction,
  - reconstruction → system ID,
  - system ID → metric inference.

This modularity is critical for interpreting invariance as a **physical property**
rather than a modeling artifact.

## Data Loading and Preprocessing

The `data/` module provides **robust, reproducible loaders** for both synthetic
and real-world time series used throughout the project.

All loaders are designed to:
- tolerate missing or malformed rows,
- preserve original timestamps,
- make preprocessing steps explicit and reversible.

No hidden resampling, filtering, or normalization is performed implicitly.

---

### Generic Time-Series Loader (`loader.py`)

Provides a lightweight, dependency-free CSV loader suitable for
instrument logs, oscilloscope data, or PMU-style measurements.

#### `load_csv_timeseries(...)`

Loads a CSV file into a unified container:

    TimeSeriesData:
      t : (N,)   time in seconds
      y : (N,p)  observed outputs
      u : (N,m)  optional known inputs

Key features:
- column selection by **name or index**,
- automatic NaN / Inf row rejection (reported, not silent),
- optional input channels,
- automatic time sorting (real logs are often shuffled),
- explicit time scaling (e.g. milliseconds → seconds).

This loader is used for:
- synthetic benchmarks with logged outputs,
- exploratory real-world datasets,
- sanity-check experiments.

---

### Detrending Preprocessing

#### `preprocess_detrend(...)`

Removes low-order polynomial trends from outputs (and inputs, if present):

    y(t) ← y(t) − polyfit(t)

This step is **critical** for fluctuation-based analyses where:
- DC offsets,
- slow ramps,
- long-term drift

would otherwise contaminate energy and dissipation estimates.

Important properties:
- uses normalized time for numerical stability,
- preserves original timestamps,
- does not resample or filter high-frequency content.

---

### Power Grid Frequency Data (`powergrid_frequency.py`)

Implements **specialized loaders** for publicly available grid-frequency data
(Europe, US, Japan), which differ substantially in format and metadata.

Supported sources:
- EU `.csv.zip` archives (power-grid-frequency.org),
- US / Japan plain `.csv` logs with quality indicators.

#### Unified Interface

    load_powergrid_frequency(path)

Returns:

    GridFrequencyData:
      t : (N,)   seconds since start
      y : (N,1)  frequency deviation in Hz
      name
      tz

Key handling:
- timestamp parsing with timezone normalization,
- automatic unit conversion (mHz → Hz),
- removal of duplicated or non-monotonic timestamps,
- optional quality-flag filtering (US/Japan data).

---

### Uniform Resampling

#### `resample_uniform(...)`

Explicitly resamples irregular time series onto a uniform grid:

    y(t) → y(t_k),   t_k = k·Δt

This is required for:
- windowed flux integration,
- consistent delay embeddings,
- cross-regime comparison.

No resampling is performed implicitly elsewhere in the pipeline.

---

### Regime Slicing

#### `split_two_regimes(...)`

Extracts two disjoint time intervals from a single dataset:

    (A, B) = split(data, tA, tB)

Used in:
- invariance tests across operating conditions,
- structural break detection,
- Level-3 sensitivity experiments.

Each slice is time-reset to start at zero, preserving internal dynamics.

---

### Real-World Datasets

The `data/realdata/` directory contains **raw, unmodified datasets**:

- `finland/`
- `germany/`
- `uk/`
- `us/`

These files are not altered by the repository.
All preprocessing steps are performed explicitly in experiment scripts.

---

### Design Rationale

Data ingestion is intentionally:
- **boring** (no magic),
- **explicit** (every transformation is named),
- **separable** from modeling and inference.

This ensures that invariance results can be traced to **physical structure**
rather than preprocessing artifacts.

## Experiments: Invariance and Flux Closure

The `experiments/` module implements the **primary validation tests** used in
the paper: invariance of an inferred quadratic metric under changes in forcing.

All experiments follow a strict protocol:
1. identify or assume a linear system,
2. fit a single metric \(H^\*\) on one operating regime,
3. evaluate physical closure on a distinct regime,
4. compare against standard baselines.

No retraining or tuning is allowed across regimes.

---

### Dataset Construction (`datasets.py`)

Defines a unified container:

    Dataset:
      t : (N,)    time
      z : (N,n)   true state
      u : (N,m)   input / forcing
      y : (N,p)   measured output

#### `build_dataset(...)`

Simulates a continuous-time LTI system

    ż = A z + B u(t)
    y = C z

using deterministic RK4 integration, returning a fully populated `Dataset`.

#### `build_experiment_datasets(...)`

Constructs **three disjoint datasets** for each system:

- `dataset_0_id`  
  Identification regime (used only if A,B are unknown)

- `dataset_A_flux`  
  **Training regime** for flux-matching (metric inference)

- `dataset_B_invariance`  
  **Test regime** with different forcing

By design:
- the metric \(H^\*\) is fit **only on Dataset A**,
- Dataset B is never seen during optimization.

If no output matrix `C` is provided, full-state observation is assumed.

---

### Invariance Experiment Runner (`run_invariance_suite.py`)

This script performs the **core invariance test** reported in the paper.

For each system:
1. Build datasets A and B with different forcing.
2. Infer \(H^\*\) via flux matching on Dataset A.
3. Evaluate energy closure on **both** datasets.
4. Compute the relative transport discrepancy:

    T_rel = |T_A − T_B| / T_A

where \(T\) is the total inferred dissipation over all windows.

Lower values indicate stronger invariance.

#### Baselines

Two standard baselines are evaluated **without refitting**:
- Lyapunov metric (solution of \(A^T H + H A \prec 0\))
- Covariance metric (state covariance inverse)

Each baseline is subjected to the same A/B test.

Results are written to:

    results/invariance/<system>.json

---

### Benchmark Variant (`run_invariance_benchmarks.py`)

Functionally equivalent to `run_invariance_suite.py`, but structured for:
- cleaner logging,
- explicit benchmarking runs,
- easier extension to additional systems.

Both scripts implement the **same invariance logic**.
Only one is used in the final paper figures.

---

### Result Aggregation (`aggregate_results.py`)

Collects all per-system JSON outputs and produces:

- `invariance_summary.csv`  
  (for tables, plotting, and supplementary material)

- `invariance_summary.json`  
  (machine-readable archive)

Metrics reported:
- dissipation estimates on A and B,
- relative transport error \(T_{rel}\),
- residual ratios (physical closure quality),
- baseline comparisons.

---

### Interpretation

An invariant metric satisfies:

- low residual ratio (good energy balance),
- low \(T_{rel}\) across forcing changes,
- consistent dissipation estimates without retraining.

Flux-matched metrics consistently outperform
Lyapunov and covariance baselines in this setting.

---

### Design Philosophy

The experiment layer is intentionally:
- **procedural** (not framework-heavy),
- **fully reproducible** (no hidden state),
- **stateless across runs**.

Every result can be regenerated by running the scripts
in this directory with no external configuration.

## Experiments: Real-Data and Output-Only Validation

The scripts in this section validate the full pipeline under **realistic constraints**:
partial observation, measurement noise, unknown inputs, and field-recorded data.

These experiments are not used to tune the method.
They exist to demonstrate *operability*, *robustness*, and *end-to-end reproducibility*.

---

### Dry-Run: Synthetic → CSV → Loader → Full Pipeline  
(`run_realdata_dryrun.py`)

This script is a **controlled end-to-end dry run** that mimics real experimental conditions
while retaining a known ground-truth system.

Pipeline structure:

1. Simulate a coupled oscillator with known (A, B).
2. Apply **partial observation**:
       y = C z   (single output channel)
3. Inject controlled measurement noise (SNR-based).
4. Export data to CSV (as if from field instrumentation).
5. Reload data using the public CSV loader.
6. Detrend outputs (safe default for real data).
7. Reconstruct state via delay-embedding subspace projection.
8. Identify surrogate (“ghost”) Â, B̂ from reconstructed state.
9. Fit a single flux-matched metric H* on regime A only.
10. Evaluate invariance on regime B using the same H*.

Key properties:

- Reconstruction basis is **fit on regime A and reused on B**.
- System identification is performed **once**, on A only.
- No retraining or refitting occurs across regimes.
- Both invariance and residual closure are reported.

Outputs:
- Console diagnostics (for inspection)
- `results/realdata_dryrun/dryrun_summary.txt`
  (minimal numerical log suitable for paper archives)

This script certifies that the full pipeline functions correctly
when driven entirely by disk-loaded, noisy, partially observed data.

---

### Public Power-Grid Frequency (Single-Run)  
(`run_public_grid_frequency.py`)

Applies the method to **real power-grid frequency deviation data**
(EU / US / Japan public datasets).

Characteristics:
- Output-only measurements (no known input u(t)).
- Non-stationary real-world signals.
- Deterministic, declared regime selection (no cherry-picking).

Procedure:

1. Load raw grid frequency data (`.csv` or `.csv.zip`).
2. Uniformly resample to fixed dt.
3. Split into two disjoint operating regimes (A and B).
4. Set dummy input u(t) = 0 (output-only formulation).
5. Reconstruct latent state via delay embedding:
       ẑ = Φ(y)
6. Identify surrogate Â, B̂ from reconstructed state (A only).
7. Fit H* via flux matching on regime A.
8. Evaluate invariance and closure on regime B.

All hyperparameters (order, horizon, smoothing, windows)
are explicit command-line arguments.

Output:
- One JSON summary per dataset:
      results/public_grid/<file>_summary.json

This script is intended for **transparent, single-case analysis**
and figure generation.

---

### Public Power-Grid Frequency (Batch Study)  
(`run_public_grid_batch.py`)

Automates the above pipeline across **all available real datasets**
using a **fixed, declared parameter set**.

Design constraints:

- Same-day, same-block regimes (no cross-day mixing).
- Identical reconstruction, identification, and flux parameters.
- No per-dataset tuning.
- Failures are logged and skipped (never hidden).

For each dataset:
- reconstruct latent state,
- identify surrogate dynamics,
- infer H* on regime A,
- evaluate invariance on regime B.

Outputs:
- `public_grid_batch_summary.csv`
- `public_grid_batch_summary.json`

Each row records:
- relative dissipation invariance error,
- residual closure error,
- minimum eigenvalue of H*,
- solver success status.

This batch study supports **statistical robustness claims**
and guards against isolated or hand-picked successes.

---

### Interpretation

Across real and synthetic data, a valid metric H* should exhibit:
- stable dissipation estimates across regimes,
- low relative transport error,
- consistent residual closure without retraining.

The real-data experiments demonstrate that the method:
- operates under partial observability,
- tolerates realistic noise,
- does not require known inputs,
- remains invariant across operating conditions.

These properties are essential for physical applicability
beyond simulation benchmarks.

## Experiments: Discovery, Invariance, and Stress Testing

This section documents the controlled discovery and stress-test experiments
that establish when a quadratic metric H* is:
- discoverable,
- invariant under forcing variation,
- robust to partial observation and noise,
- sensitive to genuine structural change,
- and invalidated by broken physical structure.

All experiments in this section use **fixed, preregistered procedures**
with no per-condition tuning.

---

## Level 1 — Discovery Under Forcing Variation (Full-State)

### Metric Discovery with Known Ground Truth  
(`run_discovery_test.py`)

This experiment tests whether the flux-matching method can **recover a known physical energy metric**
from full-state data and whether that metric remains invariant under changes in forcing.

Setup:
- True system: coupled oscillator with known physical quadratic energy H_true.
- Full state z(t) observed.
- True A, B used (no identification uncertainty).
- Single identification dataset fixes A_hat, B_hat.
- Forcing frequency and amplitude are varied.

For each forcing condition:
- Infer H* via flux matching.
- Evaluate physical closure (T_delta, residual ratio).
- Compare H* to:
  - base-condition H* (scale-aligned invariance),
  - true physical H_true (scale-aligned recovery).
- Compare against Lyapunov and covariance baselines.

Outputs:
- `results/discovery_test/discovery_test_summary.json`

Purpose:
- Demonstrates **metric discovery**, not assumption.
- Establishes **invariance to forcing variation**.
- Shows **quantitative recovery** of the true physical metric.
- Provides baseline comparisons under identical data.

---

## Level 2 — Discovery Under Partial Observation

### Partial Observation Discovery  
(`run_discovery_test_partial_obs.py`)

Repeats the Level 1 discovery experiment using **only a single observed channel**.

Setup:
- Observation: y = x₁ only.
- Latent state reconstructed using delay embedding.
- Reconstruction basis is fit **only on base condition** and reused.
- Ghost dynamics (A_hat, B_hat) identified once, from base only.
- No retraining across conditions.

For each forcing condition:
- Reconstruct latent trajectory using fixed projector.
- Infer H* via flux matching.
- Measure invariance vs base (scale-aligned Frobenius drift).
- Compare to Lyapunov and covariance baselines.

Outputs:
- `results/discovery_test_partial_obs/discovery_test_partial_obs_summary.json`

Purpose:
- Demonstrates **discoverability without full state access**.
- Shows invariance survives reconstruction error.
- Prevents data leakage by construction.

---

### Clean vs Noisy Partial Observation  
(`run_partial_obs_clean_vs_noisy.py`)

Directly compares discovery performance under:
- clean observations,
- noisy observations (additive Gaussian noise on observed channel only).

Design:
- Two independent pipelines (clean and noisy).
- Identical hyperparameters.
- Reconstruction, identification, and H inference performed separately.
- Same forcing grid and base condition.

Reported statistics:
- Invariance drift (mean / median / max).
- Residual closure degradation under noise.

Outputs:
- `results/partial_obs_clean_vs_noisy/partial_obs_clean_vs_noisy_summary.json`

Purpose:
- Quantifies **noise sensitivity**.
- Shows graceful degradation rather than collapse.
- Demonstrates robustness of flux closure under observation noise.

---

## Level 3 — Stress Tests Beyond Ideal Assumptions

### Level 3 Suite (Multi-Test, Multi-Seed)  
(`run_level3_suite.py`)

A preregistered suite of **hard stress tests** designed to falsify the method if it were brittle.

Global design:
- Multiple forcing conditions.
- Multiple random seeds.
- Fixed thresholds for invariance and residual closure.
- No manual intervention.

#### Test 3A — Nonlinear Dynamics via Lifted Linear Closure  
(Duffing oscillator)

- True system is nonlinear.
- Observation is single-channel.
- State reconstructed via embedding.
- Linear surrogate dynamics identified in reconstructed space.
- H* inferred via flux matching.

Goal:
- Test whether approximate linear closure in reconstructed coordinates
  still yields a consistent invariant metric.

#### Test 3B — Controlled Regime Switching (Piecewise LTI)

- System switches abruptly between two different linear regimes.
- Full-state observation (no reconstruction ambiguity).
- Separate H* inferred per regime.

Expected outcome:
- Low drift within regime.
- Large drift across regimes.

This test confirms **sensitivity to true structural change**.

#### Test 3C — Nonlinear Observation Maps

- True system is linear.
- Observation is nonlinear:
  - y = tanh(Cz)
  - y = (Cz)²
- Latent state reconstructed blindly.
- Dynamics identified and H* inferred.

Goal:
- Demonstrate robustness to nonlinear sensing distortions.

Outputs:
- `results/level3_suite/level3_suite_summary.json`

Includes:
- Per-seed invariance statistics.
- Pass-rate summaries at 5% and 15% drift thresholds.
- Explicit sensitivity confirmation for regime switching.

---

## Stochastic Forcing Robustness

### Ornstein–Uhlenbeck Perturbed Forcing  
(`run_discovery_test_ou.py`)

Extends the Level 1 discovery test by adding **colored stochastic forcing**
on top of deterministic inputs.

Setup:
- OU noise with declared correlation time and diffusion strength.
- Independent noise realizations across seeds.
- Forcing grid identical to deterministic tests.

Metrics:
- Invariance drift vs base (per seed).
- Quantiles (median, 90%, 95%, max).

Outputs:
- `results/discovery_test_ou/discovery_test_ou_summary.json`

Purpose:
- Demonstrates robustness under **temporally correlated stochastic forcing**.
- Shows that invariance is not an artefact of smooth deterministic inputs.

---

## Negative Controls (Falsification Tests)

### Broken Physics Controls  
(`run_negative_controls.py`)

Applies flux matching to datasets where physical structure is intentionally destroyed.

Negative cases:
- Time-shuffled state trajectories.
- Phase-scrambled trajectories.
- Incorrectly aligned input signals.

Expected outcome:
- Poor closure.
- Large residual ratios.
- Failed or unstable optimization.

Outputs:
- `results/negative_controls/negative_controls_summary.json`

Purpose:
- Demonstrates **specificity**, not just robustness.
- Confirms the method fails when physical balance laws are violated.

---

## Interpretation Across Levels

Across all discovery and stress tests, a valid metric H* exhibits:
- Low scale-aligned drift under forcing variation.
- Stable residual closure ratios.
- Robustness to partial observation and noise.
- Sensitivity to genuine structural change.
- Failure under deliberately broken physics.

These experiments collectively establish that flux-matched metrics are:
- discoverable,
- physically meaningful,
- invariant when they should be,
- and invalidated when they must be.

## Scripts: Paper Figures (Deterministic, Output-Only)

The `src/deviation_spectroscopy/scripts/` folder contains figure-generation scripts.
Each script is output-only: it loads JSON summaries produced by `experiments/` and renders a PNG
into `results/figures_paper/`. No experiment logic is duplicated inside figure scripts.

All figure scripts:
- assume the corresponding experiment has already been executed,
- are deterministic given fixed JSON outputs,
- write a single named PNG per script.

### Figure 1 — Discovery / Invariance Across Systems  
`figure1_discovery.py`  
Input: `results/invariance/*.json`  
Output: `results/figures_paper/Figure1_Discovery.png`

Bar plot of relative drift `T_rel` comparing:
- Flux-matched metric `H*`
- Lyapunov baseline `H`
- Covariance baseline `H`

Includes reference lines for 5% target and 15% bound (log scale).

### Figure 2 — Partial Observation Robustness Under Measurement Noise  
`figure2_partial_observation.py`  
Input: `results/partial_obs_clean_vs_noisy/partial_obs_clean_vs_noisy_summary.json`  
Output: `results/figures_paper/Figure2_PartialObservation.png`

Compares median invariance drift under:
- clean partial observation
- noisy partial observation (declared sigma)

Shown on log scale with the same 5% / 15% reference bounds.

### Figure 3 — Public Power Grid Frequency (Output-Only Data)  
`figure3_powergrid.py`  
Input: `results/public_grid_batch/public_grid_batch_summary.json`  
Output: `results/figures_paper/Figure3_PowerGrid.png`

Per-region drift values (`T_rel`) extracted from batch summaries.
Uses log scale and highlights successes (< 0.15) vs failures (>= 0.15).

### Discovery Test (Full-State) — Invariance and Recovery Panels  
`figure_discovery_test.py`  
Input: `results/discovery_test/discovery_test_summary.json`  
Outputs:
- `results/figures_paper/Figure_Discovery_Invariance.png`
- `results/figures_paper/Figure_Discovery_Recovery.png`

Panel A shows scale-aligned drift of inferred `H*` across forcing conditions.  
Panel B shows scale-aligned recovery error vs the physical ground truth `H_true`.

### Discovery Test (Partial Observation) — Drift, T_delta, Residual  
`figure_discovery_test_partial_obs.py`  
Input: `results/discovery_test_partial_obs/discovery_test_partial_obs_summary.json`  
Outputs:
- `results/figures_paper/Figure_Discovery_PartialObs_Invariance.png`
- `results/figures_paper/Figure_Discovery_PartialObs_Tdelta.png`
- `results/figures_paper/Figure_Discovery_PartialObs_Residual.png`

Plots invariance drift (log), stress metric `T_delta`, and flux-closure residual ratios (log)
across forcing conditions under single-channel observation + reconstruction.

### Level 3 Suite — Robustness Distributions, Sensitivity, Pass Rates  
`figure_level3_suite.py`  
Input: `results/level3_suite/level3_suite_summary.json`  
Outputs:
- `results/figures_paper/Figure_L3_Drift_Distributions.png`
- `results/figures_paper/Figure_L3_RegimeSwitch_Sensitivity.png`
- `results/figures_paper/Figure_L3_PassRates.png`

Summarizes multi-seed robustness under nonlinear dynamics and nonlinear observation,
and separately demonstrates sensitivity via regime-switch structural break detection.

### OU Forcing Robustness — Drift Distributions Across Seeds  
`figure_discovery_test_ou.py`  
Input: `results/discovery_test_ou/discovery_test_ou_summary.json`  
Output: `results/figures_paper/Figure_Discovery_OU_Drift.png`

Scatter + median bars for drift distributions under OU-perturbed forcing
(log scale, with 5% target and 15% bound).

### Negative Controls — Dual Panel (Residuals + Drift)  
`figure_negative_controls.py`  
Input: `results/negative_controls/negative_controls_summary.json`  
Output: `results/figures_paper/Figure4_NegativeControls_Dual.png`

Compares a clean physics baseline against negative controls (time shuffle, phase scramble, wrong input),
showing failure via flux residual blow-up and loss of geometric invariance (log scales).

## Tests — Core Mathematical and Numerical Validation

The `tests/` directory contains unit tests that validate the mathematical, numerical, and physical foundations of the deviation spectroscopy pipeline.  
These tests are intentionally **decoupled from experiments** and focus on correctness, invariance diagnostics, and numerical safety.

### Invariance Evaluation Helpers

**`helpers_invariance.py`**  
Defines a standardized evaluation routine used across experiments to assess inferred quadratic metrics.

- Computes windowed energy residuals and residual ratios
- Computes the stress metric `T_Δ`
- Constructs the symmetrized dissipation operator `S(H)`
- Operates on generic dataset objects (supports reconstructed / latent coordinates)

This module provides the canonical definition of invariance diagnostics used throughout the project.

---

### Linear Algebra Primitives

**`test_00_linalg.py`**  
Validates core linear-algebra utilities used to construct and normalize quadratic forms.

Tests include:
- Symmetrization of arbitrary matrices
- Projection onto the symmetric positive definite (SPD) cone
- Trace normalization of quadratic forms
- Distinction between PSD and SPD matrices
- Frobenius norm correctness

These tests ensure numerical stability and well-defined geometry for all inferred metrics.

---

### Windowing and Numerical Integration

**`test_01_windows.py`**  
Validates window construction and windowed integration routines used in flux and residual calculations.

Tests include:
- Correct construction of overlapping windows
- Proper normalization of tapering functions (boxcar, Hann, Tukey)
- Accurate integration of constant signals
- Correct handling of vector-valued signals

This prevents silent numerical bias in windowed energy and flux computations.

---

### System Base Interface

**`test_02_system_base.py`**  
Validates the abstract `LinearSystem` interface.

Tests include:
- Correct matrix dimensions
- Default absence of noise models
- Metadata propagation

Ensures all system implementations conform to a consistent and inspectable interface.

---

### Two-Mass Spring–Damper System

**`test_03_two_mass.py`**  
Validates structural and physical properties of the two-mass spring–damper model.

Tests include:
- State-space dimensionality
- Proper velocity coupling structure
- Correct parameter exposure via metadata

This guards against incorrect system construction that could invalidate physical interpretations.

---

### Forcing Signal Generation

**`test_04_forcing.py`**  
Validates deterministic and stochastic forcing utilities.

Tests include:
- Zero forcing behavior
- Sine forcing amplitude correctness
- Multi-sine signal construction
- RMS calibration of band-limited noise
- Amplitude scaling laws

These tests ensure forcing regimes used in invariance studies are well-controlled and reproducible.

---

### LTI Simulation and Time Discretization

**`test_05_simulate.py`**  
Validates the numerical LTI simulator.

Tests include:
- Output shape correctness
- Stability under zero input for stable systems
- Enforcement of uniform time grids

The uniform-grid check is critical for preventing invalid numerical integration in all experiments.

## Tests — Invariance, Identification, and Flux Geometry

This section documents the second group of tests, which validate **system identification**, **dataset construction**, and the central **invariance claims** of deviation spectroscopy under controlled perturbations.

---

### Dataset Construction and Experimental Roles

**`test_06_datasets.py`**  
Validates the construction of experiment datasets used throughout invariance studies.

This test ensures that:
- Identification (`id`) and steady-state datasets (`A`, `B`) are correctly generated
- Time grids, state trajectories, and inputs are shape-consistent
- Each dataset plays a well-defined experimental role (identification vs comparison)

This guards against silent dataset misalignment when comparing inferred metrics across regimes.

---

### Linear System Identification (A, B)

**`test_07_identify_ab.py`**  
Validates ridge-regularized estimation of system matrices from simulated data.

Checks include:
- Recovery accuracy of estimated `A` and `B`
- Bounded residual norms
- Compatibility with band-limited stochastic forcing

This establishes that downstream invariance results are not artifacts of poor system identification.

---

### Flux-Matching Metric Inference

**`test_08_flux_match.py`**  
Validates the core flux-matching optimization used to infer the quadratic metric `H*`.

This test checks that:
- Optimization converges successfully
- The inferred metric is trace-normalized and SPD
- The induced dissipation operator `S(H)` is PSD
- Flux residual ratios remain small

This is the first end-to-end validation of the metric inference pipeline.

---

### Baseline Metric Construction

**`test_09_baselines.py`**  
Validates construction of baseline quadratic metrics used for comparison.

Baselines tested:
- Lyapunov solution-based metric
- Empirical covariance-based metric

Both are required to be SPD and trace-normalized, ensuring fair comparison against `H*`.

---

### Amplitude Invariance (Flux vs Baselines)

**`test_10a_amplitude_invariance.py`**  
Tests invariance of the inferred metric under **amplitude scaling** of the forcing.

Key checks:
- Flux-matched metric preserves `T_Δ` and `S(H)` across amplitudes
- Lyapunov and covariance baselines drift more strongly
- Flux residuals remain bounded in all cases

This directly supports Figure-level claims about amplitude invariance.

---

### Spectral Invariance

**`test_10b_spectral_invariance.py`**  
Tests invariance under **spectral changes** in the forcing signal.

The inferred metric must:
- Preserve `T_Δ` within tolerance
- Preserve dissipation geometry `S(H)`

This isolates spectral content from geometric structure.

---

### Ornstein–Uhlenbeck (1D) Invariance

**`test_11a_ou_invariance.py`**  
Validates invariance behavior in the analytically degenerate 1D OU case.

Key points:
- Flux matching does not break trivial invariance
- Residuals remain bounded
- Baselines collapse to identical normalized forms

This test ensures correctness in low-dimensional edge cases.

---

### Coupled Oscillator Invariance

**`test_11b_coupled_invariance.py`**  
Validates invariance for a higher-dimensional coupled oscillator system.

Checks include:
- Preservation of `T_Δ` under spectral forcing changes
- Acceptable flux residuals across regimes

This test bridges low-dimensional theory and multi-DOF physical systems.

## Tests — End-to-End Pipelines, Partial Observation, and Real Data

This section documents the final group of tests, which validate **end-to-end experiment execution**, **partial observation robustness**, **state reconstruction**, and **real-world data ingestion**. Together, these tests demonstrate that deviation spectroscopy is not limited to idealized simulations.

---

### End-to-End Results Runner

**`test_12_results_runner.py`**  
Validates the high-level experiment runner used to generate summary JSON outputs for figures.

This test ensures that:
- Complete pipelines run without error for multiple systems
- Flux-based invariance metrics (`T_rel`) are computed correctly
- Results satisfy invariance thresholds for:
  - Two-mass systems
  - 1D Ornstein–Uhlenbeck systems
  - Coupled oscillators

This test underwrites the reproducibility of all reported result summaries.

---

### Dataset Output Mapping (Partial Observation)

**`test_13_output_map.py`**  
Validates support for explicit output maps `y = C z`.

Checks include:
- Output vectors are included in datasets when `C` is provided
- Output dimensions match expectations
- Output values correspond exactly to projected state components

This enables controlled partial-observation experiments.

---

### Delay Embedding and State Reconstruction

**`test_14_state_reconstruction.py`**  
Validates delay-coordinate embedding and subspace reconstruction.

Key guarantees:
- Correct construction of delay matrices for scalar and vector outputs
- Proper dimensionality of reconstructed state subspaces
- Energy concentration in low-rank embeddings for structured signals
- Robust handling of invalid reconstruction parameters

This test establishes the numerical stability of the reconstruction layer.

---

### System Identification from Reconstructed State

**`test_15_identify_from_reconstruction.py`**  
Validates identification of linear dynamics from reconstructed (ghost) states.

Ensures that:
- Identified `Â, B̂` have correct dimensions
- Residual norms are finite and meaningful
- Identification quality improves with increased data length

This bridges partial observation and flux-based metric inference.

---

### Partial Observation Invariance (Clean)

**`test_16_partial_observation_invariance.py`**  
Validates invariance of the flux-based metric under partial observation without noise.

Pipeline tested:
1. Simulate true physical system
2. Observe only a subset of coordinates
3. Reconstruct ghost state
4. Identify ghost dynamics
5. Fit flux-based metric on dataset A
6. Apply the same metric to dataset B

The test confirms that:
- The stress metric `T_Δ` remains invariant
- Inferred geometry survives loss of state information

---

### Partial Observation Invariance with Measurement Noise

**`test_17_partial_obs_with_noise.py`**  
Extends partial observation tests to include realistic output noise.

Key assertions:
- Invariance of `T_Δ` survives moderate measurement noise
- Flux residual degradation is consistent across forcing changes
- Absolute closure may degrade, but geometric invariance persists

This test directly supports claims of robustness under noisy sensing.

---

### Generic CSV Time-Series Loader

**`test_18_loader.py`**  
Validates ingestion of generic CSV time-series data.

Capabilities tested:
- Parsing of time, output, and input columns
- Automatic removal of NaNs
- Polynomial detrending of slow drifts

This enables application to arbitrary experimental datasets.

---

### Power Grid Frequency Data Loader

**`test_19_powergrid_frequency_loader.py`**  
Validates ingestion of real-world power grid frequency data.

Ensures that:
- Timestamped CSV files (including `.csv.zip`) are parsed correctly
- Units are converted consistently (mHz → Hz)
- Data can be resampled to a uniform time grid

This test underwrites the real-data power grid experiments reported in the paper.

## Testing Philosophy and Scope

All tests in this repository are **claim-verification tests**, not exploratory analyses.

They are designed to:
- enforce mathematical correctness,
- validate numerical stability,
- confirm invariance and robustness claims under controlled perturbations,
- and prevent regression of physical guarantees (SPD geometry, flux closure, invariance).

No test introduces new methodology or empirical conclusions beyond what is described
in the experiment and theory sections. All scientific claims originate in the
experiment runners under `experiments/` and are merely *asserted* by the test suite.

This separation ensures that:
- scientific conclusions remain traceable to declared experiments,
- tests serve as executable guarantees rather than hidden evidence,
- and reviewers can audit logic without reverse-engineering test code.

## Author

**Asaad Riaz** Affiliation: [Riaz Communications Inc.](https://riazcommunications.com) (research infrastructure)  
Email: asaad@quantumhub.solutions

## License

This project is licensed under the **Apache License 2.0**. See the [LICENSE](LICENSE) file for details.

**Copyright 2025 Asad N. Riaz (aka Asaad Riaz)**

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.