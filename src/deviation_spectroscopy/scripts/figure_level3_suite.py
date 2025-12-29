# Copyright 2025 Asaad Riaz
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

INP = Path("results/level3_suite/level3_suite_summary.json")
OUTDIR = Path("results/figures_paper")
OUTDIR.mkdir(parents=True, exist_ok=True)


def _collect_drifts_from_group(group):
    drifts = []
    for seed, blob in group.items():
        for r in blob["runs"]:
            inv = r.get("invariance_vs_base", None)
            if inv is None:
                continue
            drifts.append(float(inv["rel_fro_error"]))
    return np.array(drifts, dtype=float)


def main():
    if not INP.exists():
        raise FileNotFoundError(f"Missing {INP}. Run run_level3_suite first.")

    with open(INP, "r") as f:
        data = json.load(f)

    drift_ok = float(data["meta"]["DRIFT_OK"])
    drift_warn = float(data["meta"]["DRIFT_WARN"])

    duffing = _collect_drifts_from_group(data["tests"]["duffing_lifted"])
    tanh = _collect_drifts_from_group(data["tests"]["nonlinear_observation"]["tanh"])
    square = _collect_drifts_from_group(data["tests"]["nonlinear_observation"]["square"])

    labels = ["Duffing (lifted)", "NonlinObs tanh", "NonlinObs square"]
    series = [duffing, tanh, square]

    plt.figure(figsize=(10, 4.5))
    rng = np.random.default_rng(42)

    for i, arr in enumerate(series):
        if arr.size == 0:
            continue
        
        jitter = rng.uniform(-0.08, 0.08, size=arr.size)
        x = i + jitter
        
        plt.scatter(x, arr, s=20, alpha=0.7, edgecolors="none")
        
        med = np.median(arr)
        plt.hlines(med, i - 0.25, i + 0.25, colors='tab:blue', linewidth=3, alpha=0.8)

    plt.axhline(drift_ok, linestyle="--", linewidth=1, label="5% target")
    plt.axhline(drift_warn, linestyle=":", linewidth=1, label="15% bound")
    
    plt.xticks(range(len(labels)), labels, rotation=0) 
    plt.yscale("log")
    
    plt.ylabel("Relative Metric Drift (Log Scale)")
    
    plt.title("Level-3 Robustness: Metric Invariance Under Nonlinearity and Nonlinear Observation")
    plt.grid(axis="y", alpha=0.2)
    plt.legend(loc='upper right')
    plt.tight_layout()

    out = OUTDIR / "Figure_L3_Drift_Distributions.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"[saved] {out}")


    switch_runs = data["tests"]["regime_switch"]["runs"]
    drift_AB = np.array([float(r["drift_A_vs_B"]["rel_fro_error"]) for r in switch_runs], dtype=float)

    plt.figure(figsize=(8, 3.5))
    
    rng = np.random.default_rng(42)
    y_jitter = rng.uniform(-0.04, 0.04, size=len(drift_AB))

    plt.axvline(0.15, color="gray", linestyle="--", alpha=0.5, lw=1.5, label="15% Invariance Bound")

    plt.scatter(drift_AB, y_jitter, s=100, alpha=0.85, edgecolor="k", linewidth=1, zorder=3, label="Detected Drift")

    median_drift = np.median(drift_AB)
    plt.axvline(median_drift, color="tab:red", linestyle=":", alpha=0.8, lw=2, label=f"Median Drift: {median_drift:.2f}")

    plt.xlabel("Scale-aligned drift between regimes (Frobenius Norm)")
    plt.ylabel("") 
    plt.yticks([])
    plt.ylim(-0.15, 0.15) 
    plt.xlim(0.05, 0.26)  
    
    plt.title("Level-3 Sensitivity: Structural Break Detection (Regime Switch)")
    plt.grid(axis="x", alpha=0.3)
    plt.legend(loc="upper right", fontsize="small", framealpha=0.9)
    plt.tight_layout()

    out = OUTDIR / "Figure_L3_RegimeSwitch_Sensitivity.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"[saved] {out}")


    def _pass(arr, thr):
        return float(np.mean(arr <= thr)) if arr.size else 0.0

    pass_5 = [_pass(duffing, drift_ok), _pass(tanh, drift_ok), _pass(square, drift_ok)]
    pass_15 = [_pass(duffing, drift_warn), _pass(tanh, drift_warn), _pass(square, drift_warn)]

    x = np.arange(len(labels))
    w = 0.35

    plt.figure(figsize=(9, 4))
    plt.bar(x - w/2, pass_5, w, edgecolor="k", alpha=0.9, label="pass ≤ 5%")
    plt.bar(x + w/2, pass_15, w, edgecolor="k", alpha=0.9, label="pass ≤ 15%")
    plt.xticks(x, labels, rotation=10)
    plt.ylim(0, 1.0)
    plt.ylabel("pass rate")
    plt.title("Level-3 Multi-Seed Robustness: Pass Rates")
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out = OUTDIR / "Figure_L3_PassRates.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()