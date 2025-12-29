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

INP = Path("results/discovery_test_partial_obs/discovery_test_partial_obs_summary.json")
OUTDIR = Path("results/figures_paper")
OUTDIR.mkdir(parents=True, exist_ok=True)


def main():
    if not INP.exists():
        raise FileNotFoundError(
            f"Missing {INP}. Run discovery test partial observation first."
        )

    with open(INP, "r") as f:
        data = json.load(f)

    conds = data["conditions"]
    labels = [c["name"] for c in conds]


    drift = np.array(
        [c["invariance_vs_base"]["rel_fro_error"] for c in conds],
        dtype=float,
    )

    x = np.arange(len(labels))
    plt.figure(figsize=(10, 4))
    plt.bar(x, drift, edgecolor="k", alpha=0.9)
    plt.xticks(x, labels, rotation=15)

    plt.ylabel(r"Scale-aligned drift  $||H_0 - \alpha H^*||_F \,/\, ||H_0||_F$")

    plt.title(
        "Discovery Test (Partial Observation): "
        "Inferred Metric Invariance Under Forcing Variation"
    )
    plt.yscale("log")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    out = OUTDIR / "Figure_Discovery_PartialObs_Invariance.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"[saved] {out}")


    T = np.array([c["flux"]["T_delta"] for c in conds], dtype=float)

    plt.figure(figsize=(10, 4))
    plt.bar(x, T, edgecolor="k", alpha=0.9)
    plt.xticks(x, labels, rotation=15)
    plt.ylabel(r"$T_\Delta$")
    plt.title("Partial Observation: Invariant Stress Metric Across Forcing Conditions")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    out = OUTDIR / "Figure_Discovery_PartialObs_Tdelta.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"[saved] {out}")


    R = np.array([c["flux"]["residual_ratio"] for c in conds], dtype=float)

    plt.figure(figsize=(10, 4))
    plt.bar(x, R, edgecolor="k", alpha=0.9)
    plt.xticks(x, labels, rotation=15)
    plt.yscale("log")
    plt.ylabel(r"Residual ratio  $|r| \,/\, |\Delta E|$")
    plt.title("Partial Observation: Flux Closure Residuals")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    out = OUTDIR / "Figure_Discovery_PartialObs_Residual.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()