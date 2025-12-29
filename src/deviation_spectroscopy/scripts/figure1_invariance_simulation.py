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
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt

RESULTS = Path("results/invariance")
FIG_DIR = Path("results/figures_paper")
FIG_DIR.mkdir(parents=True, exist_ok=True)
OUT = FIG_DIR / "Figure1_Discovery.png"

THRESHOLD = 0.15

def main():
    systems = []
    flux = []
    lyap = []
    cov = []

    for p in sorted(RESULTS.glob("*.json")):
        with open(p) as f:
            r = json.load(f)

        name = r["system"]
        if name == "ou_1d":
            label = "1D OU"
        elif name == "two_mass":
            label = "2-Mass"
        elif name == "coupled_oscillator":
            label = "Coupled Osc"
        else:
            continue

        systems.append(label)
        flux.append(r["flux"]["T_rel"])
        lyap.append(r["lyapunov"]["T_rel"])
        cov.append(r["covariance"]["T_rel"])

    x = np.arange(len(systems))
    w = 0.25
    
    plt.figure(figsize=(8, 5))

    plt.bar(x - w, flux, w, label="Flux-matched $H^*$", color="#2ca02c", edgecolor='k', alpha=0.9)
    plt.bar(x, lyap, w, label="Lyapunov $H$", color="#d62728", edgecolor='k', alpha=0.9)
    plt.bar(x + w, cov, w, label="Covariance $H$", color="#1f77b4", edgecolor='k', alpha=0.9)

    plt.axhline(0.05, linestyle="--", color="tab:blue", linewidth=1, label="5% target")
    plt.axhline(THRESHOLD, linestyle=":", color="tab:blue", linewidth=1.2, label="15% bound")

    plt.yscale("log")
    
    plt.xticks(x, systems, fontsize=11)
    plt.ylabel(r"Relative drift $|T_A - T_B|/T_A$ (Log Scale)", fontsize=11)
    plt.title("Invariant Dissipative Geometry Across Excitation Regimes", fontsize=12)
    
    plt.ylim(1e-16, 1.0) 
    
    plt.grid(axis="y", which="both", alpha=0.2)
    
    plt.legend(loc="upper right", framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(OUT, dpi=300)
    plt.close()

    print(f"[saved] {OUT}")

if __name__ == "__main__":
    main()