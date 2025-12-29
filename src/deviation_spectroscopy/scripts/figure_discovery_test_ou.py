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
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

INP = Path("results/discovery_test_ou/discovery_test_ou_summary.json")
OUTDIR = Path("results/figures_paper")
OUTDIR.mkdir(parents=True, exist_ok=True)
OUT = OUTDIR / "Figure_Discovery_OU_Drift.png"

TARGET_5 = 0.05
BOUND_15 = 0.15

def main():
    if not INP.exists():
        print(f"File {INP} not found. Ensure the summary JSON exists.")
        return

    with open(INP, "r") as f:
        data = json.load(f)

    rows = data["rows"]
    plot_names = ["w2_a1", "w1_a2", "w2_a2"]

    dist = {n: [] for n in plot_names}
    for r in rows:
        if r["name"] in plot_names:
            dist[r["name"]].append(r["invariance_vs_base"]["rel_fro_error"])

    plt.figure(figsize=(8, 6))
    
    plt.axhline(TARGET_5, ls="--", color="tab:blue", alpha=0.5, label="5% target")
    plt.axhline(BOUND_15, ls=":", color="tab:blue", alpha=0.5, label="15% bound")

    x_pos = np.arange(len(plot_names))
    rng = np.random.default_rng(0)

    for i, n in enumerate(plot_names):
        y = np.array(dist[n], dtype=float)
        jitter = rng.uniform(-0.08, 0.08, size=y.size)
        
        plt.scatter(x_pos[i] + jitter, y, s=40, alpha=0.8, edgecolors='none', zorder=3)
        plt.hlines(np.median(y), x_pos[i] - 0.2, x_pos[i] + 0.2, 
                   linewidth=3, color='tab:blue', alpha=0.8, zorder=4)

    plt.yscale("log")
    plt.ylabel("Relative Metric Drift (Log Scale)", fontsize=11)
    
    plt.xticks(x_pos, plot_names, rotation=0, fontsize=11)
    
    plt.title("Discovery Robustness under OU Forcing\nInferred Metric Invariance Across Seeds", fontsize=12)
    
    plt.grid(True, which="both", axis="y", ls="-", alpha=0.15, zorder=0)
    plt.legend(loc="upper right", framealpha=0.6, fontsize='small')

    plt.tight_layout()
    plt.savefig(OUT, dpi=300)
    print(f"[saved] {OUT}")

if __name__ == "__main__":
    main()