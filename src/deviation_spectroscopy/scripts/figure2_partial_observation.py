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
import matplotlib.pyplot as plt

INP = Path("results/partial_obs_clean_vs_noisy/partial_obs_clean_vs_noisy_summary.json")
OUTDIR = Path("results/figures_paper")
OUTDIR.mkdir(parents=True, exist_ok=True)

THRESHOLD = 0.15


def main():
    if not INP.exists():
        raise FileNotFoundError(
            f"Missing {INP}. Run experiments/run_partial_obs_clean_vs_noisy.py first."
        )

    with open(INP, "r") as f:
        data = json.load(f)

    clean_med = float(data["clean"]["summary"]["drift"]["median"])
    noisy_med = float(data["noisy"]["summary"]["drift"]["median"])
    sigma = float(data["noise"]["sigma"])

    labels = ["Clean", "Noisy"]
    vals = [clean_med, noisy_med]

    plt.figure(figsize=(6.5, 4.2))
    
    plt.bar(labels, vals, edgecolor="k", alpha=0.85, width=0.6)
    
    plt.axhline(0.05, linestyle="--", color="tab:blue", linewidth=1.0, label="5% target")
    plt.axhline(THRESHOLD, linestyle=":", color="tab:blue", linewidth=1.2, label="15% bound")

    plt.yscale("log")  
    plt.ylabel("Relative drift (Log Scale)")
    
    plt.title(f"Partial Observation: Invariance Under Observation Noise (sigma={sigma:g})")
    
    plt.ylim(min(vals)*0.5, 0.5) 
    
    plt.grid(axis="y", which="both", alpha=0.25)
    plt.legend(loc="upper left")
    plt.tight_layout()

    out = OUTDIR / "Figure2_PartialObservation.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()