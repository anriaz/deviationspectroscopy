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
import matplotlib.pyplot as plt
from pathlib import Path

JSON = Path("results/public_grid_batch/public_grid_batch_summary.json")
OUT = Path("results/figures_paper/Figure3_PowerGrid.png")

def main():
    if not JSON.exists():
        print(f"Missing {JSON}")
        return

    with open(JSON) as f:
        rows = json.load(f)

    labels, vals = [], []
    for r in rows:
        name = r["file"].lower()
        if "finland" in name: l = "Finland"
        elif "germany" in name: l = "Germany"
        elif "uk" in name: l = "Great Britain"
        elif "texas" in name or "us" in name: l = "US (Eastern)"
        else: continue
        
        labels.append(l)
        vals.append(r["T_rel"])

    plt.figure(figsize=(7, 4.5))
    
    colors = ['#2ca02c' if v < 0.15 else '#d62728' for v in vals]
    
    plt.bar(labels, vals, color=colors, edgecolor="k", alpha=0.9, width=0.6)

    plt.axhline(0.05, linestyle="--", color="tab:blue", linewidth=1, label="5% target")
    plt.axhline(0.15, linestyle=":", color="tab:blue", linewidth=1.2, label="15% bound")
    
    plt.yscale("log")
    
    plt.ylabel(r"Relative drift $|T_A - T_B|/T_A$ (Log Scale)")
    plt.title("Invariant Stress Metric from Power Grid Frequency Data")
    
    for i, v in enumerate(vals):
        if v > 1.0:
            plt.text(i, v * 1.2, f"{v:.0f}", ha='center', color='#d62728', fontweight='bold')

    plt.ylim(1e-3, max(vals) * 5) 
    plt.grid(axis="y", which="both", alpha=0.2)
    plt.legend(loc="upper left")
    plt.tight_layout()
    
    plt.savefig(OUT, dpi=300)
    plt.close()
    print(f"[saved] {OUT}")

if __name__ == "__main__":
    main()