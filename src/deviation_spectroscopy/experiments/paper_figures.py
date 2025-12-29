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

RESULTS_SIM_DIR = Path("results/invariance")
RESULTS_REAL_BATCH = Path("results/public_grid_batch/public_grid_batch_summary.json")
FIG_DIR = Path("results/figures_paper")
FIG_DIR.mkdir(parents=True, exist_ok=True)

def load_sim_results():
    data = {}
    if not RESULTS_SIM_DIR.exists(): return data
    for p in RESULTS_SIM_DIR.glob("*.json"):
        with open(p, "r") as f:
            res = json.load(f)
            sys_name = res["system"]
            if sys_name == "ou_1d": label = "1D OU (Sim)"
            elif sys_name == "two_mass": label = "2-Mass (Sim)"
            elif sys_name == "coupled_oscillator": label = "Coupled Osc (Sim)"
            else: label = sys_name
            data[label] = {
                "Flux-Matching": res["flux"]["T_rel"],
                "Lyapunov": res["lyapunov"]["T_rel"],
                "Covariance": res["covariance"]["T_rel"]
            }
    return data

def load_real_results():
    data = {}
    if not RESULTS_REAL_BATCH.exists(): return data
    with open(RESULTS_REAL_BATCH, "r") as f:
        batch_res = json.load(f)

    for item in batch_res:
        fname = item["file"]
        if "germany" in fname: 
            label = "Grid: Germany (Stable)"
            category = "invariant"
        elif "uk" in fname or "greatbritain" in fname: 
            label = "Grid: UK (Stable)"
            category = "invariant"
        elif "finland" in fname:
            label = "Grid: Finland (Stable)"
            category = "invariant"
        elif "US_TX" in fname or "us" in fname: 
            label = "Grid: Texas (Volatile)"
            category = "sensitive" 
        else: 
            label = "Grid: Unknown"
            category = "invariant"

        data[label] = {
            "val": item["T_rel"],
            "category": category
        }
    return data

def plot_combined_results(sim_data, real_data):

    valid_labels = ["1D OU (Sim)", "2-Mass (Sim)", "Coupled Osc (Sim)", 
                    "Grid: Germany (Stable)", "Grid: UK (Stable)", "Grid: Finland (Stable)"]
    
    methods = ["Flux-Matching", "Lyapunov", "Covariance"]
    colors = ["#2ca02c", "#d62728", "#1f77b4"] 
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [3, 1]})
    
    x = np.arange(len(valid_labels))
    width = 0.25
    
    for i, method in enumerate(methods):
        y_vals = []
        for label in valid_labels:
            if label in sim_data:
                y_vals.append(sim_data[label].get(method, np.nan))
            elif label in real_data:
                if method == "Flux-Matching":
                    y_vals.append(real_data[label]["val"])
                else:
                    y_vals.append(np.nan)
            else:
                y_vals.append(np.nan)
        
        y_vals = np.array(y_vals, dtype=float)
        mask = np.isfinite(y_vals)
        if np.any(mask):
            ax1.bar(x[mask] + (i-1)*width, y_vals[mask], width, label=method, color=colors[i], edgecolor='k')

    ax1.axhline(0.15, color='gray', linestyle='--', label="Success Threshold")
    ax1.set_title("A. Invariance Verification (Stable Regimes)", fontsize=12, fontweight='bold')
    ax1.set_ylabel("Invariance Drift ($|T_A - T_B|/T_A$)", fontsize=11)
    ax1.set_xticks(x)
    ax1.set_xticklabels(valid_labels, rotation=15, ha='right')
    ax1.set_ylim(0, 0.45)
    ax1.legend(loc='upper left')
    ax1.grid(axis='y', alpha=0.3)

    
    tex_label = "Grid: Texas (Volatile)"
    if tex_label in real_data:
        val = real_data[tex_label]["val"]
        display_val = min(val, 1.5) 
        
        bar = ax2.bar([0], [display_val], width=0.5, color="#2ca02c", edgecolor='k')
        
        ax2.text(0, display_val + 0.05, f"Drift: {val:.0f}x\n(Structural Change)", 
                 ha='center', va='bottom', fontweight='bold', color="#d62728")
        
        ax2.set_xticks([0])
        ax2.set_xticklabels(["Grid: Texas\n(Regime Change)"])
        ax2.set_title("B. Sensitivity (Structural Shift)", fontsize=12, fontweight='bold')
        ax2.set_ylim(0, 2.0)
        ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    out_path = FIG_DIR / "Figure1_Discovery_Final.png"
    plt.savefig(out_path, dpi=300)
    print(f"Saved Figure 1 to {out_path}")

if __name__ == "__main__":
    sims = load_sim_results()
    real = load_real_results()
    plot_combined_results(sims, real)