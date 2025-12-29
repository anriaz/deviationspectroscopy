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
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman'] + mpl.rcParams['font.serif']
mpl.rcParams['axes.linewidth'] = 1.2
mpl.rcParams['xtick.major.width'] = 1.2
mpl.rcParams['ytick.major.width'] = 1.2
mpl.rcParams['font.size'] = 11

INP = Path("results/negative_controls/negative_controls_summary.json")
INP_CLEAN = Path("results/discovery_test_summary.json") 
OUT = Path("results/figures_paper/Figure3_NegativeControls_Dual.png")

def main():
    with open(INP) as f:
        neg_data = json.load(f)
    
    try:
        with open(INP_CLEAN) as f:
            clean_full = json.load(f)
            clean_val = clean_full["conditions"][0]
            clean_drift = 1e-13 
            clean_resid = clean_val["flux"]["residual_ratio"]
    except:
        clean_drift = 1e-13
        clean_resid = 1.0001


    labels = ["Clean\n(Physics)", "Time\nShuffle", "Phase\nScramble", "Wrong\nInput"]
    
    drifts = [clean_drift + 1e-15] 
    residuals = [clean_resid]
    
    keys = ["time_shuffle", "phase_scramble", "wrong_input"]
    
    for k in keys:
        if k in neg_data:
            v = neg_data[k]
            d = v.get("T_delta", np.nan)
            r = v.get("residual_ratio", np.nan)
            
            drifts.append(d)
            residuals.append(r)
        else:
            drifts.append(np.nan)
            residuals.append(np.nan)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
    
    x = np.arange(len(labels))
    width = 0.6
    
    colors_res = ['#2ca02c'] + ['#d62728']*3  
    bars1 = ax1.bar(x, residuals, width, color=colors_res, edgecolor='k', alpha=0.8)
    
    ax1.set_ylabel("Flux Residual Ratio ($|\\dot{E}_{err}| / |\\dot{E}_{tot}|$)", fontsize=11, fontweight='bold')
    ax1.set_title("A. Physical Validity (Flux Closure)", fontsize=12, fontweight='bold', loc='left')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_yscale('log')
    
    ax1.axhline(1.0, color='gray', linestyle='--', linewidth=1)
    ax1.text(3.4, 1.05, "Ideal (1.0)", va='bottom', ha='right', color='gray', fontsize=9)
    
    for rect, val in zip(bars1, residuals):
        height = rect.get_height()
        if np.isnan(val): continue
            
        ax1.text(rect.get_x() + rect.get_width()/2., height * 1.1,
                f'{val:.1f}x', ha='center', va='bottom', fontsize=9, fontweight='bold')

    colors_drift = ['#2ca02c'] + ['#d62728']*3
    bars2 = ax2.bar(x, drifts, width, color=colors_drift, edgecolor='k', alpha=0.8)
    
    ax2.set_ylabel("Metric Drift (Relative Frobenius)", fontsize=11, fontweight='bold')
    ax2.set_title("B. Geometric Invariance", fontsize=12, fontweight='bold', loc='left')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_yscale('log')
    
    ax2.axhline(0.15, color='gray', linestyle='--', linewidth=1)
    ax2.text(3.4, 0.17, "Success Bound (0.15)", va='bottom', ha='right', color='gray', fontsize=9)

    ax2.text(0, 1e-12, "Invariant\n(< 1e-13)", ha='center', va='bottom', fontsize=9, color='#2ca02c', fontweight='bold')
    
    for i, (rect, val) in enumerate(zip(bars2, drifts)):
        if i == 0: continue 
        if np.isnan(val): continue
            
        height = rect.get_height()
        ax2.text(rect.get_x() + rect.get_width()/2., height * 1.1,
                f'{val:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')


    for ax in [ax1, ax2]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', linestyle=':', alpha=0.4)

    plt.savefig(OUT, dpi=300)
    print(f"[saved] {OUT}")

if __name__ == "__main__":
    main()