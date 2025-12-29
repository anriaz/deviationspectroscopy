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
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt

IN = Path("results/discovery_test/discovery_test_summary.json")
OUTDIR = Path("results/figures_paper")
OUTDIR.mkdir(parents=True, exist_ok=True)

THRESH_INV = 0.05  


def main():
    if not IN.exists():
        raise FileNotFoundError(f"Missing {IN}. Run discovery test first.")

    with open(IN, "r") as f:
        R = json.load(f)

    conds = R["conditions"]
    labels = [c["name"] for c in conds]
    inv_err = np.array([c["invariance_vs_base"]["rel_fro_error"] for c in conds], float)
    rec_err = np.array([c["recovery_vs_truth"]["rel_fro_error"] for c in conds], float)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(labels))
    ax.bar(x, inv_err, edgecolor="k", alpha=0.9)
    ax.axhline(THRESH_INV, linestyle="--", linewidth=1.2, label=f"Invariance target ({THRESH_INV:.0%})")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel(r"$\|H^\star_0 - s H^\star\|_F / \|H^\star_0\|_F$")
    ax.set_title("Discovery Test: Inferred Metric Invariance Under Forcing Variation")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    plt.tight_layout()

    out1 = OUTDIR / "Figure_Discovery_Invariance.png"
    plt.savefig(out1, dpi=300)
    plt.close()
    print(f"[saved] {out1}")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x, rec_err, edgecolor="k", alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel(r"$\|H_{\mathrm{true}} - s H^\star\|_F / \|H_{\mathrm{true}}\|_F$")
    ax.set_title("Discovery Test: Recovery of Physical Quadratic Energy Metric (Scale-Aligned)")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    out2 = OUTDIR / "Figure_Discovery_Recovery.png"
    plt.savefig(out2, dpi=300)
    plt.close()
    print(f"[saved] {out2}")


if __name__ == "__main__":
    main()