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

SUMMARY_PATH = Path("results/summary/invariance_summary.json")
FIG_DIR = Path("results/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

NUMERICAL_FLOOR = 1e-12
TYPICAL_NONINVARIANT = 1e-2


def load():
    with open(SUMMARY_PATH, "r") as f:
        return json.load(f)


def plot_T_delta(results):
    systems = [r["system"] for r in results]
    errors = [
        abs(r["flux"]["T_delta_A"] - r["flux"]["T_delta_B"])
        / (r["flux"]["T_delta_A"] + 1e-16)
        for r in results
    ]

    x = np.arange(len(systems))
    w = 0.5

    plt.figure()
    plt.bar(x, errors, w, color="tab:purple")
    plt.xticks(x, systems, rotation=20)
    plt.yscale("log")

    plt.axhline(NUMERICAL_FLOOR, ls="--", lw=1, color="gray")
    plt.text(
        len(systems) - 0.5,
        NUMERICAL_FLOOR * 1.3,
        "Numerical floor",
        ha="right",
        va="bottom",
        fontsize=9,
        color="gray",
    )

    plt.ylabel(r"Relative Error $|T_{\Delta,A} - T_{\Delta,B}| / T_{\Delta,A}$")
    plt.title(r"T$_\Delta$ Invariance (Relative Error)")
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.tight_layout()

    out = FIG_DIR / "T_delta_invariance.png"
    plt.savefig(out)
    plt.close()
    print(f"Saved {out}")


def plot_residuals(results):
    systems = [r["system"] for r in results]
    rA = [r["flux"]["residual_A"] for r in results]
    rB = [r["flux"]["residual_B"] for r in results]

    x = np.arange(len(systems))
    w = 0.35

    plt.figure()
    plt.bar(x - w / 2, rA, w, label="Residual A")
    plt.bar(x + w / 2, rB, w, label="Residual B")
    plt.xticks(x, systems, rotation=20)
    plt.yscale("log")

    plt.ylabel(r"Residual ratio $|\dot{E}_{err}|/|\dot{E}_{tot}|$")
    plt.title("Flux-Closure Residuals")
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.tight_layout()

    out = FIG_DIR / "residuals.png"
    plt.savefig(out)
    plt.close()
    print(f"Saved {out}")


def plot_baseline_comparison(results):
    systems = [r["system"] for r in results]
    flux = [r["flux"]["T_rel"] for r in results]
    lyap = [r["lyapunov"]["T_rel"] for r in results]
    cov = [r["covariance"]["T_rel"] for r in results]

    x = np.arange(len(systems))
    w = 0.25

    plt.figure()
    plt.bar(x - w, flux, w, label="Flux H*")
    plt.bar(x, lyap, w, label="Lyapunov H")
    plt.bar(x + w, cov, w, label="Covariance H")
    plt.xticks(x, systems, rotation=20)
    plt.yscale("log")

    plt.axhline(
        TYPICAL_NONINVARIANT,
        ls="--",
        lw=1,
        color="gray",
        alpha=0.7,
    )
    plt.text(
        len(systems) - 0.5,
        TYPICAL_NONINVARIANT * 1.3,
        "Typical non-invariant drift",
        ha="right",
        va="bottom",
        fontsize=9,
        color="gray",
    )

    plt.ylabel(r"Relative drift in $T_\Delta$ (log scale)")
    plt.title("Invariant Metric Comparison Across Systems")
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.tight_layout()

    out = FIG_DIR / "baseline_comparison.png"
    plt.savefig(out)
    plt.close()
    print(f"Saved {out}")


if __name__ == "__main__":
    results = load()
    plot_T_delta(results)
    plot_residuals(results)
    plot_baseline_comparison(results)