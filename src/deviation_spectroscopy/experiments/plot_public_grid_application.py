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

SUMMARY_JSON = Path("results/public_grid_batch/public_grid_batch_summary.json")
OUTDIR = Path("results/figures_application")
OUTDIR.mkdir(parents=True, exist_ok=True)

THRESHOLD = 0.15  


def main():
    if not SUMMARY_JSON.exists():
        raise FileNotFoundError(
            "Batch summary not found. "
            "Run run_public_grid_batch first."
        )

    with open(SUMMARY_JSON, "r") as f:
        rows = json.load(f)

    labels = []
    T_rel = []

    for r in rows:
        fname = r["file"].lower()
        if "germany" in fname:
            labels.append("Germany (CE)")
        elif "greatbritain" in fname or "uk" in fname:
            labels.append("Great Britain")
        elif "finland" in fname:
            labels.append("Finland (Nordic)")
        else:
            labels.append(Path(fname).stem)

        T_rel.append(r["T_rel"])

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.bar(labels, T_rel, color="#2ca02c", edgecolor="k", alpha=0.85)

    ax.axhline(
        THRESHOLD,
        linestyle="--",
        color="gray",
        linewidth=1,
        label="Invariance bound (15%)",
    )

    ax.set_ylabel("Relative invariance drift $|T_A - T_B| / T_A$")
    ax.set_title(
        "Invariant Stress Metric from Real Power-Grid Frequency Data",
        pad=12,
    )

    ax.set_ylim(0, max(THRESHOLD * 1.4, max(T_rel) * 1.2))
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = OUTDIR / "public_grid_application.png"
    plt.savefig(out, dpi=300)
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()