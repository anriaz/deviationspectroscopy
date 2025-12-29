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
import csv

INP = Path("results/discovery_test_ou/discovery_test_ou_summary.json")
OUTDIR = Path("results/tables")
OUTDIR.mkdir(parents=True, exist_ok=True)

def main():
    if not INP.exists():
        raise FileNotFoundError(f"Missing {INP}. Run run_discovery_test_ou first.")

    with open(INP, "r") as f:
        data = json.load(f)

    stats = data["drift_stats"]

    csv_path = OUTDIR / "discovery_test_ou_table.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "median", "p90", "p95", "max", "mean"])
        for k in ["w1_a1", "w2_a1", "w1_a2", "w2_a2"]:
            s = stats[k]
            w.writerow([k, s["median"], s["p90"], s["p95"], s["max"], s["mean"]])
    print(f"[saved] {csv_path}")

    tex_path = OUTDIR / "discovery_test_ou_table.tex"
    lines = []
    lines.append(r"\begin{tabular}{lccccc}")
    lines.append(r"\toprule")
    lines.append(r"Condition & Median & P90 & P95 & Max & Mean \\")
    lines.append(r"\midrule")
    for k in ["w1_a1", "w2_a1", "w1_a2", "w2_a2"]:
        s = stats[k]
        lines.append(
            rf"{k} & {s['median']:.3e} & {s['p90']:.3e} & {s['p95']:.3e} & {s['max']:.3e} & {s['mean']:.3e} \\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    with open(tex_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[saved] {tex_path}")

if __name__ == "__main__":
    main()