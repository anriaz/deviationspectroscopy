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
import csv
from pathlib import Path
import numpy as np

INP = Path("results/level3_suite/level3_suite_summary.json")
OUTDIR = Path("results/tables")
OUTDIR.mkdir(parents=True, exist_ok=True)


def main():
    if not INP.exists():
        raise FileNotFoundError(f"Missing {INP}. Run run_level3_suite first.")

    with open(INP, "r") as f:
        data = json.load(f)

    drift_ok = float(data["meta"]["DRIFT_OK"])
    drift_warn = float(data["meta"]["DRIFT_WARN"])

    rows = []

    dsum = data["tests"]["duffing_lifted_summary"]
    rows.append(("Duffing (lifted)", dsum["n"], dsum["pass_5pct"], dsum["pass_15pct"], dsum["mean"], dsum["std"]))

    for obs in ["tanh", "square"]:
        s = data["tests"][f"nonlinear_observation_{obs}_summary"]
        rows.append((f"NonlinearObs ({obs})", s["n"], s["pass_5pct"], s["pass_15pct"], s["mean"], s["std"]))

    rs = data["tests"]["regime_switch_summary"]
    rows.append(("RegimeSwitch (A vs B drift)", rs["n"], "", "", rs["mean_drift_A_vs_B"], rs["std_drift_A_vs_B"]))

    out_csv = OUTDIR / "level3_suite_table.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["test", "n", "pass<=5%", "pass<=15%", "mean_drift", "std_drift"])
        for r in rows:
            w.writerow(r)

    out_tex = OUTDIR / "level3_suite_table.tex"
    with open(out_tex, "w") as f:
        f.write("\\begin{table}[t]\n\\centering\n")
        f.write("\\caption{Level-3 robustness and sensitivity tests (multi-seed).}\n")
        f.write("\\begin{tabular}{lrrrrr}\n\\hline\n")
        f.write("Test & $n$ & Pass $\\leq 5\\%$ & Pass $\\leq 15\\%$ & Mean drift & Std drift \\\\\n\\hline\n")
        for test, n, p5, p15, mu, sd in rows:
            def fmt(x):
                if x == "": return ""
                if isinstance(x, str): return x
                if isinstance(x, (int, np.integer)): return str(int(x))
                return f"{float(x):.3g}"
            f.write(f"{test} & {fmt(n)} & {fmt(p5)} & {fmt(p15)} & {fmt(mu)} & {fmt(sd)} \\\\\n")
        f.write("\\hline\n\\end{tabular}\n\\end{table}\n")

    print(f"[saved] {out_csv}")
    print(f"[saved] {out_tex}")


if __name__ == "__main__":
    main()