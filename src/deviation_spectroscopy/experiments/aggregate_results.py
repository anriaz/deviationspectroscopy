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
import csv


RESULTS_DIR = Path("results/invariance")
OUT_DIR = Path("results/summary")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_results():
    results = []
    for path in RESULTS_DIR.glob("*.json"):
        with open(path, "r") as f:
            results.append(json.load(f))
    return results


def write_csv(results):
    csv_path = OUT_DIR / "invariance_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "system",
            "T_delta_A",
            "T_delta_B",
            "T_rel_flux",
            "residual_A",
            "residual_B",
            "T_rel_lyapunov",
            "T_rel_covariance",
        ])

        for r in results:
            writer.writerow([
                r["system"],
                r["flux"]["T_delta_A"],
                r["flux"]["T_delta_B"],
                r["flux"]["T_rel"],
                r["flux"]["residual_A"],
                r["flux"]["residual_B"],
                r["lyapunov"]["T_rel"],
                r["covariance"]["T_rel"],
            ])

    print(f"Wrote {csv_path}")


def write_json(results):
    out_path = OUT_DIR / "invariance_summary.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    results = load_results()
    write_csv(results)
    write_json(results)