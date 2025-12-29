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
import csv
import numpy as np

from deviation_spectroscopy.data.powergrid_frequency import (
    load_powergrid_frequency,
    resample_uniform,
    split_two_regimes,
)
from deviation_spectroscopy.id.state_reconstruct import SubspaceProjector
from deviation_spectroscopy.id.identify_from_reconstruction import identify_ab_from_reconstructed_state
from deviation_spectroscopy.flux.flux_match import fit_H_flux_matching
from deviation_spectroscopy.tests.helpers_invariance import evaluate_H_on_dataset


class _DS:
    def __init__(self, name, t, z, u):
        self.name = name
        self.t = t
        self.z = z
        self.u = u


DT = 1.0
ORDER = 6
HORIZON = 25
SMOOTH_Y = 9
SMOOTH_Z = 9
WINDOW = 400
STRIDE = 400
MAXITER = 1000

REGIME_A = (2 * 3600, 6 * 3600)
REGIME_B = (6 * 3600, 10 * 3600)


def run_one(path: Path) -> dict:
    data = load_powergrid_frequency(path)
    data = resample_uniform(data, dt=DT)
    Areg, Breg = split_two_regimes(data, tA=REGIME_A, tB=REGIME_B)

    def prep(ts):
        y = ts.y
        u = np.zeros((y.shape[0], 1))
        return ts.t, y, u

    tA, yA, uA = prep(Areg)
    tB, yB, uB = prep(Breg)

    proj = SubspaceProjector(
        order=ORDER,
        horizon=HORIZON,
        scale="sqrt",
        smooth_window=SMOOTH_Y,
    )

    zA = proj.fit_transform(yA)
    zB = proj.transform(yB)

    N = min(len(zA), len(zB))
    tA, tB = tA[:N], tB[:N]
    zA, zB = zA[:N], zB[:N]
    uA, uB = uA[:N], uB[:N]

    idres = identify_ab_from_reconstructed_state(
        t=tA,
        z_hat=zA,
        u=uA,
        reg=1e-6,
        smooth_window=SMOOTH_Z,
    )

    flux = fit_H_flux_matching(
        t=tA,
        z=zA,
        u=uA,
        A=idres.A_hat,
        B=idres.B_hat,
        window=WINDOW,
        stride=STRIDE,
        maxiter=MAXITER,
    )

    dsA = _DS("A", tA, zA, uA)
    dsB = _DS("B", tB, zB, uB)

    evalA = evaluate_H_on_dataset(
        dataset=dsA,
        A=idres.A_hat,
        B=idres.B_hat,
        H=flux.H,
        window=WINDOW,
        stride=STRIDE,
    )

    evalB = evaluate_H_on_dataset(
        dataset=dsB,
        A=idres.A_hat,
        B=idres.B_hat,
        H=flux.H,
        window=WINDOW,
        stride=STRIDE,
    )

    T_rel = abs(evalA["T_delta"] - evalB["T_delta"]) / max(abs(evalA["T_delta"]), 1e-15)
    res_rel = abs(evalA["residual_ratio"] - evalB["residual_ratio"]) / max(abs(evalA["residual_ratio"]), 1e-15)

    return {
        "file": str(path),
        "T_A": evalA["T_delta"],
        "T_B": evalB["T_delta"],
        "T_rel": T_rel,
        "res_A": evalA["residual_ratio"],
        "res_B": evalB["residual_ratio"],
        "res_rel": res_rel,
        "min_eig_H": flux.min_eig_H,
        "success": flux.success,
    }


def main():
    root = Path("src/deviation_spectroscopy/data/realdata/powergrid")
    outdir = Path("results/public_grid_batch")
    outdir.mkdir(parents=True, exist_ok=True)

    rows = []

    for path in sorted(root.rglob("*")):
        if path.suffix not in {".zip", ".csv"}:
            continue

        try:
            print(f"[run] {path}")
            rows.append(run_one(path))
        except Exception as e:
            print(f"[skip] {path} :: {e}")

    if not rows:
        raise RuntimeError(
            "Batch run produced no valid results. "
            "Check earlier error messages."
        )

    csv_path = outdir / "public_grid_batch_summary.csv"

    if not rows:
        raise RuntimeError(
            "Batch run produced no valid results. "
            "Check earlier error messages."
        )

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    json_path = outdir / "public_grid_batch_summary.json"
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2)

    print(f"[saved] {csv_path}")
    print(f"[saved] {json_path}")


if __name__ == "__main__":
    main()