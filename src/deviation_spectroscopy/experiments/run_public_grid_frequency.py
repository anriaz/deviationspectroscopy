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
    def __init__(self, name: str, t: np.ndarray, z: np.ndarray, u: np.ndarray):
        self.name = name
        self.t = t
        self.z = z
        self.u = u


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--path", type=str, required=True, help="Path to .csv.zip frequency file")
    ap.add_argument("--dt", type=float, default=1.0, help="Uniform resample dt (seconds)")
    ap.add_argument("--order", type=int, default=6, help="Reconstruction order")
    ap.add_argument("--horizon", type=int, default=25, help="Delay horizon")
    ap.add_argument("--smooth_y", type=int, default=9, help="Odd moving average for y before embedding")
    ap.add_argument("--smooth_z", type=int, default=9, help="Odd moving average for z_hat before ID")
    ap.add_argument("--window", type=int, default=400, help="Flux window (samples)")
    ap.add_argument("--stride", type=int, default=400, help="Flux stride (samples)")
    ap.add_argument("--maxiter", type=int, default=1000, help="Optimizer maxiter")
    ap.add_argument("--outdir", type=str, default="results/public_grid", help="Output directory")
    ap.add_argument("--tA0", type=float, default=2 * 3600, help="Regime A start (sec)")
    ap.add_argument("--tA1", type=float, default=6 * 3600, help="Regime A end (sec)")
    ap.add_argument("--tB0", type=float, default=18 * 3600, help="Regime B start (sec)")
    ap.add_argument("--tB1", type=float, default=22 * 3600, help="Regime B end (sec)")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("=== PUBLIC GRID FREQUENCY RUN (output-only) ===")
    data = load_powergrid_frequency(args.path)
    data = resample_uniform(data, dt=args.dt)
    Areg, Breg = split_two_regimes(data, tA=(args.tA0, args.tA1), tB=(args.tB0, args.tB1))

    def _prep(ts, tag):
        y = ts.y  
        u = np.zeros((y.shape[0], 1), dtype=float)
        return ts.t, y, u, f"{ts.name}_{tag}"

    tA, yA, uA, nameA = _prep(Areg, "A")
    tB, yB, uB, nameB = _prep(Breg, "B")

    proj = SubspaceProjector(order=args.order, horizon=args.horizon, scale="sqrt", smooth_window=args.smooth_y)
    zA = proj.fit_transform(yA)
    zB = proj.transform(yB)

    N = min(zA.shape[0], zB.shape[0])
    tA = tA[:N]
    tB = tB[:N]
    uA = uA[:N]
    uB = uB[:N]
    zA = zA[:N]
    zB = zB[:N]

    idres = identify_ab_from_reconstructed_state(
        t=tA,
        z_hat=zA,
        u=uA,
        reg=1e-6,
        smooth_window=args.smooth_z,
    )
    A_hat = idres.A_hat
    B_hat = idres.B_hat

    flux = fit_H_flux_matching(
        t=tA,
        z=zA,
        u=uA,
        A=A_hat,
        B=B_hat,
        window=args.window,
        stride=args.stride,
        maxiter=args.maxiter,
    )
    H_star = flux.H

    dsA = _DS(nameA, tA, zA, uA)
    dsB = _DS(nameB, tB, zB, uB)

    evalA = evaluate_H_on_dataset(dataset=dsA, A=A_hat, B=B_hat, H=H_star, window=args.window, stride=args.stride)
    evalB = evaluate_H_on_dataset(dataset=dsB, A=A_hat, B=B_hat, H=H_star, window=args.window, stride=args.stride)

    T_rel = abs(evalA["T_delta"] - evalB["T_delta"]) / max(abs(evalA["T_delta"]), 1e-15)
    res_rel = abs(evalA["residual_ratio"] - evalB["residual_ratio"]) / max(abs(evalA["residual_ratio"]), 1e-15)

    summary = {
        "file": str(args.path),
        "dt": args.dt,
        "regimes": {"A": [args.tA0, args.tA1], "B": [args.tB0, args.tB1]},
        "reconstruction": {"order": args.order, "horizon": args.horizon, "smooth_y": args.smooth_y},
        "id": {"smooth_z": args.smooth_z, "residual_norm": float(idres.residual_norm)},
        "flux": {
            "T_A": float(evalA["T_delta"]),
            "T_B": float(evalB["T_delta"]),
            "T_rel": float(T_rel),
            "res_A": float(evalA["residual_ratio"]),
            "res_B": float(evalB["residual_ratio"]),
            "res_rel": float(res_rel),
            "min_eig_H": float(flux.min_eig_H),
            "min_eig_S": float(flux.min_eig_S),
            "objective": float(flux.objective),
            "iters": int(flux.n_iter),
            "success": bool(flux.success),
            "message": str(flux.message),
        },
    }

    out_json = outdir / (Path(args.path).stem + "_summary.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"T_delta(A) = {summary['flux']['T_A']:.6g}")
    print(f"T_delta(B) = {summary['flux']['T_B']:.6g}")
    print(f"T_rel      = {summary['flux']['T_rel']:.6g}")
    print(f"res(A)     = {summary['flux']['res_A']:.6g}")
    print(f"res(B)     = {summary['flux']['res_B']:.6g}")
    print(f"res_rel    = {summary['flux']['res_rel']:.6g}")
    print(f"[H*] min_eig(H) = {summary['flux']['min_eig_H']:.3e}, min_eig(S(H)) = {summary['flux']['min_eig_S']:.3e}")
    print(f"[H*] objective  = {summary['flux']['objective']:.6g}, iters = {summary['flux']['iters']}, success = {summary['flux']['success']}")
    print(f"[saved] {out_json}")


if __name__ == "__main__":
    main()