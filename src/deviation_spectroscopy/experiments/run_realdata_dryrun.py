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
from dataclasses import dataclass
from pathlib import Path
import csv
import numpy as np

from deviation_spectroscopy.data.loader import load_csv_timeseries, preprocess_detrend
from deviation_spectroscopy.sim.integrate import simulate_lti_rk4
from deviation_spectroscopy.sim.forcing import sine_force
from deviation_spectroscopy.id.state_reconstruct import SubspaceProjector
from deviation_spectroscopy.id.identify_from_reconstruction import identify_ab_from_reconstructed_state
from deviation_spectroscopy.flux.flux_match import fit_H_flux_matching
from deviation_spectroscopy.tests.helpers_invariance import evaluate_H_on_dataset


Array = np.ndarray


@dataclass
class _DS:
    name: str
    t: Array
    z: Array
    u: Array


def _write_csv(path: Path, t: Array, y: Array, u: Array | None = None):
    y = np.asarray(y, dtype=float)
    if y.ndim == 1:
        y = y.reshape(-1, 1)

    if u is not None:
        u = np.asarray(u, dtype=float)
        if u.ndim == 1:
            u = u.reshape(-1, 1)

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        header = ["t"] + [f"y{i}" for i in range(y.shape[1])]
        if u is not None:
            header += [f"u{i}" for i in range(u.shape[1])]
        w.writerow(header)

        for k in range(t.shape[0]):
            row = [float(t[k])] + [float(v) for v in y[k]]
            if u is not None:
                row += [float(v) for v in u[k]]
            w.writerow(row)


def _add_output_noise(y: Array, *, snr_db: float, seed: int = 0) -> Array:
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=float)
    power = float(np.mean(y * y))
    if power <= 0:
        return y.copy()
    snr = 10.0 ** (snr_db / 10.0)
    noise_power = power / snr
    noise = rng.standard_normal(y.shape) * np.sqrt(noise_power)
    return y + noise


def main():
    outdir = Path("results/realdata_dryrun")
    outdir.mkdir(parents=True, exist_ok=True)

    from deviation_spectroscopy.systems.coupled_oscillator import make_coupled_oscillator
    sys = make_coupled_oscillator()
    A_true, B_true = sys.A, sys.B
    n = A_true.shape[0]
    z0 = np.zeros(n)

    C = np.zeros((1, n))
    C[0, 0] = 1.0

    tA = np.linspace(0.0, 20.0, 5001)
    tB = np.linspace(0.0, 20.0, 5001)

    simA = simulate_lti_rk4(A_true, B_true, tA, z0, lambda tt: sine_force(tt, 1.0, 1.1))
    simB = simulate_lti_rk4(A_true, B_true, tB, z0, lambda tt: sine_force(tt, 1.7, 2.0))

    yA = (simA.z @ C.T)
    yB = (simB.z @ C.T)

    yA_noisy = _add_output_noise(yA, snr_db=20.0, seed=123)
    yB_noisy = _add_output_noise(yB, snr_db=20.0, seed=456)

    csvA = outdir / "A.csv"
    csvB = outdir / "B.csv"
    _write_csv(csvA, simA.t, yA_noisy, u=simA.u)   
    _write_csv(csvB, simB.t, yB_noisy, u=simB.u)

    dsA = load_csv_timeseries(csvA, time_col="t", data_cols=["y0"], input_cols=["u0"])
    dsB = load_csv_timeseries(csvB, time_col="t", data_cols=["y0"], input_cols=["u0"])

    dsA = preprocess_detrend(dsA, poly_order=1)
    dsB = preprocess_detrend(dsB, poly_order=1)

    horizon = 25
    order = min(6, n + 2)

    proj = SubspaceProjector(order=order, horizon=horizon, scale="sqrt", smooth_window=9)
    zhat_A = proj.fit_transform(dsA.y)
    zhat_B = proj.transform(dsB.y)

    N_eff = zhat_A.shape[0]
    tA_eff = dsA.t[:N_eff]
    uA_eff = dsA.u[:N_eff] if dsA.u is not None else np.zeros((N_eff, 1))
    tB_eff = dsB.t[:N_eff]
    uB_eff = dsB.u[:N_eff] if dsB.u is not None else np.zeros((N_eff, 1))

    id_res = identify_ab_from_reconstructed_state(
        t=tA_eff,
        z_hat=zhat_A,
        u=uA_eff,
        reg=1e-6,
        smooth_window=9,
    )
    A_hat, B_hat = id_res.A_hat, id_res.B_hat

    flux_res = fit_H_flux_matching(
        t=tA_eff,
        z=zhat_A,
        u=uA_eff,
        A=A_hat,
        B=B_hat,
        window=400,
        stride=400,
        maxiter=350,
    )
    H_star = flux_res.H

    dsA_hat = _DS("A_reconstructed", tA_eff, zhat_A, uA_eff)
    dsB_hat = _DS("B_reconstructed", tB_eff, zhat_B, uB_eff)

    evalA = evaluate_H_on_dataset(dataset=dsA_hat, A=A_hat, B=B_hat, H=H_star, window=400, stride=400)
    evalB = evaluate_H_on_dataset(dataset=dsB_hat, A=A_hat, B=B_hat, H=H_star, window=400, stride=400)

    T_rel = abs(evalA["T_delta"] - evalB["T_delta"]) / max(abs(evalA["T_delta"]), 1e-12)
    rr_rel = abs(evalA["residual_ratio"] - evalB["residual_ratio"]) / max(abs(evalA["residual_ratio"]), 1e-12)

    print("=== REALDATA DRY RUN (CSV -> loader -> pipeline) ===")
    print(f"T_delta(A) = {evalA['T_delta']:.6g}")
    print(f"T_delta(B) = {evalB['T_delta']:.6g}")
    print(f"T_rel      = {T_rel:.6g}")
    print(f"res(A)     = {evalA['residual_ratio']:.6g}")
    print(f"res(B)     = {evalB['residual_ratio']:.6g}")
    print(f"res_rel    = {rr_rel:.6g}")
    print(f"[H*] min_eig(H) = {flux_res.min_eig_H:.3e}, min_eig(S(H)) = {flux_res.min_eig_S:.3e}")
    print(f"[H*] objective  = {flux_res.objective:.6g}, iters = {flux_res.n_iter}, success = {flux_res.success}")

    summary_path = outdir / "dryrun_summary.txt"
    with open(summary_path, "w") as f:
        f.write("REALDATA DRY RUN (CSV -> loader -> pipeline)\n")
        f.write(f"T_delta(A) = {evalA['T_delta']:.12g}\n")
        f.write(f"T_delta(B) = {evalB['T_delta']:.12g}\n")
        f.write(f"T_rel      = {T_rel:.12g}\n")
        f.write(f"res(A)     = {evalA['residual_ratio']:.12g}\n")
        f.write(f"res(B)     = {evalB['residual_ratio']:.12g}\n")
        f.write(f"res_rel    = {rr_rel:.12g}\n")
        f.write(f"min_eig(H) = {flux_res.min_eig_H:.12g}\n")
        f.write(f"min_eig(S) = {flux_res.min_eig_S:.12g}\n")
        f.write(f"objective  = {flux_res.objective:.12g}\n")
        f.write(f"iters      = {flux_res.n_iter}\n")
        f.write(f"success    = {flux_res.success}\n")
        f.write(f"message    = {flux_res.message}\n")

    print(f"[saved] {summary_path}")


if __name__ == "__main__":
    main()