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
from dataclasses import dataclass
from pathlib import Path
import numpy as np

from deviation_spectroscopy.systems.coupled_oscillator import make_coupled_oscillator
from deviation_spectroscopy.sim.integrate import simulate_lti_rk4
from deviation_spectroscopy.sim.forcing import sine_force

from deviation_spectroscopy.id.state_reconstruct import SubspaceProjector
from deviation_spectroscopy.id.identify_from_reconstruction import identify_ab_from_reconstructed_state
from deviation_spectroscopy.flux.flux_match import fit_H_flux_matching
from deviation_spectroscopy.tests.helpers_invariance import evaluate_H_on_dataset
from deviation_spectroscopy.core.baselines import lyapunov_baseline_H, covariance_baseline_H

Array = np.ndarray


@dataclass
class _DS:
    name: str
    t: Array
    z: Array
    u: Array


def _rel_fro(A: Array, B: Array) -> float:
    num = float(np.linalg.norm(A - B, ord="fro"))
    den = float(np.linalg.norm(A, ord="fro"))
    return num / max(den, 1e-15)


def _best_scale_align(A_ref: Array, A_est: Array) -> tuple[float, float]:
    num = float(np.sum(A_ref * A_est))
    den = float(np.sum(A_est * A_est))
    alpha = num / max(den, 1e-15)
    err = _rel_fro(A_ref, alpha * A_est)
    return alpha, err


def main():
    outdir = Path("results/discovery_test_partial_obs")
    outdir.mkdir(parents=True, exist_ok=True)

    sys = make_coupled_oscillator()
    A_true, B_true = sys.A, sys.B
    n = A_true.shape[0]
    z0 = np.zeros(n)

    C = np.zeros((1, n))
    C[0, 0] = 1.0

    conditions = [
        ("w1_a1", 1.0, 1.0),
        ("w2_a1", 2.0, 1.0),
        ("w1_a2", 1.0, 2.0),
        ("w2_a2", 2.0, 2.0),
    ]

    t = np.linspace(0.0, 20.0, 5001)

    ORDER = 6
    HORIZON = 25
    SMOOTH_Y = 9
    SMOOTH_Z = 9

    WINDOW = 400
    STRIDE = 400
    MAXITER = 800

    obs = {}
    for name, w, a in conditions:
        sim = simulate_lti_rk4(
            A_true, B_true, t, z0,
            lambda tt, w=w, a=a: sine_force(tt, w, a),
        )
        y = sim.z @ C.T  
        obs[name] = {"t": sim.t, "y": y, "u": sim.u, "w": w, "a": a}

    base_name = "w1_a1"
    proj = SubspaceProjector(order=ORDER, horizon=HORIZON, scale="sqrt", smooth_window=SMOOTH_Y)
    zhat_base = proj.fit_transform(obs[base_name]["y"])

    N0 = zhat_base.shape[0]
    t0 = obs[base_name]["t"][:N0]
    u0 = obs[base_name]["u"][:N0] if obs[base_name]["u"] is not None else np.zeros((N0, 1))

    idres = identify_ab_from_reconstructed_state(
        t=t0,
        z_hat=zhat_base,
        u=u0,
        reg=1e-6,
        smooth_window=SMOOTH_Z,
    )
    A_hat, B_hat = idres.A_hat, idres.B_hat

    H_lyap = lyapunov_baseline_H(A_hat)
    H_cov = covariance_baseline_H(zhat_base)

    flux0 = fit_H_flux_matching(
        t=t0, z=zhat_base, u=u0,
        A=A_hat, B=B_hat,
        window=WINDOW, stride=STRIDE,
        maxiter=MAXITER,
    )
    H0 = flux0.H

    results = {"system": "coupled_oscillator_partial_obs", "base": base_name, "conditions": []}

    for name, w, a in conditions:
        zhat = proj.transform(obs[name]["y"])
        N = zhat.shape[0]
        tt = obs[name]["t"][:N]
        uu = obs[name]["u"][:N] if obs[name]["u"] is not None else np.zeros((N, 1))

        flux = fit_H_flux_matching(
            t=tt, z=zhat, u=uu,
            A=A_hat, B=B_hat,
            window=WINDOW, stride=STRIDE,
            maxiter=MAXITER,
        )
        Hs = flux.H

        alpha, relerr = _best_scale_align(H0, Hs)

        ds = _DS(name=name, t=tt, z=zhat, u=uu)
        ev_flux = evaluate_H_on_dataset(dataset=ds, A=A_hat, B=B_hat, H=Hs, window=WINDOW, stride=STRIDE)

        ev_lyap = evaluate_H_on_dataset(dataset=ds, A=A_hat, B=B_hat, H=H_lyap, window=WINDOW, stride=STRIDE)
        ev_cov = evaluate_H_on_dataset(dataset=ds, A=A_hat, B=B_hat, H=H_cov, window=WINDOW, stride=STRIDE)

        results["conditions"].append({
            "name": name,
            "w": float(w),
            "a": float(a),
            "flux": {
                "T_delta": float(ev_flux["T_delta"]),
                "residual_ratio": float(ev_flux["residual_ratio"]),
                "min_eig_H": float(flux.min_eig_H),
                "objective": float(flux.objective),
                "iters": int(flux.n_iter),
                "success": bool(flux.success),
            },
            "invariance_vs_base": {
                "scale": float(alpha),
                "rel_fro_error": float(relerr),
            },
            "baselines": {
                "lyapunov_T_delta": float(ev_lyap["T_delta"]),
                "lyapunov_residual_ratio": float(ev_lyap["residual_ratio"]),
                "cov_T_delta": float(ev_cov["T_delta"]),
                "cov_residual_ratio": float(ev_cov["residual_ratio"]),
            },
        })

    out_json = outdir / "discovery_test_partial_obs_summary.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    print("=== DISCOVERY TEST (PARTIAL OBSERVATION) ===")
    print(f"[saved] {out_json}")
    for r in results["conditions"]:
        print(
            f"{r['name']}: rel_fro_error={r['invariance_vs_base']['rel_fro_error']:.3e}, "
            f"T_delta={r['flux']['T_delta']:.6g}, res={r['flux']['residual_ratio']:.3e}, success={r['flux']['success']}"
        )


if __name__ == "__main__":
    main()