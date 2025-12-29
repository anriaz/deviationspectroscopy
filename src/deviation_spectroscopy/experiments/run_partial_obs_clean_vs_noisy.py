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


def _make_obs(A_true, B_true, t, z0, C, conditions, *, noise_sigma: float, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    obs = {}
    for name, w, a in conditions:
        sim = simulate_lti_rk4(
            A_true, B_true, t, z0,
            lambda tt, w=w, a=a: sine_force(tt, w, a),
        )
        y = sim.z @ C.T  # (N, m)
        if noise_sigma > 0.0:
            y = y + noise_sigma * rng.standard_normal(size=y.shape)
        obs[name] = {"t": sim.t, "y": y, "u": sim.u, "w": float(w), "a": float(a)}
    return obs


def _run_pipeline(obs: dict, *, base_name: str, hyper: dict) -> dict:
    ORDER = int(hyper["ORDER"])
    HORIZON = int(hyper["HORIZON"])
    SMOOTH_Y = int(hyper["SMOOTH_Y"])
    SMOOTH_Z = int(hyper["SMOOTH_Z"])
    WINDOW = int(hyper["WINDOW"])
    STRIDE = int(hyper["STRIDE"])
    MAXITER = int(hyper["MAXITER"])

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

    out_conditions = []
    for name, blob in obs.items():
        zhat = proj.transform(blob["y"])
        N = zhat.shape[0]
        tt = blob["t"][:N]
        uu = blob["u"][:N] if blob["u"] is not None else np.zeros((N, 1))

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

        out_conditions.append({
            "name": name,
            "w": float(blob["w"]),
            "a": float(blob["a"]),
            "flux": {
                "T_delta": float(ev_flux["T_delta"]),
                "residual_ratio": float(ev_flux["residual_ratio"]),
                "min_eig_H": float(flux.min_eig_H),
                "objective": float(flux.objective),
                "iters": int(flux.n_iter),
                "success": bool(flux.success),
            },
            "invariance_vs_base": {"scale": float(alpha), "rel_fro_error": float(relerr)},
            "baselines": {
                "lyapunov_T_delta": float(ev_lyap["T_delta"]),
                "lyapunov_residual_ratio": float(ev_lyap["residual_ratio"]),
                "cov_T_delta": float(ev_cov["T_delta"]),
                "cov_residual_ratio": float(ev_cov["residual_ratio"]),
            },
        })

    drifts = np.array([c["invariance_vs_base"]["rel_fro_error"] for c in out_conditions], dtype=float)
    resid = np.array([c["flux"]["residual_ratio"] for c in out_conditions], dtype=float)

    return {
        "base": base_name,
        "hyperparams": hyper,
        "A_hat_shape": [int(A_hat.shape[0]), int(A_hat.shape[1])],
        "conditions": out_conditions,
        "summary": {
            "n_conditions": int(len(out_conditions)),
            "drift": {
                "mean": float(np.mean(drifts)),
                "median": float(np.median(drifts)),
                "max": float(np.max(drifts)),
            },
            "residual_ratio": {
                "mean": float(np.mean(resid)),
                "median": float(np.median(resid)),
                "max": float(np.max(resid)),
            },
        },
    }


def main():
    outdir = Path("results/partial_obs_clean_vs_noisy")
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
    base_name = "w1_a1"

    t = np.linspace(0.0, 20.0, 5001)

    hyper = dict(
        ORDER=6, HORIZON=25,
        SMOOTH_Y=9, SMOOTH_Z=9,
        WINDOW=400, STRIDE=400, MAXITER=800,
    )

    NOISE_SIGMA = 0.05
    SEED = 0

    obs_clean = _make_obs(A_true, B_true, t, z0, C, conditions, noise_sigma=0.0, seed=SEED)
    obs_noisy = _make_obs(A_true, B_true, t, z0, C, conditions, noise_sigma=NOISE_SIGMA, seed=SEED)

    clean = _run_pipeline(obs_clean, base_name=base_name, hyper=hyper)
    noisy = _run_pipeline(obs_noisy, base_name=base_name, hyper=hyper)

    out = {
        "system": "coupled_oscillator_partial_obs",
        "comparison": "clean_vs_noisy_observation",
        "noise": {"sigma": float(NOISE_SIGMA), "seed": int(SEED), "channel": "x1_only"},
        "clean": clean,
        "noisy": noisy,
    }

    out_json = outdir / "partial_obs_clean_vs_noisy_summary.json"
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)

    print("=== PARTIAL OBSERVATION: CLEAN vs NOISY ===")
    print(f"[saved] {out_json}")
    print(f"Clean median drift: {clean['summary']['drift']['median']:.6g}")
    print(f"Noisy median drift: {noisy['summary']['drift']['median']:.6g}")
    print(f"Noise sigma: {NOISE_SIGMA:.6g}")


if __name__ == "__main__":
    main()