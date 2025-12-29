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
import json
import numpy as np

from deviation_spectroscopy.sim.integrate import simulate_lti_rk4
from deviation_spectroscopy.sim.forcing import sine_force
from deviation_spectroscopy.flux.flux_match import fit_H_flux_matching
from deviation_spectroscopy.tests.helpers_invariance import evaluate_H_on_dataset
from deviation_spectroscopy.core.baselines import lyapunov_baseline_H, covariance_baseline_H

Array = np.ndarray


@dataclass
class DS:
    name: str
    t: Array
    z: Array
    u: Array


def _align_scale(Href: Array, H: Array) -> tuple[Array, float]:
    num = float(np.sum(Href * H))
    den = float(np.sum(H * H)) + 1e-18
    c = num / den
    return c * H, c


def _rel_fro_err(Href: Array, H: Array) -> float:
    return float(np.linalg.norm(Href - H, "fro") / (np.linalg.norm(Href, "fro") + 1e-18))


def main():
    outdir = Path("results/discovery_test")
    outdir.mkdir(parents=True, exist_ok=True)

    from deviation_spectroscopy.systems.coupled_oscillator import make_coupled_oscillator
    sys = make_coupled_oscillator()
    A_true, B_true = sys.A, sys.B
    n = A_true.shape[0]
    z0 = np.zeros(n)

    H_true = getattr(sys, "H_true", None)
    if H_true is None:
        raise RuntimeError(
            "coupled_oscillator must expose sys.H_true (the physical quadratic energy matrix)."
        )

    t_id = np.linspace(0.0, 6.0, 3001)
    forcing_id = lambda tt: sine_force(tt, 1.0, 0.6)
    sim0 = simulate_lti_rk4(A_true, B_true, t_id, z0, forcing_id)

    A_hat, B_hat = A_true.copy(), B_true.copy()

    t = np.linspace(0.0, 20.0, 5001)
    conditions = [
        {"name": "w1_a1", "w": 1.0, "a": 1.0},
        {"name": "w2_a1", "w": 2.0, "a": 1.0},   
        {"name": "w1_a2", "w": 1.0, "a": 2.0},   
        {"name": "w2_a2", "w": 2.0, "a": 2.0},   
    ]

    results = {"system": "coupled_oscillator", "conditions": []}

    H_refs = []
    cond_objs = {}

    for c in conditions:
        forcing = lambda tt, w=c["w"], a=c["a"]: sine_force(tt, w, a)
        sim = simulate_lti_rk4(A_true, B_true, t, z0, forcing)

        ds = DS(c["name"], sim.t, sim.z, sim.u)

        flux = fit_H_flux_matching(
            t=ds.t,
            z=ds.z,
            u=ds.u,
            A=A_hat,
            B=B_hat,
            window=400,
            stride=400,
            maxiter=2000,
        )
        H_star = flux.H

        ev = evaluate_H_on_dataset(
            dataset=ds,
            A=A_hat,
            B=B_hat,
            H=H_star,
            window=400,
            stride=400,
        )

        cond_objs[c["name"]] = {
            "H_star": H_star,
            "eval": ev,
            "min_eig_H": float(flux.min_eig_H),
            "min_eig_S": float(flux.min_eig_S),
            "objective": float(flux.objective),
            "iters": int(flux.n_iter),
            "success": bool(flux.success),
            "message": str(flux.message),
        }
        H_refs.append(H_star)

    base = conditions[0]["name"]
    H_base = cond_objs[base]["H_star"]

    for c in conditions:
        name = c["name"]
        H = cond_objs[name]["H_star"]

        H_scaled, s = _align_scale(H_base, H)
        inv_err = _rel_fro_err(H_base, H_scaled)

        Ht_scaled, st = _align_scale(H_true, H)
        rec_err = _rel_fro_err(H_true, Ht_scaled)

        forcing = lambda tt, w=c["w"], a=c["a"]: sine_force(tt, w, a)
        sim = simulate_lti_rk4(A_true, B_true, t, z0, forcing)
        ds = DS(name, sim.t, sim.z, sim.u)

        H_lyap = lyapunov_baseline_H(A_hat)
        H_cov = covariance_baseline_H(ds.z)

        e_flux = cond_objs[name]["eval"]
        e_lyap = evaluate_H_on_dataset(
            dataset=ds,
            A=A_hat,
            B=B_hat,
            H=H_lyap,
            window=400,
            stride=400,
        )
        e_cov = evaluate_H_on_dataset(
            dataset=ds,
            A=A_hat,
            B=B_hat,
            H=H_cov,
            window=400,
            stride=400,
        )

        results["conditions"].append({
            "name": name,
            "w": c["w"],
            "a": c["a"],
            "flux": {
                "T_delta": float(e_flux["T_delta"]),
                "residual_ratio": float(e_flux["residual_ratio"]),
                "min_eig_H": cond_objs[name]["min_eig_H"],
                "objective": cond_objs[name]["objective"],
                "iters": cond_objs[name]["iters"],
                "success": cond_objs[name]["success"],
            },
            "invariance_vs_base": {
                "scale": float(s),
                "rel_fro_error": float(inv_err),
            },
            "recovery_vs_truth": {
                "scale": float(st),
                "rel_fro_error": float(rec_err),
            },
            "baselines": {
                "lyapunov_T_delta": float(e_lyap["T_delta"]),
                "lyapunov_residual_ratio": float(e_lyap["residual_ratio"]),
                "cov_T_delta": float(e_cov["T_delta"]),
                "cov_residual_ratio": float(e_cov["residual_ratio"]),
            }
        })

    out_json = outdir / "discovery_test_summary.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[saved] {out_json}")


if __name__ == "__main__":
    main()