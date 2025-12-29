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
from deviation_spectroscopy.sim.ou_noise import ou_process
from deviation_spectroscopy.flux.flux_match import fit_H_flux_matching
from deviation_spectroscopy.tests.helpers_invariance import evaluate_H_on_dataset


Array = np.ndarray

@dataclass
class _DS:
    name: str
    t: Array
    z: Array
    u: Array


def _scale_align(A: Array, B: Array) -> tuple[float, float]:
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    num = float(np.sum(A * B))
    den = float(np.sum(B * B)) + 1e-18
    alpha = num / den
    rel = float(np.linalg.norm(A - alpha * B, ord="fro") / (np.linalg.norm(A, ord="fro") + 1e-18))
    return alpha, rel


def _run_condition(A, B, z0, t, w, a, *, tau, sigma, seed_noise: int):
    ou = ou_process(t, tau=tau, sigma=sigma, seed=seed_noise)
    u = sine_force(t, w, a).reshape(-1) + ou

    sim = simulate_lti_rk4(A, B, t, z0, lambda tt: np.interp(tt, t, u))
    ds = _DS(name=f"w{w}_a{a}", t=sim.t, z=sim.z, u=sim.u)

    flux = fit_H_flux_matching(
        t=ds.t,
        z=ds.z,
        u=ds.u,
        A=A,
        B=B,
        window=400,
        stride=400,
        maxiter=1000,
    )
    H = flux.H

    ev = evaluate_H_on_dataset(dataset=ds, A=A, B=B, H=H, window=400, stride=400)

    return {
        "w": float(w),
        "a": float(a),
        "flux": {
            "T_delta": float(ev["T_delta"]),
            "residual_ratio": float(ev["residual_ratio"]),
            "min_eig_H": float(flux.min_eig_H),
            "objective": float(flux.objective),
            "iters": int(flux.n_iter),
            "success": bool(flux.success),
            "message": str(flux.message),
        },
        "H": H.tolist(),  
    }


def main():
    outdir = Path("results/discovery_test_ou")
    outdir.mkdir(parents=True, exist_ok=True)

    sys = make_coupled_oscillator()
    A_true, B_true = sys.A, sys.B
    n = A_true.shape[0]
    z0 = np.zeros(n)

    CONDITIONS = [
        ("w1_a1", 1.0, 1.0),
        ("w2_a1", 2.0, 1.0),
        ("w1_a2", 1.0, 2.0),
        ("w2_a2", 2.0, 2.0),
    ]

    TAU = 0.8      
    SIGMA = 0.35   

    SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  

    t = np.linspace(0.0, 20.0, 5001)

    rows = []
    for seed in SEEDS:
        cond_results = []
        for j, (name, w, a) in enumerate(CONDITIONS):
            r = _run_condition(
                A_true, B_true, z0, t, w, a,
                tau=TAU, sigma=SIGMA,
                seed_noise=seed * 10 + j,
            )
            r["name"] = name
            r["seed"] = int(seed)
            cond_results.append(r)

        H0 = np.array(cond_results[0]["H"], dtype=float)
        for r in cond_results:
            H = np.array(r["H"], dtype=float)
            alpha, rel = _scale_align(H0, H)
            r["invariance_vs_base"] = {"scale": float(alpha), "rel_fro_error": float(rel)}

        rows.extend(cond_results)

    by_name = {}
    for r in rows:
        by_name.setdefault(r["name"], []).append(r["invariance_vs_base"]["rel_fro_error"])

    summary = {
        "system": "coupled_oscillator",
        "ou": {"tau": float(TAU), "sigma": float(SIGMA)},
        "seeds": SEEDS,
        "conditions": CONDITIONS,
        "drift_stats": {
            k: {
                "mean": float(np.mean(v)),
                "median": float(np.median(v)),
                "p90": float(np.quantile(v, 0.90)),
                "p95": float(np.quantile(v, 0.95)),
                "max": float(np.max(v)),
            }
            for k, v in by_name.items()
        },
        "rows": rows,
    }

    out = outdir / "discovery_test_ou_summary.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()