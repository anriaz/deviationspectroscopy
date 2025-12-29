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
from typing import Callable, Dict, Any, List, Tuple

import numpy as np

from deviation_spectroscopy.sim.forcing import sine_force
from deviation_spectroscopy.sim.integrate import simulate_lti_rk4

from deviation_spectroscopy.id.state_reconstruct import SubspaceProjector
from deviation_spectroscopy.id.identify_from_reconstruction import identify_ab_from_reconstructed_state
from deviation_spectroscopy.flux.flux_match import fit_H_flux_matching
from deviation_spectroscopy.tests.helpers_invariance import evaluate_H_on_dataset
from deviation_spectroscopy.systems.coupled_oscillator import make_coupled_oscillator


Array = np.ndarray


DRIFT_OK = 0.05            
DRIFT_WARN = 0.15          
RESID_OK = 1.5             
#SEEDS = list(range(10))    
SEEDS = [0, 1]

class _DS:
    def __init__(self, name: str, t: Array, z: Array, u: Array):
        self.name = name
        self.t = t
        self.z = z
        self.u = u


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _add_measurement_noise(y: Array, snr_db: float, seed: int) -> Array:
    r = _rng(seed)
    y = np.asarray(y, dtype=float)
    p = float(np.mean(y * y))
    if p <= 0:
        return y.copy()
    snr = 10.0 ** (snr_db / 10.0)
    noise_p = p / snr
    return y + r.standard_normal(y.shape) * np.sqrt(noise_p)


def _rk4_general(
    f: Callable[[float, Array], Array],
    t: Array,
    z0: Array,
) -> Array:
    t = np.asarray(t, dtype=float)
    z0 = np.asarray(z0, dtype=float)
    z = np.zeros((t.shape[0], z0.shape[0]), dtype=float)
    z[0] = z0
    for k in range(t.shape[0] - 1):
        dt = float(t[k + 1] - t[k])
        tk = float(t[k])
        zk = z[k]
        k1 = f(tk, zk)
        k2 = f(tk + 0.5 * dt, zk + 0.5 * dt * k1)
        k3 = f(tk + 0.5 * dt, zk + 0.5 * dt * k2)
        k4 = f(tk + dt, zk + dt * k3)
        z[k + 1] = zk + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return z


def _prep_u(t: Array, w: float, a: float) -> Array:
    u = np.array([sine_force(tt, w, a) for tt in t], dtype=float).reshape(-1, 1)
    return u


def _scale_aligned_fro_error(H0: Array, H1: Array) -> Dict[str, float]:
    H0 = np.asarray(H0, float)
    H1 = np.asarray(H1, float)
    num = float(np.sum(H0 * H1))
    den = float(np.sum(H1 * H1))
    alpha = 0.0 if den <= 0 else num / den
    err = float(np.linalg.norm(H0 - alpha * H1, ord="fro") / max(np.linalg.norm(H0, ord="fro"), 1e-15))
    return {"scale": alpha, "rel_fro_error": err}


def _fit_H_on_dataset(
    *,
    name: str,
    t: Array,
    z: Array,
    u: Array,
    A_hat: Array,
    B_hat: Array,
    window: int,
    stride: int,
    maxiter: int,
) -> Dict[str, Any]:
    flux = fit_H_flux_matching(
        t=t, z=z, u=u, A=A_hat, B=B_hat,
        window=window, stride=stride, maxiter=maxiter
    )
    ev = evaluate_H_on_dataset(dataset=_DS(name, t, z, u), A=A_hat, B=B_hat, H=flux.H, window=window, stride=stride)
    return {
        "H": flux.H,
        "flux": {
            "T_delta": float(ev["T_delta"]),
            "residual_ratio": float(ev["residual_ratio"]),
            "min_eig_H": float(flux.min_eig_H),
            "min_eig_S": float(flux.min_eig_S),
            "objective": float(flux.objective),
            "iters": int(flux.n_iter),
            "success": bool(flux.success),
            "message": str(flux.message),
        }
    }


def _level3_duffing_lifted(
    *,
    seed: int,
    w: float,
    a: float,
    t: Array,
    snr_db: float,
) -> Dict[str, Any]:
    rng = _rng(seed)

    d = 0.25
    k = 1.0
    b = 0.8

    u = _prep_u(t, w, a).reshape(-1)

    def f(tt: float, z: Array) -> Array:
        i = int(round((tt - t[0]) / (t[1] - t[0])))
        i = min(max(i, 0), len(u) - 1)
        x, v = float(z[0]), float(z[1])
        dx = v
        dv = -d * v - k * x - b * (x ** 3) + float(u[i])
        return np.array([dx, dv], dtype=float)

    z0 = np.array([0.0, 0.0], dtype=float)
    z_true = _rk4_general(f, t, z0)

    y = z_true[:, [0]]
    y = _add_measurement_noise(y, snr_db=snr_db, seed=seed + 1000)

    proj = SubspaceProjector(order=6, horizon=25, scale="sqrt", smooth_window=9)
    zhat = proj.fit_transform(y)

    N = zhat.shape[0]
    t_eff = t[:N]
    u_eff = _prep_u(t_eff, w, a)

    idres = identify_ab_from_reconstructed_state(t=t_eff, z_hat=zhat, u=u_eff, reg=1e-6, smooth_window=9)
    A_hat, B_hat = idres.A_hat, idres.B_hat

    out = _fit_H_on_dataset(
        name=f"duffing_lifted_w{w}_a{a}_seed{seed}",
        t=t_eff, z=zhat, u=u_eff, A_hat=A_hat, B_hat=B_hat,
        window=400, stride=400, maxiter=800
    )
    return {
        "seed": seed,
        "w": w,
        "a": a,
        "id_residual_norm": float(idres.residual_norm),
        "flux": out["flux"],
        "H": out["H"].tolist(),  
    }


def _level3_regime_switch(
    *,
    seed: int,
    w: float,
    a: float,
    t: Array,
    switch_frac: float,
    snr_db: float,
) -> Dict[str, Any]:
    rng = _rng(seed)
    dt = float(t[1] - t[0])

    A1 = np.array([[0.0, 1.0],
                   [-2.0, -0.4]])
    A2 = np.array([[0.0, 1.0],
                   [-3.2, -0.6]])  
    B = np.array([[0.0],
                  [1.0]])

    t = np.asarray(t, float)
    N = t.shape[0]
    k_switch = int(np.floor(switch_frac * N))
    k_switch = min(max(k_switch, 2), N - 2)

    u = _prep_u(t, w, a)

    z = np.zeros((N, 2), dtype=float)
    z[0] = 0.0

    def step(A, zk, uk):
        return zk + dt * (A @ zk + (B @ uk).reshape(-1))

    for k in range(N - 1):
        A = A1 if k < k_switch else A2
        z[k + 1] = step(A, z[k], u[k])

    z_noisy = _add_measurement_noise(z, snr_db=snr_db, seed=seed + 2000)

    tA, zA, uA = t[:k_switch], z_noisy[:k_switch], u[:k_switch]
    tB, zB, uB = t[k_switch:], z_noisy[k_switch:], u[k_switch:]

    idA = identify_ab_from_reconstructed_state(t=tA, z_hat=zA, u=uA, reg=1e-6, smooth_window=9)
    idB = identify_ab_from_reconstructed_state(t=tB, z_hat=zB, u=uB, reg=1e-6, smooth_window=9)

    fitA = _fit_H_on_dataset(
        name=f"regimeA_w{w}_a{a}_seed{seed}",
        t=tA, z=zA, u=uA, A_hat=idA.A_hat, B_hat=idA.B_hat,
        window=200, stride=200, maxiter=800
    )
    fitB = _fit_H_on_dataset(
        name=f"regimeB_w{w}_a{a}_seed{seed}",
        t=tB, z=zB, u=uB, A_hat=idB.A_hat, B_hat=idB.B_hat,
        window=200, stride=200, maxiter=800
    )

    drift_AB = _scale_aligned_fro_error(np.array(fitA["H"]), np.array(fitB["H"]))

    return {
        "seed": seed,
        "w": w,
        "a": a,
        "switch_frac": switch_frac,
        "fit_A": {**fitA["flux"], "H": fitA["H"].tolist()},
        "fit_B": {**fitB["flux"], "H": fitB["H"].tolist()},
        "drift_A_vs_B": drift_AB,
    }


def _level3_nonlinear_observation(
    *,
    seed: int,
    w: float,
    a: float,
    t: Array,
    obs: str,
    snr_db: float,
) -> Dict[str, Any]:
    sys = make_coupled_oscillator()
    A, B = sys.A, sys.B
    n = A.shape[0]
    z0 = np.zeros(n)

    C = np.zeros((1, n), dtype=float)
    C[0, 0] = 1.0

    sim = simulate_lti_rk4(A=A, B=B, t=t, z0=z0, u_of_t=lambda tt: sine_force(tt, w, a))
    z = sim.z
    y_lin = z @ C.T

    if obs == "tanh":
        y = np.tanh(y_lin)
    elif obs == "square":
        y = y_lin ** 2
    else:
        raise ValueError("obs must be 'tanh' or 'square'")

    y = _add_measurement_noise(y, snr_db=snr_db, seed=seed + 3000)

    proj = SubspaceProjector(order=6, horizon=25, scale="sqrt", smooth_window=9)
    zhat = proj.fit_transform(y)

    N = zhat.shape[0]
    t_eff = t[:N]
    u_eff = np.array([sine_force(tt, w, a) for tt in t_eff], dtype=float).reshape(-1, 1)

    idres = identify_ab_from_reconstructed_state(t=t_eff, z_hat=zhat, u=u_eff, reg=1e-6, smooth_window=9)

    out = _fit_H_on_dataset(
        name=f"nonlinobs_{obs}_w{w}_a{a}_seed{seed}",
        t=t_eff, z=zhat, u=u_eff, A_hat=idres.A_hat, B_hat=idres.B_hat,
        window=400, stride=400, maxiter=1000
    )

    return {
        "seed": seed,
        "w": w,
        "a": a,
        "obs": obs,
        "id_residual_norm": float(idres.residual_norm),
        "flux": out["flux"],
        "H": out["H"].tolist(),
    }


def main():
    outdir = Path("results/level3_suite")
    outdir.mkdir(parents=True, exist_ok=True)

    conditions = [
        ("w1_a1", 1.0, 1.0),
        ("w2_a1", 2.0, 1.0),
        ("w1_a2", 1.0, 2.0),
        ("w2_a2", 2.0, 2.0),
    ]

    t = np.linspace(0.0, 20.0, 5001)

    snr_db = 20.0

    results: Dict[str, Any] = {
        "meta": {
            "DRIFT_OK": DRIFT_OK,
            "DRIFT_WARN": DRIFT_WARN,
            "RESID_OK": RESID_OK,
            "seeds": SEEDS,
            "snr_db": snr_db,
            "conditions": [{"name": n, "w": w, "a": a} for (n, w, a) in conditions],
        },
        "tests": {}
    }

    duffing_runs: List[Dict[str, Any]] = []
    for seed in SEEDS:
        for name, w, a in conditions:
            duffing_runs.append(_level3_duffing_lifted(seed=seed, w=w, a=a, t=t, snr_db=snr_db))

    duffing_by_seed: Dict[int, Dict[str, Any]] = {}
    for seed in SEEDS:
        items = [r for r in duffing_runs if r["seed"] == seed]
        base = next(r for r in items if abs(r["w"] - 1.0) < 1e-12 and abs(r["a"] - 1.0) < 1e-12)
        H0 = np.array(base["H"], dtype=float)
        for r in items:
            H = np.array(r["H"], dtype=float)
            r["invariance_vs_base"] = _scale_aligned_fro_error(H0, H)
        duffing_by_seed[seed] = {"runs": items}

    results["tests"]["duffing_lifted"] = duffing_by_seed

    switch_runs: List[Dict[str, Any]] = []
    for seed in SEEDS:
        for name, w, a in conditions:
            switch_runs.append(_level3_regime_switch(seed=seed, w=w, a=a, t=t, switch_frac=0.5, snr_db=snr_db))
    results["tests"]["regime_switch"] = {"runs": switch_runs}

    nlobs_runs: List[Dict[str, Any]] = []
    for seed in SEEDS:
        for obs in ["tanh", "square"]:
            for name, w, a in conditions:
                nlobs_runs.append(_level3_nonlinear_observation(seed=seed, w=w, a=a, t=t, obs=obs, snr_db=snr_db))

    nlobs_group: Dict[str, Dict[int, Any]] = {"tanh": {}, "square": {}}
    for obs in ["tanh", "square"]:
        for seed in SEEDS:
            items = [r for r in nlobs_runs if r["seed"] == seed and r["obs"] == obs]
            base = next(r for r in items if abs(r["w"] - 1.0) < 1e-12 and abs(r["a"] - 1.0) < 1e-12)
            H0 = np.array(base["H"], dtype=float)
            for r in items:
                H = np.array(r["H"], dtype=float)
                r["invariance_vs_base"] = _scale_aligned_fro_error(H0, H)
            nlobs_group[obs][seed] = {"runs": items}

    results["tests"]["nonlinear_observation"] = nlobs_group


    def pass_rate_from_runs(runs: List[Dict[str, Any]]) -> Dict[str, float]:
        drifts = []
        for r in runs:
            inv = r.get("invariance_vs_base", None)
            if inv is not None:
                drifts.append(float(inv["rel_fro_error"]))
        drifts = np.array(drifts, dtype=float) if drifts else np.array([], dtype=float)
        if drifts.size == 0:
            return {"n": 0, "pass_5pct": 0.0, "pass_15pct": 0.0, "mean": np.nan, "std": np.nan}
        return {
            "n": int(drifts.size),
            "pass_5pct": float(np.mean(drifts <= DRIFT_OK)),
            "pass_15pct": float(np.mean(drifts <= DRIFT_WARN)),
            "mean": float(np.mean(drifts)),
            "std": float(np.std(drifts)),
        }

    all_duffing = []
    for seed in SEEDS:
        all_duffing.extend(results["tests"]["duffing_lifted"][seed]["runs"])
    results["tests"]["duffing_lifted_summary"] = pass_rate_from_runs(all_duffing)

    for obs in ["tanh", "square"]:
        all_obs = []
        for seed in SEEDS:
            all_obs.extend(results["tests"]["nonlinear_observation"][obs][seed]["runs"])
        results["tests"][f"nonlinear_observation_{obs}_summary"] = pass_rate_from_runs(all_obs)

    drift_switch = np.array([float(r["drift_A_vs_B"]["rel_fro_error"]) for r in switch_runs], dtype=float)
    results["tests"]["regime_switch_summary"] = {
        "n": int(drift_switch.size),
        "mean_drift_A_vs_B": float(np.mean(drift_switch)),
        "std_drift_A_vs_B": float(np.std(drift_switch)),
        "min_drift_A_vs_B": float(np.min(drift_switch)),
        "max_drift_A_vs_B": float(np.max(drift_switch)),
    }

    out_json = outdir / "level3_suite_summary.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    print("=== LEVEL 3 SUITE COMPLETE ===")
    print(f"[saved] {out_json}")


if __name__ == "__main__":
    main()