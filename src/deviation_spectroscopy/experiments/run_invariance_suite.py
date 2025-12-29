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
from pathlib import Path
import numpy as np

from deviation_spectroscopy.experiments.datasets import build_experiment_datasets
from deviation_spectroscopy.flux.flux_match import fit_H_flux_matching
from deviation_spectroscopy.core.baselines import (
    lyapunov_baseline_H,
    covariance_baseline_H,
)
from deviation_spectroscopy.tests.helpers_invariance import evaluate_H_on_dataset
from deviation_spectroscopy.sim.forcing import sine_force


def run_system(
    *,
    system_name: str,
    A,
    B,
    z0,
    t_id,
    t_steady,
    forcing_id,
    forcing_A,
    forcing_B,
    window,
    stride,
):
    data = build_experiment_datasets(
        A=A,
        B=B,
        z0=z0,
        t_id=t_id,
        t_steady=t_steady,
        forcing_id=forcing_id,
        forcing_A=forcing_A,
        forcing_B=forcing_B,
    )

    H_star = fit_H_flux_matching(
        t=data["A"].t,
        z=data["A"].z,
        u=data["A"].u,
        A=A,
        B=B,
        window=window,
        stride=stride,
    ).H

    eval_A = evaluate_H_on_dataset(
        dataset=data["A"], A=A, B=B, H=H_star, window=window, stride=stride
    )
    eval_B = evaluate_H_on_dataset(
        dataset=data["B"], A=A, B=B, H=H_star, window=window, stride=stride
    )

    T_rel = abs(eval_A["T_delta"] - eval_B["T_delta"]) / eval_A["T_delta"]

    result = {
        "system": system_name,
        "flux": {
            "T_delta_A": eval_A["T_delta"],
            "T_delta_B": eval_B["T_delta"],
            "T_rel": T_rel,
            "residual_A": eval_A["residual_ratio"],
            "residual_B": eval_B["residual_ratio"],
        },
    }

    H_lyap = lyapunov_baseline_H(A)
    H_cov = covariance_baseline_H(data["A"].z)

    for name, H in {
        "lyapunov": H_lyap,
        "covariance": H_cov,
    }.items():
        eA = evaluate_H_on_dataset(
            dataset=data["A"], A=A, B=B, H=H, window=window, stride=stride
        )
        eB = evaluate_H_on_dataset(
            dataset=data["B"], A=A, B=B, H=H, window=window, stride=stride
        )
        T_rel0 = abs(eA["T_delta"] - eB["T_delta"]) / eA["T_delta"]
        result[name] = {"T_rel": T_rel0}

    return result


def save_results(result: dict, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{result['system']}.json"
    with open(path, "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    outdir = Path("results/invariance")

    A = np.array([[0.0, 1.0],
                  [-2.0, -0.5]])
    B = np.array([[0.0],
                  [1.0]])
    z0 = np.zeros(2)

    res = run_system(
        system_name="two_mass",
        A=A,
        B=B,
        z0=z0,
        t_id=np.linspace(0, 5, 2001),
        t_steady=np.linspace(0, 20, 4001),
        forcing_id=lambda t: sine_force(t, 1.0, 0.7),
        forcing_A=lambda t: sine_force(t, 1.0, 1.0),
        forcing_B=lambda t: sine_force(t, 2.5, 1.0),
        window=300,
        stride=300,
    )
    save_results(res, outdir)

    gamma = 1.3
    A = np.array([[-gamma]])
    B = np.array([[1.0]])
    z0 = np.zeros(1)

    res = run_system(
        system_name="ou_1d",
        A=A,
        B=B,
        z0=z0,
        t_id=np.linspace(0, 4, 2001),
        t_steady=np.linspace(0, 15, 4001),
        forcing_id=lambda t: sine_force(t, 1.0, 0.4),
        forcing_A=lambda t: sine_force(t, 1.0, 1.0),
        forcing_B=lambda t: sine_force(t, 2.0, 2.3),
        window=300,
        stride=300,
    )
    save_results(res, outdir)

    from deviation_spectroscopy.systems.coupled_oscillator import make_coupled_oscillator

    sys = make_coupled_oscillator()
    A = sys.A
    B = sys.B
    z0 = np.zeros(A.shape[0])

    res = run_system(
        system_name="coupled_oscillator",
        A=A,
        B=B,
        z0=z0,
        t_id=np.linspace(0, 6, 3001),
        t_steady=np.linspace(0, 20, 5001),
        forcing_id=lambda t: sine_force(t, 1.0, 0.6),
        forcing_A=lambda t: sine_force(t, 1.0, 1.1),
        forcing_B=lambda t: sine_force(t, 1.7, 2.0),
        window=400,
        stride=400,
    )
    save_results(res, outdir)