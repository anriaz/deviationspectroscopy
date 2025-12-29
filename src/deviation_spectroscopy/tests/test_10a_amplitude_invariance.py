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
import numpy as np

from deviation_spectroscopy.sim.forcing import sine_force
from deviation_spectroscopy.experiments.datasets import build_experiment_datasets
from deviation_spectroscopy.flux.flux_match import fit_H_flux_matching
from deviation_spectroscopy.core.baselines import (
    lyapunov_baseline_H,
    covariance_baseline_H,
)
from deviation_spectroscopy.tests.helpers_invariance import evaluate_H_on_dataset
from deviation_spectroscopy.core.metrics import compute_S_H_drift


def test_amplitude_invariance_flux_vs_baselines():
    A = np.array([[0.0, 1.0],
                  [-2.0, -0.5]])
    B = np.array([[0.0],
                  [1.0]])
    z0 = np.array([0.0, 0.0])

    t_id = np.linspace(0.0, 5.0, 2001)
    t_steady = np.linspace(0.0, 20.0, 4001)

    def forcing_id(t):
        return sine_force(t, amplitude=1.0, frequency=0.7)

    def forcing_A(t):
        return sine_force(t, amplitude=1.0, frequency=1.0)

    def forcing_B(t):
        return sine_force(t, amplitude=2.5, frequency=1.0) 

    data = build_experiment_datasets(
        A=A, B=B, z0=z0,
        t_id=t_id, t_steady=t_steady,
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
        window=300,
        stride=300,
        maxiter=250,
    ).H

    H_lyap = lyapunov_baseline_H(A)
    H_cov = covariance_baseline_H(data["A"].z)

    results = {}
    for name, H in {
        "flux": H_star,
        "lyap": H_lyap,
        "cov": H_cov,
    }.items():
        eval_A = evaluate_H_on_dataset(
            dataset=data["A"], A=A, B=B, H=H, window=300, stride=300
        )
        eval_B = evaluate_H_on_dataset(
            dataset=data["B"], A=A, B=B, H=H, window=300, stride=300
        )
        results[name] = (eval_A, eval_B)

    flux_A, flux_B = results["flux"]
    T_rel = abs(flux_A["T_delta"] - flux_B["T_delta"]) / flux_A["T_delta"]
    S_rel = compute_S_H_drift(flux_A["S_H"], flux_B["S_H"])

    assert T_rel < 0.1
    assert S_rel < 0.1

    for name in ["lyap", "cov"]:
        A0, B0 = results[name]

        T_rel0 = abs(A0["T_delta"] - B0["T_delta"]) / A0["T_delta"]
        assert T_rel0 > T_rel

        assert flux_A["residual_ratio"] < 0.3
        assert flux_B["residual_ratio"] < 0.3