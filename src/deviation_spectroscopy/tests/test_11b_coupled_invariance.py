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
from deviation_spectroscopy.tests.helpers_invariance import evaluate_H_on_dataset


def test_coupled_oscillator_invariance():
    k1, k2 = 2.0, 1.5
    c1, c2 = 0.4, 0.3
    k12 = 0.8

    A = np.array([
        [0.0,  1.0,  0.0,  0.0],
        [-k1, -c1,  k12,  0.0],
        [0.0,  0.0,  0.0,  1.0],
        [k12,  0.0, -k2, -c2],
    ])

    B = np.array([[0.0], [1.0], [0.0], [0.0]])
    z0 = np.zeros(4)

    t_id = np.linspace(0.0, 5.0, 2501)
    t_steady = np.linspace(0.0, 25.0, 6001)

    def forcing_id(t):
        return sine_force(t, amplitude=1.0, frequency=0.6)

    def forcing_A(t):
        return sine_force(t, amplitude=1.0, frequency=1.0)

    def forcing_B(t):
        return sine_force(t, amplitude=1.0, frequency=2.0)

    data = build_experiment_datasets(
        A=A, B=B, z0=z0,
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
        window=400,
        stride=400,
        maxiter=300,
    ).H

    eval_A = evaluate_H_on_dataset(
        dataset=data["A"], A=A, B=B, H=H_star, window=400, stride=400
    )
    eval_B = evaluate_H_on_dataset(
        dataset=data["B"], A=A, B=B, H=H_star, window=400, stride=400
    )

    T_rel = abs(eval_A["T_delta"] - eval_B["T_delta"]) / eval_A["T_delta"]

    assert T_rel < 0.15
    assert eval_A["residual_ratio"] < 0.4
    assert eval_B["residual_ratio"] < 0.4