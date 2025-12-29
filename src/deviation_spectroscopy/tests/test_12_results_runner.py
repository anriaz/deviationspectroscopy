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
from pathlib import Path
import json
import numpy as np

from deviation_spectroscopy.experiments.run_invariance_suite import run_system
from deviation_spectroscopy.sim.forcing import sine_force


def test_results_runner_two_mass(tmp_path):
    A = np.array([[0.0, 1.0],
                  [-2.0, -0.5]])
    B = np.array([[0.0],
                  [1.0]])
    z0 = np.zeros(2)

    result = run_system(
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

    assert "flux" in result
    assert "T_rel" in result["flux"]
    assert result["flux"]["T_rel"] < 0.15


def test_results_runner_ou():
    A = np.array([[-1.3]])
    B = np.array([[1.0]])
    z0 = np.zeros(1)

    result = run_system(
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

    assert result["flux"]["T_rel"] < 0.15


def test_results_runner_coupled():
    from deviation_spectroscopy.systems.coupled_oscillator import make_coupled_oscillator

    sys = make_coupled_oscillator()
    A = sys.A
    B = sys.B
    z0 = np.zeros(A.shape[0])

    result = run_system(
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

    assert result["flux"]["T_rel"] < 0.20