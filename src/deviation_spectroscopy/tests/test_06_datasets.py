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
from deviation_spectroscopy.experiments.datasets import build_experiment_datasets
from deviation_spectroscopy.sim.forcing import sine_force, band_limited_noise


def test_dataset_roles_and_shapes():
    A = np.array([[0.0, 1.0], [-1.0, -0.3]])
    B = np.array([[0.0], [1.0]])
    z0 = np.array([0.0, 0.0])

    t_id = np.linspace(0.0, 5.0, 2001)
    t_steady = np.linspace(0.0, 10.0, 4001)

    def forcing_id(t):
        return band_limited_noise(t, rms=1.0, seed=1)

    def forcing_A(t):
        return sine_force(t, amplitude=1.0, frequency=1.0)

    def forcing_B(t):
        return sine_force(t, amplitude=2.0, frequency=1.0)  

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

    assert set(data.keys()) == {"id", "A", "B"}

    for key in ["id", "A", "B"]:
        ds = data[key]
        assert ds.z.shape[0] == ds.t.shape[0]
        assert ds.u.shape[0] == ds.t.shape[0]