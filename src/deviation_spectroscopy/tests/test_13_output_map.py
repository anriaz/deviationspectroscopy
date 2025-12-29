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
from deviation_spectroscopy.sim.forcing import sine_force

def test_dataset_includes_output_y():
    A = np.array([[0.0, 1.0], [-2.0, -0.5]])
    B = np.array([[0.0], [1.0]])
    C = np.array([[1.0, 0.0]])  
    z0 = np.zeros(2)

    t_id = np.linspace(0.0, 2.0, 401)
    t_steady = np.linspace(0.0, 4.0, 801)

    data = build_experiment_datasets(
        A=A, B=B, z0=z0,
        C=C,
        t_id=t_id,
        t_steady=t_steady,
        forcing_id=lambda t: sine_force(t, 1.0, 0.7),
        forcing_A=lambda t: sine_force(t, 1.0, 1.0),
        forcing_B=lambda t: sine_force(t, 2.0, 1.3),
    )

    for key in ["id", "A", "B"]:
        d = data[key]
        assert hasattr(d, "y")
        assert d.y.shape[0] == d.z.shape[0]
        assert d.y.shape[1] == 1
        assert np.allclose(d.y[:, 0], d.z[:, 0])