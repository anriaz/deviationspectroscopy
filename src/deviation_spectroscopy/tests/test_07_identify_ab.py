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
from deviation_spectroscopy.id.identify_ab import estimate_ab_ridge
from deviation_spectroscopy.sim.integrate import simulate_lti_rk4
from deviation_spectroscopy.sim.forcing import band_limited_noise


def test_identify_ab_accuracy():
    A_true = np.array([[0.0, 1.0], [-2.0, -0.4]])
    B_true = np.array([[0.0], [1.0]])

    t = np.linspace(0.0, 10.0, 5001)
    dt = t[1] - t[0]
    z0 = np.array([0.0, 0.0])

    def forcing(t):
        return band_limited_noise(t, rms=1.0, seed=0)

    sim = simulate_lti_rk4(A_true, B_true, t, z0, forcing)

    est = estimate_ab_ridge(sim.z, sim.u, dt, ridge=1e-6)

    assert np.linalg.norm(est.A - A_true) < 0.1
    assert np.linalg.norm(est.B - B_true) < 0.1
    assert est.residual_norm < 0.5