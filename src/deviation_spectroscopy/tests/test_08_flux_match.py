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

from deviation_spectroscopy.sim.integrate import simulate_lti_rk4
from deviation_spectroscopy.sim.forcing import band_limited_noise
from deviation_spectroscopy.flux.flux_match import fit_H_flux_matching
from deviation_spectroscopy.flux.residuals import windowed_residual, residual_ratio


def test_flux_match_recovers_psd_S_and_low_residual():
    A = np.array([[0.0, 1.0],
                  [-2.0, -0.6]])
    B = np.array([[0.0],
                  [1.0]])

    t = np.linspace(0.0, 20.0, 4001)
    z0 = np.array([0.0, 0.0])

    def forcing(tt):
        return band_limited_noise(tt, rms=1.0, seed=4)

    sim = simulate_lti_rk4(A, B, t, z0, forcing)

    out = fit_H_flux_matching(
        t=sim.t,
        z=sim.z,
        u=sim.u,
        A=A,
        B=B,
        window=200,
        stride=200,
        penalty_mu=1e6,
        x0_seed=0,
        maxiter=250,
    )

    assert out.success
    assert np.isclose(np.trace(out.H), 1.0, atol=1e-10)
    assert out.min_eig_H > 0.0
    assert out.min_eig_S >= -1e-8

    dE, r = windowed_residual(sim.t, sim.z, sim.u, A, B, out.H, window=200, stride=200)
    rr = residual_ratio(dE, r)
    assert rr < 0.25