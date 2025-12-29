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
from deviation_spectroscopy.sim.forcing import sine_force
from deviation_spectroscopy.id.state_reconstruct import reconstruct_state_subspace
from deviation_spectroscopy.id.identify_from_reconstruction import (
    identify_ab_from_reconstructed_state,
)


def test_identify_from_reconstructed_dimensions():
    A = np.array([[0.0, 1.0],
                  [-2.0, -0.5]])
    B = np.array([[0.0],
                  [1.0]])
    C = np.array([[1.0, 0.0]])  

    z0 = np.zeros(2)
    t = np.linspace(0.0, 10.0, 4001)

    sim = simulate_lti_rk4(
        A=A,
        B=B,
        t=t,
        z0=z0,
        u_of_t=lambda t: sine_force(t, 1.0, 1.0),
    )

    y = sim.z @ C.T

    rec = reconstruct_state_subspace(
        y,
        order=4,
        horizon=20,
    )

    idres = identify_ab_from_reconstructed_state(
        t=t[: rec.z_hat.shape[0]],
        z_hat=rec.z_hat,
        u=sim.u[: rec.z_hat.shape[0]],
    )

    assert idres.A_hat.shape == (4, 4)
    assert idres.B_hat.shape == (4, 1)
    assert idres.residual_norm > 0.0


def test_identification_improves_with_more_data():
    A = np.array([[-1.2]])
    B = np.array([[1.0]])
    z0 = np.zeros(1)

    t_short = np.linspace(0.0, 5.0, 1001)
    t_long = np.linspace(0.0, 20.0, 4001)

    def run(t):
        sim = simulate_lti_rk4(
            A=A,
            B=B,
            t=t,
            z0=z0,
            u_of_t=lambda t: sine_force(t, 1.0, 0.8),
        )
        y = sim.z[:, [0]]  

        rec = reconstruct_state_subspace(y, order=2, horizon=15)

        return identify_ab_from_reconstructed_state(
            t=t[: rec.z_hat.shape[0]],
            z_hat=rec.z_hat,
            u=sim.u[: rec.z_hat.shape[0]],
        )

    short = run(t_short)
    long = run(t_long)

    assert long.residual_norm < short.residual_norm