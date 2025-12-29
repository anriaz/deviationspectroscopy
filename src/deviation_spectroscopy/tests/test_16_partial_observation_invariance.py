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

from deviation_spectroscopy.systems.coupled_oscillator import make_coupled_oscillator

from deviation_spectroscopy.id.state_reconstruct import SubspaceProjector
from deviation_spectroscopy.id.identify_from_reconstruction import (
    identify_ab_from_reconstructed_state,
)

from deviation_spectroscopy.flux.flux_match import fit_H_flux_matching
from deviation_spectroscopy.core.metrics import compute_T_delta


def test_partial_observation_invariance_coupled_oscillator():
    sys = make_coupled_oscillator()
    A_true = sys.A
    B_true = sys.B

    C = np.zeros((1, A_true.shape[0]))
    C[0, 0] = 1.0

    z0 = np.zeros(A_true.shape[0])

    t_A = np.linspace(0.0, 30.0, 6001)
    u_A = lambda t: sine_force(t, amplitude=1.0, frequency=1.0)

    sim_A = simulate_lti_rk4(A_true, B_true, t_A, z0, u_A)
    y_A = sim_A.z @ C.T

    t_B = np.linspace(0.0, 30.0, 6001)
    u_B = lambda t: sine_force(t, amplitude=1.0, frequency=2.2)

    sim_B = simulate_lti_rk4(A_true, B_true, t_B, z0, u_B)
    y_B = sim_B.z @ C.T

    projector = SubspaceProjector(order=4, horizon=25, scale="sqrt")

    z_hat_A = projector.fit_transform(y_A)
    z_hat_B = projector.transform(y_B)

    N_eff = z_hat_A.shape[0]
    t_eff = t_A[:N_eff]
    u_A_eff = sim_A.u[:N_eff]
    u_B_eff = sim_B.u[:N_eff]

    id_res = identify_ab_from_reconstructed_state(
        t=t_eff,
        z_hat=z_hat_A,
        u=u_A_eff,
        reg=1e-6,
    )

    A_hat, B_hat = id_res.A_hat, id_res.B_hat

    flux_res = fit_H_flux_matching(
        t=t_eff,
        z=z_hat_A,
        u=u_A_eff,
        A=A_hat,
        B=B_hat,
        window=300,
        stride=300,
        maxiter=400,
    )

    assert flux_res.success
    H_star = flux_res.H
    S_H = flux_res.S_H

    T_A = compute_T_delta(z_hat_A, H_star, S_H)
    T_B = compute_T_delta(z_hat_B, H_star, S_H)

    rel_error = abs(T_A - T_B) / T_A
    print(
        f"\nPartial Obs Invariance:"
        f" T_A={T_A:.6f}, T_B={T_B:.6f}, Drift={rel_error:.2%}"
    )

    assert T_A > 0.01
    assert rel_error < 0.20