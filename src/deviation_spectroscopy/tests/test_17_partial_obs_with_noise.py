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
from deviation_spectroscopy.id.state_reconstruct import SubspaceProjector
from deviation_spectroscopy.id.identify_from_reconstruction import identify_ab_from_reconstructed_state
from deviation_spectroscopy.flux.flux_match import fit_H_flux_matching
from deviation_spectroscopy.tests.helpers_invariance import evaluate_H_on_dataset
from deviation_spectroscopy.sim.forcing import sine_force


def _add_output_noise(y: np.ndarray, snr_db: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=float)
    if y.ndim == 1:
        y = y.reshape(-1, 1)

    p_signal = np.mean(y**2)
    if p_signal <= 0.0:
        return y.copy()

    snr_lin = 10.0 ** (snr_db / 10.0)
    p_noise = p_signal / snr_lin
    sigma = np.sqrt(p_noise)

    noise = rng.normal(loc=0.0, scale=sigma, size=y.shape)
    return y + noise


def test_partial_observation_invariance_with_output_noise():
    from deviation_spectroscopy.systems.coupled_oscillator import make_coupled_oscillator

    sys = make_coupled_oscillator()
    A_true = sys.A
    B_true = sys.B

    n = A_true.shape[0]
    z0 = np.zeros(n)

    C = np.zeros((1, n))
    C[0, 0] = 1.0

    t_id = np.linspace(0.0, 6.0, 3001)
    t_steady = np.linspace(0.0, 20.0, 5001)

    data = build_experiment_datasets(
        A=A_true, B=B_true, z0=z0,
        C=C,
        t_id=t_id,
        t_steady=t_steady,
        forcing_id=lambda t: sine_force(t, 1.0, 0.6),
        forcing_A=lambda t: sine_force(t, 1.0, 1.1),
        forcing_B=lambda t: sine_force(t, 1.7, 2.0),  
    )

    yA = data["A"].y
    yB = data["B"].y

    snr_db = 20.0
    yA_noisy = _add_output_noise(yA, snr_db=snr_db, seed=123)
    yB_noisy = _add_output_noise(yB, snr_db=snr_db, seed=456)

    horizon = 25
    order = min(6, n + 2) 

    proj = SubspaceProjector(order=order, horizon=horizon, scale="sqrt", smooth_window=9)
    zhat_A = proj.fit_transform(yA_noisy)
    zhat_B = proj.transform(yB_noisy)

    N_eff = zhat_A.shape[0]
    tA_eff = data["A"].t[:N_eff]
    uA_eff = data["A"].u[:N_eff]
    uB_eff = data["B"].u[:N_eff]

    id_res = identify_ab_from_reconstructed_state(
        t=tA_eff,
        z_hat=zhat_A,
        u=uA_eff,
        reg=1e-6,
        smooth_window=9,
    )
    A_hat = id_res.A_hat
    B_hat = id_res.B_hat

    flux_res = fit_H_flux_matching(
        t=tA_eff,
        z=zhat_A,
        u=uA_eff,
        A=A_hat,
        B=B_hat,
        window=400,
        stride=400,
        maxiter=350,
    )
    H_star = flux_res.H

    class _D:
        def __init__(self, name, t, z, u):
            self.name = name
            self.t = t
            self.z = z
            self.u = u

    dsA_hat = _D("A_reconstructed", tA_eff, zhat_A, uA_eff)
    dsB_hat = _D("B_reconstructed", data["B"].t[:N_eff], zhat_B, uB_eff)

    evalA = evaluate_H_on_dataset(dataset=dsA_hat, A=A_hat, B=B_hat, H=H_star, window=400, stride=400)
    evalB = evaluate_H_on_dataset(dataset=dsB_hat, A=A_hat, B=B_hat, H=H_star, window=400, stride=400)

    T_rel = abs(evalA["T_delta"] - evalB["T_delta"]) / evalA["T_delta"]

    assert T_rel < 0.30

    res_ratio_rel = abs(evalA["residual_ratio"] - evalB["residual_ratio"]) / evalA["residual_ratio"]
    assert res_ratio_rel < 0.30