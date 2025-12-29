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
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from deviation_spectroscopy.id.identify_ab import estimate_ab_ridge

Array = np.ndarray


@dataclass(frozen=True)
class ReconstructedIDResult:
    A_hat: Array
    B_hat: Array
    residual_norm: float


def _moving_average_z(z: Array, *, smooth_window: int) -> Array:
    if smooth_window <= 1:
        return np.asarray(z, dtype=float)
    if smooth_window % 2 == 0:
        raise ValueError("smooth_window must be odd")

    z2 = np.asarray(z, dtype=float)
    if z2.ndim != 2:
        raise ValueError("z must be 2D")

    k = smooth_window // 2
    zpad = np.pad(z2, ((k, k), (0, 0)), mode="reflect")
    kernel = np.ones(smooth_window, dtype=float) / float(smooth_window)

    out = np.empty_like(z2, dtype=float)
    for j in range(z2.shape[1]):
        out[:, j] = np.convolve(zpad[:, j], kernel, mode="valid")
    return out


def identify_ab_from_reconstructed_state(
    *,
    t: Array,
    z_hat: Array,
    u: Array,
    reg: float = 1e-8,
    smooth_window: int = 1,
) -> ReconstructedIDResult:
    z_hat = np.asarray(z_hat, dtype=float)
    u = np.asarray(u, dtype=float)

    if z_hat.ndim != 2:
        raise ValueError("z_hat must be 2D")
    if u.ndim == 1:
        u = u.reshape(-1, 1)

    if z_hat.shape[0] != u.shape[0]:
        raise ValueError("z_hat and u must have same number of samples")

    t = np.asarray(t, dtype=float)
    if t.ndim != 1 or t.size < 2:
        raise ValueError("t must be a 1D time array with at least 2 points")

    dt = float(np.mean(np.diff(t)))

    z_smooth = _moving_average_z(z_hat, smooth_window=smooth_window)

    est = estimate_ab_ridge(
        z=z_smooth,
        u=u,
        dt=dt,
        ridge=reg,
    )

    return ReconstructedIDResult(
        A_hat=est.A,
        B_hat=est.B,
        residual_norm=float(est.residual_norm),
    )