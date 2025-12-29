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
from numpy.linalg import lstsq, cond, norm

Array = np.ndarray


@dataclass(frozen=True)
class ABEstimate:
    A: Array
    B: Array
    residual_norm: float
    condition_number: float


def estimate_ab_ridge(
    z: Array,
    u: Array,
    dt: float,
    ridge: float = 1e-6,
) -> ABEstimate:
    z = np.asarray(z, dtype=float)
    u = np.asarray(u, dtype=float)

    if u.ndim == 1:
        u = u.reshape(-1, 1)

    N, n = z.shape
    m = u.shape[1]

    if u.shape[0] != N:
        raise ValueError("z and u must have the same number of samples.")

    z_dot = (z[2:] - z[:-2]) / (2.0 * dt)

    Z = z[1:-1]          
    U = u[1:-1]          

    X = np.hstack([Z, U])   

    XtX = X.T @ X
    reg = ridge * np.eye(XtX.shape[0])
    Theta = np.linalg.solve(XtX + reg, X.T @ z_dot)

    A_hat = Theta[:n, :].T
    B_hat = Theta[n:, :].T

    residual = z_dot - (Z @ A_hat.T + U @ B_hat.T)
    res_norm = norm(residual) / np.sqrt(residual.size)

    return ABEstimate(
        A=A_hat,
        B=B_hat,
        residual_norm=float(res_norm),
        condition_number=float(cond(XtX)),
    )


def identify_ab(
    z: Array,
    u: Array,
    dt: float,
    ridge: float = 1e-6,
) -> ABEstimate:
    return estimate_ab_ridge(
        z=z,
        u=u,
        dt=dt,
        ridge=ridge,
    )