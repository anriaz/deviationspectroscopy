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
from typing import Callable, Tuple, Optional
import numpy as np

Array = np.ndarray


@dataclass(frozen=True)
class SimResult:
    t: Array          
    z: Array          
    u: Array          


def _as_column(u: Array) -> Array:
    u = np.asarray(u)
    if u.ndim == 0:
        return u.reshape(1, 1)
    if u.ndim == 1:
        return u.reshape(-1, 1)
    return u


def simulate_lti_rk4(
    A: Array,
    B: Array,
    t: Array,
    z0: Array,
    u_of_t: Callable[[Array], Array],
) -> SimResult:
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    t = np.asarray(t, dtype=float)
    z0 = np.asarray(z0, dtype=float)

    n = A.shape[0]
    assert A.shape == (n, n)
    assert B.shape[0] == n
    m = B.shape[1]

    N = len(t)
    assert N >= 2
    dt = t[1] - t[0]
    if not np.allclose(np.diff(t), dt, atol=0.0, rtol=1e-12):
        raise ValueError("Time grid t must be uniformly spaced for this RK4 integrator.")

    u_raw = u_of_t(t)
    u_raw = np.asarray(u_raw, dtype=float)
    if u_raw.ndim == 1:
        u_raw = u_raw.reshape(-1, 1)
    if u_raw.shape[0] != N:
        raise ValueError("u_of_t(t) must return an array with first dimension = len(t).")
    if u_raw.shape[1] == 1 and m != 1:
        raise ValueError(f"u has 1 column but B expects m={m}.")
    if u_raw.shape[1] != m:
        raise ValueError(f"u has {u_raw.shape[1]} columns but B expects m={m}.")

    z = np.zeros((N, n), dtype=float)
    z[0] = z0

    def f(zz: Array, uu: Array) -> Array:
        return (A @ zz) + (B @ uu)

    for k in range(N - 1):
        uu0 = u_raw[k]
        uu1 = u_raw[k + 1]
        tt0 = t[k]
        uum = 0.5 * (uu0 + uu1)

        z_k = z[k]

        k1 = f(z_k, uu0)
        k2 = f(z_k + 0.5 * dt * k1, uum)
        k3 = f(z_k + 0.5 * dt * k2, uum)
        k4 = f(z_k + dt * k3, uu1)

        z[k + 1] = z_k + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    return SimResult(t=t, z=z, u=u_raw)