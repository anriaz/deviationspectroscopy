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
import numpy as np
from deviation_spectroscopy.core.linalg import sym

Array = np.ndarray


def S_of_H(A: Array, H: Array) -> Array:
    return sym(-0.5 * (A.T @ H + H @ A))


def windowed_integrals(
    t: Array,
    z: Array,
    u: Array,
    A: Array,
    B: Array,
    H: Array,
    window: int,
    stride: int | None = None,
) -> tuple[Array, Array, Array]:
    t = np.asarray(t, dtype=float)
    z = np.asarray(z, dtype=float)
    u = np.asarray(u, dtype=float)
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    H = np.asarray(H, dtype=float)

    if u.ndim == 1:
        u = u.reshape(-1, 1)

    N = t.shape[0]
    if stride is None:
        stride = window

    if window < 2:
        raise ValueError("window must be >= 2")
    if N < window:
        raise ValueError("Not enough samples for one window.")

    S = S_of_H(A, H)

    idx0 = np.arange(0, N - window + 1, stride, dtype=int)
    idx1 = idx0 + (window - 1)

    dE = np.empty(idx0.shape[0], dtype=float)
    Ipi = np.empty_like(dE)
    Isig = np.empty_like(dE)

    HB = H @ B
    pi = np.einsum("bi,ij,bj->b", z, HB, u)
    sig = np.einsum("bi,ij,bj->b", z, S, z)

    for k, (i0, i1) in enumerate(zip(idx0, idx1)):
        z0 = z[i0]
        z1 = z[i1]
        dE[k] = 0.5 * (z1 @ H @ z1 - z0 @ H @ z0)

        tk = t[i0:i1 + 1]
        pik = pi[i0:i1 + 1]
        sigk = sig[i0:i1 + 1]

        Ipi[k] = float(np.trapezoid(pik, tk))
        Isig[k] = float(np.trapezoid(sigk, tk))

    return dE, Ipi, Isig


def windowed_residual(
    t: Array,
    z: Array,
    u: Array,
    A: Array,
    B: Array,
    H: Array,
    window: int,
    stride: int | None = None,
) -> tuple[Array, Array]:
    dE, Ipi, Isig = windowed_integrals(t, z, u, A, B, H, window, stride)
    r = dE - Ipi + Isig
    return dE, r


def residual_ratio(dE: Array, r: Array, eps: float = 1e-12) -> float:
    dE = np.asarray(dE, dtype=float)
    r = np.asarray(r, dtype=float)
    num = float(np.sqrt(np.mean(r * r)))
    den = float(np.sqrt(np.mean(dE * dE)))
    return num / max(den, eps)