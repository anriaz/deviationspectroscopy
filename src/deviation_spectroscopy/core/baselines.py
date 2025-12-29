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
from scipy.linalg import solve_continuous_lyapunov

from deviation_spectroscopy.core.linalg import sym, project_spd, normalize_trace

Array = np.ndarray


def lyapunov_baseline_H(A: Array, trace: float = 1.0, eps: float = 1e-10) -> Array:
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    Q = np.eye(n)
    H = solve_continuous_lyapunov(A.T, -Q)
    H = sym(H)
    H = project_spd(H, eps=eps)
    H = normalize_trace(H, trace=trace)
    return H


def covariance_baseline_H(z: Array, trace: float = 1.0, eps: float = 1e-10) -> Array:
    z = np.asarray(z, dtype=float)
    P = (z.T @ z) / float(z.shape[0])
    P = sym(P)
    P = P + eps * np.eye(P.shape[0])
    H = np.linalg.inv(P)
    H = sym(H)
    H = project_spd(H, eps=eps)
    H = normalize_trace(H, trace=trace)
    return H