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
from scipy.optimize import minimize

from deviation_spectroscopy.core.linalg import sym, normalize_trace, project_spd
from deviation_spectroscopy.flux.residuals import S_of_H, windowed_residual

Array = np.ndarray


@dataclass(frozen=True)
class FluxMatchResult:
    H: Array
    S_H: Array
    objective: float
    min_eig_H: float
    min_eig_S: float
    n_iter: int
    success: bool
    message: str


def _pack_L(L: Array) -> Array:
    return L[np.tril_indices(L.shape[0])]


def _unpack_L(x: Array, n: int) -> Array:
    L = np.zeros((n, n), dtype=float)
    L[np.tril_indices(n)] = x
    return L


def _H_from_params(x: Array, n: int, eps: float) -> Array:
    L = _unpack_L(x, n)
    H = L @ L.T
    H = sym(H)
    H = project_spd(H, eps=eps)          
    H = normalize_trace(H, trace=1.0)    
    return H


def fit_H_flux_matching(
    *,
    t: Array,
    z: Array,
    u: Array,
    A: Array,
    B: Array,
    window: int,
    stride: int | None = None,
    eps_H: float = 1e-8,
    penalty_mu: float = 1e6,
    x0_seed: int = 0,
    maxiter: int = 300,
) -> FluxMatchResult:
    t = np.asarray(t, dtype=float)
    z = np.asarray(z, dtype=float)
    u = np.asarray(u, dtype=float)
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)

    n = z.shape[1]

    rng = np.random.default_rng(x0_seed)
    L0 = np.eye(n) + 0.05 * rng.standard_normal((n, n))
    L0 = np.tril(L0)
    x0 = _pack_L(L0)

    def objective(x: Array) -> float:
        H = _H_from_params(x, n=n, eps=eps_H)
        S = S_of_H(A, H)

        _, r = windowed_residual(t, z, u, A, B, H, window=window, stride=stride)
        obj = float(np.sum(r * r))

        w = np.linalg.eigvalsh(S)
        neg = np.minimum(w, 0.0)
        pen = float(np.sum(neg * neg))

        return obj + penalty_mu * pen

    res = minimize(
        objective,
        x0,
        method="L-BFGS-B",
        options={"maxiter": maxiter},
    )

    H = _H_from_params(res.x, n=n, eps=eps_H)
    S_H = S_of_H(A, H)

    wH = np.linalg.eigvalsh(H)
    wS = np.linalg.eigvalsh(S_H)

    return FluxMatchResult(
        H=H,
        S_H=S_H,
        objective=float(res.fun),
        min_eig_H=float(np.min(wH)),
        min_eig_S=float(np.min(wS)),
        n_iter=int(res.nit) if hasattr(res, "nit") else -1,
        success=bool(res.success),
        message=str(res.message),
    )