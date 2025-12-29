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

Array = np.ndarray


def sym(A: Array) -> Array:
    return 0.5 * (A + A.T)


def project_spd(A: Array, eps: float = 1e-12) -> Array:
    A = sym(A)
    w, V = np.linalg.eigh(A)
    w_clipped = np.maximum(w, eps)
    return V @ np.diag(w_clipped) @ V.T


def normalize_trace(H: Array, trace: float = 1.0) -> Array:
    tr = np.trace(H)
    if tr <= 0:
        raise ValueError("Cannot normalize matrix with non-positive trace")
    return H * (trace / tr)


def is_spd(H: Array, tol: float = 1e-12) -> bool:
    if not np.allclose(H, H.T, atol=tol):
        return False
    w = np.linalg.eigvalsh(H)
    return np.all(w > tol)


def is_psd(H: Array, tol: float = 1e-12) -> bool:
    if not np.allclose(H, H.T, atol=tol):
        return False
    w = np.linalg.eigvalsh(H)
    return np.all(w >= -tol)


def fro_norm(A: Array) -> float:
    return float(np.linalg.norm(A, ord="fro"))