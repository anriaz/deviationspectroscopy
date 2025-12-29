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

Array = np.ndarray


def compute_T_delta(z: Array, H: Array, S_H: Array, eps: float = 1e-15) -> float:
    z = np.asarray(z, dtype=float)
    H = np.asarray(H, dtype=float)
    S_H = np.asarray(S_H, dtype=float)

    E = 0.5 * np.einsum("bi,ij,bj->b", z, H, z)
    Sig = np.einsum("bi,ij,bj->b", z, S_H, z)

    E_mean = float(np.mean(E))
    Sig_mean = float(np.mean(Sig))
    return E_mean / max(Sig_mean, eps)


def compute_S_H_drift(S_A: Array, S_B: Array, eps: float = 1e-15) -> float:
    S_A = np.asarray(S_A, dtype=float)
    S_B = np.asarray(S_B, dtype=float)
    num = np.linalg.norm(S_A - S_B, ord="fro")
    den = max(np.linalg.norm(S_A, ord="fro"), eps)
    return float(num / den)