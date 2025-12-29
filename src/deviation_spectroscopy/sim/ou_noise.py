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

def ou_process(
    t: Array,
    *,
    tau: float = 0.5,
    sigma: float = 0.25,
    seed: int = 0,
) -> Array:
    t = np.asarray(t, dtype=float)
    if t.ndim != 1 or t.size < 2:
        raise ValueError("t must be 1D with length >= 2")

    dt = float(np.median(np.diff(t)))
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("t must be strictly increasing with finite step")

    rng = np.random.default_rng(seed)
    x = np.zeros_like(t, dtype=float)

    a = -1.0 / max(tau, 1e-12)
    s = sigma * np.sqrt(dt)

    for k in range(len(t) - 1):
        x[k + 1] = x[k] + a * x[k] * dt + s * rng.standard_normal()

    return x