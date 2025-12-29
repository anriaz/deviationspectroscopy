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
from typing import List, Tuple

Array = np.ndarray


def make_windows(
    N: int,
    L: int,
    stride: int,
) -> List[np.ndarray]:
    if L <= 0 or stride <= 0:
        raise ValueError("L and stride must be positive")
    if L > N:
        raise ValueError("Window length L cannot exceed signal length N")

    windows = []
    start = 0
    while start + L <= N:
        windows.append(np.arange(start, start + L))
        start += stride

    return windows


def make_taper(
    L: int,
    kind: str = "hann",
    tukey_alpha: float = 0.5,
) -> Array:
    if L <= 0:
        raise ValueError("L must be positive")

    if kind == "boxcar":
        w = np.ones(L)

    elif kind == "hann":
        w = np.hanning(L)

    elif kind == "tukey":
        if not (0.0 <= tukey_alpha <= 1.0):
            raise ValueError("tukey_alpha must be in [0, 1]")
        n = np.arange(L)
        w = np.ones(L)

        edge = int(tukey_alpha * (L - 1) / 2)
        if edge > 0:
            ramp = 0.5 * (1 + np.cos(np.pi * (2 * n[:edge] / (tukey_alpha * (L - 1)) - 1)))
            w[:edge] = ramp
            w[-edge:] = ramp[::-1]

    else:
        raise ValueError(f"Unknown taper kind '{kind}'")

    return w * (L / np.sum(w))


def windowed_integral(
    x: Array,
    windows: List[np.ndarray],
    taper: Array,
    dt: float,
) -> Array:
    if dt <= 0:
        raise ValueError("dt must be positive")

    x = np.asarray(x)
    taper = np.asarray(taper)

    L = taper.shape[0]
    out = []

    for idx in windows:
        if idx.shape[0] != L:
            raise ValueError("Window length and taper length mismatch")

        segment = x[idx]
        weighted = taper[:, None] * segment if segment.ndim == 2 else taper * segment
        out.append(np.sum(weighted, axis=0) * dt)

    return np.array(out)