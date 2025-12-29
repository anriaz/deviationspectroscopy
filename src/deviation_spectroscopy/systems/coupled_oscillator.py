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
from dataclasses import dataclass


@dataclass
class CoupledOscillatorSystem:
    A: np.ndarray
    B: np.ndarray
    H_true: np.ndarray  


def make_coupled_oscillator(
    *,
    k1: float = 2.0,
    k2: float = 1.5,
    c1: float = 0.4,
    c2: float = 0.3,
):

    A = np.array([
        [0.0,   1.0,   0.0,   0.0],
        [-k1 - k2, -c1,  k2,   0.0],
        [0.0,   0.0,   0.0,   1.0],
        [k2,   0.0,  -k2,  -c2],
    ])

    B = np.array([
        [0.0],
        [0.0],
        [0.0],
        [1.0],
    ])

    H_true = np.array([
        [k1 + k2, 0.0,   -k2,     0.0],
        [0.0,     1.0,    0.0,     0.0],
        [-k2,     0.0,    k2,      0.0],
        [0.0,     0.0,    0.0,     1.0],
    ])

    return CoupledOscillatorSystem(A=A, B=B, H_true=H_true)