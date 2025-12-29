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
from typing import Tuple, Dict, Any
from deviation_spectroscopy.systems.base import LinearSystem

Array = np.ndarray


class TwoMassSpringDamper(LinearSystem):

    def __init__(
        self,
        m1: float = 1.0,
        m2: float = 1.0,
        k1: float = 1.0,
        k2: float = 1.0,
        c1: float = 0.1,
        c2: float = 0.1,
    ):
        self.m1 = m1
        self.m2 = m2
        self.k1 = k1
        self.k2 = k2
        self.c1 = c1
        self.c2 = c2

    def dims(self) -> int:
        return 4

    def matrices(self) -> Tuple[Array, Array]:
        m1, m2 = self.m1, self.m2
        k1, k2 = self.k1, self.k2
        c1, c2 = self.c1, self.c2

        A = np.array([
            [0,        1,        0,        0],
            [-(k1+k2)/m1, -(c1+c2)/m1,  k2/m1,  c2/m1],
            [0,        0,        0,        1],
            [k2/m2,    c2/m2,   -k2/m2,   -c2/m2],
        ])

        B = np.array([
            [0.0],
            [1.0 / m1],
            [0.0],
            [0.0],
        ])

        return A, B

    def metadata(self) -> Dict[str, Any]:
        return {
            "system": "two_mass_spring_damper",
            "m1": self.m1,
            "m2": self.m2,
            "k1": self.k1,
            "k2": self.k2,
            "c1": self.c1,
            "c2": self.c2,
        }