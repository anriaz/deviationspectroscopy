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
import pytest
from deviation_spectroscopy.systems.base import LinearSystem


class DummySystem(LinearSystem):
    def dims(self):
        return 2

    def matrices(self):
        A = np.array([[0.0, 1.0], [-1.0, -0.1]])
        B = np.array([[0.0], [1.0]])
        return A, B


def test_base_interface():
    sys = DummySystem()
    A, B = sys.matrices()
    assert A.shape == (2, 2)
    assert B.shape == (2, 1)
    assert sys.noise_matrix() is None
    assert isinstance(sys.metadata(), dict)