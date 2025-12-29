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
from deviation_spectroscopy.systems.two_mass import TwoMassSpringDamper


def test_two_mass_dimensions():
    sys = TwoMassSpringDamper()
    A, B = sys.matrices()
    assert A.shape == (4, 4)
    assert B.shape == (4, 1)


def test_two_mass_stability_structure():
    sys = TwoMassSpringDamper()
    A, _ = sys.matrices()

    assert A[0, 1] == 1.0
    assert A[2, 3] == 1.0


def test_two_mass_metadata():
    sys = TwoMassSpringDamper(m1=2.0, k1=3.0)
    meta = sys.metadata()
    assert meta["m1"] == 2.0
    assert meta["k1"] == 3.0