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
from deviation_spectroscopy.core.baselines import lyapunov_baseline_H, covariance_baseline_H
from deviation_spectroscopy.core.linalg import is_spd


def test_lyapunov_baseline_spd_trace():
    A = np.array([[0.0, 1.0],
                  [-2.0, -0.6]])
    H = lyapunov_baseline_H(A, trace=1.0)
    assert is_spd(H)
    assert np.isclose(np.trace(H), 1.0, atol=1e-10)


def test_covariance_baseline_spd_trace():
    rng = np.random.default_rng(0)
    z = rng.standard_normal((5000, 3))
    H = covariance_baseline_H(z, trace=1.0)
    assert is_spd(H)
    assert np.isclose(np.trace(H), 1.0, atol=1e-10)