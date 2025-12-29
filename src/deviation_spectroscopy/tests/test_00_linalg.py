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
from deviation_spectroscopy.core.linalg import (
    sym, project_spd, normalize_trace, is_spd, is_psd, fro_norm
)


def test_symmetry():
    A = np.random.randn(5, 5)
    S = sym(A)
    assert np.allclose(S, S.T)


def test_project_spd():
    A = np.random.randn(4, 4)
    A = 0.5 * (A + A.T)
    H = project_spd(A, eps=1e-6)
    assert is_spd(H)
    wmin = np.min(np.linalg.eigvalsh(H))
    assert np.isclose(wmin, 1e-6, atol=1e-12) or wmin > 1e-6


def test_trace_normalization():
    H = np.eye(3)
    Hn = normalize_trace(H, trace=1.0)
    assert np.isclose(np.trace(Hn), 1.0)


def test_psd_vs_spd():
    H = np.eye(3)
    assert is_spd(H)
    assert is_psd(H)

    H2 = np.diag([1.0, 0.0, 2.0])
    assert not is_spd(H2)
    assert is_psd(H2)


def test_fro_norm():
    A = np.eye(3)
    assert fro_norm(A) == np.sqrt(3)