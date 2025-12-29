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
from deviation_spectroscopy.core.windows import (
    make_windows, make_taper, windowed_integral
)


def test_make_windows_basic():
    windows = make_windows(N=10, L=4, stride=2)
    assert len(windows) == 4
    assert np.all(windows[0] == np.array([0, 1, 2, 3]))
    assert np.all(windows[1] == np.array([2, 3, 4, 5]))


def test_taper_normalization():
    for kind in ["boxcar", "hann", "tukey"]:
        w = make_taper(L=16, kind=kind)
        assert np.isclose(np.sum(w), 16.0)


def test_constant_signal_integral():
    N = 100
    L = 20
    stride = 10
    dt = 0.1

    x = np.ones(N)
    windows = make_windows(N, L, stride)
    taper = make_taper(L, kind="hann")

    I = windowed_integral(x, windows, taper, dt)

    expected = L * dt
    assert np.allclose(I, expected)


def test_vector_signal_integral():
    N = 50
    L = 10
    stride = 5
    dt = 0.2

    x = np.ones((N, 3))
    windows = make_windows(N, L, stride)
    taper = make_taper(L, kind="boxcar")

    I = windowed_integral(x, windows, taper, dt)
    assert I.shape == (len(windows), 3)
    assert np.allclose(I, L * dt)