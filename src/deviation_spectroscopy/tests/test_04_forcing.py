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
from deviation_spectroscopy.sim.forcing import (
    zero_force,
    sine_force,
    multi_sine_force,
    band_limited_noise,
    scale_amplitude,
    rms,
)


def test_zero_force():
    t = np.linspace(0, 1, 1000)
    u = zero_force(t)
    assert np.all(u == 0.0)


def test_sine_force_amplitude():
    t = np.linspace(0, 1, 1000)
    u = sine_force(t, amplitude=2.0, frequency=1.0)
    assert np.isclose(np.max(u), 2.0, atol=1e-2)


def test_multi_sine_shape():
    t = np.linspace(0, 1, 2000)
    u = multi_sine_force(
        t,
        amplitudes=[1.0, 0.5],
        frequencies=[1.0, 3.0],
    )
    assert u.shape == t.shape


def test_band_limited_noise_rms():
    t = np.linspace(0, 10, 5000)
    u = band_limited_noise(t, rms=0.7, seed=0)
    assert np.isclose(rms(u), 0.7, atol=0.05)


def test_amplitude_scaling():
    t = np.linspace(0, 1, 1000)
    u = sine_force(t, amplitude=1.0, frequency=2.0)
    u2 = scale_amplitude(u, 3.0)
    assert np.isclose(rms(u2), 3.0 * rms(u), rtol=1e-3)