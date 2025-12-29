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
from typing import Callable, Optional

Array = np.ndarray


def zero_force(t: Array) -> Array:
    return np.zeros_like(t)


def sine_force(
    t: Array,
    amplitude: float = 1.0,
    frequency: float = 1.0,
    phase: float = 0.0,
) -> Array:
    return amplitude * np.sin(2.0 * np.pi * frequency * t + phase)


def multi_sine_force(
    t: Array,
    amplitudes: Array,
    frequencies: Array,
    phases: Optional[Array] = None,
) -> Array:
    amplitudes = np.asarray(amplitudes)
    frequencies = np.asarray(frequencies)

    if phases is None:
        phases = np.zeros_like(frequencies)
    else:
        phases = np.asarray(phases)

    u = np.zeros_like(t)
    for a, f, p in zip(amplitudes, frequencies, phases):
        u += a * np.sin(2.0 * np.pi * f * t + p)
    return u


def band_limited_noise(
    t: Array,
    rms: float = 1.0,
    f_low: float = 0.1,
    f_high: float = 5.0,
    seed: Optional[int] = None,
) -> Array:
    rng = np.random.default_rng(seed)
    n = len(t)
    dt = t[1] - t[0]

    white = rng.standard_normal(n)

    freqs = np.fft.rfftfreq(n, dt)
    spectrum = np.fft.rfft(white)

    mask = (freqs >= f_low) & (freqs <= f_high)
    spectrum[~mask] = 0.0

    u = np.fft.irfft(spectrum, n=n)

    current_rms = np.sqrt(np.mean(u**2))
    if current_rms > 0:
        u *= rms / current_rms

    return u


def scale_amplitude(u: Array, scale: float) -> Array:
    return scale * u


def rms(u: Array) -> float:
    return float(np.sqrt(np.mean(u**2)))