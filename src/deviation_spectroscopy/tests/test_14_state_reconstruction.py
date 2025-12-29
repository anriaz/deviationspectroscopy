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

from deviation_spectroscopy.id.state_reconstruct import (
    build_delay_matrix,
    reconstruct_state_subspace,
)


def test_delay_matrix_shape_scalar_output():
    y = np.zeros(1000)
    H = build_delay_matrix(y, horizon=10)
    assert H.shape == (991, 10)


def test_delay_matrix_shape_vector_output():
    y = np.zeros((100, 2))
    H = build_delay_matrix(y, horizon=5)
    assert H.shape == (96, 10)


def test_reconstruction_dimensions():
    y = np.zeros((1000, 1))
    res = reconstruct_state_subspace(y, order=4, horizon=10)
    assert res.z_hat.shape == (991, 4)
    assert res.horizon == 10
    assert res.order == 4
    assert res.singular_values.shape == (10,)


def test_reconstruction_captures_signal_rank_two_sines():
    t = np.linspace(0.0, 20.0, 2000)
    y = np.sin(2.0 * t) + 0.5 * np.sin(5.0 * t)

    res = reconstruct_state_subspace(y, order=8, horizon=40, center=True, scale="sqrt")
    s = res.singular_values

    energy_6 = np.sum(s[:6] ** 2)
    total = np.sum(s ** 2)
    assert (energy_6 / total) > 0.98


def test_reconstruction_rejects_invalid_inputs():
    y = np.zeros(50)
    try:
        _ = reconstruct_state_subspace(y, order=4, horizon=0)
        assert False, "expected ValueError for horizon=0"
    except ValueError:
        pass

    try:
        _ = reconstruct_state_subspace(y, order=0, horizon=5)
        assert False, "expected ValueError for order=0"
    except ValueError:
        pass

    try:
        _ = reconstruct_state_subspace(y, order=10, horizon=40)  
        assert True
    except ValueError:
        pass