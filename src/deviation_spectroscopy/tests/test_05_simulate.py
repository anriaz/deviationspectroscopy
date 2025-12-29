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
from deviation_spectroscopy.sim.integrate import simulate_lti_rk4
from deviation_spectroscopy.sim.forcing import sine_force


def test_sim_shapes_scalar_input():
    A = np.array([[0.0, 1.0], [-1.0, -0.2]])
    B = np.array([[0.0], [1.0]])

    t = np.linspace(0.0, 5.0, 2001)
    z0 = np.array([0.0, 0.0])

    def u_of_t(tt):
        return sine_force(tt, amplitude=1.0, frequency=1.0)

    out = simulate_lti_rk4(A, B, t, z0, u_of_t)

    assert out.t.shape == (len(t),)
    assert out.z.shape == (len(t), 2)
    assert out.u.shape == (len(t), 1)


def test_sim_zero_input_stability():
    A = np.array([[0.0, 1.0], [-2.0, -0.5]])
    B = np.array([[0.0], [1.0]])

    t = np.linspace(0.0, 10.0, 5001)
    z0 = np.array([1.0, 0.0])

    def u_of_t(tt):
        return np.zeros_like(tt)

    out = simulate_lti_rk4(A, B, t, z0, u_of_t)

    assert np.max(np.linalg.norm(out.z, axis=1)) < 10.0


def test_requires_uniform_grid():
    A = np.eye(2)
    B = np.zeros((2, 1))
    t = np.array([0.0, 0.1, 0.25]) 
    z0 = np.zeros(2)

    def u_of_t(tt):
        return np.zeros_like(tt)

    try:
        simulate_lti_rk4(A, B, t, z0, u_of_t)
        assert False, "Expected ValueError for non-uniform time grid."
    except ValueError:
        pass