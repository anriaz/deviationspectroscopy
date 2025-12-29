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
from deviation_spectroscopy.flux.residuals import windowed_residual, residual_ratio
from deviation_spectroscopy.core.metrics import compute_T_delta
from deviation_spectroscopy.flux.residuals import S_of_H


def evaluate_H_on_dataset(
    *,
    dataset,
    A,
    B,
    H,
    window,
    stride,
):
    dE, r = windowed_residual(
        dataset.t,
        dataset.z,
        dataset.u,
        A,
        B,
        H,
        window=window,
        stride=stride,
    )
    rr = residual_ratio(dE, r)

    S_H = S_of_H(A, H)
    Tdelta = compute_T_delta(dataset.z, H, S_H)

    return {
        "residual_ratio": rr,
        "T_delta": Tdelta,
        "S_H": S_H,
    }