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
from dataclasses import dataclass
from typing import Callable, Dict, Optional
import numpy as np

from deviation_spectroscopy.sim.integrate import simulate_lti_rk4

Array = np.ndarray


@dataclass(frozen=True)
class Dataset:
    name: str
    t: Array
    z: Array
    u: Array
    y: Array  


def build_dataset(
    *,
    name: str,
    A: Array,
    B: Array,
    C: Array,
    z0: Array,
    t: Array,
    forcing: Callable[[Array], Array],
) -> Dataset:
    sim = simulate_lti_rk4(
        A=A,
        B=B,
        t=t,
        z0=z0,
        u_of_t=forcing,
    )

    z = sim.z
    y = z @ C.T

    return Dataset(
        name=name,
        t=sim.t,
        z=z,
        u=sim.u,
        y=y,
    )


def build_experiment_datasets(
    *,
    A: Array,
    B: Array,
    z0: Array,
    t_id: Array,
    t_steady: Array,
    forcing_id: Callable[[Array], Array],
    forcing_A: Callable[[Array], Array],
    forcing_B: Callable[[Array], Array],
    C: Optional[Array] = None,
) -> Dict[str, Dataset]:
    n = A.shape[0]

    if C is None:
        C = np.eye(n)

    ds0 = build_dataset(
        name="dataset_0_id",
        A=A,
        B=B,
        C=C,
        z0=z0,
        t=t_id,
        forcing=forcing_id,
    )

    dsA = build_dataset(
        name="dataset_A_flux",
        A=A,
        B=B,
        C=C,
        z0=z0,
        t=t_steady,
        forcing=forcing_A,
    )

    dsB = build_dataset(
        name="dataset_B_invariance",
        A=A,
        B=B,
        C=C,
        z0=z0,
        t=t_steady,
        forcing=forcing_B,
    )

    return {
        "id": ds0,
        "A": dsA,
        "B": dsB,
    }