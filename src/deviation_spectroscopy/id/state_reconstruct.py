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
from typing import Optional, Literal
import numpy as np

Array = np.ndarray
ScaleMode = Literal["sqrt", "none"]


@dataclass(frozen=True)
class ReconstructedState:
    z_hat: Array
    singular_values: Array
    horizon: int
    order: int


def _as_2d(y: Array) -> Array:
    y = np.asarray(y, dtype=float)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    if y.ndim != 2:
        raise ValueError(f"y must be 1D or 2D, got shape={y.shape}")
    if not np.all(np.isfinite(y)):
        raise ValueError("y contains NaN or Inf")
    return y


def _moving_average(y: Array, window: int) -> Array:
    if window <= 1:
        return np.asarray(y, dtype=float)

    if window % 2 == 0:
        raise ValueError("smooth_window must be odd")

    y2 = _as_2d(y)
    k = window // 2

    ypad = np.pad(y2, ((k, k), (0, 0)), mode="reflect")
    kernel = np.ones(window, dtype=float) / float(window)

    out = np.empty_like(y2, dtype=float)
    for j in range(y2.shape[1]):
        out[:, j] = np.convolve(ypad[:, j], kernel, mode="valid")
    return out


def build_delay_matrix(y: Array, horizon: int) -> Array:
    if horizon <= 0:
        raise ValueError("horizon must be positive")

    y2 = _as_2d(y)
    N, p = y2.shape
    N_eff = N - horizon + 1
    if N_eff <= 0:
        raise ValueError("Data too short for horizon")

    blocks = [y2[i : i + N_eff, :] for i in range(horizon)]
    return np.concatenate(blocks, axis=1)


class SubspaceProjector:


    def __init__(
        self,
        *,
        order: int,
        horizon: int = 20,
        scale: ScaleMode = "sqrt",
        smooth_window: int = 1,
        inv_sqrt_eps: float = 1e-8,
    ):
        if order <= 0:
            raise ValueError("order must be positive")
        if horizon < order:
            raise ValueError("horizon must be >= order")

        if smooth_window < 1 or smooth_window % 2 == 0:
            raise ValueError("smooth_window must be odd and >= 1")

        if inv_sqrt_eps <= 0:
            raise ValueError("inv_sqrt_eps must be positive")

        self.order = int(order)
        self.horizon = int(horizon)
        self.scale = scale
        self.smooth_window = int(smooth_window)
        self.inv_sqrt_eps = float(inv_sqrt_eps)

        self.mean_: Optional[Array] = None
        self.Vr_: Optional[Array] = None
        self.S_: Optional[Array] = None
        self.weights_: Optional[Array] = None

    def fit(self, y: Array) -> SubspaceProjector:
        y2 = _moving_average(y, self.smooth_window)
        H = build_delay_matrix(y2, self.horizon)

        self.mean_ = np.mean(H, axis=0, keepdims=True)
        Hc = H - self.mean_

        U, S, Vt = np.linalg.svd(Hc, full_matrices=False)

        self.S_ = S
        self.Vr_ = Vt[: self.order, :]

        if self.scale == "sqrt":
            Sr = np.maximum(S[: self.order], self.inv_sqrt_eps)
            inv_sqrt = np.diag(1.0 / np.sqrt(Sr))
            self.weights_ = self.Vr_.T @ inv_sqrt
        elif self.scale == "none":
            self.weights_ = self.Vr_.T
        else:
            raise ValueError(f"Unknown scale: {self.scale}")

        return self

    def transform(self, y: Array) -> Array:
        if self.weights_ is None:
            raise RuntimeError("Projector not fitted")

        y2 = _moving_average(y, self.smooth_window)
        H = build_delay_matrix(y2, self.horizon)
        Hc = H - self.mean_
        return Hc @ self.weights_

    def fit_transform(self, y: Array) -> Array:
        self.fit(y)
        return self.transform(y)

    @property
    def singular_values_(self) -> Array:
        if self.S_ is None:
            raise RuntimeError("Not fitted")
        return self.S_


def reconstruct_state_subspace(
    y: Array,
    *,
    order: int,
    horizon: int = 20,
    center: bool = True,
    scale: ScaleMode = "sqrt",
) -> ReconstructedState:
    if center is not True:
        raise ValueError("center=False not supported")

    proj = SubspaceProjector(order=order, horizon=horizon, scale=scale)
    z_hat = proj.fit_transform(y)

    return ReconstructedState(
        z_hat=z_hat,
        singular_values=proj.singular_values_,
        horizon=horizon,
        order=order,
    )