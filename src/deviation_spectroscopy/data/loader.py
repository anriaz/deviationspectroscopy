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
from pathlib import Path
from typing import Optional, List, Union
import csv
import numpy as np

Array = np.ndarray


@dataclass(frozen=True)
class TimeSeriesData:
    t: Array  
    y: Array  
    u: Optional[Array] = None  
    name: str = "unknown"


ColSpec = Union[str, int]


def _parse_float_safe(val: str) -> float:
    try:
        return float(val)
    except (ValueError, TypeError):
        return np.nan


def _resolve_index(header: List[str], spec: ColSpec) -> int:
    if isinstance(spec, int):
        return spec
    if spec not in header:
        raise ValueError(f"Column '{spec}' not found in header: {header}")
    return header.index(spec)


def load_csv_timeseries(
    path: Path | str,
    *,
    time_col: ColSpec = 0,
    data_cols: List[ColSpec] | None = None,
    input_cols: List[ColSpec] | None = None,
    delimiter: str = ",",
    skip_header: bool = True,
    t_scale: float = 1.0,
) -> TimeSeriesData:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    raw_rows: List[List[str]] = []
    header: List[str] = []

    with open(path, "r", newline="") as f:
        reader = csv.reader(f, delimiter=delimiter)
        try:
            first = next(reader)
        except StopIteration:
            raise ValueError("Empty CSV file")

        if skip_header:
            header = [h.strip() for h in first]
        else:
            raw_rows.append(first)

        for row in reader:
            if row and any(cell.strip() != "" for cell in row):
                raw_rows.append(row)

    if len(raw_rows) == 0:
        raise ValueError("CSV has no data rows")

    if skip_header:
        time_idx = _resolve_index(header, time_col)

        input_indices: List[int] = []
        if input_cols:
            input_indices = [_resolve_index(header, ic) for ic in input_cols]

        if data_cols:
            data_indices = [_resolve_index(header, dc) for dc in data_cols]
        else:
            data_indices = [i for i in range(len(header)) if i != time_idx and i not in input_indices]
    else:
        if isinstance(time_col, str):
            raise ValueError("Cannot use string column names without header")
        time_idx = int(time_col)

        input_indices = [int(i) for i in input_cols] if input_cols else []
        data_indices = [int(i) for i in data_cols] if data_cols else [i for i in range(len(raw_rows[0])) if i != time_idx]

    try:
        t_raw = np.array([_parse_float_safe(r[time_idx]) for r in raw_rows], dtype=float)
    except IndexError:
        raise ValueError(f"Time index {time_idx} out of bounds for some row(s)")

    y_list: List[List[float]] = []
    for r in raw_rows:
        try:
            y_list.append([_parse_float_safe(r[i]) for i in data_indices])
        except IndexError:
            raise ValueError("Some data row is missing expected columns (ragged CSV)")
    y = np.array(y_list, dtype=float)

    u = None
    if input_indices:
        u_list: List[List[float]] = []
        for r in raw_rows:
            try:
                u_list.append([_parse_float_safe(r[i]) for i in input_indices])
            except IndexError:
                raise ValueError("Some input row is missing expected columns (ragged CSV)")
        u = np.array(u_list, dtype=float)

    valid = np.isfinite(t_raw) & np.all(np.isfinite(y), axis=1)
    if u is not None:
        valid &= np.all(np.isfinite(u), axis=1)

    dropped = int(np.sum(~valid))
    if dropped > 0:
        print(f"[loader] dropped {dropped} row(s) containing NaNs/Inf")

    t = t_raw[valid] * float(t_scale)
    y = y[valid]
    if u is not None:
        u = u[valid]

    if t.size < 2:
        raise ValueError("Not enough valid samples after dropping NaNs")

    sidx = np.argsort(t)
    t = t[sidx]
    y = y[sidx]
    if u is not None:
        u = u[sidx]

    return TimeSeriesData(t=t, y=y, u=u, name=path.stem)


def preprocess_detrend(data: TimeSeriesData, *, poly_order: int = 1) -> TimeSeriesData:
    t = np.asarray(data.t, dtype=float)
    if t.ndim != 1 or t.size < 2:
        raise ValueError("data.t must be 1D with at least 2 samples")

    denom = (t[-1] - t[0])
    if denom == 0.0:
        raise ValueError("Time array has zero duration")
    t_norm = (t - t[0]) / denom

    y = np.asarray(data.y, dtype=float)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    if y.shape[0] != t.shape[0]:
        raise ValueError("data.y and data.t must have same number of samples")

    y_new = np.empty_like(y, dtype=float)
    for j in range(y.shape[1]):
        p = np.polyfit(t_norm, y[:, j], poly_order)
        trend = np.polyval(p, t_norm)
        y_new[:, j] = y[:, j] - trend

    u_new = None
    if data.u is not None:
        u0 = np.asarray(data.u, dtype=float)
        if u0.ndim == 1:
            u0 = u0.reshape(-1, 1)
        if u0.shape[0] != t.shape[0]:
            raise ValueError("data.u and data.t must have same number of samples")

        u_new = np.empty_like(u0, dtype=float)
        for j in range(u0.shape[1]):
            p = np.polyfit(t_norm, u0[:, j], poly_order)
            trend = np.polyval(p, t_norm)
            u_new[:, j] = u0[:, j] - trend

    return TimeSeriesData(t=t, y=y_new, u=u_new, name=f"{data.name}_detrend")