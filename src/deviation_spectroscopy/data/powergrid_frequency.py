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
from typing import Optional, Tuple
import zipfile
import csv
from datetime import datetime, timezone
import numpy as np

Array = np.ndarray


@dataclass(frozen=True)
class GridFrequencyData:
    t: Array          
    y: Array          
    name: str
    tz: Optional[str] = None


def _parse_datetime(val: str) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(val.replace("Z", "")).replace(
            tzinfo=timezone.utc
        )
    except Exception:
        return None


def load_csvzip_powergrid_frequency(
    path: str | Path,
    *,
    column: int = 1,
    dropna: bool = True,
) -> GridFrequencyData:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    try:
        import pandas as pd

        df = pd.read_csv(path, index_col=0)
        if df.shape[1] <= (column - 1):
            raise ValueError(
                f"Expected at least {column+1} columns, got {df.shape[1]+1} incl. index"
            )

        idx = pd.to_datetime(df.index, errors="coerce", utc=True)
        y_mhz = pd.to_numeric(df.iloc[:, column - 1], errors="coerce")

        if dropna:
            m = idx.notna() & y_mhz.notna()
            idx = idx[m]
            y_mhz = y_mhz[m]

        if len(idx) < 3:
            raise ValueError("Not enough valid rows")

        idx_np = idx.to_numpy(dtype="datetime64[ns]")
        t0 = idx_np[0]
        t = (idx_np - t0) / np.timedelta64(1, "s")
        t = t.astype(float)
        y = (y_mhz.to_numpy(dtype=float) * 1e-3).reshape(-1, 1)

        order = np.argsort(t)
        t = t[order]
        y = y[order]

        keep = np.ones_like(t, dtype=bool)
        keep[1:] = t[1:] > t[:-1]
        t = t[keep]
        y = y[keep]

        return GridFrequencyData(
            t=t,
            y=y,
            name=path.stem,
            tz="UTC",
        )

    except ImportError:
        pass

    timestamps = []
    values = []

    with zipfile.ZipFile(path, "r") as zf:
        names = zf.namelist()
        if not names:
            raise ValueError("Zip file is empty")

        with zf.open(names[0], "r") as f:
            reader = csv.reader(line.decode("utf-8") for line in f)
            header = next(reader, None)
            if header is None:
                raise ValueError("CSV file is empty")

            for row in reader:
                if len(row) <= column:
                    continue

                ts = _parse_datetime(row[0])
                try:
                    val = float(row[column])
                except ValueError:
                    val = np.nan

                if ts is None or not np.isfinite(val):
                    if dropna:
                        continue

                timestamps.append(ts)
                values.append(val)

    if len(timestamps) < 3:
        raise ValueError("Not enough valid rows")

    t0 = timestamps[0]
    t = np.array([(ts - t0).total_seconds() for ts in timestamps], dtype=float)
    y = (np.array(values, dtype=float) * 1e-3).reshape(-1, 1)

    order = np.argsort(t)
    t = t[order]
    y = y[order]

    keep = np.ones_like(t, dtype=bool)
    keep[1:] = t[1:] > t[:-1]
    t = t[keep]
    y = y[keep]

    return GridFrequencyData(
        t=t,
        y=y,
        name=path.stem,
        tz="UTC",
    )


def load_powergrid_frequency(
    path: str | Path,
    *,
    dropna: bool = True,
) -> GridFrequencyData:
    path = Path(path)

    if path.suffix == ".zip":
        return load_csvzip_powergrid_frequency(path, dropna=dropna)

    if path.suffix == ".csv":
        import pandas as pd

        df = pd.read_csv(path, sep=None, engine="python")

        if df.shape[1] < 2:
            raise ValueError(
                f"CSV parsing failed: expected ≥2 columns, got {df.shape[1]}"
            )

        time_col = df.columns[0]
        idx = pd.to_datetime(df[time_col], errors="coerce", utc=True)

        freq_cols = [c for c in df.columns if c.lower().startswith("f60")]
        if not freq_cols:
            raise ValueError("No f60_* frequency deviation column found")

        freq_col = freq_cols[0]
        y_mhz = pd.to_numeric(df[freq_col], errors="coerce")

        qi_cols = [c for c in df.columns if c.lower().startswith("qi")]
        if qi_cols:
            qi = pd.to_numeric(df[qi_cols[0]], errors="coerce")
            good = qi == 0
            idx = idx[good]
            y_mhz = y_mhz[good]

        if dropna:
            m = idx.notna() & y_mhz.notna()
            idx = idx[m]
            y_mhz = y_mhz[m]

        if len(idx) < 3:
            raise ValueError("Not enough valid rows")

        idx_np = idx.to_numpy(dtype="datetime64[ns]")
        t0 = idx_np[0]
        t = (idx_np - t0) / np.timedelta64(1, "s")
        t = t.astype(float)
        y = (y_mhz.to_numpy(dtype=float) * 1e-3).reshape(-1, 1)

        order = np.argsort(t)
        t = t[order]
        y = y[order]

        keep = np.ones_like(t, dtype=bool)
        keep[1:] = t[1:] > t[:-1]
        t = t[keep]
        y = y[keep]

        return GridFrequencyData(
            t=t,
            y=y,
            name=path.stem,
            tz="UTC",
        )

    raise ValueError(f"Unsupported file type: {path.suffix}")


def resample_uniform(
    data: GridFrequencyData,
    *,
    dt: float,
) -> GridFrequencyData:
    if dt <= 0:
        raise ValueError("dt must be positive")

    t = data.t
    y = data.y

    t_uniform = np.arange(t[0], t[-1], dt)
    if t_uniform.size < 2:
        raise ValueError("Resampled series too short for interpolation")

    y_uniform = np.empty((t_uniform.size, y.shape[1]), dtype=float)
    for j in range(y.shape[1]):
        y_uniform[:, j] = np.interp(t_uniform, t, y[:, j])

    return GridFrequencyData(
        t=t_uniform,
        y=y_uniform,
        name=f"{data.name}_dt{dt:g}",
        tz=data.tz,
    )


def split_two_regimes(
    data: GridFrequencyData,
    *,
    tA: Tuple[float, float],
    tB: Tuple[float, float],
) -> tuple[GridFrequencyData, GridFrequencyData]:

    def _slice(t0, t1, tag):
        m = (data.t >= t0) & (data.t <= t1)
        if np.sum(m) < 200:
            raise ValueError(f"Slice {tag} too short")

        return GridFrequencyData(
            t=data.t[m] - data.t[m][0],
            y=data.y[m],
            name=f"{data.name}_{tag}",
            tz=data.tz,
        )

    return _slice(*tA, "A"), _slice(*tB, "B")