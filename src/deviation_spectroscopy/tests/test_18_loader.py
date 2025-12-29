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
from pathlib import Path

from deviation_spectroscopy.data.loader import load_csv_timeseries, preprocess_detrend


def test_loader_simple_csv(tmp_path: Path):
    p = tmp_path / "test_data.csv"
    with open(p, "w") as f:
        f.write("time,voltage,current\n")
        f.write("0.0,10.0,1.0\n")
        f.write("0.1,10.1,1.1\n")
        f.write("0.2,10.2,1.2\n")

    ds = load_csv_timeseries(
        p,
        time_col="time",
        data_cols=["voltage"],
        input_cols=["current"],
    )

    assert ds.t.shape == (3,)
    assert ds.y.shape == (3, 1)
    assert ds.u is not None
    assert ds.u.shape == (3, 1)

    assert np.isclose(ds.t[1], 0.1)
    assert np.isclose(ds.y[1, 0], 10.1)
    assert np.isclose(ds.u[2, 0], 1.2)


def test_loader_drops_nans(tmp_path: Path):
    p = tmp_path / "nan_data.csv"
    with open(p, "w") as f:
        f.write("t,x\n")
        f.write("0.0,1.0\n")
        f.write("0.1,NaN\n")
        f.write("0.2,1.2\n")

    ds = load_csv_timeseries(p, time_col="t", data_cols=["x"])
    assert ds.t.shape == (2,)
    assert ds.y.shape == (2, 1)
    assert np.all(np.isfinite(ds.t))
    assert np.all(np.isfinite(ds.y))


def test_detrending(tmp_path: Path):
    p = tmp_path / "drift.csv"
    t = np.linspace(0.0, 10.0, 200)
    y = np.sin(t) + 2.0 * t + 5.0

    with open(p, "w") as f:
        f.write("t,val\n")
        for ti, yi in zip(t, y):
            f.write(f"{ti},{yi}\n")

    ds = load_csv_timeseries(p, time_col="t", data_cols=["val"])
    ds_clean = preprocess_detrend(ds, poly_order=1)

    assert abs(float(np.mean(ds_clean.y))) < 1e-8
    assert float(np.max(np.abs(ds_clean.y))) < 2.0