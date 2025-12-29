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
import zipfile

from deviation_spectroscopy.data.powergrid_frequency import load_csvzip_powergrid_frequency, resample_uniform


def test_powergrid_loader_parses_timestamp_index(tmp_path: Path):
    csv = tmp_path / "toy.csv"
    csv.write_text(
        "time,dev_mhz\n"
        "2017-01-01 00:00:00,10.0\n"
        "2017-01-01 00:00:01,20.0\n"
        "2017-01-01 00:00:02,15.0\n"
    )
    zpath = tmp_path / "toy.csv.zip"
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(csv, arcname="toy.csv")

    data = load_csvzip_powergrid_frequency(zpath)
    assert data.t.shape == (3,)
    assert data.y.shape == (3, 1)
    assert np.isclose(data.y[0, 0], 10.0e-3)

    uni = resample_uniform(data, dt=1.0)
    assert np.allclose(np.diff(uni.t), 1.0)