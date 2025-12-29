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
import json
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

summary = Path("results/public_grid_batch/public_grid_batch_summary.csv")
df = pd.read_csv(summary)

fig, axs = plt.subplots(1, 2, figsize=(10, 4))

axs[0].scatter(range(len(df)), df["T_A"], label="Regime A", alpha=0.7)
axs[0].scatter(range(len(df)), df["T_B"], label="Regime B", alpha=0.7)
axs[0].set_title("Absolute invariant scale (context)")
axs[0].set_ylabel(r"$T_\Delta$")
axs[0].legend()

axs[1].scatter(range(len(df)), df["T_rel"], color="black")
axs[1].axhline(0.1, ls="--", lw=1, color="red", alpha=0.5)
axs[1].set_title("Relative invariance across regimes")
axs[1].set_ylabel(r"$T_{\mathrm{rel}}$")
axs[1].set_ylim(0, max(0.2, df["T_rel"].max() * 1.1))

plt.tight_layout()
plt.savefig("results/figures_paper/public_grid_application_figure.pdf")
plt.show()