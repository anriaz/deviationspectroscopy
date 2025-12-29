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
import pandas as pd
from pathlib import Path

INP = Path("results/negative_controls/negative_controls_summary.json")
OUT_CSV = Path("results/tables/negative_controls.csv")
OUT_TEX = Path("results/tables/negative_controls.tex")

def main():
    with open(INP) as f:
        data = json.load(f)

    rows = []
    for k, v in data.items():
        display_name = k.replace("_", " ").title()
        
        rows.append({
            "control": display_name,
            "success": v.get("success", False),
            "drift": v.get("T_delta"),
            "residual": v.get("residual_ratio"),
            "min_eig": v.get("min_eig_H"),
        })

    df = pd.DataFrame(rows)
    
    df_display = df.copy()
    df_display['drift'] = df_display['drift'].map('{:.2e}'.format)
    df_display['residual'] = df_display['residual'].map('{:.2f}'.format)

    df_display.to_csv(OUT_CSV, index=False)
    
    df_display.rename(columns={
        "control": "Control Condition",
        "success": "Optim. Converged",
        "drift": "Metric Drift",
        "residual": "Flux Residual Ratio"
    }, inplace=True)
    
    df_display.to_latex(OUT_TEX, index=False)

    print("[saved] negative control tables")

if __name__ == "__main__":
    main()