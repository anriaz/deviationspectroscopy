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
import numpy as np
import json
from pathlib import Path

from deviation_spectroscopy.sim.integrate import simulate_lti_rk4
from deviation_spectroscopy.sim.forcing import sine_force
from deviation_spectroscopy.flux.flux_match import fit_H_flux_matching
from deviation_spectroscopy.tests.helpers_invariance import evaluate_H_on_dataset

from deviation_spectroscopy.systems.coupled_oscillator import make_coupled_oscillator
from deviation_spectroscopy.experiments.run_discovery_test import DS


OUTDIR = Path("results/negative_controls")
OUTDIR.mkdir(parents=True, exist_ok=True)

SEED = 0
WINDOW = 400
STRIDE = 400


def time_shuffle(x):
    idx = np.random.permutation(len(x))
    return x[idx]


def phase_scramble(x):
    X = np.fft.rfft(x, axis=0)
    phase = np.exp(1j * np.random.uniform(0, 2*np.pi, size=X.shape))
    return np.fft.irfft(np.abs(X) * phase, n=len(x), axis=0)


def wrong_input(u):
    return np.roll(u, shift=200, axis=0)


def run_case(name, z, u, t, A, B):
    try:
        flux = fit_H_flux_matching(
            t=t,
            z=z,
            u=u,
            A=A,
            B=B,
            window=WINDOW,
            stride=STRIDE,
            maxiter=2000,
        )

        ds = DS(name, t, z, u)
        ev = evaluate_H_on_dataset(
            dataset=ds,
            A=A,
            B=B,
            H=flux.H,
            window=WINDOW,
            stride=STRIDE,
        )

        return {
            "success": bool(flux.success),
            "objective": float(flux.objective),
            "min_eig_H": float(flux.min_eig_H),
            "T_delta": float(ev["T_delta"]),
            "residual_ratio": float(ev["residual_ratio"]),
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


def main():
    np.random.seed(SEED)

    sys = make_coupled_oscillator()
    A, B = sys.A, sys.B
    n = A.shape[0]
    z0 = np.zeros(n)

    t = np.linspace(0.0, 20.0, 5001)
    forcing = lambda tt: sine_force(tt, 1.0, 1.0)
    sim = simulate_lti_rk4(A, B, t, z0, forcing)

    base_z = sim.z
    base_u = sim.u

    results = {}

    results["time_shuffle"] = run_case(
        "time_shuffle",
        time_shuffle(base_z),
        base_u,
        t, A, B
    )

    results["phase_scramble"] = run_case(
        "phase_scramble",
        phase_scramble(base_z),
        base_u,
        t, A, B
    )

    results["wrong_input"] = run_case(
        "wrong_input",
        base_z,
        wrong_input(base_u),
        t, A, B
    )

    out = OUTDIR / "negative_controls_summary.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)

    print(f"[saved] {out}")


if __name__ == "__main__":
    main()