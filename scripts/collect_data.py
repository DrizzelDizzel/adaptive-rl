"""Collect (x, u, x_next) data sets for learning the dynamics model (docs/decisions.md #7, #8).

Per condition (nominal, heavier, wind) one training run on a random smooth trajectory and
test runs on fixed trajectories, all with held excitation noise. Fixed seeds: re-running the
script gives the same data.

Usage:
    pixi run python scripts/collect_data.py            # writes results/data/<condition>/<run>.npz
"""

import argparse
from pathlib import Path

from adaptive_rl.plant import Plant  # isort: skip  (imports crazyflow before scipy)

import numpy as np

from adaptive_rl.controllers import GRAVITY, CascadedController
from adaptive_rl.data import save_run
from adaptive_rl.excitation import HeldNoise
from adaptive_rl.reference import Circle, SumOfSines

CENTER = (0.0, 0.0, 1.5)

# Disturbance per condition: true mass relative to nominal, constant world-frame force [N].
CONDITIONS = {
    "nominal": dict(mass_factor=1.0, force=(0.0, 0.0, 0.0)),
    "heavy": dict(mass_factor=1.3, force=(0.0, 0.0, 0.0)),  # payload, +30 %
    "wind": dict(mass_factor=1.0, force=(0.05, 0.0, 0.0)),  # ~12 % of the weight, along +x
}

# name -> (reference, duration [s], noise seed). Train and test never share a trajectory.
RUNS = {
    "train": (lambda: SumOfSines(center=CENTER, seed=0), 60.0, 0),
    "test_hover": (lambda: Circle(center=CENTER, radius=0.0), 20.0, 100),
    "test_circle": (lambda: Circle(center=CENTER, radius=0.5, period=4.0), 20.0, 101),
}


def fly(reference, duration: float, noise_seed: int, mass_factor: float, force) -> dict:
    plant = Plant(control_freq=100, p0=reference.start_position())
    m = plant.nominal_mass
    controller = CascadedController(m)  # knows only the nominal mass
    excitation = HeldNoise(std_rate=0.5, std_thrust=0.1 * m * GRAVITY, hold_steps=10, seed=noise_seed)

    n = int(duration * plant.control_freq)
    X, U, X_next, E, T = np.zeros((n, 18)), np.zeros((n, 4)), np.zeros((n, 18)), np.zeros((n, 4)), np.zeros(n)

    x = plant.reset()
    plant.set_disturbance(mass=mass_factor * m, force=force)
    for k in range(n):
        t = k * plant.dt
        e = excitation(k)
        u = controller(x, reference(t)) + e
        x_next = plant.step(u)
        X[k], U[k], X_next[k], E[k], T[k] = x, u, x_next, e, t
        x = x_next

        tilt = np.degrees(np.arccos(np.clip(x[11], -1.0, 1.0)))  # x[11] = R[2,2]
        if not np.all(np.isfinite(x)) or x[2] < 0.1 or tilt > 60.0:
            plant.close()
            raise RuntimeError(f"unsafe state at t = {t:.2f} s: z = {x[2]:.2f} m, tilt = {tilt:.0f} deg")
    plant.close()

    meta = dict(
        reference=type(reference).__name__,
        duration=duration,
        noise_seed=noise_seed,
        mass_factor=mass_factor,
        force=list(force),
        nominal_mass=m,
        control_freq=plant.control_freq,
    )
    return dict(X=X, U=U, X_next=X_next, E=E, T=T, meta=meta)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/data")
    args = parser.parse_args()

    for cond_name, disturbance in CONDITIONS.items():
        for run_name, (make_reference, duration, noise_seed) in RUNS.items():
            run = fly(make_reference(), duration, noise_seed, **disturbance)
            path = Path(args.out) / cond_name / f"{run_name}.npz"
            save_run(path, run)

            settled = run["T"] >= 2.0
            tilt = np.degrees(np.arccos(np.clip(run["X"][settled, 11], -1.0, 1.0)))
            thrust = run["U"][settled, 3] / (run["meta"]["nominal_mass"] * GRAVITY)
            print(
                f"{cond_name:8s} {run_name:12s} {len(run['T']):5d} steps | "
                f"thrust/mg [{thrust.min():.2f}, {thrust.max():.2f}] | max tilt {tilt.max():4.1f} deg | {path}"
            )


if __name__ == "__main__":
    main()
