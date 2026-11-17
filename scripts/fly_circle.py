"""Fly a circle with the baseline controller and record (x, u, x_next) tuples.

Usage:
    pixi run python scripts/fly_circle.py            # fly, print tracking error, save data
    pixi run python scripts/fly_circle.py --render   # with MuJoCo viewer
    pixi run python scripts/fly_circle.py --plot     # with matplotlib plots
"""

import argparse
from pathlib import Path

from adaptive_rl.plant import Plant  # isort: skip  (imports crazyflow before scipy)

import numpy as np

from adaptive_rl.controllers import CascadedController
from adaptive_rl.reference import Circle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=12.0, help="flight time [s]")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    ref_traj = Circle(center=(0.0, 0.0, 1.0), radius=0.5, period=4.0)
    plant = Plant(control_freq=100, p0=ref_traj.start_position())
    controller = CascadedController(mass=plant.nominal_mass)

    n_steps = int(args.duration * plant.control_freq)
    X = np.zeros((n_steps, 18))
    U = np.zeros((n_steps, 4))
    X_next = np.zeros((n_steps, 18))
    T = np.zeros(n_steps)
    P_ref = np.zeros((n_steps, 3))

    x = plant.reset()
    for k in range(n_steps):
        t = k * plant.dt
        ref = ref_traj(t)
        u = controller(x, ref)
        x_next = plant.step(u)

        X[k], U[k], X_next[k], T[k], P_ref[k] = x, u, x_next, t, ref[0]
        x = x_next  # the measurement after this step is the input for the next one

        if not np.all(np.isfinite(x)) or x[2] < 0.05:
            print(f"Abort at t = {t:.2f} s: state non-finite or drone near ground (z = {x[2]:.3f})")
            X, U, X_next, T, P_ref = X[: k + 1], U[: k + 1], X_next[: k + 1], T[: k + 1], P_ref[: k + 1]
            break
        if args.render:
            plant.render()
    plant.close()

    err = np.linalg.norm(X_next[:, 0:3] - P_ref, axis=1)
    settled = T > 2.0  # ignore the start-up transient
    print(f"tracking error after 2 s: mean {err[settled].mean() * 100:.1f} cm, max {err[settled].max() * 100:.1f} cm")
    print(f"thrust: min {U[:, 3].min():.3f} N, max {U[:, 3].max():.3f} N (hover = {plant.nominal_mass * 9.81:.3f} N)")

    out = Path("results/fly_circle.npz")
    out.parent.mkdir(exist_ok=True)
    np.savez(out, X=X, U=U, X_next=X_next, T=T, P_ref=P_ref)
    print(f"saved {len(T)} tuples to {out}")

    if args.plot:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
        ax[0].plot(P_ref[:, 0], P_ref[:, 1], "--", label="reference")
        ax[0].plot(X[:, 0], X[:, 1], label="flown")
        ax[0].set(xlabel="x [m]", ylabel="y [m]", aspect="equal", title="top view")
        ax[0].legend()
        ax[1].plot(T, err * 100)
        ax[1].set(xlabel="t [s]", ylabel="position error [cm]", title="tracking error")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
