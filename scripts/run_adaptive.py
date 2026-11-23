"""Closed loop: residual policy adapted online by first-order gradients, vs baseline.

Circle, payload switch mid-run (nominal -> +30 % mass at t = 10 s, never told to anyone).
Two runs with the same plant, reference and disturbance:
  baseline: the policy is never updated, so it stays exactly the baseline controller
  adaptive: one SGD step per control step on an H-step rollout in the hybrid model,
            starting at the state just measured
The residual is untrained (a_res = 0, hybrid == nominal) until the online model update exists.
Docs: docs/decisions.md #11, #17, #20.

Usage:
    pixi run python scripts/run_adaptive.py                  # table + results/figures/run_adaptive.png
    pixi run python scripts/run_adaptive.py --lr 3e-3 --horizon 30
"""

import argparse
from pathlib import Path

from adaptive_rl.plant import Plant  # isort: skip  (imports crazyflow before scipy)

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from flax import nnx

from adaptive_rl.controllers import CascadedController
from adaptive_rl.dynamics.residual import ResidualMLP
from adaptive_rl.pipelines.first_order import gradient
from adaptive_rl.policy import PolicyMLP, act
from adaptive_rl.reference import Circle

CENTER = (0.0, 0.0, 1.5)
T_ADAPT_START = 2.0  # [s] no learning during rotor spin-up (#4)
MAX_GRAD_NORM = 1.0
# Evaluation intervals [s]: before the switch, right after it, adapted.
INTERVALS = {"5-10 s (nominal)": (5.0, 10.0), "10-12 s (after switch)": (10.0, 12.0), "15-20 s": (15.0, 20.0)}


def fly(adapt: bool, args, seed: int = 0) -> dict:
    reference = Circle(center=CENTER, radius=0.5, period=4.0)
    plant = Plant(control_freq=100, p0=reference.start_position())
    dt = plant.dt
    baseline = CascadedController(plant.nominal_mass)  # knows only the nominal mass (#8)
    policy = PolicyMLP(rngs=nnx.Rngs(seed))
    residual = ResidualMLP(rngs=nnx.Rngs(seed))  # zero output layer: hybrid == nominal (#12)
    optimizer = gradient.make_optimizer(policy, args.lr, MAX_GRAD_NORM)
    act_jit = nnx.jit(act, static_argnames="baseline")

    n, H = int(args.duration / dt), args.horizon
    T = np.arange(n + 1) * dt
    # Reference at the time of every state x_k, precomputed: it does not depend on the flight.
    # Index k of X and of the references always refers to the same time. H steps beyond the
    # end, because the last rollouts look H steps ahead.
    T_ref = np.arange(n + H + 1) * dt
    P_ref, V_ref, A_ref = (np.array(a, dtype=np.float32) for a in zip(*[reference(t) for t in T_ref]))
    X = np.zeros((n + 1, 18), dtype=np.float32)
    U = np.zeros((n, 4), dtype=np.float32)
    U_base = np.zeros((n, 4), dtype=np.float32)
    loss = np.full(n, np.nan)
    grad_norm = np.full(n, np.nan)

    X[0] = plant.reset()
    k_adapt = int(T_ADAPT_START / dt)
    for k in range(n):
        if k == int(args.switch_time / dt):
            plant.set_disturbance(mass=args.mass_factor * plant.nominal_mass)

        ref = (P_ref[k], V_ref[k], A_ref[k])
        U[k] = act_jit(policy, baseline, X[k], ref)
        U_base[k] = baseline(X[k], ref)
        X[k + 1] = plant.step(U[k])

        # Rollout from the state just measured, x_{k+1}, with the reference at k+1 .. k+1+H.
        if adapt and k + 1 >= k_adapt:
            w = slice(k + 1, k + 2 + H)
            refs = (jnp.asarray(P_ref[w]), jnp.asarray(V_ref[w]), jnp.asarray(A_ref[w]))
            x0 = jnp.asarray(X[k + 1])
            loss_k, grad_norm_k = gradient.update(policy, optimizer, residual, baseline, x0, refs, dt)
            loss[k], grad_norm[k] = float(loss_k), float(grad_norm_k)

        if not np.all(np.isfinite(X[k + 1])) or X[k + 1, 2] < 0.1:
            print(f"  abort at t = {T[k + 1]:.2f} s (z = {X[k + 1, 2]:.2f} m)")
            n = k + 1
            break
    plant.close()

    err = np.linalg.norm(X[: n + 1, 0:3] - P_ref[: n + 1], axis=1)
    return dict(T=T[: n + 1], err=err, z_err=X[: n + 1, 2] - P_ref[: n + 1, 2], correction=U[:n] - U_base[:n],
                loss=loss[:n], grad_norm=grad_norm[:n], hover_thrust=plant.nominal_mass * 9.81)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-2, help="SGD step size")
    parser.add_argument("--horizon", type=int, default=30, help="rollout horizon H [control steps]")
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--switch-time", type=float, default=10.0)
    parser.add_argument("--mass-factor", type=float, default=1.3)
    parser.add_argument("--out", default="results/figures/run_adaptive.png")
    args = parser.parse_args()

    runs = {"baseline": fly(False, args), "adaptive (FO)": fly(True, args)}

    print(f"\nlr = {args.lr:g}, H = {args.horizon}, mass x{args.mass_factor} at t = {args.switch_time} s")
    print("mean position error [cm]   " + "".join(f"{name:>24s}" for name in INTERVALS))
    for name, run in runs.items():
        cells = []
        for t0, t1 in INTERVALS.values():
            sel = (run["T"] >= t0) & (run["T"] < t1)
            cells.append(f"{100 * run['err'][sel].mean():24.2f}" if sel.any() else f"{'-':>24s}")
        print(f"{name:27s}" + "".join(cells))

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for name, run in runs.items():
        axes[0].plot(run["T"], 100 * run["err"], label=name)
        axes[1].plot(run["T"], 100 * run["z_err"], label=name)
    adaptive = runs["adaptive (FO)"]
    t_u = adaptive["T"][:-1]
    axes[2].plot(t_u, adaptive["correction"][:, 3] / adaptive["hover_thrust"], label="thrust correction / (m_nom g)")
    axes[2].plot(t_u, np.linalg.norm(adaptive["correction"][:, :3], axis=1), label="|rate correction| [rad/s]")
    axes[0].set_ylabel("position error [cm]")
    axes[1].set_ylabel("height error z - z_ref [cm]")
    axes[2].set_ylabel("correction")
    axes[2].set_xlabel("t [s]")
    for ax in axes:
        ax.axvline(args.switch_time, color="0.5", ls="--", lw=1)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(f"Circle, mass x{args.mass_factor} at t = {args.switch_time:g} s; FO lr = {args.lr:g}, H = {args.horizon}")
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
