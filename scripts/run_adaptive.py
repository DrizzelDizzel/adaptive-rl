"""Closed loop: residual policy adapted online by first-order gradients, vs baseline.

Circle, payload switch mid-run (nominal -> +30 % mass at t = 10 s, never told to anyone).
Runs with the same plant, reference and disturbance:
  baseline:             the policy is never updated, so it stays exactly the baseline controller
  FO, nominal model:    one SGD step per control step on an H-step rollout in the model, from
                        the state just measured; the residual stays untrained (hybrid == nominal)
  FO, online model W s: the same, but first the residual takes Adam steps on a sliding window
                        of the last W s of real transitions (model first, then policy)
--noise adds held excitation noise on the thrust in the online runs (#7, #22), so the model
also sees what other thrust does.
Diagnosis: besides the model's a_res at the flown state, the mean a_res along the imagined
rollout. Only that one enters the gradient; it can be wrong although the first is right (#22).
Docs: docs/decisions.md #11, #17, #20, #22.

Usage:
    pixi run python scripts/run_adaptive.py                   # table + results/figures/run_adaptive.png
    pixi run python scripts/run_adaptive.py --windows 1 2 5 --noise 0.1 --out results/figures/noise.png
"""

import argparse
from functools import partial
from pathlib import Path

from adaptive_rl.plant import Plant  # isort: skip  (imports crazyflow before scipy)

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from flax import nnx

from adaptive_rl.controllers import CascadedController
from adaptive_rl.dynamics import fit, online
from adaptive_rl.dynamics.residual import ResidualMLP, features
from adaptive_rl.excitation import HeldNoise
from adaptive_rl.pipelines.first_order import gradient
from adaptive_rl.policy import PolicyMLP, act
from adaptive_rl.reference import Circle

CENTER = (0.0, 0.0, 1.5)
T_LEARN_START = 2.0  # [s] no learning and no model data during rotor spin-up (#4)
MAX_GRAD_NORM = 1.0
MODEL_LR = 1e-2  # Adam, as the offline fit (#14)
# Evaluation intervals [s]: before the switch, right after it, adapted.
INTERVALS = {"5-10 s (nominal)": (5.0, 10.0), "10-12 s (after switch)": (10.0, 12.0), "15-20 s": (15.0, 20.0)}


@partial(nnx.jit, static_argnames=("baseline", "dt"))
def rollout_a_res_z(policy, residual, baseline, x0, refs, dt) -> jax.Array:
    """Mean a_res,z the model predicts along the rollout the gradient is about to use: the
    residual evaluated at the imagined (x^_j, u_j), j = 0 .. H-1. Diagnosis only."""
    xs, us, _ = gradient.imagine(policy, residual, baseline, x0, refs, dt)
    states = jnp.concatenate([x0[None], xs[:-1]])  # x^_0 .. x^_{H-1}, the inputs of each step
    F = jax.vmap(features, in_axes=(0, 0, None))(states, us, baseline.mass)
    return jnp.mean(residual(F)[:, 2])


def fly(args, adapt: bool, window_s: float | None = None, seed: int = 0) -> dict:
    """adapt: update the policy. window_s: if given, update the residual online on a sliding
    window of that length; if None, the residual stays untrained (hybrid == nominal)."""
    reference = Circle(center=CENTER, radius=0.5, period=4.0)
    plant = Plant(control_freq=100, p0=reference.start_position())
    dt, mass = plant.dt, plant.nominal_mass  # the model only knows the nominal mass (#8)
    baseline = CascadedController(mass)
    policy = PolicyMLP(rngs=nnx.Rngs(seed))
    optimizer = gradient.make_optimizer(policy, args.lr, MAX_GRAD_NORM)
    residual = ResidualMLP(rngs=nnx.Rngs(seed))  # zero output layer: starts as hybrid == nominal (#12)
    model_optimizer = nnx.Optimizer(residual, optax.adam(MODEL_LR), wrt=nnx.Param)
    window = online.SlidingWindow(int(round(window_s / dt))) if window_s else None
    # Excitation only where a model learns online; held 10 steps as offline (#7).
    excitation = HeldNoise(std_rate=0.0, std_thrust=args.noise * mass * 9.81, seed=seed) if window else None
    act_jit = nnx.jit(act, static_argnames="baseline")
    predict = nnx.jit(lambda model, f: model(f))

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
    E = np.zeros((n, 4), dtype=np.float32)  # excitation noise contained in U
    a_res_rollout = np.full(n, np.nan, dtype=np.float32)  # mean a_res,z along the imagined rollout
    a_res_pred = np.zeros((n, 3), dtype=np.float32)  # model's a_res for x_k, u_k before learning from it
    a_res_target = np.zeros((n, 3), dtype=np.float32)  # a_res that explains the measured x_{k+1}

    X[0] = plant.reset()
    k_learn = int(T_LEARN_START / dt)
    for k in range(n):
        if k == int(args.switch_time / dt):
            plant.set_disturbance(mass=args.mass_factor * plant.nominal_mass)

        ref = (P_ref[k], V_ref[k], A_ref[k])
        if excitation is not None and k >= k_learn:
            E[k] = excitation(k)
        U[k] = act_jit(policy, baseline, X[k], ref) + E[k]
        U_base[k] = baseline(X[k], ref)
        X[k + 1] = plant.step(U[k])

        # Model first (#22): the newest transition is scored a priori (prediction before the
        # model has seen it), then joins the window, and the residual learns on the window.
        f, y = online.sample(X[k], U[k], X[k + 1], mass, dt)
        a_res_pred[k], a_res_target[k] = predict(residual, f), y
        if window is not None and k >= k_learn:
            window.add(f, y)
            if window.full:
                for _ in range(args.model_steps):
                    fit.train_step(residual, model_optimizer, jnp.asarray(window.F), jnp.asarray(window.Y))

        # Then the policy: rollout from the state just measured, x_{k+1}, with the reference at
        # k+1 .. k+1+H. With an online model only once the window is full (model is trained).
        model_ready = window.full if window is not None else True
        if adapt and k + 1 >= k_learn and model_ready:
            w = slice(k + 1, k + 2 + H)
            refs = (jnp.asarray(P_ref[w]), jnp.asarray(V_ref[w]), jnp.asarray(A_ref[w]))
            x0 = jnp.asarray(X[k + 1])
            a_res_rollout[k] = rollout_a_res_z(policy, residual, baseline, x0, refs, dt)
            gradient.update(policy, optimizer, residual, baseline, x0, refs, dt)

        if not np.all(np.isfinite(X[k + 1])) or X[k + 1, 2] < 0.1:
            print(f"  abort at t = {T[k + 1]:.2f} s (z = {X[k + 1, 2]:.2f} m)")
            n = k + 1
            break
    plant.close()

    err = np.linalg.norm(X[: n + 1, 0:3] - P_ref[: n + 1], axis=1)
    return dict(T=T[: n + 1], err=err, correction=U[:n] - U_base[:n] - E[:n], hover_thrust=mass * 9.81,
                a_res_pred=a_res_pred[:n], a_res_target=a_res_target[:n], a_res_rollout=a_res_rollout[:n])


def interval_means(run: dict, values: np.ndarray, scale: float = 1.0) -> str:
    """Mean of values over each evaluation interval, '-' where the run did not get that far."""
    cells = []
    for t0, t1 in INTERVALS.values():
        sel = (run["T"][: len(values)] >= t0) & (run["T"][: len(values)] < t1)
        ok = sel & np.isfinite(values)
        cells.append(f"{scale * values[ok].mean():24.2f}" if ok.any() else f"{'-':>24s}")
    return "".join(cells)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-2, help="SGD step size of the policy")
    parser.add_argument("--horizon", type=int, default=30, help="rollout horizon H [control steps]")
    parser.add_argument("--windows", type=float, nargs="+", default=[2.0], help="model window lengths [s]")
    parser.add_argument("--model-steps", type=int, default=1, help="Adam steps on the model per control step")
    parser.add_argument("--noise", type=float, default=0.0, help="held thrust noise std [m_nom g], online runs")
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--switch-time", type=float, default=10.0)
    parser.add_argument("--mass-factor", type=float, default=1.3)
    parser.add_argument("--out", default="results/figures/run_adaptive.png")
    args = parser.parse_args()

    runs = {"baseline": fly(args, adapt=False), "FO, nominal model": fly(args, adapt=True)}
    for w in args.windows:
        runs[f"FO, online model {w:g} s"] = fly(args, adapt=True, window_s=w)

    print(f"\npolicy lr = {args.lr:g}, H = {args.horizon}, model steps/tick = {args.model_steps}, "
          f"thrust noise {args.noise:g} m g, mass x{args.mass_factor} at t = {args.switch_time} s")
    header = "".join(f"{name:>24s}" for name in INTERVALS)
    print("mean position error [cm]      " + header)
    for name, run in runs.items():
        print(f"{name:30s}" + interval_means(run, run["err"], scale=100))
    print("model a-priori error |a_res_pred - a_res_target| [m/s^2]")
    for name, run in runs.items():
        print(f"{name:30s}" + interval_means(run, np.linalg.norm(run["a_res_pred"] - run["a_res_target"], axis=1)))
    print("mean a_res,z along the imagined rollout [m/s^2] (true after the switch ~ -2.9)")
    for name, run in runs.items():
        if name.startswith("FO, online"):
            print(f"{name:30s}" + interval_means(run, run["a_res_rollout"]))

    fig, axes = plt.subplots(4, 1, figsize=(10, 11), sharex=True)
    for i, (name, run) in enumerate(runs.items()):
        t_u, color = run["T"][:-1], f"C{i}"  # same color for a run in every panel
        axes[0].plot(run["T"], 100 * run["err"], color=color, label=name)
        axes[1].plot(t_u, run["correction"][:, 3] / run["hover_thrust"], color=color, label=name)
        if name.startswith("FO, online"):
            axes[2].plot(t_u, run["a_res_pred"][:, 2], color=color, label=f"predicted, {name}")
            axes[3].plot(t_u, run["a_res_rollout"], color=color, label=name)
    target_run = runs[f"FO, online model {args.windows[0]:g} s"]
    for ax in axes[2:]:
        ax.plot(target_run["T"][:-1], target_run["a_res_target"][:, 2], color="0.6", lw=0.6, zorder=0,
                label=f"measured target, {args.windows[0]:g} s run")
    axes[0].set_ylabel("position error [cm]")
    axes[1].set_ylabel("thrust correction / (m_nom g)")
    axes[2].set_ylabel("a_res,z at flown state")
    axes[3].set_ylabel("a_res,z along rollout")
    axes[3].set_xlabel("t [s]")
    for ax in axes:
        ax.axvline(args.switch_time, color="0.5", ls="--", lw=1)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(f"Circle, mass x{args.mass_factor} at t = {args.switch_time:g} s; "
                 f"policy lr = {args.lr:g}, H = {args.horizon}, model steps/tick = {args.model_steps}, "
                 f"thrust noise {args.noise:g} m g")
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
