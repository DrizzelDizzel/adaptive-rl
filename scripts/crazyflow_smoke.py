"""Smoke test for the Crazyflow setup.

Checks three things:
1. The simulator runs: a single drone is commanded to hover at 0.5 m.
2. The ground-truth dynamics are differentiable: Jacobian of the first-principles
   dynamics (linear acceleration) with respect to the rotor speeds, computed with JAX.
3. Optional: the MuJoCo viewer opens (``--render``).

Usage:
    pixi run smoke              # without visualization
    pixi run smoke-render       # with the MuJoCo viewer
"""

import argparse

# Crazyflow must be imported before scipy: it sets SCIPY_ARRAY_API=1 on import.
from crazyflow.control import Control  # isort: skip
from crazyflow.drones import Drone
from crazyflow.dynamics.core import parametrize
from crazyflow.dynamics.first_principles import dynamics
from crazyflow.sim import Sim

import jax
import jax.numpy as jnp
import numpy as np
from scipy.spatial.transform import Rotation as R


def run_hover(render: bool, duration_s: float = 3.0) -> None:
    sim = Sim(n_worlds=1, n_drones=1, control=Control.state)
    sim.reset()
    cmd = np.zeros((1, 1, 16))
    cmd[..., 2] = 0.5  # target height [m]
    cmd[..., 9:13] = R.from_euler("z", 0.0).as_quat()  # level attitude, yaw 0

    n_control_steps = int(duration_s * sim.control_freq)
    for _ in range(n_control_steps):
        sim.state_control(cmd)
        sim.step(sim.freq // sim.control_freq)
        if render:
            sim.render()

    pos = np.asarray(sim.data.states.pos[0, 0])
    print(f"[hover] position after {duration_s:.1f} s: {np.round(pos, 3)} (target z = 0.5)")
    sim.close()


def check_gradient() -> None:
    dynamics_fn = parametrize(dynamics, Drone.cf2x_L250, xp=jnp)
    pos, quat = jnp.zeros(3), jnp.array([0.0, 0.0, 0.0, 1.0])
    vel, ang_vel = jnp.zeros(3), jnp.zeros(3)

    def vel_dot(cmd):
        # rotor_vel=None: the commanded rotor speeds (RPM) act directly, no motor lag
        _, _, v_dot, _, _ = dynamics_fn(pos=pos, quat=quat, vel=vel, ang_vel=ang_vel, cmd=cmd)
        return v_dot

    # Row by row with jax.grad. Note: jax.jacfwd/jacrev (which use vmap) currently fail inside
    # scipy's Rotation.apply with this JAX version.
    cmd0 = jnp.full(4, 15_000.0)
    jac = jnp.stack([jax.grad(lambda c, i=i: vel_dot(c)[i])(cmd0) for i in range(3)])
    assert jnp.all(jnp.isfinite(jac)), "Jacobian contains non-finite values"
    assert jnp.any(jac != 0), "Jacobian is identically zero"
    print(f"[grad] d(vel_dot)/d(rotor RPM), shape {jac.shape}:\n{np.asarray(jac)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", action="store_true", help="open the MuJoCo viewer")
    args = parser.parse_args()

    print(f"JAX {jax.__version__}, devices: {jax.devices()}")
    check_gradient()
    run_hover(render=args.render)
    print("Crazyflow setup OK.")


if __name__ == "__main__":
    main()
