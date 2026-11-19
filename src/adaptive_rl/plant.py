"""The "real" drone: a thin wrapper around Crazyflow.

The rest of the project only talks to this class through two calls:

    x = plant.reset()        # once at the start
    x_next = plant.step(u)   # once per control step

Nothing outside this file imports crazyflow. All Crazyflow specifics (batch dimensions
(n_worlds, n_drones, ...), xyzw quaternions, import order, controller gains) stay here.
"""

# Crazyflow must be imported before scipy: it sets SCIPY_ARRAY_API=1 on import.
from crazyflow.control import Control  # isort: skip
from crazyflow.sim import Sim  # isort: skip

import jax.numpy as jnp
import numpy as np
from scipy.spatial.transform import Rotation

# Layout of x and u (frames, units): see state.py.
from adaptive_rl.state import CMD_DIM


class Plant:
    def __init__(
        self,
        control_freq: int = 100,
        sim_freq: int = 500,
        p0: tuple[float, float, float] = (0.0, 0.0, 1.0),
    ):
        """Create the simulated drone.

        Args:
            control_freq: Rate [Hz] at which step(u) is called. u is held constant in between
                (zero-order hold).
            sim_freq: Physics rate [Hz]. Crazyflow's inner body-rate controller also runs at
                this rate. Must be a multiple of control_freq.
            p0: Initial position [m]. Start in the air, otherwise ground contact forces enter
                the data.
        """
        if sim_freq % control_freq != 0:
            raise ValueError(f"sim_freq ({sim_freq}) must be a multiple of control_freq ({control_freq})")
        self.control_freq = control_freq
        self.dt = 1.0 / control_freq
        self._n_substeps = sim_freq // control_freq

        # One world, one drone (just as for a real drone). body_rate_freq = sim_freq: the inner rate loop runs as fast as
        # the physics, independent of how often we send new commands.
        self._sim = Sim(
            n_worlds=1,
            n_drones=1,
            control=Control.body_rate,
            freq=sim_freq,
            body_rate_freq=sim_freq,
        )

        # Crazyflow's body-rate controller (firmware Mellinger) also contains attitude terms
        # (kR, ki_m) that pull the drone back to level flight and fight the commanded rates.
        # Zero them so the inner loop is a pure rate tracker.
        data = self._sim.data
        body_rate = data.controls.body_rate
        params = body_rate.params | {"kR": jnp.zeros(3), "ki_m": jnp.zeros(3)}
        data = data.replace(controls=data.controls.replace(body_rate=body_rate.replace(params=params)))

        # Initial position in the air.
        pos = data.states.pos.at[0, 0].set(jnp.asarray(p0))
        data = data.replace(states=data.states.replace(pos=pos))
        self._sim.data = data

        # sim.reset() copies `default_data` into `data`. Store the modified data as the new
        # default, otherwise reset() would undo both changes above.
        self._sim.build_default_data()

        # Mass as known from the spec sheet. Fixed here on purpose: after set_disturbance()
        # the true mass differs, and handing that out would be an oracle for controller/model.
        self.nominal_mass = float(np.asarray(data.params.mass).ravel()[0])

    def reset(self) -> np.ndarray:
        """Back to the initial state. Also removes any disturbance."""
        self._sim.reset()
        return self._read_state()

    def set_disturbance(
        self,
        mass: float | None = None,
        force: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        """Change the true dynamics; stays active until the next call or reset().

        Args:
            mass: True mass [kg], e.g. with payload. None = nominal mass. Inertia is not scaled.
            force: Constant external force on the CoM, world frame [N], e.g. wind.
        """
        data = self._sim.data
        true_mass = self.nominal_mass if mass is None else mass
        params = data.params.replace(mass=jnp.full_like(data.params.mass, true_mass))
        f = data.states.force.at[0, 0].set(jnp.asarray(force))
        self._sim.data = data.replace(params=params, states=data.states.replace(force=f))

    def step(self, u: np.ndarray) -> np.ndarray:
        u = np.asarray(u, dtype=np.float32)
        if u.shape != (CMD_DIM,):
            raise ValueError(f"u must have shape ({CMD_DIM},), got {u.shape}")
        self._sim.body_rate_control(u.reshape(1, 1, CMD_DIM))
        self._sim.step(self._n_substeps)
        return self._read_state()

    def render(self) -> None:
        self._sim.render()

    def close(self) -> None:
        self._sim.close()

    def _read_state(self) -> np.ndarray:
        s = self._sim.data.states
        # [0, 0] = world 0, drone 0. np.asarray copies from the JAX device buffer, so the
        # returned array is independent of the simulator's internal state.
        p = np.asarray(s.pos[0, 0], dtype=np.float64)
        quat_xyzw = np.asarray(s.quat[0, 0], dtype=np.float64)
        v = np.asarray(s.vel[0, 0], dtype=np.float64)
        omega_body = np.asarray(s.ang_vel[0, 0], dtype=np.float64)
        R = Rotation.from_quat(quat_xyzw).as_matrix()  # scipy uses xyzw, same as Crazyflow
        return np.concatenate([p, R.ravel(), v, omega_body])
