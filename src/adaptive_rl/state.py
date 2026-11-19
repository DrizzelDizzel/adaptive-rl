"""Layout of the state x and the command u: the contract every module shares.

Plant, controllers and (later) the dynamics models all read and write these vectors. The layout
is defined here and nowhere else, so that none of them has to import another just to know it
(docs/decisions.md #10). Importing this must not load the simulator (JAX is fine).

State vector x, shape (18,), float64:
    x[0:3]    p      position of the CoM, world frame [m]
    x[3:12]   R      rotation matrix body -> world, flattened row-major (R[0,0], R[0,1], ...)
    x[12:15]  v      velocity of the CoM, world frame [m/s]
    x[15:18]  omega  angular velocity, BODY frame [rad/s]

Command u, shape (4,):
    [omega_x, omega_y, omega_z, thrust]  body rates [rad/s] (body frame), collective thrust [N]
    Crazyflow's internal rate controller tracks omega; it also saturates the motors, so the
    thrust/torque actually applied can differ from u for large commands.

World frame: z points up, gravity acts along -z (constants.GRAVITY).
"""

import jax
import jax.numpy as jnp
import numpy as np

STATE_DIM = 18
CMD_DIM = 4


def unpack_state(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split the flat state into (p, R, v, omega). R is returned as a 3x3 matrix.

    Only slicing and reshape: works for numpy and JAX arrays alike.
    """
    return x[0:3], x[3:12].reshape(3, 3), x[12:15], x[15:18]


def pack_state(p, R, v, omega) -> jax.Array:
    """Inverse of unpack_state: (p, R 3x3, v, omega) -> flat x, shape (18,).

    jnp instead of np: inside jit/grad the inputs are tracers (symbolic placeholders), which
    np.concatenate cannot handle. numpy inputs work too and give a JAX array.
    reshape is row-major (C order), the same as R.ravel() in plant.py.
    """
    return jnp.concatenate([p, R.reshape(9), v, omega])
