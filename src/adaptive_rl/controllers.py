"""Baseline controllers. Interface shared with every policy:

    u = controller(x, ref)

x:   flat state from plant.py (p, R, v, omega)
ref: (p_ref, v_ref, a_ref) in the world frame, from reference.py
u:   [omega_x, omega_y, omega_z, thrust] (body rates [rad/s], collective thrust [N])
"""

import numpy as np

from adaptive_rl.plant import unpack_state

GRAVITY = 9.81
E3 = np.array([0.0, 0.0, 1.0])


def vee(S: np.ndarray) -> np.ndarray:
    """Inverse of the hat map: skew-symmetric 3x3 matrix -> vector."""
    return np.array([S[2, 1], S[0, 2], S[1, 0]])


class CascadedController:
    """Position -> attitude -> body rates. The innermost loop (rates -> torques) is Crazyflow's.

    Outer loop (translation): PD on position/velocity plus feedforward acceleration gives the
    force vector the drone must produce in the world frame.
    Middle loop (attitude): rotate the thrust axis (body z) onto that force vector, yaw = 0,
    and command body rates proportional to the attitude error (geometric control on SO(3),
    no Euler angles -> no singularities).

    Stateless: no integrator. Consequence: constant disturbances (drag, wrong mass, external
    force) leave a steady-state tracking error. Acceptable for data collection.
    """

    def __init__(self, mass: float, kp: float = 6.0, kd: float = 4.0, k_att: float = 8.0):
        self.mass = mass  # nominal mass [kg]
        self.kp = kp  # [1/s^2]
        self.kd = kd  # [1/s]
        self.k_att = k_att  # [1/s] attitude error -> body rate

    def __call__(self, x: np.ndarray, ref: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
        p, R, v, _ = unpack_state(x)
        p_ref, v_ref, a_ref = ref

        # 1) Desired force in world frame [N]
        a_des = a_ref + self.kp * (p_ref - p) + self.kd * (v_ref - v)
        f_des = self.mass * (a_des + GRAVITY * E3)

        # 2) Collective thrust: only the component along the current body z axis is useful
        b3 = R[:, 2]  # body z axis expressed in world frame
        thrust = float(f_des @ b3)

        # 3) Desired attitude: body z along f_des, body x as close to world x as possible (yaw 0)
        b3_des = f_des / np.linalg.norm(f_des)
        b2_des = np.cross(b3_des, np.array([1.0, 0.0, 0.0]))
        b2_des /= np.linalg.norm(b2_des)
        b1_des = np.cross(b2_des, b3_des)
        R_des = np.column_stack([b1_des, b2_des, b3_des])

        # 4) Attitude error (body frame) -> body-rate command
        e_R = 0.5 * vee(R_des.T @ R - R.T @ R_des)
        omega_cmd = -self.k_att * e_R

        return np.concatenate([omega_cmd, [thrust]])
