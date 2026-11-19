"""Physical constants shared by all modules (docs/decisions.md #10)."""

# [m/s^2], acts along -z of the world frame. Same value as Crazyflow
# (crazyflow/dynamics/params.toml: gravity_vec = [0, 0, -9.81]), so the nominal model and the
# simulator agree exactly and the residual does not have to learn a constant offset.
GRAVITY = 9.81
