import numpy as np

from adaptive_rl.data import drop_startup, load_run, save_run, stack_runs


def make_run(n: int, offset: float = 0.0) -> dict:
    return dict(
        X=np.arange(n * 18, dtype=float).reshape(n, 18) + offset,
        U=np.ones((n, 4)),
        X_next=np.zeros((n, 18)),
        E=np.zeros((n, 4)),
        T=np.arange(n) * 0.01,
        meta={"reference": "Circle", "seed": 1},
    )


def test_save_load_roundtrip(tmp_path):
    run = make_run(5)
    save_run(tmp_path / "a" / "run.npz", run)
    loaded = load_run(tmp_path / "a" / "run.npz")
    for key in ("X", "U", "X_next", "E", "T"):
        np.testing.assert_array_equal(loaded[key], run[key])
    assert loaded["meta"] == run["meta"]


def test_drop_startup_and_stack():
    run = drop_startup(make_run(300), t_min=2.0)
    assert run["T"][0] == 2.0 and len(run["T"]) == 100
    stacked = stack_runs([run, make_run(10, offset=1e6)])
    assert stacked["X"].shape == (110, 18)
    assert len(stacked["meta"]) == 2
