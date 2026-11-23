"""Replay recorded runs as an animated GIF: reference vs every controller, one color each (#18).

Reads the .npz that run_adaptive.py writes next to its figure, so rendering never touches the
control loop. Colors match run_adaptive.py (matplotlib's C0, C1, ... in run order). The black
star is the reference point at that time; each drone leaves a 1 s trail. A drone whose run
was aborted (crash) stays where it ended, drawn as an x.

matplotlib.animation in two sentences: FuncAnimation calls `draw(frame)` once per frame, which
only moves the existing markers and lines (set_data_3d); PillowWriter grabs each drawn figure
as an image and stacks them into a GIF (Pillow is already installed, ffmpeg is not).

Usage:
    pixi run python scripts/render_runs.py results/figures/run_adaptive.npz   # -> run_adaptive.gif
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

FPS = 20  # frames per second of the GIF; with SPEED = 1 one frame = 0.05 s of flight
SPEED = 1.0  # playback speed relative to real time
TRAIL = 1.0  # [s] length of the trail behind each drone


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", help=".npz written by run_adaptive.py")
    parser.add_argument("--out", help="output GIF (default: same name as the .npz)")
    args = parser.parse_args()

    data = np.load(args.runs)
    names = list(data["names"])
    t_ref, p_ref = data["t_ref"], data["p_ref"]
    switch_time, mass_factor = float(data["switch_time"]), float(data["mass_factor"])
    runs = [(data[f"t{i}"], data[f"p{i}"]) for i in range(len(names))]
    dt = t_ref[1] - t_ref[0]
    step = max(1, round(SPEED / (FPS * dt)))  # control steps per frame
    trail = round(TRAIL / dt)
    frames = range(0, len(t_ref), step)

    fig = plt.figure(figsize=(8, 6.5))
    ax = fig.add_subplot(projection="3d")
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.02, top=1.0)  # 3D axes leave wide margins otherwise
    ax.plot(*p_ref.T, color="0.6", ls="--", lw=1, label="reference")
    ref_marker, = ax.plot([], [], [], "k*", ms=12)
    markers, trails = [], []
    for i, name in enumerate(names):
        trails.append(ax.plot([], [], [], color=f"C{i}", lw=1.5, alpha=0.7)[0])
        markers.append(ax.plot([], [], [], "o", color=f"C{i}", ms=8, label=name)[0])
    clock = ax.text2D(0.02, 0.95, "", transform=ax.transAxes, fontsize=11)

    # Fixed limits: the drones separate mainly in height after the mass switch, so z spans
    # from the floor to above the circle; the camera looks slightly from the side.
    center = p_ref.mean(axis=0)
    ax.set_xlim(center[0] - 0.7, center[0] + 0.7)
    ax.set_ylim(center[1] - 0.7, center[1] + 0.7)
    ax.set_zlim(0.0, center[2] + 0.3)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.view_init(elev=15, azim=-60)
    ax.legend(loc="upper right", fontsize=8)

    def draw(k: int):
        ref_marker.set_data_3d([p_ref[k, 0]], [p_ref[k, 1]], [p_ref[k, 2]])
        for (t, p), marker, line in zip(runs, markers, trails):
            j = min(k, len(t) - 1)  # aborted runs stay at their last position
            marker.set_data_3d([p[j, 0]], [p[j, 1]], [p[j, 2]])
            marker.set_marker("x" if k >= len(t) else "o")
            seg = p[max(0, j - trail) : j + 1]
            line.set_data_3d(seg[:, 0], seg[:, 1], seg[:, 2])
        mass = "nominal mass" if t_ref[k] < switch_time else f"mass x{mass_factor:g}"
        clock.set_text(f"t = {t_ref[k]:5.2f} s   ({mass})")
        return [ref_marker, clock, *markers, *trails]

    out = Path(args.out) if args.out else Path(args.runs).with_suffix(".gif")
    FuncAnimation(fig, draw, frames=frames).save(out, writer=PillowWriter(fps=FPS), dpi=100)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
