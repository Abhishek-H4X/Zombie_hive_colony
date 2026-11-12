"""
Watch a trained colony hunt.  Needs trained_colonies.pkl (created by train.py).

    python demo.py                # full hive mind  -> demo_hive.gif
    python demo.py individual     # compare:  random | individual | shared_policy | hive
"""
import sys
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from env import ZombieWorld, SIZE, WALLS, SHELTERS, N_HUMANS
from agents import Colony
from train import run_episode


def main(mode="hive"):
    with open("trained_colonies.pkl", "rb") as f:
        saved = pickle.load(f)[mode]
    colony = Colony(mode, np.random.default_rng(3))
    colony.q = saved["q"]
    colony.belief = saved["belief"].copy()
    colony.epsilon = 0.05 if mode != "random" else 1.0

    world = ZombieWorld(seed=21)
    best_rec, best_c = None, -1
    for _ in range(15):                       # record a few episodes, show the best
        rec = []
        c, _, _ = run_episode(world, colony, learn=False, record=rec)
        if c > best_c:
            best_c, best_rec = c, rec
    rec = best_rec[::2]                       # every 2nd frame keeps the GIF small

    fig, ax = plt.subplots(figsize=(6, 6.4))

    def draw(i):
        zs, hs, alive, targets, belief = rec[i]
        ax.clear()
        if mode == "hive":                    # the leader's belief map = faint red glow
            ax.imshow(belief, cmap="Reds", alpha=0.55, vmin=0, vmax=3,
                      extent=(-0.5, SIZE - 0.5, SIZE - 0.5, -0.5))
        for (r, c) in WALLS:
            ax.add_patch(plt.Rectangle((c - .5, r - .5), 1, 1, color="#444"))
        for (r, c) in SHELTERS:
            ax.add_patch(plt.Rectangle((c - 1.5, r - 1.5), 3, 3, fill=False,
                                       ls="--", ec="#2a7", lw=1))
        for z, t in zip(zs, targets):
            if t is not None:
                ax.plot([z[1], t[1]], [z[0], t[0]], color="#c33", alpha=.25, lw=1)
        for h, a in zip(hs, alive):
            if a:
                ax.scatter(h[1], h[0], s=130, c="#2a7fff", marker="o", zorder=3)
        for z in zs:
            ax.scatter(z[1], z[0], s=170, c="#3a8a2a", marker="X", zorder=4,
                       edgecolors="black")
        ax.set_xlim(-.5, SIZE - .5); ax.set_ylim(SIZE - .5, -.5)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
        caught = N_HUMANS - sum(alive)
        ax.set_title(f"{mode}   |   humans caught: {caught}/{N_HUMANS}", fontsize=12)
        ax.set_xlabel("green X = zombie   blue dot = human   dashed = human shelter"
                      + ("   red glow = hive memory" if mode == "hive" else ""),
                      fontsize=8)

    anim = FuncAnimation(fig, draw, frames=len(rec), interval=120)
    out = f"demo_{mode}.gif"
    anim.save(out, writer=PillowWriter(fps=8))
    print(f"Saved {out}  (caught {best_c}/{N_HUMANS} in the recorded episode)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "hive")
