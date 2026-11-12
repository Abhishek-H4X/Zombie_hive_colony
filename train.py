"""
Train all four colony types and compare them.

    python train.py                 # full experiment (about 3-6 min)
    python train.py --quick         # fast sanity run
"""
import argparse
import pickle
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from env import ZombieWorld, N_HUMANS, MAX_STEPS, dist_field
from agents import Colony, MODES, manhattan

COLORS = {"random": "#888888", "individual": "#e07b39",
          "shared_policy": "#3a86c8", "hive": "#2a9d4f"}
LABELS = {"random": "Random (no learning)", "individual": "Individual learning",
          "shared_policy": "Shared brain only", "hive": "Full hive mind"}

R_CATCH, R_STEP, R_BUMP, R_CLOSER, R_FARTHER, R_WAIT = 10.0, -0.05, -0.2, 0.5, -0.5, -0.3


def run_episode(world, colony, learn=True, record=None):
    """Plays one episode. Returns (humans caught, total reward, steps)."""
    world.reset()
    obs = colony.observe(world)
    total, done = 0.0, False
    while not done:
        if record is not None:
            record.append((list(world.zombies), list(world.humans), list(world.alive),
                           [t for _, t in obs], colony.belief.copy()))
        states = [s for s, _ in obs]
        before = [list(z) for z in world.zombies]
        actions = [colony.act(zi, states[zi]) for zi in range(len(world.zombies))]
        caught, bumped, done = world.step(actions)
        new_obs = colony.observe(world)

        for zi in range(len(world.zombies)):
            r = R_STEP + R_CATCH * len(caught[zi])
            if bumped[zi]:
                r += R_BUMP
            if actions[zi] == 4:
                r += R_WAIT   # standing still is never useful for a hunter
            tgt = obs[zi][1]
            if tgt is not None and not caught[zi]:
                f = dist_field(tgt)
                d0 = f[tuple(before[zi])]
                d1 = f[world.zombies[zi]]
                r += R_CLOSER if d1 < d0 else R_FARTHER if d1 > d0 else 0.0
            total += r
            if learn:
                colony.learn(zi, states[zi], actions[zi], r, new_obs[zi][0], done)
        obs = new_obs

    if record is not None:
        record.append((list(world.zombies), list(world.humans), list(world.alive),
                       [None] * len(world.zombies), colony.belief.copy()))
    caught_n = N_HUMANS - sum(world.alive)
    return caught_n, total, world.t


def train(mode, episodes, seed):
    rng = np.random.default_rng(seed)
    world = ZombieWorld(seed=seed + 1000)
    colony = Colony(mode, rng)
    caught = np.zeros(episodes)
    for ep in range(episodes):
        colony.new_episode(ep, episodes)
        caught[ep], _, _ = run_episode(world, colony, learn=True)
    return colony, caught


def smooth(x, w=50):
    k = np.ones(w) / w
    return np.convolve(x, k, mode="valid")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=1500)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.episodes, args.seeds = 300, 1

    results = {m: [] for m in MODES}
    best = {}
    t0 = time.time()
    for mode in MODES:
        for s in range(args.seeds):
            colony, caught = train(mode, args.episodes, seed=s)
            results[mode].append(caught)
            best[mode] = colony  # keep last seed's colony for the demo
            print(f"{mode:14s} seed {s}: last-100 avg catches = "
                  f"{caught[-100:].mean():.2f}/{N_HUMANS}   ({time.time()-t0:.0f}s)")

    # ---- learning-curve plot -------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for mode in MODES:
        arr = np.array(results[mode])                       # seeds x episodes
        curves = np.array([smooth(a) for a in arr])
        m = curves.mean(0)
        x = np.arange(len(m)) + 50
        ax.plot(x, m, color=COLORS[mode], lw=2.2, label=LABELS[mode])
        if len(curves) > 1:
            ax.fill_between(x, curves.min(0), curves.max(0), color=COLORS[mode], alpha=0.15)
    ax.set_xlabel("Training episode")
    ax.set_ylabel(f"Humans caught per episode (out of {N_HUMANS})")
    ax.set_title("Do zombies learn faster when they share a hive mind?")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig("learning_curves.png", dpi=150)

    # ---- summary table ---------------------------------------------------
    print("\nSummary (mean over seeds)")
    print(f"  {'colony':24s} {'final skill':>12s}   {'episodes to reach 2.0 catches':>30s}")
    for mode in MODES:
        arr = np.array(results[mode])
        final = arr[:, -100:].mean()
        reach = []
        for a in arr:
            idx = np.flatnonzero(smooth(a) >= 2.0)
            reach.append(idx[0] + 50 if len(idx) else np.nan)
        reach_s = "never" if np.all(np.isnan(reach)) else f"{np.nanmean(reach):.0f}"
        print(f"  {LABELS[mode]:24s} {final:9.2f}/{N_HUMANS}   {reach_s:>30s}")

    with open("trained_colonies.pkl", "wb") as f:
        pickle.dump({m: {"q": best[m].q, "belief": best[m].belief} for m in MODES}, f)
    print("\nSaved learning_curves.png and trained_colonies.pkl")


if __name__ == "__main__":
    main()
