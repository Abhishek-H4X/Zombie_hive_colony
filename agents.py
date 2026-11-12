"""
Zombie "brains".

Four colony types, so we can measure WHAT part of the hive mind actually helps:

  random       - no learning at all (baseline)
  individual   - every zombie has its own Q-table and only trusts its own eyes
  shared_policy- every zombie sees only for itself, but they all learn into ONE
                 shared Q-table (the leader merges everyone's experience)
  hive         - shared Q-table  +  the leader broadcasts what any zombie sees
                 +  the leader remembers where humans tend to hide (belief map)
                 and sends idle zombies to the likeliest hiding spots

The learning algorithm is tabular Q-learning (reinforcement learning).
"""
import numpy as np
from env import SIZE, VISION, N_ACTIONS, N_ZOMBIES, path_info

MODES = ["random", "individual", "shared_policy", "hive"]

# state = (target kind 0..2) x (sign of direction to target 3x3) x (first step of shortest path 4) x (wall bits 16)
N_STATES = 3 * 9 * 4 * 16


def _sign(x):
    return (x > 0) - (x < 0)


def encode(kind, zpos, target, walls):
    if target is None:
        sr = sc = hint = 0
    else:
        sr, sc = _sign(target[0] - zpos[0]), _sign(target[1] - zpos[1])
        hint, _ = path_info(zpos, target)
    return ((kind * 9 + (sr + 1) * 3 + (sc + 1)) * 4 + hint) * 16 + walls


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


class Colony:
    def __init__(self, mode, rng, alpha=0.2, gamma=0.95):
        assert mode in MODES
        self.mode = mode
        self.rng = rng
        self.alpha, self.gamma = alpha, gamma
        self.epsilon = 1.0
        if mode == "shared_policy" or mode == "hive":
            self.q = [np.zeros((N_STATES, N_ACTIONS))] * N_ZOMBIES  # same object
        else:
            self.q = [np.zeros((N_STATES, N_ACTIONS)) for _ in range(N_ZOMBIES)]
        # leader's memory of where humans tend to be
        self.belief = np.zeros((SIZE, SIZE))

    # ------------------------------------------------------------------
    def new_episode(self, episode, total_episodes):
        # explore a lot early, a little late
        self.epsilon = max(0.05, 1.0 - 1.6 * episode / total_episodes)
        self.belief *= 0.9          # memory fades a bit between episodes

    # ------------------------------------------------------------------
    def _update_belief(self, world):
        """Leader processes what every zombie's eyes report."""
        for z in world.zombies:
            r0, r1 = max(0, z[0] - VISION), min(SIZE, z[0] + VISION + 1)
            c0, c1 = max(0, z[1] - VISION), min(SIZE, z[1] + VISION + 1)
            self.belief[r0:r1, c0:c1] *= 0.7        # "nobody here" evidence
        for zi, z in enumerate(world.zombies):
            for hi in world.visible_humans(z):
                self.belief[world.humans[hi]] += 1.0  # "human spotted" evidence

    def observe(self, world):
        """Return (state, target) for each zombie, according to this colony's knowledge."""
        if self.mode == "hive":
            self._update_belief(world)
            broadcast = {hi for z in world.zombies for hi in world.visible_humans(z)}

        out = []
        for zi, z in enumerate(world.zombies):
            walls = world.wall_flags(z)
            kind, target = 0, None

            if self.mode == "hive":
                if broadcast:   # anyone's sighting is everyone's sighting
                    t = min(broadcast, key=lambda h: manhattan(z, world.humans[h]))
                    kind, target = 1, world.humans[t]
                else:
                    # send this zombie to the most promising hiding spot (closer = better)
                    idx = np.arange(SIZE)
                    dist = np.abs(idx[:, None] - z[0]) + np.abs(idx[None, :] - z[1])
                    score = self.belief / (1.0 + 0.1 * dist)
                    r, c = np.unravel_index(np.argmax(score), score.shape)
                    if self.belief[r, c] > 0.3 and (r, c) != z:
                        kind, target = 2, (int(r), int(c))
            else:
                vis = world.visible_humans(z)
                if vis:
                    t = min(vis, key=lambda h: manhattan(z, world.humans[h]))
                    kind, target = 1, world.humans[t]

            out.append((encode(kind, z, target, walls), target))
        return out

    # ------------------------------------------------------------------
    def act(self, zi, state):
        if self.mode == "random" or self.rng.random() < self.epsilon:
            return int(self.rng.integers(N_ACTIONS))
        q = self.q[zi][state]
        best = np.flatnonzero(q == q.max())
        return int(self.rng.choice(best))

    def learn(self, zi, s, a, r, s2, done):
        if self.mode == "random":
            return
        q = self.q[zi]
        target = r + (0.0 if done else self.gamma * q[s2].max())
        q[s, a] += self.alpha * (target - q[s, a])
