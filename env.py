"""
Grid world for the "Colony" zombie hive-mind project.

Humans hide in shelters (their "tactic") and flee when a zombie gets close.
Zombies are the learning agents. The environment only produces raw facts
(positions, walls, catches); what the zombies *know* is decided by agents.py.
"""
import numpy as np
from collections import deque

# actions: 0=up 1=down 2=left 3=right 4=wait
MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]
N_ACTIONS = len(MOVES)

SIZE = 15
N_ZOMBIES = 4
N_HUMANS = 5
VISION = 4          # how far (Chebyshev) a zombie can see
MAX_STEPS = 150

# Fixed wall layout (row, col). Gives the map some buildings.
WALLS = set()
for r in range(3, 7):
    WALLS.add((r, 3)); WALLS.add((r, 4))
for c in range(8, 12):
    WALLS.add((10, c)); WALLS.add((11, c))
for r in range(2, 5):
    WALLS.add((r, 10))
for c in range(6, 9):
    WALLS.add((6, c))

# Shelters: the places humans like to hide. Zombies must DISCOVER these.
SHELTERS = [(2, 12), (12, 2), (7, 7)]


class ZombieWorld:
    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)
        self.reset()

    # ------------------------------------------------------------------
    def free(self, pos):
        r, c = pos
        return 0 <= r < SIZE and 0 <= c < SIZE and pos not in WALLS

    def _random_free_near(self, center, radius=2):
        while True:
            r = center[0] + self.rng.integers(-radius, radius + 1)
            c = center[1] + self.rng.integers(-radius, radius + 1)
            if self.free((r, c)):
                return (int(r), int(c))

    def reset(self):
        self.t = 0
        # Humans: each picks a shelter (mostly sticks to the same ones)
        self.humans = []
        for _ in range(N_HUMANS):
            s = SHELTERS[self.rng.choice(len(SHELTERS), p=[0.4, 0.4, 0.2])]
            self.humans.append(self._random_free_near(s, 1))
        self.alive = [True] * N_HUMANS
        # Zombies spawn along the left/top edge, away from shelters
        self.zombies = []
        while len(self.zombies) < N_ZOMBIES:
            p = (int(self.rng.integers(0, SIZE)), 0)
            if self.free(p) and p not in self.zombies:
                self.zombies.append(p)
        return self

    # ------------------------------------------------------------------
    def visible_humans(self, zpos):
        """Indices of living humans this zombie can see."""
        out = []
        for i, h in enumerate(self.humans):
            if self.alive[i] and max(abs(h[0] - zpos[0]), abs(h[1] - zpos[1])) <= VISION:
                out.append(i)
        return out

    def wall_flags(self, pos):
        """4 bits: is there a wall / edge up, down, left, right?"""
        f = 0
        for k, (dr, dc) in enumerate(MOVES[:4]):
            if not self.free((pos[0] + dr, pos[1] + dc)):
                f |= 1 << k
        return f

    # ------------------------------------------------------------------
    def _move_humans(self):
        """Human tactic: stay hidden, but flee if a zombie gets within 2 cells."""
        for i, h in enumerate(self.humans):
            if not self.alive[i]:
                continue
            nearest = min(self.zombies, key=lambda z: abs(z[0] - h[0]) + abs(z[1] - h[1]))
            d = abs(nearest[0] - h[0]) + abs(nearest[1] - h[1])
            if d <= 2:
                best, best_d = h, d
                for dr, dc in MOVES[:4]:
                    n = (h[0] + dr, h[1] + dc)
                    if self.free(n) and n not in self.humans:
                        nd = abs(nearest[0] - n[0]) + abs(nearest[1] - n[1])
                        if nd > best_d:
                            best, best_d = n, nd
                self.humans[i] = best

    def step(self, actions):
        """
        actions: list of ints, one per zombie.
        returns: caught (list of human indices caught this step, per zombie),
                 bumped (list of bool per zombie), done
        """
        self.t += 1
        bumped = []
        for zi, a in enumerate(actions):
            dr, dc = MOVES[a]
            n = (self.zombies[zi][0] + dr, self.zombies[zi][1] + dc)
            if a != 4 and not self.free(n):
                bumped.append(True)
            else:
                bumped.append(False)
                self.zombies[zi] = n
        self._move_humans()

        caught = [[] for _ in self.zombies]
        for zi, z in enumerate(self.zombies):
            for hi, h in enumerate(self.humans):
                if self.alive[hi] and h == z:
                    self.alive[hi] = False
                    caught[zi].append(hi)
        done = (not any(self.alive)) or self.t >= MAX_STEPS
        return caught, bumped, done


# ----------------------------------------------------------------------
# Path-finding helpers (walls never move, so distance fields can be cached)
_FIELD_CACHE = {}


def dist_field(target):
    """BFS walking distance from every cell to `target` (inf if unreachable)."""
    if target in _FIELD_CACHE:
        return _FIELD_CACHE[target]
    f = np.full((SIZE, SIZE), np.inf)
    f[target] = 0
    q = deque([target])
    while q:
        r, c = q.popleft()
        for dr, dc in MOVES[:4]:
            n = (r + dr, c + dc)
            if 0 <= n[0] < SIZE and 0 <= n[1] < SIZE and n not in WALLS and f[n] == np.inf:
                f[n] = f[r, c] + 1
                q.append(n)
    _FIELD_CACHE[target] = f
    return f


def path_info(pos, target):
    """(first step direction 0-3 along shortest path, walking distance)."""
    f = dist_field(target)
    d = f[pos]
    if d == 0 or d == np.inf:
        return 0, d
    best, bd = 0, np.inf
    for k, (dr, dc) in enumerate(MOVES[:4]):
        n = (pos[0] + dr, pos[1] + dc)
        if 0 <= n[0] < SIZE and 0 <= n[1] < SIZE and f[n] < bd:
            best, bd = k, f[n]
    return best, d
