# Colony Hive Mind: do zombies learn faster when they share knowledge?

Inspired by the Korean movie *Colony*: a zombie leader collects what the zombies learn about humans and shares it back.
This project tests that idea with multi-agent reinforcement learning (tabular Q-learning) in a grid world.

## The world
- 15x15 grid with walls (buildings).
- 5 humans hide near 3 shelters (their "tactic") and run away when a zombie gets within 2 cells.
- 4 zombies spawn on the left edge and can see 4 cells around them.
- Reward: +10 catch, +0.5 / -0.5 for moving closer / farther from a target, small penalties for steps, wall bumps and waiting.

## Four colonies compared
| Colony | What it has |
|---|---|
| Random | Nothing learned (baseline) |
| Individual | Each zombie has its own brain and trusts only its own eyes |
| Shared brain only | All zombies learn into ONE Q-table, but see only for themselves |
| Full hive mind | Shared brain + leader broadcasts every sighting to everyone + leader remembers where humans tend to hide and sends idle zombies there |

## Run it (Windows, Python 3.9+)
```
pip install -r requirements.txt
python train.py --quick      # ~30 s sanity run
python train.py              # full experiment (~5 min) -> learning_curves.png, trained_colonies.pkl
python demo.py hive          # animation -> demo_hive.gif
python demo.py individual    # compare other colonies: random | individual | shared_policy | hive
```

## Results (3 seeds, 1200 episodes)
| Colony | Humans caught (last 100 episodes) | Episodes to reach 2.0 catches |
|---|---|---|
| Random | 0.23 / 5 | never |
| Individual | 3.48 / 5 | 290 |
| Shared brain only | 2.83 / 5 | 274 |
| Full hive mind | 3.77 / 5 | 212 |

Honest reading:
- All learners beat random by a wide margin, so this is real learning, not hand-written rules.
- The full hive mind learns fastest and peaks highest (about 4.3 catches around episode 600).
- Late in training the hive and individual curves overlap and are noisy; the final gap (3.77 vs 3.48) is modest.
- Sharing the brain alone did NOT help (it was worse than individual). The gain comes from sharing *perception* (broadcast sightings + hiding-spot memory), not just sharing the Q-table. Likely reason: identical brains make the zombies behave too alike.
- Results come from 3 seeds with a simple tabular method; more seeds would tighten the conclusion.

## Ideas to extend
- Humans that also learn (arms race), so shelters change over time.
- Replace the Q-table with a neural network (PPO / DQN via Stable-Baselines3 + PettingZoo).
- Give zombies short-term memory or role assignment to reduce clumping.
- More seeds and statistical tests.

## Files
`env.py` world | `agents.py` colonies and Q-learning | `train.py` experiment and plots | `demo.py` GIF renderer
