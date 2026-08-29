"""Does the PettingZoo wrapper actually conform, and does it change anything?

Two questions, both worth asking separately:

  1. Is it a valid ParallelEnv? PettingZoo ships a conformance test that
     checks the API contract -- space shapes, dict keys, agent lifecycle, the
     things that are easy to get subtly wrong and that still run.

  2. Does it behave identically to the raw environment? An adapter that
     quietly changes behaviour is worse than no adapter, because every result
     produced through it silently disagrees with the results produced without
     it. Same seed, same actions, same outcomes -- or the wrapper is wrong.

Run:  uv run check_pz_env.py
"""

import numpy as np

from oracle_env import OracleSpectrumEnv
from pz_env import SpectrumParallelEnv

N_CHANNELS, N_NODES, N_MCS = 4, 3, 4
EPISODE_LEN = 50

# --- 1. conformance -----------------------------------------------------------
print("PettingZoo parallel_api_test...")
from pettingzoo.test import parallel_api_test

test_env = SpectrumParallelEnv(N_CHANNELS, N_NODES, N_MCS, EPISODE_LEN,
                               seed=0)
parallel_api_test(test_env, num_cycles=200)
print("  passed\n")

# --- 2. behavioural equivalence ----------------------------------------------
# Drive both with the identical action sequence from the same seed. Any
# divergence means the adapter is doing something of its own.
SEED = 7
rng = np.random.default_rng(123)
n_actions = N_CHANNELS * N_MCS
scripted = rng.integers(0, n_actions, (EPISODE_LEN, N_NODES))

raw = OracleSpectrumEnv(N_CHANNELS, N_NODES, N_MCS, EPISODE_LEN,
                        reward_mode="collaborative", seed=SEED)
raw.reset(seed=SEED)
raw_rewards, raw_thr = [], []
for t in range(EPISODE_LEN):
    _, r, done, info = raw.step(scripted[t])
    raw_rewards.append(np.asarray(r, dtype=float))
    raw_thr.append(info["throughput"])
    if done:
        break

pz = SpectrumParallelEnv(N_CHANNELS, N_NODES, N_MCS, EPISODE_LEN,
                         reward_mode="collaborative", seed=SEED)
pz.reset(seed=SEED)
pz_rewards, pz_thr = [], []
for t in range(EPISODE_LEN):
    acts = {a: int(scripted[t][i]) for i, a in enumerate(pz.agents)}
    _, r, _, trunc, infos = pz.step(acts)
    pz_rewards.append(np.array([r[a] for a in sorted(r)], dtype=float))
    pz_thr.append(sum(infos[a]["throughput"] for a in infos))
    if any(trunc.values()):
        break

raw_rewards = np.array(raw_rewards)
pz_rewards = np.array(pz_rewards)

print(f"steps compared        : {len(raw_rewards)}")
print(f"max reward difference : "
      f"{np.abs(raw_rewards - pz_rewards).max():.3e}")
print(f"total throughput      : raw {sum(raw_thr):.3f}  "
      f"wrapped {sum(pz_thr):.3f}")

assert raw_rewards.shape == pz_rewards.shape
assert np.allclose(raw_rewards, pz_rewards), \
    "the wrapper changes rewards -- it is not a transparent adapter"
assert np.allclose(sum(raw_thr), sum(pz_thr))
print("  identical: the wrapper is transparent\n")

# --- 3. the ordering trap -----------------------------------------------------
# Actions arrive as a dict. Python dicts preserve insertion order, but nothing
# guarantees a caller builds one in agent order -- RLlib certainly does not
# promise to. If the wrapper iterated over values instead of indexing by agent
# name, node 2's action could be applied to node 0 and the environment would
# still run happily, just wrong.
pz2 = SpectrumParallelEnv(N_CHANNELS, N_NODES, N_MCS, EPISODE_LEN,
                          reward_mode="selfish", seed=SEED)
pz2.reset(seed=SEED)
forward = {a: int(scripted[0][i]) for i, a in enumerate(pz2.agents)}
_, _, _, _, info_fwd = pz2.step(forward)
channels_fwd = [info_fwd[a]["channel"] for a in pz2.agents]

pz3 = SpectrumParallelEnv(N_CHANNELS, N_NODES, N_MCS, EPISODE_LEN,
                          reward_mode="selfish", seed=SEED)
pz3.reset(seed=SEED)
reversed_dict = {a: int(scripted[0][i])
                 for i, a in reversed(list(enumerate(pz3.agents)))}
_, _, _, _, info_rev = pz3.step(reversed_dict)
channels_rev = [info_rev[a]["channel"] for a in pz3.agents]

print(f"actions given in agent order   -> channels {channels_fwd}")
print(f"same actions, dict built backwards -> channels {channels_rev}")
assert channels_fwd == channels_rev, \
    "dict ordering changes the outcome -- the wrapper indexes wrongly"
print("  dict ordering does not matter: actions are indexed by agent name\n")

print("all checks passed")
