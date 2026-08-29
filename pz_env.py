"""PettingZoo wrapper around the Frame-Oracle spectrum environment.

Stage A and B used a plain Python loop, deliberately: the mechanics stay
visible while the interesting behaviour is being established. Now that the
behaviour is established, conforming to a standard API buys two things --
other people's algorithms run against this environment without modification,
and PettingZoo's own conformance test checks the implementation is correct
rather than merely running.

ParallelEnv, not AECEnv: every node transmits in the same time slot, so all
agents act simultaneously. AEC models turn-taking games like chess, where
agents act one after another. Choosing the wrong one here would misrepresent
the physics -- radios do not wait their turn.

The underlying environment is unchanged. This is an adapter, and it holds no
logic of its own beyond translating between dict-of-agents and list-of-nodes.
"""

import functools

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

from oracle_env import OracleSpectrumEnv


class SpectrumParallelEnv(ParallelEnv):
    """PettingZoo ParallelEnv over OracleSpectrumEnv."""

    metadata = {"name": "node_parley_spectrum_v0", "render_modes": []}

    def __init__(self, n_channels=4, n_nodes=3, n_mcs=4, episode_len=100,
                 reward_mode="collaborative", seed=None):
        self.possible_agents = [f"node_{i}" for i in range(n_nodes)]
        self.agents = self.possible_agents[:]

        self._env = OracleSpectrumEnv(
            n_channels=n_channels, n_nodes=n_nodes, n_mcs=n_mcs,
            episode_len=episode_len, reward_mode=reward_mode, seed=seed)

        # Box, not MultiBinary. The data is identical -- a vector of zeros
        # and ones -- but RLlib's new API stack has no default encoder for
        # MultiBinary and fails at network construction with
        # "No default encoder config for obs space=MultiBinary(4)".
        # Box(0, 1) gets a normal MLP encoder and means exactly the same thing.
        # The raw environments keep their MultiBinary/int8 form; only this
        # adapter changes, so Stage A/B/B+ results are untouched.
        self._obs_space = spaces.Box(low=0.0, high=1.0,
                                     shape=(n_channels,), dtype=np.float32)
        self._act_space = spaces.Discrete(self._env.n_actions)

    # PettingZoo expects these to be cheap and stable, hence the cache.
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self._obs_space

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self._act_space

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        obs_list = self._env.reset(seed=seed)
        observations = {a: o.astype(np.float32)
                        for a, o in zip(self.agents, obs_list)}
        return observations, {a: {} for a in self.agents}

    def step(self, actions):
        # Dict order is not guaranteed to match agent order, so index rather
        # than iterate over values. Getting this wrong would silently swap
        # which node took which action, and the environment would still run.
        action_list = [int(actions[a]) for a in self.agents]
        obs_list, rewards, done, info = self._env.step(action_list)

        observations = {a: o.astype(np.float32)
                        for a, o in zip(self.agents, obs_list)}
        rewards_d = {a: float(r) for a, r in zip(self.agents, rewards)}
        terminations = {a: False for a in self.agents}
        truncations = {a: bool(done) for a in self.agents}

        infos = {
            a: {"throughput": float(info["per_node_throughput"][i]),
                "channel": int(info["channels"][i]),
                "mcs": int(info["mcs"][i]),
                "received": bool(info["received"][i]),
                "p_success": float(info["p_success"][i])}
            for i, a in enumerate(self.agents)
        }

        # PettingZoo requires the agent list to empty once an episode ends.
        if done:
            self.agents = []

        return observations, rewards_d, terminations, truncations, infos

    def render(self):
        pass

    def close(self):
        pass


def env(**kwargs):
    """Factory, matching PettingZoo convention."""
    return SpectrumParallelEnv(**kwargs)
