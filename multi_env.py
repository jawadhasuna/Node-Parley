"""Many agents, unequal channels, and two ways of keeping score.

Stage A had one learner against fixed interference. Here every node is
learning simultaneously, so the environment each one faces is itself changing
-- the defining difficulty of multi-agent reinforcement learning. A policy that
was optimal last episode may be wrong this one, because everyone else moved.

WHY CHANNELS HAVE DIFFERENT VALUES
----------------------------------
If every channel were equally good, avoiding collisions would be enough and
both reward rules below would agree. Real spectrum is not like that: some
channels are wider, cleaner, or less contended. Making channel 0 the best one
creates the actual tension -- everybody wants it, and only one node can have
it. That is where a scoring rule starts to matter.

THE TWO SCORING RULES
---------------------
selfish        A node is paid for its own successful transmission only.
               Collisions hurt, so nodes still learn to avoid each other, but
               nothing discourages fighting over the best channel.

collaborative  Every node is paid the AVERAGE payoff across all nodes. This
               is DARPA's SC2 rule: teams were scored on the whole ensemble's
               throughput, not just their own, so wrecking a neighbour's link
               lowered your score too.

Whether that difference actually produces better aggregate throughput is the
experiment, not an assumption. Selfish agents already dislike collisions, so
it is entirely possible the two converge to the same behaviour.
"""

import numpy as np

REWARD_MODES = ("selfish", "collaborative")


class MultiSpectrumEnv:
    """N unequal channels, M simultaneously-learning nodes.

    Deliberately not a Gymnasium env: the standard single-agent API assumes
    one action and one reward per step. PettingZoo exists for this, and Stage
    B2 wraps it -- but the plain loop here keeps the mechanics visible while
    the interesting behaviour is being established.

    Observation per node: MultiBinary(n_channels), which channels were busy
    last step. Nodes see aggregate occupancy, not who was where -- a radio can
    sense energy, not identity.
    """

    def __init__(self, n_channels=4, n_nodes=3, episode_len=200,
                 reward_mode="collaborative", channel_values=None, seed=None):
        if reward_mode not in REWARD_MODES:
            raise ValueError(f"reward_mode must be one of {REWARD_MODES}")

        self.n_channels = n_channels
        self.n_nodes = n_nodes
        self.episode_len = episode_len
        self.reward_mode = reward_mode

        # Descending quality by default: channel 0 is the prize.
        if channel_values is None:
            channel_values = np.linspace(1.0, 0.4, n_channels)
        self.channel_values = np.asarray(channel_values, dtype=np.float64)

        self._rng = np.random.default_rng(seed)
        self._step = 0
        self._occupancy = np.zeros(n_channels, dtype=np.int8)

    def reset(self, seed=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._step = 0
        # Start from a random occupancy so nodes cannot memorise one opening.
        self._occupancy = np.zeros(self.n_channels, dtype=np.int8)
        start = self._rng.integers(0, self.n_channels, self.n_nodes)
        for c in start:
            self._occupancy[c] = 1
        return [self._occupancy.copy() for _ in range(self.n_nodes)]

    def step(self, actions):
        """Advance one step.

        Args:
            actions: one channel index per node.

        Returns:
            (observations, rewards, done, info)
        """
        actions = np.asarray(actions)
        counts = np.bincount(actions, minlength=self.n_channels)

        # A transmission succeeds only if it is alone on its channel.
        alone = counts[actions] == 1
        payoff = np.where(alone, self.channel_values[actions], -1.0)

        if self.reward_mode == "selfish":
            rewards = payoff.copy()
        else:
            # Everyone is paid the ensemble average -- the SC2 rule.
            rewards = np.full(self.n_nodes, payoff.mean())

        occ = np.zeros(self.n_channels, dtype=np.int8)
        occ[actions] = 1
        self._occupancy = occ
        self._step += 1

        info = {
            "collisions": int((counts > 1).sum()),
            "n_successful": int(alone.sum()),
            # Total value actually delivered: the number to compare rules on,
            # since it is what SC2 scored and it is independent of how the
            # reward was shared out.
            "throughput": float(np.where(alone,
                                         self.channel_values[actions],
                                         0.0).sum()),
            "per_node_success": alone.astype(float),
            "actions": actions.copy(),
        }
        done = self._step >= self.episode_len
        return [occ.copy() for _ in range(self.n_nodes)], rewards, done, info

    # -- reference points ------------------------------------------------------
    def best_throughput(self) -> float:
        """Highest achievable: nodes on the best distinct channels, no clashes."""
        k = min(self.n_nodes, self.n_channels)
        return float(np.sort(self.channel_values)[::-1][:k].sum())

    def random_throughput(self, trials=20_000, seed=0) -> float:
        """Expected throughput when every node chooses uniformly at random."""
        rng = np.random.default_rng(seed)
        acts = rng.integers(0, self.n_channels, (trials, self.n_nodes))
        total = 0.0
        for row in acts:
            counts = np.bincount(row, minlength=self.n_channels)
            alone = counts[row] == 1
            total += np.where(alone, self.channel_values[row], 0.0).sum()
        return total / trials

    def greedy_throughput(self) -> float:
        """If every node always grabs the single best channel.

        The failure mode worth naming: all nodes pile onto channel 0, all
        collide, and throughput is zero. Any learning rule must beat this, and
        the fact that it is so easy to fall into is the reason the scoring
        rule matters at all.
        """
        return 0.0 if self.n_nodes > 1 else float(self.channel_values.max())


def fairness(per_node_success_rates) -> float:
    """Jain's fairness index: 1.0 is perfectly equal, 1/n is one node taking all.

    Reported alongside throughput because they can disagree. A configuration
    where one node holds the best channel forever may be efficient and deeply
    unfair, and an ensemble scoring rule is supposed to care about both.
    """
    x = np.asarray(per_node_success_rates, dtype=np.float64)
    if np.allclose(x, 0):
        return 1.0
    return float(x.sum() ** 2 / (len(x) * (x**2).sum()))
