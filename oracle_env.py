"""Multi-agent spectrum access with Frame-Oracle deciding outcomes.

The difference from multi_env.py is the action. There, a node chose a channel
and succeeded if it was alone. Here it chooses a channel AND a modulation and
coding scheme, and whether the frame lands is predicted by the model trained
in Frame-Oracle on real Colosseum measurements.

That is the actual SC2 decision. It also closes a loop through the project:
Wave-Lathe measured what higher-order modulation costs (QPSK +3.01 dB over
BPSK, 16-QAM +10.00 dB); Frame-Oracle learned from real frames when
transmissions survive; and here an agent has to live with both.

The tension is real in both directions:
  aggressive MCS  more bits when it lands, lands less often
  good channel    everyone wants it, and sharing ruins it for both

See channel_model.py for what is measured here and what is invented. The short
version: the predictor is real, the mapping of this synthetic world into its
feature space is not.
"""

import numpy as np

from channel_model import FrameOracleChannel

REWARD_MODES = ("selfish", "collaborative")


class OracleSpectrumEnv:
    """N channels x M MCS levels, several nodes, outcomes from Frame-Oracle."""

    def __init__(self, n_channels=4, n_nodes=3, n_mcs=4, episode_len=100,
                 reward_mode="selfish", seed=None, model_path=None):
        if reward_mode not in REWARD_MODES:
            raise ValueError(f"reward_mode must be one of {REWARD_MODES}")

        self.n_channels = n_channels
        self.n_nodes = n_nodes
        self.n_mcs = n_mcs
        self.episode_len = episode_len
        self.reward_mode = reward_mode

        kw = {"model_path": model_path} if model_path else {}
        self.channel = FrameOracleChannel(n_channels, n_mcs, seed=seed, **kw)

        self.n_actions = n_channels * n_mcs
        self._rng = np.random.default_rng(seed)
        self._step = 0
        self._occupancy = np.zeros(n_channels, dtype=np.int8)

        # Best achievable per node with no contention, used as the ceiling.
        table = self.channel.probability_table()
        self._best_expected = float(
            (table * self.channel.mcs_bits[None, :]).max())

    def decode(self, action):
        """Flat action index -> (channel, mcs)."""
        return int(action) // self.n_mcs, int(action) % self.n_mcs

    def reset(self, seed=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            self.channel.rng = np.random.default_rng(seed + 1)
        self._step = 0
        self._occupancy = np.zeros(self.n_channels, dtype=np.int8)
        start = self._rng.integers(0, self.n_channels, self.n_nodes)
        for c in start:
            self._occupancy[c] = 1
        return [self._occupancy.copy() for _ in range(self.n_nodes)]

    def step(self, actions):
        decoded = [self.decode(a) for a in actions]
        chans = np.array([c for c, _ in decoded])
        counts = np.bincount(chans, minlength=self.n_channels)

        throughputs = np.zeros(self.n_nodes)
        received = np.zeros(self.n_nodes, dtype=bool)
        probs = np.zeros(self.n_nodes)

        for i, (c, m) in enumerate(decoded):
            others = int(counts[c] - 1)
            ok, thr, p = self.channel.transmit(c, m, self._occupancy, others)
            received[i], throughputs[i], probs[i] = ok, thr, p

        # Reward is delivered bits, not a success flag. A node that lands a
        # 6-bit frame did more than one that landed a 1-bit frame, and the
        # agent has to feel that difference to learn the MCS tradeoff.
        payoff = throughputs.copy()

        if self.reward_mode == "collaborative":
            payoff = np.full(self.n_nodes, throughputs.mean())

        occ = np.zeros(self.n_channels, dtype=np.int8)
        occ[chans] = 1
        self._occupancy = occ
        self._step += 1

        info = {
            "throughput": float(throughputs.sum()),
            "received": received.copy(),
            "p_success": probs.copy(),
            "collisions": int((counts > 1).sum()),
            "channels": chans.copy(),
            "mcs": np.array([m for _, m in decoded]),
            "per_node_throughput": throughputs.copy(),
        }
        done = self._step >= self.episode_len
        return [occ.copy() for _ in range(self.n_nodes)], payoff, done, info

    # -- reference points ------------------------------------------------------
    def best_throughput(self):
        """Every node alone on a distinct channel at that channel's best MCS.

        The ceiling. Unreachable in practice because nodes cannot coordinate,
        but it is the number a learning curve has to be read against.
        """
        table = self.channel.probability_table() * self.channel.mcs_bits[None, :]
        per_channel_best = table.max(axis=1)
        k = min(self.n_nodes, self.n_channels)
        return float(np.sort(per_channel_best)[::-1][:k].sum())

    def random_throughput(self, episodes=30, seed=0):
        """Expected throughput when every node picks a random (channel, MCS).

        The floor. Measured rather than derived, because the channel model is
        a neural network and there is no closed form for it.
        """
        rng = np.random.default_rng(seed)
        total, steps = 0.0, 0
        for ep in range(episodes):
            self.reset(seed=50_000 + ep)
            done = False
            while not done:
                acts = rng.integers(0, self.n_actions, self.n_nodes)
                _, _, done, info = self.step(acts)
                total += info["throughput"]
                steps += 1
        return total / steps

    def greedy_throughput(self, episodes=30, seed=0):
        """Every node always takes the single best (channel, MCS) pair.

        The trap. All nodes pile onto the same channel, interference costs
        6 dB each, and everyone does badly. A learning rule that cannot beat
        this has not learned to share.
        """
        table = self.channel.probability_table() * self.channel.mcs_bits[None, :]
        best = int(np.argmax(table))
        total, steps = 0.0, 0
        for ep in range(episodes):
            self.reset(seed=60_000 + ep)
            done = False
            while not done:
                _, _, done, info = self.step([best] * self.n_nodes)
                total += info["throughput"]
                steps += 1
        return total / steps
