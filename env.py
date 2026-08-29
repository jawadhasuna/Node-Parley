"""A minimal spectrum-access environment.

N channels, one learning agent, and a few interferers that occupy channels
according to a pattern. Each step the agent picks one channel. If nobody else
is there the transmission succeeds; if an interferer is there it collides and
both lose.

The agent observes LAST step's channel occupancy, not this step's. That is
not a simplification -- it is the real constraint. A radio senses the spectrum,
then transmits; by the time it transmits, the world has moved on. Giving the
agent the current occupancy would let it cheat by reading the answer.

INTERFERER PATTERNS
-------------------
Three, chosen so that success and failure are both diagnosable:

  static  Each interferer sits on one channel forever. Trivially learnable:
          the agent should find a free channel and stay there.

  cyclic  Interferers rotate through channels on a fixed period. Last step's
          occupancy fully determines this step's, so it is learnable -- but
          only by an agent that actually uses its observation rather than
          settling on one channel.

  random  Interferers choose uniformly at random each step. NOTHING is
          learnable here. An agent that beats chance on this pattern has a
          bug, not a policy.

That last one matters. It is the same role the -20 dB point plays in
Mod-Scope's accuracy curve: a place where the correct result is "no better
than guessing", so a suspiciously good number is a warning rather than a win.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

PATTERNS = ("static", "cyclic", "random")


class SpectrumEnv(gym.Env):
    """Single-agent channel selection against patterned interference.

    Observation: MultiBinary(n_channels) -- which channels were busy LAST step.
    Action:      Discrete(n_channels)    -- which channel to transmit on.
    Reward:      +1 clear transmission, -1 collision.
    """

    metadata = {"render_modes": []}

    def __init__(self, n_channels=4, n_interferers=2, pattern="cyclic",
                 episode_len=200, seed=None):
        super().__init__()
        if pattern not in PATTERNS:
            raise ValueError(f"pattern must be one of {PATTERNS}")
        if n_interferers >= n_channels:
            raise ValueError("need at least one channel free of interferers")

        self.n_channels = n_channels
        self.n_interferers = n_interferers
        self.pattern = pattern
        self.episode_len = episode_len

        self.action_space = spaces.Discrete(n_channels)
        self.observation_space = spaces.MultiBinary(n_channels)

        self._rng = np.random.default_rng(seed)
        self._step = 0
        self._occupancy = np.zeros(n_channels, dtype=np.int8)
        self._home = None       # per-interferer base channel
        self._schedule = None   # occupancy for the whole episode

    # -- interference ---------------------------------------------------------
    def _interferer_channels(self, t):
        """Which channels the interferers occupy at step t."""
        if self.pattern == "static":
            return self._home
        if self.pattern == "cyclic":
            # Each interferer advances one channel per step from its own base,
            # so the set of busy channels rotates deterministically.
            return (self._home + t) % self.n_channels
        return self._rng.integers(0, self.n_channels, self.n_interferers)

    def _occupancy_at(self, t):
        """Occupancy at step t. A LOOKUP, not a fresh sample.

        Precomputing the whole episode at reset matters for the `random`
        pattern: drawing new numbers on every call means querying the future
        twice gives two different futures, so an oracle policy ends up
        choosing against a world that never happens. The first version of this
        did exactly that and reported a ceiling of +0.127 for random
        interference when the true ceiling is near +1.0.
        """
        return self._schedule[min(t, len(self._schedule) - 1)]

    def _build_schedule(self):
        """Fix the entire episode's interference up front."""
        sched = np.zeros((self.episode_len + 2, self.n_channels), dtype=np.int8)
        for t in range(len(sched)):
            sched[t, self._interferer_channels(t)] = 1
        return sched

    # -- gym API --------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # Distinct base channels so interferers do not stack on each other.
        self._home = self._rng.choice(self.n_channels, self.n_interferers,
                                      replace=False)
        self._step = 0
        self._schedule = self._build_schedule()
        self._occupancy = self._occupancy_at(0)
        return self._occupancy.copy(), {}

    def step(self, action):
        # The agent acted on LAST step's observation; the world has advanced.
        self._step += 1
        busy = self._occupancy_at(self._step)

        collided = bool(busy[action])
        reward = -1.0 if collided else 1.0

        # What the agent will see next is what just happened.
        self._occupancy = busy
        terminated = False
        truncated = self._step >= self.episode_len

        info = {"collided": collided, "busy": busy.copy(),
                "n_free": int(self.n_channels - busy.sum())}
        return self._occupancy.copy(), reward, terminated, truncated, info

    # -- reference policies ---------------------------------------------------
    def optimal_action(self):
        """What a policy with perfect knowledge of the NEXT step would pick.

        The ceiling. Not achievable by any agent that only sees the past, but
        the right thing to measure against: a learning curve is meaningless
        without knowing what perfect looks like.
        """
        busy = self._occupancy_at(self._step + 1)
        free = np.flatnonzero(busy == 0)
        return int(free[0]) if len(free) else 0

    def expected_random_reward(self):
        """Mean reward for choosing uniformly at random.

        The floor. With k interferers on n channels the chance of collision is
        roughly k/n, so reward is about (1 - k/n) - (k/n) = 1 - 2k/n. Computed
        exactly here because interferers can share a channel under `random`.
        """
        n, k = self.n_channels, self.n_interferers
        if self.pattern == "random":
            p_free = ((n - 1) / n) ** k  # all k interferers miss a given cell
        else:
            p_free = (n - k) / n
        return p_free - (1 - p_free)
