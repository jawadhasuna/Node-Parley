r"""Tabular Q-learning, written out rather than imported.

The whole algorithm is the one line in `update` below. Everything else is
bookkeeping. Writing it by hand once means the frameworks in Stage B are a
convenience rather than a black box.

WHAT Q-LEARNING IS
------------------
Q[state, action] estimates "total future reward if I take this action here and
behave sensibly afterwards". The table starts at zero -- the agent believes
nothing. After each step it nudges the entry it just used toward what actually
happened:

    Q[s,a]  <-  Q[s,a] + alpha * ( r + gamma * max_a' Q[s',a']  -  Q[s,a] )
                                   \_______________________/     \______/
                                    what we now think it's worth   what we
                                    (reward, plus the best we can  thought
                                     do from where we landed)

The bracketed difference is the temporal-difference error: how wrong the old
estimate was. alpha decides how much of that correction to accept.

gamma discounts future reward. At gamma = 0 the agent is myopic and chases
immediate reward only; near 1 it plans further ahead. Channel selection has
almost no long-run structure -- picking a clear channel now does not change
what is free next step -- so a low gamma is appropriate and a high one mostly
adds noise.

EXPLORATION
-----------
An agent that always takes its current best action never discovers that
something else is better. Epsilon-greedy takes a random action with
probability epsilon, decayed over training: explore early, exploit later.
"""

import numpy as np


def encode(observation) -> int:
    """Pack a binary occupancy vector into a single table index.

    MultiBinary(n) has 2^n possible values, so the observation becomes a
    base-2 integer. Fine at n = 4 (16 states); this is exactly the step that
    stops working as state spaces grow, and the reason Stage B needs a network
    instead of a table.
    """
    obs = np.asarray(observation).ravel()
    return int(np.dot(obs, 1 << np.arange(len(obs))[::-1]))


class QLearner:
    """Tabular Q-learning with epsilon-greedy exploration."""

    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.5,
                 eps_start=1.0, eps_end=0.02, eps_decay_steps=20_000,
                 seed=0):
        self.q = np.zeros((n_states, n_actions), dtype=np.float64)
        self.alpha = alpha
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay_steps = eps_decay_steps
        self.n_actions = n_actions
        self.rng = np.random.default_rng(seed)
        self.steps = 0

    @property
    def epsilon(self) -> float:
        """Linear decay from eps_start to eps_end, then flat."""
        frac = min(1.0, self.steps / max(self.eps_decay_steps, 1))
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    def act(self, state: int, greedy: bool = False) -> int:
        """Epsilon-greedy action, or purely greedy for evaluation."""
        if not greedy and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))

        row = self.q[state]
        # Break ties randomly. Without this, an all-zero table always returns
        # action 0, so the agent never explores by accident and early learning
        # is biased toward whichever action happens to be first.
        best = np.flatnonzero(row == row.max())
        return int(self.rng.choice(best))

    def update(self, state, action, reward, next_state, done=False):
        """One temporal-difference update. This is the algorithm."""
        best_next = 0.0 if done else self.q[next_state].max()
        td_error = reward + self.gamma * best_next - self.q[state, action]
        self.q[state, action] += self.alpha * td_error
        self.steps += 1
        return td_error

    def greedy_policy(self):
        """The action this agent would take in each state, for inspection.

        Worth looking at directly: a Q-table is small enough to read, and
        seeing WHICH channel it picks for each occupancy pattern tells you
        whether it learned the structure or just found one lucky channel.
        """
        return self.q.argmax(axis=1)
