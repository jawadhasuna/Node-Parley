"""Stage A: does tabular Q-learning learn to dodge interference?

Trains on all three interference patterns and measures against both a floor
(random choice) and a ceiling (a policy that knows the future). A learning
curve without those two lines is unreadable -- you cannot tell 0.6 from good.

Run:  uv run train_stage_a.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from env import PATTERNS, SpectrumEnv
from qlearning import QLearner, encode

N_CHANNELS = 4
N_INTERFERERS = 2
EPISODES = 600
EPISODE_LEN = 200
SEED = 0


def evaluate(env, agent, episodes=40, greedy=True):
    """Mean reward per step under the agent's current policy."""
    total, steps = 0.0, 0
    for ep in range(episodes):
        obs, _ = env.reset(seed=10_000 + ep)
        done = False
        while not done:
            a = agent.act(encode(obs), greedy=greedy)
            obs, r, term, trunc, _ = env.step(a)
            total += r
            steps += 1
            done = term or trunc
    return total / steps


def reference_scores(env, episodes=40):
    """Ceiling (knows the future) and floor (uniform random)."""
    opt_total, rand_total, steps = 0.0, 0.0, 0
    rng = np.random.default_rng(0)
    for ep in range(episodes):
        env.reset(seed=10_000 + ep)
        done = False
        while not done:
            a = env.optimal_action()
            _, r, term, trunc, _ = env.step(a)
            opt_total += r
            steps += 1
            done = term or trunc

        env.reset(seed=10_000 + ep)
        done = False
        while not done:
            _, r, term, trunc, _ = env.step(int(rng.integers(env.n_channels)))
            rand_total += r
            done = term or trunc
    return opt_total / steps, rand_total / steps


results = {}

for pattern in PATTERNS:
    print(f"\n{'=' * 58}")
    print(f"pattern: {pattern}")
    print("=" * 58)

    env = SpectrumEnv(N_CHANNELS, N_INTERFERERS, pattern,
                      episode_len=EPISODE_LEN, seed=SEED)
    agent = QLearner(n_states=2 ** N_CHANNELS, n_actions=N_CHANNELS,
                     alpha=0.1, gamma=0.5,
                     eps_decay_steps=EPISODES * EPISODE_LEN // 2, seed=SEED)

    ceiling, floor = reference_scores(env)
    theory_floor = env.expected_random_reward()
    print(f"ceiling (knows future) : {ceiling:+.3f} reward/step")
    print(f"floor (random)         : {floor:+.3f}  "
          f"(theory {theory_floor:+.3f})")

    curve = []
    for ep in range(EPISODES):
        obs, _ = env.reset(seed=ep)
        s = encode(obs)
        done = False
        ep_reward = 0.0
        while not done:
            a = agent.act(s)
            obs2, r, term, trunc, _ = env.step(a)
            s2 = encode(obs2)
            agent.update(s, a, r, s2, done=term)
            s, ep_reward = s2, ep_reward + r
            done = term or trunc
        curve.append(ep_reward / EPISODE_LEN)

        if ep % 150 == 0 or ep == EPISODES - 1:
            print(f"  episode {ep:>4}  reward/step {np.mean(curve[-50:]):+.3f}"
                  f"   epsilon {agent.epsilon:.3f}")

    final = evaluate(env, agent)
    span = ceiling - floor
    frac = (final - floor) / span if abs(span) > 1e-9 else float("nan")

    print(f"\n  final (greedy)  : {final:+.3f} reward/step")
    print(f"  vs floor        : {final - floor:+.3f}")
    print(f"  vs ceiling      : {final - ceiling:+.3f}")
    if abs(span) > 1e-9:
        print(f"  closed {frac * 100:.1f}% of the gap between random and perfect")

    # The learned policy is small enough to read. For `cyclic` a real policy
    # varies with the observation; a degenerate one picks the same channel
    # regardless, which scores above random only by luck.
    pol = agent.greedy_policy()
    visited = np.flatnonzero(agent.q.any(axis=1))
    distinct = len(set(pol[visited].tolist()))
    print(f"  states visited  : {len(visited)} of {2 ** N_CHANNELS}")
    print(f"  distinct actions across visited states: {distinct}")

    results[pattern] = {
        "curve": curve, "final": final, "ceiling": ceiling, "floor": floor,
        "fraction_of_gap": frac, "distinct_actions": distinct,
        "states_visited": int(len(visited)),
    }

# --- the check that matters ---------------------------------------------------
print(f"\n{'=' * 58}")
print("sanity: `random` interference must NOT be learnable")
print("=" * 58)
r = results["random"]
print(f"  final {r['final']:+.3f} vs floor {r['floor']:+.3f}  "
      f"(difference {r['final'] - r['floor']:+.3f})")
if abs(r["final"] - r["floor"]) < 0.05:
    print("  as expected: no structure to exploit, so no gain over chance.")
else:
    print("  WARNING: beating chance against random interference means the")
    print("  agent is seeing something it should not. Check the observation")
    print("  timing -- most likely it is being handed the current step.")

# --- plot ---------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
window = 25

for ax, pattern in zip(axes, PATTERNS):
    r = results[pattern]
    c = np.array(r["curve"])
    smooth = np.convolve(c, np.ones(window) / window, mode="valid")
    ax.plot(np.arange(len(smooth)) + window, smooth, lw=1.8, label="Q-learning")
    ax.axhline(r["ceiling"], ls="--", color="green", alpha=0.8,
               label="knows the future")
    ax.axhline(r["floor"], ls=":", color="red", alpha=0.8, label="random")
    ax.set_title(f"{pattern}\nclosed {r['fraction_of_gap'] * 100:.0f}% of the gap")
    ax.set_xlabel("episode")
    ax.grid(alpha=0.3)
axes[0].set_ylabel("reward per step")
axes[0].legend(fontsize=8, loc="lower right")

fig.suptitle("Stage A: tabular Q-learning against three interference patterns",
             fontsize=13)
fig.tight_layout()

out = Path("figures")
out.mkdir(exist_ok=True)
fig.savefig(out / "stage_a.png", dpi=140)
print(f"\nsaved {out / 'stage_a.png'}")

Path("results").mkdir(exist_ok=True)
with open(Path("results") / "stage_a.json", "w") as f:
    json.dump({k: {kk: vv for kk, vv in v.items() if kk != "curve"}
               for k, v in results.items()}, f, indent=2)
print("saved results/stage_a.json")

plt.show()
