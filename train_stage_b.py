"""Stage B: does DARPA's collaborative scoring rule actually help?

Independent tabular Q-learners, one per node, all learning at once. Trained
twice on identical settings -- once paid for their own success, once paid the
ensemble average as SC2 scored it -- and compared on the thing SC2 actually
measured: total throughput delivered.

This is a real question, not a demonstration. Selfish agents already dislike
collisions, so they may well discover the same polite behaviour without being
told to. If both rules converge to the same throughput, the collaborative
reward earned nothing here and that is the honest finding.

Run:  uv run train_stage_b.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from multi_env import REWARD_MODES, MultiSpectrumEnv, fairness
from qlearning import QLearner, encode

N_CHANNELS = 4
N_NODES = 3
EPISODES = 1500
EPISODE_LEN = 100
SEED = 0


def run(reward_mode, seed=SEED):
    env = MultiSpectrumEnv(N_CHANNELS, N_NODES, EPISODE_LEN,
                           reward_mode=reward_mode, seed=seed)

    # One independent learner per node. No shared table, no communication --
    # each sees only aggregate occupancy and its own reward. This is the
    # weakest possible form of multi-agent learning, which makes it the right
    # baseline: anything it achieves comes from the reward rule alone.
    agents = [QLearner(n_states=2 ** N_CHANNELS, n_actions=N_CHANNELS,
                       alpha=0.1, gamma=0.5,
                       eps_decay_steps=EPISODES * EPISODE_LEN // 2,
                       seed=seed + i)
              for i in range(N_NODES)]

    thr_curve, coll_curve = [], []

    for ep in range(EPISODES):
        obs = env.reset(seed=ep)
        states = [encode(o) for o in obs]
        ep_thr, ep_coll, success = 0.0, 0, np.zeros(N_NODES)

        done = False
        while not done:
            actions = [ag.act(s) for ag, s in zip(agents, states)]
            obs2, rewards, done, info = env.step(actions)
            states2 = [encode(o) for o in obs2]

            for ag, s, a, r, s2 in zip(agents, states, actions, rewards,
                                       states2):
                ag.update(s, a, r, s2, done=False)

            states = states2
            ep_thr += info["throughput"]
            ep_coll += info["collisions"]
            success += info["per_node_success"]

        thr_curve.append(ep_thr / EPISODE_LEN)
        coll_curve.append(ep_coll / EPISODE_LEN)

    # Evaluate greedily -- exploration noise would understate the policy.
    obs = env.reset(seed=99_999)
    states = [encode(o) for o in obs]
    thr, coll, success = 0.0, 0, np.zeros(N_NODES)
    done = False
    while not done:
        actions = [ag.act(s, greedy=True) for ag, s in zip(agents, states)]
        obs2, _, done, info = env.step(actions)
        states = [encode(o) for o in obs2]
        thr += info["throughput"]
        coll += info["collisions"]
        success += info["per_node_success"]

    return {
        "throughput": thr / EPISODE_LEN,
        "collisions": coll / EPISODE_LEN,
        "success_rates": (success / EPISODE_LEN).tolist(),
        "fairness": fairness(success / EPISODE_LEN),
        "curve": thr_curve,
        "collision_curve": coll_curve,
    }


ref = MultiSpectrumEnv(N_CHANNELS, N_NODES, EPISODE_LEN, seed=SEED)
best = ref.best_throughput()
rand = ref.random_throughput()
greedy = ref.greedy_throughput()

print(f"{N_NODES} nodes, {N_CHANNELS} channels "
      f"worth {np.round(ref.channel_values, 2).tolist()}")
print(f"\nreference throughput per step")
print(f"  best possible (distinct channels) : {best:.3f}")
print(f"  all uniformly random              : {rand:.3f}")
print(f"  all grab the best channel         : {greedy:.3f}  (total collision)")

# Repeat over seeds. A single run of a multi-agent system says very little:
# independent learners fall into whichever equilibrium they stumble on
# first, so two rules can differ by chance alone. The spread across seeds
# is what decides whether a gap is real.
SEEDS = [0, 1, 2, 3, 4, 5, 6, 7]

print()
print(f"running {len(SEEDS)} seeds per rule -- one run of independent")
print("learners lands in whichever equilibrium it finds first, so a")
print("single comparison cannot separate a real effect from luck.")

all_runs = {m: [run(m, seed=sd) for sd in SEEDS] for m in REWARD_MODES}

results = {}
for mode in REWARD_MODES:
    runs = all_runs[mode]
    thr = np.array([x['throughput'] for x in runs])
    col = np.array([x['collisions'] for x in runs])
    fair = np.array([x['fairness'] for x in runs])

    print()
    print('=' * 58)
    print(f'reward rule: {mode}   ({len(SEEDS)} seeds)')
    print('=' * 58)
    print(f'  per seed : {np.round(thr, 3).tolist()}')
    print(f'  mean {thr.mean():.3f} +- {thr.std():.3f}   '
          f'min {thr.min():.3f}  max {thr.max():.3f}')

    # Keep the median run for the curves and per-node bars.
    med = runs[int(np.argsort(thr)[len(thr) // 2])]
    results[mode] = dict(med)
    results[mode].update({
        'throughput': float(thr.mean()),
        'throughput_std': float(thr.std()),
        'throughput_all': thr.tolist(),
        'collisions': float(col.mean()),
        'fairness': float(fair.mean()),
    })
    r = results[mode]
    print(f"  throughput      {r['throughput']:.3f}  "
          f"({(r['throughput'] - rand) / max(best - rand, 1e-9) * 100:.1f}% of "
          f"the way from random to best)")
    print(f"  collisions/step {r['collisions']:.3f}")
    print(f"  per-node success {np.round(r['success_rates'], 3).tolist()}")
    print(f"  fairness        {r['fairness']:.3f}  (1.0 = perfectly equal)")

# --- the comparison -----------------------------------------------------------
s, c = results["selfish"], results["collaborative"]
print(f"\n{'=' * 58}")
print("does the SC2 rule earn its keep?")
print("=" * 58)
print(f"{'metric':<18} {'selfish':>10} {'collaborative':>14} {'diff':>9}")
print("-" * 54)
print(f"{'throughput':<18} {s['throughput']:>10.3f} "
      f"{c['throughput']:>14.3f} {c['throughput'] - s['throughput']:>+9.3f}")
print(f"{'collisions/step':<18} {s['collisions']:>10.3f} "
      f"{c['collisions']:>14.3f} {c['collisions'] - s['collisions']:>+9.3f}")
print(f"{'fairness':<18} {s['fairness']:>10.3f} "
      f"{c['fairness']:>14.3f} {c['fairness'] - s['fairness']:>+9.3f}")

delta = c["throughput"] - s["throughput"]

# Is the gap larger than the seed-to-seed scatter? Without this a
# difference between two single runs means nothing.
pooled = np.sqrt((s['throughput_std'] ** 2 + c['throughput_std'] ** 2) / 2)
n_seeds = len(s['throughput_all'])
stderr = pooled * np.sqrt(2.0 / n_seeds) if pooled > 0 else 0.0
print()
print('seed spread: selfish +-{0:.3f}, collaborative +-{1:.3f}'
      .format(s['throughput_std'], c['throughput_std']))
if stderr > 0:
    print(f'gap is {abs(delta) / stderr:.1f} standard errors ({n_seeds} seeds each)')
    if abs(delta) < 2 * stderr:
        print('-> under 2 standard errors: NOT distinguishable from noise.')
    else:
        print('-> over 2 standard errors: the gap looks real.')
else:
    print('every seed gave an identical result, so the gap is exact.')

if abs(delta) < 0.02 * max(best, 1e-9):
    print("\nThe two rules land in the same place. Selfish agents already")
    print("lose from collisions, so they learn to spread out without being")
    print("told to care about anyone else. On this setup the collaborative")
    print("reward bought nothing measurable.")
elif delta > 0:
    print(f"\nThe collaborative rule delivers {delta:+.3f} more throughput per")
    print("step. Paying every node the ensemble average makes wrecking a")
    print("neighbour costly, and the nodes spread out further because of it.")
else:
    print(f"\nThe collaborative rule delivers {delta:+.3f} LESS throughput.")
    print("Averaging the reward hides which node caused a collision, so the")
    print("credit-assignment problem got harder -- a known cost of shared")
    print("rewards in multi-agent learning.")

# --- plot ---------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
window = 50

for mode in REWARD_MODES:
    c_ = np.array(results[mode]["curve"])
    smooth = np.convolve(c_, np.ones(window) / window, mode="valid")
    ax1.plot(np.arange(len(smooth)) + window, smooth, lw=1.8, label=mode)

ax1.axhline(best, ls="--", color="green", alpha=0.8, label="best possible")
ax1.axhline(rand, ls=":", color="red", alpha=0.8, label="all random")
ax1.axhline(greedy, ls="-.", color="black", alpha=0.5,
            label="all grab best channel")
ax1.set_xlabel("episode")
ax1.set_ylabel("throughput per step")
ax1.set_title(f"{N_NODES} nodes learning at once on {N_CHANNELS} channels")
ax1.legend(fontsize=8, loc="lower right")
ax1.grid(alpha=0.3)

x = np.arange(N_NODES)
ax2.bar(x - 0.2, results["selfish"]["success_rates"], 0.4, label="selfish")
ax2.bar(x + 0.2, results["collaborative"]["success_rates"], 0.4,
        label="collaborative")
ax2.set_xticks(x, [f"node {i}" for i in range(N_NODES)])
ax2.set_ylabel("share of steps transmitting successfully")
ax2.set_title("Who actually gets to talk\n"
              f"fairness: selfish {s['fairness']:.2f}, "
              f"collaborative {c['fairness']:.2f}")
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3, axis="y")

fig.tight_layout()
out = Path("figures")
out.mkdir(exist_ok=True)
fig.savefig(out / "stage_b.png", dpi=140)
print(f"\nsaved {out / 'stage_b.png'}")

Path("results").mkdir(exist_ok=True)
with open(Path("results") / "stage_b.json", "w") as f:
    json.dump({"reference": {"best": best, "random": rand, "greedy": greedy},
               **{k: {kk: vv for kk, vv in v.items()
                      if kk not in ("curve", "collision_curve")}
                  for k, v in results.items()}}, f, indent=2)
print("saved results/stage_b.json")

plt.show()
