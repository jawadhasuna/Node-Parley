"""Stage B+: agents choosing channel AND modulation, judged by Frame-Oracle.

Same independent tabular Q-learners as Stage B, but now with 16 actions
instead of 4 (4 channels x 4 MCS levels) and success decided by the predictor
trained on real Colosseum frames rather than by a collision rule.

Both reward rules again, across multiple seeds, because Stage B showed that a
single run of independent learners reports which equilibrium it stumbled into
rather than which rule is better.

Run:  uv run train_oracle.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from oracle_env import REWARD_MODES, OracleSpectrumEnv
from qlearning import QLearner, encode

N_CHANNELS, N_NODES, N_MCS = 4, 3, 4
EPISODES = 1200
EPISODE_LEN = 100
SEEDS = [0, 1, 2, 3, 4]


def run(reward_mode, seed):
    env = OracleSpectrumEnv(N_CHANNELS, N_NODES, N_MCS, EPISODE_LEN,
                            reward_mode=reward_mode, seed=seed)
    agents = [QLearner(n_states=2 ** N_CHANNELS, n_actions=env.n_actions,
                       alpha=0.1, gamma=0.5,
                       eps_decay_steps=EPISODES * EPISODE_LEN // 2,
                       seed=seed * 100 + i)
              for i in range(N_NODES)]

    curve = []
    for ep in range(EPISODES):
        obs = env.reset(seed=ep)
        states = [encode(o) for o in obs]
        ep_thr = 0.0
        done = False
        while not done:
            actions = [ag.act(s) for ag, s in zip(agents, states)]
            obs2, rewards, done, info = env.step(actions)
            states2 = [encode(o) for o in obs2]
            for ag, s, a, r, s2 in zip(agents, states, actions, rewards,
                                       states2):
                ag.update(s, a, r, s2)
            states = states2
            ep_thr += info["throughput"]
        curve.append(ep_thr / EPISODE_LEN)

    # Greedy evaluation over several episodes -- outcomes are now stochastic,
    # so a single episode would mostly measure luck.
    thr, chans, mcs, coll, steps = 0.0, [], [], 0, 0
    for ep in range(20):
        obs = env.reset(seed=90_000 + ep)
        states = [encode(o) for o in obs]
        done = False
        while not done:
            actions = [ag.act(s, greedy=True) for ag, s in zip(agents, states)]
            obs2, _, done, info = env.step(actions)
            states = [encode(o) for o in obs2]
            thr += info["throughput"]
            chans.extend(info["channels"].tolist())
            mcs.extend(info["mcs"].tolist())
            coll += info["collisions"]
            steps += 1

    return {
        "throughput": thr / steps,
        "collisions": coll / steps,
        "channel_use": np.bincount(chans, minlength=N_CHANNELS).tolist(),
        "mcs_use": np.bincount(mcs, minlength=N_MCS).tolist(),
        "curve": curve,
    }


ref = OracleSpectrumEnv(N_CHANNELS, N_NODES, N_MCS, EPISODE_LEN, seed=0)
best = ref.best_throughput()
rand = ref.random_throughput()
greedy = ref.greedy_throughput()

print(f"{N_NODES} nodes, {N_CHANNELS} channels, {N_MCS} MCS levels "
      f"-> {ref.n_actions} actions per node")
print(f"outcomes decided by Frame-Oracle's predictor\n")
print("reference throughput (bits per step, all nodes)")
print(f"  best possible          : {best:.3f}")
print(f"  all random             : {rand:.3f}")
print(f"  all take the best pair : {greedy:.3f}  (they collide)")

results = {}
for mode in REWARD_MODES:
    runs = [run(mode, sd) for sd in SEEDS]
    thr = np.array([r["throughput"] for r in runs])
    med = runs[int(np.argsort(thr)[len(thr) // 2])]

    print(f"\n{'=' * 58}")
    print(f"reward rule: {mode}   ({len(SEEDS)} seeds)")
    print("=" * 58)
    print(f"  per seed   : {np.round(thr, 3).tolist()}")
    print(f"  mean {thr.mean():.3f} +- {thr.std():.3f}   "
          f"({(thr.mean() - rand) / max(best - rand, 1e-9) * 100:.1f}% of the "
          f"way from random to best)")
    print(f"  collisions/step : {med['collisions']:.3f}")
    print(f"  channel use     : {med['channel_use']}")
    print(f"  MCS use         : {med['mcs_use']}")

    results[mode] = {
        "throughput": float(thr.mean()), "throughput_std": float(thr.std()),
        "throughput_all": thr.tolist(), "collisions": med["collisions"],
        "channel_use": med["channel_use"], "mcs_use": med["mcs_use"],
        "curve": med["curve"],
    }

s, c = results["selfish"], results["collaborative"]
delta = c["throughput"] - s["throughput"]
pooled = np.sqrt((s["throughput_std"] ** 2 + c["throughput_std"] ** 2) / 2)
stderr = pooled * np.sqrt(2.0 / len(SEEDS)) if pooled > 0 else 0.0

print(f"\n{'=' * 58}")
print("does the SC2 rule earn its keep with MCS in the action space?")
print("=" * 58)
print("selfish      {0:.3f} +- {1:.3f}".format(s["throughput"],
                                               s["throughput_std"]))
print("collaborative {0:.3f} +- {1:.3f}".format(c["throughput"],
                                                c["throughput_std"]))
print("difference   {0:+.3f}".format(delta))
if stderr > 0:
    print("that is {0:.1f} standard errors".format(abs(delta) / stderr))
    print("-> {0}".format("NOT distinguishable from noise"
                          if abs(delta) < 2 * stderr else "the gap looks real"))

print("\nStage B found no difference between the rules when the only choice")
print("was a channel. With MCS in the action space there is more room for")
print("selfish and collective interests to diverge -- an aggressive MCS on a")
print("shared channel wastes the channel for everyone. Whether that is")
print("enough to separate them is what the numbers above answer.")

# --- plot ---------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5))
window = 40
for mode in REWARD_MODES:
    cur = np.array(results[mode]["curve"])
    sm = np.convolve(cur, np.ones(window) / window, mode="valid")
    ax1.plot(np.arange(len(sm)) + window, sm, lw=1.8, label=mode)
ax1.axhline(best, ls="--", color="green", alpha=0.8, label="best possible")
ax1.axhline(rand, ls=":", color="red", alpha=0.8, label="all random")
ax1.axhline(greedy, ls="-.", color="black", alpha=0.5, label="all take best")
ax1.set_xlabel("episode")
ax1.set_ylabel("bits delivered per step")
ax1.set_title("Learning with Frame-Oracle deciding outcomes")
ax1.legend(fontsize=8, loc="lower right")
ax1.grid(alpha=0.3)

x = np.arange(N_MCS)
ax2.bar(x - 0.2, s["mcs_use"], 0.4, label="selfish")
ax2.bar(x + 0.2, c["mcs_use"], 0.4, label="collaborative")
ax2.set_xticks(x, [f"MCS{m}" for m in range(N_MCS)])
ax2.set_ylabel("times chosen (greedy evaluation)")
ax2.set_title("Which modulation the agents settle on")
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3, axis="y")

fig.tight_layout()
out = Path("figures")
out.mkdir(exist_ok=True)
fig.savefig(out / "oracle_env.png", dpi=140)
print(f"\nsaved {out / 'oracle_env.png'}")

Path("results").mkdir(exist_ok=True)
with open(Path("results") / "oracle_env.json", "w") as f:
    json.dump({"reference": {"best": best, "random": rand, "greedy": greedy},
               **{k: {kk: vv for kk, vv in v.items() if kk != "curve"}
                  for k, v in results.items()}}, f, indent=2)
print("saved results/oracle_env.json")
plt.show()
