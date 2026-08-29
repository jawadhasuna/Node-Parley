"""Stage B2: the same environment under Ray RLlib.

The science is already done. Stage B+ established the result with a Q-table
written by hand. This checks the environment works with production tooling and
that the finding survives a different algorithm.

THE SYMMETRY PROBLEM
--------------------
This run exposed something worth more than the comparison itself.

All three nodes share one policy (standard parameter sharing -- the nodes are
identical) AND see an identical observation (aggregate channel occupancy is
the same array for everyone). So with a DETERMINISTIC policy, all three
produce the same action, pick the same channel, and jam each other on every
step. They cannot coordinate; it is structurally impossible.

Greedy evaluation -- correct everywhere else in this project -- is therefore
wrong here. It removes the only thing breaking the symmetry: the randomness
PPO trains with. Both modes are reported below, because the gap between them
IS the finding.

Three ways out, in increasing order of honesty:
    sample instead of argmax     works, but leans on noise to coordinate
    one policy per node          breaks symmetry through separate weights
    agent id in the observation  lets a shared policy tell nodes apart

The tabular runs never hit this because independent Q-learners have separate
tables, and their separate exploration histories break the symmetry for free.

Run:  uv run train_rllib.py
      uv run train_rllib.py --iterations 60
"""

import argparse
import json
from pathlib import Path

import numpy as np

N_CHANNELS, N_NODES, N_MCS = 4, 3, 4
EPISODE_LEN = 100

# Q-table results from train_oracle.py, for comparison.
TABULAR = {"selfish": 2.658, "collaborative": 3.320}


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--iterations", type=int, default=40)
    p.add_argument("--reward-mode", default="collaborative",
                   choices=["selfish", "collaborative"])
    p.add_argument("--eval-episodes", type=int, default=20)
    return p.parse_args()


def main():
    args = get_args()

    import ray
    import torch
    from ray.rllib.algorithms.ppo import PPOConfig
    from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
    from ray.tune.registry import register_env

    from oracle_env import OracleSpectrumEnv
    from pz_env import SpectrumParallelEnv

    def env_creator(cfg):
        return ParallelPettingZooEnv(SpectrumParallelEnv(
            n_channels=N_CHANNELS, n_nodes=N_NODES, n_mcs=N_MCS,
            episode_len=EPISODE_LEN,
            reward_mode=cfg.get("reward_mode", args.reward_mode)))

    register_env("spectrum", env_creator)

    # num_env_runners=0 keeps sampling in the driver process. The environment
    # is a few numpy operations and one tiny ONNX call, so worker processes
    # would cost more than they save. (Ray 2.58 removed local_mode.)
    ray.init(log_to_driver=False, include_dashboard=False)

    print("reward rule : {0}".format(args.reward_mode))
    print("iterations  : {0}".format(args.iterations))
    print("tabular Q-learning reference: {0:.3f} bits/step".format(
        TABULAR[args.reward_mode]))
    print()

    config = (
        PPOConfig()
        .environment("spectrum", env_config={"reward_mode": args.reward_mode})
        .framework("torch")
        .env_runners(num_env_runners=0)
        .multi_agent(
            policies={"shared"},
            policy_mapping_fn=lambda agent_id, *a, **kw: "shared",
        )
        .training(train_batch_size=4000, lr=3e-4, gamma=0.5)
    )

    algo = (config.build_algo() if hasattr(config, "build_algo")
            else config.build())

    print("{0:>5} {1:>16}".format("iter", "episode return"))
    print("-" * 24)
    history = []
    for i in range(args.iterations):
        result = algo.train()
        ret = None
        for path in (("env_runners", "episode_return_mean"),
                     ("env_runners", "episode_reward_mean"),
                     ("episode_reward_mean",)):
            node = result
            for k in path:
                node = node.get(k) if isinstance(node, dict) else None
                if node is None:
                    break
            if isinstance(node, (int, float)):
                ret = float(node)
                break
        history.append(ret)
        if i % 5 == 0 or i == args.iterations - 1:
            print("{0:>5} {1:>16.3f}".format(
                i, ret if ret is not None else float("nan")))

    # --- evaluate on the raw environment, scored like the Q-table ------------
    module = algo.get_module("shared")
    raw = OracleSpectrumEnv(N_CHANNELS, N_NODES, N_MCS, EPISODE_LEN,
                            reward_mode=args.reward_mode, seed=0)

    def evaluate(greedy):
        """Run the trained policy through the raw environment.

        greedy=True takes argmax; greedy=False samples, which is how PPO was
        trained and the only thing letting three identical nodes that share
        one policy pick different channels.
        """
        thr, coll, steps = 0.0, 0, 0
        chans, mcs = [], []
        gen = torch.Generator().manual_seed(0)

        for ep in range(args.eval_episodes):
            obs_list = raw.reset(seed=90_000 + ep)
            done = False
            while not done:
                batch = torch.tensor(np.array(obs_list), dtype=torch.float32)
                with torch.no_grad():
                    out = module.forward_inference({"obs": batch})
                logits = out["action_dist_inputs"]

                if greedy:
                    actions = torch.argmax(logits, dim=-1).numpy()
                else:
                    probs = torch.softmax(logits, dim=-1)
                    actions = torch.multinomial(
                        probs, 1, generator=gen).squeeze(-1).numpy()

                obs_list, _, done, info = raw.step(actions)
                thr += info["throughput"]
                coll += info["collisions"]
                chans.extend(info["channels"].tolist())
                mcs.extend(info["mcs"].tolist())
                steps += 1

        return {
            "throughput": thr / steps,
            "collisions": coll / steps,
            "channel_use": np.bincount(chans, minlength=N_CHANNELS).tolist(),
            "mcs_use": np.bincount(mcs, minlength=N_MCS).tolist(),
        }

    print()
    print("evaluating on the raw environment, scored exactly as")
    print("train_oracle.py scores the Q-table")
    print()

    greedy_r = evaluate(greedy=True)
    sample_r = evaluate(greedy=False)
    ref = TABULAR[args.reward_mode]

    print("{0:<24}{1:>10}{2:>10}{3:>12}".format(
        "", "greedy", "sampled", "tabular Q"))
    print("-" * 56)
    print("{0:<24}{1:>10.3f}{2:>10.3f}{3:>12.3f}".format(
        "throughput (bits/step)", greedy_r["throughput"],
        sample_r["throughput"], ref))
    print("{0:<24}{1:>10.3f}{2:>10.3f}".format(
        "collisions/step", greedy_r["collisions"], sample_r["collisions"]))
    print()
    print("channel use, greedy  : {0}".format(greedy_r["channel_use"]))
    print("channel use, sampled : {0}".format(sample_r["channel_use"]))
    print("MCS use, sampled     : {0}".format(sample_r["mcs_use"]))

    print()
    print("The greedy column is the symmetry problem, not a bad policy.")
    print("One shared policy plus an identical observation for every node")
    print("means a deterministic choice is the SAME choice for all three, so")
    print("they jam each other every step by construction. Sampling is the")
    print("only thing breaking that tie, and it is what PPO trained with.")
    print()
    print("Independent Q-learners never hit this: separate tables and")
    print("separate exploration histories break the symmetry for free.")

    Path("results").mkdir(exist_ok=True)
    out_path = Path("results") / "rllib_{0}.json".format(args.reward_mode)
    with open(out_path, "w") as f:
        json.dump({"greedy": greedy_r, "sampled": sample_r,
                   "tabular_reference": ref, "iterations": args.iterations,
                   "return_history": history}, f, indent=2)
    print()
    print("saved {0}".format(out_path))

    algo.stop()
    ray.shutdown()


if __name__ == "__main__":
    main()
