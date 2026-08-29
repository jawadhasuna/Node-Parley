# Node-Parley

Reinforcement learning for spectrum access: radios deciding for themselves
where to transmit, with no coordinator.

Repo 4 of six in a DARPA Spectrum Collaboration Challenge project. Where
[Frame-Oracle](https://github.com/jawadhasuna/Frame-Oracle) predicts whether a
transmission will survive, Node-Parley uses that kind of judgement to *act*.

> Status: Stages A, B, B+ and B2 complete. ns-3 (Stage C) in progress.

## Stage A: tabular Q-learning, written by hand

One learning node against patterned interference. The Q-learning update is
written out in `qlearning.py` rather than imported, because Stage B's
frameworks are only worth using once the thing they wrap is understood.

The agent observes **last** step's occupancy, never this step's. That is not a
handicap -- it is the real constraint. A radio senses, then transmits, and by
the time it transmits the world has moved.

Three interference patterns, chosen so that both success and failure are
diagnosable:

```
pattern   reward/step   ceiling   floor    gap closed
static         +1.000    +1.000   -0.011        100.0%
cyclic         +1.000    +1.000   -0.001        100.0%
random         +0.104    +1.000   +0.120         -1.8%
```

`random` is the control. Interferers choosing uniformly leave no structure, so
the correct result is **no better than chance**. An agent that beat chance
there would have an observation-timing leak, not a policy. It is the same role
the -20 dB point plays in Mod-Scope's accuracy curve.

### A bug the reference line caught

The first version reported a ceiling of **+0.127** for random interference. An
oracle that truly knew the future should dodge two interferers on four
channels almost always, so that number was impossible.

Cause: `_occupancy_at()` drew fresh random numbers on every call, so asking
"what will be busy next step?" and then stepping produced two different
futures. The oracle was choosing against a world that never happened.

Every headline result was correct. The bug was visible **only** in the
reference line. Precomputing the whole episode's interference at reset fixed
it, and the ceiling moved to +1.000 as it should.

## Stage B: many nodes, and DARPA's scoring rule

Three nodes, four channels of unequal value `[1.0, 0.8, 0.6, 0.4]`, all
learning simultaneously with no communication -- each sees only aggregate
occupancy and its own reward.

Channels are unequal on purpose. If they were identical, avoiding collisions
would be the whole game and any sensible reward rule would do. Making channel
0 the prize creates the tension: everyone wants it, one node can have it.

Two scoring rules compared:

- **selfish** -- paid for your own successful transmission
- **collaborative** -- paid the average across all nodes, as SC2 scored teams

### Result: no measurable difference

```
                per-seed throughput (8 seeds)              mean      std
selfish     [2.3, 2.4, 2.3, 2.3, 2.4, 2.4, 2.3, 2.4]      2.350    0.050
collaborative [2.4, 2.4, 2.3, 2.4, 2.4, 2.1, 2.4, 2.4]    2.350    0.100

reference: best possible 2.400, all-random 1.186,
           everyone grabs channel 0 -> 0.000
```

Identical means. The gap is **0.0 standard errors**.

A single seed had shown collaborative ahead by +0.100, and a mechanism was
ready to explain it -- paying the ensemble average makes wrecking a neighbour
costly. That explanation was written before checking whether the effect
existed. It did not.

**The honest finding:** at this scale the collaborative rule bought nothing.
Selfish agents already lose from collisions, so they learn to spread out
without being told to care about anyone else.

### One thing the means hide

Collaborative reached the optimum 6 times in 8 against selfish's 4, but had a
worse worst case (2.1) and **double the variance** (0.100 vs 0.050). Same
average, different shape.

That is consistent with the credit-assignment problem: when everyone is paid
the ensemble average, a node cannot tell whether a poor reward was its own
doing or someone else's, so learning is noisier. With 8 seeds the 6-vs-4 split
is not worth claiming; the variance difference is the more defensible half.

### What would actually test the SC2 rule

The rule should matter where selfish incentives and collective good diverge.
Here they barely do, because a collision costs both nodes equally. Setups
where it should bite, none of which Stage B tests:

- more nodes than channels, so somebody must yield and selfishly nobody will
- asymmetric demand, where one node needs bandwidth urgently and another does not
- long-horizon interaction, where holding the best channel forever is
  individually optimal and collectively poor

## Stage B+: Frame-Oracle as the channel model

Stage A and B decided success with a rule: alone on the channel means the
frame lands. That is a coin flip dressed as physics. Here outcomes come from
the predictor trained in Frame-Oracle on 6.5M real Colosseum frames, and the
action becomes the real SC2 decision -- **which channel AND which modulation**.

16 actions per node instead of 4. Reward is bits delivered, not a success
flag, so landing a 6-bit frame counts six times a 1-bit one.

### The SC2 rule earns its keep here

```
                 throughput    collisions/step   channel use               MCS use
selfish        2.658 +- 0.245        1.000       [3996, 2000, 2, 2]       mostly MCS1
collaborative  3.320 +- 0.389        0.002       [2003, 1999, 998, 1000]  mostly MCS2

difference +0.662, which is 3.2 standard errors over 5 seeds
reference: best 3.576, all-random 1.176, all-take-best 0.107
```

The mechanism is visible in the behaviour, not inferred from the number.

**Selfish agents crowd the two best channels and abandon the others** --
channels 2 and 3 saw two visits each out of 6000. Two nodes permanently stacked
on channel 0 collide on every step, and because sharing costs ~6 dB they are
forced down to MCS1 to land anything.

**Collaborative agents spread across all four channels**, collisions fall to
0.002 per step, and -- the part that matters -- their clean SNR lets them run
**MCS2, four bits per symbol instead of two**.

```
spread out -> no interference -> better SNR -> higher MCS viable -> more bits
```

The collaborative reward did not merely reduce collisions. It unlocked a
faster modulation that selfish agents could never use, because selfish agents
destroyed their own channel conditions.

That is DARPA's scoring rationale, reproduced from this project's own parts:
Wave-Lathe's measured MCS ladder, Frame-Oracle's predictor, and independent
Q-learners that were never told to cooperate.

### Why Stage B found nothing and this found something

Stage B's action was only a channel, and a collision cost both nodes equally,
so selfish incentives already pointed toward spreading out. There was no wedge
between individual and collective interest.

MCS creates the wedge: an aggressive modulation on a crowded channel wastes
the airtime for everyone, and a selfish agent feels only part of that cost.

**The structure had to permit divergence before the reward rule could matter.**
Stage B's null result is the control that makes this one interpretable.

### What is measured and what is invented

The **predictor** is real: trained on measured SC2 frames, AUC 0.68-0.71 on
unseen links, replicated across two scrimmages. The **MCS requirements**
(3, 6, 13, 19 dB) come from Wave-Lathe's measured ladder.

The **mapping of this synthetic four-channel world into the predictor's
standardised feature space is a designed approximation.** The 6 dB cost per
interferer, the 18-to-2 dB channel spread, the SNR centring -- all chosen to
be plausible, none calibrated. `check_channel_model.py` also reports a
monotonicity violation at MCS0, where the physics term saturates and the
learned model is asked about a region it never saw.

So the +0.662 magnitude means little. What is robust is the mechanism:
spreading out, near-zero collisions and the shift to higher MCS are directly
observed and do not depend on those constants being right.

## Stage B2: PettingZoo and RLlib, and what symmetry costs

The environment is wrapped as a PettingZoo `ParallelEnv` (ParallelEnv, not
AECEnv: radios transmit in the same slot, they do not take turns) and trained
with Ray RLlib's PPO using a shared policy across the three identical nodes --
standard parameter sharing.

`check_pz_env.py` verifies three things: PettingZoo's own conformance test,
that the wrapper reproduces the raw environment's rewards exactly from the
same seed, and that building the action dict in a different order does not
change the outcome. That last one guards a bug that would run perfectly and be
completely wrong -- indexing actions by position rather than agent name would
silently apply node 2's choice to node 0.

### Result: a shared policy cannot solve this

```
                    greedy   sampled   tabular Q
throughput           0.545     2.079       3.320
collisions/step      1.000     0.796
channel use    [3000,3000,0,0]   [2417,2499,906,178]
```

**Greedy evaluation collapses completely.** All three nodes share one policy
AND receive an identical observation (aggregate occupancy is the same array
for everyone), so a deterministic policy produces the SAME action for all
three. They stack on one channel and jam each other on every step. The
`[3000, 3000, 0, 0]` split is the policy switching between two channels
depending on observation -- with all three nodes switching together.

Greedy evaluation is correct everywhere else in this project. Here it strips
out the only mechanism the agents had for differing: the sampling PPO trains
with. Both modes are therefore reported.

**Sampling recovers most of it but cannot reach the Q-table**, and that ceiling
is structural rather than a tuning failure. A shared policy must emit the same
action distribution for every node, so it cannot assign roles -- only randomise
and hope. Three nodes choosing independently from four channels land on
distinct ones just

```
(4 x 3 x 2) / 4^3 = 24/64 = 37.5% of the time
```

Independent Q-learners have no such limit. Each holds its own table, so one
learns "I take channel 0" and another "I take channel 1" -- genuine role
specialisation, reached with no communication, purely because three separate
exploration histories diverged.

**So parameter sharing, the standard recommendation for identical agents, is
strictly less capable here.** It is more sample-efficient and cannot solve the
problem. The cause is symmetry, not PPO.

### The fix, named but not tested

Add each agent's own identity to its observation:

```
[occupancy..., am_I_node_0, am_I_node_1, am_I_node_2]
```

A shared policy can then tell the nodes apart and assign them different
channels, recovering role specialisation while keeping shared weights. This is
a standard technique and would very likely close the gap. It is not
implemented here, and the results above should be read as "shared policy
without agent identity", not as a limit on RLlib or on PPO.

### Three tooling constraints worth recording

RLlib blocked this stage three times, none of them bugs in this code:

1. **It hard-pins gymnasium to an exact version** -- not a range. Every RLlib
   release requires one specific gymnasium, so the project had to pin
   `gymnasium==1.2.2` and cap `requires-python < 3.14` (Ray publishes no
   wheels above 3.13).
2. **`local_mode` was removed** in Ray 2.58.
3. **No default encoder for `MultiBinary` observation spaces.** The PettingZoo
   wrapper exposes `Box(0, 1)` instead -- identical data, a space RLlib can
   build a network for.

Every one of these would have blocked Stage A entirely had the roadmap not put
hand-written Q-learning first. That ordering was chosen so the update rule
would be understood before a framework hid it; it turned out to also be why
none of this friction could touch Stages A, B or B+, which run on NumPy and a
table.

## Run it

```bash
uv run train_stage_a.py    # single agent, three interference patterns
uv run train_stage_b.py    # three agents, two scoring rules, 8 seeds each

# needs Frame-Oracle's ONNX export next door
uv run check_channel_model.py   # is the channel model a real decision?
uv run train_oracle.py          # channel + MCS, outcomes from the predictor

uv run check_pz_env.py          # PettingZoo conformance + transparency
uv run train_rllib.py           # PPO with a shared policy
```

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12. CPU only; these are
tables, not networks.

## Layout

| File | Purpose |
|------|---------|
| `qlearning.py` | Tabular Q-learning, written out with the update rule explained |
| `env.py` | Single-agent environment, three interference patterns |
| `multi_env.py` | Multi-agent environment, unequal channels, two reward rules |
| `train_stage_a.py` | Stage A against floor and ceiling references |
| `train_stage_b.py` | Stage B across 8 seeds with a noise check |
| `channel_model.py` | Frame-Oracle's predictor wired in, with its caveats |
| `check_channel_model.py` | Verifies the model presents a real decision |
| `oracle_env.py` | Channel + MCS actions, outcomes from the predictor |
| `train_oracle.py` | Stage B+ across 5 seeds, both reward rules |
| `pz_env.py` | PettingZoo ParallelEnv wrapper |
| `check_pz_env.py` | Conformance, transparency, and the dict-ordering trap |
| `train_rllib.py` | RLlib PPO, both greedy and sampled evaluation |

## Still to come

- **Stage C:** ns-3. Committed, not optional -- see the project roadmap for
  its definition of done.

## References

- F. A. P. de Figueiredo et al., "SCATTER PHY: An Open Source Physical Layer
  for the DARPA Spectrum Collaboration Challenge," *Electronics* 8(11), 2019.

Independent work. Not affiliated with or endorsed by DARPA.
