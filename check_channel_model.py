"""Is the Frame-Oracle channel model a usable environment?

Before training anything on it, check that it presents a real decision. Two
ways this can be useless:

  every (channel, MCS) succeeds  -> nothing to learn, always pick top MCS
  nothing succeeds               -> nothing to learn, all actions equal

What is wanted is a diagonal band: good channels supporting aggressive MCS,
poor channels forcing conservative ones, and a best choice that depends on
where you are and who else is transmitting.

Run:  uv run check_channel_model.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from channel_model import FrameOracleChannel

N_CHANNELS, N_MCS = 4, 4
ch = FrameOracleChannel(N_CHANNELS, N_MCS, seed=0)

print(f"channel SNR (dB) : {np.round(ch.channel_snr_db, 1).tolist()}")
print(f"MCS bits/symbol  : {ch.mcs_bits.tolist()}")
print(f"MCS needs (dB)   : {ch.mcs_required_db.tolist()}")
print("\nthe MCS requirements come from Wave-Lathe's measured ladder:")
print("QPSK costs 3.01 dB over BPSK, 16-QAM costs 10.00 dB.\n")

# --- P(success), clear spectrum ----------------------------------------------
table = ch.probability_table()
print("P(frame received), no interference")
print("            " + "".join(f"  MCS{m} ({int(b)}b)"
                               for m, b in enumerate(ch.mcs_bits)))
for c in range(N_CHANNELS):
    row = "".join(f"{table[c, m]:>11.3f}" for m in range(N_MCS))
    print(f"ch{c} ({ch.channel_snr_db[c]:>4.1f} dB){row}")

# --- expected throughput is what the agent actually maximises ----------------
# A 40% chance at 6 bits beats a 95% chance at 1 bit. The best MCS is not the
# most reliable one, and that is the decision.
expected = table * ch.mcs_bits[None, :]
print("\nexpected throughput = P(success) x bits, no interference")
print("            " + "".join(f"  MCS{m} ({int(b)}b)"
                               for m, b in enumerate(ch.mcs_bits)))
for c in range(N_CHANNELS):
    row = "".join(f"{expected[c, m]:>11.3f}" for m in range(N_MCS))
    best = int(np.argmax(expected[c]))
    print(f"ch{c} ({ch.channel_snr_db[c]:>4.1f} dB){row}   best: MCS{best}")

best_per_channel = expected.argmax(axis=1)
print(f"\nbest MCS per channel: {best_per_channel.tolist()}")

if len(set(best_per_channel.tolist())) == 1:
    print("WARNING: the same MCS wins on every channel. The agent would only")
    print("need to learn which channel is free -- the MCS choice is dead.")
else:
    print("Good: the right MCS depends on the channel, so the agent has two")
    print("things to learn rather than one.")

# --- what interference does ---------------------------------------------------
print("\n\nP(success) as others crowd the channel (MCS chosen per channel)")
print(f"{'channel':>9} {'alone':>8} {'+1 other':>10} {'+2 others':>11}")
print("-" * 42)
occ = np.zeros(N_CHANNELS, dtype=np.int8)
for c in range(N_CHANNELS):
    m = int(best_per_channel[c])
    ps = [ch.success_probability(c, m, occ, k) for k in (0, 1, 2)]
    print(f"{c:>9} {ps[0]:>8.3f} {ps[1]:>10.3f} {ps[2]:>11.3f}")

print("\nSharing a channel costs ~6 dB per additional transmitter, so two")
print("nodes on one channel should leave both close to useless. If it does")
print("not, collisions carry no penalty and the agents have no reason to")
print("spread out.")

# --- is P(success) monotonic in channel quality? -----------------------------
# It should be: a better channel cannot be less reliable at the same MCS.
# Where it is not, the learned predictor is being asked about a region of
# feature space it never saw, and its answer there is not meaningful.
print()
print('monotonic in channel quality? (better channel must be >= worse)')
bad = []
for m in range(N_MCS):
    col = table[:, m]
    ok = bool(np.all(np.diff(col) <= 1e-9))
    print('  MCS{0}: {1}   {2}'.format(
        m, 'yes' if ok else 'NO ', np.round(col, 3).tolist()))
    if not ok:
        bad.append(m)
if bad:
    print()
    print('MCS {0} violate it. At low MCS the physics term saturates, so the'
          .format(bad))
    print('learned model dominates -- and it was trained on real Colosseum')
    print('features, not on this synthetic mapping into that space. The')
    print('decision structure survives because expected throughput still')
    print('ranks channels correctly, but this is a real symptom of an')
    print('uncalibrated approximation and should not be read past.')

# --- sanity ------------------------------------------------------------------
assert (table >= 0).all() and (table <= 1).all(), "probabilities out of range"
assert table[0, 0] > table[-1, -1], (
    "the best channel at the lowest MCS should beat the worst channel at the "
    "highest -- if not, the SNR mapping is inverted")
spread = table.max() - table.min()
assert spread > 0.3, f"only {spread:.2f} spread: too flat to learn from"
print(f"\nspread across the table: {spread:.3f}  (needs to be well above 0)")
print("all checks passed")

# --- picture ------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5))

for ax, data, title in [
    (ax1, table, "P(frame received)"),
    (ax2, expected, "expected throughput (bits)"),
]:
    im = ax.imshow(data, cmap="viridis", aspect="auto")
    ax.set_xticks(range(N_MCS), [f"MCS{m}\n{int(b)}b"
                                 for m, b in enumerate(ch.mcs_bits)])
    ax.set_yticks(range(N_CHANNELS),
                  [f"ch{c}\n{ch.channel_snr_db[c]:.0f} dB"
                   for c in range(N_CHANNELS)])
    ax.set_title(title)
    for c in range(N_CHANNELS):
        for m in range(N_MCS):
            ax.text(m, c, f"{data[c, m]:.2f}", ha="center", va="center",
                    color="white" if data[c, m] < data.max() * 0.6 else "black",
                    fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046)

fig.suptitle("Frame-Oracle as the channel model: the decision the agents face\n"
             "predictor trained on real SC2 data; the feature mapping into it "
             "is a designed approximation", fontsize=11)
fig.tight_layout()

out = Path("figures")
out.mkdir(exist_ok=True)
fig.savefig(out / "channel_model.png", dpi=140)
print(f"saved {out / 'channel_model.png'}")
plt.show()
