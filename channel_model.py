"""Frame-Oracle's predictor as this environment's channel model.

Stage A and B decide success with a rule: alone on the channel means the frame
lands. That is a coin flip dressed as physics. Here the decision comes from
the model trained in Frame-Oracle on 6.5M real DARPA Colosseum frames, so
success depends on SNR, modulation and coding scheme, bandwidth, and the
spectrum around the transmission.

WHAT IS REAL AND WHAT IS NOT
----------------------------
Read this before trusting any number that comes out of an environment using
this class.

REAL: the predictor. Trained on measured SC2 scrimmage data, AUC 0.68-0.71 on
links it had never seen, replicated across two independent scrimmages. The
SHAPE of the relationship it encodes -- how success falls away as SNR drops or
MCS climbs -- comes from real radios.

NOT REAL: the inputs. Frame-Oracle's features are standardised using
statistics from the SC2 dataset. This environment is a synthetic four-channel
world that is not Colosseum, so the mapping from "node 2 is on channel 1 at
MCS 3 while channel 0 is busy" onto that standardised feature space is a
DESIGNED APPROXIMATION. Nobody calibrated it. The numbers below were chosen to
be plausible, not measured.

So: the tradeoff the agents face has a realistic shape, and the absolute
success rates do not mean anything in particular. This is a better environment
than a coin flip and a worse one than a testbed, and results should be read
that way.
"""

from pathlib import Path

import numpy as np

# Frame-Oracle's export. Feature order is fixed by that model and must not be
# rearranged: snr, mcs, center_freq, bandwidth, then 16 PSD bins.
DEFAULT_MODEL = (Path("..") / "Frame-Oracle" / "export"
                 / "scrimmage5_by_link.onnx")

N_FEATURES = 20
N_PSD_BINS = 16


class FrameOracleChannel:
    """Predicts P(frame received) for a transmission in this environment."""

    def __init__(self, n_channels=4, n_mcs=4, model_path=DEFAULT_MODEL,
                 seed=None):
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"{model_path} not found. Run Frame-Oracle's export_onnx.py "
                f"first: uv run export_onnx.py --checkpoint scrimmage5_by_link"
            )

        import onnxruntime as ort

        self.session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"])
        self.n_channels = n_channels
        self.n_mcs = n_mcs
        self.rng = np.random.default_rng(seed)

        # Channel quality in dB, best first. Spread chosen so the MCS decision
        # is non-trivial: the top channel supports the top MCS and the bottom
        # one does not.
        self.channel_snr_db = np.linspace(18.0, 2.0, n_channels)

        # Throughput multiplier per MCS, from Wave-Lathe's modulation ladder.
        # Bits per symbol: 1, 2, 4, 6. Higher pays more when it works.
        self.mcs_bits = np.array([1.0, 2.0, 4.0, 6.0])[:n_mcs]

        # SNR each MCS needs, also from Wave-Lathe: QPSK costs 3.01 dB over
        # BPSK and 16-QAM costs 10.00 dB, both measured there rather than
        # assumed here.
        self.mcs_required_db = np.array([3.0, 6.0, 13.0, 19.0])[:n_mcs]

    # -- feature construction -------------------------------------------------
    def _standardise_snr(self, snr_db):
        """Map an SNR in dB onto the standardised scale the model expects.

        The SC2 features have mean 0 and unit variance. A 10 dB centre and a
        6 dB spread put a realistic operating range across roughly +-2 sigma,
        which is where the training data lived. Chosen, not fitted.
        """
        return (snr_db - 10.0) / 6.0

    def _features(self, channel, mcs, occupancy, interferers_on_channel):
        """Build one 20-feature vector for a transmission.

        Args:
            channel: index of the channel used.
            mcs: index into mcs_bits.
            occupancy: binary array, which channels are busy.
            interferers_on_channel: how many others share this channel.
        """
        # Interference costs SNR. Each additional occupant on the same channel
        # takes roughly 6 dB, so two transmitters means neither gets through.
        snr_db = self.channel_snr_db[channel] - 6.0 * interferers_on_channel

        f = np.zeros(N_FEATURES, dtype=np.float32)
        f[0] = self._standardise_snr(snr_db)
        # MCS and channel index centred on their own ranges.
        f[1] = (mcs - (self.n_mcs - 1) / 2) / max((self.n_mcs - 1) / 2, 1)
        f[2] = (channel - (self.n_channels - 1) / 2) / \
               max((self.n_channels - 1) / 2, 1)
        f[3] = 0.0  # bandwidth held fixed; not part of the action space

        # The 16 PSD bins describe the spectrum. Spread the channel occupancy
        # across them so a busy neighbour shows up as raised power nearby --
        # the same quantity Frame-Oracle's real PSD bins carried.
        bins_per_channel = max(N_PSD_BINS // self.n_channels, 1)
        psd = np.zeros(N_PSD_BINS, dtype=np.float32)
        for ch in range(self.n_channels):
            lo = ch * bins_per_channel
            hi = min(lo + bins_per_channel, N_PSD_BINS)
            psd[lo:hi] = 1.0 if occupancy[ch] else -0.5
        f[4:] = psd
        return f

    # -- prediction ------------------------------------------------------------
    def success_probability(self, channel, mcs, occupancy,
                            interferers_on_channel):
        """P(frame received), from the trained predictor."""
        f = self._features(channel, mcs, occupancy,
                           interferers_on_channel)[None, :]
        logit = self.session.run(["logit"], {"features": f})[0].ravel()[0]
        p_model = 1.0 / (1.0 + np.exp(-logit))

        # The predictor never saw this synthetic world, so on its own it does
        # not know that MCS 3 is impossible at 2 dB. That physical constraint
        # comes from Wave-Lathe's measured ladder and is applied on top: a
        # transmission below its MCS's required SNR fails steeply.
        snr_db = self.channel_snr_db[channel] - 6.0 * interferers_on_channel
        margin_db = snr_db - self.mcs_required_db[mcs]
        p_physics = 1.0 / (1.0 + np.exp(-margin_db / 2.0))

        # Both must be satisfied: the learned model supplies the messy part,
        # the ladder supplies the hard physical limit.
        return float(p_model * p_physics)

    def transmit(self, channel, mcs, occupancy, interferers_on_channel):
        """Sample an outcome. Returns (received, throughput, p_success)."""
        p = self.success_probability(channel, mcs, occupancy,
                                     interferers_on_channel)
        received = bool(self.rng.random() < p)
        throughput = float(self.mcs_bits[mcs]) if received else 0.0
        return received, throughput, p

    # -- inspection ------------------------------------------------------------
    def probability_table(self):
        """P(success) for every (channel, MCS) pair with no interference.

        Worth printing before training. If every cell is near 1 the agent has
        nothing to decide; if every cell is near 0 it cannot learn anything.
        The interesting case is a diagonal band -- good channels supporting
        aggressive MCS, poor channels not.
        """
        occ = np.zeros(self.n_channels, dtype=np.int8)
        return np.array([[self.success_probability(c, m, occ, 0)
                          for m in range(self.n_mcs)]
                         for c in range(self.n_channels)])
