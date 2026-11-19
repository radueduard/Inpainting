import numpy as np

from hmi_infill import prep, synthetic
from hmi_infill.ccsds122 import CodecParams
from hmi_infill.channel import Bernoulli, GilbertElliott, NoLoss
from hmi_infill.infill import InfillInput, Laplace, PreviousFrame, run
from hmi_infill.metrics import evaluate
from hmi_infill.pipeline import Trial, simulate, simulate_sequence


def _frames(n=3, size=128):
    out = []
    for f in synthetic.sequence(n, size, "magnetogram", seed=4):
        q, valid, info = prep.to_int(f, content="MAGNETOGRAM")
        out.append((q, valid, info))
    return out


def test_no_loss_is_exact():
    q, valid, _ = _frames(1)[0]
    t = simulate(q, codec=CodecParams(segment_blocks=32), channel=NoLoss(), valid=valid)
    assert np.array_equal(t.damaged, q) and np.array_equal(t.truth, q)
    assert t.meta["blocks_lost"] == 0


def test_sequence_with_losses_and_save(tmp_path):
    frames = _frames(3)
    trials = simulate_sequence([f[0] for f in frames], codec=CodecParams(segment_blocks=16),
                               channel=Bernoulli(0.1), seed=1, max_packet_len=256,
                               valid=[f[1] for f in frames])
    assert len(trials) == 3
    t = trials[1]
    assert t.meta["blocks_lost"] > 0
    assert np.array_equal(t.damaged[~t.mask_uncertain], t.truth[~t.mask_uncertain])
    t2 = Trial.load(t.save(tmp_path / "f1"))
    assert np.array_equal(t2.damaged, t.damaged) and t2.meta == t.meta

    x = InfillInput.from_trials(trials, 1, n_past=1)
    for m in (Laplace(), PreviousFrame()):
        filled = run(m, x)
        assert np.array_equal(filled[~x.mask], t.damaged[~x.mask])
        r = evaluate(filled, t.truth, x.mask, valid=t.valid, scale=0.1)
        assert np.isfinite(r["rmse"])


def test_gilbert_elliott_mean_loss():
    ch = GilbertElliott.from_mean(0.05, 8)
    pat = ch.loss_pattern(400_000, np.random.default_rng(0))
    assert abs(pat.mean() - 0.05) < 0.005
    runs = np.diff(np.flatnonzero(np.diff(np.r_[0, pat.astype(int), 0])))[::2]
    assert abs(runs.mean() - 8) < 0.8
