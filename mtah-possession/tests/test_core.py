"""Unit tests. None of them need the gated match data."""

import hashlib
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

from possession.constants import (A, BASELINE_DIR, D, H, N_FRAMES, OFFICIAL_DIR, X,  # noqa: E402
                                  graded_blocks, pm)
from possession.decode import fit_duration_model, hsmm_viterbi, majority_smooth, decode_blocks  # noqa: E402
from possession.external import derive_labels  # noqa: E402
from possession.match import Match  # noqa: E402

OFFICIAL_SHA256 = {
    "games.py": "3b2122794e9e0476e2b3a871d14880e5fa47c4feaeacf99caef054c2309076ce",
    "possession_metrics.py": "341bad48904b489d6e875397b27687f5d6a4d22a397f3dc8af3a00d48f16874d",
    "run_evaluation.py": "1c61564fe52499ba4af3006fb5eb4147b825e92bf516651add80de49d267b19f",
    "validate_submission.py": "07ac29499acbd337e1dc1de1832d63f6e1fa1d981e22f6ad08212529e2bbee14",
}


# -- the vendored official code is byte-identical to upstream ----------------------------------

@pytest.mark.parametrize("name,digest", sorted(OFFICIAL_SHA256.items()))
def test_official_files_unmodified(name, digest):
    with open(os.path.join(OFFICIAL_DIR, name), "rb") as f:
        # compare content, not line endings: Git for Windows may check files out with CRLF
        assert hashlib.sha256(f.read().replace(b"\r\n", b"\n")).hexdigest() == digest


def test_graded_blocks_match_docs():
    assert graded_blocks() == [(11185, 90652), (110808, 190198)]
    assert N_FRAMES == 199319


# -- the baseline port reproduces the committed files exactly ----------------------------------

@pytest.mark.parametrize("window", [25, 50, 125])
def test_majority_smooth_reproduces_committed_baselines(window):
    raw = pd.read_csv(os.path.join(BASELINE_DIR, "baseline_raw.csv"))["label"].values
    codes = np.where(raw == "H", 0, np.where(raw == "A", 1, -1)).astype(np.int8)
    ours = np.array(["H", "A", "D", "X"])[majority_smooth(codes, window)]
    ref = pd.read_csv(os.path.join(BASELINE_DIR, "baseline_smooth{}.csv".format(window)))
    assert np.array_equal(ours, ref["label"].values)


# -- EPTS loader -------------------------------------------------------------------------------

def test_epts_loader_on_synthetic_bundle(tmp_path):
    from fake_bundle import write_bundle
    from possession.epts import load_epts
    write_bundle(str(tmp_path), n_frames=3000)
    stem = os.path.join(str(tmp_path), "tactical", "BAR-ATH_tactical")
    m, rep = load_epts(stem + "_atd.xml", stem + "_atd.txt", pitch=(105.3, 68.0), n_frames=3000)
    assert rep["first_frame_id"] == 1 and rep["contiguous"] and rep["n_rows"] == 3000
    assert m.px.shape == (3000, 46)
    assert rep["n_home_slots"] == 23
    assert rep["player_channels"] == ["speed", "x", "y"]
    assert rep["ball_channels"] == ["speed", "x", "y"]
    assert rep["player_order_matches_layout"] is True
    # metres, NaN-preserving
    assert np.nanmax(m.px) <= 105.3 + 1e-3 and np.isnan(m.px).any()
    assert np.isnan(m.ball[:, 0]).any() and not np.all(np.isnan(m.ball[:, 0]))


def test_epts_layout_order_is_respected(tmp_path):
    """Swap two registers in the layout: values must follow the layout, not <Players> order."""
    from fake_bundle import write_bundle
    from possession.epts import load_epts
    write_bundle(str(tmp_path), n_frames=200)
    stem = os.path.join(str(tmp_path), "tactical", "BAR-ATH_tactical")
    xml = open(stem + "_atd.xml").read()
    r1 = '<PlayerChannelRef playerChannelId="P1_x"/><PlayerChannelRef playerChannelId="P1_y"/>' \
         '<PlayerChannelRef playerChannelId="P1_speed"/>'
    r2 = r1.replace("P1_", "P30_")
    xml = xml.replace(r1, "@@").replace(r2, r1).replace("@@", r2)
    open(stem + "_atd.xml", "w").write(xml)
    m, rep = load_epts(stem + "_atd.xml", stem + "_atd.txt", pitch=(100.0, 100.0))
    first = open(stem + "_atd.txt").readline().split(":")[1].split(";")
    g0 = first[0].split(",")
    assert rep["player_order_matches_layout"] is False
    if g0[0] != "NaN":
        assert np.isclose(m.px[0, 29], float(g0[0]) * 100.0, atol=1e-3)  # slot P30 <- 1st group


# -- event-derived labels follow the stated possession rules ----------------------------------

def _ev(rows):
    return pd.DataFrame(rows, columns=["team", "type", "subtype", "period", "start", "end"])


def test_derive_labels_rules():
    ev = _ev([
        (0, "SET PIECE", "KICK OFF", 1, 10, 10),
        (0, "PASS", "", 1, 10, 20),
        (1, "CHALLENGE", "AERIAL-WON", 1, 30, 30),    # duel alone: no change
        (0, "PASS", "", 1, 40, 50),
        (1, "RECOVERY", "INTERCEPTION", 1, 60, 60),   # change to away
        (1, "BALL OUT", "", 1, 80, 85),               # dead from end frame
        (0, "SET PIECE", "THROW IN", 1, 120, 120),    # restart, home
        (0, "PASS", "", 1, 121, 140),
    ])
    lab, blocks = derive_labels(ev, 200, frame_base=0, period_starts={1: 0})
    assert blocks == [(0, 141)]
    assert np.all(lab[0:10] == D)              # before kick-off, inside the block
    assert np.all(lab[10:60] == H)             # duel at 30 did not flip it
    assert np.all(lab[60:85] == A)
    assert np.all(lab[85:120] == D)
    assert np.all(lab[120:141] == H)
    assert np.all(lab[141:] == X)


def test_derive_labels_unconfirmed_stoppage_is_dropped():
    ev = _ev([
        (0, "SET PIECE", "KICK OFF", 1, 0, 0),
        (1, "FAULT RECEIVED", "", 1, 50, 50),
        (0, "PASS", "", 1, 60, 70),                # play on, no set piece: advantage
        (0, "PASS", "", 1, 90, 99),
    ])
    lab, _ = derive_labels(ev, 120, frame_base=0)
    assert not np.any(lab[0:100] == D)


# -- decoder -----------------------------------------------------------------------------------

def _toy_model(k=2):
    lab = np.array([H] * 300 + [A] * 300 + [D] * 100 + [H] * 300)
    return fit_duration_model(lab, [(0, len(lab))], k=k)


def _logp(seq, conf=0.9):
    p = np.full((len(seq), 3), (1 - conf) / 2)
    p[np.arange(len(seq)), seq] = conf
    return np.log(p)


def test_hsmm_removes_one_frame_blip_keeps_real_switch():
    seq = np.array([H] * 200 + [A] + [H] * 200 + [A] * 200)
    out = hsmm_viterbi(_logp(seq), _toy_model(), alpha=1.0)
    assert np.all(out[:401] == H)
    assert np.mean(out[401:] == A) > 0.95


def test_hsmm_keeps_short_genuine_possession_with_strong_evidence():
    seq = np.array([H] * 300 + [A] * 30 + [H] * 300)
    out = hsmm_viterbi(_logp(seq, conf=0.995), _toy_model(), alpha=1.0)
    assert np.mean(out[300:330] == A) > 0.8


def test_decode_blocks_never_crosses_halves_and_fills_x():
    T = 1000
    lp = _logp(np.array([H] * T))
    out = decode_blocks(lp, [(100, 400), (600, 900)], T, _toy_model(), 1.0)
    assert np.all(out[:100] == X) and np.all(out[400:600] == X) and np.all(out[900:] == X)
    assert np.all(out[100:400] == H)


def test_hsmm_is_team_symmetric():
    rng = np.random.RandomState(0)
    p = rng.dirichlet([1, 1, 1], size=500)
    out = hsmm_viterbi(np.log(p), _toy_model(), 0.7)
    out_sw = hsmm_viterbi(np.log(p[:, [1, 0, 2]]), _toy_model(), 0.7)
    sw = out_sw.copy()
    sw[out_sw == H], sw[out_sw == A] = A, H
    assert np.array_equal(out, sw)


# -- abstention semantics of the official scorer (documents the X design decision) -------------

def test_x_is_forward_filled_for_spellf1():
    gt = np.array([X] * 5 + [H] * 50 + [A] * 50 + [X] * 5, dtype=np.int8)
    pred = gt.copy()
    pred[50:80] = X            # going quiet across the true H->A hand-over at 55
    r = pm.evaluate(gt, pred)
    filled = pm.fill_abstentions(pred, pm.valid_runs(gt))
    assert np.all(filled[50:80] == H)         # the old label bleeds into the new possession
    assert r["spell_f1_mean"] < 1.0           # ... and costs IoU on both spells
    assert r["macro_f1"] < 1.0


# -- features ----------------------------------------------------------------------------------

def _toy_match(T=600, seed=0):
    rng = np.random.RandomState(seed)
    px = (rng.rand(T, 22) * 100).astype(np.float32)
    py = (rng.rand(T, 22) * 60).astype(np.float32)
    px[:, 0], px[:, 11] = 2.0, 100.0      # keepers: home defends x=0
    ball = np.stack([np.linspace(10, 90, T), np.full(T, 30.0)], 1).astype(np.float32)
    ball[100:200] = np.nan
    team = np.array([0] * 11 + [1] * 11, dtype=np.int8)
    lab = np.array([H] * T, dtype=np.int8)
    return Match("toy", 25, (105.0, 68.0), ball, px, py, team, [(0, T)], lab).validate()


def test_missing_ball_gives_nan_not_zero():
    from possession.features import frame_features
    names, F = frame_features(_toy_match())
    for n in ("d1_h", "ball_speed", "ball_x", "nearest_is_home"):
        col = F[:, names.index(n)]
        assert np.all(np.isnan(col[110:190])), n
    assert np.all(F[110:190, names.index("ball_vis")] == 0)
    assert np.all(F[110:190, names.index("ball_gap_len")] > 0)


def test_team_swap_mirrors_features():
    from possession.features import frame_features, swap_teams
    m = _toy_match()
    n1, F1 = frame_features(m)
    n2, F2 = frame_features(swap_teams(m))
    assert n1 == n2
    i, j = n1.index("d1_h"), n1.index("d1_a")
    np.testing.assert_allclose(F1[:, i], F2[:, j], equal_nan=True, rtol=1e-5)
    k = n1.index("control_sign")
    np.testing.assert_allclose(F1[:, k], -F2[:, k], equal_nan=True, atol=1e-6)


# -- submission file contract ------------------------------------------------------------------

def test_submission_contract_and_official_validator(tmp_path):
    from possession.pipeline import sha256, write_submission
    codes = np.full(N_FRAMES, X, dtype=np.int8)
    for a, b in graded_blocks():
        codes[a:b] = np.where(np.arange(b - a) % 900 < 450, H, A)
    p1 = write_submission(str(tmp_path / "T.csv"), codes)
    p2 = write_submission(str(tmp_path / "T2.csv"), codes)
    assert sha256(p1) == sha256(p2)                       # deterministic bytes
    df = pd.read_csv(p1, dtype={"label": str})
    assert list(df.columns) == ["frame", "label"]
    assert len(df) == 199319
    assert np.array_equal(df["frame"].values, np.arange(199319))
    assert set(df["label"].unique()) <= {"H", "A", "D", "X"}
    assert not df["label"].isna().any()
    out = subprocess.run([sys.executable, os.path.join(OFFICIAL_DIR, "validate_submission.py"), p1],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr
