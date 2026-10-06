#!/usr/bin/env python3
"""Focused behaviour checks for skill_calc. Run: python3 test_skill_calc.py"""
import os
import sys
import tempfile
from unittest.mock import patch

import skill_calc as S
import difficulty_model as D

_TMP = tempfile.mkdtemp(prefix="maniascope-test-")
_N = [0]


def chart(notes, keys=7, mode=3, shuffle=False, timing=("0,500,4,1,0,100,1,0",)):
    """notes: [(ms, col)] or [(ms, col, end_ms)] → path of a generated .osu"""
    lines = []
    for n in notes:
        x = int((n[1] + 0.5) * 512 / keys)
        if len(n) > 2:
            lines.append(f"{x},192,{n[0]},128,0,{n[2]}:0:0:0:0:")
        else:
            lines.append(f"{x},192,{n[0]},1,0,0:0:0:0:")
    if shuffle:
        lines.reverse()
    _N[0] += 1
    path = os.path.join(_TMP, f"{_N[0]}.osu")
    with open(path, "w") as fh:
        fh.write(f"osu file format v14\n\n[General]\nMode: {mode}\n\n[Metadata]\nTitle:t\n"
                 f"Version:v\n\n[Difficulty]\nCircleSize:{keys}\n\n[TimingPoints]\n"
                 + "\n".join(timing) + "\n\n[HitObjects]\n"
                 + "\n".join(lines) + "\n")
    return path


def rate(notes, r=1.0, **kw):
    return S.compute(S.parse_osu(chart(notes, **kw)), r)["scores"]


def top(notes, n=1, **kw):
    return S.ranked(S.compute(S.parse_osu(chart(notes, **kw)), 1.0))[:n]


def stream(n, gap, keys=7, start=1000):
    return [(start + gap * i, i % keys) for i in range(n)]


def test_parsing_and_rate():
    base = stream(1200, 100)
    a = rate(base)
    assert a["overall"] > 1 and all(v == v and v >= 0 for v in a.values())
    # simultaneous-line order and absolute offset do not matter
    chords = [(1000 + 150 * i, c) for i in range(300) for c in (i % 3, 4, 6)]
    assert rate(chords) == rate(chords, shuffle=True)
    assert abs(rate(stream(1200, 100, start=91234))["overall"] - a["overall"]) < 1e-9
    # rate applied exactly once: 100 ms stream at 1.25x == 80 ms stream at 1.0x, plus the rate factor
    # This is the physical core invariant, before the fitted score-link transform.
    with patch.object(D, "parameters", return_value={}):
        assert abs(rate(base, 1.25)["overall"] / 1.25 ** S.RATE_G - rate(stream(1200, 80))["overall"]) < 1e-6
    # every keymode parses; non-mania is explicit
    for k in range(4, 11):
        assert rate(stream(400, 100, keys=k), keys=k)["overall"] > 0
    try:
        rate(base, mode=0)
        raise AssertionError("std chart accepted")
    except S.ChartError:
        pass
    # short charts are rated, not zeroed; one note is simply 0
    lns = [(1000 + 100 * i, i % 7, 1050 + 100 * i) for i in range(7)]
    assert rate(lns)["overall"] > 0 and rate([(1000, 0)])["overall"] == 0


def test_rate_sweep_is_smooth_and_uncapped():
    path = chart([(1000 + 120 * i, c) for i in range(900) for c in ((i % 7,) if i % 4 else (i % 7, (i + 3) % 7))])
    ch = S.parse_osu(path)
    prev = 0
    for r100 in range(70, 201, 2):
        v = S.compute(ch, r100 / 100)["scores"]["overall"]
        # 0.5 s bins shift under the notes as rate changes: allow a sub-display wobble. The learned
        # stages can dip slightly: on 48 sampled ranked charts the pre-tree calculator already dips
        # >0.3 % in 22 of 3,120 steps (worst 4.5 %); soft tree splits add 6 (hard ones added 46).
        # This chart dips 1.1 % at 1.70→1.72× (2026-10-02).
        assert prev * 0.985 < v < prev * 1.07 or prev == 0, (r100, prev, v)
        prev = v


def test_horizon_units_match_rate_adjusted_overall():
    import math
    import recdata
    path = chart(stream(1200, 100))
    base = S.parse_osu(path)
    sped = S.parse_osu(chart(stream(1200, 80)))
    with patch.object(D, 'parameters', return_value={}):
        a, b = S.compute(base, 1.25), S.compute(sped, 1.)
        feature = recdata.chart_feats(path, [1.25], analyses={1.25:a})[1.25]
    for horizon, value in a['levels'].items():
        assert math.isclose(value/1.25**S.RATE_G, b['levels'][horizon], rel_tol=1e-10)
    assert math.isclose(a['levels'][120.]/a['scores']['overall'],
                        b['levels'][120.]/b['scores']['overall'], rel_tol=1e-10)
    assert math.isclose(feature['stam'], a['levels'][120.]/a['scores']['overall']/1.25**S.RATE_G,
                        rel_tol=1e-10)


def test_recovery_and_burst_vs_sustained():
    base = stream(1200, 100)
    rested = [(t + (60000 if i >= 600 else 0), c) for i, (t, c) in enumerate(base)]
    a, b = rate(base), rate(rested)
    assert b["stamina"] < a["stamina"] and b["overall"] <= a["overall"]
    # same pattern: 3 s burst in a 2 min map < 3 min sustained; easy padding does not erase a bottleneck
    burst, long = rate(stream(30, 100) + [(125000, 0)]), rate(stream(1800, 100))
    assert burst["overall"] < 0.9 * long["overall"]
    padded = stream(600, 400) + stream(100, 70, start=300000)
    assert rate(padded)["overall"] > 1.3 * rate(stream(600, 400))["overall"]
    assert S.compute(S.parse_osu(chart(stream(60, 70) + [(125000, 0)])), 1.0)["horizon"] == 2.0
    # Stamina is endurance, not a relabelled overall: a short burst has little, a long even run
    # nearly all of its rating; rest never adds to it
    short, marathon = rate(stream(60, 100)), rate(stream(3600, 100))
    assert short["stamina"] < 0.6 * short["overall"] and marathon["stamina"] > 0.95 * marathon["overall"]
    assert rate([(t + (60000 if i >= 1800 else 0), c) for i, (t, c) in enumerate(stream(3600, 100))])["stamina"] \
        < marathon["stamina"]


def test_hand_patterns():
    # same row rate, different hands: one-hand trill > 7-key staircase roll
    trill = [(1000 + 100 * i, i % 2) for i in range(600)]
    assert rate(trill)["overall"] > 1.2 * rate(stream(600, 100))["overall"]
    # single-column jack is Jack (7K) / Jackspeed (4K); repeated chords are chordjack
    assert top([(1000 + 150 * i, 2) for i in range(400)]) == ["jack"]
    assert top([(1000 + 150 * i, 2) for i in range(400)], keys=4) == ["jackspeed"]
    # an anchor with the other hand in between is still a jack in 7K (the hand repeats)
    anchor = [(1000 + 120 * i, 1 if i % 2 else 5 + (i // 2) % 2) for i in range(800)]
    assert top(anchor, 3)[0] == "jack" or "jack" in top(anchor, 2)
    cj = [(1000 + 160 * i, c) for i in range(400) for c in ((0, 2, 4, 6) if i % 2 else (0, 3, 4, 5))]
    assert top(cj) == ["chordjack"] and "bracket" not in top(cj, 3)
    # chords without repeats are chordstream, fast light rice is delay (7777's Delay packs)
    hs = rate([(1000 + 160 * i, c) for i in range(400) for c in ((0, 2, 5) if i % 2 else (1, 3, 6))])
    assert hs["chordstream"] > hs["chordjack"]
    delay = [(1000 + 60 * i, c) for i in range(1500) for c in ((i * 3) % 7,) + (((i * 3 + 4) % 7,) if i % 3 == 0 else ())]
    assert top(delay) == ["delay"]
    # the same texture slower is Stream, not Delay
    slow = [(1000 + 100 * i, c) for i in range(1500) for c in ((i * 3) % 7,) + (((i * 3 + 4) % 7,) if i % 3 == 0 else ())]
    assert top(slow) == ["stream"]
    # bracket: each hand trills between disjoint finger groups, one a chord (13 / 2)
    br = [(1000 + 120 * i, c) for i in range(800) for c in ((0, 2, 4, 6) if i % 2 else (1, 5))]
    assert "bracket" in top(br, 3) and rate(br)["bracket"] > 2 * rate(stream(800, 120))["bracket"]
    # a smooth roll is not technical; broken rhythm is
    assert rate(stream(600, 100))["technical"] < 0.25
    swing = [(1000 + 100 * i + (37 if i % 3 == 1 else 0), i % 7) for i in range(600)]
    assert rate(swing)["technical"] > 1 and top(swing, 3).count("technical")
    # smooth fast rolls back onto the 1/4 grid and tap -> LN accents are not technical
    rolls = [(1000 + 300 * i + dt, c) for i in range(300) for dt, c in ((0, 0), (19, 1), (38, 2), (57, 3), (150, 5), (225, 6))]
    assert rate(rolls)["technical"] < 0.25
    accent = [n for i in range(500) for n in ((1000 + 150 * i, (i * 3) % 7), (1075 + 150 * i, (i * 3) % 7, 1135 + 150 * i))]
    assert rate(accent)["technical"] < 0.25


def test_sv_reading():
    base = stream(1200, 100)
    plain = rate(base)
    # decorative lines (SV 1, or changes after the last note) change nothing
    harmless = ("0,500,4,1,0,100,1,0", "5000,-100,4,1,0,100,0,0", "999000,-25,4,1,0,100,0,0")
    assert rate(base, timing=harmless) == plain
    # scroll jumping between 0.5x and 2x while notes approach is reading work
    active = ("0,500,4,1,0,100,1,0",) + tuple(f"{1000 + 400 * k},{-200 if k % 2 else -50},4,1,0,100,0,0" for k in range(300))
    sv = S.compute(S.parse_osu(chart(base, timing=active)), 1.0)
    assert sv["scores"]["overall"] > 1.05 * plain["overall"] and sv["scores"]["sv"] > 0 and sv["sv"] == "included"
    # Constant Speed: the same chart without scroll changes
    cs = S.compute(S.parse_osu(chart(base, timing=active)), 1.0, sv=False)
    assert abs(cs["scores"]["overall"] - plain["overall"]) < 1e-9 and cs["sv"] == "ignored"


def test_ln_interaction():
    taps = [(1000 + 300 * i, c) for i in range(300) for c in (0, 1)]
    sync = [(t, c, t + 250) for t, c in taps]                       # released together
    stag = [(t, c, t + 250 - 70 * c) for t, c in taps]              # awkward 70 ms stagger
    a, b, c = rate(taps), rate(sync), rate(stag)
    assert a["ln"] == 0 and a["overall"] < b["overall"] < c["overall"]
    # a long simple hold is not harder than a shorter one
    assert abs(rate([(1000 + 3000 * i, 3, 3400 + 3000 * i) for i in range(40)])["overall"]
               - rate([(1000 + 3000 * i, 3, 2400 + 3000 * i) for i in range(40)])["overall"]) < 0.12
    # tapping under a same-hand hold costs more than under an other-hand hold
    hold = [(1000 + 4000 * i, 0, 4900 + 4000 * i) for i in range(30)]
    same = hold + [(1000 + 200 * i, 1 + i % 2) for i in range(600) if (200 * i) % 4000 > 200]
    other = hold + [(1000 + 200 * i, 5 + i % 2) for i in range(600) if (200 * i) % 4000 > 200]
    assert rate(same)["ln"] > 1.3 * rate(other)["ln"]


def test_rate_from_mods():
    from skillsets import rate_from_mods as r
    assert r({"array": []}) == (1.0, "")
    assert r({"array": [{"acronym": "HT", "settings": {"speed_change": 0.98}}]})[0] == 0.98
    assert r({"array": [{"acronym": "DT", "settings": {"speed_change": 1.02}}]})[0] == 1.02
    assert r({"array": [{"acronym": "NC"}]})[0] == 1.5 and r({"array": [{"acronym": "DC"}]})[0] == 0.75
    # unknown or variable speed is never silently No Mod
    assert r(None)[0] is None and r({})[0] is None
    assert r({"array": [{"acronym": "WU"}]})[0] is None



def names(notes, **kw):
    return [e["name"] for e in S.card(S.compute(S.parse_osu(chart(notes, **kw)), 1.0))]


def test_mashed_rolls_read_as_vibro_not_stream():
    # "Vibro Dansen": a 4-key roll 21 ms apart, short holds, each column every 83 ms, is mashed
    roll = [(1000 + 21 * i, (2, 3, 0, 1)[i % 4], 1000 + 21 * i + 42) for i in range(1600)]
    r = S.compute(S.parse_osu(chart(roll, keys=4)), 1.0)
    sc = r["scores"]
    assert S.card(r)[0]["name"] == "Vibro" and sc["vibro"] > 2 * max(sc["stream"], sc["dump"], sc["ln"])
    # the same roll at a hittable 60 ms per row stays rice
    slow = [(1000 + 60 * i, (2, 3, 0, 1)[i % 4]) for i in range(600)]
    assert S.compute(S.parse_osu(chart(slow, keys=4)), 1.0)["scores"]["vibro"] < .1


def test_card_taxonomy():
    # a long single-column run is a Longjack (it replaces Jack); a lone pair inside rice is a Minijack
    assert names([(1000 + 150 * i, 2) for i in range(400)])[0] == "Longjack"
    mini = [(1000 + 110 * i, (i // 2) * 3 % 7 if i % 4 < 2 else (i * 3) % 7) for i in range(1200)]
    assert "Minijack" in names(mini) and "Longjack" not in names(mini)
    # tap -> hold in the same column (1/4 apart) is Shield; release -> re-press right away is Inverse
    shield = [n for i in range(300) for n in ((1000 + 400 * i, i % 7), (1100 + 400 * i, i % 7, 1350 + 400 * i))]
    assert "Shield" in names(shield)
    inverse = [(1000 + 300 * i, c, 1250 + 300 * i) for i in range(300) for c in (1, 5)]
    assert "Inverse" in names(inverse)
    # broken rhythm reads as Tech <pattern>, never a bare Technical; a component never repeats
    swing = [(1000 + 100 * i + (37 if i % 3 == 1 else 0), i % 7) for i in range(600)]
    card = S.card(S.compute(S.parse_osu(chart(swing)), 1.0))
    assert card[0]["name"].startswith("Tech ") and "Technical" not in [e["name"] for e in card]
    parts = [p for e in card for p in e["parts"]]
    assert len(parts) == len(set(parts))
    for n in list(S.NAMES.values()) + list(S.TECH_NAMES.values()) + list(S.STAMINA_NAMES.values()):
        assert len(n) <= 16 and len(n.split()) <= 3, n


def test_chord_trills_and_dump():
    # names only: overall comes from the same demand whatever the rows are called
    def rows(pat, gap, keys, n=800):
        return [(1000 + gap * i, c) for i in range(n) for c in pat[i % len(pat)]]
    assert top(rows([(0, 1), (2, 3)], 90, 4), keys=4) == ["jumptrill"]
    assert top(rows([(0, 2), (1, 3)], 90, 4), keys=4) == ["splittrill"]
    assert top(rows([(0, 1, 2), (4, 5, 6)], 90, 7)) == ["jumptrill"]
    assert "splittrill" not in S.skills(7) and "dump" not in S.skills(7)      # 7K: Bracket, Delay
    assert top(rows([(0, 2, 4), (1, 3, 5)], 90, 7))[0] in ("chordstream", "bracket")
    # A B A inside jumpstream is not a trill; A B A B is
    js = rows([(0, 2), (1, 3), (0, 2), (1,), (3,), (0, 1), (2,), (3,)], 90, 4)
    assert rate(js, keys=4)["splittrill"] == 0
    # 4K Dump: fast one-finger-per-hand rice, overlapping the stream family (Delay's rule)
    fast, slow = rate(stream(1200, 45, keys=4), keys=4), rate(stream(1200, 110, keys=4), keys=4)
    assert fast["dump"] >= 0.9 * fast["stream"] and slow["dump"] == 0


def test_changing_jumptrill():
    from unittest.mock import patch

    def rows(pat, gap=90, n=400):
        return [(1000 + gap * i, c) for i in range(n) for c in pat[i % len(pat)]]

    changing = [(0, 1), (4, 5), (1, 2), (5, 6)]  # 12 / 56 / 23 / 67
    for keys in (7, 8):
        ch = S.parse_osu(chart(rows(changing), keys=keys))
        # Recognition does not change physical demand. The score calibration is
        # explicitly pattern-sensitive, and is checked separately.
        with patch.object(D, "parameters", return_value={}):
            result = S.compute(ch)
        assert S.ranked(result)[0] == "jumptrill"
        with patch.object(S, "TRILL_GAP", 0), patch.object(D, "parameters", return_value={}):
            plain = S.compute(ch)
        for key in ("overall", "stamina"):
            assert result["scores"][key] == plain["scores"][key]
        for key in ("levels", "horizon", "timeline"):
            assert result[key] == plain[key]
    assert top(rows([(0, 1, 2), (4, 5), (1, 2), (4, 5, 6)])) == ["jumptrill"]
    # Alternating hands already costs far less than repeating chords at the same row speed.
    chordjack = [(0, 1), (0, 1), (1, 2), (1, 2)]
    assert rate(rows(chordjack, gap=100))["overall"] > 2 * rate(rows(changing, gap=100))["overall"]
    for pat in ([(0, 1), (1, 2), (4, 5), (5, 6)],  # same hand twice
                [(0, 1), (4,), (1, 2), (5, 6)],    # a single note
                [(0, 4), (1, 5), (2, 6), (1, 5)]):  # chords span both hands
        assert rate(rows(pat))["jumptrill"] == 0
    assert rate(rows(changing, n=3))["jumptrill"] == 0
    assert rate(rows(changing, gap=220))["jumptrill"] == 0


def test_fast_changing_chords_lose_mirror_discount():
    from unittest.mock import patch

    changing = [(1000 + 110 * i, c) for i in range(400)
                for c in ((0, 1, 4, 5) if i % 2 else (1, 2, 5, 6))]
    fixed = [(1000 + 110 * i, c) for i in range(400) for c in (0, 1, 4, 5)]
    slow = [(t * 2, c) for t, c in changing]
    a, b = rate(changing)["overall"], rate(fixed)["overall"]
    s = rate(slow)["overall"]
    with patch.object(S, "MIRROR", 1.0):
        assert rate(changing)["overall"] == a   # fast partial repeats are separate movements
        assert rate(fixed)["overall"] > b      # complete repeats still share one bounce
        assert rate(slow)["overall"] > s       # slower changing chords keep the discount


def res(notes, r=1.0, **kw):
    return S.compute(S.parse_osu(chart(notes, **kw)), r)


def dump(gaps, n=900, keys=4, rows=None):
    """one note per row, hands alternating irregularly, gaps (ms) cycling"""
    cols = rows or [0, 3, 1, 2, 3, 0, 2, 1, 0, 2, 3, 1]
    t, out = 1000, []
    for i in range(n):
        out.append((t, cols[i % len(cols)]))
        t += gaps[i % len(gaps)]
    return out


def test_rhythm_tech():
    # 1/4 <-> 1/6 switching inside fast rice is rhythm tech; the same rice on one pulse is not
    mixed, even = res(dump([40, 40, 40, 60, 60]), keys=4)["scores"], res(dump([50]), keys=4)["scores"]
    assert mixed["rhythmtech"] > 0.8 * mixed["overall"] and even["rhythmtech"] == 0
    # a whole multiple of the pulse stays on it (50 50 50 150); a long roll changing speed is one sweep
    assert res(dump([50, 50, 50, 150]), keys=4)["scores"]["rhythmtech"] == 0
    roll = [(1000 + sum((25, 37, 50)[(k // 14) % 3] for k in range(i)), 6 - i % 7) for i in range(900)]
    assert res(roll)["scores"]["rhythmtech"] < 0.2 * res(roll)["scores"]["overall"]
    # scroll changes do not move Tech: the same notes with and without SV
    sv = ("0,500,4,1,0,100,1,0",) + tuple(f"{2000 * i},-{50 + 50 * (i % 3)},4,1,0,100,0,0" for i in range(1, 30))
    with patch.object(D, "parameters", return_value={}):
        a, b = res(dump([40, 40, 40, 60, 60]), keys=4)["scores"], res(dump([40, 40, 40, 60, 60]), keys=4, timing=sv)["scores"]
    assert b["sv"] > 0 and abs(a["rhythmtech"] - b["rhythmtech"]) < 1e-6 * a["rhythmtech"]


def test_pattern_tech():
    # a minijack inserted into fast moving rice is pattern tech; the same rice without it, or every note
    # doubled (a template), is not
    base = [0, 2, 1, 3, 0, 3, 1, 2]
    ins = [0, 2, 1, 3, 3, 0, 1, 2, 0, 3, 1, 2]              # 3 3: a repeat entered and left from motion
    plain, inserted = res(dump([60], rows=base), keys=4)["scores"], res(dump([60], rows=ins), keys=4)["scores"]
    doubled = res(dump([60], rows=[0, 0, 2, 2, 1, 1, 3, 3]), keys=4)["scores"]
    assert plain["patterntech"] == 0 and doubled["patterntech"] == 0
    assert inserted["patterntech"] > 0.8 * inserted["overall"]
    # Smooth rice <-> jumptrill transitions alone are NOT technical. They must
    # have independent repeated-column/rhythm-control evidence at the boundary.
    sw = [c for i in range(160) for c in ([(0,), (2,), (1,), (3,)] if i % 2 else [(0, 1), (2, 3), (0, 1), (2, 3)])]
    jt = [(0, 1), (2, 3)] * 320
    rows = lambda pat: [(1000 + 70 * i, c) for i, cs in enumerate(pat) for c in cs]
    assert res(rows(sw), keys=4)["scores"]["patterntech"] < .1 * res(rows(sw), keys=4)["scores"]["overall"]
    assert res(rows(jt), keys=4)["scores"]["patterntech"] == 0


def test_hand_descriptors():
    # 1H Trill: one hand carrying the rows alternating two fingers; two-hand alternation is not
    one = [(1000 + 90 * i, i % 2) for i in range(600)]
    two = [(1000 + 90 * i, (0, 3, 1, 2)[i % 4]) for i in range(600)]
    assert res(one, keys=4)["scores"]["trill1h"] > 0.9 * res(one, keys=4)["scores"]["overall"]
    assert res(two, keys=4)["scores"]["trill1h"] == 0
    # Anchor: one column every other row around the other three; an even roll is not
    anc = [(1000 + 80 * i, 1 if i % 2 == 0 else (0, 2, 3)[(i // 2) % 3]) for i in range(600)]
    roll = [(1000 + 80 * i, i % 4) for i in range(600)]
    assert res(anc, keys=4)["scores"]["anchor"] > 0.9 * res(anc, keys=4)["scores"]["overall"]
    assert res(roll, keys=4)["scores"]["anchor"] == 0


def test_tech_card():
    # tech inside the dump passage: Tech Dump replaces Dump; Technical, Pattern/Rhythm Tech never bare;
    # no part twice; a partial Tech X never sits next to its own X
    r = res(dump([40, 40, 40, 60, 60]), keys=4)
    names = [e["name"] for e in S.card(r)]
    assert names[0] == "Tech Dump" and "Dump" not in names, names
    for n in ("Technical", "Pattern Tech", "Rhythm Tech"):
        assert n not in names
    parts = [p for e in S.card(r) for p in e["parts"]]
    assert len(parts) == len(set(parts))
    assert r["tech_where"] and r["tech_where"][0]["rhythm"] > 0.9
    # tech in one half, harder plain dump in the other: Dump keeps its entry, no Tech Dump beside it
    half = dump([60], n=400) + [(t + 30000, c) for t, c in dump([40, 40, 40, 60, 60], n=300)]
    names = [e["name"] for e in S.card(res(half, keys=4))]
    assert not ("Dump" in names and "Tech Dump" in names), names


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and (len(sys.argv) < 2 or name in sys.argv[1:]):
            fn()
            print("ok", name)




def test_shipped_dan_table_is_stamped_for_this_calculator():
    # A calculator change without `python3 calib/dan_table.py` hides every dan reading.
    import dans
    assert dans._table(), "rerun python3 calib/dan_table.py"


def test_dan_labels_follow_course_order_and_chart_type(monkeypatch):
    import dans, math
    tiers = dans.monotone([(i, t, math.log(r)) for i, (t, r) in enumerate(
        [("1st", 3.), ("2nd", 4.), ("3rd", 3.9), ("4th", 6.)])])
    table = {"7K Regular": tiers, "7K LN": tiers[:2], "4K REFORM": tiers,
             "4K Vibro": dans.monotone([(1, "Vibro 1", 2.3), (2, "Vibro 2", 2.5)])}
    monkeypatch.setattr(dans, "_table", lambda: table)
    monkeypatch.setattr(dans, "_data", lambda: {"vibro_weights": (1., 0.)})
    # The 2nd/3rd reversal is pooled but both tiers stay reachable, in order.
    values = [v for _t, v in table["7K Regular"]]
    assert values == sorted(values) and len(set(values)) == 4
    # nearest tier: a course rated exactly at its anchor shows that tier, not the one below
    assert dans.label(7, 3., 0.) == "Reg 1st" and dans.label(7, 3.3, 0.) == "Reg 1st+"
    assert dans.label(7, 9., 0.) == "Reg 4th+" and dans.label(7, 2., 0.) == "below Reg 1st"
    assert dans.label(7, 3.5, .6).startswith("LN ") and dans.label(6, 5., 0.) is None
    # Mixed: mostly taps with LN-dan amounts of holds read on both, tap dan first
    assert dans.label(7, 3.5, .3).startswith("Reg ") and " · LN " in dans.label(7, 3.5, .3)
    # vibro is read from the notes: a 4-column run at 11.5 hits/s is vibro, shown beside the rice dan
    run = [(t * 87, t * 87, c) for t in range(40) for c in range(4)]
    vib = dans.vibro_runs(run)
    assert vib["share"] == 1. and 11 < vib["speed"] < 12
    assert dans.label(4, 3., 0., vib) == "Vibro 2−"     # all vibro: the rice dan would name the wrong skill
    mixed = run + [(4000 + t * 87, 4000 + t * 87, t % 4) for t in range(600)]      # 21 % vibro runs
    assert dans.label(4, 3., 0., dans.vibro_runs(mixed)) == "1st · Vibro 2−"
    assert dans.ability(4, 3.) == "1st" and dans.ability(7, 3.5, "ln/release").startswith("LN ")
    assert dans.ability(4, 3., "jackspeed/vibro") is None and dans.ability(4, None) is None
    assert dans.vibro_runs(run, rate=.8)["share"] == 0.          # slowed to 9.2 hits/s: jacks, not vibro
    stream = [(t * 87, t * 87, t % 4) for t in range(160)]
    assert dans.label(4, 3., 0., dans.vibro_runs(stream)) == "1st"


def test_interleaved_bracket_is_harder_than_a_plain_bracket():
    # user 2026-10-06: "they get rated the same as regular brackets but they constantly alternate
    # fingers". 13↔2 per hand vs 12↔3 per hand, same rows, notes and timing.
    inter = [(1000 + 110 * i, c) for i in range(800) for c in ((0, 2, 4, 6) if i % 2 else (1, 5))]
    plain = [(1000 + 110 * i, c) for i in range(800) for c in ((0, 1, 5, 6) if i % 2 else (2, 4))]
    assert rate(inter)["overall"] > 1.1 * rate(plain)["overall"]
