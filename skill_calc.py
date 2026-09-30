#!/usr/bin/env python3
"""
ManiaScope difficulty model — native osu!mania, 7K-first, 4K-10K supported.

Rating convention: the skill a well-rounded player needs for controlled,
consistently accurate play. Chart-derived endurance and supported OD interactions
are calibrated against player scores; HP is not a difficulty input.
Every displayed number — overall, each skill, stamina — comes from the SAME
demand unit, the SAME time aggregation and the SAME scale curve, so 20 in LN
and 20 in jacks mean the same required level as far as the weights below are
right. The unit anchor is specific to each keymode, not a cross-keymode ability
claim. Those weights are PROVISIONAL (see README "Calibration").

Pipeline:
  1. parse once: notes (start, end, column), LN tails kept, non-mania rejected;
     timing points → the scroll multiplier lazer's mania playfield uses.
  2. per-action effort, hand/finger aware, in played time (source time / rate):
       hand   speed of the hand between different columns (streams, rolls, trills);
              a finger that repeats its column is not charged for hand movement
       jack   same finger repeating with nothing else on that hand in between;
              fingers repeating together on one hand are one bounce (shared
              strongly), both hands repeating at once is one mirrored bounce;
              a repeat with the other hand in between costs W_IJACK of that
       split  a chord spanning both hands must land together: coordination on
              top of the busier hand
       cross  overall row rate (reading / two-hand coordination)
       tech   rhythm irregularity and flams, scaled by local speed
       sv     reading: scroll changes that shorten/stretch a note's time on screen
              or change its speed just before it is hit, scaled by row rate
       ln     releases (cheap when synchronous, costly when staggered against
              other actions of the hand), presses while the hand is holding,
              and fast release->re-press of one column. Heads are ordinary
              presses and are NOT charged again.
     Notes of one row share effort sub-linearly (a chord is one motion).
     Odd keymodes: the centre key is evaluated as whole-chart left-thumb and
     whole-chart right-thumb, and the two demand series are averaged.
  3. effort is summed into 0.5 s sections per hand; hands combine by 2-norm so a
     one-handed pattern is not diluted by an idle hand.
  4. capacity curve: the hardest 2 / 8 / 30 / 120 s window of demand, each
     divided by how far above sustainable level a player can go for that long.
     The binding horizon is the rating. Rest lowers it, length alone never raises it.
  5. skills (attribution only, overall is untouched): each ±1 s section's demand
     counts toward every pattern it shows, relative to its dominant one, so a
     skill is "how hard the sections of that kind are" on the overall scale.
     Technical / Bracket are descriptors of the same demand; Stamina is the
     hardest 4 minutes (a shorter chart counted with rest).
  6. chronological hand-local fatigue/recovery and pattern-sensitive timing
     features feed a score-trained correction. SR anchors the median units only.

Usage: skill_calc.py <file.osu> [rate]
"""
from array import array
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import cython

import bisect
import functools
import math
import os

MODEL_VERSION = "3.5-technical-presence"

# Skills per keymode. 4K keeps its vocabulary. 5K+: Delay (fast rice, rows < 70 ms apart, each hand
# pressing one finger at a time — 7777's Delay packs), Stream (the same texture slower), Chordstream (chord-led rice), Bracket (a hand
# trilling between disjoint finger groups, one of them a chord), Jack (a single finger repeating on
# its hand: anchors, jack runs), Chordjack (repetition inside chords). Stamina is last so a tie with a
# pattern skill shows the pattern first. Both: Jumptrill (2+ note one-hand chords, the hands taking turns;
# chords may change: 12/34, 123/567, 12/56/23/67). 4K: Split Trill (chords over both hands alternating: 13/24, 14/23; 5K+ that is
# Bracket), and Dump — Delay's texture (fast rice, one finger per hand) as a descriptor over the stream
# family, so a fast handstream is Handstream AND Dump.
PARTS_4K = ("stream", "jumpstream", "handstream", "jumptrill", "splittrill", "jackspeed", "chordjack", "ln", "sv")
PARTS_5K = ("delay", "stream", "chordstream", "jumptrill", "jack", "chordjack", "ln", "sv")
TRILL_PARTS = ("stream", "delay", "chordstream", "jack", "chordjack", "jackspeed")   # a trill row's rice


RAWS = ("ln_rel",)                                   # the "ln" part's release share
COUNTS = ("mj", "sus", "lj", "shield", "inv", "hyb", "vib")
EVIDENCE = ("odd", "rodd", "pins", "psw", "oht", "anc", "rows", "sv_raw")   # per-bin counts / raws, never scaled as demand
NON_DEMAND = frozenset(EVIDENCE + COUNTS)
SV_KINDS = ("sv_fast", "sv_slow", "sv_accel", "sv_stutter", "sv_brake")
# Decelerating approaches deliberately keep the neutral label until a more
# specific user-facing vocabulary is useful. A brake is not an oscillation.
SV_NAMES = ("Fast SV", "Slowjam SV", "Accel SV", "Stutter SV", "SV")
SV_KIND_SHARE = 0.5  # a single SV type names the SV entry when it carries this share of SV demand        # event counts per bin


JACKS = ("minijack", "longjack", "vibro")
JACK_PARTS = ("chordjack", "jack", "jackspeed")                         # refinements of the jack parts by run length
LN_KINDS = ("release", "inverse", "hybrid", "shield")   # refinements of the LN part


def skills(keys):
    """Skill keys measured for a keymode, in display order for ties."""
    base = PARTS_4K[:3] + ("dump",) + PARTS_4K[3:] if keys < 5 else PARTS_5K[:3] + ("bracket",) + PARTS_5K[3:]
    i = base.index("ln")
    return (base[:i] + (("quadstream",) if keys == 4 else ()) + JACKS + base[i:i + 1] + LN_KINDS + base[i + 1:]
            + SV_KINDS
            + ("trill1h", "anchor", "patterntech", "rhythmtech", "technical", "mash", "stamina"))

# ---- calibration knobs (few, on purpose) ----------------------------------
SCALE = 2.2         # level = SCALE * sqrt(demand)
STAR_A, STAR_B = 0.1743, 1.095   # rating = STAR_A * level**STAR_B: osu! star-rating realm
                                 # (fit on 4.4k ranked 7K maps at 1.0x; ordering unchanged)
W_HAND = 1.0
V_FAST_KEYS = 2.0   # 8K+: V_FAST × (keys / 7)**2 — five fingers a hand spread fast rice (public scores:
                    # 10K Delay-heavy maps read 8–20 % too hard at 12 Hz)
V_FAST, SPEED_P = 12.0, 1.5   # hand speed above V_FAST Hz costs (v/V_FAST)**SPEED_P more (public scores:
                              # 7K maps with rows < 55 ms apart read 10-13 % too easy at 8*+ without it)
W_JACK = 3.3
W_JACK_7K = 3.0     # 7K: recalibrated with the speed-dependent discount for changing two-hand chords
JACK_P = 0.5        # jack cost × (speed / 12 Hz)**JACK_P: slow jacks read too hard, fast ones too easy (public scores)
W_IJACK = 0.4       # a hand-local jack with the other hand in between, below 7 Hz; full weight from 11 Hz
                    # (public scores: slow interleaved jacks read too hard, dense-chordstream ones too easy)
JACK_SHARE = 0.7    # a hand jacking n fingers at once counts n**(1-JACK_SHARE) jacks (one bounce)
MIRROR = 0.25       # base jack weight of the lesser hand when both hands repeat in the same row
W_SPLIT = 1.5       # coordination of a chord split over both hands, × the lesser hand's effort
W_CROSS = 0.15
W_CROSS_WIDE = 0.05   # 5K+ row-rate cost (public scores: 7K fast rice read too easy at 0.15; 4K keeps 0.15)
W_TECH = 0.3         # public scores: rhythm-irregular maps read too hard at 0.55 / 0.4
W_SV = 2.8          # reading cost per unit of scroll strain, × row rate (public scores: SV-heavy maps
                    # read too easy; SV/noSV chart pairs: SV adds +29 %, the 2.9 model +17 %)
W_REL = 0.50        # plain release, relative to a press at the same hand speed
W_STAG = 0.30       # staggered release against another action of the hand
W_HOLD_SAME = 0.30  # press while another finger of the same hand is holding
W_HOLD_OTHER = 0.08
W_REPRESS = 0.30    # release -> re-press of the same column
LN_BASE = 10.0       # Hz: speed-independent coordination cost of an LN action
ROW_SHARE = 0.3     # notes of an n-row each count n**-ROW_SHARE
FLAM = 0.045        # s: gaps shorter than this taper toward a chord (speed × (gap/FLAM)**2); dense 1/4 rice at
                    # 45-55 ms read 10-20 % too easy with a sqrt taper from 60 ms, 26 ms rolls too hard (public scores)
V_FINGER = 8.0      # Hz: a finger re-pressing its column faster than this costs proportionally more (public scores)
HORIZONS = ((2.0, 1.75), (8.0, 1.30), (30.0, 1.12), (120.0, 1.00))  # (window s, tolerance)
BIN = 0.5
PHASES = 8          # bin-grid offsets tried for the horizon windows (rate sweeps stay smooth)
H_WEIGHTS = (0.0, 0.2, 0.25, 0.35)   # pull of each horizon's level on the rating: a map hard at every
                                     # horizon plays harder than one peaking at a single one (public scores)
RATE_G = 0.12                        # every rating × rate**RATE_G: DT/HT plays are harder/easier than the
                                     # same note speeds at 1.0× (public scores: DT +5-6 % underrated without it)
G_NOTES = -0.03                      # rating × (notes / 2000)**G_NOTES: accuracy averages over every note
# Attribution only (never changes overall): a section's demand counts toward every skill it shows,
# relative to its dominant pattern; Technical saturates when this share of rows is awkward, Bracket
# when bracket transitions carry this share of the demand.
TECH_FULL = 0.09
RT_WIN = 4           # bins: Rhythm Tech's share is taken over ±2 s (sustained, not one odd row)
RT_FULL = 0.2        # Rhythm Tech saturates when this share of rows are off-grid events
OHT_FULL = 0.5      # 1H Trill saturates when this share of rows are one-hand-trill rows
ANC_FULL = 0.5      # Anchor: this share of rows inside anchor runs
PT_FULL = 0.05       # Pattern Tech saturates when this share of rows carry pattern evidence
PT_CHAIN = 1.5       # s: a pattern switch counts when another came this recently
ROLL_MIN = 6         # rows: a single-note roll this long is a sweep for Rhythm Tech
R_EVENT = 0.025      # s: rows closer than this are one rhythmic event (flam, rolled chord) for Rhythm Tech
DELAY_GAP = 0.07     # s between rows: faster one-finger-per-hand rice is Delay (4K: Dump), slower is Stream
TRILL_GAP = 0.2      # s between rows: a chord trill (Jumptrill, Split Trill) is faster than this
BRACKET_FULL = 0.4
DUMP_FULL = 0.5      # 4K Dump saturates when Delay-rule rice carries this share of the demand
MJ_BURST = 2         # rows: a chain of at most this many column-repeating rows, then a break, is a minijack
MJ_FULL = 0.2        # Minijack saturates when this share of rows are minijack-burst rows
VIBRO_GAP = 0.095    # s: vibro repeats are faster than this ...
VIBRO_MIN = 5        # ... for at least this many notes in one column
VIB_FULL = 0.3       # Vibro saturates when this share of rows holds a vibro note
LJ_FULL = 0.5        # Longjack: this share of rows repeating inside a 4+ note run
SHIELD_FULL = 0.5    # Shield: this share of rows is a tap followed, within 0.12 s, by a hold in its column
INV_FULL = 1.5       # Inverse: this share of rows re-presses a column right after its hold ends
HYB_FULL = 0.8       # Hybrid LN: this share of rows are taps under a hold
REL_FULL = 0.6       # Release: releases carry this share of the section's LN demand
STAMINA_WINDOW = 240.0   # s: Stamina = the hardest 4 minutes of demand at the 120 s tolerance, a
                         # shorter chart padded with rest — endurance, not a relabelled overall

class ChartError(ValueError):
    """Input is not a usable native osu!mania chart."""


class AnalysisCancelled(Exception):
    """The viewer selected another chart; no partial analysis is a result."""


def _check_cancelled(cancelled):
    if cancelled is not None and cancelled():
        raise AnalysisCancelled()


class Chart:
    __slots__ = ("keys", "notes", "title", "version", "artist", "creator", "sv", "od")


def _parse(path):
    meta = {}
    keys = None
    od = 5.0
    mode = 0
    section = ""
    raw, tps = [], []
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        if not fh.readline().lstrip("﻿").startswith("osu file format"):
            raise ChartError("not an .osu file")
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith("["):
                section = line
                continue
            if section == "[HitObjects]":
                p = line.split(",")
                try:
                    x, t, typ = int(float(p[0])), int(float(p[2])), int(p[3])
                    end = int(float(p[5].split(":", 1)[0])) if typ & 128 else t
                except (ValueError, IndexError):
                    continue
                raw.append((t, max(end, t), x))
            elif section == "[TimingPoints]":
                p = line.split(",")
                try:
                    tps.append((float(p[0]), float(p[1]), p[6].strip()[:1] != "0" if len(p) > 6 else True))
                except (ValueError, IndexError):
                    continue
            elif ":" in line:
                k, v = line.split(":", 1)
                if section == "[General]" and k == "Mode":
                    mode = int(v.strip() or 0)
                elif section == "[Difficulty]" and k == "CircleSize":
                    keys = int(round(float(v)))
                elif section == "[Difficulty]" and k == "OverallDifficulty":
                    od = float(v)
                elif section == "[Metadata]":
                    meta[k] = v.strip()
    if mode != 3:
        raise ChartError("not a native osu!mania chart (converts are unsupported)")
    if keys is None or not 1 <= keys <= 18:
        raise ChartError(f"unsupported key count: {keys}")
    # Normalise: column, dedupe, order-independent sort, no same-column overlap.
    notes = sorted({(t, e, min(keys - 1, max(0, x * keys // 512))) for t, e, x in raw},
                   key=lambda n: (n[0], n[2], n[1]))
    out, last = [], {}
    for t, e, c in notes:
        i = last.get(c)
        if i is not None:
            pt, pe, _ = out[i]
            if pt == t:
                continue  # duplicate head
            if pe > t:
                out[i] = (pt, t, c)  # impossible overlap: cut the earlier hold
        last[c] = len(out)
        out.append((t, e, c))
    ch = Chart()
    ch.keys, ch.notes = keys, out
    ch.od = od if math.isfinite(od) else 5.0
    ch.title = meta.get("TitleUnicode") or meta.get("Title", "")
    ch.artist = meta.get("ArtistUnicode") or meta.get("Artist", "")
    ch.version, ch.creator = meta.get("Version", ""), meta.get("Creator", "")
    ch.sv = _scroll(tps, max((e for _, e, _ in out), default=0))
    return ch


def _scroll(tps, last):
    """[(ms, scroll multiplier)] as osu!lazer's mania playfield scrolls (ppy/osu 577d29f,
    DrawableScrollingRuleset + LegacyBeatmapDecoder): multiplier = SV (100 / -beatLength of a
    green line, 1 at a red line, clamped 0.01-10) × most common beat length / current beat length;
    last point at a timestamp wins; points after the last object are dropped. [] = constant."""
    pts, bl, sv = [], None, 1.0
    for t, v, red in tps:
        if red and v > 0:
            bl, sv = min(60000.0, max(6.0, v)), 1.0
        elif not red:
            sv = min(10.0, max(0.01, 100.0 / -v)) if v < 0 else 1.0
        if bl is not None and t <= last:
            pts.append((t, bl, sv))
    reds = [(t, v) for t, v, red in tps if red and v > 0]
    if not pts or not reds:
        return []
    dur = {}
    for i, (t, v) in enumerate(reds):     # most common beat length by duration; first red starts at 0
        if t > last:
            continue
        end = reds[i + 1][0] if i + 1 < len(reds) else last
        k = round(min(60000.0, max(6.0, v)), 3)
        dur[k] = dur.get(k, 0.0) + max(0.0, min(end, last) - (0.0 if i == 0 else t))
    base = max(dur, key=dur.get) if dur else pts[0][1]
    out = {}
    for t, b, v in pts:
        out[t] = v * base / b
    sv = sorted(out.items())
    return [] if all(abs(m - 1.0) < 1e-3 for _, m in sv) else sv


@functools.lru_cache(maxsize=16)
def _parse_cached(path, _mtime):
    return _parse(path)


def parse_osu(path):
    """Parse by file revision, including replacements that retain their mtime."""
    s = os.stat(path)
    return _parse_cached(path, (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns))


SV_PIECES = 3       # a note's approach in pieces: fast scroll jitter averages out within one
W_WOB = 2.0         # reading cost of speed changes during the approach (stutters), per unit of log speed
SV_VIS = 450.0      # ms of chart time a note is visible at multiplier 1 (nominal reading setup)


def _legacy_sv_reading(chart, kinds=None):
    """Per-note reading strain from scroll changes (0 on constant scroll): how far the time a note
    is visible departs from nominal (compressed: less time to read; stretched: notes pile up), and
    how much its speed at the judgement line differs from its average approach speed (a note that
    brakes or lunges just before it is hit is timed wrong). Rate-free: ratios of chart time."""
    if not chart.sv:
        if kinds is not None:
            kinds.extend([(0.0,) * len(SV_KINDS)] * len(chart.notes))
        return [0.0] * len(chart.notes)
    ts = [t for t, _ in chart.sv]
    ms = [m for _, m in chart.sv]
    xs = [0.0]                            # scroll position at each control point
    for i in range(1, len(ts)):
        xs.append(xs[-1] + (ts[i] - ts[i - 1]) * ms[i - 1])

    def seg(t):
        return max(0, bisect.bisect_right(ts, t) - 1)

    def pos(t):
        i = seg(t)
        return xs[i] + (t - ts[i]) * ms[i]

    def time_at(x):                       # inverse of pos (positions are monotonic)
        i = max(0, bisect.bisect_right(xs, x) - 1)
        return ts[i] + (x - xs[i]) / ms[i]
    out = []
    for t, _e, _c in chart.notes:
        seen = t - time_at(pos(t) - SV_VIS)          # ms the note is on screen
        ratio = SV_VIS / max(seen, 1.0)             # average approach speed / nominal
        # the approach in pieces (the eye averages scroll jitter within a piece): arrival speed is
        # the last piece's, wobble how far the pieces' speeds stray from the average
        d = seen / SV_PIECES
        sp = [max((pos(t - k * d) - pos(t - (k + 1) * d)) / d, 0.01) / ratio for k in range(SV_PIECES)]
        brake = abs(math.log(sp[0]))
        wob = W_WOB * sum(abs(math.log(v)) for v in sp) / SV_PIECES
        lr = math.log(ratio)
        out.append(min(3.0, abs(lr) + brake + wob))
        if kinds is not None:
            tot = abs(lr) + brake + wob or 1.0
            # Wobble magnitude alone also fires on a smooth brake. Stutter
            # needs a meaningful direction reversal during the approach.
            logs = [math.log(v) for v in reversed(sp)]
            d1, d2 = logs[1]-logs[0], logs[2]-logs[1]
            stutter = wob if d1*d2 < 0 and min(abs(d1), abs(d2)) > .1 else 0.
            directed = brake + wob - stutter
            accel = directed if sp[0] >= 1 else 0.
            decel = directed if sp[0] < 1 else 0.
            kinds.append((max(lr, 0.0)/tot, max(-lr, 0.0)/tot, accel/tot, stutter/tot, decel/tot))
    return out


def sv_reading(chart, kinds=None, rate=1.0, detail=None, *, cancelled=None):
    """Reading at the actual rate; Constant Speed bypasses this entirely."""
    from skill_calc import _check_cancelled
    import scroll_reading
    return scroll_reading.reading(chart, rate, kinds, detail, cancelled=cancelled)


def _v(d, knee=FLAM, taper=2.0):
    """Speed (Hz) of a gap; below the knee it tapers to 0 (flam ≈ chord)."""
    if d >= knee:
        return 1.0 / d
    return (max(d, 0.0) / knee) ** taper / knee


def _smooth(x, lo, hi):
    x = (x - lo) / (hi - lo)
    if not x > 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    return x * x * (3 - 2 * x)


def _off_grid(g, ref, pulse=True):
    """0..1: how far an event gap is off a reference. Slower: a whole multiple of it is on it (50 50 150
    skips two slots); faster than a pulse (a repeated gap): a binary subdivision is on it (1/4 -> 1/8), a
    triplet or 1/4 <-> 1/6 switch is not; faster than a lone previous gap (it may itself have skipped
    slots): any whole division is on it."""
    r = g / ref
    if r < 1 and not pulse:
        r = 1 / r
    x = abs(math.log2(r / max(1, round(r)))) if r >= 1 else abs(math.log2(r))
    return min(1.0, 2.2 * max(0.0, abs(x - round(x)) - 0.06))


def _pattern_evidence(rowsets, trill):
    """Pattern Tech evidence per row (recognition only), each 0..1:
    ins  an inserted repeat: a row repeating a column of the row before, entered from and left into
         moving rice (the two transitions before and the one after do not repeat), both fast — a
         minijack the flow has to make room for. Doubling every note, jack runs and chordjack are
         templates, not insertions;
    sw   a chained switch: the row changes between rice and a chord trill (each side held 3 rows),
         fast, within PT_CHAIN s of another such switch — patterns changing faster than they settle."""
    n = len(rowsets)
    rep = [r > 0 and bool(rowsets[r][1] & rowsets[r - 1][1]) and rowsets[r][0] - rowsets[r - 1][0] < 0.25
           for r in range(n)]
    fam = [x or "rice" for x in trill]
    ins, sws = [0.0] * n, [0.0] * n
    last = None
    for r in range(n):
        t = rowsets[r][0]
        g = t - rowsets[r - 1][0] if r else 1.0
        if rep[r] and r >= 3 and not (rep[r - 1] or rep[r - 2] or r + 1 < n and rep[r + 1]):
            ins[r] = _smooth(1 / g, 6.0, 12.0) * _smooth(1 / (rowsets[r - 1][0] - rowsets[r - 2][0]), 6.0, 12.0)
        if (3 <= r < n - 2 and fam[r] != fam[r - 1] and fam[r - 1] == fam[r - 2] == fam[r - 3]
                and fam[r] == fam[r + 1] == fam[r + 2]):
            sw = _smooth(1 / g, 5.0, 10.0)
            if last is not None and t - last < PT_CHAIN:
                sws[r] = sw
            if sw > 0.3:
                last = t
    return ins, sws


def _hand_patterns(rowsets, hand_of, keys, *, cancelled=None):
    """Row membership (0/1) of two descriptors:
    oht  1H Trill: one hand carries 4+ consecutive rows alternating two single fingers (a b a b),
         5 Hz+, steady; the other hand plays in at most one of the four rows;
    anc  Anchor: a column recurring every 2-4 rows (never adjacent: that is a jack), 5+ times,
         in a third of the rows it spans, around 3+ other columns none as frequent as it (a
         two-column trill is not an anchor, and a roll's even recurrence is not either)."""
    from skill_calc import _check_cancelled
    n: cython.Py_ssize_t = len(rowsets)
    r: cython.Py_ssize_t
    c: cython.Py_ssize_t
    h: cython.Py_ssize_t
    hand_mask: cython.uint
    a_mask: cython.uint
    b_mask: cython.uint
    mask_data = array('I', (sum(1 << c for c in cs) for _,cs in rowsets))
    masks: cython.uint[::1] = mask_data
    hs: cython.uint[::1]
    oht, anc = [0] * n, [0] * n
    for h in (0, 1):
        hand_mask = sum(1 << c for c in range(keys) if hand_of[c] == h)
        hs = array('I', (mask & hand_mask for mask in mask_data))
        for r in range(3, n):
            if cancelled is not None and r%256 == 0:
                _check_cancelled(cancelled)
            a_mask, b_mask = hs[r-3], hs[r-2]
            if (a_mask and b_mask and not a_mask & (a_mask-1) and not b_mask & (b_mask-1)
                    and a_mask == hs[r-1] and b_mask == hs[r] and a_mask != b_mask
                    and ((masks[r-3] != a_mask) + (masks[r-2] != b_mask)
                         + (masks[r-1] != a_mask) + (masks[r] != b_mask)) <= 1):
                gaps = [rowsets[i][0] - rowsets[i-1][0] for i in range(r-2,r+1)]
                if max(gaps) <= 0.2 and max(gaps) <= 1.6 * min(gaps):
                    oht[r - 3:r + 1] = [1] * 4
    for c in range(keys):
        occ = [r for r in range(n) if masks[r] & (1 << c)]
        run = occ[:1]
        for a, z in zip(occ, occ[1:] + [None]):
            if z is not None and 2 <= z - a <= 4 and rowsets[z][0] - rowsets[a][0] < 0.5:
                run.append(z)
                continue
            if len(run) >= 5 and len(run) * 3 >= run[-1] - run[0] + 1:
                others = {}
                for r in range(run[0], run[-1] + 1):
                    if cancelled is not None and r%256 == 0:
                        _check_cancelled(cancelled)
                    for x in rowsets[r][1] - {c}:
                        others[x] = others.get(x, 0) + 1
                if len(others) >= 3 and max(others.values()) <= 0.6 * len(run):
                    anc[run[0]:run[-1] + 1] = [1] * (run[-1] - run[0] + 1)
            run = [z]
    return oht, anc


def _demand_geometry(chart, rate, *, cancelled=None):
    """Hand-independent rows and repeat evidence for one chart/rate calculation."""
    from skill_calc import _check_cancelled
    inv: cython.double = .001/rate
    t: cython.double
    e: cython.double
    notes = [(t*inv,e*inv,c) for t,e,c in chart.notes]
    bycol = {}
    for k,(_,_,c) in enumerate(notes):
        if cancelled is not None and k%256 == 0:
            _check_cancelled(cancelled)
        bycol.setdefault(c,[]).append(k)
    vib = array('i',[0])*len(notes)
    for ks in bycol.values():
        run = [ks[0]]
        for a,k in zip(ks,ks[1:]+[None]):
            if k is not None and notes[k][0]-notes[a][0] < VIBRO_GAP and notes[a][1]-notes[a][0] <= .003:
                run.append(k)
                continue
            if len(run) >= VIBRO_MIN:
                for xid in run:vib[xid] = 1
            run = [k]
    rowsets,boundaries,k = [],[],0
    while k < len(notes):
        m = k+1
        while m < len(notes) and notes[m][0]-notes[k][0] <= .003:
            m += 1
        rowsets.append((notes[k][0],frozenset(c for _,_,c in notes[k:m])))
        boundaries.append((k,m))
        k = m
    roll = [False]*len(rowsets)
    run,run_step = [],None
    for r in range(len(rowsets)+1):
        cs = rowsets[r][1] if r < len(rowsets) else ()
        step = (next(iter(cs))-next(iter(rowsets[r-1][1])))%chart.keys if len(cs)==1 and run and len(rowsets[r-1][1])==1 else None
        if step in (1,chart.keys-1) and (len(run)<2 or step==run_step):
            run.append(r);run_step = step
            continue
        if len(run) >= ROLL_MIN:
            for xid in run:roll[xid] = True
        run,run_step = ([r] if len(cs)==1 else [],None)
    return dict(notes=notes,times=array('d',(v[0] for v in notes)),
                ends=array('d',(v[1] for v in notes)),columns=array('i',(v[2] for v in notes)),
                vib=vib,rows=rowsets,boundaries=boundaries,roll=roll)


def _demand(chart, rate, centre_left, strain, kinds=None, *, cancelled=None, plain_runs=None, motor_actions=None, prepared=None):
    """Per-bin attributed and total demand, with bounded column state and numeric buffers.

    Parts partition demand; technical is the rhythm surcharge. Bracket, odd
    rows and the remaining evidence counters describe that same physical work.
    Local annotations are optional Cython hints; normal Python ignores them.
    """
    from skill_calc import _check_cancelled
    _constant_BIN: cython.double = BIN
    _constant_DELAY_GAP: cython.double = DELAY_GAP
    _constant_JACK_P: cython.double = JACK_P
    _constant_JACK_SHARE: cython.double = JACK_SHARE
    _constant_LN_BASE: cython.double = LN_BASE
    _constant_MIRROR: cython.double = MIRROR
    _constant_MJ_BURST: cython.Py_ssize_t = MJ_BURST
    _constant_PHASES: cython.Py_ssize_t = PHASES
    _constant_ROW_SHARE: cython.double = ROW_SHARE
    _constant_R_EVENT: cython.double = R_EVENT
    _constant_SPEED_P: cython.double = SPEED_P
    _constant_TRILL_GAP: cython.double = TRILL_GAP
    _constant_V_FAST: cython.double = V_FAST
    _constant_V_FAST_KEYS: cython.double = V_FAST_KEYS
    _constant_V_FINGER: cython.double = V_FINGER
    _constant_W_CROSS: cython.double = W_CROSS
    _constant_W_CROSS_WIDE: cython.double = W_CROSS_WIDE
    _constant_W_HAND: cython.double = W_HAND
    _constant_W_HOLD_OTHER: cython.double = W_HOLD_OTHER
    _constant_W_HOLD_SAME: cython.double = W_HOLD_SAME
    _constant_W_IJACK: cython.double = W_IJACK
    _constant_W_JACK: cython.double = W_JACK
    _constant_W_JACK_7K: cython.double = W_JACK_7K
    _constant_W_REL: cython.double = W_REL
    _constant_W_REPRESS: cython.double = W_REPRESS
    _constant_W_SPLIT: cython.double = W_SPLIT
    _constant_W_STAG: cython.double = W_STAG
    _constant_W_SV: cython.double = W_SV
    _constant_W_TECH: cython.double = W_TECH
    c: cython.Py_ssize_t
    keys: cython.Py_ssize_t = chart.keys
    half: cython.Py_ssize_t = keys // 2
    wide: cython.bint = keys >= 5
    v_fast: cython.double = _constant_V_FAST * (max(keys, 7) / 7) ** _constant_V_FAST_KEYS
    w_jack: cython.double = _constant_W_JACK_7K if keys == 7 else _constant_W_JACK
    hand_of: cython.int[::1] = array('i', (0 if c < half or (keys % 2 and c == half and centre_left) else 1 for c in range(keys)))
    if prepared is None:
        prepared = _demand_geometry(chart,rate,cancelled=cancelled)
    notes = prepared['notes']
    n: cython.Py_ssize_t = len(notes)
    t0: cython.double = notes[0][0]
    nbins: cython.Py_ssize_t = int((max((e for _, e, _ in notes)) - t0) / _constant_BIN) + 1
    nfine: cython.Py_ssize_t = nbins * _constant_PHASES
    times: cython.double[::1] = prepared['times']
    ends: cython.double[::1] = prepared['ends']
    columns: cython.int[::1] = prepared['columns']
    reading: cython.double[::1] = array('d', strain if strain is not None else [0.0] * n)
    order: cython.int[::1] = array('i', [0] * keys)
    seen: cython.int[::1] = array('i', [0] * keys)
    last_head: cython.double[::1] = array('d', [0.0] * keys)
    last_end: cython.double[::1] = array('d', [0.0] * keys)
    last_size: cython.int[::1] = array('i', [0] * keys)
    hold_weight: cython.double[::1] = array('d', [0.0] * keys)
    nseen: cython.Py_ssize_t = 0
    previous_hand: cython.double[::1] = array('d', [0.0, 0.0])
    previous_mask: cython.int[::1] = array('i', [0, 0])
    hand_mask: cython.int[::1] = array('i', [sum((1 << c for c in range(keys) if hand_of[c] == h)) for h in (0, 1)])
    jrun: cython.int[::1] = array('i', [0] * keys)
    run_time: cython.double[::1] = array('d', [0.0] * keys)
    runs = [[] for _ in range(keys)]
    burst = []
    vib: cython.int[::1] = prepared['vib']
    rowsets, boundaries = prepared['rows'], prepared['boundaries']
    trill = [None] * len(rowsets)
    for r in range(3, len(rowsets)):
        _slice_values = rowsets[r - 3:r + 1]
        (ta, a), (_tb, bb), (_tc, cc), (td, dd) = (_slice_values[0], _slice_values[1], _slice_values[2], _slice_values[3])
        if min(map(len, (a, bb, cc, dd))) >= 2 and td - ta < 3 * _constant_TRILL_GAP:
            ha, hb, hc, hd = ({hand_of[col] for col in cs} for cs in (a, bb, cc, dd))
            kind = 'jumptrill' if len(ha) == len(hb) == 1 and ha != hb and (ha == hc) and (hb == hd) else 'splittrill' if not wide and cc == a and (dd == bb) and (not a & bb) and (len(ha) == len(hb) == 2) else None
            if kind:
                trill[r - 3] = trill[r - 2] = trill[r - 1] = trill[r] = kind
    p_ins, p_sw = _pattern_evidence(rowsets, trill)
    roll = prepared['roll']
    oht, anc = _hand_patterns(rowsets, hand_of, keys, cancelled=cancelled)
    part_keys = (PARTS_5K if wide else PARTS_4K) + ('technical', 'bracket', 'dump') + EVIDENCE + RAWS + COUNTS + SV_KINDS
    all_keys = tuple(dict.fromkeys(PARTS_4K + PARTS_5K + ('technical', 'bracket', 'dump') + EVIDENCE + RAWS + COUNTS + SV_KINDS))
    index = {s: j for j, s in enumerate(all_keys)}
    stream: cython.Py_ssize_t = index['stream'] * nbins
    chordstream: cython.Py_ssize_t = index['chordstream'] * nbins
    chordjack: cython.Py_ssize_t = index['chordjack'] * nbins
    jack: cython.Py_ssize_t = index['jack'] * nbins
    jackspeed: cython.Py_ssize_t = index['jackspeed'] * nbins
    ln: cython.Py_ssize_t = index['ln'] * nbins
    sv: cython.Py_ssize_t = index['sv'] * nbins
    delay: cython.Py_ssize_t = index['delay'] * nbins
    technical: cython.Py_ssize_t = index['technical'] * nbins
    bracket: cython.Py_ssize_t = index['bracket'] * nbins
    dump: cython.Py_ssize_t = index['dump'] * nbins
    odd_part: cython.Py_ssize_t = index['odd'] * nbins
    rodd: cython.Py_ssize_t = index['rodd'] * nbins
    pins: cython.Py_ssize_t = index['pins'] * nbins
    psw: cython.Py_ssize_t = index['psw'] * nbins
    oht_part: cython.Py_ssize_t = index['oht'] * nbins
    anc_part: cython.Py_ssize_t = index['anc'] * nbins
    rows_part: cython.Py_ssize_t = index['rows'] * nbins
    sv_raw: cython.Py_ssize_t = index['sv_raw'] * nbins
    ln_rel: cython.Py_ssize_t = index['ln_rel'] * nbins
    mj: cython.Py_ssize_t = index['mj'] * nbins
    sus: cython.Py_ssize_t = index['sus'] * nbins
    lj: cython.Py_ssize_t = index['lj'] * nbins
    shield: cython.Py_ssize_t = index['shield'] * nbins
    inv_part: cython.Py_ssize_t = index['inv'] * nbins
    hyb: cython.Py_ssize_t = index['hyb'] * nbins
    vib_part: cython.Py_ssize_t = index['vib'] * nbins
    part_data = array('d', [0.0]) * (len(all_keys) * nbins)
    comp: cython.double[::1] = part_data
    hand_data = array('d', [0.0]) * (3 * nbins)
    hands: cython.double[::1] = hand_data
    fine_data = array('d', [0.0]) * (3 * nfine)
    fine: cython.double[::1] = fine_data
    plain_x = [0.0] * nbins if plain_runs is not None else None
    plain_fine_x = [0.0] * nfine if plain_runs is not None else None
    rows_by_size = ([0] * nbins, [0] * nbins, [0] * nbins)
    n_hand: cython.int[::1] = array('i', [0, 0])
    n_jack: cython.int[::1] = array('i', [0, 0])
    row_hand: cython.int[::1] = array('i', [0, 0])
    jack_effort: cython.double[::1] = array('d', [0.0, 0.0])
    inserted: cython.double[::1] = array('d', [0.0, 0.0])
    mirror: cython.double[::1] = array('d', [1.0, 1.0])
    eh: cython.double[::1] = array('d', [0.0, 0.0])
    prev_row_t: cython.double = 0.0
    prev_gap: cython.double = 0.0
    pp_gap: cython.double = 0.0
    grid: cython.double = 0.0
    ev_t: cython.double = 0.0
    ev_gap: cython.double = 0.0
    ev_grid: cython.double = 0.0
    ev_seen: cython.bint = False
    after_roll: cython.bint = False
    prev_row_mask: cython.int = 0
    ri: cython.Py_ssize_t
    i: cython.Py_ssize_t
    j: cython.Py_ssize_t
    ni: cython.Py_ssize_t
    c: cython.Py_ssize_t
    h: cython.Py_ssize_t
    oc: cython.Py_ssize_t
    oi: cython.Py_ssize_t
    b: cython.Py_ssize_t
    fb: cython.Py_ssize_t
    rb: cython.Py_ssize_t
    mh: cython.Py_ssize_t
    top: cython.Py_ssize_t
    part: cython.Py_ssize_t
    tr: cython.Py_ssize_t
    size: cython.Py_ssize_t
    row_mask: cython.int
    shared_mask: cython.int
    shared_count: cython.Py_ssize_t
    old_width: cython.Py_ssize_t
    partial: cython.bint
    jacked: cython.bint
    rolling: cython.bint
    judged: cython.bint
    any_vib: cython.bint
    t: cython.double
    e: cython.double
    share: cython.double
    gap: cython.double
    cross: cython.double
    tech: cython.double
    read: cython.double
    x: cython.double
    frac: cython.double
    irr: cython.double
    odd: cython.double
    nxt: cython.double
    eg: cython.double
    speed: cython.double
    vj: cython.double
    dh: cython.double
    dj: cython.double
    vdj: cython.double
    vdh: cython.double
    e_hand: cython.double
    e_jack: cython.double
    rice: cython.double
    e_ln: cython.double
    held_same: cython.double
    held_other: cython.double
    g: cython.double
    rep: cython.double
    jack_w: cython.double
    split: cython.double
    jf: cython.double
    cs_sum: cython.double
    ds_sum: cython.double
    other: cython.double
    fast: cython.double
    r: cython.Py_ssize_t
    amount: cython.double
    for ri, (i, j) in enumerate(boundaries):
        if cancelled is not None and i % 256 == 0:
            _check_cancelled(cancelled)
        t = times[i]
        size = j - i
        share = size ** (-_constant_ROW_SHARE)
        b = int((t - t0) / _constant_BIN)
        fb = min(nfine - 1, int((t - t0) * _constant_PHASES / _constant_BIN))
        tr = index[trill[ri]] * nbins if trill[ri] else -1
        rows_by_size[min(size, 3) - 1][b] += 1
        gap = t - prev_row_t if ri else 0.0
        cross = (_constant_W_CROSS_WIDE if wide else _constant_W_CROSS) * _v(gap) if gap else 0.0
        tech = 0.0
        if gap and gap < 0.5:
            if gap < 0.04:
                irr = 1.0
            elif prev_gap and prev_gap < 0.5:
                x = abs(math.log2(gap / prev_gap))
                frac = abs(x - round(x))
                irr = 0.0 if x < 0.08 else min(1.0, 2.2 * max(0.0, frac - 0.06) + 0.25 * min(x, 1.0))
            else:
                irr = 0.0
            tech = _constant_W_TECH * _v(gap) * irr
        comp[technical + b] += tech
        read = 0.0
        top = i
        for ni in range(i, j):
            if reading[ni] > reading[top]:
                top = ni
        if gap:
            read = _constant_W_SV * reading[top] * math.sqrt(6.0 * _v(gap))
        comp[sv + b] += read
        comp[sv_raw + b] += read
        if read and kinds:
            for key, f in zip(SV_KINDS, kinds[top]):
                comp[index[key] * nbins + b] += read * f
        hands[2 * nbins + b] += cross + tech + read
        fine[2 * nfine + fb] += cross + tech + read
        if plain_x is not None:
            plain_x[b] += cross + tech + 0.0
            plain_fine_x[fb] += cross + tech + 0.0
        odd = 0.0
        if gap and gap < 0.04:
            nxt = times[j] - t if j < n else 1.0
            odd = float(not (prev_gap and prev_gap < 0.06 or nxt < 0.06))
        elif gap and 0.06 <= gap < 0.5 and (grid or (prev_gap and 0.06 <= prev_gap < 0.5 and (not (pp_gap and pp_gap < 0.06)))) and (not (prev_gap and prev_gap < 0.06 and pp_gap and (pp_gap < 0.06))):
            x = abs(math.log2(gap / (grid or prev_gap)))
            odd = min(1.0, 2.2 * max(0.0, abs(x - round(x)) - 0.06))
        if gap and 0.06 <= gap < 0.5 and prev_gap and (abs(math.log2(gap / prev_gap)) < 0.06):
            grid = gap
        nxt = times[j] - t if j < n else 0.0
        rolling = bool(roll[ri] and gap and (prev_gap and abs(math.log2(gap / prev_gap)) < 0.06 or (nxt and abs(math.log2(gap / nxt)) < 0.06)))
        if not ri or gap >= _constant_R_EVENT:
            eg = t - ev_t if ev_seen else 0.0
            judged = not (rolling or after_roll)
            if judged and eg and (eg < 0.5) and (ev_grid or (ev_gap and ev_gap < 0.5)):
                comp[rodd + b] += max(odd if gap and gap < 0.04 else 0.0, _off_grid(eg, ev_grid or ev_gap, ev_grid != 0.0))
            elif judged and gap and (gap < 0.04):
                comp[rodd + b] += odd
            if rolling:
                pass
            elif eg and eg < 0.5 and ev_gap and (abs(math.log2(eg / ev_gap)) < 0.06):
                ev_grid = eg
            elif not eg or eg >= 0.5:
                ev_grid = 0.0
            ev_t = t
            ev_gap = eg
            after_roll = rolling
            ev_seen = True
        else:
            comp[rodd + b] += odd
        comp[rows_part + b] += 1
        comp[pins + b] += p_ins[ri]
        comp[psw + b] += p_sw[ri]
        comp[oht_part + b] += oht[ri]
        comp[anc_part + b] += anc[ri]
        row_mask = 0
        any_vib = False
        for ni in range(i, j):
            row_mask |= 1 << columns[ni]
            any_vib = any_vib or bool(vib[ni])
        comp[vib_part + b] += any_vib
        if gap and gap < 0.25 and row_mask & prev_row_mask:
            burst.append(b)
        else:
            part = mj if len(burst) <= _constant_MJ_BURST else sus
            for rb in burst:
                comp[part + rb] += 1
            burst.clear()
        shared_mask = row_mask & prev_row_mask
        shared_count = 0
        for c in range(keys):
            shared_count += bool(shared_mask & 1 << c)
        for h in (0, 1):
            n_hand[h] = 0
            n_jack[h] = 0
            row_hand[h] = row_mask & hand_mask[h]
            mirror[h] = 1.0
            eh[h] = 0.0
        for ni in range(i, j):
            c = columns[ni]
            h = hand_of[c]
            n_hand[h] += 1
            n_jack[h] += bool(previous_mask[h] & 1 << c)
        if n_jack[0] and n_jack[1]:
            mh = 0 if n_jack[0] <= n_jack[1] else 1
            mirror[mh] = _constant_MIRROR
            partial = False
            if keys == 7:
                for h in (0, 1):
                    old_width = 0
                    for c in range(keys):
                        old_width += bool(previous_mask[h] & 1 << c)
                    partial = partial or not n_jack[h] == n_hand[h] == old_width
                if partial:
                    speed = min(_v(t - previous_hand[0], 0.07, 0.5), _v(t - previous_hand[1], 0.07, 0.5))
                    mirror[mh] += (1 - _constant_MIRROR) * _smooth(speed, 7.0, 9.0)
        for h in (0, 1):
            jack_effort[h] = 0.0
            inserted[h] = 0.0
            if n_jack[h]:
                vj = _v(t - previous_hand[h], 0.07, 0.5)
                jack_effort[h] = w_jack * vj * (vj / 12.0) ** _constant_JACK_P * n_jack[h] ** (-_constant_JACK_SHARE) * mirror[h]
                inserted[h] = _constant_W_IJACK + (1 - _constant_W_IJACK) * _smooth(vj, 7.0, 11.0)
        pending = []
        jack_w = 0.0
        for ni in range(i, j):
            e = ends[ni]
            c = columns[ni]
            h = hand_of[c]
            jacked = bool(previous_mask[h] & 1 << c)
            dh = 0.0
            for oc in range(keys):
                if seen[oc] and hand_of[oc] == h and (oc != c) and (not (jacked and (row_mask | previous_mask[h]) & 1 << oc)):
                    g = t - last_head[oc]
                    if not dh or g < dh:
                        dh = g
            dj = min(t - last_head[c], 1.0) if seen[c] else 1.0
            vdj = _v(dj)
            vdh = _v(dh) if dh else 0.0
            e_hand = _constant_W_HAND * math.sqrt(vdh * vdj) * share * max(1.0, vdh / v_fast) ** _constant_SPEED_P if dh else 0.0
            eh[h] += e_hand
            e_jack = jack_effort[h] if jacked else 0.0
            if jacked and (not prev_row_mask & 1 << c):
                e_jack *= inserted[h]
            rice = (e_hand + e_jack) * max(1.0, vdj / _constant_V_FINGER)
            held_same = 0.0
            held_other = 0.0
            for oi in range(nseen):
                oc = order[oi]
                if oc != c and last_head[oc] < t - 0.003 and (last_end[oc] > t + 0.003):
                    if hand_of[oc] == h:
                        held_same += hold_weight[oc]
                    else:
                        held_other += hold_weight[oc]
            e_ln = (rice + _constant_LN_BASE * share) * (_constant_W_HOLD_SAME * held_same ** 0.7 + _constant_W_HOLD_OTHER * min(1.0, held_other))
            if e - t <= 0.003 and held_same + held_other >= 0.5 and gap and (gap < 0.15):
                comp[hyb + b] += 1
            if seen[c] and last_end[c] - last_head[c] <= 0.003 and (e - t > 0.003) and (t - last_head[c] < 0.12):
                comp[shield + b] += 1
            if seen[c] and last_end[c] - last_head[c] > 0.003:
                g = t - last_end[c]
                if g < 0.25:
                    rep = _constant_W_REPRESS * _v(g, 0.04, 0.5) * (1 - _smooth(g, 0.12, 0.25)) * _smooth(last_end[c] - last_head[c], 0.06, 0.25) * share
                    e_ln += rep
                    if g < 0.12 and last_end[c] - last_head[c] >= 0.1:
                        comp[inv_part + b] += 1
            hands[h * nbins + b] += rice + e_ln
            fine[h * nfine + fb] += rice + e_ln
            comp[ln + b] += e_ln
            if motor_actions is not None:
                motor_actions[ni] = (h, rice, (cross + tech) / size, bool(jacked or e_ln > 0 or e - t > 0.003), fb)
            jrun[c] = jrun[c] + 1 if jacked else 0
            if jacked and prev_row_mask & 1 << c and (shared_count == 1) and (e - t <= 0.003) and (last_end[c] - last_head[c] <= 0.003) and (t - run_time[c] < 0.25):
                runs[c].append(b)
            else:
                if len(runs[c]) >= 3:
                    for rb in runs[c]:
                        comp[lj + rb] += 1
                runs[c].clear()
            run_time[c] = t
            if wide and e_jack and (jrun[c] >= 2 or n_hand[h] >= 2):
                jack_w += 1
                part = chordjack if n_jack[h] >= 2 or n_hand[h] >= 2 else jack
                comp[(tr if tr >= 0 else part) + b] += rice
            elif wide and e_jack:
                pending.append((rice, h))
            elif not wide and e_jack and prev_row_mask & 1 << c:
                jack_w += 1
                part = chordjack if size >= 2 and seen[c] and (last_size[c] >= 2) else jackspeed
                comp[(tr if tr >= 0 else part) + b] += rice
            else:
                pending.append((rice, h))
        comp[odd_part + b] += odd
        split = _constant_W_SPLIT * min(eh[0], eh[1])
        hands[2 * nbins + b] += split
        fine[2 * nfine + fb] += split
        if plain_x is not None:
            plain_x[b] += split
            plain_fine_x[fb] += split
        if wide:
            jf = jack_w / size
            cs_sum = 0.0
            ds_sum = 0.0
            for amount, h in pending:
                if n_hand[h] >= 2:
                    cs_sum += amount
                else:
                    ds_sum += amount
            cs_sum += cross + split if size >= 3 else 0.0
            ds_sum += cross + split if size < 3 else 0.0
            comp[(tr if tr >= 0 else chordstream) + b] += cs_sum * (1 - jf)
            fast = 1 - _smooth(gap or 1.0, _constant_DELAY_GAP - 0.01, _constant_DELAY_GAP + 0.01)
            comp[(tr if tr >= 0 else delay) + b] += ds_sum * (1 - jf) * fast
            comp[(tr if tr >= 0 else stream) + b] += ds_sum * (1 - jf) * (1 - fast)
            part = chordjack if size >= 2 else jack
            comp[(tr if tr >= 0 else part) + b] += (cs_sum + ds_sum) * jf
            for h in (0, 1):
                old_width = 0
                for c in range(keys):
                    old_width += bool(previous_mask[h] & 1 << c)
                if n_hand[h] and (not n_jack[h]) and previous_mask[h] and (previous_hand[h] == prev_row_t) and (max(n_hand[h], old_width) >= 2):
                    other = 0.0
                    for amount, hh in pending:
                        if hh == h:
                            other += amount
                    comp[bracket + b] += other * (1 - jf)
        else:
            jf = min(1.0, 2.0 * jack_w / size) if size >= 2 else jack_w
            other = 0.0
            for amount, h in pending:
                other += amount
            other = other + cross + split
            comp[(tr if tr >= 0 else stream) + b] += other * (1 - jf)
            part = chordjack if size >= 2 else jackspeed
            comp[(tr if tr >= 0 else part) + b] += other * jf
            if tr < 0:
                ds_sum = 0.0
                for amount, h in pending:
                    if n_hand[h] < 2:
                        ds_sum += amount
                ds_sum += cross + split if size < 3 else 0.0
                comp[dump + b] += ds_sum * (1 - jf) * (1 - _smooth(gap or 1.0, _constant_DELAY_GAP - 0.01, _constant_DELAY_GAP + 0.01))
        for ni in range(i, j):
            c = columns[ni]
            e = ends[ni]
            if not seen[c]:
                order[nseen] = c
                nseen += 1
                seen[c] = 1
            last_head[c] = t
            last_end[c] = e
            last_size[c] = size
            hold_weight[c] = _smooth(e - t, 0.06, 0.25)
        for h in (0, 1):
            if row_hand[h]:
                previous_hand[h] = t
                previous_mask[h] = row_hand[h]
        pp_gap = prev_gap
        prev_gap = gap
        prev_row_t = t
        prev_row_mask = row_mask
    _check_cancelled(cancelled)
    acts = ([], [])
    for _cancel_index, (t, e, c) in enumerate(notes):
        if cancelled is not None and _cancel_index % 256 == 0:
            _check_cancelled(cancelled)
        acts[hand_of[c]].append((t, c))
        if e - t > 0.003:
            acts[hand_of[c]].append((e, c))
    for a in acts:
        a.sort()
    ats = [array('d', (v[0] for v in a)) for a in acts]
    acs = [array('i', (v[1] for v in a)) for a in acts]
    at: cython.double[::1]
    ac: cython.int[::1]
    lo: cython.Py_ssize_t
    hi: cython.Py_ssize_t
    sync: cython.Py_ssize_t
    stag: cython.double
    prev_d: cython.double
    ln_len: cython.double
    d: cython.double
    ot: cython.double
    base: cython.double
    eff: cython.double
    for ni in range(n):
        t = times[ni]
        e = ends[ni]
        c = columns[ni]
        ln_len = e - t
        if ln_len <= 0.003:
            continue
        h = hand_of[c]
        at = ats[h]
        ac = acs[h]
        lo = bisect.bisect_left(at, e - 0.18)
        hi = bisect.bisect_right(at, e + 0.18)
        sync = 1
        stag = 0.0
        prev_d = 0.0
        for oi in range(lo, hi):
            if ac[oi] == c:
                continue
            ot = at[oi]
            d = abs(ot - e)
            if d <= 0.005:
                sync += 1
            else:
                stag += _smooth(d, 0.005, 0.03) * (1 - _smooth(d, 0.08, 0.18))
            if ot < e - 0.005:
                prev_d = e - ot if not prev_d else min(prev_d, e - ot)
        base = _constant_W_REL * (_constant_LN_BASE + (_v(prev_d) if prev_d else 0.0))
        eff = (base * sync ** (-_constant_ROW_SHARE) + _constant_W_STAG * 10.0 * min(stag, 2.0)) * _smooth(ln_len, 0.06, 0.25)
        b = min(nbins - 1, int((e - t0) / _constant_BIN))
        fb = min(nfine - 1, int((e - t0) * _constant_PHASES / _constant_BIN))
        hands[h * nbins + b] += eff
        fine[h * nfine + fb] += eff
        comp[ln + b] += eff
        comp[ln_rel + b] += eff
    for c in range(keys):
        if len(runs[c]) >= 3:
            for rb in runs[c]:
                comp[lj + rb] += 1
    part = mj if len(burst) <= _constant_MJ_BURST else sus
    for rb in burst:
        comp[part + rb] += 1
    handstream: cython.Py_ssize_t = index['handstream'] * nbins
    jumpstream: cython.Py_ssize_t = index['jumpstream'] * nbins
    low: cython.Py_ssize_t
    high: cython.Py_ssize_t
    singles: cython.Py_ssize_t
    jumps: cython.Py_ssize_t
    chords: cython.Py_ssize_t
    count: cython.Py_ssize_t
    hs: cython.double
    js: cython.double
    work: cython.double
    raw: cython.double
    factor: cython.double
    left: cython.double
    right: cython.double
    cross_value: cython.double
    for b in range(nbins if not wide else 0):
        low = max(0, b - 2)
        high = min(nbins, b + 3)
        singles = sum(rows_by_size[0][low:high])
        jumps = sum(rows_by_size[1][low:high])
        chords = sum(rows_by_size[2][low:high])
        count = singles + jumps + chords
        if count and comp[stream + b]:
            hs = min(1.0, 3.0 * chords / count)
            js = (1 - hs) * min(1.0, 3.0 * jumps / count)
            work = comp[stream + b]
            comp[handstream + b] = work * hs
            comp[jumpstream + b] = work * js
            comp[stream + b] = work * (1 - hs - js)
    offsets: cython.int[::1] = array('i', (index[k] * nbins for k in part_keys if k not in NON_DEMAND))
    oi: cython.Py_ssize_t
    plain_data = array('d', part_data) if plain_x is not None else None
    plain_comp: cython.double[::1]
    totals: cython.double[::1] = array('d', [0.0]) * nbins
    plain_totals: cython.double[::1]
    if plain_x is not None:
        plain_comp = plain_data
        for key in ('sv', 'sv_raw') + SV_KINDS:
            part = index[key] * nbins
            for b in range(nbins):
                plain_comp[part + b] = 0.0
        plain_totals = array('d', [0.0]) * nbins
        for b in range(nbins):
            left = hands[b]
            right = hands[nbins + b]
            cross_value = plain_x[b]
            raw = left + right + cross_value
            if raw > 0:
                plain_totals[b] = (math.hypot(left, right) + cross_value) / _constant_BIN
                factor = plain_totals[b] / raw
                for oi in range(len(offsets)):
                    plain_comp[offsets[oi] + b] *= factor
    for b in range(nbins):
        left = hands[b]
        right = hands[nbins + b]
        cross_value = hands[2 * nbins + b]
        raw = left + right + cross_value
        if raw > 0:
            totals[b] = (math.hypot(left, right) + cross_value) / _constant_BIN
            factor = totals[b] / raw
            for oi in range(len(offsets)):
                comp[offsets[oi] + b] *= factor
    fine_output = tuple((fine_data[h * nfine:(h + 1) * nfine].tolist() for h in range(3)))
    if plain_x is not None:
        plain_output = {k: plain_data[index[k] * nbins:(index[k] + 1) * nbins].tolist() for k in part_keys}
        plain_runs.append((plain_output, list(plain_totals), (fine_output[0], fine_output[1], plain_fine_x)))
    output = {k: part_data[index[k] * nbins:(index[k] + 1) * nbins].tolist() for k in part_keys}
    return (output, list(totals), fine_output)


def _phase_total(fine, p):
    phases: cython.Py_ssize_t = PHASES
    offset: cython.Py_ssize_t = p
    n: cython.Py_ssize_t = (len(fine[0]) + offset + phases - 1) // phases
    side: cython.Py_ssize_t
    i: cython.Py_ssize_t
    b: cython.Py_ssize_t
    v: cython.double
    f: cython.double[::1]
    acc: cython.double[::1] = array('d', [0.0]) * (3 * n)
    for side in range(3):
        f = fine[side] if isinstance(fine[side], array) and fine[side].typecode == 'd' else array('d', fine[side])
        for i in range(len(f)):
            v = f[i]
            if v:
                acc[side * n + (i + offset) // phases] += v
    return [(math.hypot(acc[b], acc[n + b]) + acc[2 * n + b]) / BIN for b in range(n)]


def _star(level):
    return STAR_A * level ** STAR_B


def _horizon_levels(series):
    """Hardest window of each length (mean demand) → required level per horizon.
    A chart shorter than the window is judged whole at that window's tolerance: public
    scores say short maps earn no discount (a tolerance interpolated by length fit worse)."""
    n: cython.Py_ssize_t
    i: cython.Py_ssize_t
    d: cython.double
    acc: cython.double
    peak: cython.double
    previous: cython.double
    win: cython.double
    tol: cython.double
    out = []
    for win, tol in HORIZONS:
        n = min(max(1, int(win / BIN)), len(series))
        acc = peak = 0.0
        for i, d in enumerate(series):
            previous = series[i - n] if i >= n else 0.0
            acc += d - previous
            if acc > peak:
                peak = acc
        out.append(_star(SCALE * math.sqrt(peak / n) / tol))
    return out


def _combine(levels, notes):
    """Rating from the horizon levels: the binding (max) level, pulled toward the other horizons
    (a map hard at every horizon plays harder than one peaking at a single one) and discounted
    for note count (accuracy is averaged over every note)."""
    top = max(levels)
    if top <= 0:
        return 0.0
    f = 1.0
    for (win, _tol), lv, b in zip(HORIZONS, levels, H_WEIGHTS):
        f *= (max(lv, 1e-9) / top) ** b
    return top * f * (max(notes, 1) / 2000) ** G_NOTES


# Archetypes (card names for demands that meet in the same passages; at most 16 chars).
# Tech X: pattern X in passages with pattern or rhythm tech. X Stamina: pattern X sustained over 4 minutes.
TECH_NAMES = {"stream": "Tech Stream", "delay": "Tech Delay", "chordstream": "Tech Chordstream",
              "dump": "Tech Dump", "jumptrill": "Tech Jumptrill", "splittrill": "Tech Split Trill",
              "jumpstream": "Tech Jumpstream", "handstream": "Tech Handstream", "jack": "Tech Jacks",
              "jackspeed": "Tech Jacks", "chordjack": "Tech Chordjack", "minijack": "Tech Minijack",
              "longjack": "Tech Longjack", "bracket": "Tech Bracket", "hybrid": "Tech Hybrid LN",
              "inverse": "Tech Inverse", "release": "Tech LN Release", "ln": "Tech LN",
              "trill1h": "Tech 1H Trill", "anchor": "Tech Anchor"}
STAMINA_NAMES = {"stream": "Stream Stamina", "delay": "Delay Stamina", "chordstream": "Chordstream Stam",
                 "dump": "Dump Stamina", "jumptrill": "JT Stamina", "splittrill": "Split Stamina",
                 "jumpstream": "JS Stamina", "handstream": "HS Stamina", "jack": "Jack Stamina",
                 "jackspeed": "Jack Stamina", "longjack": "Jack Stamina", "chordjack": "CJ Stamina",
                 "bracket": "Bracket Stamina", "release": "LN Stamina", "inverse": "LN Stamina",
                 "hybrid": "LN Stamina", "ln": "LN Stamina"}
TECH_CONTROL = "Tech Control"   # rapid switching between patterns, no single one explaining the tech
TECH_KEYS = ("patterntech", "rhythmtech", "technical")   # never a bare card entry: shown through Tech X
# refinements: a specific pattern replaces its general parent on the card when it carries nearly all
# of the parent's rating (Bracket is how that chordstream moves, a longjack is that jack section).
# Minijack is not a refinement: bursty jack demand moves out of Chordjack/Jack into it.
REFINES = {"bracket": ("chordstream",), "quadstream": ("handstream", "minijack"),
           "longjack": ("jack", "jackspeed", "chordjack"),
           "vibro": ("jack", "jackspeed", "chordjack", "longjack"), "release": ("ln",), "inverse": ("ln",),
           "hybrid": ("ln",), "shield": ("ln",)}
# descriptors over single-note rice: their card entry already says that Stream / Delay is there
# (chord texture — Jumpstream, Handstream, Chordstream — still adds information)
DESCRIBES = {"dump": ("stream",), "trill1h": ("stream", "delay"), "anchor": ("stream", "delay")}
REFINE_Q = 0.9
ALSO_SHARE = 0.6     # the grey "also" line: LN at this share of the top card entry ...
ALSO_SPECIAL = 0.4   # ... SV and Vibro (specialist skills players seek out or avoid) already at this share
ARCH_MATERIAL = 0.6   # an archetype's pattern must reach this share of the strongest skill
ARCH_TECH = 0.75    # Tech X: X's technical passages rate at least this share of X; at TECH_WHOLE they
TECH_WHOLE = 0.85   # are X's hard passages and Tech X takes X's place, below it X keeps its own entry
STAMINA_SUSTAIN = 0.9  # X Stamina: X's hardest 4 minutes reach this share of X's rating


def _archetypes(series, sc, notes, switch=None):
    """Qualified archetypes [{"name", "rating", "parts"}] from the per-skill demand series.
    Co-location: a combination is rated on the passages where both are present (per-bin minimum of
    two series of the same total demand, so shared work is never counted twice).
    Tech X is rated on X's technical passages; when those are X's hard passages (TECH_WHOLE) it takes
    X's place, otherwise it is rated on them alone and shown only when X itself is not.
    Tech Control: Technical where patterns switch faster than they settle (the switch series),
    unless one Tech X explains nearly as much."""
    out = []
    top = max(v for k, v in sc.items() if k in series and k not in TECH_KEYS + ("ln",))
    # Quadstream is the name of a material short-jack texture, not a separate
    # additive skill. It can explain the hard minijack passages even when the
    # rest of a song is ordinary rice. Isolated quads do not reach this threshold.
    if "quadstream" in series and sc["quadstream"] >= .8*sc["minijack"] > 0 \
            and sc["quadstream"] >= ARCH_MATERIAL*sc["overall"]:
        out.append({"name":"Quadstream", "rating":sc["minijack"],
                    "parts":("quadstream","minijack")})
    tech = series["technical"]
    best = 0.0
    for k, name in TECH_NAMES.items():
        if k not in series or sc[k] < ARCH_MATERIAL * top:
            continue
        r = _combine(_horizon_levels([min(a, b) for a, b in zip(series[k], tech)]), notes)
        # A few awkward passages in a secondary component must not rename a
        # mostly smooth map. Technical remains measured in Details/Stats even
        # when it does not carry enough of the whole-map challenge for its title.
        if r >= ARCH_TECH * sc[k] and r >= .80 * sc["overall"]:
            whole = r >= TECH_WHOLE * sc[k]
            out.append({"name": name, "rating": sc[k] if whole else r,
                        "parts": (k, "technical") if whole else ("technical",), "of": k})
            best = max(best, sc[k] if whole else r)
    if switch:
        r = _combine(_horizon_levels([min(a, b) for a, b in zip(switch, tech)]), notes)
        if r >= ARCH_MATERIAL * top and best < REFINE_Q * r:
            out.append({"name": TECH_CONTROL, "rating": r, "parts": ("technical", "patterntech")})
    for k, parents in REFINES.items():
        if k not in series:
            continue
        ps = [p for p in parents if p in series and sc[k] >= REFINE_Q * sc[p] > 0]
        if k in series and ps:
            out.append({"name": NAMES[k], "rating": sc[k], "parts": (k, max(ps, key=sc.get))})
    teched = {a["of"] for a in out if a.get("of") in a["parts"]}
    for k, name in STAMINA_NAMES.items():
        # Tech X names X's passages more specifically than X Stamina; plain Stamina stays available
        if k not in series or sc[k] < ARCH_MATERIAL * top or k in teched:
            continue
        lv = _horizon_levels(series[k])
        if max(lv) > 0 and _stamina(series[k]) * sc[k] / max(lv) >= STAMINA_SUSTAIN * sc[k]:
            out.append({"name": name, "rating": sc[k], "parts": (k, "stamina")})
    return out


TECH_WHY = (("rodd", "rhythm: gaps off the pulse (e.g. 1/4 <-> 1/6)"), ("pins", "minijacks inserted in rice"),
            ("psw", "pattern transitions with repeated-column or irregular-hand pressure"))


def _tech_where(series, parts, start, rate, n=3):
    """The hardest Technical passages (2 s each, non-overlapping, at least half the hardest one):
    [{"t": song time s of the passage start (source chart time, not played time), "pattern", "rhythm":
    their shares of the passage's tech, "why": the leading evidence}]."""
    tech, w = series["technical"], 4
    sums = [sum(tech[i:i + w]) for i in range(max(1, len(tech) - w + 1))]
    out, taken = [], []
    for i in sorted(range(len(sums)), key=lambda i: -sums[i]):
        if len(out) == n or not sums[i] or sums[i] < 0.5 * sums[0 if not out else taken[0]]:
            break
        if any(abs(i - j) < w for j in taken):
            continue
        taken.append(i)
        pt, rt = sum(series["patterntech"][i:i + w]), sum(series["rhythmtech"][i:i + w])
        ev = {k: sum(parts[k][i:i + w]) / (RT_FULL if k == "rodd" else PT_FULL) for k, _ in TECH_WHY}
        out.append({"t": start + i * BIN * rate, "pattern": pt / (pt + rt or 1), "rhythm": rt / (pt + rt or 1),
                    "why": dict(TECH_WHY)[max(ev, key=ev.get)] if any(ev.values()) else ""})
    return out


def _stamina(total):
    n: cython.Py_ssize_t
    i: cython.Py_ssize_t
    d: cython.double
    acc: cython.double
    peak: cython.double
    previous: cython.double
    win: cython.double
    tol: cython.double
    n = int(STAMINA_WINDOW / BIN)
    acc = peak = 0.0
    for i, d in enumerate(total):
        previous = total[i - n] if i >= n else 0.0
        acc += d - previous
        peak = max(peak, acc)
    return _star(SCALE * math.sqrt(peak / n) / HORIZONS[-1][1])


def _attribute(parts, total, keys, *, technical_only=False):
    """Per-skill demand series. A ±1 s section counts toward each pattern in proportion to its
    share relative to the section's dominant pattern (a section can be Delay AND Chordstream;
    overall is unaffected — it is the total, charged once). Technical and Bracket are
    descriptors of the same total: Technical by the share of awkward rows (off-grid
    rhythm, lone flams) saturating at TECH_FULL, Bracket by the share of demand in bracket transitions saturating at BRACKET_FULL."""
    nb: cython.Py_ssize_t = len(total)
    b: cython.Py_ssize_t
    tot: cython.double
    top: cython.double
    rows: cython.double
    rt: cython.double
    pt: cython.double
    jr: cython.double
    burst: cython.double
    moved: cython.double
    pk = PARTS_4K if keys < 5 else PARTS_5K

    def win(v, h: int=2):            # ±h bins sums (±1 s)
        acc: cython.double = 0.0
        i: cython.Py_ssize_t
        out = [0.0] * nb
        for i in range(nb + h):
            if i < nb:
                acc += v[i]
            if i >= 2 * h + 1:
                acc -= v[i - 2 * h - 1]
            if i >= h:
                out[i - h] = acc
        return out
    needed = pk + ("technical", "rows")
    if not technical_only:
        needed += ("mj", "lj", "vib", "shield", "inv", "hyb", "oht", "anc", "sus", "ln_rel", "dump", "bracket")
    P = {k: win(v) for k, v in parts.items() if k in needed}
    rodd, rrows = win(parts["rodd"], RT_WIN), win(parts["rows"], RT_WIN)
    podd = win([a + c for a, c in zip(parts["pins"], parts["psw"])], RT_WIN)
    psw = win(parts["psw"], RT_WIN)
    # Tech is pattern/rhythm difficulty: measured on the demand without the scroll-reading part
    nosv = [max(0.0, t - s / BIN) for t, s in zip(total, parts["sv_raw"])]
    names = ("technical", "patterntech", "rhythmtech", "_switch") if technical_only else skills(keys) + ("_switch",)
    series = {k: [0.0] * nb for k in names if k != "stamina"}
    for b in range(nb):
        tot = sum(P[k][b] for k in pk) + P["technical"][b]
        if not tot or not total[b]:
            continue
        if not technical_only:
            top = max(P[k][b] for k in pk)
            for k in pk:
                series[k][b] = total[b] * P[k][b] / top
        rows = P["rows"][b]
        if rows:
            rt = min(1.0, max(0.0, rodd[b]) / (RT_FULL * rrows[b])) if rrows[b] > 0.5 else 0.0
            pt = min(1.0, max(0.0, podd[b]) / (PT_FULL * rrows[b])) if rrows[b] > 0.5 else 0.0
            series["rhythmtech"][b] = nosv[b] * rt
            series["patterntech"][b] = nosv[b] * pt
            series["_switch"][b] = nosv[b] * min(1.0, max(0.0, psw[b]) / (PT_FULL * rrows[b])) if rrows[b] > 0.5 else 0.0
            series["technical"][b] = nosv[b] * (1 - (1 - rt) * (1 - pt))
        if technical_only:
            continue
        if rows:
            series["minijack"][b] = total[b] * min(1.0, P["mj"][b] / (MJ_FULL * rows))
            series["longjack"][b] = total[b] * min(1.0, P["lj"][b] / (LJ_FULL * rows))
            series["vibro"][b] = total[b] * min(1.0, P["vib"][b] / (VIB_FULL * rows))
            series["shield"][b] = total[b] * min(1.0, P["shield"][b] / (SHIELD_FULL * rows))
            series["inverse"][b] = total[b] * min(1.0, P["inv"][b] / (INV_FULL * rows))
            series["hybrid"][b] = total[b] * min(1.0, P["hyb"][b] / (HYB_FULL * rows))
            series["trill1h"][b] = total[b] * min(1.0, P["oht"][b] / (OHT_FULL * rows))
            series["anchor"][b] = total[b] * min(1.0, P["anc"][b] / (ANC_FULL * rows))
        # jack strain in short bursts (then a break) is Minijack; Chordjack/Jack keep the sustained part
        jr = P["mj"][b] + P["sus"][b]
        if jr:
            burst = P["mj"][b] / jr
            for k in JACK_PARTS:
                if k in series:
                    moved = series[k][b] * burst
                    series[k][b] -= moved
                    series["minijack"][b] = max(series["minijack"][b], moved)
        if P["ln"][b] > 0:
            series["release"][b] = series["ln"][b] * min(1.0, P["ln_rel"][b] / (REL_FULL * P["ln"][b]))
        if "dump" in series:
            series["dump"][b] = total[b] * min(1.0, P["dump"][b] / (DUMP_FULL * tot))
        if "bracket" in series:
            series["bracket"][b] = total[b] * min(1.0, P["bracket"][b] / (BRACKET_FULL * tot))
    return series


def compute(chart, rate=1.0, sv=True, *, cancelled=None):
    """Full analysis of a parsed Chart at a constant rate. sv=False: scroll changes ignored
    (Constant Speed). Cancelled selections return no result and are never cached."""
    import difficulty_model
    config=difficulty_model.parameters().get('rolled_execution',{}).get('modes',{}).get(str(chart.keys),{})
    strength=config.get('strength',0.)
    if strength:
        import rolled_execution
        if not rolled_execution.possible(chart,rate):strength=0.
    shared={'trace':{},'execution_plans':True} if strength else {}
    result=_compute(chart, rate, sv, shared, cancelled)
    if strength and result.get('horizon'):
        import rolled_execution
        result=rolled_execution.apply(chart,rate,sv,result,shared,strength,cancelled)
    import archetype_names
    result['archetype_profile'] = archetype_names.profile(chart, rate)
    model=difficulty_model.parameters().get('score_units',{}).get('modes',{}).get(str(chart.keys))
    if model and result.get('horizon'):
        import score_units
        result['unit_card']=card(result)
        description=describe(result)[0]
        result=score_units.apply(result,difficulty_model.parameters())
        result['preunit_description']=description
    return result


def _compute(chart, rate, sv, shared, cancelled):
    # Only this calculation's identical chart/rate shares geometry with its
    # no-SV reference. Nothing survives a mutated chart or a different rate.
    from skill_calc import _check_cancelled
    left_value: cython.double
    right_value: cython.double
    if not rate > 0 or not math.isfinite(rate):
        raise ValueError(f"invalid rate: {rate}")
    _check_cancelled(cancelled)
    res = {"model": MODEL_VERSION, "sv_name": "SV", "keys": chart.keys, "rate": rate,
           "notes": len(chart.notes),
           "ln_notes": sum(1 for t, e, _ in chart.notes if e > t),
           "scores": {k: 0.0 for k in ("overall",) + skills(chart.keys)}, "horizon": None, "timeline": [],
           "sv": "constant" if not chart.sv else "included" if sv else "ignored"}
    if len(chart.notes) < 2:
        return res
    kinds = [] if sv else None
    reading = {}
    strain = sv_reading(chart, kinds, rate, reading, cancelled=cancelled) if sv else None
    _check_cancelled(cancelled)
    if not sv and 'plain_runs' in shared:
        runs = shared['plain_runs']
    else:
        plain_runs = [] if sv and chart.sv and any(strain) else None
        runs=[]
        prepared = _demand_geometry(chart,rate,cancelled=cancelled)
        shared['demand_geometry'] = prepared
        for cl in ([True,False] if chart.keys%2 else [True]):
            actions=[None]*len(chart.notes) if shared.get('execution_plans') else None
            runs.append(_demand(chart,rate,cl,strain,kinds,cancelled=cancelled,
                                plain_runs=plain_runs,motor_actions=actions,prepared=prepared))
            if actions is not None:shared.setdefault('motor_actions',{})[cl]=actions
        if plain_runs is not None:
            shared['plain_runs'] = plain_runs
    m = len(runs)
    if m == 2:
        total = [(left_value+right_value)/2 for left_value,right_value in zip(runs[0][1],runs[1][1])]
        parts = {k: [(left_value+right_value)/2 for left_value,right_value in zip(runs[0][0][k],runs[1][0][k])] for k in runs[0][0]}
    else:
        total = runs[0][1].copy()
        parts = {k: v.copy() for k,v in runs[0][0].items()}

    # hardest window over PHASES bin-grid offsets: a burst must not rate lower because the grid split it
    levels = _horizon_levels(total)
    phase_totals=[total] if 'trace' in shared else None
    fine_runs = [tuple(array('d', side) for side in run[2]) for run in runs]
    for p in range(1, PHASES):
        tp = _phase_total(fine_runs[0], p) if m == 1 else \
            [sum(x) / m for x in zip(*(_phase_total(fine, p) for fine in fine_runs))]
        levels = [max(a, b) for a, b in zip(levels, _horizon_levels(tp))]
        if phase_totals is not None: phase_totals.append(tp)
    sc = res["scores"]
    sc["overall"] = _combine(levels, len(chart.notes))
    res["horizon"] = HORIZONS[levels.index(max(levels))][0]
    res["levels"] = dict(zip((h[0] for h in HORIZONS), levels))
    sc["stamina"] = _stamina(total) * sc["overall"] / max(levels) if max(levels)>0 else 0.
    series = _attribute(parts, total, chart.keys)
    for k, v in series.items():
        if cancelled is not None:
            _check_cancelled(cancelled)
        if not k.startswith("_"):
            sc[k] = _combine(_horizon_levels(v), len(chart.notes))
    # Preserve the released score calibration's original physical coordinates.
    # New structural execution effects are fitted separately, so an unvalidated
    # keymode can retain identical difficulty while still getting honest labels.
    res["physical_sk"] = {k: v/max(.05, sc["overall"]) for k,v in sc.items()}
    import execution
    _check_cancelled(cancelled)
    series, parts, res["execution"] = execution.attribute(parts, total, series, chart, rate,
                                                         shared=shared, cancelled=cancelled)
    import pattern_control
    if 'tech_profile' not in shared:
        shared['tech_profile'] = pattern_control.tech_profile(chart, rate, shared['execution'])
    res['tech_profile'] = shared['tech_profile']
    switch = series.pop("_switch")
    for k, v in series.items():
        if cancelled is not None:
            _check_cancelled(cancelled)
        sc[k] = _combine(_horizon_levels(v), len(chart.notes))
    res["archetypes"] = _archetypes(series, sc, len(chart.notes), switch)
    res["tech_where"] = _tech_where(series, parts, chart.notes[0][0] / 1000.0, rate)
    sv_tot = sum(parts["sv"])
    shares = [sum(parts[k]) / sv_tot for k in SV_KINDS] if sv_tot else []
    # Several kinds occurring together are not proof of chaotic reading.
    # Without a clear dominant subtype, keep the honest generic name.
    res["sv_name"] = next((n for n, f in zip(SV_NAMES, shares) if f >= SV_KIND_SHARE),
                          "SV")
    # difficulty over time: the trailing 2 s window of the same demand series
    n = max(1, int(HORIZONS[0][0] / BIN))
    acc, tl = 0.0, []
    for i, d in enumerate(total):
        acc = max(0.0, acc + d - (total[i - n] if i >= n else 0.0))   # float drift
        tl.append(_star(SCALE * math.sqrt(acc / min(n, i + 1)) / HORIZONS[0][1]))
    res["timeline"] = tl
    k = rate ** RATE_G
    for key in sc:
        sc[key] *= k
    for a in res["archetypes"]:
        a["rating"] *= k
    res["timeline"] = [x * k for x in tl]
    import difficulty_model
    _check_cancelled(cancelled)
    if 'endurance' not in shared:
        # SV is added to the cross-hand reading track, not either hand's effort.
        shared['endurance'] = difficulty_model.endurance(runs, BIN, PHASES)
    res["endurance"], res["fatigue_timeline"] = shared['endurance']
    masks = {}
    for (_cancel_index, (t, _e, c)) in enumerate(chart.notes):
        if cancelled is not None and _cancel_index%256 == 0:
            _check_cancelled(cancelled)
        masks[t] = masks.get(t, 0) | (1 << c)
    rs = list(masks.items())
    pressure, pairs = 0., 0
    for (ta, a), (tb, b) in zip(rs, rs[1:]):
        gap = (tb - ta) / 1000. / rate
        if 0 < gap < .25:
            pressure += ((a & b).bit_count() / chart.keys) ** 2 * min(2., .125 / gap)
            pairs += 1
    res["jack_width"] = pressure / max(1, pairs)
    if sv and chart.sv and any(strain):
        # The hand/LN calibration must not change merely because an identical
        # physical chart acquired SV. Calibrate its physical coordinates once,
        # then add the locally aggregated reading demand in those same units.
        physical = _compute(chart, rate, False, shared, cancelled)
        res["no_sv_overall"] = physical["scores"]["overall"]
        res["reading"] = reading
        res["execution"]["scroll_base"] = {
            "raw": physical["raw_overall"], "features": physical["calibration_features"],
            "execution": physical["execution"]}
    import chart_structure
    _check_cancelled(cancelled)
    if 'structure' not in shared:
        shared['structure'] = chart_structure.structure(chart, rate)
    res['structure'] = shared['structure']
    res['length'] = max(e for _t,e,_c in chart.notes) - chart.notes[0][0]
    res=difficulty_model.apply(res, getattr(chart, "od", 8.), rate)
    # Keep the fitted critic's original coordinates above, then put horizon
    # ratings in the same rate-adjusted units as Overall and the timeline.
    res['levels'] = {h: v*k for h,v in res['levels'].items()}
    if 'trace' in shared:
        shared['trace'][bool(sv)]={'phases':phase_totals,'total':total,'series':series,'switch':switch,
                                   'parts':parts,'result':res}
        if shared.get('execution_plans'):
            shared['trace'][bool(sv)]['action_runs']={cl:(shared['motor_actions'][cl],run[2])
                for cl,run in zip([True,False] if chart.keys%2 else [True],runs)}
    return res


NAMES = {"stream": "Stream", "jumpstream": "Jumpstream", "handstream": "Handstream",
         "quadstream": "Quadstream", "sv_fast": "Fast SV", "sv_slow": "Slowjam SV",
         "sv_accel": "Accel SV", "sv_stutter": "Stutter SV", "sv_brake": "Brakes",
         "jackspeed": "Jackspeed", "chordjack": "Chordjack", "delay": "Delay",
         "chordstream": "Chordstream", "bracket": "Bracket", "jack": "Jack",
         "dump": "Dump", "jumptrill": "Jumptrill", "splittrill": "Split Trill",
         "technical": "Technical", "patterntech": "Pattern Tech", "rhythmtech": "Rhythm Tech",
         "trill1h": "1H Trill", "anchor": "Anchor", "ln": "LN", "sv": "SV", "stamina": "Stamina",
         "minijack": "Minijack", "longjack": "Longjack", "release": "LN Release", "inverse": "Inverse",
         "hybrid": "Hybrid LN", "shield": "Shield", "vibro": "Vibro", "mash": "Mash"}


def skill_names(keys):
    """Display names of the keymode's skills."""
    return {k: NAMES[k] for k in skills(keys)}


def ranked(res):
    """The keymode's skills, strongest first (ties keep display order: patterns before Stamina)."""
    sc = res["scores"]
    return sorted(skills(res["keys"]), key=lambda k: -round(sc[k], 2))


def card(res):
    """Up to 3 non-redundant card entries [{"name", "rating", "parts"}]. Archetypes and plain skills
    compete by rating (an archetype first on a tie); an entry is shown only if none
    of its parts is already shown, and only shown entries use up their parts. A descriptor entry
    (Dump, 1H Trill, Anchor) also covers the stream-family parts it describes. Technical, Pattern Tech
    and Rhythm Tech are shown only through Tech X / Tech Control."""
    if 'unit_card' in res:
        return [dict(e) for e in res['unit_card']]
    sc = res["scores"]
    if sc["overall"] < 0.1:
        return []
    names = dict(skill_names(res["keys"]), sv=res.get("sv_name", "SV"))
    # an archetype competes at the rating of its strongest component, so X Stamina / Longjack stand in
    # for Stamina / Jack wherever those would have been shown
    # (a descriptor's archetype — Tech Dump — before one of the pattern it describes on a tie)
    cands = [(max([a["rating"]] + [sc[p] for p in a["parts"] if p not in TECH_KEYS]),
              0 if a.get("of") in DESCRIBES or "ln" in a["parts"] and any(
                  p in LN_KINDS for p in a["parts"]) else 1, a) for a in res.get("archetypes", ())]
    cands += [(sc[k], 2, {"name": names[k], "rating": sc[k], "parts": (k,)})
              for k in ranked(res) if k not in TECH_KEYS + SV_KINDS and sc[k] > 0]
    picked, used = [], set()
    for _r, _o, e in sorted(cands, key=lambda c: (-round(c[0], 6), c[1])):
        if len(picked) == 3 or picked and e["rating"] < 0.3 * picked[0]["rating"]:
            break
        if used & set(e["parts"]) or e.get("of") in used:
            continue
        picked.append({k: v for k, v in e.items() if k != "of"})
        used |= set(e["parts"]) | {m for p in e["parts"] + (e.get("of", ""),) for m in DESCRIBES.get(p, ())}
    # An LN refinement represents the LN family that won the selection above;
    # its slightly smaller release-only rating must not demote that family.
    picked = sorted(picked, key=lambda e: -max(e["rating"], sc["ln"] if "ln" in e["parts"]
        and any(p in LN_KINDS for p in e["parts"]) else 0.))
    if picked:
        import pattern_control
        first = picked[0]
        name = pattern_control.technical_title(first['name'],
            {'technical': round(sc['technical']/max(.05,sc['overall']),4)}, res.get('tech_profile'))
        if name != first['name']:
            first['name'] = name
            first['parts'] = tuple(dict.fromkeys((*first['parts'], 'technical')))
    import archetype_names
    return archetype_names.card(picked, {key:value for key,value in sc.items() if key != 'overall'},
                                res.get('archetype_profile'), res['keys'])


def describe(res):
    """→ (dominant text, [structural tags]) from a compute() result."""
    sc = res["scores"]
    entries = card(res)
    if not entries:
        return "", []
    a = entries[0]
    dom = a["name"] + (f" / {entries[1]['name']}" if len(entries) > 1 and entries[1]["rating"] >= 0.85 * a["rating"] else "")
    dom = res.get('preunit_description', dom)
    tags = []
    lv = res["levels"]
    if res["horizon"] == 2.0 and lv[30.0] < 0.75 * sc["overall"]:
        tags.append("short burst")
    # demands that change how the map plays, strong but not on the card (SV, LN, vibro)
    shown = {p for e in entries for p in e["parts"]}
    names = dict(skill_names(res["keys"]), sv=res.get("sv_name", "SV"))
    extra = []
    for group, share in ((("sv",), ALSO_SPECIAL), (LN_KINDS + ("ln",), ALSO_SHARE),
                         (("vibro",), ALSO_SPECIAL), (("mash",), .25)):
        if not shown & set(group):
            k = max((g for g in group if g in sc), key=sc.get)
            if k == "mash" and res.get("execution", {}).get("mash", 0.) < .05 and sc[k] < .8*sc["overall"]:
                continue                 # a few ordinary full chords are not a map identity
            if sc[k] >= share * a["rating"]:
                extra.append(f"{names[k]} {sc[k]:.1f}")
    if extra:
        tags.append("also " + " · ".join(extra))
    return dom, tags


def compute_all(osu_path, rate=1.0):
    """Flat {skill: rating} for a file (CLI / scripting)."""
    return {k: round(v, 2) for k, v in compute(parse_osu(osu_path), rate)["scores"].items()}


def main():
    import sys
    import json
    if len(sys.argv) < 2:
        print("usage: skill_calc.py <file.osu> [rate]", file=sys.stderr)
        return 2
    rate = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    res = compute(parse_osu(sys.argv[1]), rate)
    res["scores"] = {k: round(v, 2) for k, v in res["scores"].items()}
    res["described"] = describe(res) if res["horizon"] else None
    res.pop("timeline")
    print(json.dumps(res, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
