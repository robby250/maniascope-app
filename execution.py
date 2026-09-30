"""Structural execution evidence shared by difficulty, labels and prediction.

Compact pseudo-chords are recognised from time/column occupancy, never titles.
Smooth hand-local pulses do not become Tech merely because their interleaved
global rows change rhythm. None of these descriptors is an unconditional map
difficulty discount: their scalar effects are fitted to player scores.
"""
import bisect
import collections
import math
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import cython


def smooth(x, lo, hi):
    x = (x-lo)/(hi-lo)
    if not x > 0.:
        return 0.
    if x >= 1.:
        return 1.
    return x*x*(3-2*x)


def analyse(chart, rate: float, bin_seconds: float=0.5, *, cancelled=None, prepared=None):
    from skill_calc import _check_cancelled
    inv: cython.double
    first: cython.double
    width: cython.double
    coverage: cython.double
    similarity: cython.double
    compact: cython.double
    value: cython.double
    tail_count: cython.double
    synced: cython.double
    ref: cython.double
    chosen: cython.double
    original_value: cython.double
    pulse: cython.double
    original_pulse: cython.double
    gap: cython.double
    nearest: cython.double
    irregular: cython.double
    nt: cython.double
    end: cython.double
    t: cython.double
    e: cython.double
    v: cython.double
    following: cython.double
    bins: cython.Py_ssize_t
    i: cython.Py_ssize_t
    p: cython.Py_ssize_t
    q: cython.Py_ssize_t
    ri: cython.Py_ssize_t
    b: cython.Py_ssize_t
    mask: cython.Py_ssize_t
    half: cython.Py_ssize_t
    hand: cython.Py_ssize_t
    col: cython.Py_ssize_t
    import skill_calc
    inv = .001/rate
    notes = prepared['notes'] if prepared is not None else [(t*inv, e*inv, c) for t, e, c in chart.notes]
    if not notes:
        return {"mash": [], "pseudo": [], "regular": [], "switch": [], "flow": 0., "release_sync": 0.}
    first = notes[0][0]
    bins = int((max(e for _, e, _ in notes)-first)/bin_seconds)+1
    # At most a great-window-wide timing offset on either side. The compactness
    # check below rejects merging a continuous jumptrill's alternating hands.
    width = 2*max(.015, min(.045, (64-3*chart.od)/1000))
    groups, group, mask = [], [], 0
    for i, (t, e, c) in enumerate(notes):
        if cancelled is not None and i%256 == 0:
            _check_cancelled(cancelled)
        if group and (t-group[0][0] > width or mask & (1 << c)):
            groups.append(group); group = []; mask = 0
        group.append((t, e, c, i)); mask |= 1 << c
    if group:
        groups.append(group)
    masks = [sum(1 << c for t, e, c, i in g) for g in groups]
    centers = [sum(t for t, e, c, i in g)/len(g) for g in groups]
    spans = [g[-1][0]-g[0][0] for g in groups]
    note_mash, note_pseudo = [0.]*len(notes), [0.]*len(notes)
    for i in range(1, len(groups)-1):
        gs = groups[i-1:i+2]
        gaps = [centers[i]-centers[i-1], centers[i+1]-centers[i]]
        if min(gaps) < .045 or max(gaps) > .22 or max(gaps) > 1.65*min(gaps):
            continue
        coverage = sum(len(g) for g in gs)/(3*chart.keys)
        similarity = sum((a & b).bit_count()/max(1, (a | b).bit_count())
                         for a, b in zip(masks[i-1:i+1], masks[i:i+2]))/2
        compact = sum(1-smooth(spans[j]/max(.001, sum(gaps)/2), .30, .47)
                      for j in range(i-1, i+2))/3
        value = smooth(coverage, .65, .95)*smooth(similarity, .55, .9)*compact
        # Long independently ending holds still require actual finger control.
        # Co-releases/short shields can share a bounce, independent tails cannot.
        ends = [e for t, e, c, n in groups[i] if e-t > .04]
        if ends:
            value *= 1-.85*smooth(max(ends)-min(ends), width, .25)
        for t, e, c, n in groups[i]:
            note_mash[n] = value
            note_pseudo[n] = value if spans[i] > .006 else 0.

    sums = {k: [0.]*bins for k in ("mash", "pseudo", "regular", "switch", "regular_cal", "switch_cal", "count")}
    if prepared is None:
        rowsets, indices, p = [], [], 0
        while p < len(notes):
            q = p+1
            while q < len(notes) and notes[q][0]-notes[p][0] <= .003:
                q += 1
            rowsets.append((notes[p][0], frozenset(c for _, _, c in notes[p:q])))
            indices.append((p,q)); p=q
    else:
        rowsets, indices = prepared['rows'], prepared['boundaries']
    # Track each hand's actual pulse separately. Interleaving two regular hand
    # rhythms can create irregular global gaps without irregular hand timing.
    half = chart.keys//2
    import pattern_control
    roll_membership = pattern_control.smooth_rolls(rowsets, chart.keys)
    last, history = [None,None], [collections.deque(maxlen=6), collections.deque(maxlen=6)]
    head_times = [t for t, e, c in notes]
    tail_count, synced = 0., 0.
    for ri, ((t, cols), (p,q)) in enumerate(zip(rowsets, indices)):
        if cancelled is not None and ri%256 == 0:
            _check_cancelled(cancelled)
        b = min(bins-1, int((t-first)/bin_seconds))
        active = {0 if c < half else 1 for c in cols}
        awkward, original_awkward = [], []
        for hand in active:
            g = None if last[hand] is None else t-last[hand]
            if g and .006 < g < .5:
                recent = sorted(history[hand])
                ref = recent[len(recent)//2] if len(recent) >= 2 else g
                original_value = skill_calc._off_grid(g, ref)
                original_awkward.append(original_value)
                # A skip in a hand's pulse may be a whole multiple of the
                # shorter pulse, not a new syncopation relative to its median.
                chosen = ref
                # The first repeated value in sorted history must have an
                # adjacent match; there is no need to rescan the neighbourhood.
                for v, following in zip(recent, recent[1:]):
                    if v > ref:
                        break
                    if following-v < .004:
                        chosen = v
                        break
                value = original_value if chosen == ref else skill_calc._off_grid(g, chosen)
                awkward.append(value*(1.-roll_membership[hand][ri]))
                history[hand].append(g)
            else:
                history[hand].clear()
            last[hand] = t
        # A family transition only becomes pattern-control evidence when the
        # boundary also forces a repeat or non-grid hand pulse. Speed alone is
        # retained in the physical calculator, not counted again as Tech.
        prev_cols = rowsets[ri-1][1] if ri else frozenset()
        gap = t-rowsets[ri-1][0] if ri else 1.
        repeat = bool(cols & prev_cols) and gap < .15
        pulse = max(awkward, default=0.)
        original_pulse = max(original_awkward, default=0.)
        sums["regular"][b] += pulse
        sums["switch"][b] += max(pulse, .8 if repeat else 0.)
        sums["regular_cal"][b] += original_pulse
        sums["switch_cal"][b] += max(original_pulse, .8 if repeat else 0.)
        sums["mash"][b] += sum(note_mash[p:q])/(q-p)
        sums["pseudo"][b] += sum(note_pseudo[p:q])/(q-p)
        sums["count"][b] += 1
        for nt, end, col in notes[p:q]:
            if end-nt < .06:
                continue
            tail_count += 1
            at = bisect.bisect_left(head_times, end)
            nearest = min((abs(head_times[x]-end) for x in (at-1, at) if 0 <= x < len(head_times)), default=1.)
            synced += 1-smooth(nearest, .003, .025)
    for k in ("mash", "pseudo", "regular", "switch", "regular_cal", "switch_cal"):
        sums[k] = [v/max(1., n) for v, n in zip(sums[k], sums["count"])]
    sums["release_sync"] = synced/max(1., tail_count)
    # Synchronous tails in a stable pulse are a distinct, score-testable kind
    # of LN coordination, rather than an assertion that all LNs are easier.
    irregular = sum(sums["regular_cal"][i]*n for i,n in enumerate(sums["count"]))/max(1.,sum(sums["count"]))
    sums["flow"] = sums["release_sync"]*(1-min(1., irregular))*tail_count/max(1.,len(notes))
    return sums


def attribute(parts, total, original, chart, rate: float, *, shared=None, cancelled=None):
    """Reassign overlapping descriptions, without double-charging any demand."""
    import skill_calc
    if shared is None:
        shared = {}
    if 'execution' not in shared:
        shared['execution'] = analyse(chart, rate, cancelled=cancelled, prepared=shared.get('demand_geometry'))
    info = shared['execution']
    skill_calc._check_cancelled(cancelled)
    changed = dict(parts)
    m = info["mash"]
    # Keep inserted repeats (genuine jack control); remove only their mash
    # overlap. Smooth switches need independent physical boundary evidence.
    changed["pins"] = [v*(1-s) for v,s in zip(parts["pins"], m)]
    changed["psw"] = [v*g*(1-s) for v,g,s in zip(parts["psw"],info["switch"],m)]
    # Evidence belongs to the actual affected interval. Taking the maximum
    # anywhere in a four-second neighbourhood made one awkward hand impulse
    # validate every unrelated global-grid deviation around a smooth sweep.
    hand_gate = [min(1., value/skill_calc.RT_FULL) for value in info["regular"]]
    changed["rodd"] = [v*g*(1-s) for v,g,s in zip(parts["rodd"],hand_gate,m)]
    # Only these four series change; the other labels retain their first pass.
    corrected = skill_calc._attribute(changed, total, chart.keys, technical_only=True)
    for name in ("technical", "patterntech", "rhythmtech", "_switch"):
        original[name] = corrected[name]
    original["mash"] = [max(0.,t-sv/skill_calc.BIN)*s for t,sv,s in zip(total,parts["sv_raw"],m)]
    # The special category owns the compact bounce portions, not independent
    # normal chordstreams/LN control elsewhere in the map.
    for name in ("stream", "delay", "dump", "jumpstream", "handstream", "chordstream", "minijack", "bracket"):
        if name in original:
            original[name] = [v*(1-s) for v,s in zip(original[name],m)]
    ln_shared = info["release_sync"]
    for name in ("ln", "release", "shield"):
        if name in original:
            original[name] = [v*(1-s*ln_shared) for v,s in zip(original[name],m)]
    weight = max(1., sum(total))
    mash = sum(t*v for t,v in zip(total,m))/weight
    pseudo = sum(t*v for t,v in zip(total,info["pseudo"]))/weight
    smooth_switch = sum(v*(1-g) for v,g in zip(parts["psw"],info["switch_cal"]))/max(1.,sum(parts["rows"]))
    # Short high-demand transitions are weighted by the demand they carry,
    # not by map length or the mere existence of a single note burst.
    mean = sum(total)/max(1,len(total))
    peak = sum(t*max(0., math.log(max(1., t/max(1.,mean)))) for t in total)/weight
    surge = sum(max(0., t-previous)*math.log1p(max(0., t-previous)/max(1.,mean))
                for previous,t in zip(total,total[1:]))/weight
    extra = {"mash": mash, "pseudo_mash": pseudo, "ln_flow": info["flow"],
             "smooth_switch": smooth_switch, "peak_pressure": peak, "transition_pressure": surge}
    import gestures
    if 'gestures' not in shared:
        shared['gestures'] = (gestures.analyse(chart, rate, skill_calc.BIN,with_plans=True, cancelled=cancelled)
                              if shared.get('execution_plans') else gestures.analyse(chart,rate,skill_calc.BIN, cancelled=cancelled))
    motion = shared['gestures']
    skill_calc._check_cancelled(cancelled)
    g = motion["series"]
    nosv = [max(0.,t-sv/skill_calc.BIN) for t,sv in zip(total,parts["sv_raw"])]
    # The old per-column frequency test also labelled ordinary ultra-fast rolls
    # Vibro. Only actual repeated-row execution now owns that descriptor.
    original["vibro"] = [t*min(1.,v/.8) for t,v in zip(nosv,g["vibro"])]
    rice = [k for k in ("stream","delay","dump","jumpstream","handstream","chordstream") if k in original]
    # Relative-time gesture binning can include an exact endpoint that the
    # legacy floating-point demand horizon rounds down. Never index past the
    # existing physical timeline; completed in-range results are unchanged.
    for b,strength in enumerate(g["rolled"][:len(total)]):
        if strength <= 0:
            continue
        # Move only the eligible rice share. Anchors, jacks, split trills and LN
        # releases retain their independent demands; no duplicate strain charge.
        moved = max((original[k][b] for k in rice), default=0.)*strength
        original["jumptrill"][b] = max(original["jumptrill"][b], moved)
        for k in rice:
            original[k][b] *= 1.-strength
    extra.update(motion["features"])
    import pattern_control
    if 'quadstream' not in shared:
        shared['quadstream'] = pattern_control.quadstream(chart, rate, skill_calc.BIN)
    quad, share = shared['quadstream']
    extra["full_chord_stream_share"] = share
    if chart.keys == 4:
        # Use the existing short-jack materiality scale: this names a minijack
        # texture, not a new independent strength or an extra strain surcharge.
        original["quadstream"] = [t*min(1.,s/skill_calc.MJ_FULL) for t,s in zip(nosv,quad)]
    # Subtypes are views of the existing SV demand, never new additive charges.
    for key in skill_calc.SV_KINDS:
        original[key] = [v*min(1.,parts[key][b]/max(1e-9,parts['sv'][b]))
                         for b,v in enumerate(original['sv'])]
    return original, changed, extra
