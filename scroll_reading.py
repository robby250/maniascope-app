"""Scroll-reading geometry in played time, not a proxy for player specialization.

The sequential-scroll integral follows ppy/osu. The analysis separates shortened
preview, additional same-column crowding, broad acceleration/braking and fine
displacement hidden by coarse averaging. An adjustable constant scroll multiplier
does not itself create SV difficulty. Coefficients belong to score calibration.
"""
import bisect
import math
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import cython
import statistics
from collections import deque

FEATURES = ("preview", "crowding", "motion", "stutter")
DEFAULT_WEIGHTS = (.8, .7, .8, 1.6)
VISIBILITY_MS = 450.


def geometry(chart, rate: float=1.0, *, cancelled=None):
    from skill_calc import _check_cancelled
    t: cython.double
    start: cython.double
    end: cython.double
    half_approach: cython.double
    reference: cython.double
    nominal: cython.double
    p: cython.double
    entry: cython.double
    seen: cython.double
    preview: cython.double
    look: cython.double
    a: cython.double
    distance: cython.double
    mean: cython.double
    deviation: cython.double
    visible_fraction: cython.double
    total_variation: cython.double
    broad: cython.double
    fine: cython.double
    peak: cython.double
    point: cython.double
    expected: cython.double
    novelty: cython.double
    motion: cython.double
    preview_pressure: cython.double
    crowd: cython.double
    normal: cython.double
    actual: cython.double
    extra: cython.double
    n: cython.Py_ssize_t
    sample_count: cython.Py_ssize_t
    i: cython.Py_ssize_t
    j: cython.Py_ssize_t
    lo: cython.Py_ssize_t
    hi: cython.Py_ssize_t
    if rate <= 0 or not math.isfinite(rate):
        raise ValueError("rate must be positive and finite")
    n = len(chart.notes)
    empty = {"vectors": [(0.,)*4]*n, "kinds": [(0.,)*5]*n,
             "summary": dict.fromkeys(FEATURES, 0.)}
    if not chart.sv or not n:
        return empty
    # Redundant control points have no visible effect. Apart from saving work,
    # removing them prevents mapper sampling resolution from affecting features.
    controls = []
    for t, v in chart.sv:
        if not controls or v != controls[-1][1]:
            controls.append((t, v))
    ts, velocities = zip(*controls)
    # Estimate one normal scroll setting from sustained approach-speed samples,
    # not instantaneous control-point velocities. A mode can choose the stopped
    # half of a stop/lunge cycle; an overall mean lets a few extreme lunges hide
    # all the normal reading. The median of time-uniform approach averages avoids
    # both, and does not depend on how densely a mapper sampled their curves.
    start, end = chart.notes[0][0], max(e for _,e,_ in chart.notes)
    if max(velocities)-min(velocities) < 1e-6 or end <= start:
        return empty
    raw_positions = [0.]
    for i in range(1,len(ts)):
        raw_positions.append(raw_positions[-1]+(ts[i]-ts[i-1])*velocities[i-1])
    def raw_pos(t):
        i=max(0,bisect.bisect_right(ts,t)-1)
        return raw_positions[i]+(t-ts[i])*velocities[i]
    sample_count=min(2048,max(64,n))
    half_approach=VISIBILITY_MS*rate/2
    averages=[]
    for i in range(sample_count):
        t=start+(end-start)*(i+.5)/sample_count
        left,right=max(start,t-half_approach),min(end,t+half_approach)
        averages.append((raw_pos(right)-raw_pos(left))/(right-left))
    reference=max(1e-6,statistics.median(averages))
    vs=[max(1e-6,v/reference) for v in velocities]
    xs=[p/reference for p in raw_positions]

    def pos(t):
        i = max(0,bisect.bisect_right(ts,t)-1)
        return xs[i]+(t-ts[i])*vs[i]

    def at(position):
        i = max(0,bisect.bisect_right(xs,position)-1)
        return ts[i]+(position-xs[i])/vs[i]

    # Lazer scales chart-time TimeRange by tempo/frequency to retain the chosen
    # played-time scroll speed under HT/DT. Multiplying here applies rate once.
    nominal = VISIBILITY_MS*rate
    positions = [pos(t) for t,_,_ in chart.notes]
    neighbours = [[] for _ in chart.notes]
    previous = {}
    for i,(t,e,c) in enumerate(chart.notes):
        if cancelled is not None and i%256 == 0:
            _check_cancelled(cancelled)
        previous_index = previous.get(c)
        if previous_index is not None and t>chart.notes[previous_index][0]:
            neighbours[i].append(previous_index); neighbours[previous_index].append(i)
        previous[c] = i
    by_time, vectors, kinds = {}, [], []
    history = deque(maxlen=12)
    for i,(t,e,c) in enumerate(chart.notes):
        if cancelled is not None and i%256 == 0:
            _check_cancelled(cancelled)
        current = by_time.get(t)
        if current is None:
            p = positions[i]
            entry = at(p-nominal)
            seen = max(.001,t-entry)
            preview = max(0.,1.-seen/nominal)
            # Very long slow approaches should not require scanning tens of
            # seconds per note. Crowding is measured separately from neighbours.
            look = min(seen,2*nominal)
            a = t-look
            distance = max(1e-6,p-pos(a))
            mean = distance/look
            lo,hi = bisect.bisect_right(ts,a),bisect.bisect_left(ts,t)
            if lo == hi:
                # A straight approach has no acceleration or stutter. Keep
                # preview/crowding costs, including a sustained faster segment.
                fingerprint = (1/3, 1/3, 1/3, min(1.,seen/nominal))
                current = preview,0.,0.,False,min(1.,nominal/seen)
            else:
                boundaries = [a]
                boundaries.extend(ts[lo:hi]); boundaries.append(t)
                deviation = 0.
                turns = []
                last_direction, previous_v = 0, None
                for left,right in zip(boundaries,boundaries[1:]):
                    v = vs[max(0,bisect.bisect_right(ts,(left+right)*.5)-1)]
                    deviation += abs(v-mean)*(right-left)
                    # Tiny control-line flicker is only relevant in proportion to
                    # actual displacement/time, not because there are many lines.
                    direction = (1 if v>previous_v*1.05 else -1 if v<previous_v/1.05 else 0) if previous_v else 0
                    if direction and last_direction and direction!=last_direction:
                        turns.append(left)
                    if direction: last_direction=direction
                    previous_v=v
                # When a long slow approach is clipped to the look-back horizon,
                # its last tiny movement is not a whole playfield traversal. Using
                # that tiny distance as the sole denominator amplifies invisible
                # jitter into a full-strength stutter. Keep the actual screen scale.
                visible_fraction = min(1.,distance/nominal)
                total_variation = min(2.,deviation/distance)*visible_fraction
                thirds = [max(0.,pos(a+(j+1)*look/3)-pos(a+j*look/3))/distance for j in range(3)]
                broad = min(total_variation,sum(abs(v-1/3) for v in thirds)*visible_fraction)
                fast_cycle = any(b-a < nominal/3 for a,b in zip(turns,turns[2:]))
                fine = max(0.,total_variation-broad) if fast_cycle else 0.
                if fine:
                    # Velocity can oscillate wildly while the note barely deviates
                    # from a straight trajectory. Measure the actual on-screen
                    # displacement from each coarse third's linear approach. The
                    # extrema of a piecewise-linear path occur at its boundaries.
                    # Six scales a half-third-playfield displacement to one unit.
                    edges = [a+j*look/3 for j in range(4)]
                    edge_positions = [pos(x) for x in edges]
                    peak = 0.
                    for point in boundaries[1:-1]:
                        j = min(2,int((point-a)*3/look))
                        expected = edge_positions[j] + (edge_positions[j+1]-edge_positions[j])*(point-edges[j])/(look/3)
                        peak = max(peak,abs(pos(point)-expected))
                    fine = min(fine,6*peak/nominal)
                elif not fast_cycle:
                    broad = total_variation
                fingerprint = tuple(thirds)+(min(1.,seen/nominal),)
                recent = [v for when,v in history if t-when <= 1500*rate]
                novelty = 0. if fingerprint in recent else min(
                    (sum(abs(x-y) for x,y in zip(fingerprint,old)) for old in recent), default=1.)
                novelty = min(1.,novelty/.5)
                # A repeated broad, predictable brake is easier to anticipate, but
                # this never removes its crowding or the fine stutter displacement.
                motion = broad*(.4+.6*novelty)
                accel = thirds[-1] > thirds[0]
                current = preview,motion,fine,accel,min(1.,nominal/seen)
            by_time[t] = current
            history.append((t,fingerprint))
        preview,motion,fine,accel,preview_pressure = current
        crowd = 0.
        for j in neighbours[i]:
            normal = abs(chart.notes[j][0]-t)/nominal
            actual = abs(positions[j]-positions[i])/nominal
            # Additional overlap relative to these SAME notes at constant speed;
            # ordinary fast jacks were already charged by the physical model.
            extra = max(0.,1.-actual/.10)-max(0.,1.-normal/.10)
            crowd = max(crowd, max(0.,extra))
        # Crowding is a decoding deadline, not just small pixel spacing. The
        # same stack visible several seconds in advance offers more preparation
        # than a last-moment compressed arrival. Do not charge both identically.
        crowd *= preview_pressure
        vector = (preview,crowd,motion,fine)
        vectors.append(vector)
        kinds.append((preview,crowd,motion if accel else 0.,fine,motion if not accel else 0.))
    # Note-weighted diagnostic features only. The actual rating retains the
    # complete time series and existing hard-section/endurance aggregation.
    return {"vectors":vectors, "kinds":kinds,
            "summary":{key:sum(v[j] for v in vectors)/n for j,key in enumerate(FEATURES)}}


def reading(chart, rate: float=1.0, kinds=None, detail=None, *, cancelled=None):
    from skill_calc import _check_cancelled
    import difficulty_model
    info = geometry(chart,rate, cancelled=cancelled)
    config = difficulty_model.parameters().get("scroll_reading", {})
    config = config.get("modes", {}).get(str(chart.keys), config)
    weights = config.get("weights", DEFAULT_WEIGHTS)
    if len(weights)!=4 or any(not math.isfinite(w) or w<0 for w in weights):
        raise ValueError("Invalid scroll-reading coefficients")
    out = []
    for (_cancel_index, (vector, kind)) in enumerate(zip(info['vectors'], info['kinds'])):
        if cancelled is not None and _cancel_index%256 == 0:
            _check_cancelled(cancelled)
        values = [v*w for v,w in zip(vector,weights)]
        total = sum(values)
        out.append(total)
        if kinds is not None:
            # Macro motion occupies exactly one directional subtype.
            ks = (values[0],values[1],values[2] if kind[2] else 0.,values[3],values[2] if kind[4] else 0.)
            kinds.append(tuple(v/total for v in ks) if total>0 else (0.,)*5)
    if detail is not None:
        detail.update(info["summary"])
    return out
