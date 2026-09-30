"""Timing-feasible hand gestures, not a claim about a player's muscles.

An ordinary stream stays a stream. Extremely compact taps can share an
alternating hand pulse, even when roll directions change. Repeated columns,
independent holds and incompatible timing still require their original control.
Vibro evidence requires an actual repeating-row run, not merely a fast column
recurrence inside a roll. These signals do not themselves discount difficulty.
"""
import math
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import cython


FEATURES = ("vibro_easy", "vibro_control", "vibro_sustain",
            "jumptrill_easy", "roll_control", "roll_anchor", "jumptrill_sustain",
            "vibro_locked")
SERIES = ("rolled", "jumptrill", "control", "anchor", "vibro") + FEATURES


def smooth(x, lo, hi):
    x = (x-lo)/(hi-lo)
    if not x > 0.:
        return 0.
    if x >= 1.:
        return 1.
    return x*x*(3.-2.*x)


def great_window(od):
    """Nominal half-window in played seconds; never divided by playback rate.

    ppy/osu ManiaHitWindows: Great ranges 64/49/34 ms at OD 0/5/10.
    A feasible Great is not proof of a Perfect or of a player's chosen technique.
    """
    return (64.-3.*min(10., max(0., od)))/1000.


def _packets(notes, rows, hand_of, window: float, *, cancelled=None):
    """Each hand's distinct-column impulses, with original note times retained.

    Greedy packing stops at a repeated column, a hold head, a literal chord
    boundary, or a timing-infeasible span. Other held fingers add control demand
    but do not prohibit the free fingers from rolling. It never changes a hold.
    """
    from skill_calc import _check_cancelled
    group_hold: cython.double
    t: cython.double
    e: cython.double
    segment: cython.Py_ssize_t
    h: cython.Py_ssize_t
    i: cython.Py_ssize_t
    packets = []
    # Global stream continuity, not same-hand adjacency: at 25 ms per row,
    # 1 4 2 3 can be rolled as alternating hands even though each hand's notes
    # are 50 ms apart. A real gap still ends the candidate gesture.
    segments, segment = [], 0
    for i,(t,e,c) in enumerate(notes):
        if cancelled is not None and i%256 == 0:
            _check_cancelled(cancelled)
        if i and t-notes[i-1][0] >= .045:
            segment += 1
        segments.append(segment)
    for h in (0, 1):
        group, used, holds = [], set(), {}
        group_hold = 0.

        def close():
            direction: cython.double
            nonlocal group_hold
            if group:
                if notes[group[0]][0] == notes[group[-1]][0]:
                    # Preserve the original floating-point mean even for a
                    # literal chord; its span, direction and inner gaps are zero.
                    ts = [notes[group[0]][0]] * len(group)
                    packets.append((sum(ts)/len(ts), h, tuple(group),
                                    frozenset(notes[i][2] for i in group), 0., 0., 0., group_hold))
                else:
                    ts = [notes[i][0] for i in group]
                    order = [notes[i][2] for i in group]
                    directions = [1 if b > a else -1 for a,b in zip(order,order[1:])]
                    # Simultaneous chord line order is not a roll direction.
                    direction = (sum(directions)/len(directions)
                                 if directions and ts[-1]-ts[0] > .003 else 0.)
                    packets.append((sum(ts)/len(ts), h, tuple(group), frozenset(order),
                                    ts[-1]-ts[0], direction,
                                    max((notes[i][0]-notes[i-1][0]
                                         for i in range(group[0]+1,group[-1]+1)), default=0.), group_hold))
                group.clear(); used.clear()
                group_hold = 0.

        for (_row_number, (t, ids)) in enumerate(rows):
            if cancelled is not None and _row_number%256 == 0:
                _check_cancelled(cancelled)
            local = [i for i in ids if hand_of[notes[i][2]] == h]
            if not local:
                continue
            long = [i for i in local if notes[i][1]-t > .003]
            if long:
                close()
                holds.update({notes[i][2]:notes[i][1] for i in long})
                local = [i for i in local if i not in long]
                if not local:
                    continue
            cols = {notes[i][2] for i in local}
            if group and (used & cols or t-notes[group[0]][0] > 2*window
                          or segments[local[0]] != segments[group[-1]] or len(local) > 1):
                close()
            group.extend(local); used.update(cols)
            group_hold = max(group_hold, sum(e > t+.003 for e in holds.values()))
            if len(local) > 1:
                close()
        close()
    return sorted(packets, key=lambda p: (p[0], p[1]))


def _jumptrills(notes, rows, hand_of, window: float, plans=None, *, cancelled=None):
    from skill_calc import _check_cancelled
    gap: cython.double
    pl: cython.double
    ph: cython.double
    ps: cython.double
    lo: cython.double
    hi: cython.double
    speed: cython.double
    margin: cython.double
    strength: cython.double
    quick: cython.double
    n: cython.Py_ssize_t
    r: cython.Py_ssize_t
    j: cython.Py_ssize_t
    i: cython.Py_ssize_t
    n = len(notes)
    result = {k: [0.]*n for k in ("rolled", "jumptrill", "control", "anchor", "jumptrill_easy")}
    packets = _packets(notes, rows, hand_of, window, cancelled=cancelled)
    support = [0.]*len(packets) if plans is not None else None
    positions = [[] for _ in packets] if plans is not None else None
    # A genuine split trill cannot become a jumptrill merely because its two
    # simultaneous hands can be shifted into the Great window.
    cross = [False]*n
    for (_row_number, (t, ids)) in enumerate(rows):
        if cancelled is not None and _row_number%256 == 0:
            _check_cancelled(cancelled)
        if len({hand_of[notes[i][2]] for i in ids}) > 1:
            for i in ids:
                cross[i] = True
    previous = [None, None]
    previous_packets = []
    for p in packets:
        previous_packets.append(previous[p[1]])
        previous[p[1]] = p
    controls, anchors = [None]*len(packets), [0.]*len(packets)

    def control_at(j):
        # Most packets never form a feasible Jumptrill. Work out their control
        # only if they contribute evidence or an execution plan, once each.
        center: cython.double
        span: cython.double
        direction: cython.double
        held: cython.double
        control: cython.double
        anchor: cython.double
        cycle: cython.double
        change: cython.double
        turn: cython.double
        if controls[j] is not None:
            return
        p = packets[j]
        center,h,ids,cols,span,direction,_gap,held = p
        prev = previous_packets[j]
        control = .18*min(1., span/(2*window)) + .25*min(1.,held)
        anchor = 0.
        if prev and 0 < center-prev[0] < .4:
            cycle = center-prev[0]
            change = 1.-len(cols & prev[3])/max(1,len(cols | prev[3]))
            turn = max(0., -direction*prev[5])
            # Reversing a roll is some control; re-hitting its endpoint before
            # the next full hand cycle is a distinct, stronger constraint.
            last = {notes[i][2]: notes[i][0] for i in prev[2]}
            anchor = max((max(0., 1.-(notes[i][0]-last[notes[i][2]])/cycle)
                          for i in ids if notes[i][2] in last), default=0.)
            control += .25*change + .3*turn + .5*anchor
        controls[j] = min(1., control); anchors[j] = anchor

    for r in range(3,len(packets)):
        if cancelled is not None and r%256 == 0:
            _check_cancelled(cancelled)
        segment = packets[r-3:r+1]
        if any(len(p[3]) < 2 for p in segment):
            continue
        if any(a[1] == b[1] for a,b in zip(segment,segment[1:])):
            continue
        # Estimate the pulse from full same-hand cycles. Averaging three
        # adjacent center gaps misreads interleaved 1 4 2 3 (25/75/25 ms) as
        # 41.7 ms, although its underlying alternating pulse is 50 ms.
        gap = ((segment[2][0]-segment[0][0])+(segment[3][0]-segment[1][0]))/4
        if not 0 < gap < .2 or any(b[0]-a[0] > .25 for a,b in zip(segment,segment[1:])):
            continue
        ids = [i for p in segment for i in p[2]]
        if sum(cross[i] for i in ids) > .25*len(ids):
            continue
        rolled = any(p[4] > .003 for p in segment)
        if plans is not None and rolled:
            # Numerical execution keeps a sweep's internal finger offsets. A
            # simultaneous-chord feasibility test alone is too strict for an
            # interleaved 1-3-2-4 roll, whose hands still have a regular cycle.
            pl = max(p[0]-window-j*gap for j,p in enumerate(segment))
            ph = min(p[0]+window-j*gap for j,p in enumerate(segment))
            ps = min(1.-smooth(p[6],.025,.045) for p in segment if p[4]>.003)
            ps *= smooth((ph-pl)/(2*window),0.,.35)
            if ph>pl and ps>0:
                for j,p in enumerate(segment,r-3):
                    if p[4]>.003:
                        support[j]=max(support[j],ps)
                        positions[j].append(((pl+ph)*.5+(j-(r-3))*gap,ps))
        if rolled:
            # Explicit feasible alternating schedule, not "high NPS => JT".
            # A pulse origin must lie in the intersection of all note windows.
            lo = max(notes[i][0]-window-j*gap for j,p in enumerate(segment) for i in p[2])
            hi = min(notes[i][0]+window-j*gap for j,p in enumerate(segment) for i in p[2])
            if lo >= hi:
                continue
            speed = min(1.-smooth(p[6], .025, .045) for p in segment if p[4] > .003)
            margin = (hi-lo)/(2*window)
            strength = speed*smooth(margin, 0., .35)
        else:
            strength = 1.
        quick = smooth(1./(2*gap), 6., 11.)
        for j,p in enumerate(segment, r-3):
            control_at(j)
            for i in p[2]:
                result["jumptrill"][i] = max(result["jumptrill"][i], strength)
                if p[4] > .003:
                    result["rolled"][i] = max(result["rolled"][i], strength)
                result["control"][i] = max(result["control"][i], strength*controls[j])
                result["anchor"][i] = max(result["anchor"][i], strength*anchors[j])
                result["jumptrill_easy"][i] = max(result["jumptrill_easy"][i],
                                                  strength*quick*(1.-controls[j]))
    if plans is not None:
        row_sizes = [0]*n
        for _,ids in rows:
            for i in ids: row_sizes[i] = len(ids)
        for j,p in enumerate(packets):
            if not support[j] or p[4] <= .003 or p[7] or any(row_sizes[i] != 1 for i in p[2]):
                continue
            # Every selected packet is a set of singleton tap heads, not a
            # split trill, repeated column, chord row or independently held key.
            # Averaging feasible pulse shifts preserves each head's window when
            # its recorded internal sweep offset is restored.
            target = sum(t*w for t,w in positions[j])/sum(w for _,w in positions[j])
            control_at(j)
            plans.append({'ids':p[2], 'time':target, 'strength':support[j],
                          'control':controls[j], 'anchor':anchors[j], 'span':p[4],
                          'offsets':tuple(notes[i][0]-p[0] for i in p[2])})
    return result


def _repetition_lock(rows, masks, clean):
    """Stable repetition support beyond the short jack detector's horizon.

    A few identical chords inside changing CJ are not an extended fixed-column
    oscillation. Decaying support from both sides recognises the latter without
    a hard run-length/rate cliff. Real rests decay support even with equal masks.
    This measures chart predictability, not a physiological motor threshold.
    """
    left, right = [0.]*len(rows), [0.]*len(rows)
    for r in range(1,len(rows)):
        if clean[r] and clean[r-1] and masks[r]==masks[r-1]:
            left[r]=math.exp(-(rows[r][0]-rows[r-1][0])/.5)*(1.+left[r-1])
    for r in range(len(rows)-2,-1,-1):
        if clean[r] and clean[r+1] and masks[r]==masks[r+1]:
            right[r]=math.exp(-(rows[r+1][0]-rows[r][0])/.5)*(1.+right[r+1])
    return [smooth(1.+a+b,6.,12.) for a,b in zip(left,right)]


def _vibro(notes, rows):
    """Stable and changing repeating chords share evidence, not equal ease."""
    n = len(rows)
    cols = [{notes[i][2] for i in ids} for _,ids in rows]
    masks = [sum(1 << c for c in cs) for cs in cols]
    clean = [len(cs)==len(ids) and all(notes[i][1]-notes[i][0] <= .003 for i in ids)
             for cs,(_,ids) in zip(cols,rows)]
    speed, similarity = [0.]*n, [0.]*n
    connected = [False]*n
    for r in range(1,n):
        gap = rows[r][0]-rows[r-1][0]
        if gap <= 0 or not (clean[r] and clean[r-1]):
            continue
        shared = (masks[r] & masks[r-1]).bit_count()
        if not shared:
            continue
        connected[r] = True
        speed[r] = smooth(1./gap, 9., 13.)
        similarity[r] = shared/(masks[r] | masks[r-1]).bit_count()
    result = {k: [0.]*len(notes) for k in ("vibro", "vibro_easy", "vibro_control", "vibro_sustain", "vibro_locked")}
    if not any(speed):
        return result
    locked = _repetition_lock(rows, masks, clean)
    pressure = [0.]*n
    start = 1
    while start < n:
        # Geometry defines connectivity; a speed threshold must not suddenly
        # join two runs and increase the support of every neighbouring note.
        if not connected[start]:
            start += 1; continue
        end = start+1
        while end < n and connected[end]:
            end += 1
        for r in range(start,end):
            if not speed[r]:
                continue
            mass = sum(math.exp(-abs(rows[j][0]-rows[r][0])/.3)
                       for j in range(max(start,r-4),min(end,r+5)))
            support = smooth(mass, 1.6, 4.)
            local = similarity[max(start,r-3):min(end,r+4)]
            stable = sum(x**4 for x in local)/len(local)
            gaps = [rows[j][0]-rows[j-1][0] for j in range(max(start,r-3),min(end,r+4))]
            pulse = sorted(gaps)[len(gaps)//2]
            active_gaps = [g for g in gaps if g <= 2*pulse]
            rhythm = min(active_gaps)/max(active_gaps)
            easy = stable*rhythm
            v = speed[r]*support
            pressure[r] = v
            # A changing chord remains potentially vibroable. Its public label
            # should not swallow Chordjack just because one column repeats fast.
            vals = (v*(.25+.75*easy), v*easy, v*(1.-easy))
            for key,value in zip(("vibro","vibro_easy","vibro_control"),vals):
                for i in rows[r][1]:
                    result[key][i] = value
            for i in rows[r][1]:
                result['vibro_locked'][i] = v*easy*locked[r]
        start = end
    # A short slower interval or changing chord does not reset accumulated
    # effort. Real rests have zero pressure and recover in actual played time.
    state = 0.
    for r in range(1,n):
        v = pressure[r]
        decay = math.exp(-(rows[r][0]-rows[r-1][0])/(20. if v > state else 4.))
        state = v+(state-v)*decay
        for i in rows[r][1]:
            result["vibro_sustain"][i] = v*state
    return result


def analyse(chart, rate: float=1.0, bin_seconds: float=0.5, *, with_plans=False, cancelled=None):
    """Return per-bin execution membership and standalone calibration features.

    Odd keymodes average both fixed center-key assignments, as the physical
    calculator does. Scores/replays cannot identify arm versus wrist technique.
    """
    from skill_calc import _check_cancelled
    origin: cython.double
    first: cython.double
    state: cython.double
    decay: cython.double
    denominator: cython.double
    t: cython.double
    e: cython.double
    value: cython.double
    v: cython.double
    w: cython.double
    left: cython.double
    right: cython.double
    bins: cython.Py_ssize_t
    half: cython.Py_ssize_t
    i: cython.Py_ssize_t
    b: cython.Py_ssize_t
    if not math.isfinite(rate) or rate <= 0 or not math.isfinite(bin_seconds) or bin_seconds <= 0:
        raise ValueError("rate and bin_seconds must be positive and finite")
    origin = chart.notes[0][0] if chart.notes else 0.
    notes = [((t-origin)*.001/rate,(e-origin)*.001/rate,c) for t,e,c in chart.notes]
    if not notes:
        return {"series": {k: [] for k in SERIES}, "features": dict.fromkeys(FEATURES,0.)}
    first = notes[0][0]
    bins = int((max(e for _,e,_ in notes)-first)/bin_seconds+1e-9)+1
    rows, ids, at = [], [], None
    for i,(t,e,c) in enumerate(notes):
        if cancelled is not None and i%256 == 0:
            _check_cancelled(cancelled)
        if ids and t-at > .003:
            rows.append((at,ids)); ids=[]
        if not ids:
            at=t
        ids.append(i)
    rows.append((at,ids))
    half = chart.keys//2
    assignments = [False, True] if chart.keys % 2 else [False]
    js = []
    plans = {}
    for center_left in assignments:
        hand_of = [0 if c < half or (center_left and c == half) else 1 for c in range(chart.keys)]
        selected = [] if with_plans else None
        js.append(_jumptrills(notes, rows, hand_of, great_window(chart.od), selected, cancelled=cancelled))
        if with_plans: plans[center_left] = selected
    point = ({k: [(left+right)/2 for left,right in zip(js[0][k], js[1][k])] for k in js[0]}
             if len(js) == 2 else js[0])
    point.update(_vibro(notes,rows))
    point["roll_control"] = point["control"]
    point["roll_anchor"] = point["anchor"]
    point["jumptrill_sustain"] = [0.]*len(notes)
    series = {k: [0.]*bins for k in SERIES}
    counts = [0]*bins
    note_bins = [min(bins-1,int((t-first)/bin_seconds+1e-9)) for t,e,c in notes]
    for b in note_bins:
        counts[b] += 1
    for k, values in series.items():
        if cancelled is not None:
            _check_cancelled(cancelled)
        for b, value in zip(note_bins, point[k]):
            values[b] += value
        series[k] = [v/max(1,c) for v,c in zip(values,counts)]
    # A reversal/anchor can interrupt grouping without instantly restoring
    # endurance. Quiet chart sections recover it; pauses are not chart notes.
    state = 0.
    for b,value in enumerate(series["jumptrill"]):
        decay = math.exp(-bin_seconds/(15. if value > state else 4.))
        state = value+(state-value)*decay
        series["jumptrill_sustain"][b] = value*state
    # Note-pressure weighting makes a high-NPS climax matter without labelling
    # its easy introduction. Rest contributes neither demand nor free stamina.
    weights = [c*c for c in counts]
    denominator = max(1,sum(weights))
    features = {k: sum(v*w for v,w in zip(series[k],weights))/denominator for k in FEATURES}
    result = {"series": series, "features": features}
    if with_plans: result['plans'] = plans
    return result
