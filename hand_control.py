"""Local hand-control measurements for blind-spot investigation.

These describe actual note transitions, not skill labels or map identity. Pure
repeated chords, fixed alternating groups, and empty space are controls. Costs
are hypotheses until fitted on TRAIN and accepted against held-out songs.
"""
import math

VERSION = 1
NAMES = ('selective_repeats', 'bracket_retargets', 'release_opposition')


def smooth(x, low, high):
    x = min(1., max(0., (x-low)/(high-low)))
    return x*x*(3-2*x)


def events(chart, rate=1., centre_left=True):
    """Per-row (played seconds, left/right three-component control work).

    Selective repeats: a finger repeats while the same hand changes which other
    fingers participate. It cannot use the fully shared, fixed-chord bounce.
    Bracket retargets: changing the two disjoint groups while alternating them;
    a fixed ABAB bracket has zero retarget work after entering it.
    Release opposition: a hold ending close to a different finger's press,
    rather than co-releasing as a group. Timed in real seconds at the given rate.
    """
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError('Rate must be positive and finite')
    half = chart.keys//2
    hand = [int(c >= half and not (chart.keys%2 and c == half and centre_left)) for c in range(chart.keys)]
    groups, current = [], None
    for t,e,c in chart.notes:
        t,e = t*.001/rate, e*.001/rate
        if current is None or t-current[0] > .003:
            current = (t, [])
            groups.append(current)
        current[1].append((e,c))
    previous, before = [None,None], [None,None]
    holds = {}
    output = []
    for t, row in groups:
        masks = [0,0]
        for e,c in row:
            masks[hand[c]] |= 1 << c
        work = [[0.,0.,0.],[0.,0.,0.]]
        for h, mask in enumerate(masks):
            if not mask:
                continue
            last = previous[h]
            if last is not None:
                dt = t-last[0]
                if .006 < dt < .25:
                    old = last[1]
                    repeated = (mask & old).bit_count()
                    changed = (mask ^ old).bit_count()
                    union = (mask | old).bit_count()
                    # Tap-rate pressure fades at slow transitions and saturates
                    # below the meaningful independent-bounce timing scale.
                    speed = 1./max(.045, dt)
                    speed_weight = speed*math.sqrt(speed/8)*smooth(speed,4.,9.)
                    work[h][0] = speed_weight*repeated*changed/max(1, union)
                    prevprev = before[h]
                    if not repeated and max(mask.bit_count(), old.bit_count()) >= 2 and prevprev:
                        retarget = (mask ^ prevprev[1]).bit_count()/max(1,(mask | prevprev[1]).bit_count())
                        work[h][1] = speed_weight*retarget
            # Do not turn a sustained held finger into a new repeat just because
            # another finger is tapping next to it; only actual heads form rows.
            for col, (head,end) in holds.items():
                d = abs(end-t)
                if hand[col] == h and not (mask & (1<<col)) and head < t-.003 and d < .10:
                    duration = end-head
                    work[h][2] += 8.*smooth(duration,.06,.25)*(1.-smooth(d,.02,.10))
            before[h], previous[h] = previous[h], (t,mask)
        holds = {c:v for c,v in holds.items() if v[1] >= t-.10}
        for end,c in row:
            if end-t > .003:
                holds[c] = (t,end)
        output.append((t,work))
    return output


def analyse(chart, rate=1., bin_seconds=.5):
    """Aggregate local work plus representative sections; independent of scores."""
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError('Rate must be positive and finite')
    if not math.isfinite(bin_seconds) or bin_seconds <= 0:
        raise ValueError('Bin duration must be positive and finite')
    if not chart.notes:
        return {'version':VERSION, 'summary':dict.fromkeys(NAMES,0.), 'sections':[], 'series':[]}
    first = chart.notes[0][0]*.001/rate
    count = max(1, int((max(e for t,e,c in chart.notes)*.001/rate-first)/bin_seconds)+1)
    variants = [events(chart,rate,c) for c in ((True,False) if chart.keys%2 else (True,))]
    series = [[0.]*len(NAMES) for _ in range(count)]
    for event_rows in variants:
        for t,work in event_rows:
            b = min(count-1,max(0,int((t-first)/bin_seconds)))
            for j in range(len(NAMES)):
                series[b][j] += math.hypot(work[0][j],work[1][j])/len(variants)/bin_seconds
    totals = [sum(v[j] for v in series)*bin_seconds/max(1,len(chart.notes)) for j in range(len(NAMES))]
    sections = []
    for j,name in enumerate(NAMES):
        width = max(1,round(4./bin_seconds))
        sums = [sum(v[j] for v in series[max(0,i-width+1):i+1]) for i in range(len(series))]
        end = max(range(len(sums)),key=sums.__getitem__)
        if sums[end]:
            sections.append({'mechanic':name,'start':(first+max(0,end-width+1)*bin_seconds)*rate,
                             'end':(first+(end+1)*bin_seconds)*rate,
                             'work':sums[end]*bin_seconds})
    return {'version':VERSION, 'summary':dict(zip(NAMES,totals)), 'sections':sections, 'series':series}
