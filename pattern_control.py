"""Local execution patterns used for recognition, never map-name overrides."""
import math

TECH_PROFILE_VERSION = 1


def smooth(x, lo, hi):
    x = min(1., max(0., (x-lo)/(hi-lo)))
    return x*x*(3.-2.*x)


def quadstream(chart, rate, bin_seconds=.5):
    """Full-column chords between single notes; isolated hits keep low coverage.

    The wider-key equivalent is measured as evidence but has no invented public
    name. Doubles around quads remain chordjack/minijack, as do repeated quads.
    Existing physical demand already pays for the repeated fingers once.
    """
    if not chart.notes:
        return [], 0.
    first = chart.notes[0][0]
    rows = []
    for t,e,c in chart.notes:
        if rows and t-rows[-1][0] <= 3:
            rows[-1][1].add(c); rows[-1][2] |= e-t > 3
        else:
            rows.append([t, {c}, e-t > 3])
    bins = int((max(e for _,e,_ in chart.notes)-first)*.001/rate/bin_seconds)+1
    hits, counts, episodes = [0.]*bins, [0.]*bins, [0.]*bins
    marked = [0.]*len(rows)
    for i in range(1,len(rows)-1):
        a,b,c = rows[i-1],rows[i],rows[i+1]
        if b[2] or a[2] or c[2] or len(b[1]) != chart.keys or len(a[1]) != 1 or len(c[1]) != 1:
            continue
        ga,gb = (b[0]-a[0])*.001/rate, (c[0]-b[0])*.001/rate
        if min(ga,gb) <= .006 or max(ga,gb) > .23:
            continue
        strength = (1.-smooth(max(ga,gb),.16,.23))*(1.-smooth(abs(math.log(ga/gb)),.15,.6))
        for j in (i-1,i,i+1):
            marked[j] = max(marked[j],strength)
        episodes[min(bins-1,int((b[0]-first)*.001/rate/bin_seconds))] += strength
    for (t,cs,_),v in zip(rows,marked):
        b = min(bins-1,int((t-first)*.001/rate/bin_seconds))
        counts[b] += len(cs); hits[b] += len(cs)*v
    out = []
    for b in range(bins):
        lo,hi = max(0,b-2),min(bins,b+3)
        coverage = sum(hits[lo:hi])/max(1.,sum(counts[lo:hi]))
        support = min(1.,sum(episodes[lo:hi])/3.)
        out.append(coverage*support)
    return out, sum(hits)/max(1.,sum(counts))


def smooth_rolls(rowsets, keys):
    """Notes inside stable adjacent-finger rolls, separately for each hand.

    A three-finger sweep is a real wide-key execution primitive. It need not
    cross six global rows to stop being labelled a series of timing surprises.
    Unordered rice, inserted jacks and direction reversals keep their evidence.
    """
    assignments = [False, True] if keys%2 else [False]
    output = [[0.]*len(rowsets) for _ in (0,1)]
    for center_left in assignments:
        half = keys//2
        hand = lambda c: 0 if c<half or (center_left and c==half) else 1
        for h in (0,1):
            local = [(i,t,tuple(sorted(c for c in cs if hand(c)==h)))
                     for i,(t,cs) in enumerate(rowsets) if any(hand(c)==h for c in cs)]
            run = []
            step = None
            def close():
                if len(run)>=3:
                    for index,_,_ in run:
                        output[h][index] += 1./len(assignments)
            for row in local:
                i,t,cs = row
                if len(cs)!=1:
                    close(); run=[]; step=None; continue
                delta = cs[0]-run[-1][2][0] if run else 0
                gap = t-run[-1][1] if run else 1.
                prevgap = run[-1][1]-run[-2][1] if len(run)>1 else gap
                if run and abs(delta)==1 and (step is None or delta==step) and .006<gap<.16 \
                        and abs(gap-prevgap)<max(.003,.12*prevgap):
                    run.append(row); step=delta
                else:
                    close()
                    run = [run[-1],row] if run and abs(delta)==1 and .006<gap<.16 else [row]
                    step = delta if len(run)>1 else None
            close()
    return output


def tech_profile(chart, rate=1., info=None):
    """Whole-map technical character, separate from peak difficulty units.

    Count the actual hand-rhythm/control material across the chart, not the
    maximum rating of one technical passage. Reuse the execution pass in live
    analysis; standalone enrichment does not run the full difficulty calculator.
    """
    import skill_calc as S
    if info is None:
        import execution
        info = execution.analyse(chart, rate)
    counts = info.get('count', [])
    empty = {'version': TECH_PROFILE_VERSION, 'technical': 0., 'patterntech': 0., 'rhythmtech': 0.}
    if not counts or not sum(counts):
        return empty
    rows = []
    for t, _e, col in chart.notes:
        t = t*.001/rate
        if rows and t-rows[-1][0] <= .003:
            rows[-1][1].add(col)
        else:
            rows.append((t, {col}))
    trills = [None]*len(rows)
    half = chart.keys//2
    for i in range(3, len(rows)):
        window = rows[i-3:i+1]
        if min(len(cs) for t, cs in window) < 2 or window[-1][0]-window[0][0] >= 3*S.TRILL_GAP:
            continue
        hands = [{int(c >= half) for c in cs} for t, cs in window]
        a,b,c,d = [cs for t,cs in window]
        kind = ('jumptrill' if len(hands[0]) == len(hands[1]) == 1 and hands[0] != hands[1]
                and hands[0] == hands[2] and hands[1] == hands[3] else
                'splittrill' if chart.keys == 4 and a == c and b == d and not a & b else None)
        if kind:
            trills[i-3:i+1] = [kind]*4
    inserted, switched = S._pattern_evidence(rows, trills)
    pattern = [0.]*len(counts)
    first = rows[0][0]
    for (t, _cs), ins, sw in zip(rows, inserted, switched):
        b = min(len(counts)-1, int((t-first)/S.BIN))
        pattern[b] += (ins + sw*info['switch'][b])*(1.-info['mash'][b])
    rhythm = [v*n*(1.-m) for v,n,m in zip(info['regular'], counts, info['mash'])]
    def window(values):
        out=[];total=0.;h=2
        for i in range(len(values)+h):
            if i < len(values): total += values[i]
            if i >= 2*h+1: total -= values[i-2*h-1]
            if i >= h: out.append(max(0.,total))
        return out
    denominator, pr, rr = window(counts), window(pattern), window(rhythm)
    totals = {'technical': 0., 'patterntech': 0., 'rhythmtech': 0.}
    for n, den, p, r in zip(counts, denominator, pr, rr):
        pt = min(1.,p/max(1.,den)/S.PT_FULL)
        rt = min(1.,r/max(1.,den)/S.RT_FULL)
        totals['patterntech'] += n*pt
        totals['rhythmtech'] += n*rt
        totals['technical'] += n*(1.-(1.-pt)*(1.-rt))
    return {'version': TECH_PROFILE_VERSION, **{k:round(v/sum(counts),6) for k,v in totals.items()}}


def technical_title(name, skills, profile):
    """A persistent technical texture refines a pattern, without adding strain.

    The peak-rated skill still supplies the number. Prevalence is required as
    well as material technical demand, so a short awkward passage cannot name
    an otherwise smooth map. Inputs are identical in fresh and cached features.
    """
    if not profile or profile.get('technical',0.) < .5 or skills.get('technical',0.) < .6:
        return name
    if name.startswith('Tech ') or name in ('SV','Fast SV','Slowjam SV','Accel SV','Stutter SV','Brakes','Vibro','Mash','Stamina'):
        return name
    return 'Tech ' + name


def technical_description(description, skills, profile):
    first, sep, rest = description.partition(' / ')
    return technical_title(first, skills, profile) + sep + rest
