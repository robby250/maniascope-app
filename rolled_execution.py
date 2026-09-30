"""A timing-feasible rolled hand pulse shares literal-jumptrill motor work.

Reference notes are private demand inputs, never substituted into a real chart,
score, replay or reading timeline. Only selected singleton taps can share work;
original repeated-finger and LN actions keep their physical cost. The fitted
strength interpolates local action demand, not a map-level percentage discount.
"""
import math

VERSION = 2


def possible(chart,rate):
    """Cheap conservative screen. A rolled packet needs a nonzero sub-45ms gap.

    This only skips impossible charts; it does not decide gesture membership.
    Simultaneous chords are included in the scan but cannot create a gap alone.
    """
    maximum=45.*rate
    return any(0 < b[0]-a[0] < maximum for a,b in zip(chart.notes,chart.notes[1:]))


def reference(chart, rate, plans):
    """Build a one-to-one motor-pulse reference for one hand assignment.

    Feasibility retains the sweep's internal offsets, not simultaneous hits.
    The reference coalesces its motor pulse ONLY to compare calibrated work.
    Overlapping four-packet schedules vote on each pulse's position.
    Reject a packet if that projection could cross/collide with another action
    of one of its own fingers. Unchanged note ordering/hold lengths are retained.
    """
    import skill_calc as S
    import gestures as G
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError('Invalid playback rate')
    if not plans: return None
    origin = chart.notes[0][0]
    times = [t for t,e,c in chart.notes]
    targets = times.copy()
    weights = [0.]*len(times)
    bycol = [[] for _ in range(chart.keys)]
    for i,(t,e,c) in enumerate(chart.notes): bycol[c].append(i)
    neighbours = {}
    for ids in bycol:
        for j,i in enumerate(ids):
            neighbours[i] = (ids[j-1] if j else None, ids[j+1] if j+1<len(ids) else None)
    for p in plans:
        ids = p['ids']; target = origin+p['time']*rate*1000.
        if 0 in ids or len(times)-1 in ids: continue
        # Packet membership changes discretely when a sweep exceeds the timing
        # span. Its numerical credit must fade to zero BEFORE that boundary.
        # Otherwise a 0.01% rate change can suddenly remove an entire star.
        margin=1.-p['span']/(2*G.great_window(chart.od))
        w = p['strength']*(1.-p['control'])**2*G.smooth(margin,0.,.25)
        if w <= 0 or p['anchor'] >= .8: continue
        safe = True
        for i in ids:
            left,right = neighbours[i]
            # A real repeated endpoint cannot disappear into the pulse; retain
            # ordering and a margin on both neighbouring attacks/hold tails.
            if (left is not None and target <= chart.notes[left][1]+3*rate or
                    right is not None and target >= chart.notes[right][0]-3*rate):
                safe = False; break
        if safe:
            for i in ids: targets[i],weights[i] = target,w
    if not any(weights): return None
    # Check final simultaneous projection too (neighbours can also have moved).
    bad=set()
    for ids in bycol:
        for a,b in zip(ids,ids[1:]):
            if targets[b]-targets[a] <= 3*rate and (weights[a] or weights[b]):bad.update((a,b))
    if bad:
        for p in plans:
            if bad.intersection(p['ids']):
                for i in p['ids']:targets[i],weights[i] = times[i],0.
    if not any(weights): return None
    ordered=sorted(range(len(times)),key=lambda i:(targets[i],chart.notes[i][2],i))
    ref=S.Chart()
    for key in ('keys','title','version','artist','creator','od'):setattr(ref,key,getattr(chart,key))
    ref.sv=[]
    ref.notes=[(targets[i], targets[i] if weights[i] else chart.notes[i][1], chart.notes[i][2]) for i in ordered]
    return ref,ordered,weights


def basis(chart,rate,sv,result,shared,cancelled=None):
    """Local removable work relative to a feasible literal-pulse reference.

    Normalize both views into the SAME calibrated physical-demand units before
    comparing them. Otherwise the old stream-versus-chord calibration itself
    restores the excess cost after the hand gestures have already been shared.
    Never use public map IDs, personal scores or exact known-map residuals.
    """
    import skill_calc as S
    physical=shared['trace'].get(False,shared['trace'][bool(sv)])
    original=shared['trace'][bool(sv)]
    reference_phases=[]; confidences={}
    for center,plans in shared['gestures'].get('plans',{}).items():
        S._check_cancelled(cancelled)
        built=reference(chart,rate,plans)
        if built is None:
            reference_phases.append(physical['phases']);confidences[center]=[0.]*len(chart.notes);continue
        ref,order,weights=built
        ref_shared={'trace':{}}
        rr=S._compute(ref,rate,False,ref_shared,cancelled)
        # Only physical calibration is normalized here. The chart-only critic
        # remains the original chart's critic, not a lookup/second correction.
        p=physical['result']
        conversion=((rr['baseline_overall']/rr['raw_overall'])/(p['baseline_overall']/p['raw_overall']))**(2./S.STAR_B)
        reference_phases.append([[v*conversion for v in row] for row in ref_shared['trace'][False]['phases']])
        confidences[center]=weights
    if not confidences or not any(any(c) for c in confidences.values()):return None
    origin=chart.notes[0][0]*.001/rate
    fine=[int((t*.001/rate-origin)*S.PHASES/S.BIN) for t,e,col in chart.notes]
    deltas=[];support_series=None
    for phase,full in enumerate(original['phases']):
        plain=physical['phases'][phase]
        delta=[0.]*len(full)
        support=[0.]*len(full)
        for ref,weights in zip(reference_phases,confidences.values()):
            totals=[0.]*len(full);count=[0]*len(full)
            for i,index in enumerate(fine):
                b=min(len(full)-1,max(0,(index+phase)//S.PHASES))
                totals[b]+=weights[i];count[b]+=1
            for b in range(len(full)):
                # Min keeps the ordinary tapping option; no chart is forced to
                # use a less efficient technique. Reading is never subtracted.
                target=ref[phase][b] if b<len(ref[phase]) else plain[b]
                delta[b]+=max(0.,plain[b]-target)*totals[b]/max(1,count[b])/len(confidences)
                support[b]+=totals[b]/max(1,count[b])/len(confidences)
        deltas.append(delta)
        if phase==0:support_series=support
    # A bin can also contain an unselected jack, hold, release or reading event.
    # Cap its credit by removing only eligible original tap work, before the
    # hand norm and phase pooling. All protected work stays at its actual time.
    floors=[]
    for center,(actions,tracks) in original['action_runs'].items():
        weights=confidences.get(center,next(iter(confidences.values())))
        minimum=[list(row) for row in tracks]
        for w,(hand,work,cross,protected,at) in zip(weights,actions):
            if protected:continue
            minimum[hand][at]=max(0.,minimum[hand][at]-w*work)
            minimum[2][at]=max(0.,minimum[2][at]-w*cross)
        floors.append([S._phase_total(minimum,p) for p in range(S.PHASES)])
    for p,row in enumerate(deltas):
        floor=[sum(v)/len(floors) for v in zip(*(f[p] for f in floors))]
        deltas[p]=[min(d,max(0.,a-b)) for d,a,b in zip(row,original['phases'][p],floor)]
    return {'phases':original['phases'],'removable':deltas,
            'support':support_series,
            'mean_support':sum(map(sum,confidences.values()))/len(confidences)/max(1,len(chart.notes))}


def evaluate(basis,strength,notes):
    import skill_calc as S
    if not math.isfinite(strength) or not 0<=strength<=1:
        raise ValueError('Rolled execution strength must be in [0,1]')
    levels=[0.]*len(S.HORIZONS);series=[]
    for row,credit in zip(basis['phases'],basis['removable']):
        mixed=[max(0.,v-strength*d) for v,d in zip(row,credit)]
        series.append(mixed)
        levels=[max(a,b) for a,b in zip(levels,S._horizon_levels(mixed))]
    old=[0.]*len(S.HORIZONS)
    for row in basis['phases']:old=[max(a,b) for a,b in zip(old,S._horizon_levels(row))]
    denominator=S._combine(old,notes)
    return S._combine(levels,notes)/denominator if denominator else 1.,levels,series[0]


def apply(chart,rate,sv,result,shared,strength,cancelled=None,*,candidate=None):
    import skill_calc as S
    candidate=basis(chart,rate,sv,result,shared,cancelled) if candidate is None else candidate
    if candidate is None:return result
    factor,levels,total=evaluate(candidate,strength,len(chart.notes))
    if math.isclose(factor,1.,rel_tol=0.,abs_tol=1e-12):return result
    source=shared['trace'][bool(sv)]
    ratio=[v/old if old else 1. for v,old in zip(total,source['total'])]
    series={k:[v*r for v,r in zip(row,ratio)] for k,row in source['series'].items()}
    rice=[k for k in ('stream','delay','dump','jumpstream','handstream','chordstream') if k in series]
    for b,support in enumerate(candidate['support']):
        weight=strength*support
        moved=max((series[k][b] for k in rice),default=0.)*weight
        series['jumptrill'][b]=max(series['jumptrill'][b],moved)
        for k in rice:series[k][b]*=1.-weight
    switch=[v*r for v,r in zip(source['switch'],ratio)]
    # Re-pool locally modified attributed work, not a uniform all-skills nerf.
    calibration=result['scores']['overall']/result['raw_overall']
    scale=calibration*rate**S.RATE_G
    raw=S._combine(levels,len(chart.notes))
    scores={k:S._combine(S._horizon_levels(v),len(chart.notes)) for k,v in series.items()}
    scores['overall']=raw
    scores['stamina']=S._stamina(total)*raw/max(levels) if max(levels)>0 else 0.
    result['archetypes']=S._archetypes(series,scores,len(chart.notes),switch)
    result['tech_where']=S._tech_where(series,source['parts'],chart.notes[0][0]/1000.,rate)
    for a in result['archetypes']:a['rating']*=scale
    result['scores']={k:v*scale for k,v in scores.items()}
    result['levels']={h[0]:v*scale for h,v in zip(S.HORIZONS,levels)}
    result['horizon']=S.HORIZONS[levels.index(max(levels))][0]
    width=max(1,int(S.HORIZONS[0][0]/S.BIN));acc=0.;timeline=[]
    for i,d in enumerate(total):
        acc=max(0.,acc+d-(total[i-width] if i>=width else 0.))
        timeline.append(S._star(S.SCALE*math.sqrt(acc/min(width,i+1))/S.HORIZONS[0][1])*scale)
    result['timeline']=timeline
    result['execution']['rolled_factor']=factor
    result['execution']['rolled_support']=candidate['mean_support']
    if sv and False in shared['trace']:
        physical=shared['trace'][False]
        pf,_,_=evaluate(dict(candidate,phases=physical['phases']),strength,len(chart.notes))
        result['no_sv_overall']=physical['result']['scores']['overall']*pf
    result['rolled_correction']=result['scores']['overall']*(1.-1./factor)
    result['baseline_overall']*=factor
    result['structural_correction']=result['scores']['overall']-result['baseline_overall']
    return result
