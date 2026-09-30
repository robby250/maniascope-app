"""Apply local retarget/rearticulation work before time-horizon aggregation.

The hand-control detector supplies per-action work in the same speed units as
the base hand model. This module changes hand demand, not a map-name lookup or
an additive correction to a whole-map skill label. A zero gain is exactly neutral.
"""
import math
import numpy as np
import hand_control as H


def prepare(chart, rate, runs):
    import skill_calc as S
    assignments = (True,False) if chart.keys%2 else (True,)
    work=[]
    first=chart.notes[0][0]*.001/rate
    for assignment,run in zip(assignments,runs):
        extra=np.zeros((2,len(run[2][0]),len(H.NAMES)),dtype=float)
        for t,values in H.events(chart,rate,assignment):
            i=min(extra.shape[1]-1,max(0,int((t-first)*S.PHASES/S.BIN)))
            extra[:,i,:]+=values
        work.append(extra)
    return work


def aggregate(runs, extra, gains):
    """Return raw horizon levels, rating and fine-grained demand timeline.

    Hands combine at every actual time bin before windows are taken. A control
    spike in an easy section cannot be pasted onto the hardest unrelated phrase.
    Every bin-grid phase used by the base calculator is preserved.
    """
    import skill_calc as S
    gains=np.asarray(gains,dtype=float)
    if gains.shape!=(len(H.NAMES),) or not np.all(np.isfinite(gains)) or np.any(gains<0):
        raise ValueError('Control gains must be finite nonnegative work weights')
    levels=np.zeros(len(S.HORIZONS))
    main=None
    for phase in range(S.PHASES):
        tracks=[]
        for run,work in zip(runs,extra):
            fine=np.asarray(run[2],dtype=float).copy()
            fine[:2]+=np.einsum('htj,j->ht',work,gains)
            indices=(np.arange(fine.shape[1])+phase)//S.PHASES
            bins=np.stack([np.bincount(indices,weights=v) for v in fine])
            total=(np.hypot(bins[0],bins[1])+bins[2])/S.BIN
            tracks.append(total)
        timeline=np.mean(tracks,axis=0)
        if phase==0:main=timeline
        sums=np.r_[0.,np.cumsum(timeline)]
        ends=np.arange(1,len(timeline)+1)
        for h,(seconds,tolerance) in enumerate(S.HORIZONS):
            width=min(len(timeline),max(1,int(seconds/S.BIN)))
            maximum=np.max((sums[ends]-sums[np.maximum(0,ends-width)])/width)
            value=S._star(S.SCALE*math.sqrt(max(0.,maximum))/tolerance)
            levels[h]=max(levels[h],value)
    return levels.tolist(),main.tolist()


def factors(chart, rate, runs, settings):
    """Evaluate several predeclared hypotheses against one physical extraction."""
    import skill_calc as S
    work=prepare(chart,rate,runs)
    zero,original=aggregate(runs,work,[0.]*len(H.NAMES))
    baseline=S._combine(zero,len(chart.notes))
    out=[]
    for gains in settings:
        levels,timeline=aggregate(runs,work,gains)
        ratio=S._combine(levels,len(chart.notes))/baseline if baseline>0 else 1.
        out.append({'factor':ratio,'levels':levels,'timeline':timeline})
    return out
