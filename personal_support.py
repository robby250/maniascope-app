"""Separate repeatable chart affinity from a single session's condition.

All inputs are the caller's training history. Same-session retries are one block
of evidence, and a chart cannot explain away its own errors as session form.
"""
import collections
import hashlib
import math
import numpy as np


def fit_shape(z, target, weights, groups):
    """Chart-family-blocked regularisation, with a conservative one-SE choice."""
    n,p=z.shape
    grid=(.02,.05,.2,.8)
    fold=np.array([int(hashlib.sha256(str(g).encode()).hexdigest()[:8],16)%4 for g in groups])
    all_a=z.T@(weights[:,None]*z)
    all_b=z.T@(weights*target)
    def solve(a,b,penalty):
        matrix=a+penalty*np.eye(p)
        beta=np.linalg.solve(matrix,b)
        if beta[-1]<0:
            beta[-1]=0.
            beta[:-1]=np.linalg.solve(matrix[:-1,:-1],b[:-1])
        return beta
    if len(set(groups)) < 8 or len(set(fold)) < 2:
        # One heavily retried chart cannot validate a structural specialization.
        return np.zeros(p), {'ridge':None,'chart_families':len(set(groups)),
                             'cv_mse':{},'reason':'insufficient independent chart families'}
    errors=np.empty((n,len(grid)))
    for f in range(4):
        mask=fold==f;train=~mask
        if not mask.any() or not train.any():
            continue
        a=all_a-z[mask].T@(weights[mask,None]*z[mask])
        b=all_b-z[mask].T@(weights[mask]*target[mask])
        mass=weights[train].sum()
        for j,lam in enumerate(grid):
            errors[mask,j]=(target[mask]-z[mask]@solve(a,b,lam*mass))**2
    by=collections.defaultdict(list)
    for i,g in enumerate(groups):by[g].append(i)
    grouped=np.array([np.average(errors[ix],axis=0,weights=weights[ix]) for ix in by.values()])
    risk=grouped.mean(axis=0);best=int(np.argmin(risk))
    eligible=[]
    for j in range(len(grid)):
        difference=grouped[:,j]-grouped[:,best]
        se=float(difference.std()/math.sqrt(max(1,len(grouped))))
        if risk[j]-risk[best] <= se+1e-12:eligible.append(j)
    chosen=max(eligible,key=lambda j:grid[j])
    beta=solve(all_a,all_b,grid[chosen]*weights.sum())
    return beta,{'ridge':grid[chosen],'chart_families':len(grouped),
                 'cv_mse':dict(zip(map(str,grid),map(float,risk)))}


def session_context(rows, residual):
    """Leave-one-chart-out condition from OTHER charts in that historical session."""
    order=sorted((i for i,r in enumerate(rows) if r.get('src') in ('realm','live')
                  and r.get('ts') and r.get('completed',True) and not r.get('rd')),
                 key=lambda i:rows[i]['ts'])
    ids=[None]*len(rows);last=None;session=-1
    for i in order:
        r=rows[i]
        seconds=max(0.,min(1800.,r['f'].get('length',0.)/1000/max(.01,r['rate'])))
        if last is None or r['ts']-seconds-last>3000:session+=1
        ids[i]=(session,r['keys'])
        last=r['ts']
    groups=collections.defaultdict(lambda:collections.defaultdict(list))
    for i,g in enumerate(ids):
        if g is not None:groups[g][rows[i]['chart']].append(i)
    block={g:[(chart,ix,float(np.mean(residual[ix])),float(np.mean([rows[i]['ts'] for i in ix])))
              for chart,ix in charts.items()] for g,charts in groups.items()}
    multiple=[v for v in block.values() if len(v)>=3]
    if len(multiple)>=10:
        means=np.array([np.mean([r[2] for r in v]) for v in multiple])
        within=float(np.mean([np.var([r[2] for r in v],ddof=1) for v in multiple]))
        between=max(0.,float(np.var(means)-np.mean([within/len(v) for v in multiple])))
        shrink=within/max(1e-6,between)
    else:
        shrink=float('inf');within=between=0.
    adjustment=np.zeros(len(rows))
    for group,values in block.items():
        for chart,ix,_value,t in values:
            other=[(e,math.exp(-abs(t-t2)/1800.)) for c,_ii,e,t2 in values if c!=chart]
            if other:
                offset=sum(e*w for e,w in other)/(shrink+sum(w for e,w in other))
                adjustment[ix]=offset
    local_variance=[]
    import warmup
    for values in multiple:
        pooled=float(np.mean([v[2] for v in values]));families=collections.defaultdict(list)
        for _chart,ix,value,_t in values:
            families[warmup.dominant(rows[ix[0]]['f'])].append(value)
        for observations in families.values():
            if len(observations)<2:continue
            noise=within*max(0.,1./len(observations)-1./len(values))
            local_variance.append((float(np.mean(observations))-pooled)**2-noise)
    skill_variance=max(0.,float(np.mean(local_variance))) if local_variance else 0.
    by_key={}
    for keys in sorted({r['keys'] for r in rows}):
        sessions=[v for g,v in block.items() if g[1]==keys and len(v)>=3]
        family_samples=collections.defaultdict(list)
        for values in sessions:
            all_values=[v[2] for v in values]
            pooled=float(np.mean(all_values));local_noise=float(np.var(all_values,ddof=1))
            families=collections.defaultdict(list)
            for _chart,ix,value,_t in values:
                families[warmup.dominant(rows[ix[0]]['f'])].append(value)
            for family,observations in families.items():
                if len(observations)<2:continue
                noise=local_noise*max(0.,1./len(observations)-1./len(values))
                family_samples[family].append((float(np.mean(observations))-pooled)**2-noise)
        if sessions:
            means=np.array([np.mean([r[2] for r in v]) for v in sessions])
            error=np.mean([np.var([r[2] for r in v],ddof=1)/len(v) for v in sessions])
            own=max(0.,float(np.var(means)-error))
            global_variance=(len(sessions)*own+10*between)/(len(sessions)+10)
        else:global_variance=between
        # Unobserved families borrow map uncertainty, not near-zero variance
        # from well-modelled rice. Evidence gradually replaces that weak prior.
        prior_family=max(skill_variance,within)
        variances=[];support=[]
        for family in warmup.FAMILIES:
            samples=family_samples[family];n=len(samples)
            estimate=max(0.,float(np.mean(samples))) if n else 0.
            variances.append((n*estimate+5*prior_family)/(n+5))
            support.append(n)
        by_key[keys]={'global_variance':global_variance,'skill_variances':variances,
                      'family_sessions':support,'sessions':len(sessions)}
    return adjustment,ids,{'sessions':len(groups),'supported_sessions':len(multiple),
                           'condition_variance':between,'within_variance':within,
                           'skill_condition_variance':skill_variance,
                           'by_key':by_key,
                           'prior_observations':shrink if math.isfinite(shrink) else None}


def chart_blocks(rows, residual, session_ids, now):
    """One weighted observation per chart/rate/session, not per retry."""
    by=collections.defaultdict(list)
    for i,r in enumerate(rows):
        chart=r['chart']+('|RD' if r.get('rd') else '')
        session=session_ids[i]
        # Date-only bests do not establish warmup/session order. Their score
        # identity remains useful, but is weaker evidence of typical performance.
        block=session if session is not None else ('best',r.get('key',i))
        by[chart,round(r['rate'],3),block].append(i)
    output=collections.defaultdict(list)
    for (chart,rate,_block),ix in by.items():
        recent=max(rows[i]['t'] for i in ix)
        age=.5**(max(0.,now-recent)/12.)
        weight=age*(.5 if all(rows[i]['src']=='public' for i in ix) else 1.)
        if any(rows[i].get('rd') for i in ix):weight*=.35
        output[chart].append((rate,float(np.mean(residual[ix])),weight,len(ix)))
    return output


def effective_count(values):
    """Independent blocks for variance estimation, distinct from recency precision.

    Three old sessions still estimate a between-chart variance from three
    observations. Their age weakens the final prediction; it must not also inflate
    the variance subtraction and erase the entire population's chart effects.
    """
    weights=[w for _r,_v,w,_n in values]
    return sum(weights)**2/max(1e-12,sum(w*w for w in weights))
