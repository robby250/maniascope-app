"""Robust shared and skill-local session condition from frozen forecasts.

One unfamiliar chart can be a model error; several independently underperformed
charts can establish poor form. No other keymode, quit or replay supplies an
accuracy observation. All historical condition variances come from the fitter.
"""
import collections
import math
from functools import lru_cache
import numpy as np
import warmup


def loading(skill=None, features=None):
    if features is not None:
        profile = warmup.family_weights(features)
    else:
        profile = ((warmup.family(skill), 1.),)
    return _loading(profile)


@lru_cache(maxsize=32768)
def _loading(profile):
    profile = dict(profile)
    values = [profile.get(k, 0.) for k in warmup.FAMILIES]
    total=sum(values)
    if total<=0:values=[1./len(values)]*len(values)
    else:values=[v/total for v in values]
    result = np.array([1.,*values])
    result.flags.writeable = False
    return result


def posterior(attempts, now, keys, prior):
    dimension=len(warmup.FAMILIES)+1
    local=prior.get('by_key',{}).get(keys,{})
    variances=np.array([local.get('global_variance',prior['global_variance'])]+
                       local.get('skill_variances',[prior['skill_variance']]*(dimension-1)))
    precision=np.diag(1./np.maximum(1e-6,variances))
    blocks=collections.defaultdict(list)
    for a in attempts:
        if a.get('keys')!=keys or not a.get('meaningful') or not a.get('ability_evidence',True) \
                or a.get('mu') is None or a.get('y') is None:
            continue
        error=a['y']-a.get('base_mu',a['mu'])-(a.get('cold_penalty') or 0.)
        variance=max(.0001,float(a.get('sd') or math.sqrt(prior['noise_variance']))**2)
        age=math.exp(-max(0.,now-a['end'])/1800.)
        blocks[a.get('beatmap') or a.get('key')].append((loading(a.get('skill'),a.get('features')),error,variance,age))
    xs,ys,vs,ages=[],[],[],[]
    for observations in blocks.values():
        mass=sum(a for x,y,v,a in observations)
        if mass<=0:continue
        xs.append(sum(x*a for x,y,v,a in observations)/mass)
        ys.append(sum(y*a for x,y,v,a in observations)/mass)
        # Retries reduce timing noise, not uncertainty about that chart itself.
        vs.append(sum(v*a for x,y,v,a in observations)/mass)
        ages.append(max(a for x,y,v,a in observations))
    if not xs:return np.zeros(dimension),variances
    X=np.array(xs);y=np.array(ys);variance=np.array(vs)
    base=np.array(ages)/variance;robust=np.ones(len(y));mean=np.zeros(dimension)
    # Unknown session/skill state can legitimately explain a large departure.
    # Judging outliers against timing noise alone would reject all the evidence
    # of an initially unmeasured weak skill (e.g. SV) as "unlikely" mistakes.
    innovation_variance=variance+np.sum(X*X*variances,axis=1)
    # Start from pooled evidence, then a Student-t likelihood can identify a
    # single map outlier without rejecting a coherent multi-map condition shift.
    for _ in range(5):
        weight=base*robust
        gram=precision+X.T@(weight[:,None]*X)
        mean=np.linalg.solve(gram,X.T@(weight*y))
        standardized=(y-X@mean)**2/innovation_variance
        robust=np.minimum(1.,5./(4.+standardized))
    return mean,np.diag(np.linalg.inv(gram))
