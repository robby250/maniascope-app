"""Completed-score cold-to-warm learning, independent of quits and targets.

Session starts are inferred from timestamped local completions, never dated
public bests. Chart/rate, month, pattern mix and session intercepts absorb
familiarity, skill drift and good/bad days. Only earlier sessions select the
decay curve; a late-session holdout is diagnostic, not a fitting target.
"""
import collections
import functools
import math
import numpy as np

VERSION = 2
CAPACITY_PRIOR = -math.log(.90)
GAP = 50*60
COOLING = 40*60
TAUS = (.5, 1., 2., 4., 8., 16.)
FAMILIES = ('rice','chords','chordjack','jack','ln','trill','technical','sv')
MEMBERS = {
    'rice':('stream','delay','dump','jumpstream','handstream'),
    'chords':('chordstream','bracket'), 'chordjack':('chordjack',),
    'jack':('jack','jackspeed','minijack','longjack','vibro'),
    'ln':('ln','release','inverse','hybrid','shield'),
    'trill':('jumptrill','trill1h','splittrill'),
    'technical':('technical','patterntech','rhythmtech'), 'sv':('sv',),
}


def family(skill):
    return next((k for k,v in MEMBERS.items() if skill in v), skill if skill in FAMILIES else 'rice')


def dominant(f):
    sk=f.get('sk', {})
    # Descriptive stamina/tech refinements should not erase the underlying
    # execution family, just as in the shared recommendation model.
    names=[k for k in sk if k not in ('stamina','technical','patterntech','rhythmtech','mash')]
    return family(max(names,key=lambda k:sk[k])) if names else 'rice'


def transfer(a,b):
    if a==b:return 1.
    if {a,b}<={'rice','chords','trill'}:return .55
    if {a,b}<={'chordjack','jack'}:return .65
    return .20


def rate_elasticities(features):
    """Rate response learned from the same chart at multiple analysed rates.

    These are chart derivatives, not player outcomes or SR targets. Sparse
    families borrow their keymode response instead of a universal accuracy tax.
    """
    by = collections.defaultdict(list)
    for fs in (features or {}).values():
        rates = sorted(r for r, f in fs.items() if isinstance(r, (int, float)) and r > 0
                       and isinstance(f, dict) and f.get('overall', 0) >= 2.)
        for left, right in zip(rates, rates[1:]):
            a, b = fs[left], fs[right]
            derivative = math.log(b['overall']/a['overall']) / math.log(right/left)
            if .2 <= derivative <= 3.:
                key = a['keys']
                by[str(key)].append(derivative)
                by[f'{key}:{dominant(a)}'].append(derivative)
    return {k: float(np.median(v)) for k, v in by.items() if len(v) >= 20}


def family_weights(f):
    """Continuous execution mix; descriptive Technical is not extra work."""
    # The mix depends only on chart skills, not session form or either score
    # target. Value keys also invalidate correctly if a feature dict changes.
    return _family_weights(tuple(f.get('sk', {}).items()))


@functools.lru_cache(maxsize=32768)
def _family_weights(items):
    sk = dict(items)
    values = []
    for name, members in MEMBERS.items():
        if name == 'technical':
            continue  # description of the same work, not a second motor family
        weight = max((max(0., sk.get(x, 0.)) for x in members), default=0.)**2
        if weight:
            values.append((name, weight))
    total = sum(w for name,w in values)
    return tuple((name,w/total) for name,w in values) if total else ((dominant({'sk':sk}),1.),)


def rate_elasticity(f, elasticities):
    """Blend measured chart-family responses without a winner-label cliff."""
    k = f['keys']; default = elasticities.get(str(k), 1.)
    return sum(w*elasticities.get(f'{k}:{name}', default) for name,w in family_weights(f))


def score_rate_slope(f, population, elasticities):
    """Local log-error / log-rate response; no map-specific score residuals."""
    k = f['keys']; pop = population or {}
    slope = pop.get('slope', {}).get(k, pop.get('slope', {}).get(0, 2.))
    curve = pop.get('curve', {}).get(k, pop.get('curve', {}).get(0, 0.))
    slope += 2 * curve * max(0., math.log(max(.05, f['overall']) / pop.get('curve_knee', 4.)))
    elasticity = rate_elasticity(f, elasticities)
    return max(1., min(8., slope * elasticity))


def duration(row):
    return max(0.,min(1800.,row['f'].get('length',0.)/1000/max(.01,row.get('rate',1.))))


def completions(rows):
    """Exact local timestamps only; public bests cannot establish session order."""
    out=[];seen=set()
    for r in sorted(rows,key=lambda r:r.get('ts') or 0):
        if r.get('src') not in ('realm','live') or not r.get('ts') or r.get('rd') or not r.get('completed',True):
            continue
        # Imported/live copies of one completion are not two warmup maps.
        key=(r['chart'],round(r['ts']),round(r['rate'],3))
        if key in seen:continue
        seen.add(key)
        out.append(dict(r,seconds=duration(r),family=dominant(r['f'])))
    return out


def history_design(rows, tau):
    observations=[];state={};last_end=None;session=-1
    for r in completions(rows):
        end=r['ts'];start=end-r['seconds']
        if last_end is None or start-last_end>GAP:
            state={};session+=1
        else:
            elapsed=max(0.,start-last_end)
            decay=math.exp(-elapsed/COOLING)
            state={k:v*decay for k,v in state.items()}
        k=r['keys']; fam=r['family'];minutes=state.get((k,fam),0.)
        seconds=r['seconds'] if last_end is None else min(r['seconds'],max(0.,end-last_end))
        d=min(10.,seconds/60)
        # The completed score averages cold and warming portions of that map.
        exposure=math.exp(-minutes/tau)*(-math.expm1(-d/tau))*tau/d if d>.001 else math.exp(-minutes/tau)
        observations.append(dict(r,session=session,activation=minutes,exposure=exposure))
        decay=math.exp(-seconds/COOLING)
        state={key:value*decay for key,value in state.items()}
        for target in FAMILIES:
            state[k,target]=state.get((k,target),0.)+d*transfer(target,fam)
        last_end=end
    return observations


def activation(history, now, keys, skill=None):
    """Same elapsed-time accounting as history_design; quits never enter."""
    state=0.;last_end=None;seen=set()
    for a in sorted(history,key=lambda a:a['end']):
        if a.get('kind')!='finish' or not a.get('ability_evidence',True) or a['end']>now:
            continue
        identity=a.get('key') or (a.get('beatmap'),round(a['end'],1))
        if identity in seen:continue
        seen.add(identity)
        seconds=max(0.,min(1800.,a.get('played_seconds') or a.get('length') or 0.))
        start=a.get('t',a['end']-seconds)
        if last_end is None or start-last_end>GAP:
            state=0.
        else:
            state*=math.exp(-max(0.,a['end']-last_end)/COOLING)
        if a.get('keys')==keys:
            played_family = dominant(a['features']) if a.get('features') else family(a.get('skill'))
            related=1. if skill is None else transfer(family(skill),played_family)
            state+=min(10.,seconds/60)*related
        last_end=a['end']
    if last_end is None or now-last_end>GAP:return 0.
    return state*math.exp(-max(0.,now-last_end)/COOLING)


_NUISANCE={}


def nuisance(row):
    # Every refit evaluates the same feature dicts ~25 times (taus × splits): ~1/4 of startup.
    f=row['f'];hit=_NUISANCE.get(id(f))
    if hit is not None and hit[0] is f:return hit[1]
    sk=f.get('sk',{})
    lr=math.log(max(.1,f['overall']))
    v=(1.,lr,lr*lr,f.get('ln',0.),f.get('stam',1.),f.get('endurance',{}).get('load',0.))+tuple(
        max((sk.get(x,0.) for x in MEMBERS[k]),default=0.) for k in FAMILIES)
    if len(_NUISANCE)>200000:_NUISANCE.clear()
    _NUISANCE[id(f)]=(f,v)
    return v


def _fit(rows, tau, use_cold=True):
    """Small mixed ridge model. Nuisance effects are never exposed as ability."""
    n=len(rows);groups={kind:{} for kind in ('month','chart','session')}
    group_keys={
        'month':[int(r['t']) for r in rows],
        'chart':[(r['chart'],round(r['rate'],2)) for r in rows],
        'session':[r['session'] for r in rows],
    }
    indices={kind:np.array([groups[kind].setdefault(x,len(groups[kind])) for x in values]) for kind,values in group_keys.items()}
    Z=np.array([nuisance(r) for r in rows]);W=np.zeros((n,len(FAMILIES)))
    for i,r in enumerate(rows):
        W[i,FAMILIES.index(r['family'])]=r['exposure']*r.get('speed_slope',1.) if use_cold else 0.
    X=np.column_stack([Z,W]); y=np.array([r['y']-r['base'] for r in rows])
    # Repeat/favourite maps cannot dominate; old sessions remain useful.
    count=collections.Counter(group_keys['chart']);recent=max(r['t'] for r in rows)
    weights=np.array([.5**(max(0,recent-r['t'])/18)/math.sqrt(count[c]) for r,c in zip(rows,group_keys['chart'])])
    weights*=len(rows)/weights.sum()
    penalty=np.eye(X.shape[1])*max(1.,.005*weights.sum());penalty[0,0]=1e-6
    # Shrink rare families toward their keymode's curve, not toward perfect
    # cold performance. No coefficient is chosen from the reported failed quit.
    speed_units = 'speed_slope' in rows[0]
    prior_precision = 12. if speed_units else 2.
    g=len(FAMILIES);penalty[-g:,-g:]=12.*(np.eye(g)-np.ones((g,g))/g)+np.eye(g)*prior_precision
    A=X.T@(weights[:,None]*X)+penalty
    effects={kind:np.zeros(len(ids)) for kind,ids in groups.items()}
    coeff=np.zeros(X.shape[1]);ridge={'month':4.,'chart':5.,'session':8.}
    for _ in range(16):
        offsets=sum(effects[k][idx] for k,idx in indices.items())
        rhs=X.T@(weights*(y-offsets))
        if speed_units and use_cold:
            rhs[-g:] += prior_precision * CAPACITY_PRIOR
        coeff=np.linalg.solve(A,rhs)
        # Coordinate projection solves the same convex objective with the
        # physically meaningful nonnegative cold-loss constraints.
        for _sweep in range(40):
            previous=coeff.copy()
            for j in range(len(coeff)):
                value=coeff[j]+(rhs[j]-A[j]@coeff)/A[j,j]
                coeff[j]=max(0.,min(.4 if speed_units else 2.,value)) if j>=len(nuisance(rows[0])) else value
            if np.max(abs(previous-coeff))<1e-7:break
        for kind,idx in indices.items():
            residual=y-X@coeff-sum(effects[k][ii] for k,ii in indices.items() if k!=kind)
            effects[kind]=np.bincount(idx,weights*residual)/(np.bincount(idx,weights)+ridge[kind])
    return {'tau':tau,'amplitudes':coeff[-g:].tolist(),'theta':coeff[:-g],
            'effects':{k:dict(zip(groups[k],v)) for k,v in effects.items()}}


def _evaluate(model, rows):
    errors=[];cold=[]
    months=model['effects']['month'];month_keys=sorted(months)
    for r in rows:
        month=float(np.interp(r['t'],month_keys,[months[k] for k in month_keys]))
        chart=model['effects']['chart'].get((r['chart'],round(r['rate'],2)),0.)
        predicted=float(np.array(nuisance(r))@model['theta'])+month+chart
        predicted+=model['amplitudes'][FAMILIES.index(r['family'])]*r['exposure']*r.get('speed_slope',1.)
        e=r['y']-r['base']-predicted;errors.append(e)
        if r['activation']<1.:cold.append(e)
    return {'rmse':float(np.sqrt(np.mean(np.square(errors)))),
            'cold_rmse':float(np.sqrt(np.mean(np.square(cold)))) if cold else None,
            'rows':len(errors),'cold_rows':len(cold)}


class Model:
    def __init__(self, modes=None, history=(), report=None, population=None, elasticities=None):
        self.modes=modes or {};self.history=list(history);self.report=report or {}
        self.population=population or {};self.elasticities=elasticities or {}
        self.fallback = ({'tau':float(np.median([v['tau'] for v in self.modes.values()])),
                          'amplitudes':np.median([v['amplitudes'] for v in self.modes.values()],axis=0).tolist(),
                          'units':next(iter(self.modes.values())).get('units','log_error'),
                          'support':'pooled'} if self.modes else
                         {'tau':8.,'amplitudes':[CAPACITY_PRIOR]*len(FAMILIES),
                          'units':'log_rate','support':'0.90 capacity prior'})

    def parameters(self, keys):
        if keys in self.modes:return self.modes[keys]
        # No cross-keymode activation. Only borrow a weak pooled curve when a
        # mode has too few complete sessions to identify its own parameters.
        return self.fallback

    def rate_slope(self, f):
        return score_rate_slope(f, self.population, self.elasticities)

    def capacity(self, keys, skill, minutes):
        p=self.parameters(keys);a=p['amplitudes']
        amplitude=float(np.mean(a)) if skill is None else a[FAMILIES.index(family(skill))]
        if p.get('units') != 'log_rate':
            amplitude /= 3.  # legacy diagnostic models only
        return math.exp(-amplitude*math.exp(-max(0.,minutes)/p['tau']))

    def penalty(self, keys, skill, minutes, seconds=0., slope=None):
        p=self.parameters(keys);tau=p['tau'];a=p['amplitudes']
        amplitude=float(np.mean(a)) if skill is None else a[FAMILIES.index(family(skill))]
        d=max(0.,min(10.,seconds/60))
        mean=(-math.expm1(-d/tau))*tau/d if d>.001 else 1.
        if p.get('units') == 'log_rate':
            amplitude *= 3. if slope is None else slope
        return amplitude*math.exp(-max(0.,minutes)/tau)*mean


def fit(rows, population=None, features=None):
    # Retain the established model as a chronological control. Sparse modes
    # must not inherit a large rate deficit from a different keymode; a new
    # response is enabled only when its own earlier-session DEV check wins.
    baseline, baseline_adjust = fit(rows) if population is not None else (None, {})
    elasticities=rate_elasticities(features) if population is not None else {}
    if not rows:return Model(population=population,elasticities=elasticities),{}
    if population is not None:
        rows=[dict(r,speed_slope=score_rate_slope(r['f'],population,elasticities)) for r in rows]
    layouts={tau:history_design(rows,tau) for tau in TAUS}
    modes={};report={};adjust={}
    for keys in sorted({r['keys'] for r in layouts[8.]}):
        ds=[r for r in layouts[8.] if r['keys']==keys]
        sessions=sorted({r['session'] for r in ds})
        if len(ds)<100 or len(sessions)<25:continue
        cut=sessions[max(1,int(len(sessions)*.7))]
        trials=[]
        for tau,obs in layouts.items():
            train=[r for r in obs if r['keys']==keys and r['session']<cut]
            dev=[r for r in obs if r['keys']==keys and r['session']>=cut]
            candidate=_fit(train,tau);stats=_evaluate(candidate,dev)
            score=stats['rmse']+(stats['cold_rmse'] or stats['rmse'])
            trials.append((score,tau,stats))
        _,tau,devstats=min(trials)
        obs=[r for r in layouts[tau] if r['keys']==keys]
        train=[r for r in obs if r['session']<cut];dev=[r for r in obs if r['session']>=cut]
        null=_fit(train,tau,False);no_cold=_evaluate(null,dev)
        final=_fit(obs,tau)
        modes[keys]={'tau':tau,'amplitudes':final['amplitudes'],'support':'local completed sessions',
                    'units':'log_rate' if population is not None else 'log_error'}
        report[keys]={'sessions':len(sessions),'completed_rows':len(ds),'cold_rows':sum(r['activation']<1. for r in ds),
                      'selected_tau':tau,'dev':devstats,'no_cold_dev':no_cold,
                      'amplitudes':dict(zip(FAMILIES,final['amplitudes'])), 'units':modes[keys]['units']}
        if population is not None:
            report[keys]['cold_capacity']={k:math.exp(-v) for k,v in zip(FAMILIES,final['amplitudes'])}
        for r in obs:
            adjust[r['key']]=final['amplitudes'][FAMILIES.index(r['family'])]*r['exposure']*r.get('speed_slope',1.)
        if baseline is not None and keys in baseline.report:
            control=baseline.report[keys]['dev']
            objective=lambda v: v['rmse']+(v['cold_rmse'] or v['rmse'])
            accepted=objective(devstats)<objective(control) and all(
                devstats[k]<=1.01*control[k] for k in ('rmse','cold_rmse') if control[k] is not None)
            report[keys]['rate_response_accepted']=accepted
            report[keys]['constant_response_dev']=control
            if not accepted:
                modes[keys]=baseline.modes[keys]
                report[keys]['retained_units']='log_error'
                for r in obs:adjust[r['key']]=baseline_adjust.get(r['key'],0.)
    # Keep only compact recent completions to seed a live session after restart.
    history=[{'end':r['ts'],'t':r['ts']-r['seconds'],'keys':r['keys'],'skill':r['family'],
              'played_seconds':r['seconds'],'key':r['key'],'beatmap':'md5:'+r['chart'],
              'kind':'finish','ability_evidence':True,'meaningful':True}
             for r in completions(rows)[-64:]]
    model=Model(modes,history,report,population,elasticities)
    if baseline is not None:
        model.fallback=baseline.fallback
    return model,adjust
