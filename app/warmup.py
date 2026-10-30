"""Completed-score cold-to-warm learning, independent of quits and targets.

Session starts are inferred from timestamped local completions, never dated
public bests. Chart/rate, month, pattern mix and session intercepts absorb
familiarity, skill drift and good/bad days. Only earlier sessions select the
decay curve; a late-session holdout is diagnostic, not a fitting target.
The fit itself runs natively with recommend.Personal's (native/src/personal.rs).
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


# Descriptive stamina/tech refinements should not erase the underlying
# execution family, just as in the shared recommendation model.
DESCRIPTIVE = ('stamina','technical','patterntech','rhythmtech','mash')


@functools.cache
def _families():
    import native_backend, recdata
    return native_backend.load().Families(list(recdata.SKILLS), list(FAMILIES),
                                          [(k, list(v)) for k, v in MEMBERS.items()], list(DESCRIPTIVE))


def dominant(f):
    """The family of the chart's strongest non-descriptive skill ('rice' without one)."""
    return _families().dominant(f)


def transfer(a,b):
    if a==b:return 1.
    if {a,b}<={'rice','chords','trill'}:return .55
    if {a,b}<={'chordjack','jack'}:return .65
    return .20


_ELASTICITIES = [None, None]


def rate_elasticities(features):
    """Rate response learned from the same chart at multiple analysed rates.

    These are chart derivatives, not player outcomes or SR targets. Sparse
    families borrow their keymode response instead of a universal accuracy tax.
    """
    # The public features are fixed for a snapshot; every refit fits two models from them.
    if features is not None and _ELASTICITIES[0] is features:
        return dict(_ELASTICITIES[1])
    out = _rate_elasticities(features)
    if features is not None:
        _ELASTICITIES[:] = features, out
    return dict(out)


def _rate_elasticities(features):
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
    """Continuous execution mix ((family, weight), ...) in MEMBERS order: each family's strongest member
    squared, normalised; descriptive Technical is not extra work. Natively (~160k calls per fit)."""
    return _families().weights(f)


def _elasticity_of(elasticities):
    """f, family_weights(f) → the chart's blended rate elasticity; the per-keymode family table built once."""
    tables = {}
    def of(f, weights):
        k = f['keys']; table = tables.get(k)
        if table is None:
            default = elasticities.get(str(k), 1.)
            table = tables[k] = {name: elasticities.get(f'{k}:{name}', default) for name in FAMILIES}
        return sum(w*table[name] for name,w in weights)
    return of


def rate_elasticity(f, elasticities):
    """Blend measured chart-family responses without a winner-label cliff."""
    return _elasticity_of(elasticities)(f, family_weights(f))


def score_rate_slopes(fs, population, elasticities):
    """Local log-error / log-rate response per chart; no map-specific score residuals."""
    pop = population or {}; elasticity = _elasticity_of(elasticities); out = []
    for f, weights in zip(fs, _families().weights_many(fs)):
        k = f['keys']
        slope = pop.get('slope', {}).get(k, pop.get('slope', {}).get(0, 2.))
        curve = pop.get('curve', {}).get(k, pop.get('curve', {}).get(0, 0.))
        slope += 2 * curve * max(0., math.log(max(.05, f['overall']) / pop.get('curve_knee', 4.)))
        out.append(max(1., min(8., slope * elasticity(f, weights))))
    return out


def score_rate_slope(f, population, elasticities):
    return score_rate_slopes([f], population, elasticities)[0]


def native_args(population, elasticities):
    """What the native warm-up fit (native/src/personal.rs, run by recommend.Personal) reads: transfer between
    FAMILIES, GAP, COOLING, TAUS, CAPACITY_PRIOR and the population's rate response (score_rate_slope)."""
    pop = population or {}
    return ([[transfer(target, played) for played in FAMILIES] for target in FAMILIES], float(GAP), float(COOLING),
            list(TAUS), CAPACITY_PRIOR, {"slope": pop.get("slope", {}), "curve": pop.get("curve", {}),
                                         "curve_knee": pop.get("curve_knee", 4.), "elasticities": elasticities})


def timeline(rows):
    """(row, session, readiness minutes, exposure minutes, seconds, family) per completion in time order: exact
    local completions only (public bests cannot establish session order), readiness per (keymode, family)
    cooling with COOLING between and during maps, reset after a GAP (the native fit's own accounting)."""
    import native_backend, recommend
    transfer_, gap, cooling, *_ = native_args(None, {})
    return [(rows[i], session, minutes, d, seconds, FAMILIES[family]) for i, session, minutes, d, seconds, family in
            native_backend.load().warmup_steps(recommend._native_consts(), rows, transfer_, gap, cooling)]


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


def models(fitted, population, elasticities):
    """warmup.Model of a native fit's ((control modes, report), (modes, report), history): completed-score
    cold-to-warm learning; keymodes whose rate response lost its earlier-session DEV check keep the
    constant-response control, and the fallback is the control's."""
    (control_modes, control_report), (modes, report), history = fitted
    model = Model(modes, history, report, population, elasticities)
    model.fallback = Model(control_modes, history, control_report).fallback
    return model
