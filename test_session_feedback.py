"""Small controls for signed skill choices and gentle density-tail preferences."""
import math

import skill_practice as S
import nps
import recommend as R
import warmup as W


def test_signed_cycle_parent_and_child_exceptions():
    selected = S.cycle((), 'sv')
    assert selected == ('sv',)
    assert all(S.state(selected, c.key) == 1 for c in S.SV.children)
    selected = S.cycle(selected, 'sv')
    assert selected == ('!sv',)
    assert all(S.state(selected, c.key) == -1 for c in S.SV.children)
    assert S.cycle(selected, 'sv') == ()
    selected = S.cycle(('ln',), 'ln/inverse')
    assert S.state(selected, 'ln/inverse') == -1
    assert S.state(selected, 'ln/release') == 1
    assert S.mixed(selected, 'ln')
    selected = S.cycle(selected, 'ln/inverse')
    assert S.state(selected, 'ln/inverse') == 0
    assert S.state(selected, 'ln/release') == 1
    assert S.cycle(selected, 'ln') == ('ln',)


def test_blacklist_is_material_and_works_without_positive_choices():
    f = {'keys': 7, 'sk': {'chordstream': 1., 'sv': .05}}
    assert S.match(f, ['!sv'])[0] == 1
    assert S.match(f, ['chordstream', '!sv'])[0] > 0
    f['sk']['sv'] = .8
    assert S.match(f, ['!sv'])[0] == 0
    assert S.match(f, ['chordstream', '!sv'])[0] == 0
    assert 'exclude SV' in S.label(['!sv'])
    assert not S.match({'keys': 7, 'sk': {'vibro': 1.}}, ['!jackspeed'])[0]
    assert not S.wants_vibro(['jackspeed'])  # no implicit dedicated vibro drills


def test_restored_parent_exceptions_agree_with_the_visible_children():
    selected = S.normalize(['!ln', 'ln/inverse'])
    assert '!ln' not in selected
    assert S.state(selected, 'ln/inverse') == 1
    assert S.state(selected, 'ln/release') == -1
    assert S.match({'keys': 7, 'sk': {'ln': 1., 'inverse': .9}}, selected)[0] > 0
    assert S.normalize(selected) == selected


def test_trills_are_independent_and_old_choices_migrate():
    assert {'jumptrill', 'splittrill'} <= {n.key for n in S.tree(4)}
    assert 'jumptrill' in {n.key for n in S.tree(7)}
    assert S.normalize(['!chordstream/jumptrill']) == ('!jumptrill',)
    assert S.normalize(['jumpstream/splittrill']) == ('splittrill',)
    f = {'keys': 7, 'sk': {'jumptrill': 1.}}
    assert S.match(f, ['jumptrill'])[0] == 1
    assert S.match(f, ['chordstream'])[0] == 0
    assert S.match(f, ['splittrill'])[0] == 0


def test_nps_lower_tail_is_rare_not_banned_and_twenty_stays_variety():
    rows = [dict(keys=7, family=str(i), md5=str(i), rate=1., nps=ns,
                 base_value=math.log(ns), length=180., profile=[1.],
                 description='Mixed', group='chords', acc_mid=.94)
            for i, ns in enumerate([16., 18., 20., 23., 27., 30., 32., 33., 34., 35.])]
    result = nps.weighted_candidates(rows, R.Session([], 1000), [7])
    by = {r['nps']: r['weight'] for r in result}
    assert 0 < by[16.] / by[20.] < .005
    assert 0 < by[18.] / by[20.] < .08
    assert .6 < by[20.] / by[32.] < .7
    assert len(result) == len(rows)
    sparse = [dict(r, nps=r['nps']/3, base_value=math.log(r['nps']/3)) for r in rows]
    lower = nps.weighted_candidates(sparse, R.Session([],1000), [7])
    assert len(lower) == len(rows) and all(r['weight'] > 0 for r in lower)
    # An easier eligible pool lowers the reference; it does not reject every
    # chart or funnel all mass into its single densest map.
    assert max(r['weight'] for r in lower) / sum(r['weight'] for r in lower) < .25
    skills = nps.weighted_candidates([dict(r, practice_description='Chords') for r in rows],
                                    R.Session([],1000), [7], mode='skills')
    sw = {r['nps']: r['weight'] for r in skills}
    assert abs(sw[16.] / sw[20.] - .8) < 1e-12


def test_capacity_prior_changes_rate_demand_not_accuracy_goal():
    m = W.Model()
    assert abs(m.capacity(7, 'chordstream', 0.) - .9) < 1e-12
    assert .9 < m.capacity(7, 'chordstream', 4.) < m.capacity(7, 'chordstream', 12.) < 1.
    assert m.penalty(7, 'chordstream', 0., 120., 5.) > m.penalty(7, 'chordstream', 0., 120., 2.)
    assert m.penalty(7, 'chordstream', 0., 240., 3.) < m.penalty(7, 'chordstream', 0., 0., 3.)
    assert nps.central_target(R.Session([], 1000)) == .94


def test_cold_opening_cannot_borrow_warmth_from_the_same_maps_ending():
    s = R.Session([], 1000)
    f = {'keys': 7, 'sk': {'chordstream': 1.}}
    assert not nps.eligible({'acc_mid': .944, 'opening_acc': .91}, s, f)
    assert nps.eligible({'acc_mid': .946, 'opening_acc': .94}, s, f)
    s.warm_flag = True
    assert nps.eligible({'acc_mid': .944, 'opening_acc': .91}, s, f)


def test_rate_search_actually_moves_toward_an_eligible_cold_opening():
    from types import SimpleNamespace
    from test_skill_practice import feature, inst
    f = feature({'chordstream': 1.})
    class Predictor:
        model = SimpleNamespace(pop={'slope': {0: 3.}})
        md5_bid = {}
        def predict(self, f, chart, bid, rate, session, *, accuracy_only=False):
            return {'acc_mid': .94, 'opening_acc': .91, 'sd_model': .1}
    s = R.Session([], 1000)
    candidates, proposals = nps.candidates_from({'a': inst('a')}, {'a': {1.: f}}, Predictor(), s)
    assert not candidates and len(proposals) == 1
    assert .85 < proposals[0][2] < .95
    assert nps.rate_shift({'acc_mid': .94, 'opening_acc': .91}, s, f, .94) < 0
    s.warm_flag = True
    assert nps.rate_shift({'acc_mid': .94, 'opening_acc': .91}, s, f, .94) == 0


def test_cold_refinement_does_not_stop_in_an_ineligible_rounding_cell():
    from types import SimpleNamespace
    from test_skill_practice import feature, inst
    f = feature({'chordstream': 1.})
    class Predictor:
        model = SimpleNamespace(pop={'slope': {0: 3.}})
        md5_bid = {}
        def predict(self, *args, accuracy_only=False):
            return {'acc_mid': .95, 'opening_acc': .9369, 'sd_model': .1}
    choices, proposals = nps.candidates_from({'a': inst('a')}, {'a': {1.35: f}}, Predictor(), R.Session([],1000))
    assert not choices and proposals[0][2] == 1.34


def test_readiness_uses_the_same_execution_family_as_historical_fitting():
    f = {'keys': 7, 'overall': 8., 'sk': {'technical': 1., 'mash': .95, 'chordjack': .8}}
    history = [{'key': 'one', 't': 1000., 'end': 1120., 'keys': 7,
                'kind': 'finish', 'played_seconds': 120., 'skill': 'technical', 'features': f}]
    assert W.dominant(f) == 'chordjack'
    assert W.activation(history, 1120., 7, 'chordjack') == 2.
    assert W.activation(history, 1120., 7, 'rice') < 2.


def test_capacity_does_not_jump_when_two_leading_skills_trade_places():
    m = W.Model({7:{'tau':4., 'units':'log_rate', 'amplitudes':[.08,.12]+[.1]*6}})
    s = R.Session([],1000,warmup_model=m)
    a = {'keys':7,'overall':8.,'sk':{'delay':.500001,'chordstream':.5}}
    b = {'keys':7,'overall':8.,'sk':{'delay':.5,'chordstream':.500001}}
    pa = s.warmup_penalty(7,'delay',f=a,response=lambda loss:loss)
    pb = s.warmup_penalty(7,'chordstream',f=b,response=lambda loss:loss)
    assert abs(pa-pb) < 1e-6 and abs(pa-.1) < 1e-6


def test_cached_family_capacity_matches_direct_model_integration():
    import math
    m = W.Model({7:{'tau':4., 'units':'log_rate', 'amplitudes':[.08,.12,.14,.13,.18,.09,.1,.11]}})
    s = R.Session([],1000,warmup_model=m)
    minutes = dict(zip(W.FAMILIES,[2.,3.,.5,1.,0.,2.,1.,0.]))
    s.activation = lambda keys,fam=None: minutes[W.family(fam)]
    f = {'keys':7,'overall':8.,'sk':{'delay':.7,'chordstream':.8,'ln':.4,'sv':.2}}
    weights = W.family_weights(f)
    for seconds in (0.,30.,180.,900.):
        direct = sum(w*m.penalty(7,fam,minutes[fam],seconds,1.) for fam,w in weights)
        actual = s.warmup_penalty(7,'chordstream',seconds,f,response=lambda loss:loss)
        assert math.isclose(actual,direct,abs_tol=1e-14)
    assert len(s._corrections['capacity-families',7]) == len(W.FAMILIES)


def test_rapid_filter_changes_skip_superseded_catalogue_searches():
    from unittest.mock import patch
    w = R.Worker(lambda *_: None)
    modes = {'pp': [7], 'nps': [7], 'skills': [7]}
    w.put('configure', 'skills', modes, 'auto', 1, ['sv'])
    w.put('configure', 'skills', modes, 'auto', 2, ['!sv'])
    with patch.object(w, '_publish') as publish, patch.object(w, '_request_nps') as build:
        task = w.q.get_nowait(); w.t_configure(*task[1:])
        assert not publish.called and not build.called
        task = w.q.get_nowait(); w.t_configure(*task[1:])
        assert publish.call_count == build.call_count == 1
    assert w.practice_skills == ('!sv',) and w.revision == 2


def test_technical_practice_requires_presence_not_only_a_local_peak():
    f = {'keys': 7, 'sk': {'chordstream': 1., 'technical': .85, 'patterntech': .8},
         'tech_profile': {'technical': .10, 'patterntech': .08, 'rhythmtech': .02}}
    assert S.match(f, ['technical'])[0] == 0
    assert S.match(f, ['technical/patterntech'])[0] == 0
    # A material forbidden hard passage is still forbidden, even though that
    # passage alone does not make this a good whole-map practice example.
    assert S.match(f, ['!technical'])[0] == 0
    f['tech_profile'].update(technical=.85, patterntech=.72)
    assert S.match(f, ['technical'])[0] > .8
    assert S.match(f, ['technical/patterntech'])[0] >= .8


def test_whole_map_technical_presence_distinguishes_regular_from_irregular_hands():
    import pattern_control as P
    import skill_calc as C
    from test_skill_calc import chart
    regular = [(i*80, i%4) for i in range(400)]
    irregular=[];t=0.
    for i in range(400):
        t += (60.,93.,47.,111.,75.)[i%5]
        irregular.append((t,i%4))
    a=C.parse_osu(chart(regular,keys=4));b=C.parse_osu(chart(irregular,keys=4))
    pa,pb=P.tech_profile(a),P.tech_profile(b)
    assert pa['technical'] < .1
    assert pb['rhythmtech'] > .5 and pb['technical'] > .5
    assert P.technical_description('Dump / Minijack',{'technical':.72},pb) == 'Tech Dump / Minijack'
    assert P.technical_description('Dump / Minijack',{'technical':.72},pa) == 'Dump / Minijack'
