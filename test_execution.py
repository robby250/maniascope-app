"""Focused structural controls for compact Mash and honest Tech/SV attribution."""
from unittest.mock import patch

import difficulty_model as D
import execution
import skill_calc as S


def chart(notes, keys=7, od=5):
    c=S.Chart();c.keys=keys;c.od=od;c.sv=[]
    c.title=c.artist=c.version=c.creator="synthetic control"
    c.notes=sorted(notes)
    return c


def test_compact_pseudo_chords_are_not_continuous_delay_or_jumptrills():
    pseudo=chart([(1000+180*i+8*j,1000+180*i+8*j,j) for i in range(160) for j in range(7)])
    precise=chart([(1000+35*i,1000+35*i,i%7) for i in range(1000)])
    alternating=chart([(1000+85*i,1000+85*i,c) for i in range(400)
                       for c in ((0,1,2) if i%2 else (3,4,5,6))])
    p=S.compute(pseudo,1)
    assert p['execution']['pseudo_mash'] > .5
    assert p['scores']['mash'] > p['scores']['technical']
    assert S.compute(precise,1)['execution']['mash'] < .02
    assert S.compute(alternating,1)['execution']['mash'] < .02


def test_changing_columns_and_independent_holds_still_require_control():
    changing=chart([(1000+160*i,1000+160*i,c) for i in range(300)
                    for c in ((0,2,4,6) if i%2 else (0,3,4,5))])
    fixed=chart([(1000+160*i,1080+160*i,c) for i in range(200) for c in range(7)])
    a=S.compute(changing,1);b=S.compute(fixed,1)
    assert a['execution']['mash'] < .03
    assert a['scores']['chordjack'] > a['scores']['mash']
    assert b['execution']['mash'] > .7
    # Tail synchronization is evidence separate from head width, not a rule
    # erasing independent LN mechanics on every wide chord.
    async_ln=chart([(1000+400*i,1070+400*i+45*c,c) for i in range(50) for c in range(7)])
    assert execution.analyse(async_ln,1)['release_sync'] < execution.analyse(fixed,1)['release_sync'] or \
        S.compute(async_ln,1)['execution']['mash'] < b['execution']['mash']


def test_execution_effect_is_fitted_not_an_unconditional_mash_discount():
    f=[0.]*len(D.FEATURES)
    with patch.object(D,'parameters',return_value={}):
        assert D.final_rating(7,8.,f,{'pseudo_mash':1.}) == 8.
    params={'execution':{'modes':{'7':{'slope':2.,'curve':0.,'coeff':[0.,-.4,0.,0.,.3,0.]}}}}
    with patch.object(D,'parameters',return_value=params):
        assert D.final_rating(7,8.,f,{'pseudo_mash':1.}) < 8.
        assert D.final_rating(7,8.,f,{'peak_pressure':1.}) > 8.
        assert D.final_rating(9,8.,f,{'pseudo_mash':1.}) == 8.  # unvalidated mode stays unchanged


def test_monotone_brake_is_not_stutter_and_total_sv_strain_is_retained():
    c=chart([(1000+i*400,1000+i*400,i%7) for i in range(80)])
    c.sv=sorted((t,v) for i in range(80) for t,v in
                ((650+i*400,1.),(800+i*400,.7),(920+i*400,.4)))
    kinds=[];strain=S.sv_reading(c,kinds)
    assert sum(strain)>0
    assert sum(s*k[3] for s,k in zip(strain,kinds)) < .1*sum(strain)
    assert all(abs(sum(k)-1)<1e-8 for k,s in zip(kinds,strain) if s>0)


def test_zero_length_two_note_chart_does_not_crash_the_catalogue_builder():
    result=S.compute(chart([(1000,1000,0),(1000,1000,6)]),1.)
    assert result['scores']['overall']==0.
    assert result['scores']['stamina']==0.
