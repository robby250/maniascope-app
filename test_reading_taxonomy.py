"""Focused controls for the shared taxonomy and observable reading geometry."""
import math
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import skill_practice as T
import skill_calc as S
import scroll_reading as R
import pattern_control as P
from test_execution import chart


def test_keymode_taxonomy_and_narrowing_do_not_change_union_semantics():
    four={p.name for p in T.tree(4)};wide={p.name for p in T.tree(7)}
    assert {'Stream','Jumpstream','Handstream','Jackspeed','Chordjack'} <= four
    assert {'Delay','Chordstream','Bracket','Jackspeed','Chordjack'} <= wide
    assert not {'Chords','Jacks','Trills'} & (four|wide)
    parent=T.toggle((), 'ln', True)
    child=T.toggle((),'ln/release',True)
    assert child==('ln/release',)
    selected=T.toggle(child,'ln/inverse',True)
    assert set(selected)=={'ln/release','ln/inverse'}
    assert T.toggle(selected,'ln',True)==('ln',)
    f={'keys':7,'sk':{'ln':1.,'release':.8,'inverse':.1}}
    assert T.match(f,selected)[1]==('ln/release',)
    assert T.match(f,['handstream/quadstream'])[0]==0
    assert T.normalize(['trill']) and 'trill' not in T.normalize(['trill'])
    assert T.wants_vibro(['jackspeed/vibro']) and not T.wants_vibro(['jackspeed'])


def test_quadstream_recognition_has_no_new_physical_charge_and_no_wide_name():
    def make(pattern, keys=4):
        return chart([(1000+80*i,1000+80*i,c) for i in range(800) for c in pattern[i%len(pattern)]],keys=keys)
    c=make(((0,),tuple(range(4)),(2,),tuple(range(4))))
    a=S.compute(c)
    assert 'Quadstream' in S.describe(a)[0]
    with patch.object(P,'quadstream',return_value=([0.]*(int((c.notes[-1][0]-1000)/500)+1),0.)):
        b=S.compute(c)
    assert abs(a['scores']['overall']-b['scores']['overall'])<1e-9
    assert S.compute(make(((0,1),tuple(range(4)),(2,3),tuple(range(4)))))['scores']['quadstream']==0
    wide=make(((0,),tuple(range(7)),(6,),tuple(range(7))),7)
    assert P.quadstream(wide,1)[1]>.8 and 'quadstream' not in S.compute(wide)['scores']
    rare=chart([(1000+i*90,1000+i*90,i%4) for i in range(1600)],keys=4)
    rare.notes=sorted(rare.notes+[(46090,46090,c) for c in range(4) if c!=1])
    assert 'Quadstream' not in S.describe(S.compute(rare))[0]


def test_constant_scroll_setting_and_uniform_multiplier_do_not_create_sv():
    c=chart([(1000+i*100,1000+i*100,i%4) for i in range(100)],keys=4)
    c.sv=[(0,2.)]
    assert not any(R.reading(c))
    assert S.compute(c)['scores']['overall']==S.compute(c,sv=False)['scores']['overall']
    c.sv=[(0,1.),(3000,.3),(3400,1.),(6000,2.),(6600,1.)]
    a=R.geometry(c)
    c.sv=[(t,m*3) for t,m in c.sv]
    b=R.geometry(c)
    assert a['summary']==pytest.approx(b['summary'],abs=1e-9)


def test_sv_rate_compensation_matches_retimed_chart_and_preserves_no_sv_mode():
    c=chart([(1000+i*80,1000+i*80,i%4) for i in range(200)],keys=4)
    c.sv=[(0,1.),(3000,.25),(3400,2.5),(3500,1.),(7000,.1),(7700,1.)]
    retimed=chart([(t/1.23,e/1.23,k) for t,e,k in c.notes],keys=4)
    retimed.sv=[(t/1.23,m) for t,m in c.sv]
    assert R.geometry(c,1.23)['summary']==pytest.approx(R.geometry(retimed)['summary'],abs=1e-9)
    plain=S.compute(c,1.23,sv=False)
    assert plain['scores']['sv']==0 and not plain['execution'].get('scroll_base')
    included=S.compute(c,1.23)
    assert included['scores']['overall']>=plain['scores']['overall']
    assert included['no_sv_overall']==plain['scores']['overall']


def test_fine_stutter_is_not_averaged_into_constant_speed_or_infinite_slow_cost():
    c=chart([(1000+i*80,1000+i*80,i%4) for i in range(200)],keys=4)
    c.sv=[(0,1.)]+[(t,m) for i in range(300) for t,m in ((2000+10*i,.02),(2005+10*i,1.98))]+[(5000,1.)]
    assert R.geometry(c)['summary']['stutter']>0
    c.sv=[(0,1.),(2000,.001),(3000,1.)]
    slow=R.geometry(c)
    assert max(v[1] for v in slow['vectors'])<=1
    kinds=[];strain=R.reading(c,kinds=kinds)
    assert all(math.isfinite(v) for v in strain)
    assert all(abs(sum(k)-1)<1e-9 for k,v in zip(kinds,strain) if v>0)
    for rate in (0,-1,float('nan'),float('inf')):
        with pytest.raises(ValueError):R.geometry(c,rate)


def test_selected_native_failure_is_an_error_not_viewer_exit():
    import selected_analysis
    bad=SimpleNamespace(returncode=-6,stderr='native helper aborted',stdout='')
    with patch.object(selected_analysis.subprocess,'run',return_value=bad):
        with pytest.raises(RuntimeError,match='native helper aborted'):
            selected_analysis.calculate('/nonexistent',1.,'fixture')
    good=SimpleNamespace(returncode=0,stderr='',stdout='{"0.98":{"overall":8.0}}')
    with patch.object(selected_analysis.subprocess,'run',return_value=good):
        assert selected_analysis.calculate('/nonexistent',.98,'fixture')=={.98:{'overall':8.}}


def test_native_pp_batch_retains_completed_scores_after_a_later_abort():
    import selected_analysis
    partial=SimpleNamespace(returncode=-6,stderr='native abort',stdout='["first",123.5]\n')
    with patch.object(selected_analysis.subprocess,'run',return_value=partial):
        assert selected_analysis.performance([{'key':'first'},{'key':'second'}])=={'first':123.5}


def test_native_pp_timeout_preserves_completed_results():
    import selected_analysis
    import subprocess
    timeout=subprocess.TimeoutExpired('pp helper',120,output=b'["first",42.0]\n["second",')
    with patch.object(selected_analysis.subprocess,'run',side_effect=timeout):
        assert selected_analysis.performance([{'key':'first'},{'key':'second'}])=={'first':42.}


def test_native_pp_child_reproduces_the_calculator():
    import selected_analysis
    import rosu_pp_py as rp
    from test_skill_calc import chart as chart_file
    path=chart_file([(1000+80*i,i%4) for i in range(1000)],keys=4)
    stats=[900,90,5,2,1,2]
    expected=rp.Performance(lazer=True,n_geki=900,n300=90,n_katu=5,n100=2,n50=1,misses=2).calculate(rp.Beatmap(path=path)).pp
    actual=selected_analysis.performance([{'key':'fixture','path':path,'stats':stats,'mods':[]}])
    assert actual['fixture']==pytest.approx(expected,rel=1e-10)


def test_invisible_velocity_flicker_costs_less_than_visible_stutter():
    def flicker(period):
        c=chart([(1000+i*80,1000+i*80,i%4) for i in range(160)],keys=4)
        c.sv=[(0,1.)]+[(t,m) for base in range(1000,11000,period)
                      for t,m in ((base,.02),(base+period/2,1.98))]+[(11000,1.)]
        return R.geometry(c)['summary']['stutter']
    assert flicker(4) < .3*flicker(80)


def test_redundant_sv_control_points_do_not_change_reading():
    c=chart([(1000+i*80,1000+i*80,i%4) for i in range(120)],keys=4)
    c.sv=[(0,1.),(3000,.2),(4000,2.),(4300,1.)]
    first=R.geometry(c)['summary']
    c.sv=[(0,1.),(1500,1.),(2000,1.),(3000,.2),(3500,.2),(4000,2.),(4100,2.),(4300,1.)]
    assert R.geometry(c)['summary']==pytest.approx(first,abs=1e-12)


def test_straight_approaches_keep_preview_but_have_no_motion():
    c=chart([(1000+i*80,1000+i*80,i%4) for i in range(200)],keys=4)
    c.sv=[(0,1.),(6000,1.3),(9000,1.)]
    result=R.geometry(c)
    flat=[v for (t,_,_),v in zip(c.notes,result['vectors']) if 7000<t<8500]
    assert flat and all(v[0]>0 and v[2:]==(0.,0.) for v in flat)
    assert any(v[2]>0 for (t,_,_),v in zip(c.notes,result['vectors']) if 6000<t<6500)


def test_scroll_weights_select_only_calibrated_keymode():
    config={'scroll_reading': {'weights':[1.,2.,3.,4.],
                              'modes': {'4': {'weights':[2.,3.,4.,5.]}}}}
    geometry={'vectors':[(1.,1.,1.,1.)], 'kinds':[(0.,0.,1.,0.,0.)]}
    with patch('difficulty_model.parameters',return_value=config), patch.object(R,'geometry',return_value=geometry):
        assert R.reading(SimpleNamespace(keys=4))==[14.]
        assert R.reading(SimpleNamespace(keys=7))==[10.]
        assert R.reading(SimpleNamespace(keys=10))==[10.]
