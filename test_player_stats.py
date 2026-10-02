import math
from types import SimpleNamespace

import player_stats as P
import recdata
import recommend as R
import score_units


def fixture(tmp_path, high=10., count=12):
    db=recdata.connect(str(tmp_path/'stats.db'))
    class Model:
        keys=[7];now=320.;level_var={7:.01}
        def predict(self,f,*args):
            return math.log(.06)+2*math.log(f['overall']/8.), .1, .2
    rows=[]
    for i in range(count):
        f={'overall':high*(.7+.3*i/max(1,count-1)), 'sk':{'ln':1.,'release':.8,'stream':.15},'keys':7}
        rows.append({'chart':str(i),'f':f,'keys':7,'ts':1000+i,'t':319.,'rate':1.,'src':'realm'})
    return SimpleNamespace(db=db,rows=rows,model=Model(),_shown=(0.,1.),installed={},pub={'maps':{}})


def test_ability_number_inverts_the_shared_model_and_session_overlay_is_separate(tmp_path):
    rec=fixture(tmp_path)
    profile=P.build(rec)
    baseline=profile[7]['groups']['ln']['value']
    assert abs(baseline-8.)<1e-4
    cold=P.current(profile,R.Session([],1000))['pages'][7]['groups']['ln']
    displayed=score_units.displayed_value(baseline,7)
    assert cold['current']<displayed and profile[7]['groups']['ln']['value']==baseline
    warm=R.Session([],1000);warm.warm_flag=True
    assert abs(P.current(profile,warm)['pages'][7]['groups']['ln']['current']-displayed)<1e-6
    rec.db.close()


def test_sparse_and_unmeasured_limits_are_not_confident_numeric_ratings(tmp_path):
    rec=fixture(tmp_path,high=3.)
    profile=P.build(rec)
    assert profile[7]['groups']['ln']['value'] is None
    assert profile[7]['groups']['ln']['extrapolated']
    assert profile[10]['groups']['overall']['value'] is None
    rec.rows=rec.rows[:2]
    assert P.build(rec)[7]['groups']['ln']['value'] is None
    rec.db.close()


def test_history_follows_the_monthly_level_and_marks_the_peak(tmp_path):
    import numpy as np
    rec=fixture(tmp_path)
    # Month 300 was 0.2 better (fewer misses) than now; month 301 has too few recorded plays to show.
    # 100 plays a month; month 301 unplayed; month 300 went 0.2 better (fewer misses) than the rest.
    sw=np.array([100.,0.,100.,100.,100.,100.,100.]);se=np.array([-20.,0.,0.,0.,0.,0.,0.])
    rec.model.monthly={7:(300,se,sw)}
    profile=P.build(rec)
    history=profile[7]['history'];now=profile[7]['groups']['overall']
    assert [m for m,_v in history]==[300,302,303,304,305,306] and abs(history[-1][1]-now['value'])<1e-9
    assert profile[7]['peak'][0]==300 and history[0][1]>history[1][1]>now['value']
    page=P.current(profile,R.Session([],1000))['pages'][7]
    assert not page['session'] and page['peak'][1]>page['groups']['overall']['value']
    rec.db.close()
