import numpy as np
import session_form as F


PRIOR={'global_variance':.05,'skill_variance':.12,'noise_variance':.04}


def attempt(i,error,skill='sv',keys=7,chart=None):
    return {'keys':keys,'skill':skill,'kind':'finish','meaningful':True,'ability_evidence':True,
            'mu':-3.,'base_mu':-3.,'y':-3.+error,'cold_penalty':0.,'sd':.25,
            'end':1000+i*100,'beatmap':chart or str(i),'rate':1.}


def test_repeated_bad_skill_evidence_is_not_dismissed_as_unrelated_outliers():
    rows=[attempt(i,.8) for i in range(4)]
    mean,_=F.posterior(rows,1400,7,PRIOR)
    assert F.loading('sv')@mean>.5
    assert F.loading('chordstream')@mean < F.loading('sv')@mean


def test_one_bad_chart_does_not_override_several_good_different_charts():
    rows=[attempt(i,-.15,'chordstream') for i in range(5)]+[attempt(5,2.,'ln')]
    mean,_=F.posterior(rows,1600,7,PRIOR)
    assert abs(F.loading('chordstream')@mean)<.15
    assert F.loading('ln')@mean>F.loading('chordstream')@mean


def test_other_keymode_and_repeated_chart_do_not_fake_independent_evidence():
    a,_=F.posterior([attempt(0,.5)],1200,7,PRIOR)
    b,_=F.posterior([attempt(0,.5),attempt(0,.5,keys=4)],1200,7,PRIOR)
    assert np.array_equal(a,b)
    repeated=[attempt(0,.5,chart='same') for i in range(6)]
    c,_=F.posterior(repeated,1200,7,PRIOR)
    assert np.allclose(a,c)
