from types import SimpleNamespace
import skill_calc as S


def chart(rows,keys=4):
    return SimpleNamespace(keys=keys,notes=sorted((t,t,c) for t,cols in rows for c in cols))


def test_rice_has_no_ln_chordstream_profile():
    c=chart([(i*80,[0,1] if i%2 else [2,3]) for i in range(100)])
    assert S.N.archetype_profile(c)==0


# Archetype renaming rules: native/src/card.rs tests.


def test_ln_release_refinement_precedes_generic_tech_ln():
    scores=dict.fromkeys(S.skills(7),0.)
    scores.update(overall=10.,ln=10.,release=9.3,delay=9.5,technical=8.,hybrid=6.3,chordstream=7.2)
    result={'keys':7,'scores':scores,'archetype_profile':{'ln_chordstream':.4},
            'archetypes':[{'name':'Tech LN','rating':10.,'parts':('ln','technical'),'of':'ln'},
                          {'name':'LN Release','rating':9.3,'parts':('release','ln')}]}
    assert S.card(result)[0]['name']=='LN Release'


def test_local_holds_and_chords_qualify_but_separate_sections_do_not():
    notes=[(0,12000,0)]+[(i*80,i*80,c) for i in range(1,150) for c in (2,3)]
    assert S.N.archetype_profile(SimpleNamespace(keys=7,notes=sorted(notes)))>.9
    separate=[(0,1000,0)]+[(2000+i*80,2000+i*80,c) for i in range(150) for c in (2,3)]
    assert S.N.archetype_profile(SimpleNamespace(keys=7,notes=sorted(separate)))==0


def test_ln_tech_requires_the_original_prevalence_and_demand_without_promotion():
    profile={'technical':.43,'ln_technical':.8}
    assert S.N.technical_title('LN', {'technical':.75}, profile) == 'LN'
    assert S.N.technical_title('LN', {'technical':.55}, {'technical':.8}) == 'LN'
    assert S.N.technical_title('LN', {'technical':.75}, {'technical':.8}) == 'Tech LN'
    scores=dict.fromkeys(S.skills(7),0.)
    scores.update(overall=10.,delay=8.54,ln=8.27,release=7.18,technical=5.17,bracket=7.49)
    result={'keys':7,'scores':scores,'notes':100,'ln_notes':83,'tech_profile':profile}
    assert S.card(result)[0]['name']=='Delay'
