from types import SimpleNamespace
import archetype_names as A


def chart(rows,keys=4):
    return SimpleNamespace(keys=keys,notes=sorted((t,t,c) for t,cols in rows for c in cols))


def test_rice_has_no_ln_chordstream_profile():
    c=chart([(i*80,[0,1] if i%2 else [2,3]) for i in range(100)])
    p=A.profile(c)
    assert p['ln_chordstream']==0
    assert set(p)=={'version','ln_chordstream'}


def test_old_jackstream_evidence_does_not_rename_patterns():
    p={'jackstream':.7}
    sk={'stream':1.,'minijack':.8,'technical':.8}
    entries=[{'name':'Tech Stream','rating':8.,'parts':('stream','technical')},
             {'name':'Minijack','rating':7.,'parts':('minijack',)}]
    changed=A.card(entries,sk,p,4)
    assert changed==entries
    assert A.description('Tech Stream / Minijack',sk,p,4)=='Tech Stream / Minijack'


def test_ln_release_refinement_precedes_generic_tech_ln():
    import skill_calc as S
    scores=dict.fromkeys(S.skills(7),0.)
    scores.update(overall=10.,ln=10.,release=9.3,delay=9.5,technical=8.,hybrid=6.3,chordstream=7.2)
    result={'keys':7,'scores':scores,'archetype_profile':{'ln_chordstream':.4},
            'archetypes':[{'name':'Tech LN','rating':10.,'parts':('ln','technical'),'of':'ln'},
                          {'name':'LN Release','rating':9.3,'parts':('release','ln')}]}
    assert S.card(result)[0]['name']=='LN Release'


def test_local_holds_and_chords_qualify_but_separate_sections_do_not():
    notes=[(0,12000,0)]+[(i*80,i*80,c) for i in range(1,150) for c in (2,3)]
    p=A.profile(SimpleNamespace(keys=7,notes=sorted(notes)))
    sk={'chordstream':1.,'ln':.9,'hybrid':.85}
    assert p['ln_chordstream']>.9
    assert A.description('Chordstream / LN',sk,p,7)=='LN Chordstream'
    separate=[(0,1000,0)]+[(2000+i*80,2000+i*80,c) for i in range(150) for c in (2,3)]
    assert A.profile(SimpleNamespace(keys=7,notes=sorted(separate)))['ln_chordstream']==0


def test_compound_card_parts_are_valid_for_the_keymode_tooltip():
    import skill_calc as S
    for keys,texture in ((4,'handstream'),(7,'chordstream')):
        sk={texture:1.,'ln':.9,'hybrid':.85}
        desc=A.descriptor(sk,{'ln_chordstream':.8},keys)
        assert desc is not None
        names=S.skill_names(keys)
        assert all(p in names for p in desc['parts'])


def test_ln_tech_requires_the_original_prevalence_and_demand_without_promotion():
    import pattern_control as P
    import skill_calc as S
    profile={'technical':.43,'ln_technical':.8}
    assert P.technical_title('LN', {'technical':.75}, profile) == 'LN'
    assert P.technical_title('LN', {'technical':.55}, {'technical':.8}) == 'LN'
    assert P.technical_title('LN', {'technical':.75}, {'technical':.8}) == 'Tech LN'
    scores=dict.fromkeys(S.skills(7),0.)
    scores.update(overall=10.,delay=8.54,ln=8.27,release=7.18,technical=5.17,bracket=7.49)
    result={'keys':7,'scores':scores,'notes':100,'ln_notes':83,'tech_profile':profile}
    assert S.card(result)[0]['name']=='Delay'
