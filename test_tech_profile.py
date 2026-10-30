"""Whole-chart Tech qualification leaves peak demand and its numbers alone."""
import pytest

import skill_calc as S


def chart(rows, keys=7):
    c = S.Chart()
    c.keys, c.od, c.sv = keys, 8., []
    c.notes = sorted((float(t), float(t), col) for t, cols in rows for col in cols)
    c.title = c.version = c.artist = c.creator = ''
    return c


@pytest.mark.parametrize('rate', [.83, 1., 1.24])
@pytest.mark.parametrize('offset', [1000, 61037])
def test_constant_global_pulse_with_skipped_hand_strokes_is_not_rhythm_tech(rate, offset):
    c = chart([(offset + 80*i, [(0, 4, 5, 1, 6)[i % 5]]) for i in range(400)])
    info = S.N.execution_analyse(c, rate)
    assert sum(v*n for v, n in zip(info['regular'], info['count'])) > 50
    assert S.N.tech_profile(c, rate)['rhythmtech'] == 0.


def test_irregular_five_gap_control_remains_technical():
    t, rows = 1000, []
    for i in range(400):
        t += (60, 93, 47, 111, 75)[i % 5]
        rows.append((t, [i % 4]))
    assert S.N.tech_profile(chart(rows, 4))['rhythmtech'] > .9


def test_regular_inserted_jacks_retain_pattern_evidence():
    rows = [(1000 + 40*i, [(0, 1), (4, 5), (4, 6), (1, 2), (5, 6)][i % 5])
            for i in range(400)]
    profile = S.N.tech_profile(chart(rows))
    assert profile['patterntech'] == 1.
    assert profile['technical'] == 1.
    assert profile['rhythmtech'] == 0.


def test_preexisting_tech_archetype_obeys_same_profile_qualification():
    sc = {k: 0. for k in S.skills(7)}
    sc.update(overall=10., delay=9., stamina=8.8, technical=8.7)
    res = {'keys': 7, 'scores': sc,
           'archetypes': [{'name': 'Tech Delay', 'rating': 9.,
                           'parts': ('delay', 'technical'), 'of': 'delay'}]}
    legacy = S.card(res)
    assert legacy[0]['name'] == 'Tech Delay'
    res['tech_profile'] = {'technical': .3}
    qualified = S.card(res)
    assert qualified[0]['name'] == 'Delay'
    assert [e['rating'] for e in qualified] == [e['rating'] for e in legacy]
    assert qualified[0]['parts'] == ('delay',)
    res['tech_profile'] = {'technical': .8}
    assert S.card(res) == legacy


def test_unqualified_technical_only_archetype_does_not_consume_a_card_slot():
    sc = {k: 0. for k in S.skills(7)}
    sc.update(overall=10., ln=9., delay=8., stamina=7.8, technical=9.5, patterntech=9.)
    res = {'keys': 7, 'scores': sc, 'tech_profile': {'technical': .3},
           'archetypes': [{'name': 'Tech Control', 'rating': 9.5,
                           'parts': ('technical', 'patterntech')}]}
    before = dict(res['archetypes'][0])
    assert S.card(res)[0] == {'name': 'LN', 'rating': 9., 'parts': ('ln',)}
    assert all(e['name'] != 'Control' for e in S.card(res))
    assert res['archetypes'][0] == before


def test_existing_tech_description_can_be_requalified_and_legacy_is_preserved():
    assert S.N.technical_title('Tech LN', {'technical': .8}, {'technical': .3}) == 'LN'
    assert S.N.technical_title('Tech Delay', {'technical': .8}, None) == 'Tech Delay'
    assert S.N.technical_title('Delay', {'technical': .8}, {'technical': .8}) == 'Tech Delay'
