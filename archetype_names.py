"""Readable names for co-occurring chart mechanics; no added difficulty cost.

This recognises a small set of useful combinations from actual local geometry.
It does not combine unrelated whole-map maxima or use chart titles as features.
"""
import collections
import math

VERSION=3


def profile(chart,rate=1.):
    if not math.isfinite(rate) or rate<=0:raise ValueError('invalid rate')
    notes=chart.notes
    if not notes:return {'version':VERSION,'ln_chordstream':0.}
    rows=[]
    for t,e,c in notes:
        if rows and t-rows[-1][0]<=3.:
            rows[-1][1].add(c);rows[-1][2].append((c,e))
        else:rows.append([t,{c},[(c,e)]])
    bins=collections.defaultdict(lambda:[0.,0.])
    held={};origin=rows[0][0]
    for t,cols,ends in rows:
        bucket=int((t-origin)/1000/rate)
        b=bins[bucket];n=len(cols);b[0]+=n
        taps=sum(e-t<=3 for c,e in ends)
        locked=any(e>t+3 for c,e in held.items() if c not in cols)
        if locked and n>=2 and taps:
            b[1]+=n
        for c,e in ends:
            if e>t+3:held[c]=e
    # Neighbouring local evidence creates one mixed texture, not disconnected
    # skill islands. Note weighting prevents long empty rests diluting it.
    total=0.;den=sum(b[0] for b in bins.values())
    for index,b in bins.items():
        neighbourhood=[bins.get(j,[0.,0.]) for j in (index-1,index,index+1)]
        mass=sum(v[0] for v in neighbourhood)
        total+=b[0]*sum(v[1] for v in neighbourhood)/max(1.,mass)
    return {'version':VERSION,'ln_chordstream':round(total/max(1.,den),6)}


def descriptor(sk,profile,keys):
    """A compound replaces its primary texture, never adds a new numeric skill."""
    if not profile:return None
    flow=('stream','dump','jumpstream','handstream') if keys==4 else ('stream','delay','chordstream','bracket')
    primary=max(flow,key=lambda k:sk.get(k,0.))
    top=max(sk.values(),default=0.)
    if top<=0 or sk.get(primary,0.)<.6*top:return None
    if profile.get('ln_chordstream',0.)>=.25 and sk.get('hybrid',0.)>=.6*top \
            and sk.get('ln',0.)>=.65*top and max(sk.get('chordstream',0.),sk.get('handstream',0.))>=.65*top:
        return {'name':'LN Chordstream','parts':('ln','hybrid','handstream' if keys==4 else 'chordstream')}
    return None


NAMES={'stream':'Stream','dump':'Dump','jumpstream':'Jumpstream','handstream':'Handstream',
       'delay':'Delay','chordstream':'Chordstream','bracket':'Bracket','minijack':'Minijack',
       'anchor':'Anchor','ln':'LN','hybrid':'Hybrid LN','release':'LN Release'}


def rename(name,descriptor):
    if not descriptor:return name
    prefix='Tech ' if name.startswith('Tech ') else ''
    text=name[len(prefix):]
    suffix=' Stamina' if text.endswith(' Stamina') else ''
    plain=text[:-len(suffix)] if suffix else text
    if plain not in {NAMES.get(k,k) for k in descriptor['parts']}:return name
    return prefix+descriptor['name']+suffix


def card(entries,sk,profile,keys):
    desc=descriptor(sk,profile,keys)
    if not entries or not desc:return entries
    name=rename(entries[0]['name'],desc)
    if name==entries[0]['name']:return entries
    first=dict(entries[0],name=name,parts=tuple(dict.fromkeys((*entries[0]['parts'],*desc['parts']))))
    rest=[e for e in entries[1:] if not set(e['parts']).intersection(desc['parts'])]
    return [first,*rest]


def description(text,sk,profile,keys):
    desc=descriptor(sk,profile,keys)
    entries=text.split(' / ')
    if not entries or not desc:return text
    changed=rename(entries[0],desc)
    if changed==entries[0]:return text
    return ' / '.join([changed,*[name for name in entries[1:] if rename(name,desc)==name]])
