"""Shared keymode-specific taxonomy for practice and player Stats.

Parent selections mean the whole family. Selecting a child narrows that family;
separate selections remain a union. These are views of chart demand, not extra
independent strain charges or new fitted difficulty coefficients.
"""
import math
from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True)
class Skill:
    key: str
    name: str
    members: tuple
    children: tuple = ()
    tip: str = ""


def family(key, name, members, children=(), tip=""):
    return Skill(key, name, tuple(members), tuple(
        Skill(key+"/"+child, label, (child,), tip=description)
        for child, label, description in children), tip)


LN = family("ln", "LN", ("ln", "release", "inverse", "hybrid", "shield"), (
    ("release", "LN Release", "Timing and coordinating hold endings."),
    ("inverse", "Inverse", "Release followed by a re-press of the held column."),
    ("hybrid", "Hybrid LN", "Independent taps while other fingers hold."),
    ("shield", "Shields", "A tap immediately followed by a hold in the same column.")))
TECH = family("technical", "Technical", ("technical", "patterntech", "rhythmtech"), (
    ("patterntech", "Pattern Tech", "Awkward finger control and genuinely difficult transitions."),
    ("rhythmtech", "Rhythm Tech", "Irregular execution timing, not merely changing global note gaps.")))
SV = family("sv", "SV", ("sv",), (
    ("sv_fast", "Fast SV", "Shortened readable approach time."),
    ("sv_slow", "Slowjam SV", "Compressed spacing and crowded notes at low scroll speed."),
    ("sv_accel", "Accelerations", "Increasing scroll speed during note approach."),
    ("sv_stutter", "Stutters", "Repeated meaningful speed-direction changes."),
    ("sv_brake", "Brakes", "Decreasing approach speed and timing uncertainty near judgement.")))
STAMINA = family("stamina", "Stamina", ("stamina",), tip="Sustained effort, not simply a long song.")
JACKS = family("jackspeed", "Jackspeed", ("jackspeed", "jack", "minijack", "longjack", "vibro"), (
    ("minijack", "Minijacks", "Short repeated-column bursts inside other patterns."),
    ("longjack", "Longjacks", "Sustained single-column repetition."),
    ("vibro", "Vibro", "Specialist ultra-fast repetition. Only explicitly selecting Vibro includes pure vibro drills.")),
    "Single-column repetition and short jack bursts; distinct from changing overlapping chords.")
CHORDJACK = family("chordjack", "Chordjack", ("chordjack",), tip="Repeated or partly overlapping chords.")
JUMPTRILL = family("jumptrill", "Jumptrill", ("jumptrill",),
    tip="Alternating hand impulses, including timing-feasible rolls. Separate from chordstream/jumpstream.")
SPLITTRILL = family("splittrill", "Split Trill", ("splittrill",),
    tip="Both hands alternate fingers independently; not ordinary Jumptrill. The wide-key counterpart is Bracket.")

FOUR = (
    family("stream", "Stream", ("stream", "dump", "trill1h", "anchor"), (
        ("dump", "Dump", "Very fast rice that still needs individual finger control."),
        ("trill1h", "1H Trill", "Alternation between fingers on one hand."),
        ("anchor", "Anchors", "A repeated column embedded in flowing notes."))),
    family("jumpstream", "Jumpstream", ("jumpstream",)),
    JUMPTRILL, SPLITTRILL,
    family("handstream", "Handstream", ("handstream", "quadstream"), (
        ("quadstream", "Quadstream", "Single → quad → single stream texture with embedded minijacks; not every quad or chordjack."),)),
    JACKS, CHORDJACK, LN, TECH, STAMINA, SV,
)
WIDE = (
    family("delay", "Delay", ("delay", "stream", "trill1h", "anchor"), (
        ("stream", "Stream", "Slower flowing single-note rice."),
        ("trill1h", "1H Trill", "One-hand finger alternation, not an alternating-hand Jumptrill."),
        ("anchor", "Anchors", "Repeated columns inside rice."))),
    family("chordstream", "Chordstream", ("chordstream",)),
    JUMPTRILL,
    family("bracket", "Bracket", ("bracket",), tip="A hand alternating between disjoint finger groups, with a chord in at least one group."),
    CHORDJACK, JACKS, LN, TECH, STAMINA, SV,
)


def tree(keys):
    return FOUR if keys == 4 else WIDE


@lru_cache(maxsize=8)
def dimensions(keys):
    return {node.key: node for parent in tree(keys) for node in (parent, *parent.children)}


_ALL = {node.key: node for keys in (4, 7) for node in dimensions(keys).values()}
NAMES = {key: node.name for key, node in _ALL.items()}
MEMBERS = {key: node.members for key, node in _ALL.items()}
TIPS = {key: node.tip for key, node in _ALL.items()}
LEGACY = {
    "rice": ("stream", "delay", "jumpstream", "handstream"),
    "chords": ("jumpstream", "handstream", "chordstream", "bracket"),
    "jack": ("jackspeed",),
    "trill": ("stream/trill1h", "delay/trill1h", "jumptrill", "splittrill", "bracket"),
    "jumpstream/jumptrill": ("jumptrill",),
    "chordstream/jumptrill": ("jumptrill",),
    "jumpstream/splittrill": ("splittrill",),
}


def normalize(selected=None):
    """Canonical signed choices; ! means excluded. Opposite child overrides survive.

    Existing positive-only preferences migrate without losing unavailable-mode
    selections. No new preference store or separate blacklist is needed.
    """
    if isinstance(selected, str):
        selected = [selected]
    if selected is not None and not isinstance(selected, (list, tuple, set)):
        return ()
    return _normalize(tuple(token for token in selected or () if isinstance(token, str)))


@lru_cache(maxsize=512)
def _normalize(selected):
    states = {}
    for token in selected:
        negative = token.startswith("!")
        key = token[1:] if negative else token
        for k in LEGACY.get(key, (key,)):
            if k in NAMES:
                # An explicit exclusion wins conflicting restored preferences.
                states[k] = min(states.get(k, 1), -1 if negative else 1)
    # Canonicalise inherited exceptions the same way as a live child click.
    # Keeping !parent + child would display an included child while the match
    # filter still rejected its whole family. Expand those states once instead.
    for parent, node in _ALL.items():
        inherited = states.get(parent)
        if inherited and node.children and any(states.get(c.key) == -inherited for c in node.children):
            for child in node.children:
                states.setdefault(child.key, inherited)
            del states[parent]
    return tuple(("!" if states[k] < 0 else "") + k for k in NAMES if k in states
                 and ("/" not in k or states.get(k.split("/")[0]) != states[k]))


def state(selected, key):
    values = normalize(selected)
    if "!" + key in values:
        return -1
    if key in values:
        return 1
    if "/" in key:
        parent = key.split("/")[0]
        return -1 if "!" + parent in values else 1 if parent in values else 0
    return 0


def mixed(selected, key):
    node = _ALL.get(key)
    if not node or not node.children:
        return False
    states = {state(selected, child.key) for child in node.children}
    return len(states) > 1 or states != {state(selected, key)}


def set_state(selected, key, new):
    """Parent operates on the whole family; a child can override or narrow it."""
    if key not in NAMES or new not in (-1, 0, 1):
        return normalize(selected)
    values = set(normalize(selected))
    if "/" in key:
        parent = key.split("/")[0]
        # Expand inherited state before editing a child. Neutral really means
        # untargeted, not silently re-included by its selected parent.
        inherited = state(values, parent)
        if inherited:
            for child in _ALL[parent].children:
                s = state(values, child.key)
                if s:
                    values.add(("!" if s < 0 else "") + child.key)
            values.discard(parent); values.discard("!" + parent)
    else:
        values = {v for v in values if not v.lstrip("!").startswith(key + "/")}
    values.discard(key); values.discard("!" + key)
    if new:
        values.add(("!" if new < 0 else "") + key)
    return normalize(values)


def cycle(selected, key):
    current = 0 if mixed(selected, key) else state(selected, key)
    return set_state(selected, key, {0: 1, 1: -1, -1: 0}[current])


def toggle(selected, key, active):
    """Compatibility for programmatic callers; user clicks use cycle()."""
    return set_state(selected, key, 1 if active else 0)


def label(selected):
    keys = normalize(selected)
    positive = [NAMES[k] for k in keys if not k.startswith("!")]
    negative = [NAMES[k[1:]] for k in keys if k.startswith("!")]
    text = " + ".join(dict.fromkeys(positive)) or "Any"
    return text + (" · exclude " + ", ".join(dict.fromkeys(negative)) if negative else "")


def strengths(f):
    return dict(_strengths(f.get('keys', 7), tuple(f.get('sk', {}).items())))


@lru_cache(maxsize=32768)
def _strengths(keys, skills):
    sk = dict(skills)
    return tuple((key, max((max(0., sk.get(k, 0.)) for k in node.members), default=0.))
                 for key, node in dimensions(keys).items())


def practice_strengths(f, values=None):
    values = strengths(f) if values is None else dict(values)
    profile = f.get('tech_profile') or {}
    # Technical practice needs actual technical character, not just an awkward
    # local peak in an otherwise smooth chart. Rare smaller but still material
    # matches remain eligible and receive less sampling weight.
    for key in ('technical', 'technical/patterntech', 'technical/rhythmtech'):
        leaf = key.split('/')[-1]
        if leaf in profile and key in values:
            values[key] = min(values[key], math.sqrt(max(0.,profile[leaf])))
    return values


def match(f, selected=None):
    """Use hard-passage demand, not one incidental occurrence or a forced rotation."""
    return _match(f.get('keys', 7), tuple(f.get('sk', {}).items()),
                  tuple((f.get('tech_profile') or {}).items()), normalize(selected))


@lru_cache(maxsize=32768)
def _match(keys, skills, profile, selected):
    f = {'keys': keys, 'sk': dict(skills), 'tech_profile': dict(profile)}
    values = strengths(f)
    maximum = max(values.values(), default=0.)
    floor = max(.45, .60*maximum)
    # Tiny incidental occurrences remain allowed; a material excluded demand
    # rejects the chart even when a different requested skill also matches.
    excluded = [k[1:] for k in selected if k.startswith("!")]
    if any(values.get(k, 0.) >= floor for k in excluded):
        return 0., ()
    values = practice_strengths(f, values)
    positive = tuple(k for k in selected if not k.startswith("!"))
    requested = positive or tuple(parent.key for parent in tree(f.get("keys", 7)))
    matched = tuple(k for k in requested if values.get(k, 0.) >= floor)
    if not positive:
        return 1., matched
    return max((min(1., values[k]) for k in matched), default=0.), matched


def demanded(f):
    """Skills g for which match(f, [g]) matches, from one strengths pass (Stats tests every group per chart)."""
    return _demanded(f.get('keys', 7), tuple(f.get('sk', {}).items()), tuple((f.get('tech_profile') or {}).items()))


@lru_cache(maxsize=32768)
def _demanded(keys, skills, profile):
    f = {'keys': keys, 'sk': dict(skills), 'tech_profile': dict(profile)}
    values = strengths(f)
    floor = max(.45, .60*max(values.values(), default=0.))
    return frozenset(k for k, v in practice_strengths(f, values).items() if v >= floor)


def session_skill(key, keys):
    node = dimensions(keys).get(key)
    return node.members[0] if node else None


def wants_vibro(selected):
    return "jackspeed/vibro" in normalize(selected)


def value(f, e, rate, target, selected=None):
    import nps
    strength, matched = match(f, selected)
    if not strength:
        return None
    st = f.get("nps", {})
    length = st.get("play_span", f.get("length", 0.)/1000.)/rate
    # Prominent examples lead; weaker-but-material matches remain a smaller
    # source of variety instead of becoming an unrelated skill playlist.
    prominence = strength
    preference = 2.2*math.log(max(.01, min(1., prominence))) if selected else 0.
    return (1.1*math.log(max(.01, strength))+preference-14*abs(e["acc_mid"]-target)
            -.40*e["sd_model"]-nps.quality_penalty(length, st, rate)
            -.15*abs(math.log(rate)))


def describe_match(f, selected=None):
    _strength, matched = match(f, selected)
    values = strengths(f)
    strongest = sorted(matched, key=lambda k: -values[k])[:2]
    return " / ".join(NAMES[k] for k in strongest) or "Mixed skills"
