//! skill_practice: the keymode-specific Skills taxonomy. Parent selections mean the whole family; selecting a child
//! narrows it; separate selections remain a union. Views of chart demand, not extra strain charges or fitted
//! coefficients. A selection is a list of signed keys: "!" means excluded.
use std::collections::{BTreeSet, HashMap};
use std::sync::OnceLock;

#[derive(Clone, Debug)]
pub struct Skill {
    pub key: String,
    pub name: &'static str,
    pub members: Vec<&'static str>,
    pub children: Vec<Skill>,
    pub tip: &'static str,
}

fn family(key: &str, name: &'static str, members: &[&'static str], children: &[(&'static str, &'static str, &'static str)], tip: &'static str) -> Skill {
    Skill {
        key: key.to_string(), name, members: members.to_vec(), tip,
        children: children.iter().map(|(child, label, about)| Skill { key: format!("{key}/{child}"), name: label, members: vec![child], children: Vec::new(), tip: about }).collect(),
    }
}

fn shared() -> [Skill; 9] {
    [
        family("ln", "LN", &["ln", "release", "inverse", "hybrid", "shield"], &[
            ("release", "LN Release", "Timing and coordinating hold endings."),
            ("inverse", "Inverse", "Release followed by a re-press of the held column."),
            ("hybrid", "Hybrid LN", "Independent taps while other fingers hold."),
            ("shield", "Shields", "A tap immediately followed by a hold in the same column.")], ""),
        family("technical", "Technical", &["technical", "patterntech", "rhythmtech"], &[
            ("patterntech", "Pattern Tech", "Awkward finger control and genuinely difficult transitions."),
            ("rhythmtech", "Rhythm Tech", "Irregular execution timing, not merely changing global note gaps.")], ""),
        family("sv", "SV", &["sv"], &[
            ("sv_fast", "Fast SV", "Shortened readable approach time."),
            ("sv_slow", "Slowjam SV", "Compressed spacing and crowded notes at low scroll speed."),
            ("sv_accel", "Accelerations", "Increasing scroll speed during note approach."),
            ("sv_stutter", "Stutters", "Repeated meaningful speed-direction changes."),
            ("sv_brake", "Brakes", "Decreasing approach speed and timing uncertainty near judgement.")], ""),
        family("stamina", "Stamina", &["stamina"], &[], "Sustained effort, not simply a long song."),
        family("jackspeed", "Jackspeed", &["jackspeed", "jack", "minijack", "longjack", "vibro"], &[
            ("minijack", "Minijacks", "Short repeated-column bursts inside other patterns."),
            ("longjack", "Longjacks", "Sustained single-column repetition."),
            ("vibro", "Vibro", "Specialist ultra-fast repetition. Only explicitly selecting Vibro includes pure vibro drills.")],
            "Single-column repetition and short jack bursts; distinct from changing overlapping chords."),
        family("chordjack", "Chordjack", &["chordjack"], &[], "Repeated or partly overlapping chords."),
        family("jumptrill", "Jumptrill", &["jumptrill"], &[], "Alternating hand impulses, including timing-feasible rolls. Separate from chordstream/jumpstream."),
        family("splittrill", "Split Trill", &["splittrill"], &[], "Both hands alternate fingers independently; not ordinary Jumptrill. The wide-key counterpart is Bracket."),
        family("bracket", "Bracket", &["bracket"], &[], "A hand alternating between disjoint finger groups, with a chord in at least one group."),
    ]
}

struct Trees {
    four: Vec<Skill>,
    wide: Vec<Skill>,
    all: Vec<Skill>, // every node of both trees, 4K's order first, each key once
}

fn trees() -> &'static Trees {
    static T: OnceLock<Trees> = OnceLock::new();
    T.get_or_init(|| {
        let [ln, tech, sv, stamina, jacks, chordjack, jumptrill, splittrill, bracket] = shared();
        let four = vec![
            family("stream", "Stream", &["stream", "dump", "trill1h", "anchor"], &[
                ("dump", "Dump", "Very fast rice that still needs individual finger control."),
                ("trill1h", "1H Trill", "Alternation between fingers on one hand."),
                ("anchor", "Anchors", "A repeated column embedded in flowing notes.")], ""),
            family("jumpstream", "Jumpstream", &["jumpstream"], &[], ""),
            jumptrill.clone(), splittrill,
            family("handstream", "Handstream", &["handstream", "quadstream"], &[
                ("quadstream", "Quadstream", "Single → quad → single stream texture with embedded minijacks; not every quad or chordjack.")], ""),
            jacks.clone(), chordjack.clone(), ln.clone(), tech.clone(), stamina.clone(), sv.clone(),
        ];
        let wide = vec![
            family("delay", "Delay", &["delay", "stream", "trill1h", "anchor"], &[
                ("stream", "Stream", "Slower flowing single-note rice."),
                ("trill1h", "1H Trill", "One-hand finger alternation, not an alternating-hand Jumptrill."),
                ("anchor", "Anchors", "Repeated columns inside rice.")], ""),
            family("chordstream", "Chordstream", &["chordstream"], &[], ""),
            jumptrill, bracket, chordjack, jacks, ln, tech, stamina, sv,
        ];
        let mut all: Vec<Skill> = Vec::new();
        for tree in [&four, &wide] {
            for parent in tree {
                for node in std::iter::once(parent).chain(&parent.children) {
                    if !all.iter().any(|n| n.key == node.key) {
                        all.push(node.clone());
                    }
                }
            }
        }
        Trees { four, wide, all }
    })
}

pub fn tree(keys: i64) -> &'static [Skill] {
    if keys == 4 { &trees().four } else { &trees().wide }
}

/// The tree's nodes, parents before their children.
pub fn dimensions(keys: i64) -> Vec<&'static Skill> {
    tree(keys).iter().flat_map(|p| std::iter::once(p).chain(&p.children)).collect()
}

pub fn node(key: &str) -> Option<&'static Skill> {
    trees().all.iter().find(|n| n.key == key)
}

pub fn name(key: &str) -> &'static str {
    node(key).map_or("", |n| n.name)
}

/// skill_practice.strengths and practice_strengths of one chart-rate: (key, raw, practice) per dimension.
fn strengths(f: &ms_calc::val::Dict) -> Vec<(&'static str, f64, f64)> {
    let keys = f.get("keys").map_or(7, |k| k.f() as i64);
    let (sk, profile) = (f.dict("sk"), f.dict("tech_profile"));
    dimensions(keys).into_iter().map(|node| {
        let raw = node.members.iter().filter_map(|m| sk.and_then(|d| d.get(m))).fold(0f64, |a, v| a.max(v.f().max(0.)));
        // Technical practice needs actual technical character, not just an awkward local peak in a smooth chart.
        let leaf = node.key.rsplit('/').next().unwrap_or("");
        let technical = matches!(node.key.as_str(), "technical" | "technical/patterntech" | "technical/rhythmtech");
        let practice = profile.and_then(|p| p.get(leaf)).filter(|_| technical).map_or(raw, |v| raw.min(v.f().max(0.).sqrt()));
        (node.key.as_str(), raw, practice)
    }).collect()
}

fn floor(values: &[(&'static str, f64, f64)]) -> f64 {
    0.45f64.max(0.60 * values.iter().map(|v| v.1).fold(0., f64::max))
}

/// Skills g for which match(f, [g]) matches, from one strengths pass.
pub fn demanded(f: &ms_calc::val::Dict) -> Vec<&'static str> {
    let values = strengths(f);
    let bar = floor(&values);
    values.into_iter().filter(|v| v.2 >= bar).map(|v| v.0).collect()
}

/// match(f)[1]: the top-level skills a chart demands.
pub fn matched_families(f: &ms_calc::val::Dict) -> Vec<&'static str> {
    demanded(f).into_iter().filter(|k| !k.contains('/')).collect()
}

/// session_skill: the session form skill of a Stats group.
pub fn session_skill(key: &str, keys: i64) -> Option<&'static str> {
    dimensions(keys).into_iter().find(|n| n.key == key).and_then(|n| n.members.first().copied())
}

/// dimensions() of 4K and of wider keymodes as [(key, members)], for FeatStore's match.
pub fn feat_trees() -> Vec<Vec<(String, Vec<String>)>> {
    [4, 7].iter().map(|&k| dimensions(k).iter().map(|n| (n.key.clone(), n.members.iter().map(|m| m.to_string()).collect())).collect()).collect()
}

const LEGACY: [(&str, &[&str]); 7] = [
    ("rice", &["stream", "delay", "jumpstream", "handstream"]),
    ("chords", &["jumpstream", "handstream", "chordstream", "bracket"]),
    ("jack", &["jackspeed"]),
    ("trill", &["stream/trill1h", "delay/trill1h", "jumptrill", "splittrill", "bracket"]),
    ("jumpstream/jumptrill", &["jumptrill"]),
    ("chordstream/jumptrill", &["jumptrill"]),
    ("jumpstream/splittrill", &["splittrill"]),
];

/// Canonical signed choices. Opposite child overrides survive; positive-only preferences migrate.
pub fn normalize(selected: &[String]) -> Vec<String> {
    let mut states: HashMap<String, i8> = HashMap::new();
    for token in selected {
        let (negative, key) = token.strip_prefix('!').map_or((false, token.as_str()), |k| (true, k));
        let names: Vec<&str> = LEGACY.iter().find(|l| l.0 == key).map_or_else(|| vec![key], |l| l.1.to_vec());
        for k in names {
            if node(k).is_some() {
                // An explicit exclusion wins conflicting restored preferences.
                let at = states.entry(k.to_string()).or_insert(1);
                *at = (*at).min(if negative { -1 } else { 1 });
            }
        }
    }
    // Inherited exceptions the same way as a live child click: keeping !parent + child would display an included
    // child while the match filter still rejected its whole family.
    for n in &trees().all {
        if let Some(&inherited) = states.get(&n.key) {
            if !n.children.is_empty() && n.children.iter().any(|c| states.get(&c.key) == Some(&-inherited)) {
                for child in &n.children {
                    states.entry(child.key.clone()).or_insert(inherited);
                }
                states.remove(&n.key);
            }
        }
    }
    trees().all.iter().filter(|n| {
        states.get(&n.key).is_some_and(|s| !n.key.contains('/') || states.get(n.key.split('/').next().unwrap()) != Some(s))
    }).map(|n| format!("{}{}", if states[&n.key] < 0 { "!" } else { "" }, n.key)).collect()
}

pub fn state(selected: &[String], key: &str) -> i8 {
    let values = normalize(selected);
    let has = |k: String| values.contains(&k);
    if has(format!("!{key}")) {
        return -1;
    }
    if has(key.to_string()) {
        return 1;
    }
    if let Some((parent, _)) = key.split_once('/') {
        return if has(format!("!{parent}")) { -1 } else if has(parent.to_string()) { 1 } else { 0 };
    }
    0
}

pub fn mixed(selected: &[String], key: &str) -> bool {
    let Some(n) = node(key).filter(|n| !n.children.is_empty()) else { return false };
    let states: BTreeSet<i8> = n.children.iter().map(|c| state(selected, &c.key)).collect();
    states.len() > 1 || !states.contains(&state(selected, key))
}

/// Parent operates on the whole family; a child can override or narrow it.
pub fn set_state(selected: &[String], key: &str, new: i8) -> Vec<String> {
    if node(key).is_none() || !(-1..=1).contains(&new) {
        return normalize(selected);
    }
    let mut values: BTreeSet<String> = normalize(selected).into_iter().collect();
    let as_list = |v: &BTreeSet<String>| v.iter().cloned().collect::<Vec<_>>();
    if let Some((parent, _)) = key.split_once('/') {
        // Expand inherited state before editing a child: neutral really means untargeted, not re-included by its parent.
        if state(&as_list(&values), parent) != 0 {
            for child in &node(parent).unwrap().children {
                let s = state(&as_list(&values), &child.key);
                if s != 0 {
                    values.insert(format!("{}{}", if s < 0 { "!" } else { "" }, child.key));
                }
            }
            values.remove(parent);
            values.remove(&format!("!{parent}"));
        }
    } else {
        let prefix = format!("{key}/");
        values.retain(|v| !v.trim_start_matches('!').starts_with(&prefix));
    }
    values.remove(key);
    values.remove(&format!("!{key}"));
    if new != 0 {
        values.insert(format!("{}{key}", if new < 0 { "!" } else { "" }));
    }
    normalize(&as_list(&values))
}

/// none → ✓ → ✕ → none
pub fn cycle(selected: &[String], key: &str) -> Vec<String> {
    let current = if mixed(selected, key) { 0 } else { state(selected, key) };
    set_state(selected, key, match current { 0 => 1, 1 => -1, _ => 0 })
}

pub fn label(selected: &[String]) -> String {
    let keys = normalize(selected);
    let mut positive: Vec<&str> = keys.iter().filter(|k| !k.starts_with('!')).map(|k| name(k)).collect();
    let mut negative: Vec<&str> = keys.iter().filter_map(|k| k.strip_prefix('!')).map(name).collect();
    let once = |v: &mut Vec<&str>| {
        let mut seen = BTreeSet::new();
        v.retain(|x| seen.insert(*x));
    };
    once(&mut positive);
    once(&mut negative);
    let text = if positive.is_empty() { "Any".to_string() } else { positive.join(" + ") };
    if negative.is_empty() { text } else { format!("{text} · exclude {}", negative.join(", ")) }
}

/// What an avoided skill may reach, as a share of a map's strongest skill.
pub const AVOID: [(f64, &str); 3] = [(0.9, "Led by it"), (0.6, "A lot of it"), (0.3, "Any of it")];

/// The sliders that change which maps are in the list.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Tuning {
    pub avoid: HashMap<String, f64>, // avoided skill: see AVOID
    pub other: f64,                  // slider of maps with no included skill; 0 keeps only included skills
}

impl Tuning {
    pub fn hold(&self, key: &str) -> f64 {
        self.avoid.get(key).copied().unwrap_or(0.6)
    }
}

/// {included skill: how-often slider, "": other maps} once something differs, else None (plain best-first list).
pub fn shares(selected: &[String], weights: &HashMap<String, f64>, other: f64) -> Option<HashMap<String, f64>> {
    let mut out: HashMap<String, f64> = normalize(selected).into_iter().filter(|k| !k.starts_with('!'))
        .map(|k| { let w = weights.get(&k).copied().unwrap_or(1.); (k, w) }).collect();
    if out.is_empty() {
        return None; // Any: nothing to weigh against
    }
    let first = out.values().next().copied();
    let differ = out.values().any(|v| Some(*v) != first);
    if other > 0. {
        out.insert(String::new(), other);
    }
    (differ || other > 0.).then_some(out)
}

pub fn wants_vibro(selected: &[String]) -> bool {
    normalize(selected).iter().any(|k| k == "jackspeed/vibro")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(x: &[&str]) -> Vec<String> {
        x.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn a_signed_cycle_covers_the_children_and_their_exceptions() {
        let sv = node("sv").unwrap();
        let on = cycle(&[], "sv");
        let off = cycle(&on, "sv");
        assert_eq!((on.clone(), off.clone()), (v(&["sv"]), v(&["!sv"])));
        for child in &sv.children {
            assert_eq!((state(&on, &child.key), state(&off, &child.key)), (1, -1));
        }
        assert!(cycle(&off, "sv").is_empty());
        let excepted = cycle(&v(&["ln"]), "ln/inverse");
        assert_eq!((state(&excepted, "ln/inverse"), state(&excepted, "ln/release")), (-1, 1));
        assert!(mixed(&excepted, "ln"));
        let neutral = cycle(&excepted, "ln/inverse");
        assert_eq!((state(&neutral, "ln/inverse"), state(&neutral, "ln/release")), (0, 1));
        assert_eq!(cycle(&neutral, "ln"), v(&["ln"]));
        assert!(label(&v(&["!sv"])).contains("exclude SV"));
    }

    #[test]
    fn cycling_a_parent_and_a_child() {
        let one = cycle(&[], "ln");
        assert_eq!(one, v(&["ln"]));
        assert_eq!(cycle(&one, "ln"), v(&["!ln"]));
        assert!(cycle(&v(&["!ln"]), "ln").is_empty());
        // Python's result: a child cycled out of an included parent expands the parent, the child neutral.
        assert_eq!(cycle(&cycle(&one, "ln/release"), "ln/release"), v(&["ln/inverse", "ln/hybrid", "ln/shield"]));
    }

    #[test]
    fn legacy_and_conflicts() {
        assert_eq!(normalize(&v(&["jack"])), v(&["jackspeed"]));
        assert_eq!(normalize(&v(&["ln", "!ln"])), v(&["!ln"]));
        let n = normalize(&v(&["!ln", "ln/release"]));
        assert!(n.contains(&"ln/release".to_string()) && n.contains(&"!ln/inverse".to_string()) && !n.contains(&"!ln".to_string()));
    }

    #[test]
    fn labels() {
        assert_eq!(label(&[]), "Any");
        assert_eq!(label(&v(&["ln", "!chordjack"])), "LN · exclude Chordjack");
    }
}
