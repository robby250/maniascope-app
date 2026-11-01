//! The NPS/Skills choice list from the Builder's scoring to the shown list: one record per chart (its best exact
//! rate) and the worker's prepare → weighting → shuffle → web mix over all of them.
use crate::predict::{HashMap, Prediction, MAX_RATE, MIN_RATE};
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::sync::Arc;

/// What the worker's ranking and the shown list read of one chart's choice.
#[derive(Clone)]
pub struct Choice {
    pub sha: String,
    pub md5: String, // "" when the chart has none
    pub keys: i64,
    pub rate: f64,
    pub value: f64, // base_value
    pub installed: bool,
    pub convert: bool, // a keymode convert, imported or not
    pub playcount: f64,
    pub family: String,
    pub description: String,
    pub group: String, // nps.group
    pub nps: f64,
    pub length: f64, // play span at this rate, s
    pub profile: Vec<f64>,
    pub acc_mid: f64,
    pub matched: Vec<String>, // included Skills the chart matches, for the how-often sliders
    pub practice: String,     // skill_practice.describe_match, Skills lists only
    pub pred: Prediction,
    pub generated_at: f64,
}

pub const REAL: u8 = 0;
pub const WEB: u8 = 1;
pub const CONVERT: u8 = 2;

impl Choice {
    /// Which how-often slider the chart answers to. An imported convert is still a convert.
    pub fn stream(&self) -> u8 {
        if self.convert { CONVERT } else if self.installed { REAL } else { WEB }
    }

    /// recommend.event_key
    pub fn identity(&self) -> String {
        if self.md5.is_empty() { format!("sha:{}", self.sha) } else { format!("md5:{}", self.md5) }
    }
}

/// A rate proposal: (quality, rate, (keys, group)).
#[derive(Clone)]
pub struct Proposal {
    pub quality: f64,
    pub rate: f64,
    pub keys: i64,
    pub group: String,
}

/// One request's scored charts (the Builder's memo across its passes): per catalogue chart, unscored or
/// (choice, proposal). The Builder drops a chart whose analyses change.
pub struct Choices {
    order: Vec<String>,
    at: HashMap<String, usize>,
    pub slots: Vec<Option<(Option<Arc<Choice>>, Option<Proposal>)>>,
    forecast: Arc<Forecast>,
}

/// accuracy_forecast provenance of a request: calculator, predictor, BAND_Z.
#[derive(Clone, Default)]
pub struct Forecast {
    pub calc: String,
    pub predictor: String,
    pub band_z: f64,
}

impl Choices {
    pub fn slot(&self, sha: &str) -> Option<usize> {
        self.at.get(sha).copied()
    }

    /// The catalogue's charts in order, and the request's accuracy_forecast provenance.
    pub fn new(order: Vec<String>, forecast: Forecast) -> Self {
        let at = order.iter().enumerate().map(|(i, s)| (s.clone(), i)).collect();
        let slots = vec![None; order.len()];
        Choices { order, at, slots, forecast: Arc::new(forecast) }
    }

    /// Charts not scored yet, in catalogue order.
    pub fn unscored(&self) -> Vec<String> {
        self.order.iter().zip(&self.slots).filter(|(_, s)| s.is_none()).map(|(o, _)| o.clone()).collect()
    }

    /// A chart whose analyses changed is scored again.
    pub fn drop_chart(&mut self, sha: &str) {
        if let Some(i) = self.slot(sha) {
            self.slots[i] = None;
        }
    }

    /// candidates_from's frontier: the proposals not refined or blocked, the best 24 overall and 4 of every
    /// (keys, group) (retention, not quotas: the eligible population keeps its proportions) → [(quality, sha,
    /// rate)], best first.
    pub fn proposals(&self, refined: &[(String, f64)], blocked: &HashSet<String>) -> Vec<(f64, String, f64)> {
        let refined: HashSet<(&str, u64)> = refined.iter().map(|(s, r)| (s.as_str(), r.to_bits())).collect();
        let mut all: Vec<(f64, &str, f64, &Proposal)> = Vec::new();
        for (sha, slot) in self.order.iter().zip(&self.slots) {
            if let Some((_, Some(p))) = slot {
                if !blocked.contains(sha) && !refined.contains(&(sha.as_str(), p.rate.to_bits())) {
                    all.push((p.quality, sha, p.rate, p));
                }
            }
        }
        // heapq.nlargest's tuple order: quality, then sha, rate and group.
        all.sort_by(|a, b| b.0.total_cmp(&a.0).then(b.1.cmp(a.1)).then(b.2.total_cmp(&a.2))
            .then((b.3.keys, &b.3.group).cmp(&(a.3.keys, &a.3.group))));
        let mut kept: Vec<(f64, &str, f64)> = all.iter().take(24).map(|x| (x.0, x.1, x.2)).collect();
        let mut per: HashMap<(i64, &str), usize> = HashMap::default();
        for x in &all {
            let n = per.entry((x.3.keys, x.3.group.as_str())).or_insert(0);
            if *n < 4 {
                *n += 1;
                kept.push((x.0, x.1, x.2));
            }
        }
        kept.sort_by(|a, b| b.0.total_cmp(&a.0).then(b.1.cmp(a.1)).then(b.2.total_cmp(&a.2)));
        kept.dedup_by(|a, b| a.1 == b.1 && a.2 == b.2);
        kept.into_iter().map(|(q, s, r)| (q, s.to_string(), r)).collect()
    }

    /// The choices as they stand, in catalogue order (what a publication carries; later passes don't touch it).
    pub fn snapshot(&self) -> ChoiceList {
        ChoiceList {
            items: self.slots.iter().filter_map(|s| s.as_ref().and_then(|(c, _)| c.clone())).collect(),
            forecast: self.forecast.clone(),
        }
    }
}

/// One publication's choices (immutable): what the worker prepares, ranks and shows.
#[derive(Clone, Default)]
pub struct ChoiceList {
    pub items: Vec<Arc<Choice>>,
    pub forecast: Arc<Forecast>,
}

/// Newer exact features than the Builder had (Worker.t_nps_update).
pub struct Metadata {
    pub family: String,
    pub group: String,
    pub nps: f64,
    pub length: f64,
    pub profile: Vec<f64>,
    pub description: String,
    pub matched: Vec<String>,
    pub practice: String,
}

impl ChoiceList {
    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// The `n` charts of the highest base_value: what the next request re-predicts first.
    pub fn lead(&self, n: usize) -> Vec<String> {
        let mut order: Vec<&Arc<Choice>> = self.items.iter().collect();
        order.sort_by(|a, b| b.value.total_cmp(&a.value));
        order.into_iter().take(n).map(|c| c.sha.clone()).collect()
    }

    /// (sha, rate) of every choice.
    pub fn refs(&self) -> Vec<(String, f64)> {
        self.items.iter().map(|c| (c.sha.clone(), c.rate)).collect()
    }

    /// The restart cache's references: the 512 best by base_value and 2048 spread by sha256(sha), sorted.
    pub fn restart_refs(&self) -> Vec<(String, f64)> {
        let mut best: Vec<&Arc<Choice>> = self.items.iter().collect();
        best.sort_by(|a, b| b.value.total_cmp(&a.value));
        let mut spread: Vec<(Vec<u8>, &Arc<Choice>)> = self.items.iter().map(|c| (Sha256::digest(c.sha.as_bytes()).to_vec(), c)).collect();
        spread.sort_by(|a, b| a.0.cmp(&b.0));
        let mut refs: Vec<(String, f64)> = best.into_iter().take(512).chain(spread.into_iter().take(2048).map(|x| x.1))
            .map(|c| (c.sha.clone(), c.rate)).collect();
        refs.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.total_cmp(&b.1)));
        refs.dedup();
        refs
    }

    /// This list with refreshed metadata over the choices of {(event_key, rate as bits): metadata}.
    pub fn patched(&self, patches: &HashMap<(String, u64), Metadata>) -> ChoiceList {
        let mut items = self.items.clone();
        for c in items.iter_mut() {
            let Some(m) = patches.get(&(c.identity(), c.rate.to_bits())) else { continue };
            *c = Arc::new(Choice { family: m.family.clone(), group: m.group.clone(), nps: m.nps, length: m.length, profile: m.profile.clone(), description: m.description.clone(),
                                  matched: m.matched.clone(), practice: m.practice.clone(), ..(**c).clone() });
        }
        ChoiceList { items, forecast: self.forecast.clone() }
    }

    /// The worker's prepared pool: choices of `keys` at an offered rate whose chart is still here (a website
    /// chart, or installed and in `local_shas`).
    pub fn prepare(self: &Arc<Self>, keys: &[i64], local_shas: &dyn Fn(&str) -> bool) -> Prepared {
        let kept: Vec<usize> = self.items.iter().enumerate()
            .filter(|(_, c)| !(c.installed && !local_shas(&c.sha)) && keys.contains(&c.keys) && (MIN_RATE..=MAX_RATE).contains(&c.rate))
            .map(|x| x.0).collect();
        let by_identity = kept.iter().map(|&i| (self.items[i].identity(), i)).collect();
        Prepared { list: self.clone(), kept, by_identity }
    }
}

/// A ChoiceList prepared for one worker state (keymodes, local charts).
pub struct Prepared {
    pub list: Arc<ChoiceList>,
    kept: Vec<usize>,
    by_identity: HashMap<String, usize>,
}

/// The worker's ranking inputs besides the pool (Worker._pool).
pub struct Ranking<'a> {
    pub keys: Vec<i64>,
    pub nps: bool, // NPS (density floor, shuffle) or Skills
    pub web: f64,
    pub convert: f64, // odds of converts against your own maps, as web
    pub weighted: [bool; 2], // website, converts: the odds are per map, not per pick
    pub taste: &'a HashMap<(i64, String), f64>, // {(keys, group): preference}
    pub eventful: &'a dyn Fn(&str) -> bool,     // identities with session events: the others are fresh
    pub freshness: &'a dyn Fn(&str, f64, f64) -> f64, // (identity, length, pool) → Session.freshness
    pub recent_families: Vec<String>,
    pub recent_profiles: Vec<Vec<f64>>,
    pub current_keys: Option<i64>,
    pub consumed: &'a dyn Fn(&Choice) -> bool,
    pub shuffle: Option<(String, f64)>, // pp_playlist.preview's (str(seed), focus)
    pub count: Option<usize>,
    pub recent: Vec<u8>, // this tab's last offers, oldest first: REAL, WEB or CONVERT
    pub skill_weights: Option<std::collections::HashMap<String, f64>>, // {included skill: how often} when the sliders differ
}

impl Prepared {
    pub fn len(&self) -> usize {
        self.kept.len()
    }

    pub fn is_empty(&self) -> bool {
        self.kept.is_empty()
    }

    /// The prepared choice of a chart revision (recommend.event_key), as an entry index.
    pub fn find(&self, identity: &str) -> Option<usize> {
        self.by_identity.get(identity).copied()
    }

    /// the old nps.weighted_candidates (family best, quality weights), the unconsumed filter, the NPS shuffle and the web mix (Worker._pool) →
    /// ([(entry index, value, weight, category_share)], freshness relaxed), in shown order.
    #[allow(clippy::type_complexity)]
    pub fn rank(&self, r: Ranking<'_>) -> (Vec<(usize, f64, f64, f64)>, bool, [f64; 3]) {
        let items = &self.list.items;
        // Each chart family's best rate; equally good maps then reproduce library proportions.
        let mut best: Vec<usize> = Vec::new();
        let mut family_at: HashMap<(i64, &str), usize> = HashMap::default();
        for &i in &self.kept {
            let c = &items[i];
            if !r.keys.contains(&c.keys) || (r.web <= 0. && c.stream() == WEB) || (r.convert <= 0. && c.stream() == CONVERT) {
                continue;
            }
            match family_at.get(&(c.keys, c.family.as_str())) {
                Some(&at) => {
                    if c.value > items[best[at]].value {
                        best[at] = i;
                    }
                }
                None => {
                    family_at.insert((c.keys, c.family.as_str()), best.len());
                    best.push(i);
                }
            }
        }
        let mut rows = best;
        // Twenty NPS is ordinary 7K variety: follow the eligible pool down for lower ability or cold states,
        // and far below that floor is not NPS practice at all (user 2026-10-22: a 6 NPS map "should never appear").
        let mut floor: HashMap<i64, f64> = HashMap::default();
        if r.nps {
            for &k in &r.keys {
                let mut ns: Vec<f64> = rows.iter().map(|&i| &items[i]).filter(|c| c.keys == k && c.nps > 0.).map(|c| c.nps).collect();
                if !ns.is_empty() {
                    floor.insert(k, (20. * k as f64 / 7.).min(0.85 * quantile(&mut ns, 0.70)));
                }
            }
            rows.retain(|&i| {
                let c = &items[i];
                !(0. < c.nps && c.nps < 0.75 * floor.get(&c.keys).copied().unwrap_or(0.))
            });
        }
        if rows.is_empty() {
            return (Vec::new(), false, [1., 0., 0.]);
        }
        let mut counts: HashMap<(i64, &str), usize> = HashMap::default();
        rows.iter().for_each(|&i| *counts.entry((items[i].keys, items[i].group.as_str())).or_insert(0) += 1);
        let top = rows.iter().map(|&i| items[i].value).fold(f64::NEG_INFINITY, f64::max);
        let w: Vec<f64> = rows.iter().map(|&i| (items[i].value - top).exp()).collect();
        let pool = w.iter().sum::<f64>().powi(2) / w.iter().map(|x| x * x).sum::<f64>();
        let mut fresh = Vec::with_capacity(rows.len());
        for &i in &rows {
            let identity = items[i].identity();
            fresh.push(if (r.eventful)(&identity) { (r.freshness)(&identity, items[i].length, pool) } else { 1. });
        }
        let relax = fresh.iter().sum::<f64>() < 3.;
        let several = r.keys.len() > 1 && r.current_keys.is_some_and(|k| r.keys.contains(&k));
        let mut values = Vec::with_capacity(rows.len());
        for (&i, fr) in rows.iter().zip(&fresh) {
            let c = &items[i];
            let mut value = c.value + 0.65 * fr.powf(if relax { 0.3 } else { 1. }).max(0.001).ln();
            let f = floor.get(&c.keys).copied().unwrap_or(0.);
            if f != 0. && c.nps > 0. {
                value -= 24. * (f / c.nps).ln().max(0.);
            }
            if r.recent_families.contains(&c.family) {
                value -= 1.2;
            }
            if c.stream() == WEB {
                // Mild popularity prior for website charts only: ×0.7 near 10 plays → ×1 from 10k.
                value += (0.6 + 0.1 * 4f64.min((1. + c.playcount).log10())).ln();
            }
            let similar = r.recent_profiles.iter().filter(|a| a.len() == c.profile.len())
                .map(|a| a.iter().zip(&c.profile).map(|(x, y)| x * y).sum::<f64>()).fold(None, |m: Option<f64>, x| Some(m.map_or(x, |m| m.max(x))));
            if let Some(s) = similar {
                value -= 0.18 * s;
            }
            if several && Some(c.keys) != r.current_keys {
                value -= 0.12;
            }
            values.push(value);
        }
        let maximum = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mut ranked: Vec<(usize, f64, f64, f64)> = Vec::with_capacity(rows.len());
        for (&i, &value) in rows.iter().zip(&values) {
            let c = &items[i];
            let preference = r.taste.get(&(c.keys, c.group.clone())).map_or(1., |v| (0.65 * v).exp());
            ranked.push((i, value, (value - maximum).max(-20.).exp() * preference, counts[&(c.keys, c.group.as_str())] as f64 / rows.len() as f64));
        }
        ranked.sort_by(|a, b| b.2.total_cmp(&a.2));
        let mut available: Vec<(usize, f64, f64, f64)> = ranked.iter().filter(|x| !(r.consumed)(&items[x.0])).copied().collect();
        if available.is_empty() {
            available = ranked;
        }
        if let Some((seed, focus)) = &r.shuffle {
            let ids: Vec<&str> = available.iter().map(|x| {
                let c = &items[x.0];
                if c.md5.is_empty() { c.sha.as_str() } else { c.md5.as_str() }
            }).collect();
            let slots: Vec<((i64, &str), f64)> = available.iter().map(|x| slot(&items[x.0], r.skill_weights.as_ref())).collect();
            let weights = balanced(&available.iter().map(|x| x.2).collect::<Vec<_>>(), &slots, *focus);
            let order = shuffle_order(&ids, &weights, seed, *focus);
            available = order.into_iter().map(|i| available[i]).collect();
        }
        let streams: Vec<u8> = available.iter().map(|x| items[x.0].stream()).collect();
        let odds = stream_odds([r.web, r.convert], r.weighted, &streams);
        let order = mix_order(&streams, odds, &r.recent);
        let mut out: Vec<(usize, f64, f64, f64)> = order.into_iter().map(|i| available[i]).collect();
        if let Some(n) = r.count {
            out.truncate(n);
        }
        (out, relax, odds)
    }
}

/// np.quantile (linear)
fn quantile(v: &mut [f64], q: f64) -> f64 {
    v.sort_by(|a, b| a.total_cmp(b));
    let pos = q * (v.len() - 1) as f64;
    let (lo, hi) = (pos.floor() as usize, pos.ceil() as usize);
    let t = pos - lo as f64;
    let d = v[hi] - v[lo];
    if t >= 0.5 { v[hi] - d * (1. - t) } else { v[lo] + d * t }
}

/// pp_playlist.preview's order: a stable weighted shuffle keyed by `seed` and each identity (focus 0 samples by
/// weight, 1 keeps the highest weights in order).
fn shuffle_order(ids: &[&str], weights: &[f64], seed: &str, focus: f64) -> Vec<usize> {
    let priority: Vec<f64> = ids.iter().zip(weights).map(|(id, &w)| {
        let h = blake2b8(format!("{seed}:{id}").as_bytes());
        let u = (u64::from_be_bytes(h) as f64 + 1.) / (2f64.powi(64) + 1.);
        (1. - focus) * (-u.ln()).max(1e-300).ln() - w.max(1e-12).ln()
    }).collect();
    let mut order: Vec<usize> = (0..ids.len()).collect();
    order.sort_by(|&a, &b| priority[a].total_cmp(&priority[b]));
    order
}

/// What a chart's weight is balanced by: its (keys, pattern group) on NPS; on Skills with how-often sliders its
/// most wanted matched skill, across keymodes, with that slider as the skill's share.
fn slot<'a>(c: &'a Choice, skills: Option<&std::collections::HashMap<String, f64>>) -> ((i64, &'a str), f64) {
    let Some(skills) = skills else { return ((c.keys, c.group.as_str()), 1.) };
    let at = |k: &String| skills.get(k).copied().unwrap_or(1.);
    c.matched.iter().max_by(|a, b| at(a).total_cmp(&at(b))).map_or(((0, ""), skills.get("").copied().unwrap_or(1.)), |k| ((0, k.as_str()), at(k)))
}

/// Weights with each slot's total mass divided out as focus falls, then its share applied. At focus 0 every slot has
/// its share of the sampled list; at 1 the weights are untouched. Library proportions alone made low variety
/// "doar chordstreams si brackets" (Less, 2026-10-31): those own the top of NPS, so a harder shuffle changed nothing.
fn balanced(weights: &[f64], slots: &[((i64, &str), f64)], focus: f64) -> Vec<f64> {
    let mut mass: HashMap<(i64, &str), f64> = HashMap::default();
    for (w, s) in weights.iter().zip(slots) {
        *mass.entry(s.0).or_insert(0.) += w;
    }
    let evening = (1. - focus).powi(2);     // squared: the slider's middle keeps most of the library's mix
    weights.iter().zip(slots).map(|(w, s)| w * s.1 / mass[&s.0].powf(evening)).collect()
}

/// The odds of each stream against your own maps. Weighted, the sliders set a chance per map: at the middle every map
/// is equally likely, so twice as many converts the list can offer now means twice as many convert picks (user
/// 2026-11-01). Off, they are plain shares of the picks.
pub fn stream_odds(sliders: [f64; 2], weighted: [bool; 2], streams: &[u8]) -> [f64; 3] {
    let count = |k: u8| streams.iter().filter(|&&s| s == k).count() as f64;
    let per_map = |k: usize, stream: u8| if weighted[k] { sliders[k] * count(stream) / count(REAL).max(1.) } else { sliders[k] };
    [1., per_map(0, WEB), per_map(1, CONVERT)]
}

/// A stream's share still owed after a tab's recent offers, oldest first: the mix keeps the slider's share across
/// picks. A list rebuilt from zero after every pick opened with an installed map each time, and Next takes the
/// first row, so 0 of 167 picks at 15 % were website charts (G835LX 2026-10-27..31, user 2026-10-31).
fn credit(recent: &[u8], stream: u8, share: f64) -> f64 {
    recent.iter().fold(0., |c: f64, &s| (c + share - if s == stream { 1. } else { 0. }).clamp(-1., 1.))
}

/// nps.mix: your maps, website charts and converts interleaved, each stream in its ranked order, at its share of the
/// odds plus `credit` (sampling all from one weighted pool let whichever had more mass fill the list, user 2026-10-25).
/// recent: as Ranking.recent.
pub fn mix_order(streams: &[u8], odds: [f64; 3], recent: &[u8]) -> Vec<usize> {
    let mut pools: [std::collections::VecDeque<usize>; 3] = Default::default();
    for (i, &s) in streams.iter().enumerate() {
        pools[s as usize].push_back(i);
    }
    let total: f64 = odds.iter().sum();
    let share = odds.map(|o| o / total);
    let credits = [0., credit(recent, WEB, share[1]), credit(recent, CONVERT, share[2])];
    let mut taken = [0.; 3];
    (0..streams.len()).map(|i| {
        let owed = |k: usize| credits[k] + share[k] * (i + 1) as f64 - taken[k];
        // A website chart or a convert only once its share is due; the one owed most goes first.
        let due = [WEB as usize, CONVERT as usize].into_iter().filter(|&k| !pools[k].is_empty() && (pools[0].is_empty() || owed(k) >= 1. - 1e-9))
            .fold(None::<usize>, |best, k| if best.is_none_or(|b| owed(k) > owed(b)) { Some(k) } else { best });
        let side = due.unwrap_or(REAL as usize);
        taken[side] += 1.;
        pools[side].pop_front().unwrap()
    }).collect()
}

/// pp_playlist.preview's order of [(identity, weight)].
pub fn preview_order(rows: &[(String, f64)], seed: &str, focus: f64) -> Vec<usize> {
    let ids: Vec<&str> = rows.iter().map(|r| r.0.as_str()).collect();
    shuffle_order(&ids, &rows.iter().map(|r| r.1).collect::<Vec<_>>(), seed, focus)
}


/// BLAKE2b (RFC 7693) with an 8-byte digest, unkeyed: hashlib.blake2b(data, digest_size=8).digest().
fn blake2b8(data: &[u8]) -> [u8; 8] {
    const IV: [u64; 8] = [0x6a09e667f3bcc908, 0xbb67ae8584caa73b, 0x3c6ef372fe94f82b, 0xa54ff53a5f1d36f1,
                          0x510e527fade682d1, 0x9b05688c2b3e6c1f, 0x1f83d9abfb41bd6b, 0x5be0cd19137e2179];
    const SIGMA: [[usize; 16]; 10] = [
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], [14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3],
        [11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4], [7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8],
        [9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13], [2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9],
        [12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11], [13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10],
        [6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5], [10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0],
    ];
    let mut h = IV;
    h[0] ^= 0x0101_0000 ^ 8;
    let compress = |h: &mut [u64; 8], block: &[u8; 128], t: u128, last: bool| {
        let m: Vec<u64> = block.chunks_exact(8).map(|c| u64::from_le_bytes(c.try_into().unwrap())).collect();
        let mut v = [0u64; 16];
        v[..8].copy_from_slice(h);
        v[8..].copy_from_slice(&IV);
        v[12] ^= t as u64;
        v[13] ^= (t >> 64) as u64;
        if last {
            v[14] = !v[14];
        }
        let g = |v: &mut [u64; 16], a: usize, b: usize, c: usize, d: usize, x: u64, y: u64| {
            v[a] = v[a].wrapping_add(v[b]).wrapping_add(x);
            v[d] = (v[d] ^ v[a]).rotate_right(32);
            v[c] = v[c].wrapping_add(v[d]);
            v[b] = (v[b] ^ v[c]).rotate_right(24);
            v[a] = v[a].wrapping_add(v[b]).wrapping_add(y);
            v[d] = (v[d] ^ v[a]).rotate_right(16);
            v[c] = v[c].wrapping_add(v[d]);
            v[b] = (v[b] ^ v[c]).rotate_right(63);
        };
        for round in 0..12 {
            let s = &SIGMA[round % 10];
            g(&mut v, 0, 4, 8, 12, m[s[0]], m[s[1]]);
            g(&mut v, 1, 5, 9, 13, m[s[2]], m[s[3]]);
            g(&mut v, 2, 6, 10, 14, m[s[4]], m[s[5]]);
            g(&mut v, 3, 7, 11, 15, m[s[6]], m[s[7]]);
            g(&mut v, 0, 5, 10, 15, m[s[8]], m[s[9]]);
            g(&mut v, 1, 6, 11, 12, m[s[10]], m[s[11]]);
            g(&mut v, 2, 7, 8, 13, m[s[12]], m[s[13]]);
            g(&mut v, 3, 4, 9, 14, m[s[14]], m[s[15]]);
        }
        for i in 0..8 {
            h[i] ^= v[i] ^ v[i + 8];
        }
    };
    let mut t: u128 = 0;
    let mut chunks = data.chunks(128).peekable();
    if data.is_empty() {
        compress(&mut h, &[0; 128], 0, true);
    }
    while let Some(c) = chunks.next() {
        let mut block = [0u8; 128];
        block[..c.len()].copy_from_slice(c);
        t += c.len() as u128;
        compress(&mut h, &block, t, chunks.peek().is_none());
    }
    h[0].to_le_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blake2b_matches_hashlib() {
        // hashlib.blake2b(data, digest_size=8).hexdigest()
        let hex = |b: [u8; 8]| b.iter().map(|x| format!("{x:02x}")).collect::<String>();
        assert_eq!(hex(blake2b8(b"9:abc")), "813f29f73b38f7d5");
        assert_eq!(hex(blake2b8(b"")), "e4a6a0577479b2b4");
        assert_eq!(hex(blake2b8(&[b'x'; 300])), "12b539c1cad94615");
        assert_eq!(hex(blake2b8(b"123456789:0123456789abcdef0123456789abcdef")), "a9633b2b960a21fb");
    }
}

#[cfg(test)]
mod stream_tests {
    use super::*;

    fn picks(streams: &[u8], odds: [f64; 3], rounds: usize) -> [usize; 3] {
        let (mut recent, mut got) = (Vec::new(), [0; 3]);
        for _ in 0..rounds {
            let first = streams[mix_order(streams, odds, &recent)[0]];
            got[first as usize] += 1;
            recent.push(first);
            if recent.len() > 40 {
                recent.remove(0);
            }
        }
        got
    }

    fn pool(real: usize, web: usize, convert: usize) -> Vec<u8> {
        let mut v = vec![REAL; real];
        v.extend(vec![WEB; web]);
        v.extend(vec![CONVERT; convert]);
        v
    }

    /// User 2026-11-01: "if there's 5000 10k beatmaps and 10000 7k beatmaps then if you put it in the middle ... 7k converts would be twice as likely".
    #[test]
    fn the_middle_of_a_weighted_slider_is_the_same_chance_per_map() {
        let streams = pool(5000, 0, 10000);
        let odds = stream_odds([0., 1.], [true, true], &streams);
        assert_eq!(odds, [1., 0., 2.]);
        let got = picks(&streams, odds, 300);
        assert!((got[2] as i64 - 200).abs() <= 4, "{got:?}"); // the spacing window is 40 offers
        assert_eq!(stream_odds([0., 1.], [false, false], &streams), [1., 0., 1.]);
    }

    #[test]
    fn each_stream_keeps_its_share_across_picks() {
        for share in [0.15, 0.5, 0.8] {
            let odds = [1., 0., share / (1. - share)];
            let got = picks(&pool(100, 0, 100), odds, 40);
            assert!((got[2] as f64 - 40. * share).abs() <= 1., "{share} {got:?}");
        }
        let got = picks(&pool(50, 50, 50), [1., 1., 2.], 40);
        assert!((got[1] as i64 - 10).abs() <= 1 && (got[2] as i64 - 20).abs() <= 1, "{got:?}");
    }

    #[test]
    fn two_streams_order_as_before() {
        // nps.mix's order at 1:1 from zero: a website chart once its share is due.
        assert_eq!(mix_order(&[REAL, REAL, WEB, WEB], [1., 1., 0.], &[]), [0, 2, 1, 3]);
    }
}

/// Family best, density floor, focus, popularity prior and sliders on synthetic choices.
#[cfg(test)]
mod list_tests {
    use super::*;

    fn choice(sha: &str, nps: f64, group: &str) -> Choice {
        Choice { sha: sha.into(), md5: sha.into(), keys: 7, rate: 1., value: 0., installed: true, convert: false, playcount: -1., family: sha.into(), description: "Mixed".into(),
                 group: group.into(), nps, length: 180., profile: vec![1.], acc_mid: 0.94, matched: Vec::new(), practice: String::new(), pred: Prediction::default(), generated_at: 0. }
    }

    fn prepared(rows: Vec<Choice>, keys: &[i64]) -> Prepared {
        let order: Vec<String> = rows.iter().map(|c| c.sha.clone()).collect();
        let mut memo = Choices::new(order, Forecast::default());
        for (slot, c) in memo.slots.iter_mut().zip(rows) {
            *slot = Some((Some(Arc::new(c)), None));
        }
        Arc::new(memo.snapshot()).prepare(keys, &|_| true)
    }

    struct Ask {
        keys: Vec<i64>,
        nps: bool,
        web: f64,
        shuffle: Option<(String, f64)>,
        count: Option<usize>,
        weights: Option<std::collections::HashMap<String, f64>>,
    }

    impl Default for Ask {
        fn default() -> Ask {
            Ask { keys: vec![7], nps: true, web: 1., shuffle: None, count: None, weights: None }
        }
    }

    /// (choice, weight) in shown order.
    fn ranked(p: &Prepared, a: Ask) -> Vec<(Arc<Choice>, f64)> {
        let taste = HashMap::default();
        let (rows, _, _) = p.rank(Ranking { keys: a.keys, nps: a.nps, web: a.web, convert: 0., weighted: [false, false], taste: &taste, eventful: &|_| false, freshness: &|_, _, _| 1.,
                                            recent_families: Vec::new(), recent_profiles: Vec::new(), current_keys: None, consumed: &|_| false, shuffle: a.shuffle, count: a.count,
                                            recent: Vec::new(), skill_weights: a.weights });
        rows.into_iter().map(|(i, _, weight, _)| (p.list.items[i].clone(), weight)).collect()
    }

    #[test]
    fn ranking_keeps_one_choice_per_chart_family_and_keymode() {
        let mut rows: Vec<Choice> = (0..35).map(|i| {
            let g = if i < 25 { 0 } else if i < 30 { 1 } else { 2 };
            let mut c = choice(&i.to_string(), if g == 0 { 60. } else { 40. }, &g.to_string());
            c.profile = vec![0., 0., 0.];
            c.profile[g] = 1.;
            c.value = c.nps.ln();
            c.length = 240.;
            c
        }).collect();
        let mut copy = rows[30].clone();
        copy.sha = "copy30".into();
        copy.md5 = "c".repeat(32);
        rows.push(copy);
        let p = prepared(rows, &[7]);
        let got = ranked(&p, Ask::default());
        let families: HashSet<&str> = got.iter().map(|c| c.0.family.as_str()).collect();
        assert!(got.len() == 35 && families.len() == 35);
        assert!(ranked(&p, Ask { keys: vec![4], ..Ask::default() }).is_empty());
    }

    #[test]
    fn nps_never_offers_a_chart_far_below_the_density_floor() {
        let mut rows = vec![choice("floor", 6.2, "sv")];
        rows.extend((0..20).map(|i| choice(&i.to_string(), 25., "chords")));
        let got = ranked(&prepared(rows, &[7]), Ask::default());
        assert!(got.len() == 20 && got.iter().all(|c| c.0.nps == 25.));
    }

    #[test]
    fn nps_focus_changes_selection_without_changing_rate_targets() {
        let rows: Vec<Choice> = (0..100).map(|i| {
            let mut c = choice(&i.to_string(), if i < 80 { 22. } else { 34. }, "chords");
            c.rate = 1.1;
            c.value = c.nps.ln();
            c
        }).collect();
        let p = prepared(rows, &[7]);
        let list = |focus: f64| ranked(&p, Ask { shuffle: Some(("9".into(), focus)), count: Some(20), ..Ask::default() });
        assert!(list(1.).iter().all(|c| c.0.nps == 34.));
        let variety = list(0.);
        assert!(variety.iter().any(|c| c.0.nps == 22.) && variety.iter().any(|c| c.0.nps == 34.));
        let shas = |rows: &[(Arc<Choice>, f64)]| rows.iter().map(|c| c.0.sha.clone()).collect::<Vec<_>>();
        assert_eq!(shas(&variety), shas(&list(0.)));
        assert!(variety.iter().all(|c| c.0.rate == 1.1 && c.0.acc_mid == 0.94));
        assert_eq!(variety.iter().map(|c| &c.0.family).collect::<HashSet<_>>().len(), 20);
    }

    #[test]
    fn website_charts_get_a_mild_popularity_prior_and_the_left_end_drops_them() {
        let row = |sha: &str, installed: bool, playcount: f64| {
            let mut c = choice(sha, 30., "chords");
            (c.installed, c.playcount) = (installed, playcount);
            c
        };
        let p = prepared(vec![row("local", true, -1.), row("rare", false, 10.), row("popular", false, 20000.)], &[7]);
        let got = ranked(&p, Ask::default());
        let w = |s: &str| got.iter().find(|c| c.0.sha == s).unwrap().1;
        assert!((w("popular") / w("local") - 1.).abs() < 1e-9);
        assert!(0.65 < w("rare") / w("local") && w("rare") / w("local") < 0.75);
        assert_eq!(ranked(&p, Ask { web: 0., ..Ask::default() }).iter().map(|c| c.0.sha.as_str()).collect::<Vec<_>>(), ["local"]);
    }

    #[test]
    fn the_website_slider_mixes_both_kinds() {
        let streams: Vec<u8> = std::iter::repeat_n(WEB, 30).chain(std::iter::repeat_n(REAL, 10)).collect();
        let order = |web: f64| mix_order(&streams, [1., web, 0.], &[]);
        let half = order(1.);
        assert_eq!(half.len(), 40);
        assert_eq!(half[..20].iter().map(|&i| streams[i]).collect::<Vec<_>>(), [REAL, WEB].repeat(10));
        assert_eq!(order(4.)[..20].iter().filter(|&&i| streams[i] == WEB).count(), 16);
        assert_eq!(&half[20..23], &[10, 11, 12], "locals ran out");
        assert!(order(1. / 19.)[..10].iter().all(|&i| streams[i] == REAL), "5 %: one in 20");
    }

    #[test]
    fn low_variety_evens_out_pattern_groups() {
        let mut rows: Vec<Choice> = (0..90).map(|i| choice(&format!("c{i}"), 30., "chords")).collect();
        rows.extend((0..10).map(|i| choice(&format!("j{i}"), 30., "jack")));
        let p = prepared(rows, &[7]);
        let jacks = |focus: f64| (0..40).map(|seed| ranked(&p, Ask { shuffle: Some((seed.to_string(), focus)), ..Ask::default() }).iter().take(20).filter(|c| c.0.group == "jack").count()).sum::<usize>() as f64 / 800.;
        assert!(jacks(1.) < 0.1 && jacks(0.) > 0.3, "10 % of the pool; the 10 jack maps cap a list of 20 at half");
    }

    #[test]
    fn skill_sliders_share_the_list() {
        let rows: Vec<Choice> = (0..100).map(|i| {
            let mut c = choice(&format!("a{i}"), 30., "chords");
            c.matched = vec![if i < 50 { "jackspeed" } else { "delay" }.to_string()];
            c
        }).collect();
        let p = prepared(rows, &[7]);
        let share = |jackspeed: f64| {
            let weights: std::collections::HashMap<String, f64> = [("jackspeed".to_string(), jackspeed), ("delay".to_string(), 1.)].into();
            (0..200).filter(|seed| ranked(&p, Ask { nps: false, shuffle: Some((seed.to_string(), 0.5)), weights: Some(weights.clone()), ..Ask::default() })[0].0.matched[0] == "jackspeed").count() as f64 / 200.
        };
        assert!(share(4.) > 0.7 && share(0.25) < 0.3);
    }

    #[test]
    fn a_list_without_skill_settings_keeps_its_order() {
        let rows: Vec<Choice> = (0..40).map(|i| choice(&format!("a{i}"), 20. + i as f64, "chords")).collect();
        let p = prepared(rows, &[7]);
        let order = |weights: Option<std::collections::HashMap<String, f64>>| ranked(&p, Ask { nps: false, weights, ..Ask::default() }).iter().map(|x| x.0.sha.clone()).collect::<Vec<_>>();
        assert!(crate::skills::shares(&[], &Default::default(), 0.).is_none(), "no settings ask for no shares");
        assert!(crate::skills::shares(&["chordjack".to_string(), "delay".to_string()], &Default::default(), 0.).is_none(), "equal sliders neither");
        assert_eq!(order(None).len(), 40);
    }

    #[test]
    fn other_maps_take_their_slider_share_of_the_list() {
        let rows: Vec<Choice> = (0..100).map(|i| {
            let mut c = choice(&format!("a{i}"), 30., "chords");
            c.matched = if i < 50 { vec!["chordjack".to_string()] } else { Vec::new() };
            c
        }).collect();
        let p = prepared(rows, &[7]);
        let weights = crate::skills::shares(&["chordjack".to_string()], &[("chordjack".to_string(), 2.)].into(), 0.5).unwrap();
        let share = (0..300).filter(|seed| ranked(&p, Ask { nps: false, shuffle: Some((seed.to_string(), 0.)), weights: Some(weights.clone()), ..Ask::default() })[0].0.matched.len() == 1).count() as f64 / 300.;
        assert!((share - 0.8).abs() < 0.08, "{share}");
        assert!(crate::skills::shares(&["chordjack".to_string()], &Default::default(), 0.).is_none());
    }
}
