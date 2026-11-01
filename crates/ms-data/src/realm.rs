//! Read-only reader for lazer's client.realm (Realm file format 24), and lazer_scores.js's dump over it. Layout from
//! realm-core 14.14: nodes are 8-byte headers ("AAAA", flags, 24-bit big-endian size), tables are cluster trees.
//! realm-db-reader only reads format 9.9, from before cluster trees; realm-js meant shipping Node.
use chrono::{TimeZone, Utc};
use memmap2::Mmap;
use serde_json::{json, Value};
use std::path::Path;

type R<T> = Result<T, String>;

pub const FORMAT: u8 = 24;
const FIRST_COL: usize = 1; // Cluster::s_first_col_index
const INNER_FIRST: usize = 3; // ClusterNodeInner::s_first_node_index
const SHIFT: u32 = 8; // Cluster::node_shift_factor
const NULL_LINK_MASK: i64 = 1 << 62; // unresolved links point at tombstones, which the dump treats as absent

#[derive(Clone, Copy)]
struct Node<'a> {
    file: &'a [u8],
    flags: u8,
    width: usize, // bits per element
    size: usize,
    data: usize,
}

impl<'a> Node<'a> {
    fn at(file: &'a [u8], at: u64) -> R<Node<'a>> {
        let at = usize::try_from(at).map_err(|_| "ref out of range")?;
        let head = file.get(at..at + 8).filter(|_| at % 8 == 0).ok_or_else(|| format!("bad node ref {at}"))?;
        let (flags, size) = (head[4], (head[5] as usize) << 16 | (head[6] as usize) << 8 | head[7] as usize);
        let width = (1usize << (flags & 7)) >> 1;
        let node = Node { file, flags, width, size, data: at + 8 };
        let bytes = match (flags >> 3) & 3 {
            0 => (size * width).div_ceil(8),
            1 => size * width,
            _ => size,
        };
        file.get(node.data..node.data + bytes).ok_or_else(|| format!("node {at} runs past the end of the file"))?;
        Ok(node)
    }

    fn inner(&self) -> bool {
        self.flags & 0x80 != 0
    }

    fn has_refs(&self) -> bool {
        self.flags & 0x40 != 0
    }

    fn context(&self) -> bool {
        self.flags & 0x20 != 0
    }

    fn multiply(&self) -> bool {
        (self.flags >> 3) & 3 == 1
    }

    fn bytes(&self) -> &'a [u8] {
        &self.file[self.data..self.data + self.size]
    }

    /// Array::get: sub-byte widths are unsigned, the rest signed.
    fn int(&self, i: usize) -> i64 {
        if i >= self.size {
            return 0;
        }
        let d = &self.file[self.data..];
        match self.width {
            0 => 0,
            1 | 2 | 4 => {
                let bit = i * self.width;
                ((d[bit / 8] >> (bit % 8)) & ((1u16 << self.width) - 1) as u8) as i64
            }
            8 => d[i] as i8 as i64,
            16 => i16::from_le_bytes(d[i * 2..i * 2 + 2].try_into().unwrap()) as i64,
            32 => i32::from_le_bytes(d[i * 4..i * 4 + 4].try_into().unwrap()) as i64,
            _ => i64::from_le_bytes(d[i * 8..i * 8 + 8].try_into().unwrap()),
        }
    }

    /// ArrayUnsigned::get (key arrays).
    fn unsigned(&self, i: usize) -> i64 {
        let d = &self.file[self.data..];
        match self.width {
            8 => d[i] as i64,
            16 => u16::from_le_bytes(d[i * 2..i * 2 + 2].try_into().unwrap()) as i64,
            32 => u32::from_le_bytes(d[i * 4..i * 4 + 4].try_into().unwrap()) as i64,
            _ => self.int(i),
        }
    }

    /// The child at `i`, None for a null ref or a tagged integer.
    fn child(&self, i: usize) -> R<Option<Node<'a>>> {
        let v = self.int(i);
        if v == 0 || v & 1 == 1 { Ok(None) } else { Node::at(self.file, v as u64).map(Some) }
    }

    fn need(&self, i: usize) -> R<Node<'a>> {
        self.child(i)?.ok_or_else(|| format!("missing child {i}"))
    }

    /// One string of a string leaf; None = null.
    fn string(&self, i: usize) -> R<Option<String>> {
        let text = |b: &[u8]| Some(String::from_utf8_lossy(b).into_owned());
        if !self.has_refs() {
            if !self.multiply() {
                return Err("enumerated string column".to_string());
            }
            if self.width == 0 {
                return Ok(None);
            }
            let cell = &self.file[self.data + i * self.width..self.data + (i + 1) * self.width];
            let pad = cell[self.width - 1] as usize;
            return Ok(if pad == self.width { None } else { text(&cell[..self.width - 1 - pad]) });
        }
        if self.context() {
            let Some(blob) = self.child(i)? else { return Ok(None) };
            let b = blob.bytes();
            return Ok(text(&b[..b.len().saturating_sub(1)]));
        }
        let (offsets, blob, nulls) = (self.need(0)?, self.need(1)?, self.need(2)?);
        if nulls.int(i) != 0 {
            return Ok(None);
        }
        let begin = if i > 0 { offsets.int(i - 1) as usize } else { 0 };
        let end = offsets.int(i) as usize;
        let b = blob.bytes().get(begin..end.saturating_sub(1)).ok_or("bad string offsets")?;
        Ok(text(b))
    }

    /// The values of a list (BPlusTree), in order.
    fn tree(&self, out: &mut Vec<i64>) -> R<()> {
        if self.inner() {
            for i in 1..self.size.saturating_sub(1) {
                self.need(i)?.tree(out)?;
            }
        } else {
            out.extend((0..self.size).map(|i| self.int(i)));
        }
        Ok(())
    }
}

#[derive(Clone)]
pub struct Column {
    index: usize,
    kind: u8,
    nullable: bool,
    list: bool,
}

/// A table's clusters and keys. Keys ascend through the tree, so a link resolves by binary search.
pub struct Table<'a> {
    columns: Vec<(String, Column)>,
    leaves: Vec<(Node<'a>, usize)>, // cluster, rows
    pub keys: Vec<i64>,
}

impl<'a> Table<'a> {
    fn open(top: Node<'a>) -> R<Table<'a>> {
        let spec = top.need(0)?;
        let (names, keys) = (spec.need(1)?, spec.need(5)?);
        let mut columns = Vec::new();
        for i in 0..keys.size {
            let key = keys.int(i);
            let attrs = (key >> 22) & 0xff;
            // Backlink columns sit after the named ones and have no name.
            if i < names.size {
                columns.push((names.string(i)?.unwrap_or_default(), Column { index: (key & 0xffff) as usize, kind: ((key >> 16) & 0x3f) as u8, nullable: attrs & 16 != 0, list: attrs & 32 != 0 }));
            }
        }
        let mut table = Table { columns, leaves: Vec::new(), keys: Vec::new() };
        if let Some(root) = top.child(2)? {
            table.walk(root, 0)?;
        }
        Ok(table)
    }

    fn walk(&mut self, node: Node<'a>, offset: i64) -> R<()> {
        if node.inner() {
            let keys = node.child(0)?;
            let shift = (node.int(1) >> 1) as u32 * SHIFT;
            for c in INNER_FIRST..node.size {
                let i = c - INNER_FIRST;
                let at = keys.map_or((i as i64) << shift, |k| k.unsigned(i));
                self.walk(node.need(c)?, offset + at)?;
            }
            return Ok(());
        }
        let first = node.int(0);
        let keys = if first & 1 == 1 { None } else { Some(node.need(0)?) };
        let rows = keys.map_or((first >> 1) as usize, |k| k.size);
        self.keys.extend((0..rows).map(|i| offset + keys.map_or(i as i64, |k| k.unsigned(i))));
        self.leaves.push((node, rows));
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.keys.len()
    }

    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }

    /// The row of a key.
    pub fn row(&self, key: i64) -> Option<usize> {
        self.keys.binary_search(&key).ok()
    }

    fn column(&self, name: &str, kinds: &[u8]) -> R<&Column> {
        let c = self.columns.iter().find(|c| c.0 == name).map(|c| &c.1).ok_or_else(|| format!("no column {name}"))?;
        if kinds.contains(&c.kind) { Ok(c) } else { Err(format!("column {name} has type {}", c.kind)) }
    }

    fn cells<T>(&self, c: &Column, f: impl Fn(Node<'a>, usize) -> R<T>) -> R<Vec<T>> {
        let mut out = Vec::with_capacity(self.keys.len());
        for (cluster, rows) in &self.leaves {
            let leaf = cluster.need(c.index + FIRST_COL)?;
            for i in 0..*rows {
                out.push(f(leaf, i)?);
            }
        }
        Ok(out)
    }

    pub fn ints(&self, name: &str) -> R<Vec<Option<i64>>> {
        let c = self.column(name, &[0])?.clone();
        let nullable = c.nullable;
        self.cells(&c, |leaf, i| Ok(if nullable {
            let v = leaf.int(i + 1);
            (v != leaf.int(0)).then_some(v)
        } else {
            Some(leaf.int(i))
        }))
    }

    pub fn bools(&self, name: &str) -> R<Vec<bool>> {
        let c = self.column(name, &[1])?.clone();
        if c.nullable {
            return Err(format!("nullable bool {name}"));
        }
        self.cells(&c, |leaf, i| Ok(leaf.int(i) != 0))
    }

    pub fn strings(&self, name: &str) -> R<Vec<Option<String>>> {
        let c = self.column(name, &[2])?.clone();
        self.cells(&c, |leaf, i| leaf.string(i))
    }

    fn floats(&self, name: &str) -> R<Vec<f64>> {
        let c = self.column(name, &[9, 10])?.clone();
        let double = c.kind == 10;
        self.cells(&c, move |leaf, i| {
            let d = &leaf.file[leaf.data..];
            Ok(if double {
                f64::from_le_bytes(d[i * 8..i * 8 + 8].try_into().unwrap())
            } else {
                f32::from_le_bytes(d[i * 4..i * 4 + 4].try_into().unwrap()) as f64
            })
        })
    }

    /// (seconds, nanoseconds)
    pub fn times(&self, name: &str) -> R<Vec<Option<(i64, i64)>>> {
        let c = self.column(name, &[8])?.clone();
        self.cells(&c, |leaf, i| {
            let (seconds, nanos) = (leaf.need(0)?, leaf.need(1)?);
            let s = seconds.int(i + 1);
            Ok((s != seconds.int(0)).then(|| (s, nanos.int(i))))
        })
    }

    pub fn links(&self, name: &str) -> R<Vec<Option<i64>>> {
        let c = self.column(name, &[12])?.clone();
        self.cells(&c, |leaf, i| {
            let v = leaf.int(i) - 1;
            Ok((v >= 0 && v & NULL_LINK_MASK == 0).then_some(v))
        })
    }

    /// A list column: link keys or integers per row.
    pub fn lists(&self, name: &str) -> R<Vec<Vec<i64>>> {
        let c = self.column(name, &[0, 12])?.clone();
        if !c.list {
            return Err(format!("{name} is not a list"));
        }
        self.cells(&c, |leaf, i| {
            let mut out = Vec::new();
            if let Some(tree) = leaf.child(i)? {
                tree.tree(&mut out)?;
            }
            Ok(out)
        })
    }

    pub fn uuids(&self, name: &str) -> R<Vec<Option<String>>> {
        const WIDTH: usize = 16;
        let c = self.column(name, &[17])?.clone();
        self.cells(&c, |leaf, i| {
            let (block, offset) = (i / 8, i % 8);
            let base = block * (WIDTH * 8 + 1);
            let d = leaf.bytes();
            if leaf.width == 0 || d[base] & (1 << offset) != 0 {
                return Ok(None);
            }
            let b = &d[base + 1 + offset * WIDTH..base + 1 + (offset + 1) * WIDTH];
            let hex = |x: &[u8]| x.iter().map(|v| format!("{v:02x}")).collect::<String>();
            Ok(Some(format!("{}-{}-{}-{}-{}", hex(&b[..4]), hex(&b[4..6]), hex(&b[6..8]), hex(&b[8..10]), hex(&b[10..]))))
        })
    }
}

pub struct Realm {
    map: Mmap,
}

impl Realm {
    pub fn open(path: &Path) -> R<Realm> {
        let file = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
        // SAFETY: a private snapshot (dump's copy); nothing else writes it.
        let map = unsafe { Mmap::map(&file) }.map_err(|e| e.to_string())?;
        if map.len() < 24 || &map[16..20] != b"T-DB" {
            return Err("not a Realm file".to_string());
        }
        let select = (map[23] & 1) as usize;
        if map[20 + select] != FORMAT {
            return Err(format!("client.realm uses Realm file format {}, this version reads {FORMAT}: update ManiaScope", map[20 + select]));
        }
        Ok(Realm { map })
    }

    fn top(&self) -> R<Node<'_>> {
        let select = (self.map[23] & 1) as usize;
        let at = u64::from_le_bytes(self.map[select * 8..select * 8 + 8].try_into().unwrap());
        if at == u64::MAX {
            return Err("streaming-form Realm file".to_string());
        }
        Node::at(&self.map, at)
    }

    pub fn table(&self, name: &str) -> R<Table<'_>> {
        let top = self.top()?;
        let (names, tables) = (top.need(0)?, top.need(1)?);
        let wanted = format!("class_{name}");
        for i in 0..names.size {
            if names.string(i)?.as_deref() == Some(wanted.as_str()) {
                return Table::open(tables.need(i)?);
            }
        }
        Err(format!("no table {name}"))
    }
}

/// A Timestamp as a JS Date's JSON: the same double arithmetic, because played times are part of score keys (integer
/// milliseconds moved 2 of 12k plays by 1 ms, which would have duplicated them).
fn iso((seconds, nanos): (i64, i64)) -> Value {
    Utc.timestamp_millis_opt((seconds as f64 * 1000. + nanos as f64 / 1e6) as i64).single().map_or(Value::Null, |t| json!(t.format("%Y-%m-%dT%H:%M:%S%.3fZ").to_string()))
}

/// lazer_scores.js over a snapshot: one object per mania score, then with `maps` one {"map": …} per installed mania
/// difficulty. `recent`: only scores from this JS timestamp on, as (seconds, nanoseconds).
/// The files of the set a chart belongs to: (name in the set, content hash).
pub fn set_files(path: &Path, sha: &str) -> R<Vec<(String, String)>> {
    let realm = Realm::open(path)?;
    let (beatmap, sets, usage, files) = (realm.table("Beatmap")?, realm.table("BeatmapSet")?, realm.table("RealmNamedFileUsage")?, realm.table("File")?);
    let (hashes, links) = (beatmap.strings("Hash")?, beatmap.links("BeatmapSet")?);
    let row = hashes.iter().position(|h| h.as_deref() == Some(sha)).ok_or("the chart isn't in the client's library")?;
    let set = links[row].and_then(|k| sets.row(k)).ok_or("the chart has no set")?;
    let (lists, names, file_links, file_hash) = (sets.lists("Files")?, usage.strings("Filename")?, usage.links("File")?, files.strings("Hash")?);
    Ok(lists[set].iter().filter_map(|&k| usage.row(k)).filter_map(|u| {
        Some((names[u].clone()?, file_links[u].and_then(|k| files.row(k)).and_then(|r| file_hash[r].clone())?))
    }).collect())
}

pub fn dump(path: &Path, maps: bool, recent: Option<(i64, i64)>) -> R<Vec<Value>> {
    let realm = Realm::open(path)?;
    let rules = realm.table("Ruleset")?;
    let names = rules.strings("ShortName")?;
    let mania = names.iter().position(|n| n.as_deref() == Some("mania")).map(|i| rules.keys[i]);
    let score = realm.table("Score")?;
    let beatmap = realm.table("Beatmap")?;
    let users = realm.table("RealmUser")?;
    let usage = realm.table("RealmNamedFileUsage")?;
    let files = realm.table("File")?;
    let (user_name, user_id) = (users.strings("Username")?, users.ints("OnlineID")?);
    let (usage_name, usage_file) = (usage.strings("Filename")?, usage.links("File")?);
    let file_hash = files.strings("Hash")?;
    let hash_of = |key: Option<i64>| key.and_then(|k| files.row(k)).and_then(|r| file_hash[r].clone());
    let (b_online, b_md5, b_status) = (beatmap.ints("OnlineID")?, beatmap.strings("MD5Hash")?, beatmap.ints("Status")?);
    let mut out = Vec::new();
    let Some(mania) = mania else { return Ok(out) };
    let (s_rule, s_deleted, s_id, s_hash, s_acc) = (score.links("Ruleset")?, score.bools("DeletePending")?, score.uuids("ID")?, score.strings("BeatmapHash")?, score.floats("Accuracy")?);
    let (s_total, s_combo, s_rank, s_date) = (score.ints("TotalScore")?, score.ints("MaxCombo")?, score.ints("Rank")?, score.times("Date")?);
    let (s_mods, s_stats, s_user, s_online) = (score.strings("Mods")?, score.strings("Statistics")?, score.links("User")?, score.ints("OnlineID")?);
    let (s_legacy_id, s_legacy, s_client, s_max) = (score.ints("LegacyOnlineID")?, score.bools("IsLegacyScore")?, score.strings("ClientVersion")?, score.strings("MaximumStatistics")?);
    let (s_files, s_pauses, s_map) = (score.lists("Files")?, score.lists("Pauses")?, score.links("BeatmapInfo")?);
    for i in 0..score.len() {
        if s_rule[i] != Some(mania) || s_deleted[i] || recent.is_some_and(|cut| s_date[i].is_some_and(|d| d < cut)) {
            continue;
        }
        let b = s_map[i].and_then(|k| beatmap.row(k));
        let user = s_user[i].and_then(|k| users.row(k));
        let replay = s_files[i].iter().filter_map(|&k| usage.row(k)).find(|&u| usage_name[u].as_deref() == Some("replay.osr")).and_then(|u| hash_of(usage_file[u]));
        out.push(json!({
            "id": s_id[i], "hash": s_hash[i], "acc": s_acc[i], "total": s_total[i], "combo": s_combo[i], "rank": s_rank[i], "date": s_date[i].map_or(Value::Null, iso),
            "mods": s_mods[i], "stats": s_stats[i], "user": user.and_then(|u| user_name[u].clone()), "user_id": user.and_then(|u| user_id[u]), "online_id": s_online[i],
            "legacy_online_id": s_legacy_id[i], "legacy": s_legacy[i], "client": s_client[i], "max_stats": s_max[i], "replay_sha256": replay, "pauses": s_pauses[i],
            "beatmap_id": b.and_then(|b| b_online[b]), "md5": b.and_then(|b| b_md5[b].clone()), "status": b.and_then(|b| b_status[b]),
        }));
    }
    if !maps {
        return Ok(out);
    }
    let sets = realm.table("BeatmapSet")?;
    let meta = realm.table("BeatmapMetadata")?;
    let diff = realm.table("BeatmapDifficulty")?;
    let (set_deleted, set_online, set_files) = (sets.bools("DeletePending")?, sets.ints("OnlineID")?, sets.lists("Files")?);
    let (m_title, m_artist, m_audio, m_author) = (meta.strings("Title")?, meta.strings("Artist")?, meta.strings("AudioFile")?, meta.links("Author")?);
    let (d_cs, d_od) = (diff.floats("CircleSize")?, diff.floats("OverallDifficulty")?);
    let (b_rule, b_hidden, b_set, b_meta, b_diff) = (beatmap.links("Ruleset")?, beatmap.bools("Hidden")?, beatmap.links("BeatmapSet")?, beatmap.links("Metadata")?, beatmap.links("Difficulty")?);
    let (b_hash, b_online_md5, b_length, b_name, b_last) = (beatmap.strings("Hash")?, beatmap.strings("OnlineMD5Hash")?, beatmap.floats("Length")?, beatmap.strings("DifficultyName")?, beatmap.times("LastPlayed")?);
    for i in 0..beatmap.len() {
        let set = b_set[i].and_then(|k| sets.row(k));
        if b_rule[i] != Some(mania) || b_hidden[i] || !set.is_some_and(|s| !set_deleted[s]) {
            continue;
        }
        let set = set.unwrap();
        let m = b_meta[i].and_then(|k| meta.row(k));
        let d = b_diff[i].and_then(|k| diff.row(k));
        let wanted = m.and_then(|m| m_audio[m].as_deref()).filter(|a| !a.is_empty());
        let audio = wanted.and_then(|name| {
            let name = name.to_lowercase();
            set_files[set].iter().filter_map(|&k| usage.row(k)).find(|&u| usage_name[u].as_deref().unwrap_or("").to_lowercase() == name).and_then(|u| hash_of(usage_file[u]))
        });
        let author = m.and_then(|m| m_author[m]).and_then(|k| users.row(k));
        out.push(json!({
            "map": b_hash[i], "beatmap_id": b_online[i], "set_id": set_online[set], "md5": b_md5[i], "online_md5": b_online_md5[i], "audio_sha": audio,
            "audio_required": wanted.is_some(), "creator": author.and_then(|a| user_name[a].clone()), "status": b_status[i], "keys": d.map(|d| d_cs[d]), "od": d.map(|d| d_od[d]),
            "length": b_length[i], "title": m.and_then(|m| m_title[m].clone()), "artist": m.and_then(|m| m_artist[m].clone()), "version": b_name[i], "last_played": b_last[i].map_or(Value::Null, iso),
        }));
    }
    Ok(out)
}
