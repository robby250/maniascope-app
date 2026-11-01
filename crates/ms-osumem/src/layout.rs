//! Field offsets computed from the game's own DLLs: their .NET metadata, then CoreCLR's rules for laying out an object
//! (MethodTableBuilder::PlaceInstanceFields). Nothing here touches the running game.
use std::collections::HashMap;
use std::path::{Path, PathBuf};

const SLOT: u32 = 8;

#[derive(Clone, Copy)]
enum Col { U2, U4, Str, Guid, Blob, Idx(usize), Coded(usize) }
use Col::{Blob, Coded, Guid, Idx, Str, U2, U4};

/// (tag bits, tables) of each coded index, from ECMA-335 II.24.2.6.
const CODED: [(u32, &[usize]); 13] = [
    (2, &[2, 1, 27]), (2, &[4, 8, 23]), (5, &[6, 4, 1, 2, 8, 9, 10, 0, 14, 23, 20, 17, 26, 27, 32, 35, 38, 39, 40, 42, 44, 43]), (1, &[4, 8]), (2, &[2, 6, 32]), (3, &[2, 1, 26, 6, 27]),
    (1, &[20, 23]), (1, &[6, 10]), (1, &[4, 6]), (2, &[38, 35, 39]), (3, &[6, 10]), (2, &[0, 26, 35, 1]), (1, &[2, 6]),
];

const SCHEMA: [&[Col]; 45] = [
    &[U2, Str, Guid, Guid, Guid], &[Coded(11), Str, Str], &[U4, Str, Str, Coded(0), Idx(4), Idx(6)], &[Idx(4)], &[U2, Str, Blob], &[Idx(6)], &[U4, U2, U2, Str, Blob, Idx(8)], &[Idx(8)], &[U2, U2, Str],
    &[Idx(2), Coded(0)], &[Coded(5), Str, Blob], &[U2, Coded(1), Blob], &[Coded(2), Coded(10), Blob], &[Coded(3), Blob], &[U2, Coded(4), Blob], &[U2, U4, Idx(2)], &[U4, Idx(4)], &[Blob],
    &[Idx(2), Idx(20)], &[Idx(20)], &[U2, Str, Coded(0)], &[Idx(2), Idx(23)], &[Idx(23)], &[U2, Str, Blob], &[U2, Idx(6), Coded(6)], &[Idx(2), Coded(7), Coded(7)], &[Str], &[Blob],
    &[U2, Coded(8), Str, Idx(26)], &[U4, Idx(4)], &[U4, U4], &[U4], &[U4, U2, U2, U2, U2, U4, Blob, Str, Str], &[U4], &[U4, U4, U4], &[U2, U2, U2, U2, U4, Blob, Str, Str, Blob], &[U4, Idx(35)],
    &[U4, U4, U4, Idx(35)], &[U4, Str, Blob], &[U4, U4, Str, Str, Coded(9)], &[U4, U4, Str, Coded(9)], &[Idx(2), Idx(2)], &[U2, U2, Coded(12), Str], &[Coded(7), Blob], &[Idx(42), Coded(0)],
];

struct Table { start: usize, rows: u32, width: usize, cols: Vec<(usize, usize)> } // (offset, size)

/// One DLL's metadata.
struct Asm {
    data: Vec<u8>,
    strings: usize,
    blobs: usize,
    tables: Vec<Table>,
    types: HashMap<String, u32>,    // full name → TypeDef row
    forwarders: HashMap<String, u32>, // full name → ExportedType row
}

fn u16_at(d: &[u8], i: usize) -> u32 { u16::from_le_bytes([d[i], d[i + 1]]) as u32 }
fn u32_at(d: &[u8], i: usize) -> u32 { u32::from_le_bytes([d[i], d[i + 1], d[i + 2], d[i + 3]]) }

impl Asm {
    fn parse(data: Vec<u8>) -> Option<Asm> {
        let d = &data;
        let pe = u32_at(d, 0x3c) as usize;
        let (sections, opt_size) = (u16_at(d, pe + 6) as usize, u16_at(d, pe + 20) as usize);
        let opt = pe + 24;
        let dirs = opt + if u16_at(d, opt) == 0x20b { 112 } else { 96 };
        let first = opt + opt_size;
        let file = |rva: usize| (0..sections).map(|i| first + 40 * i).find_map(|s| {
            let (va, size, raw) = (u32_at(d, s + 12) as usize, u32_at(d, s + 8).max(u32_at(d, s + 16)) as usize, u32_at(d, s + 20) as usize);
            (rva >= va && rva < va + size).then(|| rva - va + raw)
        });
        let cli = file(u32_at(d, dirs + 14 * 8) as usize)?;
        let meta = file(u32_at(d, cli + 8) as usize)?;
        if u32_at(d, meta) != 0x424a_5342 {
            return None;
        }
        let version = u32_at(d, meta + 12) as usize;
        let count = u16_at(d, meta + 18 + version) as usize;
        let (mut at, mut stream) = (meta + 20 + version, HashMap::new());
        for _ in 0..count {
            let (offset, end) = (u32_at(d, at) as usize, d[at + 8..].iter().position(|b| *b == 0)?);
            stream.insert(String::from_utf8_lossy(&d[at + 8..at + 8 + end]).into_owned(), meta + offset);
            at += 8 + (end + 4) / 4 * 4;
        }
        let root = *stream.get("#~")?;
        let heaps = d[root + 6] as u32;
        let (valid, mut at) = (u32_at(d, root + 8) as u64 | (u32_at(d, root + 12) as u64) << 32, root + 24);
        let mut rows = [0u32; 64];
        for (i, row) in rows.iter_mut().enumerate() {
            if valid >> i & 1 == 1 {
                *row = u32_at(d, at);
                at += 4;
            }
        }
        let big = |t: &[usize], bits: u32| t.iter().map(|t| rows[*t]).max().unwrap_or(0) >= 1 << (16 - bits);
        let size = |c: Col| match c {
            U2 => 2,
            U4 => 4,
            Str => if heaps & 1 != 0 { 4 } else { 2 },
            Guid => if heaps & 2 != 0 { 4 } else { 2 },
            Blob => if heaps & 4 != 0 { 4 } else { 2 },
            Idx(t) => if rows[t] >= 1 << 16 { 4 } else { 2 },
            Coded(k) => if big(CODED[k].1, CODED[k].0) { 4 } else { 2 },
        };
        let mut tables = Vec::new();
        for (i, schema) in SCHEMA.iter().enumerate() {
            let (mut cols, mut width) = (Vec::new(), 0);
            for c in schema.iter() {
                cols.push((width, size(*c)));
                width += size(*c);
            }
            tables.push(Table { start: at, rows: rows[i], width, cols });
            at += width * rows[i] as usize;
        }
        let mut asm = Asm { strings: *stream.get("#Strings")?, blobs: *stream.get("#Blob")?, data: Vec::new(), tables, types: HashMap::new(), forwarders: HashMap::new() };
        asm.data = data;
        let enclosing: HashMap<u32, u32> = (1..=asm.tables[41].rows).map(|r| (asm.get(41, r, 0), asm.get(41, r, 1))).collect();
        for row in 1..=asm.tables[2].rows {
            let mut name = asm.type_name(row);
            let mut at = row;
            while let Some(outer) = enclosing.get(&at) {
                name = format!("{}+{name}", asm.type_name(*outer));
                at = *outer;
            }
            asm.types.insert(name, row);
        }
        for row in 1..=asm.tables[39].rows {
            let (space, name) = (asm.string(asm.get(39, row, 3)), asm.string(asm.get(39, row, 2)));
            asm.forwarders.insert(if space.is_empty() { name } else { format!("{space}.{name}") }, row);
        }
        Some(asm)
    }

    fn get(&self, table: usize, row: u32, col: usize) -> u32 {
        let t = &self.tables[table];
        let (offset, size) = t.cols[col];
        let at = t.start + t.width * (row as usize - 1) + offset;
        if size == 2 { u16_at(&self.data, at) } else { u32_at(&self.data, at) }
    }

    fn string(&self, at: u32) -> String {
        let bytes = &self.data[self.strings + at as usize..];
        String::from_utf8_lossy(&bytes[..bytes.iter().position(|b| *b == 0).unwrap_or(0)]).into_owned()
    }

    fn blob(&self, at: u32) -> &[u8] {
        let mut i = self.blobs + at as usize;
        let (len, used) = compressed(&self.data[i..]);
        i += used;
        &self.data[i..i + len as usize]
    }

    fn type_name(&self, row: u32) -> String {
        let (name, space) = (self.string(self.get(2, row, 1)), self.string(self.get(2, row, 2)));
        if space.is_empty() { name } else { format!("{space}.{name}") }
    }

    /// (first row, one past the last) of a TypeDef's fields.
    fn fields(&self, row: u32) -> (u32, u32) {
        let next = if row < self.tables[2].rows { self.get(2, row + 1, 4) } else { self.tables[4].rows + 1 };
        (self.get(2, row, 4), next)
    }

    fn class_layout(&self, row: u32) -> Option<(u32, u32)> {
        (1..=self.tables[15].rows).find(|r| self.get(15, *r, 2) == row).map(|r| (self.get(15, r, 0), self.get(15, r, 1)))
    }

    fn explicit_offset(&self, field: u32) -> Option<u32> {
        (1..=self.tables[16].rows).find(|r| self.get(16, *r, 1) == field).map(|r| self.get(16, r, 0))
    }
}

fn compressed(b: &[u8]) -> (u32, usize) {
    match b[0] {
        x if x & 0x80 == 0 => (x as u32, 1),
        x if x & 0xc0 == 0x80 => (((x as u32 & 0x3f) << 8) | b[1] as u32, 2),
        x => (((x as u32 & 0x1f) << 24) | (b[1] as u32) << 16 | (b[2] as u32) << 8 | b[3] as u32, 4),
    }
}

/// A type as far as layout cares.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
enum Ty {
    Prim(u32),            // size
    Ref,                  // a GC reference
    Named(usize, u32, Vec<Ty>, bool), // assembly, TypeDef row, type arguments, written as a value type
}

#[derive(Clone, Default)]
pub struct Layout {
    size: u32,        // instance bytes, the method table pointer not counted
    align: u32,
    refs: bool,
    sequential: bool,
    pub fields: HashMap<String, u32>, // name → offset from the object's address; a subclass's field hides its parent's
}

impl Layout {
    /// The first 8 bytes' upper half in a live object's method table: the object's size.
    pub fn base_size(&self) -> u32 {
        self.size.div_ceil(SLOT) * SLOT + 16
    }
}

pub struct Layouts {
    dir: PathBuf,
    asms: Vec<Asm>,
    names: HashMap<String, Option<usize>>,
    done: HashMap<(usize, u32, Vec<Ty>), Layout>,
}

impl Layouts {
    pub fn new(dir: &Path) -> Layouts {
        Layouts { dir: dir.to_path_buf(), asms: Vec::new(), names: HashMap::new(), done: HashMap::new() }
    }

    fn assembly(&mut self, name: &str) -> Option<usize> {
        if let Some(hit) = self.names.get(name) {
            return *hit;
        }
        let found = std::fs::read(self.dir.join(format!("{name}.dll"))).ok().and_then(Asm::parse).map(|a| {
            self.asms.push(a);
            self.asms.len() - 1
        });
        self.names.insert(name.to_string(), found);
        found
    }

    /// Where a name really lives, across type forwarders.
    fn find(&mut self, assembly: &str, name: &str) -> Option<(usize, u32)> {
        let a = self.assembly(assembly)?;
        if let Some(row) = self.asms[a].types.get(name) {
            return Some((a, *row));
        }
        let row = *self.asms[a].forwarders.get(name)?;
        let implementation = self.asms[a].get(39, row, 4);
        if implementation & 3 != 1 {
            return None;
        }
        let target = self.asms[a].string(self.asms[a].get(35, implementation >> 2, 6));
        self.find(&target, name)
    }

    fn resolve_ref(&mut self, a: usize, row: u32) -> Option<(usize, u32)> {
        let (scope, name, space) = (self.asms[a].get(1, row, 0), self.asms[a].string(self.asms[a].get(1, row, 1)), self.asms[a].string(self.asms[a].get(1, row, 2)));
        let full = if space.is_empty() { name.clone() } else { format!("{space}.{name}") };
        match scope & 3 {
            2 => {
                let target = self.asms[a].string(self.asms[a].get(35, scope >> 2, 6));
                self.find(&target, &full)
            }
            3 => {
                let (outer_asm, outer) = self.resolve_ref(a, scope >> 2)?;
                let outer_name = self.asms[outer_asm].types.iter().find(|t| *t.1 == outer)?.0.clone();
                self.asms[outer_asm].types.get(&format!("{outer_name}+{name}")).map(|r| (outer_asm, *r))
            }
            _ => None,
        }
    }

    fn type_def_or_ref(&mut self, a: usize, coded: u32, args: &[Ty]) -> Option<Ty> {
        let row = coded >> 2;
        match coded & 3 {
            0 => Some(Ty::Named(a, row, Vec::new(), false)),
            1 => self.resolve_ref(a, row).map(|(x, r)| Ty::Named(x, r, Vec::new(), false)),
            _ => {
                let blob = self.asms[a].get(27, row, 0);
                let sig = self.asms[a].blob(blob).to_vec();
                let mut at = 0;
                self.sig(a, &sig, &mut at, args)
            }
        }
    }

    fn sig(&mut self, a: usize, s: &[u8], at: &mut usize, args: &[Ty]) -> Option<Ty> {
        let code = *s.get(*at)?;
        *at += 1;
        let number = |at: &mut usize| {
            let (v, used) = compressed(&s[*at..]);
            *at += used;
            v
        };
        Some(match code {
            0x02 | 0x04 | 0x05 => Ty::Prim(1),
            0x03 | 0x06 | 0x07 => Ty::Prim(2),
            0x08 | 0x09 | 0x0c => Ty::Prim(4),
            0x0a | 0x0b | 0x0d | 0x18 | 0x19 => Ty::Prim(8),
            0x0e | 0x1c => Ty::Ref,
            0x0f => {
                self.sig(a, s, at, args); // a pointer: as big as a slot, unknown to the GC
                Ty::Prim(8)
            }
            0x1d | 0x10 => {
                self.sig(a, s, at, args);
                Ty::Ref
            }
            0x11 | 0x12 => {
                let coded = number(at);
                if code == 0x12 {
                    Ty::Ref
                } else {
                    match self.type_def_or_ref(a, coded, args)? {
                        Ty::Named(x, r, n, _) => Ty::Named(x, r, n, true),
                        other => other,
                    }
                }
            }
            0x13 => {
                let i = number(at) as usize;
                args.get(i).cloned().unwrap_or(Ty::Ref)
            }
            0x1e => {
                number(at);
                Ty::Ref
            }
            0x15 => {
                let kind = s[*at];
                *at += 1;
                let coded = number(at);
                let n = number(at);
                let list: Vec<Ty> = (0..n).map(|_| self.sig(a, s, at, args).unwrap_or(Ty::Ref)).collect();
                if kind == 0x12 {
                    // the instantiation still matters: a class's parent is written like this
                    match self.type_def_or_ref(a, coded, args) {
                        Some(Ty::Named(x, r, _, _)) => Ty::Named(x, r, list, false),
                        _ => Ty::Ref,
                    }
                } else {
                    match self.type_def_or_ref(a, coded, args)? {
                        Ty::Named(x, r, _, _) => Ty::Named(x, r, list, true),
                        other => other,
                    }
                }
            }
            0x1b => {
                // a function pointer's signature: sized as a slot; its bytes are skipped by the caller's end
                *at = s.len();
                Ty::Prim(8)
            }
            0x14 => {
                self.sig(a, s, at, args);
                number(at); // rank
                for _ in 0..number(at) {
                    number(at);
                }
                for _ in 0..number(at) {
                    number(at);
                }
                Ty::Ref
            }
            0x1f | 0x20 => {
                number(at);
                self.sig(a, s, at, args)?
            }
            _ => return None,
        })
    }

    fn enum_size(&mut self, a: usize, row: u32) -> Option<u32> {
        // an enum's one instance field is its underlying value
        let (first, end) = self.asms[a].fields(row);
        for f in first..end {
            if self.asms[a].get(4, f, 0) & 0x10 == 0 {
                let sig = self.asms[a].blob(self.asms[a].get(4, f, 2)).to_vec();
                let mut at = 1;
                if let Some(Ty::Prim(size)) = self.sig(a, &sig, &mut at, &[]) {
                    return Some(size);
                }
            }
        }
        None
    }

    fn parent(&mut self, a: usize, row: u32, args: &[Ty]) -> Option<(usize, u32, Vec<Ty>)> {
        let extends = self.asms[a].get(2, row, 3);
        if extends == 0 || extends >> 2 == 0 {
            return None;
        }
        match self.type_def_or_ref(a, extends, args)? {
            Ty::Named(x, r, n, _) => Some((x, r, n)),
            _ => None,
        }
    }

    fn base_name(&mut self, a: usize, row: u32) -> String {
        self.parent(a, row, &[]).map_or(String::new(), |(x, r, _)| self.asms[x].type_name(r))
    }

    /// Instance layout of a class or struct.
    fn layout(&mut self, a: usize, row: u32, args: &[Ty]) -> Layout {
        let key = (a, row, args.to_vec());
        if let Some(hit) = self.done.get(&key) {
            return hit.clone();
        }
        let flags = self.asms[a].get(2, row, 0);
        let base = self.base_name(a, row);
        let value = base == "System.ValueType" || base == "System.Enum";
        let mut out = Layout { align: 1, ..Layout::default() };
        let mut pos = 0;
        if !value {
            if let Some((x, r, n)) = self.parent(a, row, args) {
                let parent = self.layout(x, r, &n);
                out.fields = parent.fields;
                pos = parent.size;
                out.refs = parent.refs;
                out.align = parent.align.max(1);
            }
        }
        let (first, end) = self.asms[a].fields(row);
        let mut prims: Vec<(String, u32, u32)> = Vec::new();
        let mut refs: Vec<String> = Vec::new();
        let mut nested: Vec<(String, Layout)> = Vec::new();
        let mut all_blittable = true;
        for f in first..end {
            let field_flags = self.asms[a].get(4, f, 0);
            if field_flags & 0x10 != 0 || field_flags & 0x40 != 0 {
                continue;
            }
            let name = self.asms[a].string(self.asms[a].get(4, f, 1));
            let sig = self.asms[a].blob(self.asms[a].get(4, f, 2)).to_vec();
            let mut at = 1;
            match self.sig(a, &sig, &mut at, args).unwrap_or(Ty::Ref) {
                Ty::Prim(size) => prims.push((name, size, f)),
                Ty::Named(x, r, n, true) => {
                    let parent = self.base_name(x, r);
                    if parent == "System.Enum" {
                        prims.push((name, self.enum_size(x, r).unwrap_or(4), f));
                    } else {
                        let inner = self.layout(x, r, &n);
                        all_blittable &= !inner.refs && inner.sequential;
                        nested.push((name, inner));
                    }
                }
                _ => {
                    all_blittable = false;
                    refs.push(name);
                }
            }
        }
        let layout_kind = flags & 0x18;
        let (pack, class_size) = self.asms[a].class_layout(row).unwrap_or((0, 0));
        if layout_kind == 0x10 {
            // explicit: the offsets are written down
            for (name, size, f) in &prims {
                let o = self.asms[a].explicit_offset(*f).unwrap_or(0);
                out.fields.insert(name.clone(), 8 + o);
                pos = pos.max(o + size);
            }
            for name in &refs {
                out.refs = true;
                pos = pos.max(SLOT);
                out.fields.insert(name.clone(), 8);
            }
            for (name, inner) in &nested {
                out.fields.insert(name.clone(), 8);
                pos = pos.max(inner.size);
            }
            out.size = pos.max(class_size);
            out.align = prims.iter().map(|p| p.1).chain(nested.iter().map(|n| n.1.align)).max().unwrap_or(1).min(SLOT);
            self.done.insert(key, out.clone());
            return out;
        }
        if value && layout_kind == 8 && all_blittable {
            // managed sequential: declaration order, natural alignment up to the pack
            let pack = if pack == 0 { SLOT } else { pack };
            let mut in_order: Vec<(u32, u32, u32, String)> = prims.iter().map(|p| (p.2, p.1, p.1, p.0.clone())).collect();
            let rows: HashMap<String, u32> = (first..end).map(|f| (self.asms[a].string(self.asms[a].get(4, f, 1)), f)).collect();
            in_order.extend(nested.iter().map(|n| (rows[&n.0], n.1.size, n.1.align, n.0.clone())));
            in_order.sort();
            let mut max_align = 1;
            for (_, size, align, name) in in_order {
                let align = align.max(1).min(pack);
                pos = pos.next_multiple_of(align);
                max_align = max_align.max(align);
                out.fields.insert(name, 8 + pos);
                pos += size;
            }
            out.size = pos.next_multiple_of(max_align).max(class_size).max(1);
            out.align = max_align;
            out.sequential = true;
            self.done.insert(key, out.clone());
            return out;
        }
        // automatic: CoreCLR's order. Gaps a parent left are filled by small fields, then pointer-sized GC references,
        // then primitives from large to small, structs last.
        let size_count = |prims: &[(String, u32, u32)], s: u32| prims.iter().filter(|p| p.1 == s).count();
        if pos % SLOT != 0 {
            let mut i = 0;
            while i < 3 {
                if pos % (2 << i) == 0 {
                    i += 1;
                    continue;
                }
                if !(i + 1..=3).any(|j| size_count(&prims, 1 << j) != 0 || (j == 3 && !refs.is_empty())) {
                    break;
                }
                let Some(j) = (0..=i).rev().find(|j| size_count(&prims, 1 << j) != 0) else { break };
                i = j;
                let k = prims.iter().position(|p| p.1 == 1 << i).unwrap();
                let (name, size, _) = prims.remove(k);
                pos = pos.next_multiple_of(size);
                out.fields.insert(name, 8 + pos);
                pos += size;
                if size_count(&prims, 1 << i) == 0 {
                    i += 1;
                }
            }
        }
        if !refs.is_empty() || prims.iter().any(|p| p.1 == 8) {
            pos = pos.next_multiple_of(SLOT);
        }
        for name in refs.iter() {
            out.refs = true;
            out.fields.insert(name.clone(), 8 + pos);
            pos += SLOT;
        }
        for size in [8, 4, 2, 1] {
            if size_count(&prims, size) == 0 {
                continue;
            }
            pos = pos.next_multiple_of(size);
            for p in prims.iter().filter(|p| p.1 == size) {
                out.fields.insert(p.0.clone(), 8 + pos);
                pos += size;
            }
            out.align = out.align.max(size);
        }
        for (name, inner) in &nested {
            let align = if inner.refs { SLOT } else { inner.align.max(1) };
            pos = pos.next_multiple_of(align);
            out.fields.insert(name.clone(), 8 + pos);
            pos += inner.size;
            out.refs |= inner.refs;
            out.align = out.align.max(align);
        }
        if !refs.is_empty() {
            out.align = SLOT;
        }
        if value {
            let min = if pos == 0 { 1 } else if pos > SLOT { if out.refs { SLOT } else { out.align } } else { pos.next_power_of_two() };
            pos = pos.max(1).next_multiple_of(min);
        }
        out.size = pos;
        self.done.insert(key, out.clone());
        out
    }

    /// `osu.Game.OsuGame`, nested types with `+`, searched through `assemblies`.
    pub fn class(&mut self, assemblies: &[&str], name: &str) -> Option<Layout> {
        let (a, row) = assemblies.iter().find_map(|x| self.find(x, name))?;
        Some(self.layout(a, row, &[]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compressed_numbers_decode_in_all_three_widths() {
        assert_eq!(compressed(&[0x03]), (3, 1));
        assert_eq!(compressed(&[0x80, 0x80]), (0x80, 2));
        assert_eq!(compressed(&[0xc0, 0x00, 0x40, 0x00]), (0x4000, 4));
    }

    #[test]
    fn the_object_size_counts_the_header_and_rounds_up() {
        assert_eq!(Layout { size: 1805, ..Layout::default() }.base_size(), 1824);
    }
}
