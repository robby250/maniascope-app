//! One mapped file: a JSON header, a sorted index and feature blobs read only when asked.
use crate::codec;
use memmap2::Mmap;
use ms_calc::val::Val;
use std::fs::File;
use std::io::Write;
use std::path::Path;

const ENTRY: usize = 28; // id i64, rate f64, offset u64, length u32

pub struct Blobs {
    map: Mmap,
    index: usize, // offset of the first entry
    entries: usize,
    blobs: usize,
}

/// Records sorted by (id, rate). Written beside the target and renamed: an interrupted write leaves the old file.
pub fn write(path: &Path, magic: &[u8; 8], header: &[u8], records: &[(i64, f64, Vec<u8>)]) -> Result<(), String> {
    let e = |x: std::io::Error| x.to_string();
    let partial = path.with_extension("partial");
    let mut out = std::io::BufWriter::new(File::create(&partial).map_err(e)?);
    out.write_all(magic).map_err(e)?;
    out.write_all(&(header.len() as u64).to_le_bytes()).map_err(e)?;
    out.write_all(header).map_err(e)?;
    out.write_all(&(records.len() as u64).to_le_bytes()).map_err(e)?;
    let mut at = 0u64;
    for (id, rate, blob) in records {
        out.write_all(&id.to_le_bytes()).map_err(e)?;
        out.write_all(&rate.to_le_bytes()).map_err(e)?;
        out.write_all(&at.to_le_bytes()).map_err(e)?;
        out.write_all(&(blob.len() as u32).to_le_bytes()).map_err(e)?;
        at += blob.len() as u64;
    }
    records.iter().try_for_each(|r| out.write_all(&r.2)).map_err(e)?;
    out.flush().map_err(e)?;
    drop(out);
    std::fs::rename(&partial, path).map_err(e)
}

impl Blobs {
    /// The mapped file and the header's byte range.
    pub fn open(path: &Path, magic: &[u8; 8]) -> Result<(Blobs, std::ops::Range<usize>), String> {
        let file = File::open(path).map_err(|e| e.to_string())?;
        // Safety: the file is only ever replaced by rename, never changed in place.
        let map = unsafe { Mmap::map(&file) }.map_err(|e| e.to_string())?;
        if map.len() < 16 || &map[..8] != magic {
            return Err(format!("{} is not this kind of data file", path.display()));
        }
        let cut = || format!("{} is cut short", path.display());
        let header = u64::from_le_bytes(map[8..16].try_into().unwrap()) as usize;
        let end = 16usize.checked_add(header).filter(|&e| e.checked_add(8).is_some_and(|x| x <= map.len())).ok_or_else(cut)?;
        let entries = u64::from_le_bytes(map[end..end + 8].try_into().unwrap()) as usize;
        let index = end + 8;
        let blobs = entries.checked_mul(ENTRY).and_then(|n| n.checked_add(index)).filter(|&b| b <= map.len()).ok_or_else(cut)?;
        Ok((Blobs { map, index, entries, blobs }, 16..end))
    }

    /// Drop the mapped pages from RSS; the file reads them back when needed.
    pub fn release(&self) {
        #[cfg(unix)]
        // Safety: a read-only file mapping: DONTNEED only forgets clean pages.
        let _ = unsafe { self.map.unchecked_advise(memmap2::UncheckedAdvice::DontNeed) };
    }

    pub fn header(&self, range: std::ops::Range<usize>) -> &[u8] {
        &self.map[range]
    }

    fn entry(&self, i: usize) -> (i64, f64, usize, usize) {
        let e = &self.map[self.index + i * ENTRY..self.index + (i + 1) * ENTRY];
        (i64::from_le_bytes(e[..8].try_into().unwrap()), f64::from_le_bytes(e[8..16].try_into().unwrap()),
         u64::from_le_bytes(e[16..24].try_into().unwrap()) as usize, u32::from_le_bytes(e[24..28].try_into().unwrap()) as usize)
    }

    fn range(&self, id: i64) -> std::ops::Range<usize> {
        let (mut lo, mut hi) = (0, self.entries);
        while lo < hi {
            let mid = (lo + hi) / 2;
            if self.entry(mid).0 < id { lo = mid + 1 } else { hi = mid }
        }
        let start = lo;
        while lo < self.entries && self.entry(lo).0 == id {
            lo += 1;
        }
        start..lo
    }

    pub fn rates(&self, id: i64) -> Vec<f64> {
        self.range(id).map(|i| self.entry(i).1).collect()
    }

    pub fn feature(&self, id: i64, rate: f64) -> Option<Val> {
        let i = self.range(id).find(|&i| self.entry(i).1 == rate)?;
        self.blob(i)
    }

    fn blob(&self, i: usize) -> Option<Val> {
        let (_, _, at, len) = self.entry(i);
        let start = self.blobs.checked_add(at)?;
        codec::decode(self.map.get(start..start.checked_add(len)?)?).ok()
    }

    /// Every record in (id, rate) order.
    pub fn each(&self) -> impl Iterator<Item = (i64, f64, Val)> + '_ {
        (0..self.entries).filter_map(|i| {
            let (id, rate, ..) = self.entry(i);
            Some((id, rate, self.blob(i)?))
        })
    }
}
