//! Binary form of stored feature dicts (rec.db `feats`), byte-compatible with the Python app's: one tagged value;
//! integers and lengths little-endian. Tags: 0 None, 1 False, 2 True, 3 i64, 4 f64, 5 str (u32 + UTF-8),
//! 6 list (u32 + values), 7 dict (u32 + (u16 key length + UTF-8, value)). Tuples are written as lists.
use ms_calc::val::{Dict, Key, Val};

const MAGIC: u8 = 0xB7; // first byte: never the start of a JSON text

fn put(v: &Val, out: &mut Vec<u8>) {
    match v {
        Val::None => out.push(0),
        Val::Bool(false) => out.push(1),
        Val::Bool(true) => out.push(2),
        Val::Int(i) => {
            out.push(3);
            out.extend_from_slice(&i.to_le_bytes());
        }
        Val::F(x) => {
            out.push(4);
            out.extend_from_slice(&x.to_le_bytes());
        }
        Val::Str(s) => {
            out.push(5);
            out.extend_from_slice(&(s.len() as u32).to_le_bytes());
            out.extend_from_slice(s.as_bytes());
        }
        Val::List(items) | Val::Tuple(items) => {
            out.push(6);
            out.extend_from_slice(&(items.len() as u32).to_le_bytes());
            items.iter().for_each(|x| put(x, out));
        }
        Val::Dict(d) => {
            out.push(7);
            out.extend_from_slice(&(d.0.len() as u32).to_le_bytes());
            for (k, x) in &d.0 {
                // JSON's dict keys are strings; a float key takes the text json.dumps gave it.
                let key = match k {
                    Key::S(s) => s.clone(),
                    Key::F(f) => format!("{f:?}"),
                    Key::B(b) => if *b { "true".into() } else { "false".into() },
                };
                out.extend_from_slice(&(key.len() as u16).to_le_bytes());
                out.extend_from_slice(key.as_bytes());
                put(x, out);
            }
        }
    }
}

pub fn encode(v: &Val) -> Vec<u8> {
    let mut out = vec![MAGIC];
    put(v, &mut out);
    out
}

struct Reader<'a> {
    b: &'a [u8],
    at: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        let end = self.at.checked_add(n).filter(|&e| e <= self.b.len()).ok_or("truncated feature row")?;
        let out = &self.b[self.at..end];
        self.at = end;
        Ok(out)
    }
    fn u32(&mut self) -> Result<usize, String> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()) as usize)
    }
    fn value(&mut self) -> Result<Val, String> {
        let tag = self.take(1)?[0];
        Ok(match tag {
            0 => Val::None,
            1 => Val::Bool(false),
            2 => Val::Bool(true),
            3 => Val::Int(i64::from_le_bytes(self.take(8)?.try_into().unwrap())),
            4 => Val::F(f64::from_le_bytes(self.take(8)?.try_into().unwrap())),
            5 => {
                let n = self.u32()?;
                Val::Str(std::str::from_utf8(self.take(n)?).map_err(|e| e.to_string())?.to_owned())
            }
            6 => {
                let n = self.u32()?;
                Val::List((0..n).map(|_| self.value()).collect::<Result<_, _>>()?)
            }
            7 => {
                let n = self.u32()?;
                let mut d = Vec::with_capacity(n);
                for _ in 0..n {
                    let len = u16::from_le_bytes(self.take(2)?.try_into().unwrap()) as usize;
                    let key = std::str::from_utf8(self.take(len)?).map_err(|e| e.to_string())?.to_owned();
                    d.push((Key::S(key), self.value()?));
                }
                Val::Dict(Dict(d))
            }
            t => return Err(format!("bad feature tag {t}")),
        })
    }
}

pub fn decode(data: &[u8]) -> Result<Val, String> {
    if data.first() != Some(&MAGIC) {
        return Err("not a binary feature row".into());
    }
    let mut r = Reader { b: data, at: 1 };
    let v = r.value()?;
    if r.at != data.len() {
        return Err("trailing bytes in feature row".into());
    }
    Ok(v)
}

/// serde_json value → Val (rows written before the binary form are JSON text).
pub fn from_json(v: &serde_json::Value) -> Val {
    use serde_json::Value as J;
    match v {
        J::Null => Val::None,
        J::Bool(b) => Val::Bool(*b),
        J::Number(n) => n.as_i64().map_or_else(|| Val::F(n.as_f64().unwrap_or(f64::NAN)), Val::Int),
        J::String(s) => Val::Str(s.clone()),
        J::Array(a) => Val::List(a.iter().map(from_json).collect()),
        J::Object(o) => Val::Dict(Dict(o.iter().map(|(k, x)| (Key::S(k.clone()), from_json(x))).collect())),
    }
}

/// Val → serde_json value (feature dicts as JSON).
pub fn to_json(v: &Val) -> serde_json::Value {
    use serde_json::Value as J;
    match v {
        Val::None => J::Null,
        Val::Bool(b) => J::Bool(*b),
        Val::Int(i) => J::from(*i),
        Val::F(x) => J::from(*x),
        Val::Str(s) => J::from(s.as_str()),
        Val::List(a) | Val::Tuple(a) => J::Array(a.iter().map(to_json).collect()),
        Val::Dict(d) => J::Object(d.0.iter().map(|(k, x)| (match k {
            Key::S(s) => s.clone(),
            Key::F(f) => format!("{f:?}"),
            Key::B(b) => b.to_string(),
        }, to_json(x))).collect()),
    }
}

/// A rec.db `feats` cell: binary, or JSON text from before it.
pub fn load(cell: &[u8]) -> Result<Val, String> {
    if cell.first() == Some(&MAGIC) {
        return decode(cell);
    }
    serde_json::from_slice::<serde_json::Value>(cell).map(|v| from_json(&v)).map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip() {
        let mut d = Dict::new();
        d.set("overall", 5.5);
        d.set("keys", Val::Int(7));
        d.set("sk", Val::List(vec![Val::None, Val::Bool(true), Val::Str("ü".into())]));
        let v = Val::Dict(d);
        assert_eq!(decode(&encode(&v)).unwrap(), v);
        assert!(decode(b"{}").is_err() && load(b"{\"a\": 1}").is_ok());
    }

    #[test]
    fn reads_the_python_bytes() {
        // recdata.feats_dump({'a': 1, 'b': [1.5, None]})
        let raw: &[u8] = &[0xB7, 7, 2, 0, 0, 0, 1, 0, b'a', 3, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, b'b', 6, 2, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0xF8, 0x3F, 0];
        let v = decode(raw).unwrap();
        let Val::Dict(d) = v else { panic!() };
        assert_eq!(d.get("a"), Some(&Val::Int(1)));
        assert_eq!(d.get("b"), Some(&Val::List(vec![Val::F(1.5), Val::None])));
    }
}
