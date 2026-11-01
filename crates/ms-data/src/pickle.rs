//! Reads the plain-data pickles of the Python app (public.pkl, web.pkl): dicts, lists, tuples, numbers, strings.
//! A global or any other opcode is an error: these files come from a download, and only data may load.
use std::rc::Rc;

#[derive(Clone, Debug, PartialEq)]
pub enum P {
    None,
    Bool(bool),
    Int(i64),
    F(f64),
    Str(Rc<str>),
    Bytes(Rc<[u8]>),
    List(Rc<Vec<P>>),
    Tuple(Rc<Vec<P>>),
    Dict(Rc<Vec<(P, P)>>),
}

impl P {
    pub fn f(&self) -> Option<f64> {
        match self {
            P::F(x) => Some(*x),
            P::Int(i) => Some(*i as f64),
            P::Bool(b) => Some(f64::from(u8::from(*b))),
            _ => None,
        }
    }

    pub fn i(&self) -> Option<i64> {
        match self {
            P::Int(i) => Some(*i),
            _ => None,
        }
    }

    pub fn s(&self) -> Option<&str> {
        match self {
            P::Str(s) => Some(s),
            _ => None,
        }
    }

    pub fn items(&self) -> &[(P, P)] {
        match self {
            P::Dict(d) => d,
            _ => &[],
        }
    }

    pub fn seq(&self) -> &[P] {
        match self {
            P::List(v) | P::Tuple(v) => v,
            _ => &[],
        }
    }

    pub fn get(&self, key: &str) -> Option<&P> {
        self.items().iter().find(|(k, _)| k.s() == Some(key)).map(|(_, v)| v)
    }
}

/// A list or dict under construction: appended to in place, frozen when the next operation needs it as a value.
enum Slot {
    Value(P),
    List(Vec<P>, Option<usize>), // memo index
    Dict(Vec<(P, P)>, Option<usize>),
    Mark,
}

struct Machine<'a> {
    b: &'a [u8],
    at: usize,
    stack: Vec<Slot>,
    memo: Vec<Option<P>>,
}

impl<'a> Machine<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        let end = self.at.checked_add(n).filter(|&e| e <= self.b.len()).ok_or("truncated pickle")?;
        let out = &self.b[self.at..end];
        self.at = end;
        Ok(out)
    }

    fn le<const N: usize>(&mut self) -> Result<[u8; N], String> {
        Ok(self.take(N)?.try_into().unwrap())
    }

    fn text(&mut self, n: usize) -> Result<P, String> {
        let raw = self.take(n)?;
        Ok(P::Str(Rc::from(std::str::from_utf8(raw).map_err(|e| e.to_string())?)))
    }

    fn push(&mut self, v: P) {
        self.stack.push(Slot::Value(v));
    }

    /// A finished value off the stack; a finished container is memoised as it closes.
    fn pop(&mut self) -> Result<P, String> {
        match self.stack.pop().ok_or("pickle stack empty")? {
            Slot::Value(v) => Ok(v),
            Slot::List(v, memo) => {
                let out = P::List(Rc::new(v));
                memo.iter().for_each(|&m| self.memo[m] = Some(out.clone()));
                Ok(out)
            }
            Slot::Dict(v, memo) => {
                let out = P::Dict(Rc::new(v));
                memo.iter().for_each(|&m| self.memo[m] = Some(out.clone()));
                Ok(out)
            }
            Slot::Mark => Err("mark where a value belongs".into()),
        }
    }

    /// Everything above the last mark, in stack order.
    fn to_mark(&mut self) -> Result<Vec<P>, String> {
        let mark = self.stack.iter().rposition(|s| matches!(s, Slot::Mark)).ok_or("no mark")?;
        let mut out = Vec::with_capacity(self.stack.len() - mark - 1);
        while self.stack.len() > mark + 1 {
            out.push(self.pop()?);
        }
        self.stack.pop();
        out.reverse();
        Ok(out)
    }

    /// A memo reference. A container memoised earlier may still sit on the stack, finished and waiting for its
    /// parent's SETITEMS (a dict shared by two keys): it freezes here. Nothing in this data refers to itself.
    fn get(&mut self, at: usize) -> Result<P, String> {
        if let Some(Some(v)) = self.memo.get(at) {
            return Ok(v.clone());
        }
        let is = |s: &Slot| matches!(s, Slot::List(_, Some(m)) | Slot::Dict(_, Some(m)) if *m == at);
        let slot = self.stack.iter().rposition(is).ok_or("memo reference to an unknown value")?;
        let out = match std::mem::replace(&mut self.stack[slot], Slot::Mark) {
            Slot::List(v, _) => P::List(Rc::new(v)),
            Slot::Dict(v, _) => P::Dict(Rc::new(v)),
            _ => unreachable!(),
        };
        self.stack[slot] = Slot::Value(out.clone());
        self.memo[at] = Some(out.clone());
        Ok(out)
    }
}

pub fn load(data: &[u8]) -> Result<P, String> {
    let mut m = Machine { b: data, at: 0, stack: Vec::new(), memo: Vec::new() };
    loop {
        let op = m.take(1)?[0];
        match op {
            0x80 => { m.take(1)?; }                                           // PROTO
            0x95 => { m.take(8)?; }                                           // FRAME
            b'.' => return m.pop(),                                           // STOP
            b'N' => m.push(P::None),
            0x88 => m.push(P::Bool(true)),
            0x89 => m.push(P::Bool(false)),
            b'K' => { let v = m.take(1)?[0]; m.push(P::Int(i64::from(v))) }   // BININT1
            b'M' => { let v = u16::from_le_bytes(m.le()?); m.push(P::Int(i64::from(v))) }
            b'J' => { let v = i32::from_le_bytes(m.le()?); m.push(P::Int(i64::from(v))) }
            0x8a => {                                                         // LONG1
                let n = m.take(1)?[0] as usize;
                if n > 8 {
                    return Err("integer over 64 bits in pickle".into());
                }
                let raw = m.take(n)?;
                let mut wide = [if raw.last().is_some_and(|b| b & 0x80 != 0) { 0xFF } else { 0 }; 8];
                wide[..n].copy_from_slice(raw);
                m.push(P::Int(i64::from_le_bytes(wide)));
            }
            b'G' => { let v = f64::from_be_bytes(m.le()?); m.push(P::F(v)) }  // BINFLOAT
            0x8c => { let n = m.take(1)?[0] as usize; let v = m.text(n)?; m.push(v) } // SHORT_BINUNICODE
            b'X' => { let n = u32::from_le_bytes(m.le()?) as usize; let v = m.text(n)?; m.push(v) }
            0x8d => { let n = u64::from_le_bytes(m.le()?) as usize; let v = m.text(n)?; m.push(v) }
            b'C' => { let n = m.take(1)?[0] as usize; let v = m.take(n)?; m.push(P::Bytes(Rc::from(v))) }   // SHORT_BINBYTES
            b'B' => { let n = u32::from_le_bytes(m.le()?) as usize; let v = m.take(n)?; m.push(P::Bytes(Rc::from(v))) }
            0x8e => { let n = u64::from_le_bytes(m.le()?) as usize; let v = m.take(n)?; m.push(P::Bytes(Rc::from(v))) }
            b']' => m.stack.push(Slot::List(Vec::new(), None)),
            b'}' => m.stack.push(Slot::Dict(Vec::new(), None)),
            b')' => m.push(P::Tuple(Rc::new(Vec::new()))),
            b'(' => m.stack.push(Slot::Mark),
            0x85..=0x87 => {                                                  // TUPLE1..3
                let n = (op - 0x84) as usize;
                let mut v = Vec::with_capacity(n);
                for _ in 0..n {
                    v.push(m.pop()?);
                }
                v.reverse();
                m.push(P::Tuple(Rc::new(v)));
            }
            b't' => { let v = m.to_mark()?; m.push(P::Tuple(Rc::new(v))) }
            b'l' => { let v = m.to_mark()?; m.push(P::List(Rc::new(v))) }
            b'd' => {
                let mut v = m.to_mark()?.into_iter();
                let mut pairs = Vec::new();
                while let (Some(a), Some(b)) = (v.next(), v.next()) {
                    pairs.push((a, b));
                }
                m.push(P::Dict(Rc::new(pairs)));
            }
            0x94 => {                                                         // MEMOIZE
                let at = m.memo.len();
                m.memo.push(None);
                match m.stack.last_mut().ok_or("memoize on an empty stack")? {
                    Slot::Value(v) => m.memo[at] = Some(v.clone()),
                    Slot::List(_, memo) | Slot::Dict(_, memo) => *memo = Some(at),
                    Slot::Mark => return Err("memoize of a mark".into()),
                }
            }
            b'q' => {                                                         // BINPUT
                let at = m.take(1)?[0] as usize;
                put(&mut m, at)?;
            }
            b'r' => {
                let at = u32::from_le_bytes(m.le()?) as usize;
                put(&mut m, at)?;
            }
            b'h' => { let at = m.take(1)?[0] as usize; let v = m.get(at)?; m.push(v) } // BINGET
            b'j' => { let at = u32::from_le_bytes(m.le()?) as usize; let v = m.get(at)?; m.push(v) }
            b'a' => {                                                         // APPEND
                let v = m.pop()?;
                match m.stack.last_mut() {
                    Some(Slot::List(l, _)) => l.push(v),
                    _ => return Err("append outside a list".into()),
                }
            }
            b'e' => {                                                         // APPENDS
                let items = m.to_mark()?;
                match m.stack.last_mut() {
                    Some(Slot::List(l, _)) => l.extend(items),
                    _ => return Err("appends outside a list".into()),
                }
            }
            b's' => {                                                         // SETITEM
                let (v, k) = (m.pop()?, m.pop()?);
                match m.stack.last_mut() {
                    Some(Slot::Dict(d, _)) => d.push((k, v)),
                    _ => return Err("setitem outside a dict".into()),
                }
            }
            b'u' => {                                                         // SETITEMS
                let mut items = m.to_mark()?.into_iter();
                match m.stack.last_mut() {
                    Some(Slot::Dict(d, _)) => {
                        while let (Some(k), Some(v)) = (items.next(), items.next()) {
                            d.push((k, v));
                        }
                    }
                    _ => return Err("setitems outside a dict".into()),
                }
            }
            other => return Err(format!("pickle opcode {other:#x} is not plain data")),
        }
    }
}

fn put(m: &mut Machine<'_>, at: usize) -> Result<(), String> {
    if m.memo.len() <= at {
        m.memo.resize(at + 1, None);
    }
    match m.stack.last_mut().ok_or("put on an empty stack")? {
        Slot::Value(v) => m.memo[at] = Some(v.clone()),
        Slot::List(_, memo) | Slot::Dict(_, memo) => *memo = Some(at),
        Slot::Mark => return Err("put of a mark".into()),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_data_and_shared_references() {
        // pickle.dumps({'a': [1, 2.5, None, True], 3: ('x', 'x'), 0.5: {'k': -1}, 'big': 2**40}, protocol=4)
        let raw = b"\x80\x04\x95\x38\x00\x00\x00\x00\x00\x00\x00}\x94(\x8c\x01a\x94]\x94(K\x01G@\x04\x00\x00\x00\x00\x00\x00N\x88eK\x03\x8c\x01x\x94h\x03\x86\x94G?\xe0\x00\x00\x00\x00\x00\x00}\x94\x8c\x01k\x94J\xff\xff\xff\xffs\x8c\x03big\x94\x8a\x06\x00\x00\x00\x00\x00\x01u.";
        let p = load(raw).unwrap();
        assert_eq!(p.get("a").unwrap().seq()[1], P::F(2.5));
        assert_eq!(p.items()[1].1.seq()[0], p.items()[1].1.seq()[1]);
        assert_eq!(p.items()[2].0, P::F(0.5));
        assert_eq!(p.items()[2].1.get("k"), Some(&P::Int(-1)));
        assert_eq!(p.get("big"), Some(&P::Int(1 << 40)));
    }

    #[test]
    fn code_is_refused() {
        // pickle.dumps(os.system)
        assert!(load(b"\x80\x04\x95\x10\x00\x00\x00\x00\x00\x00\x00\x8c\x05posix\x94\x8c\x06system\x94\x93\x94.").is_err());
    }
}
