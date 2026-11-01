//! Ordered dynamic values for the result dict (the Python-facing shape of compute()).

#[derive(Clone, Debug, PartialEq)]
pub enum Key {
    S(String),
    F(f64),
    B(bool),
}
impl From<&str> for Key {
    fn from(s: &str) -> Key {
        Key::S(s.to_string())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Val {
    None,
    Bool(bool),
    Int(i64),
    F(f64),
    Str(String),
    List(Vec<Val>),
    Tuple(Vec<Val>),
    Dict(Dict),
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Dict(pub Vec<(Key, Val)>);

impl Dict {
    pub fn new() -> Dict {
        Dict(Vec::new())
    }
    pub fn get(&self, k: &str) -> Option<&Val> {
        self.0.iter().find(|(kk, _)| matches!(kk, Key::S(s) if s == k)).map(|(_, v)| v)
    }
    pub fn get_mut(&mut self, k: &str) -> Option<&mut Val> {
        self.0.iter_mut().find(|(kk, _)| matches!(kk, Key::S(s) if s == k)).map(|(_, v)| v)
    }
    pub fn has(&self, k: &str) -> bool {
        self.get(k).is_some()
    }
    /// Python `d[k] = v`: replace in place, else append.
    pub fn set(&mut self, k: &str, v: impl Into<Val>) {
        let v = v.into();
        match self.get_mut(k) {
            Some(slot) => *slot = v,
            None => self.0.push((Key::S(k.to_string()), v)),
        }
    }
    pub fn setb(&mut self, k: bool, v: impl Into<Val>) {
        self.0.push((Key::B(k), v.into()));
    }
    pub fn setf(&mut self, k: f64, v: impl Into<Val>) {
        let v = v.into();
        match self.0.iter_mut().find(|(kk, _)| matches!(kk, Key::F(x) if *x == k)) {
            Some((_, slot)) => *slot = v,
            None => self.0.push((Key::F(k), v)),
        }
    }
    pub fn pop(&mut self, k: &str) -> Option<Val> {
        let i = self.0.iter().position(|(kk, _)| matches!(kk, Key::S(s) if s == k))?;
        Some(self.0.remove(i).1)
    }
    pub fn f(&self, k: &str) -> f64 {
        self.get(k).map_or(0.0, |v| v.f())
    }
    pub fn fget(&self, k: &str, default: f64) -> f64 {
        self.get(k).map_or(default, |v| v.f())
    }
    pub fn dict(&self, k: &str) -> Option<&Dict> {
        match self.get(k) {
            Some(Val::Dict(d)) => Some(d),
            _ => None,
        }
    }
    pub fn dict_mut(&mut self, k: &str) -> Option<&mut Dict> {
        match self.get_mut(k) {
            Some(Val::Dict(d)) => Some(d),
            _ => None,
        }
    }
    /// Values of a float-keyed dict.
    pub fn values_f(&self) -> Vec<f64> {
        self.0.iter().map(|(_, v)| v.f()).collect()
    }
    pub fn map_values(&mut self, f: impl Fn(f64) -> f64) {
        for (_, v) in &mut self.0 {
            *v = Val::F(f(v.f()));
        }
    }
}

impl Val {
    pub fn f(&self) -> f64 {
        match self {
            Val::F(x) => *x,
            Val::Int(i) => *i as f64,
            Val::Bool(b) => *b as u8 as f64,
            _ => panic!("not a number: {self:?}"),
        }
    }
    pub fn is_none(&self) -> bool {
        matches!(self, Val::None)
    }
    pub fn list_f(&self) -> Vec<f64> {
        match self {
            Val::List(v) | Val::Tuple(v) => v.iter().map(|x| x.f()).collect(),
            _ => panic!("not a list"),
        }
    }
    pub fn truthy(&self) -> bool {
        match self {
            Val::None => false,
            Val::Bool(b) => *b,
            Val::Int(i) => *i != 0,
            Val::F(x) => *x != 0.0,
            Val::Str(s) => !s.is_empty(),
            Val::List(v) | Val::Tuple(v) => !v.is_empty(),
            Val::Dict(d) => !d.0.is_empty(),
        }
    }
    pub fn floats(v: &[f64]) -> Val {
        Val::List(v.iter().map(|&x| Val::F(x)).collect())
    }
    pub fn strs(v: &[&str]) -> Val {
        Val::Tuple(v.iter().map(|&x| Val::Str(x.to_string())).collect())
    }
}

impl From<f64> for Val {
    fn from(x: f64) -> Val {
        Val::F(x)
    }
}
impl From<i64> for Val {
    fn from(x: i64) -> Val {
        Val::Int(x)
    }
}
impl From<usize> for Val {
    fn from(x: usize) -> Val {
        Val::Int(x as i64)
    }
}
impl From<bool> for Val {
    fn from(x: bool) -> Val {
        Val::Bool(x)
    }
}
impl From<&str> for Val {
    fn from(x: &str) -> Val {
        Val::Str(x.to_string())
    }
}
impl From<String> for Val {
    fn from(x: String) -> Val {
        Val::Str(x)
    }
}
impl From<Dict> for Val {
    fn from(x: Dict) -> Val {
        Val::Dict(x)
    }
}
impl From<Vec<f64>> for Val {
    fn from(x: Vec<f64>) -> Val {
        Val::floats(&x)
    }
}
