//! Binary form of stored feature dicts (rec.db `feats`): JSON's types, ~2× smaller than its text and decoded
//! with interned keys. Not part of the calculator identity (native_backend.identity_files).
//!
//! Layout: one tagged value; integers and lengths little-endian. Tags: 0 None, 1 False, 2 True, 3 int (i64),
//! 4 float (f64), 5 str (u32 length + UTF-8), 6 list (u32 count + values), 7 dict (u32 count + (key, value)),
//! a key being u16 length + UTF-8. Tuples are written as lists, as JSON would.
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyBytes, PyDict, PyFloat, PyInt, PyList, PyString, PyTuple};
use std::cell::RefCell;
use std::collections::HashMap;

const MAGIC: u8 = 0xB7; // first byte: never the start of a JSON text

fn put(v: &Bound<'_, PyAny>, out: &mut Vec<u8>) -> PyResult<()> {
    if v.is_none() {
        out.push(0);
    } else if let Ok(b) = v.cast::<PyBool>() {
        out.push(if b.is_true() { 2 } else { 1 });
    } else if let Ok(i) = v.cast::<PyInt>() {
        out.push(3);
        out.extend_from_slice(&i.extract::<i64>()?.to_le_bytes());
    } else if let Ok(f) = v.cast::<PyFloat>() {
        out.push(4);
        out.extend_from_slice(&f.value().to_le_bytes());
    } else if let Ok(s) = v.cast::<PyString>() {
        let s = s.to_str()?;
        out.push(5);
        out.extend_from_slice(&(s.len() as u32).to_le_bytes());
        out.extend_from_slice(s.as_bytes());
    } else if let Ok(d) = v.cast::<PyDict>() {
        out.push(7);
        out.extend_from_slice(&(d.len() as u32).to_le_bytes());
        for (k, x) in d.iter() {
            // Non-string keys as json.dumps writes them.
            let owned;
            let k = if let Ok(k) = k.cast::<PyString>() {
                k.to_str()?
            } else if k.is_none() {
                "null"
            } else if let Ok(b) = k.cast::<PyBool>() {
                if b.is_true() { "true" } else { "false" }
            } else if k.cast::<PyInt>().is_ok() || k.cast::<PyFloat>().is_ok() {
                owned = k.repr()?.to_string();
                owned.as_str()
            } else {
                return Err(PyValueError::new_err("feature keys must be strings or numbers"));
            };
            out.extend_from_slice(&u16::try_from(k.len()).map_err(|_| PyValueError::new_err("key too long"))?.to_le_bytes());
            out.extend_from_slice(k.as_bytes());
            put(&x, out)?;
        }
    } else if v.cast::<PyList>().is_ok() || v.cast::<PyTuple>().is_ok() {
        out.push(6);
        out.extend_from_slice(&(v.len()? as u32).to_le_bytes());
        for x in v.try_iter()? {
            put(&x?, out)?;
        }
    } else if let Ok(x) = v.extract::<f64>() {
        // numpy scalars and other numbers: json.dumps writes them as floats
        out.push(4);
        out.extend_from_slice(&x.to_le_bytes());
    } else {
        return Err(PyValueError::new_err(format!("cannot store {}", v.get_type().name()?)));
    }
    Ok(())
}

/// A feature dict (JSON types) → its binary form.
#[pyfunction]
pub fn encode_feats<'py>(py: Python<'py>, f: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyBytes>> {
    let mut out = vec![MAGIC];
    put(f, &mut out)?;
    Ok(PyBytes::new(py, &out))
}

struct Reader<'a> {
    b: &'a [u8],
    at: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> PyResult<&'a [u8]> {
        let s = self.b.get(self.at..self.at + n).ok_or_else(|| PyValueError::new_err("truncated feature row"))?;
        self.at += n;
        Ok(s)
    }
    fn u32(&mut self) -> PyResult<usize> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()) as usize)
    }
}

thread_local! {
    // Feature dicts repeat a few hundred keys: share one str object per key.
    static KEYS: RefCell<HashMap<Vec<u8>, Py<PyString>>> = RefCell::new(HashMap::new());
}

fn get<'py>(py: Python<'py>, r: &mut Reader<'_>) -> PyResult<Bound<'py, PyAny>> {
    let tag = r.take(1)?[0];
    Ok(match tag {
        0 => py.None().into_bound(py),
        1 => PyBool::new(py, false).to_owned().into_any(),
        2 => PyBool::new(py, true).to_owned().into_any(),
        3 => i64::from_le_bytes(r.take(8)?.try_into().unwrap()).into_pyobject(py)?.into_any(),
        4 => PyFloat::new(py, f64::from_le_bytes(r.take(8)?.try_into().unwrap())).into_any(),
        5 => {
            let n = r.u32()?;
            let s = std::str::from_utf8(r.take(n)?).map_err(|e| PyValueError::new_err(e.to_string()))?;
            PyString::new(py, s).into_any()
        }
        6 => {
            let n = r.u32()?;
            let items = (0..n).map(|_| get(py, r)).collect::<PyResult<Vec<_>>>()?;
            PyList::new(py, items)?.into_any()
        }
        7 => {
            let n = r.u32()?;
            let d = PyDict::new(py);
            for _ in 0..n {
                let len = u16::from_le_bytes(r.take(2)?.try_into().unwrap()) as usize;
                let raw = r.take(len)?;
                let key = KEYS.with(|keys| -> PyResult<Py<PyString>> {
                    let mut keys = keys.borrow_mut();
                    if let Some(k) = keys.get(raw) {
                        return Ok(k.clone_ref(py));
                    }
                    let s = std::str::from_utf8(raw).map_err(|e| PyValueError::new_err(e.to_string()))?;
                    let k = PyString::intern(py, s).unbind();
                    keys.insert(raw.to_vec(), k.clone_ref(py));
                    Ok(k)
                })?;
                d.set_item(key.bind(py), get(py, r)?)?;
            }
            d.into_any()
        }
        t => return Err(PyValueError::new_err(format!("bad feature tag {t}"))),
    })
}

/// encode_feats' bytes → the feature dict.
#[pyfunction]
pub fn decode_feats<'py>(py: Python<'py>, data: &[u8]) -> PyResult<Bound<'py, PyAny>> {
    if data.first() != Some(&MAGIC) {
        return Err(PyValueError::new_err("not a binary feature row"));
    }
    let mut r = Reader { b: data, at: 1 };
    let v = get(py, &mut r)?;
    if r.at != data.len() {
        return Err(PyValueError::new_err("trailing bytes in feature row"));
    }
    Ok(v)
}
