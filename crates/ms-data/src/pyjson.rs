//! json.dumps' text: the event log and rec.db text columns stay byte-identical with what the Python app wrote.
use serde_json::ser::{CompactFormatter, Formatter, Serializer};
use serde_json::Value;
use std::io::{self, Write};

struct Spaced;

impl Formatter for Spaced {
    fn begin_array_value<W: ?Sized + Write>(&mut self, w: &mut W, first: bool) -> io::Result<()> {
        if first { Ok(()) } else { w.write_all(b", ") }
    }
    fn begin_object_key<W: ?Sized + Write>(&mut self, w: &mut W, first: bool) -> io::Result<()> {
        if first { Ok(()) } else { w.write_all(b", ") }
    }
    fn begin_object_value<W: ?Sized + Write>(&mut self, w: &mut W) -> io::Result<()> {
        w.write_all(b": ")
    }
    fn write_string_fragment<W: ?Sized + Write>(&mut self, w: &mut W, fragment: &str) -> io::Result<()> {
        for c in fragment.chars() {
            if c.is_ascii() {
                w.write_all(&[c as u8])?;
            } else {
                for unit in c.encode_utf16(&mut [0; 2]) {
                    write!(w, "\\u{unit:04x}")?;
                }
            }
        }
        Ok(())
    }
    fn write_f64<W: ?Sized + Write>(&mut self, w: &mut W, value: f64) -> io::Result<()> {
        CompactFormatter.write_f64(w, value)
    }
}

pub fn dumps(v: &Value) -> String {
    let mut out = Vec::new();
    serde::Serialize::serialize(v, &mut Serializer::with_formatter(&mut out, Spaced)).expect("a JSON value always serialises");
    String::from_utf8(out).expect("the formatter writes ASCII")
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn matches_python_text() {
        assert_eq!(dumps(&json!([1, 2, 3])), "[1, 2, 3]");
        assert_eq!(dumps(&json!({"a": "ü☆", "b": [1.5, null]})), r#"{"a": "\u00fc\u2606", "b": [1.5, null]}"#);
        assert_eq!(dumps(&json!("😀")), r#""\ud83d\ude00""#);
    }
}
