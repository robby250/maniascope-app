//! ManiaScope difficulty calculator (Rust); Python bindings in bind.rs.
pub mod background;
pub mod bind;
pub mod card;
pub mod chart;
pub mod compute;
pub mod demand;
pub mod execution;
pub mod gestures;
pub mod kern;
pub mod model;
pub mod params;
pub mod pattern;
pub mod predict;
pub mod py;
pub mod scroll;
pub mod skills;
pub mod structure;
pub mod val;

use pyo3::prelude::*;

#[pymodule]
fn maniascope_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<predict::PredModel>()?;
    m.add_class::<predict::PredSession>()?;
    bind::register(m)
}
