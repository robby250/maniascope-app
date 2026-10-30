//! ManiaScope difficulty calculator (Rust); Python bindings in bind.rs.
pub mod background;
pub mod codec;
pub mod bind;
pub mod card;
pub mod chart;
pub mod choices;
pub mod compute;
pub mod demand;
pub mod execution;
pub mod fit;
pub mod gestures;
pub mod kern;
pub mod model;
pub mod params;
pub mod pattern;
pub mod personal;
pub mod predict;
pub mod py;
pub mod scroll;
pub mod skills;
pub mod stable;
pub mod structure;
pub mod val;

use pyo3::prelude::*;

#[pymodule]
fn maniascope_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fit::box_cd, m)?)?;
    m.add_function(wrap_pyfunction!(fit::warmup_ridge, m)?)?;
    m.add_function(wrap_pyfunction!(codec::encode_feats, m)?)?;
    m.add_function(wrap_pyfunction!(codec::decode_feats, m)?)?;
    m.add_function(wrap_pyfunction!(personal::warmup_steps, m)?)?;
    m.add_function(wrap_pyfunction!(personal::personal_fit, m)?)?;
    m.add_class::<predict::Families>()?;
    m.add_class::<predict::PredModel>()?;
    m.add_class::<predict::PredSession>()?;
    m.add_class::<predict::FeatStore>()?;
    m.add_function(wrap_pyfunction!(stable::stable_library, m)?)?;
    m.add_function(wrap_pyfunction!(stable::stable_scores, m)?)?;
    m.add_function(wrap_pyfunction!(stable::sha256_files, m)?)?;
    choices::register(m)?;
    bind::register(m)
}
