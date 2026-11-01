//! ManiaScope difficulty calculator. calc_id is a constant: moving source files must not re-analyse a library.
pub mod background;
pub mod card;
pub mod chart;
pub mod compute;
pub mod demand;
pub mod execution;
pub mod features;
pub mod flow;
pub mod gestures;
pub mod kern;
pub mod model;
pub mod params;
pub mod pattern;
pub mod py;
pub mod scroll;
pub mod skills;
pub mod structure;
pub mod val;

pub const CALC_ID: &str = "9148bd02bce2";
