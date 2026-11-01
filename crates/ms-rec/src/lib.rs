//! The recommender: evidence from the player's scores, the fitted models, lists.
pub mod builder;
pub mod catalog;
pub mod converts;
pub mod dans;
pub mod engine;
pub mod evidence;
pub mod history;
pub mod navigation;
pub mod pp;
pub mod recommender;
pub mod stats;
pub mod worker;

#[cfg(test)]
extern crate self as ms_rec;
#[cfg(test)]
#[allow(dead_code)]
mod testkit;
