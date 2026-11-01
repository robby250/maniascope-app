//! ManiaScope's store: rec.db, imports, public evidence, paths. Schema and file formats match the Python app's, so
//! an existing data folder needs no migration.
pub mod blobs;
pub mod codec;
pub mod events;
pub mod feedback;
pub mod imports;
pub mod lazer_index;
pub mod paths;
pub mod pickle;
pub mod public;
pub mod pyjson;
pub mod realm;
pub mod replays;
pub mod stable;
pub mod store;
pub mod tools;
pub mod webmaps;
