//! Calculator knobs: module constants (overridable by name for calibration trials) and the fitted
//! stages from calib/difficulty.json.
use serde_json::Value;

pub trait Knob: Sized {
    fn from_value(v: &Value) -> Option<Self>;
}
impl Knob for f64 {
    fn from_value(v: &Value) -> Option<Self> {
        v.as_f64()
    }
}
impl Knob for usize {
    fn from_value(v: &Value) -> Option<Self> {
        v.as_u64().map(|x| x as usize)
    }
}
impl<const N: usize> Knob for [f64; N] {
    fn from_value(v: &Value) -> Option<Self> {
        let a = v.as_array()?;
        if a.len() != N {
            return None;
        }
        let mut out = [0.0; N];
        for (o, x) in out.iter_mut().zip(a) {
            *o = x.as_f64()?;
        }
        Some(out)
    }
}
impl<const N: usize> Knob for [(f64, f64); N] {
    fn from_value(v: &Value) -> Option<Self> {
        let a = v.as_array()?;
        if a.len() != N {
            return None;
        }
        let mut out = [(0.0, 0.0); N];
        for (o, x) in out.iter_mut().zip(a) {
            let p = x.as_array()?;
            *o = (p.first()?.as_f64()?, p.get(1)?.as_f64()?);
        }
        Some(out)
    }
}

pub trait Scalar {
    fn scalar(&self) -> Option<f64>;
    fn json(&self) -> Value;
}
impl Scalar for f64 {
    fn scalar(&self) -> Option<f64> {
        Some(*self)
    }
    fn json(&self) -> Value {
        Value::from(*self)
    }
}
impl Scalar for usize {
    fn scalar(&self) -> Option<f64> {
        Some(*self as f64)
    }
    fn json(&self) -> Value {
        Value::from(*self)
    }
}
impl<const N: usize> Scalar for [f64; N] {
    fn scalar(&self) -> Option<f64> {
        None
    }
    fn json(&self) -> Value {
        Value::from(self.to_vec())
    }
}
impl<const N: usize> Scalar for [(f64, f64); N] {
    fn scalar(&self) -> Option<f64> {
        None
    }
    fn json(&self) -> Value {
        Value::from(self.iter().map(|&(a, b)| Value::from(vec![a, b])).collect::<Vec<Value>>())
    }
}

macro_rules! knobs {
    ($($name:ident : $ty:ty = $val:expr),* $(,)?) => {
        #[allow(non_snake_case)]
        #[derive(Clone, Debug)]
        pub struct Knobs { $(pub $name: $ty,)* }
        impl Default for Knobs {
            fn default() -> Self { Knobs { $($name: $val,)* } }
        }
        impl Knobs {
            pub fn set(&mut self, name: &str, v: &Value) -> Result<(), String> {
                match name {
                    $(stringify!($name) => {
                        self.$name = <$ty as Knob>::from_value(v)
                            .ok_or_else(|| format!("bad value for {}: {}", name, v))?;
                        Ok(())
                    })*
                    _ => Err(format!("unknown knob: {}", name)),
                }
            }
            /// Every knob as JSON (name order of the definition).
            pub fn dump(&self) -> Vec<(&'static str, Value)> {
                vec![$((stringify!($name), self.$name.json()),)*]
            }
            /// A scalar knob's value (None: unknown or not scalar).
            pub fn get(&self, name: &str) -> Option<f64> {
                match name {
                    $(stringify!($name) => self.$name.scalar(),)*
                    _ => None,
                }
            }
        }
    };
}

knobs! {
    // skill_calc
    SCALE: f64 = 2.2,
    STAR_A: f64 = 0.1743,
    STAR_B: f64 = 1.095,
    W_HAND: f64 = 1.0,
    V_FAST_KEYS: f64 = 2.0,
    V_FAST: f64 = 12.0,
    SPEED_P: f64 = 1.5,
    W_JACK: f64 = 3.3,
    W_JACK_7K: f64 = 3.0,
    JACK_P: f64 = 0.5,
    W_IJACK: f64 = 0.4,
    JACK_SHARE: f64 = 0.7,
    MIRROR: f64 = 0.25,
    W_SPLIT: f64 = 1.5,
    W_ALT: f64 = 0.75,
    W_CROSS: f64 = 0.15,
    W_CROSS_WIDE: f64 = 0.05,
    W_TECH: f64 = 0.3,
    W_SV: f64 = 2.8,
    W_REL: f64 = 0.50,
    W_STAG: f64 = 0.30,
    W_HOLD_SAME: f64 = 0.30,
    W_HOLD_OTHER: f64 = 0.08,
    W_REPRESS: f64 = 0.30,
    LN_BASE: f64 = 10.0,
    ROW_SHARE: f64 = 0.3,
    FLAM: f64 = 0.045,
    V_FINGER: f64 = 8.0,
    HORIZONS: [(f64, f64); 4] = [(2.0, 1.75), (8.0, 1.30), (30.0, 1.12), (120.0, 1.00)],
    BIN: f64 = 0.5,
    PHASES: usize = 8,
    H_WEIGHTS: [f64; 4] = [0.0, 0.2, 0.25, 0.35],
    RATE_G: f64 = 0.12,
    G_NOTES: f64 = -0.03,
    UNIFORM_WIN: f64 = 240.0,
    UNIFORM_R0: f64 = 0.95,
    UNIFORM_W: f64 = 0.3,
    TECH_FULL: f64 = 0.09,
    RT_WIN: usize = 4,
    RT_FULL: f64 = 0.2,
    COLUMN_ORIGIN: f64 = 0.0, // 1: column-run bins count from the first note, as the demand does
    OHT_FULL: f64 = 0.5,
    ANC_FULL: f64 = 0.5,
    PT_FULL: f64 = 0.05,
    PT_CHAIN: f64 = 1.5,
    ROLL_MIN: usize = 6,
    R_EVENT: f64 = 0.025,
    DELAY_GAP: f64 = 0.07,
    TRILL_GAP: f64 = 0.2,
    BRACKET_FULL: f64 = 0.4,
    BRACKET_IL_FULL: f64 = 0.45,
    DUMP_FULL: f64 = 0.5,
    MJ_BURST: usize = 2,
    MJ_FULL: f64 = 0.2,
    VIBRO_GAP: f64 = 0.095,
    VIBRO_MIN: usize = 5,
    VIB_FULL: f64 = 0.3,
    LJ_FULL: f64 = 0.5,
    SHIELD_FULL: f64 = 0.5,
    INV_FULL: f64 = 1.5,
    HYB_FULL: f64 = 0.8,
    REL_FULL: f64 = 0.6,
    STAMINA_WINDOW: f64 = 240.0,
    SV_KIND_SHARE: f64 = 0.5,
    REFINE_Q: f64 = 0.9,
    ALSO_SHARE: f64 = 0.6,
    ALSO_SPECIAL: f64 = 0.4,
    ARCH_MATERIAL: f64 = 0.6,
    ARCH_TECH: f64 = 0.75,
    TECH_WHOLE: f64 = 0.85,
    STAMINA_SUSTAIN: f64 = 0.9,
    SV_PIECES: usize = 3,
    W_WOB: f64 = 2.0,
    SV_VIS: f64 = 450.0,
}

/// Components an ablation can switch off (overrides {"off": [...]}): each then contributes its neutral
/// value and every stage reading it still runs. "fit:<section>" drops a fitted stage of difficulty.json.
pub const COMPONENTS: [&str; 8] =
    ["sv_reading", "gestures", "execution", "quadstream", "tech_profile", "archetype_profile", "rolled_execution", "endurance"];
pub const FITTED: [&str; 7] = ["modes", "execution", "gestures", "scroll_reading", "structural_residual", "structural_trees", "score_units"];

pub struct Params {
    pub k: Knobs,
    off: Vec<&'static str>,
    /// calib/difficulty.json (or a calibration candidate).
    pub model: Value,
    trees: Vec<std::sync::OnceLock<Option<crate::model::Trees>>>,
    overrides: String,
    id: std::sync::OnceLock<u64>,
}

impl Params {
    /// Equal for equal calculators, whichever instance holds them.
    pub fn id(&self) -> u64 {
        *self.id.get_or_init(|| {
            use std::hash::{Hash, Hasher};
            let mut hash = std::collections::hash_map::DefaultHasher::new();
            (self.model.to_string(), &self.overrides).hash(&mut hash);
            hash.finish()
        })
    }

    pub fn new(model: Value, overrides: &Value) -> Result<Params, String> {
        let mut k = Knobs::default();
        let mut model = model;
        let mut off = Vec::new();
        if let Some(o) = overrides.as_object() {
            for (name, v) in o {
                if name != "off" {
                    k.set(name, v)?;
                    continue;
                }
                for c in v.as_array().ok_or("off: a list of component names")? {
                    let c = c.as_str().unwrap_or("");
                    if let Some(&known) = COMPONENTS.iter().find(|&&x| x == c) {
                        off.push(known);
                    } else if let Some(section) = c.strip_prefix("fit:").filter(|s| FITTED.contains(s)) {
                        if let Some(m) = model.as_object_mut() {
                            m.remove(section);
                        }
                    } else {
                        return Err(format!("unknown component: {c} (known: {COMPONENTS:?}, fit:{FITTED:?})"));
                    }
                }
            }
        }
        Ok(Params { k, off, model, trees: (0..=18).map(|_| std::sync::OnceLock::new()).collect(), overrides: overrides.to_string(), id: std::sync::OnceLock::new() })
    }

    pub fn off(&self, component: &str) -> bool {
        self.off.contains(&component)
    }

    /// Typed structural trees of a keymode, parsed once.
    pub fn trees(&self, keys: usize) -> Option<&crate::model::Trees> {
        self.trees.get(keys)?.get_or_init(|| crate::model::trees(&self.model["structural_trees"]["modes"][keys.to_string()])).as_ref()
    }
}
