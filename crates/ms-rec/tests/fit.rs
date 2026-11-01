//! The Personal fit on known synthetic histories.
use ms_model::personal::{personal_fit, steps, Best, Fit, Row, WarmupArgs, COOLING, FATIGUE_ONSET, GAP, SESSION_TAU};
use ms_model::predict::{ByKeys, Consts, Feats, HashMap, Mode, Slopes};
use ms_model::warmup::{activation, Done, FormPrior, Model};
use ms_rec::recommender::{family_transfer, KNOBS};

fn feats(c: &Consts, skill: &str) -> Feats {
    let mut f = Feats::bare(7, 6., 8., c);
    f.length = 120000.;
    f.sk[c.idx[skill]] = 1.;
    f.z_tail[1] = 1.;
    f.dom = c.family(skill);
    f
}

fn row(key: &str, chart: &str, ts: f64, t: f64, y: f64) -> Row {
    Row { keys: 7, y, base: 0., t, ts, public: false, local: true, rd: false, completed: true, chart: chart.to_string(), rate: 1., rate3: 1., best: Best::Key(key.to_string()), family: chart.to_string(), f: 0 }
}

fn args(c: &Consts) -> WarmupArgs {
    WarmupArgs { transfer: family_transfer(c), slopes: Slopes::new(ByKeys { map: [(0, 3.)].into_iter().collect() }, ByKeys::default(), 4., &Default::default(), &c.families) }
}

fn fit(c: &Consts, rows: Vec<Row>, f: &[Feats], now: f64) -> Fit {
    personal_fit(c, rows, f, now, &HashMap::default(), KNOBS, &args(c))
}

/// Forty three-hour sessions of two-minute maps with a known warm-up and fatigue; the same maps recur.
fn played(cold: f64, fatigue: f64) -> Mode {
    let c = Consts::standard();
    let mut seed = 1u64;
    let mut uniform = move || { seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); ((seed >> 11) as f64 + 0.5) / (1u64 << 53) as f64 };
    let mut rows = Vec::new();
    for s in 0..40 {
        for i in 0..36 {
            let clock = i as f64 * 5.;
            let noise = 0.03 * (-2. * uniform().ln()).sqrt() * (std::f64::consts::TAU * uniform()).cos();
            let y = -3. + cold * (-clock / SESSION_TAU).exp() + fatigue * (clock - FATIGUE_ONSET).max(0.) / 60. + noise;
            let (key, chart) = (format!("{s}.{i}"), format!("{:032x}", (s * 7 + i) % 300));
            rows.push(Row { base: -3., ..row(&key, &chart, 86400. * s as f64 + clock * 60. + 120., 299. + s as f64 / 30., y) });
        }
    }
    fit(&c, rows, &[feats(&c, "chordstream")], 301.).warmup.modes[&7].clone()
}

#[test]
fn the_fit_recovers_a_session_long_warm_up_and_fatigue() {
    let mode = played(0.3, 0.1);
    let scale = if mode.log_rate { 3. } else { 1. };
    assert!(mode.session * scale > 0.1, "{}", mode.session * scale);
    assert!((mode.fatigue * scale - 0.1).abs() < 0.03, "{}", mode.fatigue * scale);
    let none = played(0., 0.);
    assert!(none.session * scale < 0.03 && none.fatigue * scale < 0.02, "{} {}", none.session, none.fatigue);
}

#[test]
fn shuffled_rows_keep_a_reduced_broad_ability_weight() {
    let c = Consts::standard();
    let rows: Vec<Row> = (0..4).map(|i| Row { rd: i > 0, ..row(&i.to_string(), "same", 1000. + i as f64 * 130., 300., if i == 0 { -3. } else { -1. }) }).collect();
    let f = fit(&c, rows, &[feats(&c, "chordstream")], 300.);
    let weight = 3. * 0.35 / 3f64.sqrt();
    assert_eq!(f.n_rows, 4);
    assert!((f.glob - (-3. - weight) / (1. + weight)).abs() < 1e-9, "{}", f.glob);
    assert_eq!(f.aff["same"].1, 1.);
    assert!((f.aff["same|RD"].1 - 3. * 0.35).abs() < 1e-12);
}

#[test]
fn historical_and_live_exposure_match_without_future_knowledge() {
    let c = Consts::standard();
    let transfer = family_transfer(&c);
    let f = [feats(&c, "chordjack")];
    let rows: Vec<Row> = (0..4).map(|i| row(&i.to_string(), &i.to_string(), 1120. + 150. * i as f64, 300., -3.)).collect();
    let timeline = steps(&c, &rows, &f, &transfer, GAP, COOLING);
    assert_eq!(timeline.len(), 4);
    let family = c.family("chordjack");
    let mode = Mode { tau: 6., amplitudes: vec![0.5, 0.6, 0.9, 0.8, 0.3, 0.4, 0.4, 0.4], ..Mode::default() };
    let prior = FormPrior { global_variance: 0.05, skill_variance: 0.12, by_key: Default::default(), noise_variance: 0.04 };
    let cold = Model::new(&ms_model::personal::WarmModels { modes: [(7, mode)].into_iter().collect(), ..Default::default() }, Slopes::default(), prior, c.families.len());
    let mut history: Vec<Done> = Vec::new();
    for s in &timeline {
        let r = &rows[s.row];
        let active = activation(&history, r.ts - s.seconds, 7, Some(family), &transfer);
        assert!((active - s.minutes).abs() < 1e-12);
        let exposure = (-s.minutes / 6.).exp() * -(-s.d / 6.).exp_m1() * 6. / s.d;
        assert!((cold.penalty(&c, 7, Some("chordjack"), active, 120., None, 1e9) - cold.fatigue(7, 1e9) - 0.9 * exposure).abs() < 1e-12);
        history.push(Done { end: r.ts, begun: r.ts - 120., seconds: 120., keys: Some(7), family, identity: r.chart.clone() });
    }
    assert!(steps(&c, &[Row { public: true, local: false, ..rows[0].clone() }], &f, &transfer, GAP, COOLING).is_empty());
    assert!(steps(&c, &[Row { completed: false, ..rows[0].clone() }], &f, &transfer, GAP, COOLING).is_empty());
}
