#!/usr/bin/env python3
"""
Personal PP recommendations: what is worth playing now.

  value     expected change of the whole account's pp (0.95^i weighting + bonus) over a small set of
            plausible outcomes of the NEXT attempt; a refarm replaces that map's counted score only when better
  outcomes  y = log(1 − acc320) ~ personal level (per keymode, drifting with time) + public slope × log(ManiaScope
            rating) + hit window + public map effect + personal skill mix + personal chart affinity + session form,
            spread by the user's own attempt-to-attempt variability; model uncertainty only makes it pessimistic
  pp        lazer mania: 8·max(stars − 0.15, 0.05)^2.2 · max(0, 5·acc320 − 4) · (1 + 0.1·min(1, hits/1500))
            (checked against rosu-pp and the snapshot's stored pp; see test_recommend.py)
  session   warmup → build → push, recovery after a sustained dip; keymode continuity; retries return after
            other maps when long; Skip is "not now", never "can't"

usage: recommend.py list [N]      the current pool, ledger coverage and why
       recommend.py retro [CUT]   held-out check of predictions on the user's own later attempts (G533QR: needs the dump)
"""
import bisect
import collections
import gc
import json
import math
import os
import queue
import pickle
import random
import subprocess
import sys
import threading
import time
import traceback

import numpy as np

import recdata
import navigation
import skill_practice
import score_units

MODES = ("pp", "nps", "skills")
LOCAL_MODES = ("nps", "skills")
MODE_NAMES = {"pp": "PP", "nps": "NPS", "skills": "Skills"}

SESSION_GAP = 50 * 60          # seconds without gameplay that start a new session
ATT_SD = 1.0                   # attempt spread inflation over the within-chart residual sd (retro: 1.0 covers 83 %)
QUANTS = (np.arange(21) + 0.5) / 21
HT_MIN_P_UP = .5
HT_MIN_GAIN = 5.      # pp: HT only for a real gain, at least an even chance (user 2026-10-03)
MIN_GAIN = .1         # pp: below this expected gain a PP map is not worth offering (user 2026-10-03; was .001)
# Push is a weighted up-down staircase (Kaernbach 1991), no threshold (user 2026-10-01):
# a beaten PP-offered best is one step up; a PP attempt that neither beats its best nor
# meets the prediction, or any clearly bad play, is PUSH_DOWN steps down. The walk
# settles where P(beat) = PUSH_DOWN/(1+PUSH_DOWN) = 2/3 ("likely improvements far more
# often than long shots", 2026-09-30; 10 of 12 beaten on 09-30 was too easy).
# One step is half an attempt sd of challenge: small enough that noise can't swing
# the target, large enough to cross the user's safe→peak range within a session.
PUSH_DOWN = 2.
PUSH_STEP = .5
# Shown "usual" accuracy range: ±BAND_Z × predicted sd. The full sd covered 89 % of the user's
# 303 played predictions (2026-09-25..10-03), not 80 %; × .75 fitted on the first half covered
# 80.7 % of the second half with better log-loss (−1.130 vs −1.054). Display only: p_beat and
# PP quantiles keep the full spread.
BAND_Z = 1.2816 * .75
PESSIMISM = 0.25               # model uncertainty shifts the mean (in its sd), never widens the upside
HALF_LIFE_MONTHS = 12.0
LEVEL_BW = 4.0                 # months: level drift kernel
HISTORY_ZONE = .15             # Stats history counts plays at most this far (log stars) below the competence level
WARMUP_MAX = 3
COLD_PRIOR = .1     # Σ penalty² the historical cold curve is worth against tracked results (~40 openings at .05)
EVIDENCE_MODS = {"NF", "HD", "FI", "FL", "MR", "SD", "PF", "DT", "NC", "HT", "DC", "CL", "RD"}
import paths  # noqa: E402
LAZER_FILES = os.path.join(paths.lazer_data(), "files")
Z_Q = np.array([__import__("statistics").NormalDist().inv_cdf(float(q)) for q in QUANTS])


def p_beat(mu, sd, acc_needed):
    """P(acc320 > acc_needed) for y = log(1 − acc) ~ N(mu, sd) (vectorised; acc_needed ≥ 1 → 0)."""
    from math import erf
    y = np.log(np.clip(1 - np.asarray(acc_needed, float), 1e-9, None))
    z = (y - mu) / sd
    return np.where(np.asarray(acc_needed) >= 1, 0.0, 0.5 * (1 + np.vectorize(erf)(z / math.sqrt(2))))


# ---------------------------------------------------------------------------
# pp and the account ledger
# ---------------------------------------------------------------------------
def pp_scale(stars, hits, mult=1.0):
    return 8.0 * max(stars - 0.15, 0.05) ** 2.2 * (1 + 0.1 * min(1.0, hits / 1500)) * mult


def pp_at(stars, hits, acc, mult=1.0):
    return pp_scale(stars, hits, mult) * np.maximum(0.0, 5 * np.asarray(acc) - 4)


def bonus(n):
    return 416.6667 * (1 - 0.995 ** min(n, 1000))



def _interp(x, xs, ys):
    """np.interp for one scalar on a short sorted list: ~20x cheaper per call (PP scoring calls it per chart)."""
    i = bisect.bisect_right(xs, x)
    if i == 0:
        return ys[0]
    if i == len(xs):
        return ys[-1]
    x0, x1 = xs[i-1], xs[i]
    return ys[i-1] + (ys[i]-ys[i-1]) * (x-x0) / (x1-x0) if x1 > x0 else ys[i]

class Ledger:
    """Best counted pp per beatmap → total, and the exact change of replacing/adding one map's best."""

    def __init__(self, best):
        self.best = dict(best)                      # bid → (pp, source, when)
        self.P = np.array(sorted((v[0] for v in self.best.values()), reverse=True), float)
        self.w = 0.95 ** np.arange(len(self.P) + 1)
        self.cum = np.concatenate([[0.0], np.cumsum(self.P * self.w[:len(self.P)])])

    def total(self):
        return float(self.cum[-1]) + bonus(len(self.P))

    def delta(self, bid, new):
        """new: possible pp results on bid → account pp changes (0 when not better)."""
        return self.delta_many(np.array([self.best.get(bid, (0.0,))[0]]), np.atleast_2d(np.asarray(new, float)))[0]

    def delta_many(self, a, new):
        """a: (N,) current best per candidate (0 = none), new: (N, Q) outcomes → (N, Q) account changes.
        Replacing a with b > a shifts the scores between them down one place (×0.95); a new map also
        shifts everything below it and adds its bonus step."""
        n, negP, w, cum = len(self.P), -self.P, self.w, self.cum
        j = np.searchsorted(negP, -new, "left")                 # scores strictly above the new result
        i = np.searchsorted(negP, -a, "left")                   # first position holding a
        has = (a > 0)[:, None]
        jj = np.where(has, np.minimum(j, i[:, None]), j)
        rep = new * w[jj] - (a * w[np.minimum(i, n)])[:, None] - 0.05 * (cum[np.minimum(i, n)][:, None] - cum[jj])
        add = new * w[jj] - 0.05 * (cum[n] - cum[jj]) + bonus(n + 1) - bonus(n)
        return np.where(new > a[:, None], np.where(has, rep, add), 0.0)


def chart_path(sha=None, bid=None, md5=None):
    """A local copy of the chart: lazer's store by sha256, else the frozen dump by id (md5-checked)."""
    if sha:
        p = os.path.join(LAZER_FILES, sha[0], sha[:2], sha)
        if os.path.exists(p):
            return p
    if bid:
        p = recdata.frozen_chart(bid)
        if p and md5:
            import hashlib
            with open(p, "rb") as fh:
                if hashlib.md5(fh.read()).hexdigest() != md5:
                    return None
        return p
    return None


def ranked_status(r, pub, installed):
    st = r["status"]
    if st is None and r["md5"]:
        st = installed.get(r["md5"], {}).get("status")
    if st is None and r["beatmap_id"] and pub:
        st = pub["maps"].get(r["beatmap_id"], {}).get("approved")
    return st


def build_ledger(db, pub, installed=None, compute=True):
    """Counted bests: the snapshot's stable pp, plus lazer scores that were submitted (online id) or seen
    live, on ranked/approved maps with ranked mods at default settings. pp of lazer scores via rosu-pp."""
    installed = installed if installed is not None else installed_by_md5(db)
    best, missing = {}, 0
    rows = db.execute("SELECT * FROM scores WHERE ranked=1 AND beatmap_id>0").fetchall()
    pending = []
    for r in rows:
        if compute and r["pp"] is None and r["src"] != "public" \
                and ranked_status(r,pub,installed) in recdata.RANKED_STATUS \
                and (r["src"] != "realm" or r["online_id"] or r["client"] == "lazer-live"):
            item = pp_job(r)
            if item: pending.append(item)
    completed = {}
    if pending:
        try:
            import selected_analysis
            completed = selected_analysis.performance(pending)
        except Exception:
            traceback.print_exc()
    for r in rows:
        if ranked_status(r, pub, installed) not in recdata.RANKED_STATUS:
            continue
        if r["src"] in ("realm",) and not r["online_id"] and r["client"] != "lazer-live":
            continue                                  # never submitted: not on the account
        pp = r["pp"]
        if pp is None and r["src"] != "public" and compute:
            pp = completed.get(r["key"])
            if pp is None:
                missing += 1
                continue
            with db:
                db.execute("UPDATE scores SET pp=? WHERE key=?", (pp, r["key"]))
        if pp is None:
            continue
        b = r["beatmap_id"]
        if pp > best.get(b, (0,))[0]:
            best[b] = (pp, "lazer" if r["src"] != "public" else "stable", r["played"])
    return Ledger(best), missing


def pp_job(r):
    p = chart_path(r["sha256"], r["beatmap_id"], r["md5"])
    if not p:
        return None
    return {"key":r["key"],"path":p,"stats":json.loads(r["stats"]),"mods":json.loads(r["mods"] or "[]")}


def score_pp(r):
    try:
        import selected_analysis
        job=pp_job(r)
        return selected_analysis.performance([job]).get(r["key"]) if job else None
    except Exception:
        return None


def installed_by_md5(db, md5=None):
    return {r["md5"]: dict(r) for r in db.execute(
        "SELECT i.*,l.online_md5,l.creator FROM installed i LEFT JOIN installed_local l USING(sha256)"
        + (" WHERE i.md5=?" if md5 is not None else ""),
        (md5,) if md5 is not None else ()) if r["md5"]}


# ---------------------------------------------------------------------------
# personal model
# ---------------------------------------------------------------------------
def months(iso):
    try:
        y, m, d = int(iso[:4]), int(iso[5:7]), int(iso[8:10])
        return (y - 2000) * 12 + m - 1 + d / 31
    except (TypeError, ValueError):
        return None


def now_months():
    return months(time.strftime("%F"))


def base_of(pop, f, b=None, rate=None, level=None):
    """Public part of the prediction: slope × log rating + hit window + the map's effect → (value, n public rows).
    level: the player's competence level in this keymode (log stars) for the pattern level curves (recdata.gap_term)."""
    k = f["keys"]
    s = pop["slope"].get(k, pop["slope"][0])
    c = pop["window"].get(k, pop["window"][0])
    mb, n_mb = pop["mb"].get(b, (0.0, 0)) if b is not None else (0.0, 0)
    # Preserve fitted effects at their exact training rates and interpolate
    # between them. A custom 0.99× must not discard the 1.00× map evidence.
    anchors = [(r, pop["mbr"][(b, r)]) for r, _v in recdata.VARIANTS if (b, r) in pop["mbr"]]
    if rate is not None and rate > 0 and anchors:
        xs = [math.log(r) for r, _v in anchors]
        mb = _interp(math.log(rate), xs, [v[0] for _r, v in anchors])
        n_mb = _interp(math.log(rate), xs, [v[1] for _r, v in anchors])
    lr = math.log(max(.05, f["overall"]))
    curve = pop.get("curve", {}).get(k, pop.get("curve", {}).get(0, 0.))
    bend = max(0., lr-math.log(pop.get("curve_knee", 4.))) ** 2
    return s * lr + curve*bend + c * math.log(max(10.0, 64 - 3 * f["od"])) + mb + recdata.gap_term(pop, f, level), n_mb


def play_levels(rows, now=None):
    """{keys: competence level (log stars)} from the player's plays of the last year (all plays when
    fewer than 20), the same zone the population fit centres its gap curves on."""
    now = now_months() if now is None else now
    by = collections.defaultdict(list)
    for r in rows:
        by[r["keys"]].append(r)
    out = {}
    for k, rs in by.items():
        recent = [r for r in rs if r["t"] >= now - 12]
        use = recent if len(recent) >= 20 else rs
        out[k] = recdata.competence_level([math.log(max(.05, r["f"]["overall"])) for r in use])
    return out


PERSONAL_GEOMETRY = ('log_density_per_key', 'repeat',
                     'burst', 'active', 'all_columns', 'chordstream:bracket',
                     'chordjack:overlap', 'chordjack:width', 'chordjack:sustain',
                     'minijack:density', 'ln:release', 'ln:hybrid', 'ln:flow',
                     'rice:anchor', 'trill:control')
from structural_residual import NAMES as _STRUCTURE_NAMES
_GEOMETRY_INDICES=tuple(_STRUCTURE_NAMES.index(name) for name in PERSONAL_GEOMETRY)


def zvec(f, names, mu=-3.):
    load = f.get("endurance", {}).get("load", 0.)
    structure = f.get("nps", {})
    features = f.get("calibration_features", [])
    width = features[8] if len(features) > 8 else 0.
    # Near-limit fatigue is a performance interaction, not a 94%-specific
    # intrinsic chart rating. It is small when comfortable and grows near capacity.
    near_limit = 1. / (1. + math.exp(max(-20., min(20., -2. * (mu + 3.2)))))
    vector=f.get('execution',{}).get('residual_vector')
    geometry = [vector[i] for i in _GEOMETRY_INDICES] if vector else [0.]*len(PERSONAL_GEOMETRY)
    return np.array([f["sk"].get(k, 0.0) for k in names] + [f["ln"], f["stam"], load,
                     structure.get("chord", 1.) / max(1, f.get("keys", 7)),
                     structure.get("overlap", 0.), math.sqrt(max(0., width)) * load] + geometry + [load * near_limit])


class Personal:
    """Fitted from the user's own rows. Everything is per keymode with a pooled fallback."""
    levels = {}     # competence level per keymode (log stars); the rows' base already used it

    def __init__(self, pub, rows, now=None, levels=None):
        # Both forecasts target lazer plays. Stable merges a hold's head/tail
        # into one judgement; neither 305 nor 320 reweighting reconstructs the
        # two separate judgements. Stable tap scores remain transfer evidence.
        rows = [r for r in rows if not r['f'].get('ln') or
                (not r.get('key', '').startswith('legacy|') and
                 (str(r.get('client', '')).endswith('-lazer') or r.get('client') == 'lazer-live'))]
        import warmup
        self.warmup, cold = warmup.fit(rows, pub['pop'], pub.get('feats'))
        # Estimate a typical warmed baseline, not a mixture that counts the
        # historical cold deficit once here and again at next-session startup.
        rows = [dict(r, y=r['y']-cold.get(r.get('key'),0.)) for r in rows]
        pop = pub["pop"]
        self.pop, self.now = pop, now if now is not None else now_months()
        self.levels = levels or {}
        self.names = recdata.SKILLS
        K = np.array([r["keys"] for r in rows])
        y = np.array([r["y"] for r in rows])
        base = np.array([r["base"] for r in rows])
        t = np.array([r["t"] for r in rows])
        src = np.array([r["src"] == "public" for r in rows])
        randomized = np.array([bool(r.get("rd")) for r in rows])
        # RD permutes columns, hence hand assignments/pattern character. Retain
        # its broad ability evidence at reduced weight, but never let shuffled
        # performances become exact-normal-chart familiarity or retry evidence.
        chart = [r["chart"] + ("|RD" if r.get("rd") else "") for r in rows]
        e = y - base
        lr_all = np.array([math.log(max(.05, r["f"]["overall"])) for r in rows])
        # stable high scores are bests: shift them by the best-of-several gap measured on lazer attempts
        by = collections.defaultdict(list)
        for i, c in enumerate(chart):
            if not src[i]:
                by[c].append(e[i])
        gaps = [min(v) - np.mean(v) for v in by.values() if len(v) >= 3]
        self.best_gap = float(np.mean(gaps)) if gaps else -0.15
        e = np.where(src, e - self.best_gap, e)
        w_rec = 0.5 ** (np.maximum(0, self.now - t) / HALF_LIFE_MONTHS)
        cnt = collections.Counter(chart)
        lev = np.array([1 / math.sqrt(cnt[c]) for c in chart])      # one heavily retried chart can't dominate
        w = np.where(src, 0.5, 1.0) * lev * np.where(randomized, .35, 1.)
        self.keys = sorted(k for k, n in collections.Counter(K.tolist()).items() if n >= 30)
        self.level, self.level_var, self.curve, self.sd, self.beta, self.zmean, self.zscale = {}, {}, {}, {}, {}, {}, {}
        self.monthly = {}                  # keymode → (first month, Σw·e, Σw) of every-play rows (Stats history)
        self.shape_validation = {}
        glob = float(np.average(e, weights=w * w_rec)) if len(e) else 0.0
        self.glob = glob
        L_at = np.zeros_like(e)
        for k in self.keys:
            s = K == k
            mb = np.floor(t[s]).astype(int)
            grid = np.arange(mb.min(), max(mb.max(), int(self.now)) + 1)
            sw = np.bincount(mb - grid[0], w[s], minlength=len(grid))
            se = np.bincount(mb - grid[0], w[s] * e[s], minlength=len(grid))
            d = grid[:, None] - grid[None, :]
            ker = np.exp(-0.5 * (d / LEVEL_BW) ** 2)
            num, den = ker @ se, ker @ sw
            prior = float(np.average(e[s], weights=w[s]))
            curve = (num + 3 * prior) / (den + 3)
            self.curve[k] = (grid[0], curve)
            # Stats history: every recorded play, month by month. Website/stable bests survive only when
            # they were good (a 2020 "Azimuth" peak from top-100s), and this curve's 4-month kernel with
            # stable bests flattened a May peak into a steady rise (user 2026-10-01).
            own = w[s] * ~src[s]
            if k in self.levels:
                # Only maps near the player's limit say where the limit is: 97 % on maps 30 % below it
                # made an easy month look like the peak (user 2026-10-01).
                own = own * (lr_all[s] >= self.levels[k] - HISTORY_ZONE)
            self.monthly[k] = (grid[0], np.bincount(mb - grid[0], own * e[s], minlength=len(grid)),
                               np.bincount(mb - grid[0], own, minlength=len(grid)))
            L_at[s] = curve[mb - grid[0]]
            self.level[k] = float(curve[-1])
            resid = e[s] - L_at[s]
            self.sd[k] = float(np.sqrt(np.average(resid ** 2, weights=w[s])))
            self.level_var[k] = self.sd[k] ** 2 / max(1.0, den[-1])
        # skill mix: ridge on what the level leaves, per keymode with enough rows
        Z = np.array([zvec(r["f"], self.names, base[i] + L_at[i]) for i, r in enumerate(rows)]) \
            if rows else np.zeros((0, len(self.names) + 7 + len(PERSONAL_GEOMETRY)))
        u = e - L_at
        for k in self.keys:
            s = K == k
            if s.sum() < 200:
                continue
            ww = w[s] * w_rec[s]
            zm = np.average(Z[s], axis=0, weights=ww)
            scale=np.maximum(.05,np.sqrt(np.average((Z[s]-zm)**2,axis=0,weights=ww)))
            Zc = (Z[s] - zm)/scale
            import personal_support
            families = [r['f'].get('nps',{}).get('family') or r['chart'] for r,keep in zip(rows,s) if keep]
            self.beta[k], self.shape_validation[k] = personal_support.fit_shape(Zc,u[s],ww,families)
            self.zmean[k] = zm
            self.zscale[k] = scale
            u[s] = u[s] - Zc @ self.beta[k]
        # Historical session condition is not a permanent property of a chart.
        # Exclude the target chart when estimating that context, and count
        # retries in one session as one block of affinity evidence.
        import personal_support
        context, session_ids, self.session_validation = personal_support.session_context(rows,u)
        u = u-context
        blocks = personal_support.chart_blocks(rows,u,session_ids,self.now)
        # chart affinity (pooled over rates), and attempt-to-attempt spread from charts played repeatedly
        by = collections.defaultdict(list)
        for i, c in enumerate(chart):
            by[c].append(i)
        within = [u[ix] - u[ix].mean() for ix in by.values() if len(ix) >= 2]
        sd_att = float(np.sqrt(np.mean(np.concatenate(within) ** 2) * ATT_SD)) if within else 0.35
        means = np.array([np.average([v for r,v,w,n in values],weights=[w for r,v,w,n in values]) for values in blocks.values()])
        ns = np.array([personal_support.effective_count(values) for values in blocks.values()])
        self.tau2_c = max(0.005, float(np.mean(means ** 2 - sd_att ** 2 / ns))) if len(ns) else 0.05
        self.kap_c = sd_att ** 2 / self.tau2_c
        self.aff = {c: (sum(v*w for r,v,w,n in values)/(sum(w for r,v,w,n in values)+self.kap_c),
                        sum(w for r,v,w,n in values)) for c,values in blocks.items()}
        self.rate_aff = {}
        self.rate_support = {}
        self.rate_bandwidth, self.rate_validation = self._rate_transfer(rows,u,by,self.kap_c)
        for c, ix in by.items():
            rates = collections.defaultdict(list)
            for i in ix:
                rates[round(rows[i]["rate"], 3)].append(i)
            self.rate_support[c] = (min(rates),max(rates))
            if len(rates) > 1:
                # A directly observed DT must not inherit three pseudo-scores
                # from comfortable NM. Borrow only rate-nearby evidence using
                # the same TRAIN-learned transfer as extrapolation. The prior
                # precision is the measured between-chart shrinkage, not 3.
                stats = [(rate, sum(v*w for r,v,w,n in blocks[c] if r==rate),
                          sum(w for r,v,w,n in blocks[c] if r==rate)) for rate in rates]
                bandwidth = self.rate_bandwidth.get(K[ix[0]], self.rate_bandwidth[0])
                self.rate_aff[c] = self._rate_anchors(stats, bandwidth, self.kap_c)
        self.sd_att = {k: float(np.sqrt(np.mean(np.concatenate([u[ix] - u[ix].mean() for ix in by.values()
                                                               if len(ix) >= 2 and K[ix[0]] == k] or [np.zeros(1)]) ** 2) * ATT_SD))
                       for k in self.keys}
        self.sd_att = {k: (v if v > 0.05 else sd_att) for k, v in self.sd_att.items()}
        self.sd_all = sd_att
        self.warmup.form_prior={
            'global_variance':max(1e-4,self.session_validation['condition_variance']),
            'skill_variance':max(1e-4,self.session_validation['skill_condition_variance']),
            'by_key':self.session_validation['by_key'],
            'noise_variance':max(1e-4,sd_att**2)}
        self.retry = self._retry(rows, e - L_at)
        self.n_rows = len(rows)

    @staticmethod
    def _rate_anchors(stats, bandwidth, shrinkage):
        anchors = []
        for rate, total, count in stats:
            prior_sum, prior_n = 0., 0.
            for other, value, n in stats:
                if other == rate:
                    continue
                transfer = math.exp(-.5 * (math.log(other/rate) / bandwidth)**2)
                prior_sum += transfer * value
                prior_n += transfer * n
            prior = prior_sum / max(1e-9, prior_n + shrinkage)
            value = (total + shrinkage * prior) / max(1e-9, count + shrinkage)
            anchors.append((rate, value, count + min(shrinkage, prior_n)))
        return sorted(anchors)

    @staticmethod
    def _rate_transfer(rows, residual, by_chart, shrinkage):
        """Choose extrapolation transfer from TRAIN history, leaving out one
        chart-rate group at a time. Exact-rate evidence is never discounted.

        An easy NM's affinity is not automatically an easy DT's affinity; a
        difficult NM result is not a permanent penalty at an easier HT rate.
        Underprediction at a lower rate remains evidence at faster rates.
        Retain an infinite-bandwidth option when the history supports transfer.
        """
        folds=collections.defaultdict(list)
        for chart,indices in by_chart.items():
            rates=collections.defaultdict(list)
            for i in indices:
                rates[round(rows[i]['rate'],3)].append(i)
            if len(rates)<2:continue
            total=float(residual[indices].sum());count=len(indices)
            for rate,ix in rates.items():
                others=[r for r in rates if r!=rate]
                distance=max(0.,math.log(min(others)/rate),math.log(rate/max(others)))
                if distance < .02:continue
                estimate=(total-float(residual[ix].sum()))/(count-len(ix)+shrinkage)
                if estimate>0 and rate>max(others):distance=0.
                target=float(residual[ix].mean())
                folds[rows[ix[0]]['keys']].append((distance,estimate,target,min(3,len(ix))/len(rates)))
        bands=np.array([.08,.12,.18,.27,.40,.60,1.,float('inf')])
        def select(examples):
            data=np.array(examples,float)
            prediction=data[:,1,None]*np.exp(-.5*(data[:,0,None]/bands[None,:])**2)
            error=np.average((prediction-data[:,2,None])**2,axis=0,weights=data[:,3])
            at=int(np.argmin(error))
            return float(bands[at]), {'folds':len(examples),'train_mse':float(error[at]),'full_transfer_mse':float(error[-1])}
        pooled=[x for examples in folds.values() for x in examples]
        default,meta=select(pooled) if len(pooled)>=25 else (.27,{'folds':len(pooled)})
        chosen={0:default};reports={0:meta}
        for key,examples in folds.items():
            chosen[key],reports[key]=select(examples) if len(examples)>=40 else (default,{'folds':len(examples),'pooled':True})
        return chosen,reports

    @staticmethod
    def _retry(rows, u):
        """Bounded familiarity effect: the 2nd attempt of a chart within a session vs the 1st."""
        seq = sorted(range(len(rows)), key=lambda i: rows[i]["ts"] or 0)
        last, diffs = {}, []
        for i in seq:
            r = rows[i]
            if r["src"] == "public" or not r["ts"]:
                continue
            p = last.get(r["chart"])
            if p is not None and 0 < r["ts"] - rows[p]["ts"] < SESSION_GAP:
                diffs.append(u[p] - u[i])
            last[r["chart"]] = i
        if len(diffs) < 20:
            return 0.0
        return float(np.clip(np.mean(diffs) * len(diffs) / (len(diffs) + 50), 0.0, 0.15))

    def level_of(self, k):
        if k in self.level:
            return self.level[k], self.level_var[k], self.sd_att.get(k, self.sd_all)
        # a keymode without evidence: the average level, far less certain (not mastery, not a ban)
        return self.glob + 0.3, 0.25, self.sd_all

    def predict(self, f, chart, b=None, rate=None):
        """→ (mu, model sd, attempt sd) of y = log(1 − acc320) for the next attempt, before session form."""
        k = f["keys"]
        base, n_mb = base_of(self.pop, f, b, rate, self.levels.get(k))
        L, lv, sd = self.level_of(k)
        mu = L + base
        if k in self.beta:
            mu += float(((zvec(f, self.names, mu) - self.zmean[k])/self.zscale[k]) @ self.beta[k])
        a, n_c = self.chart_affinity(chart, rate, k)
        mu += a
        var = lv + self.pop["tau2"] * self.pop["kappa"] / (n_mb + self.pop["kappa"]) + self.tau2_c * self.kap_c / (n_c + self.kap_c)
        return mu, math.sqrt(var), sd

    def chart_affinity(self, chart, rate, k):
        a, n_c = self.aff.get(chart, (0.0, 0))
        anchors = self.rate_aff.get(chart, [])
        if anchors and rate:
            a = _interp(math.log(rate), [math.log(r) for r, _a, _n in anchors], [v for _r, v, _n in anchors])
            # Confidence follows the local anchor too. Three NM scores are
            # not three extra observations of the lone DT performance.
            n_c = _interp(math.log(rate), [math.log(r) for r, _a, _n in anchors], [n for _r, _a, n in anchors])
        support=self.rate_support.get(chart)
        if support and rate:
            distance=max(0.,math.log(support[0]/rate),math.log(rate/support[1]))
            transfer=math.exp(-.5*(distance/self.rate_bandwidth.get(k,self.rate_bandwidth[0]))**2)
            # A bad lower-rate result cannot vanish on an uprate and make it
            # look easier. Confidence still falls beyond measured rates.
            if a<=0 or rate<support[0]:a*=transfer
            n_c*=transfer*transfer  # uncertainty increases away from measured rates
        return a, n_c

    def rate_response(self, f, chart, bid, rate):
        """Rate-equivalent cold capacity uses the actual supported rate curve.

        Chart geometry supplies the physical derivative. Known public/personal
        rate effects also change with rate; ignoring these made a nominal 10%
        cold-capacity loss become only a few percent on steep familiar charts.
        No extra chart parse or native calculation is needed.
        """
        cache = self.__dict__.setdefault('_rate_response_cache', {})
        key = (chart, bid, rate)
        cached = cache.get(key)
        if cached is not None and cached[0] is f:
            return cached[1]
        physical = self.warmup.rate_slope(f)
        lo, hi = rate/1.01, rate*1.01
        level = self.levels.get(f['keys'])
        left = base_of(self.pop, f, bid, lo, level)[0] + self.chart_affinity(chart, lo, f['keys'])[0]
        right = base_of(self.pop, f, bid, hi, level)[0] + self.chart_affinity(chart, hi, f['keys'])[0]
        value = max(1., min(8., physical + (right-left)/math.log(hi/lo)))
        cache[key] = (f, value)
        return value

    def rate_loss(self, f, chart, bid, rate, log_rate_loss):
        """Integrate the rate response over a cold capacity shift.

        Multiplying an instantaneous derivative by the deficit made the cold
        forecast jump at measured-rate anchors. Compare the endpoints instead:
        physical demand follows the learned chart-rate elasticity, and public/
        personal rate evidence is evaluated at both rates. This is a prediction
        approximation, never a substituted chart analysis or displayed rating.
        """
        import warmup
        if log_rate_loss <= 0:
            return 0.
        # The curve, measured anchors and current-rate affinity are immutable
        # within this personal-model snapshot. Reuse them across opening/average
        # forecasts and filter updates; unplayed charts need only scalar maths.
        cache = self.__dict__.setdefault('_rate_loss_cache', {})
        key = (chart, bid, rate)
        context = cache.get(key)
        if context is None or context[0] is not f:
            pop = self.pop; k = f['keys']
            slope = pop['slope'].get(k, pop['slope'][0])
            curve = pop.get('curve', {}).get(k, pop.get('curve', {}).get(0, 0.))
            anchors = [(math.log(r), pop['mbr'][bid,r][0]) for r,_ in recdata.VARIANTS if (bid,r) in pop['mbr']]
            xs, ys = zip(*anchors) if anchors else ((), ())
            mb = float(np.interp(math.log(rate),xs,ys)) if anchors else 0.
            affinity = self.chart_affinity(chart,rate,k)[0] if chart in self.aff else None
            context = (f, slope, curve, math.log(pop.get('curve_knee',4.)),
                       warmup.rate_elasticity(f,self.warmup.elasticities), xs, ys, mb, affinity)
            cache[key] = context
        _, slope, curve, knee, elasticity, xs, ys, mb, affinity = context
        # ponytail: the pattern level curves (recdata.gap_term) are left out of this cold-capacity step; add
        # their difference here if cold forecasts on jack, LN or SV maps show a bias.
        old = math.log(max(.05,f['overall']))
        new = math.log(max(.05,f['overall']*math.exp(elasticity*log_rate_loss)))
        shift = slope*(new-old) + curve*(max(0.,new-knee)**2-max(0.,old-knee)**2)
        if xs:
            shift += _interp(math.log(rate)+log_rate_loss,xs,ys)-mb
        if affinity is not None:
            shift += self.chart_affinity(chart,rate*math.exp(log_rate_loss),f['keys'])[0]-affinity
        return max(0., shift)


def evidence_rows(db, pub, feats_of, since=None, until=None):
    """Personal rows with features: → list of dicts (y, base, keys, t, chart, f, src, ts)."""
    out, skipped = [], collections.Counter()
    installed = installed_by_md5(db)
    for r in db.execute("SELECT * FROM scores"):
        if until and (r["played"] or "") >= until:
            continue
        mods = json.loads(r["mods"] or "[]")
        if isinstance(mods, int):
            ac = set()                                 # stable bitmask: stage1 rules already drop the odd ones
            if mods & ~(1 | 8 | 64 | 256 | 512 | 1024 | 1048576 | 1073741824 | 16384 | 32):
                skipped["mods"] += 1
                continue
        else:
            ac = {str(m.get("acronym", "")).upper() for m in mods}
            if ac - EVIDENCE_MODS or r["rate"] is None:
                skipped["mods"] += 1
                continue
        if not r["acc"] or r["acc"] < 0.5 or (r["n"] or 0) < 100:
            skipped["short/failed"] += 1
            continue
        f, b = feats_of(r, installed)
        if not f:
            skipped["no features"] += 1
            continue
        t = months(r["played"] or "")
        if t is None:
            continue
        chart = r["md5"] or r["sha256"]
        out.append({"y": math.log(max(1e-3, 1 - min(r["acc"], 0.999))),
                    "y_lazer": math.log(max(1e-3,1-min(recdata.acc_lazer(json.loads(r['stats'])),.999))),
                    "keys": f["keys"], "t": t, "chart": chart, "f": f,
                    "src": r["src"], "client": r["client"], "ts": _ts(r["played"]), "b": b, "rate": r["rate"], "key": r["key"],
                    "rd": "RD" in ac,
                    "completed": (r['n'] or 0) >= .95*f['notes']*(1+f.get('ln',0.))})
    return out, skipped


def _ts(iso):
    return recdata.timestamp(iso)


class FeatureSource:
    """Chart features for a (chart, rate): the public table when the md5 matches, else the local cache
    (filled in the background from local chart files by `fill`)."""

    def __init__(self, db, pub):
        self.db, self.pub = db, pub
        self.calc = recdata.calc_id()
        self.md5_bid = {m["md5"]: b for b, m in pub["maps"].items() if m.get("md5")}
        # The exact-rate library can contain hundreds of thousands of entries.
        # Only load personal/selected features here; the catalogue builder owns
        # the full-library scan. Recreating this object must not decode that
        # entire catalogue after every score.
        self.cache = {}

    def lookup(self, key):
        f = self.cache.get(key)
        if f is None:
            row = self.db.execute("SELECT data FROM feats WHERE key=?", (key,)).fetchone()
            if row:
                try:
                    f = self.cache[key] = json.loads(row[0])
                except (ValueError, TypeError):
                    return None
        return f

    def key(self, sha, rate):
        return f"{sha}@{rate:.3f}|{self.calc}"

    def __call__(self, r, installed=None):
        if r["rate"] is None:
            return None, None
        md5 = r["md5"]
        b = r["beatmap_id"] if r["beatmap_id"] and r["beatmap_id"] > 0 else self.md5_bid.get(md5)
        if b and (not md5 or self.pub["maps"].get(b, {}).get("md5") != md5):
            b = None                                   # a stale id: not the snapshot's chart
        rate = round(r["rate"] or 1.0, 3)
        if r["sha256"]:
            f = self.lookup(self.key(r["sha256"], rate))
            if f:
                pf = self.pub["feats"].get(b, {}) if b else {}
                if "stars" not in f and pf.get("calc") == self.calc and "stars" in pf.get(rate, {}):
                    f = dict(f, stars=pf[rate]["stars"], hits=pf[rate]["hits"])
                return f, b
        pf = self.pub["feats"].get(b) if b else None
        if pf and pf.get(rate) and pf.get("calc") == self.calc:
            return pf[rate], b
        if r["sha256"]:
            f = self.lookup(self.key(r["sha256"], rate))
            if f:
                return f, b
        return None, b

    def missing(self, rows):
        """(sha, rate) pairs of local scores lacking features whose chart file is here."""
        need = set()
        for r in rows:
            if r["sha256"] and not self(r)[0] and os.path.exists(chart_path(r["sha256"]) or ""):
                need.add((r["sha256"], round(r["rate"] or 1.0, 3)))
        return sorted(need)

    def fill(self, pairs, stop=lambda: False, workers=3, chunk=40, cancel=lambda: False, progress=None):
        """Compute features in niced worker processes, a chunk at a time (stop() pauses between chunks).
        progress(charts_done, charts_total) after each chunk."""
        from multiprocessing import get_context
        by = collections.defaultdict(list)
        for sha, rate in pairs:
            by[sha].append(rate)
        jobs = [(sha, rates, self.calc) for sha, rates in by.items()]
        done = 0
        with get_context("spawn").Pool(workers, initializer=paths.lower_priority) as pool:
            for i in range(0, len(jobs), chunk):
                if cancel():
                    return done
                while stop():
                    if cancel():
                        return done
                    time.sleep(2)
                for sha, res in pool.map(_feat_job, jobs[i:i + chunk]):
                    done += self._store(sha, res)
                if progress:
                    progress(min(len(jobs), i + chunk), len(jobs))
        return done

    def _store(self, sha, res):
        if not isinstance(res, dict):
            return 0
        with self.db:
            for rate, f in res.items():
                k = self.key(sha, rate)
                self.cache[k] = f
                self.db.execute("INSERT OR REPLACE INTO feats VALUES (?, ?)", (k, json.dumps(f)))
        return 1

    def ensure(self, sha, md5, bid, rate):
        """A chart just played (any map, any status) is evidence at once, not after the next start's fill."""
        if rate is None:
            return
        rate = round(rate, 3)
        if sha and not self({"sha256": sha, "md5": md5, "beatmap_id": bid, "rate": rate})[0] \
                and os.path.exists(chart_path(sha) or ""):
            self._store(*_feat_job((sha, [rate], self.calc)))


def _feat_job(job):
    sha, rates, *identity = job
    try:
        if identity and identity[0] != recdata.calc_id():
            import selected_analysis
            if selected_analysis.previous_helper(identity[0]) is None:
                return sha, "Calculator changed on disk; reopen ManiaScope to activate it"
            features = {}
            for rate in rates:
                features.update(selected_analysis.calculate(chart_path(sha), rate, identity[0]))
            return sha, features
        return sha, recdata.chart_feats(chart_path(sha), rates)
    except Exception as exc:
        return sha, str(exc)


# ---------------------------------------------------------------------------
# session
# ---------------------------------------------------------------------------
class Session:
    """Transient state rebuilt from the event log: attempts since the last gap/reset, their residuals,
    exposures (next / open / skip). A reset clears readiness, never history or bests."""

    def __init__(self, events, now=None, warmup_model=None):
        now = time.time() if now is None else now
        self.now = now
        ev = sorted((e for e in events if e["t"] <= now), key=lambda e: e["t"])
        start = 0
        last_play, active_until, active_key, active_id = None, None, None, None
        for i, e in enumerate(ev):
            if e["kind"] == "reset":
                start = i + 1
                last_play, active_until, active_key, active_id = None, None, None, None
            elif e["kind"] in ("start", "finish", "abort"):
                terminal_matches = e["kind"] in ("finish", "abort") and e["beatmap"] == active_key \
                    and (e["info"].get("start_id", active_id) == active_id)
                continuing = active_until is not None and (terminal_matches or e["t"] <= active_until)
                if last_play is not None and e["t"] - last_play > SESSION_GAP and not continuing:
                    start = i
                last_play = e["t"]
                if e["kind"] == "start":
                    active_until = e["t"] + (e["info"].get("length") or 900) + SESSION_GAP
                    active_key, active_id = e["beatmap"], e.get("id")
                else:
                    active_until = None
            elif e["kind"] == "void":
                active_until = None
        if last_play is not None and now - last_play > SESSION_GAP and (active_until is None or now > active_until):
            start = len(ev)
        self.events = ev[start:]
        self.all = ev
        self.attempts = []                            # current-session form only
        self.history_attempts = []                    # freshness also survives gaps/restarts
        self._corrections = {}
        self._warmup_model = warmup_model
        notes = {e["info"].get("start_id"): e["info"] for e in ev
                 if e["kind"] == "attempt_note" and e["info"].get("start_id") is not None}
        results = {e["info"].get("start_id"): e["info"] for e in ev if e["kind"] == "pp_result"}
        pending = {}
        session_begin = self.events[0]["t"] if self.events else float("inf")
        for e in ev:
            info = e["info"]
            if e["kind"] == "start":
                pending = dict(info, t=e["t"], beatmap=e["beatmap"], start_id=e.get("id"))
            elif e["kind"] in ("void", "reset"):
                pending = {}                          # a replay or an old result: not an attempt
            elif e["kind"] in ("finish", "abort") and pending and e["beatmap"] == pending["beatmap"] \
                    and ("start_id" not in info or info["start_id"] == pending["start_id"]):
                a = dict(pending, end=e["t"], kind=e["kind"], **info)
                seconds = info.get("played_seconds", (info.get("progress") or 0.) * (a.get("length") or 0.))
                a["meaningful"] = e["kind"] == "finish" or seconds >= min(20., .3 * (a.get("length") or 120.))
                annotation = notes.get(a.get("start_id"), {})
                a["reason"] = annotation.get("reason", info.get("reason"))
                a["reason_confidence"] = annotation.get("confidence",1.)
                a["ability_evidence"] = e['kind']=='finish' and a["reason"] not in ("interrupted", "experiment", "bored")
                result = results.get(a.get("start_id"))
                # Push evidence: an improved counted best on a map the PP playlist offered.
                a["beat"] = bool(result and a.get("mode") == "pp" and a.get("purpose") != "warmup"
                                 and result["after"] > result["before"] + .05)
                if e["kind"] == "finish" and info.get("prediction_valid", True) and pending.get("mu") is not None \
                        and info.get("y") is not None and pending.get("sd"):
                    central = pending.get("shown_mu", pending["mu"] + PESSIMISM * (pending.get("sdm") or 0.))
                    a["z"] = (central - info["y"]) / pending["sd"]
                # An interruption is effort, not a measured accuracy or dislike.
                self.history_attempts.append(a)
                if a["t"] >= session_begin:
                    self.attempts.append(a)
                pending = {}
        self.warm_flag = any(e["kind"] == "skip_warmup" for e in self.events)
        self.lock = None                 # old scalar locks do not constrain per-tab choices
        self._events_by_map = collections.defaultdict(list)
        self._attempts_by_map = collections.defaultdict(list)
        self._offer_times = collections.defaultdict(list)
        for e in self.all:
            self._events_by_map[str(e["beatmap"])].append(e)
            if e["kind"] == "offer":
                self._offer_times[e["info"].get("mode")].append(e["t"])
        for a in self.history_attempts:
            self._attempts_by_map[str(a["beatmap"])].append(a)
        self._downrates = collections.defaultdict(list)
        for a in self.attempts:
            actual, offered = a.get("rate"), a.get("offered_rate")
            if a.get("meaningful") and a.get("ability_evidence", True) and actual and offered and actual < offered-.025:
                self._downrates[str(a["beatmap"])].append(a)
        self.played = {a["beatmap"] for a in self.attempts if a.get("meaningful") and a.get("ability_evidence", True)}

    def bind_warmup(self, model):
        if model is not None and self._warmup_model is not model:
            self._warmup_model=model
            self._corrections.clear()
        return self

    def for_accuracy(self, model):
        import accuracy_targets
        return accuracy_targets.session_view(self,model)

    def model_key(self):
        """A click changes freshness, not ability. Clock decay gets a cheap 30s bucket."""
        last = max((e.get("id", e["t"]) for e in self.events
                    if e["kind"] in ("finish", "abort", "reset", "skip_warmup", "attempt_note", "pp_result")), default=0)
        return last, int(self.now / 30), bool(self.events), self.warm_flag

    def activation(self, keys, skill=None):
        """Relevant minutes in this keymode. Other keymodes cannot erase cold readiness."""
        if self.warm_flag:
            return 10.
        if self._warmup_model is not None:
            import warmup
            key=('activation',keys,skill)
            if key not in self._corrections:
                history=list(self._warmup_model.history)+self.attempts
                # Explicit reset overrides restored prior gameplay.
                resets=[e['t'] for e in self.all if e['kind']=='reset']
                if resets:history=[a for a in history if a['end']>max(resets)]
                self._corrections[key]=warmup.activation(history,self.now,keys,skill)
            return self._corrections[key]
        minutes = 0.
        for a in self.attempts:
            if not a.get("meaningful") or not a.get("ability_evidence", True) or a.get("keys") != keys:
                continue
            seconds = a.get("played_seconds") or (a.get("length") if a["kind"] == "finish" else 0.) or 0.
            transfer = 1. if skill is None else .3 + .7 * skill_related(skill, a.get("skill"))
            minutes += min(300., seconds) / 60 * transfer * math.exp(-max(0., self.now-a["end"])/2400.)
        return minutes

    def zs(self):
        return [(a, a["z"], a.get("wz", 1.0)) for a in self.attempts
                if a.get("z") is not None and a.get("ability_evidence", True)]

    def form(self, keys, skill):
        """Positive means good form; actual predictions use log-error correction directly."""
        return -self.correction(keys, skill) / .3

    def correction(self, keys, skill, f=None):
        """Fast downward adaptation, cautious upward adaptation, partial skill transfer.

        Residuals are against the frozen pre-session base, not a prediction that
        already contains this correction. Browsing/rate toggles supply no result.
        """
        prior=getattr(self._warmup_model,'form_prior',None)
        if prior is not None and keys is not None:
            import session_form
            ident=('posterior',keys)
            if ident not in self._corrections:
                self._corrections[ident]=session_form.posterior(self.attempts,self.now,keys,prior)
            mean,_variance=self._corrections[ident]
            return float(session_form.loading(skill,f)@mean)
        cache_key = (keys, skill)
        if cache_key in self._corrections:
            return self._corrections[cache_key]
        evidence = [a for a in self.attempts if a.get("meaningful") and a.get("ability_evidence", True)]
        good = sum(a.get("z", 0.) >= .5 and a.get("keys") == keys
                   and skill_related(skill, a.get("skill")) >= .45 for a in evidence[-5:])
        num, den = 0., 1.2
        for i, a in enumerate(evidence):
            if a.get("y") is not None and a.get("mu") is not None:
                residual = a["y"] - a.get("base_mu", a["mu"]) - a.get('cold_penalty',0.)
                residual = max(-.65, min(1.8, residual))
                strength = 1.8 if residual > 0 else (.45 if good >= 3 or keys is None else .05)
            else:
                continue
            related = skill_related(skill, a.get("skill"))
            transfer = (.16 + .84 * related) if keys == a.get("keys") else .06
            if keys is None:
                transfer = .25
            w = strength * transfer * .78 ** (len(evidence)-1-i) * math.exp(-max(0., self.now-a["end"]) / 1800.)
            num += w * residual
            den += w
        out = max(-.4, min(1.4, num / den))
        self._corrections[cache_key] = out
        return out

    def rate_penalty(self, chart, rate):
        """A substantially played downrate constrains faster proposals of that exact chart.

        This is censored preference/ability evidence, not a fabricated score at
        the unplayed rate. A successful easier play remains useful evidence.
        """
        penalty = 0.
        for a in (*self._downrates.get("md5:"+str(chart), ()), *self._downrates.get("sha:"+str(chart), ())):
            actual, offered = a.get("rate"), a.get("offered_rate")
            if not a.get("meaningful") or not actual or not offered or actual >= offered-.025:
                continue
            if a.get("beatmap") not in ("md5:"+str(chart), "sha:"+str(chart)) or rate <= actual:
                continue
            penalty = max(penalty, min(.8, 2.8 * math.log(rate / actual))
                          * math.exp(-max(0., self.now-a["end"]) / 3600.))
        return penalty

    def chart_correction(self, chart, rate, f, model, variance):
        """Update this chart from its completed plays without poisoning others.

        The shared form posterior marginalizes chart error. Conditional on that
        form, repeated measurements can update the chart-specific residual. A
        bad forecast is still evidence about THIS chart, even when its large
        error is appropriately downweighted as evidence about every other map.
        """
        if not hasattr(model,'sd_att') or variance <= 0:return 0.
        cache_key=('chart_correction',chart,rate,id(model))
        if cache_key in self._corrections:return self._corrections[cache_key]
        attempts=(*self._attempts_by_map.get('md5:'+str(chart),()),
                  *self._attempts_by_map.get('sha:'+str(chart),()))
        if not attempts:return 0.
        active={a.get('start_id') for a in self.attempts}
        bandwidth=model.rate_bandwidth.get(f['keys'],model.rate_bandwidth[0])
        precision=1./max(1e-6,variance);numerator=0.
        for a in attempts:
            if a.get('start_id') not in active or not a.get('ability_evidence',True) \
                    or a.get('y') is None or a.get('base_mu') is None or not a.get('rate'):
                continue
            transfer=math.exp(-.5*(math.log(rate/a['rate'])/bandwidth)**2)
            age=math.exp(-max(0.,self.now-a['end'])/3600.)
            noise=max(1e-4,model.sd_att.get(f['keys'],model.sd_all)**2)
            weight=transfer*age/noise
            error=a['y']-a['base_mu']-(a.get('cold_penalty') or 0.)
            error-=self.correction(f['keys'],a.get('skill'),a.get('features'))
            numerator+=weight*error;precision+=weight
        result=numerator/precision
        self._corrections[cache_key]=result
        return result

    def activation_for(self, f, profile=None):
        import warmup
        keys = f['keys']
        if self.warm_flag:
            return 10.
        if self._warmup_model is None or self._warmup_model.parameters(keys).get('units') != 'log_rate':
            return self.activation(keys, warmup.dominant(f))
        key = ('activation-families',keys)
        if key not in self._corrections:
            self._corrections[key] = {fam:self.activation(keys,fam) for fam in warmup.FAMILIES}
        values = self._corrections[key]
        return sum(w*values[fam] for fam,w in (profile or warmup.family_weights(f)))

    def cold_scale(self, keys):
        """Share of the historical cold curve that tracked results still show (user 2026-10-01: first
        maps ~.05–.1× too easy; G835LX 7K forecasts with .045 / .124 cold penalty came out .15 / .26
        better than predicted, warm ones .04). Least squares of result on penalty through the warm
        baseline, prior 1 worth COLD_PRIOR. Recorded penalties are unscaled by their own scale."""
        key = ('cold_scale', keys)
        if key not in self._corrections:
            obs = [(a['cold_penalty'] / (a.get('cold_scale') or 1.), a['y'] - a['base_mu']) for a in self.history_attempts
                   if a.get('keys') == keys and a['kind'] == 'finish' and a.get('ability_evidence', True)
                   and a.get('y') is not None and a.get('base_mu') is not None and a.get('cold_penalty') is not None
                   and (a.get('cold_scale') or 1.) > 0]
            warm = [e for c, e in obs if c < .02]
            base = sum(warm) / len(warm) if len(warm) >= 5 else 0.
            num = sum(c * (e - base) for c, e in obs) + COLD_PRIOR
            den = sum(c * c for c, _e in obs) + COLD_PRIOR
            self._corrections[key] = max(0., min(1.5, num / den))
        return self._corrections[key]

    def warmup_penalty(self, keys, skill, seconds=0., f=None, speed_slope=None, response=None, profile=None):
        return self._historical_cold(keys, skill, seconds, f, speed_slope, response, profile) * self.cold_scale(keys)

    def _historical_cold(self, keys, skill, seconds=0., f=None, speed_slope=None, response=None, profile=None):
        """Historical completed-score curve; no target changes or rate-up bans.

        Difficulty/rate comes down to preserve the same ~94% target. The chart's
        intrinsic rating and the selected practice skill remain unchanged.
        """
        if self.warm_flag:
            return 0.
        if self._warmup_model is not None:
            import warmup
            slope=speed_slope if speed_slope is not None else self._warmup_model.rate_slope(f) if f is not None else None
            if self._warmup_model.parameters(keys).get('units') == 'log_rate':
                # A small rate change can swap the leading skill without
                # changing the player's hands. Blend cold loss over the mix,
                # rather than jump from one family curve to another at that tie.
                if f is None:
                    loss = self._warmup_model.penalty(keys,skill,self.activation(keys,skill),seconds,1.)
                else:
                    # Family readiness is constant across the catalogue in this
                    # frozen Session. Compute its exponentials once, not once per
                    # family per chart per opening/average forecast.
                    key = ('capacity-families',keys)
                    if key not in self._corrections:
                        self._corrections[key] = {fam:self._warmup_model.penalty(
                            keys,fam,self.activation(keys,fam),0.,1.) for fam in warmup.FAMILIES}
                    values = self._corrections[key]
                    loss = sum(w*values[fam] for fam,w in (profile or warmup.family_weights(f)))
                    tau = self._warmup_model.parameters(keys)['tau']
                    d = max(0.,min(10.,seconds/60))
                    loss *= -math.expm1(-d/tau)*tau/d if d>.001 else 1.
                return response(loss) if response is not None else loss*(3. if slope is None else slope)
            if f is not None:
                skill = warmup.dominant(f)
            return self._warmup_model.penalty(keys,skill,self.activation(keys,skill),seconds,slope)
        cache_key = ("warmup", keys, skill)
        if cache_key not in self._corrections:
            minutes = self.activation(keys, skill)
            self._corrections[cache_key] = .10 * max(0., 1. - minutes/6.)
        return self._corrections[cache_key]

    def push(self, keys=None, skill=None):
        """Staircase position this session in steps, ≥ 0 (see PUSH_DOWN). Ability itself is
        tracked by correction(); after a good or bad play the next one is predicted within
        noise (Sept: next z +.23±.29 after a beat, +.00±.16 after z ≤ −1), so this is target only."""
        cache_key = ('push', keys)
        if cache_key in self._corrections:
            return self._corrections[cache_key]
        push = 0.
        for a in self.attempts:
            if not (a.get("meaningful") and a.get("ability_evidence", True)
                    and (keys is None or a.get("keys") == keys)):
                continue
            z = a.get("z", 0.)
            offered = a.get("mode") == "pp" and a.get("purpose") != "warmup"
            if a.get("beat"):
                push += 1.
            elif z <= -1. or (offered and z < 0.):
                push = max(0., push - PUSH_DOWN)
        self._corrections[cache_key] = push
        return push

    def phase(self):
        zs = self.zs()
        n = len([a for a in self.attempts if a["kind"] == "finish"])
        warmups = sum(a.get("purpose") == "warmup" and a['kind']=='finish' for a in self.attempts)
        minutes = sum(min(max(0., a.get("end", a["t"]) - a["t"]), a.get("length") or 900) / 60
                      for a, z, _w in zs if z > -1.)
        strong = any(z >= 0.5 and a.get("purpose") != "warmup" for a, z, _w in zs)
        if not (self.warm_flag or strong or n >= 3 or warmups >= WARMUP_MAX or minutes >= 8):
            return "warmup"
        recent = [z for _a, z, _w in zs[-2:]]
        if len(zs) >= 2 and np.mean(recent) < -1.0:
            return "recover"
        if recent and recent[-1] < -1.8:
            return "recover"
        return "push" if self.push() > 0 else "build"

    def current_keys(self):
        for a in reversed(self.attempts):
            if a.get("keys"):
                return a["keys"]
        return None

    def freshness(self, bid, length_s, mode=None, pool=None):
        """1 = fresh. A restart/reset affects form, never recent-play avoidance.

        pool: the playlist's effective size. An offered chart then recovers by how many other offers
        came since, (n/(n + pool/2))²: a soft shuffle bag — a quarter back after half the pool, 44% after
        all of it, so a favourite can return before the pool is exhausted but never right away. Not a
        hard bag ("not guaranteed … but it shouldn't come back very soon", user 2026-10-01), and not a
        clock — a 40-minute recovery brought charts back within one session."""
        f, now = 1.0, self.now
        att = [a for a in self._attempts_by_map.get(str(bid), []) if a.get("meaningful")]
        last_offer = None
        for e in self._events_by_map.get(str(bid), []):
            dt = now - e["t"]
            if e["kind"] in ("offer", "next", "open"):
                # One offer may also log next/open. Use its latest timestamp,
                # not three accidental independent votes for the same exposure.
                last_offer = e["t"]
            elif e["kind"] == "skip":
                if mode and e["info"].get("mode", mode) != mode:
                    continue
                f *= (0.1 if e in self.events else 1 - 0.9 * math.exp(-dt / 43200))
        if last_offer is not None and pool:
            n = len(self._offer_times.get(mode, ())) - bisect.bisect_right(self._offer_times.get(mode, []), last_offer)
            f *= (n / (n + .5 * pool)) ** 2
        elif last_offer is not None:
            f *= 1 - .98 * math.exp(-max(0., now-last_offer) / 2400.)
        if att:
            last = att[-1]
            since = {a.get("beatmap") for a in self.history_attempts
                     if a["t"] > last["t"] and a.get("meaningful") and a.get("beatmap") != str(bid)}
            need = min(10, 5 + math.ceil(length_s / 180))
            recovery = 1 - .97 * math.exp(-max(0., now-last["end"]) / (18 * 3600.))
            f *= max(recovery, min(1., len(since)/need) * .6)
            if len(att) >= 3:
                f *= 0.5 ** (len(att) - 2)                # boredom, even for short maps
        return max(1e-6, f)


def load_events(db, since=None):
    since = since if since is not None else time.time() - 3 * 86400
    events = [{"id": r["id"], "t": r["t"], "kind": r["kind"], "beatmap": r["beatmap"], "info": dict(recdata.event_info(r["info"]))}
            for r in recdata.tracked_events(db, since)]
    offers = {e["id"]: e["info"] for e in events if e["kind"] == "offer"}
    for e in events:
        if e['kind']=='finish' and e['info'].get('y_lazer') is None and e['info'].get('key'):
            row=db.execute('SELECT stats FROM scores WHERE key=?',(e['info']['key'],)).fetchone()
            if row and row[0]:
                import accuracy_targets
                e['info']['y_lazer']=accuracy_targets.log_loss(recdata.acc_lazer(json.loads(row[0])))
        if e["kind"] == "start" and e["info"].get("offer_id") in offers:
            e["info"].setdefault("offered_rate", offers[e["info"]["offer_id"]].get("rate"))
    aliases, starts, observed, sha_aliases = {}, {}, collections.defaultdict(set), {}
    for e in events:
        info = e["info"]
        if info.get("sha") and not info.get("md5"):
            sha = info["sha"]
            if sha not in sha_aliases:
                row = db.execute("SELECT md5 FROM installed WHERE sha256=?", (sha,)).fetchone()
                sha_aliases[sha] = row[0] if row else None
            if sha_aliases[sha]:
                info = e["info"] = dict(info, md5=sha_aliases[sha])
        if info.get("md5") or info.get("sha"):
            key = event_key(info)
            if e["kind"] == "start":
                starts[e["id"]] = key
            if navigation.positive_id(e["beatmap"]):
                observed[e["beatmap"]].add(key)
    aliases.update({old: next(iter(values)) for old, values in observed.items() if len(values) == 1})
    for e in events:
        old, info = e["beatmap"], e["info"]
        if e["kind"] in ("finish", "abort") and info.get("start_id") in starts:
            e["beatmap"] = starts[info["start_id"]]
        elif info.get("md5") or info.get("sha"):
            e["beatmap"] = event_key(info)
        elif old is not None:
            if old not in aliases:
                bid = navigation.positive_id(old)
                matches = db.execute("SELECT DISTINCT md5 FROM installed WHERE beatmap_id=? AND md5 IS NOT NULL LIMIT 2",
                                     (bid,)).fetchall() if bid else []
                aliases[old] = ("md5:" + matches[0][0]) if len(matches) == 1 else legacy_key(old)
            e["beatmap"] = aliases[old]
    return events


def recover_results(db):
    """Pair imported results only with a matching, already recorded attempt. Never predict after a play."""
    starts = db.execute("SELECT * FROM events WHERE kind='start' AND t>=? ORDER BY t, id",
                        (time.time() - 3 * 86400,)).fetchall()
    finished = {json.loads(r[0]).get("key") for r in db.execute("SELECT info FROM events WHERE kind='finish'")}
    added = 0
    for i, start in enumerate(starts):
        info = json.loads(start["info"])
        end = starts[i + 1]["t"] if i + 1 < len(starts) else time.time()
        end = min(end, start["t"] + (info.get("length") or 900) + 120)
        matches = [r for r in db.execute("SELECT * FROM scores WHERE sha256=?", (info.get("sha"),))
                   if r["key"] not in finished and r["rate"] == info.get("rate")
                   and sorted(m.get("acronym", "") for m in json.loads(r["mods"] or "[]")) == sorted(info.get("mods") or [])
                   and start["t"] < (_ts(r["played"]) or 0) <= end]
        if len(matches) != 1:
            continue
        r = matches[0]
        t = _ts(r["played"])
        if not recdata.tracking_allowed(db, t, start["t"]) or info.get("predicted_at", start["t"]) > t:
            continue
        recdata.log_event(db, "finish", start["beatmap"], t=t, key=r["key"], start_id=start["id"], recovered=True,
                          y=math.log(max(1e-3, 1 - min(r["acc"], 0.999))),
                          y_lazer=math.log(max(1e-3,1-min(recdata.acc_lazer(json.loads(r['stats'])),.999))))
        with db:
            db.execute("UPDATE scores SET fresh=1 WHERE key=?", (r["key"],))
        finished.add(r["key"])
        added += 1
    return added


def play_history(db, limit=100, outliers=False):
    """Saved results and their frozen predictions; unobserved historical plays have no prediction."""
    pauses = {}
    if db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='replay_evidence'").fetchone():
        pauses = {r[0]: json.loads(r[1]) if r[1] else None
                  for r in db.execute("SELECT score_key,pauses FROM replay_evidence")}
    paired, pending = {}, None
    events = db.execute("SELECT * FROM events WHERE kind IN ('start','finish','void') ORDER BY t,id").fetchall()
    starts = {r["id"]: r for r in events if r["kind"] == "start"}
    for e in events:
        info = json.loads(e["info"])
        if e["kind"] == "start":
            pending = e
        elif e["kind"] == "void":
            pending = None
        elif e["kind"] == "finish":
            start = starts.get(info.get("start_id")) if "start_id" in info else pending
            if start and start["beatmap"] == e["beatmap"]:
                paired[info.get("key")] = (start, info)
            pending = None
    out = []
    for r in db.execute("SELECT s.*, i.artist,i.title,i.version,i.keys FROM scores s "
                        "LEFT JOIN installed i ON i.sha256=s.sha256 WHERE s.src!='public' ORDER BY s.played DESC"):
        st = json.loads(r["stats"])
        actual = recdata.acc_lazer(st)
        start, finish = paired.get(r["key"], (None, {}))
        info = json.loads(start["info"]) if start else {}
        t = _ts(r["played"])
        if start and (info.get("sha") != r["sha256"] or info.get("rate") != r["rate"]
                      or (t is not None and info.get("predicted_at", start["t"]) > t)):
            info = {}
        z = None
        if info.get('display_mu') is not None and info.get('display_sd'):
            z=(info['display_mu']-math.log(max(1e-3,1-min(actual,.999))))/info['display_sd']
        elif info.get("mu") is not None and info.get("sd"):
            z = (info.get("shown_mu", info["mu"] + PESSIMISM * (info.get("sdm") or 0)) -
                 math.log(max(1e-3, 1 - min(r["acc"], 0.999)))) / info["sd"]
        title = f"{r['artist']} - {r['title']} [{r['version']}]" if r["title"] else info.get("title") or r["sha256"]
        out.append({"key": r["key"], "played": r["played"], "title": title, "rate": r["rate"], "keys": r["keys"],
                    "actual": actual, "expected": info.get("expected"), "z": z, "prediction": info,
                    "sha256": r["sha256"], "mods": json.loads(r["mods"] or "[]"), "stats": st,
                    "pause_positions_ms": pauses.get(r["key"]),
                    "recovered": finish.get("recovered", False)})
    out.sort(key=lambda r: _ts(r["played"]) or 0, reverse=True)
    if outliers:
        out = sorted((r for r in out if r["z"] is not None and abs(r["z"]) >= 1.5), key=lambda r: -abs(r["z"]))
    return out[:limit] if limit else out


def history_text(rows):
    lines = []
    for r in rows:
        when = time.strftime("%m-%d %H:%M", time.localtime(_ts(r["played"]))) if _ts(r["played"]) else "unknown time"
        rate = f"{r['rate']:.2f}×" if r["rate"] else "unknown rate"
        expected = f"expected {100 * r['expected']:.2f}%" if r["expected"] is not None else "expected % not recorded"
        z = f" · {r['z']:+.2f}σ ({'above' if r['z'] > 0 else 'below'} expected)" if r["z"] is not None else ""
        pauses = r.get("pause_positions_ms")
        pause_note = (f" · {len(pauses)} play pause{'s' if len(pauses) != 1 else ''} (duration unknown)"
                      if pauses else "")
        lines.append(f"{when} · {r['title']} · {rate}\n  {100 * r['actual']:.2f}% · {expected}{z}"
                     + (" · recovered from lazer" if r["recovered"] else "") + pause_note)
    return "\n\n".join(lines) or "No plays match this view yet."


# ---------------------------------------------------------------------------
# the recommender
# ---------------------------------------------------------------------------
class Recommender:
    def __init__(self, db, pub=None, keys=None, progress=None):
        """progress(text, done, total): startup steps a first run waits on (download, fit)."""
        self.db = db
        self._candidate_keys = recdata.normalize_keys(keys)
        self.pub = pub or recdata.load_public()
        cid = recdata.calc_id()
        stale = self.pub and self.pub.get("pop", {}).get("version", 1) < recdata.POP_VERSION
        # Source installs download too: a Linux git clone has no evidence until it does (friend 2026-10-01
        # sat at "Loading prediction model"), and after `git pull` its evidence belongs to the old calculator.
        if not self.pub or self.pub.get("calc") != cid or (stale and getattr(sys, "frozen", False)):
            def report(done, total, t0=time.monotonic()):
                if progress:
                    rate = done / max(1e-3, time.monotonic() - t0)
                    left = f" · ~{(total - done) / rate:.0f} s left" if total and done < total and rate > 0 else ""
                    progress(f"Downloading population data (what other players score) · {done / 1e6:.0f}"
                             + (f" / {total / 1e6:.0f}" if total else "") + f" MB{left}", done, total)
            try:
                self.pub = recdata.fetch_release_public(report)
            except (OSError, ValueError, pickle.UnpicklingError) as exc:
                if not self.pub:   # older evidence still works; only a missing one is fatal
                    raise RuntimeError(f"could not download the population data from {recdata.RELEASE_REPO} "
                                       f"({exc}); check the internet connection and restart ManiaScope")
        if not self.pub:
            raise RuntimeError("no population data yet; restart ManiaScope while online to download it")
        if self.pub.get("calc") != cid:
            # its skill features mean something else (or lack skills): never mix them with this calculator's
            raise RuntimeError(f"the population data belongs to another ManiaScope version ({self.pub.get('calc')}, "
                               f"now {cid}) and none is published for this one yet; update with git pull")
        if progress:
            progress("Fitting your prediction model to your scores…", 0, 0)
        recdata.import_public_user(db, self.pub)
        self.refit()

    # -- fitting ---------------------------------------------------------
    def refit(self):
        self.__dict__.pop("_typ", None)
        if not hasattr(self, "feats") or self.feats.calc != recdata.calc_id():
            self.feats = FeatureSource(self.db, self.pub)
        self.installed = installed_by_md5(self.db)
        self.local_shas = {r[0] for r in self.db.execute("SELECT sha256 FROM installed")}
        rows, self.skipped = evidence_rows(self.db, self.pub, self.feats)
        self.rows = rows
        session = Session(load_events(self.db))
        begin = min((e["t"] for e in session.events if e["kind"] == "start"), default=float("inf"))
        historical = [r for r in rows if not r["ts"] or r["ts"] < begin]
        levels = play_levels(historical)
        self.model = Personal(self.pub, [dict(r, base=base_of(self.pub["pop"], r["f"], r["b"], round(r["rate"], 3),
                                                              levels.get(r["keys"]))[0])
                                         for r in historical], levels=levels)
        self.accuracy_model=None
        if self.pub.get('pop_lazer'):
            display_pub=dict(self.pub,pop=self.pub['pop_lazer'])
            display_rows=[dict(r,y=r['y_lazer'],base=base_of(display_pub['pop'],r['f'],r['b'],round(r['rate'],3),
                                                             levels.get(r['keys']))[0])
                          for r in historical]
            self.accuracy_model=Personal(display_pub,display_rows,levels=levels)
        self._baseline_begin = begin
        self.ledger, self.ledger_missing = build_ledger(self.db, self.pub, self.installed)
        import pp_playlist
        self.pp_ranks, self.pp_reference = pp_playlist.reference(self.db, self.ledger)
        self._score_cache = {}
        self._typ = {}
        self._fit_shown()
        self._candidates()
        self.predictor = Predictor(self)

    def observe(self, key):
        """A result updates the ledger and session, not the frozen baseline model.

        Rebuilding that model on every play both blocked Next and decoded the
        entire exact-rate cache. Session correction already consumes this result.
        Historical refits remain for startup/import/newly analysed old scores.
        """
        r = self.db.execute("SELECT * FROM scores WHERE key=?", (key,)).fetchone()
        if r is None:
            return
        bid = r["beatmap_id"]
        if r["ranked"] and bid and bid > 0 and ranked_status(r, self.pub, self.installed) in recdata.RANKED_STATUS \
                and (r["src"] != "realm" or r["online_id"] or r["client"] == "lazer-live"):
            pp = r["pp"] if r["pp"] is not None else score_pp(r)
            if pp is not None:
                with self.db:
                    self.db.execute("UPDATE scores SET pp=? WHERE key=?", (pp, key))
                if pp > self.ledger.best.get(bid, (0.,))[0]:
                    self.ledger = Ledger(self.ledger.best | {bid: (pp, "lazer", r["played"])})
                    import pp_playlist
                    self.pp_ranks, self.pp_reference = pp_playlist.reference(self.db, self.ledger)
        self._score_cache.clear()

    def _fit_shown(self):
        """The model works in acc320 (pp accuracy); the player reads lazer's accuracy. Map one to the other
        by the user's own judgement mix: log(1 − shown) = a + b·log(1 − acc320), fit on their scores."""
        xs, zs = [], []
        until = getattr(self, '_baseline_begin', float('inf'))
        query = "SELECT stats FROM scores WHERE src IN ('realm', 'live') AND stats IS NOT NULL"
        args = ()
        if math.isfinite(until):
            query += " AND julianday(played)<julianday(?,'unixepoch')"
            args = (until,)
        for (st,) in self.db.execute(query,args):
            st = json.loads(st)
            a, n = recdata.acc320(st)
            al = recdata.acc_lazer(st)
            if n and 0 < a < 1 and 0 < al < 1:
                xs.append(math.log(1 - a))
                zs.append(math.log(1 - al))
        self._shown = tuple(np.polyfit(xs, zs, 1)[::-1]) if len(xs) >= 20 else (math.log(0.75), 1.0)

    def shown(self, acc):
        """acc320 → the accuracy lazer would show for it."""
        a, b = self._shown
        return max(0., min(1., 1 - math.exp(min(20., a + b * math.log(max(1e-9, 1 - acc))))))

    def _candidates(self):
        """Every ranked/loved difficulty × NM/HT/DT with features, predicted once (before session form)."""
        self._candidate_keys = recdata.normalize_keys(getattr(self, '_candidate_keys', None))
        m, C = self.model, []
        for b, fs in self.pub["feats"].items():
            meta = self.pub["maps"].get(b)
            if meta:
                live=self.installed.get(meta.get('md5'),{})
                if live.get('status') is not None:
                    # The archive may still call a newly ranked map Qualified.
                    # An installed exact revision carries the newer live status;
                    # do not drop its already available features from PP.
                    meta=dict(meta,approved=live['status'])
            if not meta or meta["approved"] not in recdata.RANKED_STATUS + (4,):
                continue
            for rate, var in recdata.VARIANTS:
                f = fs.get(rate)
                if f and f['keys'] in self._candidate_keys and "stars" in f and f["notes"] >= 100:
                    C.append((b, rate, var, f, meta) + m.predict(f, meta["md5"], b, rate))
        for inst in self.newer_ranked():               # ranked after the snapshot: chart features only
            meta = {"approved": inst["status"], "md5": inst["md5"], "set": inst["set_id"],
                    "file": f'{inst["artist"]} - {inst["title"]} [{inst["version"]}]'}
            for rate, var in recdata.VARIANTS:
                f = self.feats.lookup(self.feats.key(inst["sha256"], rate))
                if f and f['keys'] in self._candidate_keys and "stars" in f and f["notes"] >= 100:
                    C.append((inst["beatmap_id"], rate, var, f, meta)
                             + m.predict(f, inst["md5"], self.feats.md5_bid.get(inst["md5"]), rate))
        self.cands = C
        self.A = {k: np.array(v) for k, v in {
            "bid": [c[0] for c in C], "rate": [c[1] for c in C], "keys": [c[3]["keys"] for c in C],
            "mu": [c[5] for c in C], "sdm": [c[6] for c in C], "sda": [c[7] for c in C],
            "scale": [pp_scale(c[3]["stars"], c[3]["hits"]) for c in C],
            "ranked": [c[4]["approved"] in recdata.RANKED_STATUS for c in C]}.items()}
        self.A["skill"] = [top_skill(c[3]) for c in C]
        self.A["identity"] = np.array([event_key({"md5": c[4]["md5"], "bid": c[0]}) for c in C])
        display=getattr(self,'accuracy_model',None)
        if display is not None:
            predictions=[display.predict(c[3],c[4]['md5'],c[0],c[1]) for c in C]
            for i,name in enumerate(('display_mu','display_sdm','display_sda')):
                self.A[name]=np.array([p[i] for p in predictions])
        self._score_cache = {}

    def newer_ranked(self):
        """Installed ranked/approved maps without snapshot features: ranked since, or their chart was not in
        the dump's archive (lazer keeps the status current)."""
        known, feats = self.feats.md5_bid, self.pub["feats"]
        return [i for i in self.installed.values() if i["status"] in recdata.RANKED_STATUS and i["beatmap_id"] > 0
                and 4 <= (i["keys"] or 0) <= 10 and known.get(i["md5"]) not in feats]

    def score(self, session=None, now=None, keys=None):
        """→ (list of dicts, one per difficulty = its best variant, with value and reasons; session).
        Vectorised over all variants; every credible difficulty stays eligible."""
        session = session or Session(load_events(self.db), now)
        session.bind_warmup(getattr(self.model,'warmup',None))
        cache_key = (recdata.normalize_keys(keys), session.model_key())
        cached = self._score_cache.get(cache_key)
        if cached is not None:
            return cached, session
        if not cache_key[0]:
            return [], session
        if cache_key[0] != self._candidate_keys:
            self._candidate_keys = cache_key[0]
            self._candidates()
        if not self.cands:
            return [], session
        A, cands, L = self.A, self.cands, self.ledger
        F = np.array([session.correction(k,sk,c[3])+session.warmup_penalty(k,sk,c[3]['length']/1000/c[1],c[3],
                          self.model.rate_response(c[3],c[4]['md5'],c[0],c[1]),
                          lambda loss, c=c: self.model.rate_loss(c[3],c[4]['md5'],c[0],c[1],loss))
                      for k,sk,c in zip(A['keys'].tolist(),A['skill'],cands)])
        tried = np.isin(A["identity"], list(session.played))
        penalties = np.array([session.rate_penalty(c[4]["md5"], c[1]) for c in cands])
        learned=np.array([session.chart_correction(c[4]['md5'],c[1],c[3],self.model,A['sdm'][i]**2)
                          for i,c in enumerate(cands)])
        mu = A["mu"] + F + penalties + learned - self.model.retry * tried
        Y = (mu + PESSIMISM * A["sdm"])[:, None] + A["sda"][:, None] * Z_Q[None, :]
        PP = A["scale"][:, None] * np.maximum(0.0, 5 * (1 - np.exp(Y)) - 4) * A["ranked"][:, None]
        best = np.array([L.best.get(b, (0.0,))[0] for b in A["bid"].tolist()])
        gain = L.delta_many(best, PP).mean(axis=1)
        need = (best / np.maximum(A["scale"], 1e-9) + 4) / 5          # acc320 that would beat the counted best
        p_up = p_beat(mu + PESSIMISM * A["sdm"], A["sda"], need) * A["ranked"]
        typ = np.array([self.typical(k) if self.typical(k) is not None else np.nan for k in A["keys"].tolist()])
        chal = np.nan_to_num((mu - typ) / A["sda"])
        import pp_playlist
        normal = chal <= 1.25
        ready = {(k, sk): session.push(k, sk) for k, sk in set(zip(A["keys"].tolist(), A["skill"]))}
        # Challenge is in attempt sds; each staircase step raises the ceiling by one step.
        ceiling = np.array([1.25 + PUSH_STEP * min(4., ready[(k, sk)])
                            for k, sk in zip(A["keys"].tolist(), A["skill"])])
        sustainable = chal <= ceiling
        credible = sustainable & (p_up >= np.where(normal, .12, .25))
        # The user dislikes HT: offer it only as a likely improvement (2026-09-30).
        credible &= (A["rate"] >= 1.) | ((p_up >= HT_MIN_P_UP) & (gain >= HT_MIN_GAIN))
        # The first play in an unactivated keymode is a comfortable probe. A
        # long 7K session never authorises cold 10K DT through global warmup.
        cold = {k: session.activation(k) < 1.5 for k in recdata.SUPPORTED_KEYS}
        display=getattr(self,'accuracy_model',None)
        if display is not None:
            view=session.for_accuracy(display.warmup)
            shift=np.array([view.correction(k,sk,c[3])+view.warmup_penalty(k,sk,c[3]['length']/1000/c[1],c[3],
                            display.rate_response(c[3],c[4]['md5'],c[0],c[1]),
                            lambda loss,c=c:display.rate_loss(c[3],c[4]['md5'],c[0],c[1],loss))
                            for k,sk,c in zip(A['keys'].tolist(),A['skill'],cands)])
            local=np.array([view.chart_correction(c[4]['md5'],c[1],c[3],display,A['display_sdm'][i]**2)
                            for i,c in enumerate(cands)])
            display_mu=A['display_mu']+shift+penalties+local-display.retry*tried
            display_sd=np.hypot(A['display_sdm'],A['display_sda'])
        else:
            display_mu=self._shown[0]+self._shown[1]*mu
            display_sd=self._shown[1]*np.hypot(A['sdm'],A['sda'])
        shown_acc = np.clip(1-np.exp(np.minimum(20.,display_mu)), 0., 1.)
        credible &= np.array([not cold.get(k, True) or a >= pp_playlist.comfort(self, k)
                              for k, a in zip(A["keys"].tolist(), shown_acc)])
        # Choose the variant for this readiness BEFORE reducing a chart to one
        # entry. Otherwise a high-gain DT variant hides its safer NM/HT option,
        # even though the playlist later tries to prefer comfortable choices.
        state = {(k,sk): pp_playlist.readiness(session,k,sk) for k,sk in ready}
        acc_target, pp_target = {}, {}
        for (k,sk), readiness in state.items():
            ref = getattr(self,"pp_reference",{}).get(k,{})
            acc_target[k,sk] = (1-readiness)*ref.get("comfort",.975)+readiness*ref.get("peak_acc",.93)
            pp_target[k,sk] = ((1-readiness)*ref["safe_pp"]+readiness*ref["peak_pp"]) if ref else None
        central_pp = A["scale"] * np.maximum(0., 1.-5.*np.exp(np.minimum(20.,mu)))
        target_a = np.array([acc_target[k,sk] for k,sk in zip(A["keys"].tolist(),A["skill"])])
        target_p = np.array([pp_target[k,sk] or p for k,sk,p in zip(A["keys"].tolist(),A["skill"],central_pp)])
        accuracy_fit = np.exp(-.5*(np.maximum(0.,target_a-shown_acc)/.025)**2)
        tier_fit = np.exp(-.5*(np.log(np.maximum(1.,central_pp)/np.maximum(1.,target_p))/.35)**2)
        utility = .3+np.log1p(np.maximum(0.,gain)/.05)**.65
        variant_value = utility * (.2+.8*tier_fit) * accuracy_fit * p_up
        # A comfortable zero-gain variant must not hide another variant which
        # passes readiness and offers a real gain. Keep the former only as a
        # possible warmup when no useful variant of that chart exists.
        order = np.lexsort(((A["rate"] != 1.0).astype(int), -variant_value, -(gain>=MIN_GAIN).astype(int)))
        order = order[credible[order]]
        seen, out = set(), []
        for ix in order:
            b = int(A["bid"][ix])
            if b in seen:
                continue
            seen.add(b)
            b_, rate, var, f, meta = cands[ix][:5]
            pp = PP[ix]
            inst = self.installed.get(meta["md5"], {})
            out.append({"bid": b, "rate": rate, "var": var, "keys": f["keys"], "title": meta["file"], "set": meta["set"],
                        "sha": inst.get("sha256"), "mode": "pp",
                        "online_match": not inst or bool(inst.get("online_md5") == meta["md5"] or
                                                          self.pub["maps"].get(b, {}).get("md5") == meta["md5"]),
                        "search_meta": {"Artist": inst.get("artist"), "Title": inst.get("title"),
                                        "Version": inst.get("version"), "Creator": inst.get("creator")} if inst else {},
                        "md5": meta["md5"], "ranked": bool(A["ranked"][ix]), "gain": float(gain[ix]), "p_up": float(p_up[ix]),
                        "best": float(best[ix]) or None, "pp_lo": float(pp[-5]), "pp_hi": float(pp[4]),     # outcomes run best → worst
                        "best_rank": getattr(self, "pp_ranks", {}).get(b, 100000),
                        "pp_mid": float(A["scale"][ix] * max(0., 1.-5.*math.exp(min(20., mu[ix])))),
                        "acc_mid": float(shown_acc[ix]), "mu": float(mu[ix]),
                        "acc_lo": max(0.,1-math.exp(min(20.,display_mu[ix]+BAND_Z*display_sd[ix]))),
                        "acc_hi": max(0.,1-math.exp(min(20.,display_mu[ix]-BAND_Z*display_sd[ix]))),
                        "push_only": not bool(normal[ix]),
                        "sd_model": float(A["sdm"][ix]), "sd": float(A["sda"][ix]), "challenge": float(chal[ix]),
                        "skill": A["skill"][ix], "length": f["length"] / 1000 / rate, "installed": meta["md5"] in self.installed,
                        "stars": f["stars"], "f": f})
        self._score_cache = {cache_key: out}
        return out, session

    def typical(self, k):
        """The keymode's recent typical y (median of the last ~60 days of attempts, else all)."""
        cache = self.__dict__.setdefault("_typ", {})
        if k not in cache:
            ys = [r["y"] for r in self.rows if r["keys"] == k and r["src"] != "public" and r["t"] >= self.model.now - 2]
            ys = ys or [r["y"] for r in self.rows if r["keys"] == k]
            cache[k] = float(np.median(ys)) if ys else None
        return cache[k]

    def pool(self, session=None, n_show=20, now=None, rng=None, keys=None, choose=True, avoid=None):
        """Preview a broad pool; Next samples all credible maps, never just its top 25."""
        import pp_playlist
        cands, session = self.score(session, now) if keys is None else self.score(session, now, keys=keys)
        allowed = recdata.normalize_keys(keys)
        phase = session.phase()
        context = (session, session.model_key(), session._warmup_model, allowed,
                   getattr(self, 'ledger', None), getattr(self, 'pp_reference', None))
        cached = getattr(self, '_pp_pool_cache', None)
        if cached is not None and cached[0] is cands and cached[1] == context:
            pick_from, relaxed = cached[2]
        else:
            filtered = [c for c in cands if c["keys"] in allowed]
            pick_from, relaxed = pp_playlist.prepare(self, filtered, session)
            if not pick_from and phase == "warmup":
                k0 = session.current_keys() or self.home_keys()
                if k0 not in allowed:
                    k0 = next(iter(allowed), None)
                pick_from = self.warmups(filtered, session, k0) if k0 is not None else []
            self._pp_pool_cache = (cands, context, (pick_from, relaxed))
        seed = max((e.get("id", e["t"]) for e in session.all
                    if e["kind"] in ("offer", "finish", "abort", "reset")), default=0)
        self.pp_choices = pick_from
        shown = pp_playlist.preview(pick_from, n_show, seed) if n_show else []
        top = sum(c.get("best_rank", 101) <= 100 for c in pick_from)
        note = (f"{len(pick_from):,} eligible opportunities · {top} current top-100 maps · "
                "Next follows the displayed order; predictions refresh in the background" if pick_from else
                "No credible PP opportunities at this readiness — try skill practice or an easier manual map")
        if relaxed:
            note += " · freshness relaxed: the credible pool has been cycled"
        pick = None
        if pick_from and choose:
            pick_from = [c for c in pick_from if event_key(c) != avoid] or pick_from
            rng = rng or random.Random()
            wts = [max(1e-9, c["weight"]) for c in pick_from]
            pick = rng.choices(pick_from, weights=wts)[0]
        return phase, shown, pick, note

    def home_keys(self):
        contribution = collections.Counter()
        for i, (bid, best) in enumerate(sorted(self.ledger.best.items(), key=lambda p: -p[1][0])):
            k = self.pub["maps"].get(bid, {}).get("keys")
            if k in recdata.SUPPORTED_KEYS:
                contribution[k] += best[0] * .95 ** i
        if contribution:
            return contribution.most_common(1)[0][0]
        recent = [r["keys"] for r in self.rows if r["src"] != "public" and r["t"] >= self.model.now - 1]
        return collections.Counter(recent).most_common(1)[0][0] if recent else 7

    def warmups(self, cands, session, k0):
        """Comfortable maps the user already knows (installed, any status, their own rate), else easy ranked/loved."""
        known = collections.defaultdict(list)
        for r in self.rows:
            if r["src"] != "public" and r["keys"] == k0:
                known[(r["chart"], round(r["rate"], 3))].append(r)
        out = []
        for (chart, rate), rs in known.items():
            r = rs[-1]
            inst = self.installed.get(chart) or next((v for v in self.installed.values() if v["sha256"] == chart), None)
            if not inst:
                continue
            expected = self.predictor.predict(r["f"], chart, r["b"], rate, session)
            mu, sda = expected['mu'], expected['sda']
            if expected['opening_acc'] < .937:
                continue
            typ = self.typical(r["keys"])
            ch = (mu - typ) / sda if typ is not None else 0
            if not -2.2 <= ch <= -0.3:
                continue
            fr = session.freshness(event_key({"md5": chart, "sha": inst["sha256"]}), r["f"]["length"] / 1000 / rate, mode="pp")
            out.append({"bid": navigation.positive_id(inst["beatmap_id"]), "sha": inst["sha256"], "rate": rate, "var": rate_label(rate),
                        "mode": "pp", "online_match": bool(inst.get("online_md5") and inst["online_md5"] == inst["md5"]),
                        "search_meta": {"Artist": inst["artist"], "Title": inst["title"], "Version": inst["version"], "Creator": inst.get("creator")},
                        "keys": r["keys"], "title": f'{inst["artist"]} - {inst["title"]} [{inst["version"]}]',
                        "ranked": inst["status"] in recdata.RANKED_STATUS, "gain": 0.0, "best": None, "installed": True,
                        "acc_mid": expected['acc_mid'], "opening_acc": expected['opening_acc'],
                        "challenge": ch, "length": r["f"]["length"] / 1000 / rate,
                        "purpose": "warmup", "weight": math.exp(-(ch + 1.0) ** 2) * fr * (1 + min(len(rs), 5) / 5)
                        * math.exp(-max(0.0, r["f"]["length"] / 1000 / rate - 240) / 120),      # short ones warm up
                        "why": f"warmup · played {len(rs)}×", "skill": top_skill(r["f"]), "md5": chart, "f": r["f"]})
        out.sort(key=lambda c: -c["weight"])
        return out

    # -- attempts --------------------------------------------------------
    def expectation(self, sha, path, rate, compute=True):
        """Pre-play expectation for a chart about to be played (frozen into the start event)."""
        import hashlib
        with open(path, "rb") as fh:
            md5 = hashlib.md5(fh.read()).hexdigest()
        if rate is None:
            return {"md5": md5}
        r = {"md5": md5, "beatmap_id": None, "rate": rate, "sha256": sha}
        f, b = self.feats(r)
        if not f:
            if not compute:
                return {"md5": md5}
            try:
                f = recdata.chart_feats(path, (round(rate, 3),))[round(rate, 3)]
            except Exception:
                return {"md5": md5}
        session = Session(load_events(self.db))
        e = self.predictor.predict(f, md5, b, round(rate, 3), session)
        meta = self.pub["maps"].get(b, {}) if b else {}
        return {**e, "md5": md5, "bid": b, "keys": f["keys"], "skill": top_skill(f),
                "ranked": meta.get("approved") in recdata.RANKED_STATUS, "length": f["length"] / 1000 / rate,
                "f": f, "sdm": e["sdm"]}

    def card(self, path, rate, compute=True):
        """Stats for the chart lazer has selected, at lazer's rate: the same numbers a list entry shows."""
        import hashlib
        with open(path, "rb") as fh:
            sha = hashlib.sha256(fh.read()).hexdigest()
        e = self.expectation(sha, path, rate, compute=compute)
        if "mu" not in e:
            return None
        f, b = e["f"], e["bid"] or self.installed.get(e["md5"], {}).get("beatmap_id") or None
        ranked = (e["ranked"] or self.installed.get(e["md5"], {}).get("status") in recdata.RANKED_STATUS) \
            and round(rate, 2) in (0.75, 1.0, 1.5) and "stars" in f
        inst = self.installed.get(e["md5"])
        metadata = None
        if not inst:
            import lazer_index
            metadata = lazer_index._read_metadata(path)
        c = {"bid": b, "md5": e["md5"], "keys": f["keys"], "title": f'{inst["artist"]} - {inst["title"]} [{inst["version"]}]'
             if inst else os.path.basename(path), "var": rate_label(rate), "rate": rate, "ranked": ranked,
             "search_meta": {"Artist": inst["artist"], "Title": inst["title"], "Version": inst["version"]} if inst else {},
             "acc_mid": e["acc_mid"], "acc_lo": e["acc_lo"],
            "acc_hi": e["acc_hi"], "length": e["length"], "skill": e["skill"],
            "nps": f.get("nps", {}).get("nps")}
        if metadata:
            c['search_meta'] = metadata
            c['title'] = f"{metadata.get('Artist', '')} - {metadata.get('Title', '')} [{metadata.get('Version', '')}]"
        if ranked:
            c.update(pp_estimates(e, f, b, self.ledger))
        c["plays_like"] = plays_like(self.pub.get("pop_lazer") or self.pub["pop"], f, b, rate)
        return c


def plays_like(pop, f, b, rate, min_scores=30, min_shift=.05):
    """(stars, scores) the public scores say a chart plays like at this rate — its stars moved by its
    map effect — or None with few scores or a shift under min_shift (user 2026-10-01: Aim Burst 12.5★
    has 98 % top scores; the same players score 92 % on a 12.5★ Camellia map)."""
    if b is None:
        return None
    mb, n = pop["mbr"].get((b, round(rate, 3)), pop["mb"].get(b, (0., 0)))
    if n < min_scores:
        return None
    # Invert the whole link (slope·x + curve·(x − knee)²), not the slope alone: above the knee the
    # link is ~3× steeper at 9★, and slope-only turned Chinmoku [Midnight Symphony] 1.5× (displayed
    # 9.2★, mb −.17) into "plays like 7.4★" where the full link gives 8.9★ (user 2026-10-02).
    k = f["keys"]
    s = pop["slope"].get(k, pop["slope"][0])
    c = pop.get("curve", {}).get(k, pop.get("curve", {}).get(0, 0.))
    knee = math.log(pop.get("curve_knee", 4.))
    g = lambda x: s * x + c * max(0., x - knee) ** 2
    lr = math.log(max(.05, f["overall"]))
    target = g(lr) + mb                                 # more misses = harder
    if target <= g(knee) or c <= 0:
        x = target / s                                  # the link is linear below the knee
    else:
        x = knee + (-s + math.sqrt(s * s + 4 * c * (target - s * knee))) / (2 * c)
    ratio = math.exp(x - lr)
    if abs(ratio - 1) < min_shift:
        return None
    # The card shows display stars; scale the rating before the display conversion, not after it.
    shown = dict(f, overall=f["overall"] * ratio,
                 **({"preunit_overall": f["preunit_overall"] * ratio} if "preunit_overall" in f else {}))
    return score_units.displayed_feature(shown), int(n)


def pp_estimates(e, f, bid, ledger):
    acc = 1 - np.exp(e["mu"] + PESSIMISM * e["sdm"] + e["sda"] * Z_Q)
    scale = pp_scale(f["stars"], f["hits"])
    pp = scale * np.maximum(0.0, 5 * acc - 4)
    best = ledger.best.get(bid, (0.0,))[0] if bid else 0.0
    return dict(pp_lo=float(pp[-5]), pp_hi=float(pp[4]), best=best or None,
                gain=float(ledger.delta_many(np.array([best]), pp[None])[0].mean()),
                p_up=float(p_beat(e["mu"] + PESSIMISM * e["sdm"], e["sda"], (best / max(scale, 1e-9) + 4) / 5)))


def top_skill(f):
    sk = {k: v for k, v in f["sk"].items() if k != "stamina"}
    return max(sk, key=sk.get) if sk else None


def skill_family(skill):
    groups = {"rice": ("stream", "delay", "dump", "jumpstream", "handstream"),
              "chords": ("chordstream", "bracket", "jumptrill", "splittrill"),
              "jack": ("jack", "jackspeed", "minijack", "longjack", "chordjack", "quadstream"),
              "sv": ("sv", "sv_fast", "sv_slow", "sv_accel", "sv_stutter", "sv_brake"),
              "ln": ("ln", "release", "hybrid", "inverse", "shield"),
              "tech": ("technical", "patterntech", "rhythmtech", "trill1h", "anchor")}
    return next((g for g, names in groups.items() if skill in names), skill)


def skill_related(a, b):
    if a == b:
        return 1.
    x, y = skill_family(a), skill_family(b)
    if x == y:
        return .85
    if x in ("rice", "chords", "tech") and y in ("rice", "chords", "tech"):
        return .5
    if {x, y} == {"jack", "chords"}:
        return .3
    return 0.


def rate_label(rate):
    return {0.75: "HT", 1.0: "NM", 1.5: "DT"}.get(round(rate, 2), f"{rate:.2f}×")


def reason(c):
    if c["best"]:
        return f"refarm · best {c['best']:.0f}pp · {100 * c['p_up']:.0f}% to beat"
    return f"new map · {100 * c['p_up']:.0f}% to count"


lazer_running = navigation.lazer_running


def event_key(c):
    """Exact revision across rates; a sentinel online ID is never an identity."""
    if c.get("md5"):
        return "md5:" + str(c["md5"])
    if c.get("sha") or c.get("sha256"):
        return "sha:" + str(c.get("sha") or c.get("sha256"))
    bid = navigation.positive_id(c.get("bid"))
    return f"bid:{bid}" if bid else None


def legacy_key(value):
    text = str(value or "")
    if text.startswith(("md5:", "sha:", "bid:")):
        return text
    if len(text) == 32:
        return "md5:" + text
    if len(text) == 64:
        return "sha:" + text
    bid = navigation.positive_id(value)
    return f"bid:{bid}" if bid else None


class Predictor:
    """Immutable model snapshot shared with rate work; no DB or GTK ownership."""
    def __init__(self, rec, *, model=None, shown=None):
        self.model, self.pub = model or rec.model, rec.pub
        self.calc, self.md5_bid = rec.feats.calc, rec.feats.md5_bid
        self.shown = rec._shown if shown is None else shown
        self._base_cache = {}
        direct=getattr(rec,'accuracy_model',None)
        self.display=Predictor(rec,model=direct,shown=(0.,1.)) if model is None and direct is not None else None

    def predict(self, f, chart, bid, rate, session, *, accuracy_only=False):
        # NPS/Skills rank displayed accuracy; PP and recorded play forecasts
        # still compute both targets. Preserve the session's readiness model.
        session.bind_warmup(getattr(self.model,'warmup',None))
        out={} if accuracy_only and getattr(self,'display',None) is not None else self._predict(f,chart,bid,rate,session)
        if getattr(self,'display',None) is not None:
            import accuracy_targets
            view=session.for_accuracy(self.display.model.warmup)
            expected=self.display._predict(f,chart,bid,rate,view)
            out.update({'display_'+k:expected[k] for k in accuracy_targets.FIELDS})
            # PP means/spreads remain untouched; playlist quality and displayed
            # intervals use the directly fitted display target.
            out.update({k:expected[k] for k in ('acc_mid','acc_lo','acc_hi','opening_acc','activation_minutes','cold_scale',
                                               'rate_slope','sd_model','warmup_tau')})
        return out

    def _predict(self, f, chart, bid, rate, session):
        import warmup
        session.bind_warmup(getattr(self.model,'warmup',None))
        cache = self.__dict__.setdefault("_base_cache", {})
        ck = (chart, bid, rate)
        cached = cache.get(ck)
        if cached is None or cached[0] is not f:
            slope = self.model.rate_response(f, chart, bid, rate) if hasattr(self.model, 'rate_response') else None
            cached = (f, self.model.predict(f, chart, bid, rate), slope, warmup.family_weights(f), top_skill(f))
            cache[ck] = cached
        mu, sdm, sda = cached[1]
        base_mu = mu
        slope = cached[2]
        skill = cached[4]
        response = (lambda loss: self.model.rate_loss(f,chart,bid,rate,loss)) if hasattr(self.model,'rate_loss') else None
        cold = session.warmup_penalty(f['keys'],skill,f.get('length',0.)/1000/max(.01,rate),f,slope,response,cached[3])
        mu += (session.correction(f["keys"], skill, f) + cold
               + session.rate_penalty(chart, rate)
               + session.chart_correction(chart,rate,f,self.model,sdm**2))
        if "md5:" + chart in session.played or "sha:" + chart in session.played:
            mu -= self.model.retry
        a, b = self.shown
        def shown(y):
            return max(0., min(1., 1 - math.exp(min(20., a + b * y))))
        mid, spread = mu, math.hypot(sda, sdm)
        # A first map cannot borrow readiness from its own future outro. Keep
        # the full-map prediction, but also expose the opening-state estimate
        # so the playlist can reject an overly hard cold opening.
        opening = session.warmup_penalty(f['keys'], skill, 0., f,slope,response,cached[3])
        return {"mu": mu, "base_mu": base_mu, "cold_penalty": cold, "cold_scale": session.cold_scale(f['keys']),
                "activation_minutes":session.activation_for(f,cached[3]),
                "warmup_tau":self.model.warmup.parameters(f['keys'])['tau'] if hasattr(self.model,'warmup') else 6.,
                "opening_acc": shown(mu + max(0., opening-cold)),
                "rate_slope": slope,
                "sd_model": sdm, "sdm": sdm, "sda": sda,
                "sd": math.sqrt(sda * sda + sdm * sdm),
                "acc_mid": shown(mid), "acc_lo": shown(mid + BAND_Z * spread),
                "acc_hi": shown(mid - BAND_Z * spread)}


class Worker(threading.Thread):
    """Owns the recommendation state and every write to its event log; the UI only queues tasks and
    renders what emit() hands back ({"type": "status" | "pool" | "target", ...})."""

    def __init__(self, emit, busy=lambda: False):
        super().__init__(daemon=True, name="recommend")
        self.q, self.emit, self.busy = queue.Queue(), emit, busy
        self.rec = self.target = None
        self.mode, self.action, self.revision = "pp", "auto", 0
        self._queued_config_revision = 0
        self.keys = {"pp": recdata.SUPPORTED_KEYS, "nps": (7,), "skills": (7,)}
        self.targets = {m: None for m in MODES}
        self.playlists = {m: [] for m in MODES}
        self._playlist_consumed = collections.deque(maxlen=20)
        self.nps_focus = .5
        self.web_bias = {"nps": .5, "skills": .5}
        self.acc_targets = {"nps": .94, "skills": .94}
        self._selection_id = 0
        self.practice_skills = ()
        self.nps_state = {"candidates": [], "total": 0, "analyzed": 0, "failed": 0, "error": ""}
        self.skills_state = dict(self.nps_state)
        self._local_request_id = 0
        self.builder, self.generation = None, 0
        self._halt = threading.Event()
        self._feature_lock = threading.Lock()
        self._feature_pending, self._feature_thread = None, None
        self._feature_errors = {}
        self._last_idle_check = time.monotonic()
        self._taste_cache = (None, {})
        self._replay_due = time.monotonic() + 10
        self._replay_retries = 0
        self._replay_thread = None
        self._import_thread = None
        self._fill_thread, self._learning, self._first_import = None, None, False
        self._refit_due = False
        self._nps_prediction_context, self._nps_predictions = None, {}
        self._local_pool_cache = {}
        self._local_weights = {}
        self._local_prepared = {}
        self._restored_modes = set()
        self._session_cache = (None, None)
        self._last_refit = time.time()
        self._stats_visible = False
        self._stats_cache = (None, None)
        import hashlib
        import warmup
        digest=hashlib.sha256()
        import personal_support, accuracy_targets, session_form
        for path in (__file__,warmup.__file__,personal_support.__file__,accuracy_targets.__file__,session_form.__file__):
            with open(path,'rb') as fh:digest.update(fh.read())
        self.predictor_id=digest.hexdigest()[:12]

    def put(self, *task):
        if task[0] == 'configure' and len(task) > 4:
            self._queued_config_revision = max(self._queued_config_revision, task[4])
        self.q.put(task)

    def _stage(self, text, done=0, total=0, blocking=False):
        """What startup is waiting on: the status line, its progress bar, and the selected map's card."""
        self._loading = text
        self.emit({"type": "status", "text": text or "", "progress": (done, total) if total else None,
                   "blocking": blocking})
        if not self.rec and getattr(self, "_sel", (None,))[0]:
            self._card()

    def run(self):
        self.db = recdata.connect()
        self._first_import = self.db is not None and recdata.kv_get(self.db, 'realm_import') is None
        self._stage("Opening your saved plays and recommendations…")
        try:
            # The UI queues its real tab/filters before start(). Honour them
            # before launching any invisible local-playlist work.
            pending = []
            while not self.q.empty():
                task = self.q.get_nowait()
                if task[0] == "configure":
                    self.t_configure(*task[1:])
                else:
                    pending.append(task)
            for task in pending:
                self.q.put(task)
            public = recdata.load_public()
            if public:
                # This catalogue lives for the window's lifetime. Repeated
                # full GC scans caused 0.4–0.5 s Next stalls, freeing nothing.
                # Freeze before creating replaceable personal/session models;
                # new objects still receive normal cycle collection.
                gc.collect()
                gc.freeze()
            self.rec = Recommender(self.db, public, keys=self.keys['pp'], progress=self._stage)
            self.generation += 1
            import nps
            self.builder = nps.Builder(lambda kind, data: self.put("nps_update", kind, data), self.busy)
            self.builder.start()
            self._restore_local(self.mode)
            self._request_nps()
            self._publish()
            self._start_fill()
            self._stage(None)
        except Exception as exc:
            traceback.print_exc()
            self._stage(f"Recommendations unavailable: {str(exc)[:240]}", blocking=True)
        self._begin_import()
        while not self._halt.is_set():
            if self._refit_due and not self.busy():
                self.t_refit()
            self._catch_up_replays()
            try:
                task = self.q.get(timeout=5)
            except queue.Empty:
                if self.rec and not self.busy():
                    last_play = self.db.execute("SELECT MAX(t) FROM events WHERE kind IN ('finish','abort')").fetchone()[0]
                    if last_play and self._last_refit < last_play and time.time()-last_play > SESSION_GAP:
                        self.t_refit()
                if not self.busy() and time.monotonic() - self._last_idle_check > 3600:
                    self._last_idle_check = time.monotonic()
                    self._daily_check()
                continue
            started = time.monotonic()
            try:
                getattr(self, "t_" + task[0])(*task[1:])
                took = time.monotonic() - started
                if took > 1.5:      # every later tab switch / Next waits behind it (run.log evidence)
                    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] slow recommender task {task[0]} {took:.1f} s"
                          f" · {self.q.qsize()} queued", flush=True)
            except Exception:
                traceback.print_exc()
                self.emit({"type": "status", "text": f"recommendation task {task[0]} failed (see run.log)"})

    def _catch_up_replays(self):
        """Archive independently: neither a busy queue nor slow Realm I/O blocks Next."""
        if not self._replay_due or self.busy() or time.monotonic() < self._replay_due \
                or self._replay_thread is not None and self._replay_thread.is_alive() \
                or self._import_thread is not None and self._import_thread.is_alive():
            return
        self._replay_due = None
        path = self.db.execute("PRAGMA database_list").fetchone()[2]
        def archive():
            db = None
            try:
                import replays
                db = recdata.connect(path)
                replays.recent(db)
                pending = bool(db.execute("SELECT 1 FROM replay_evidence WHERE status='missing' "
                                          "AND julianday(played)>=julianday('now','-3 days') LIMIT 1").fetchone())
                fresh=[r[0] for r in db.execute("SELECT key FROM scores WHERE fresh=1 AND julianday(played)>=julianday('now','-3 days')")]
                self.put("replay_done", pending, None, fresh)
            except Exception as exc:
                self.put("replay_done", True, str(exc)[:120])
            finally:
                if db is not None:
                    db.close()
        self._replay_thread = threading.Thread(target=archive, daemon=True, name="replay-archive")
        self._replay_thread.start()

    def t_replay_done(self, pending, error=None, fresh=()):
        if self.rec and fresh:
            for key in fresh:
                self.rec.observe(key)
            self._request_nps()
            self._publish()
        if pending:
            self._replay_retries += 1
            if self._replay_retries <= 3:
                self._replay_due = time.monotonic() + 20
        else:
            self._replay_retries = 0
        if error:
            self.emit({"type": "status", "text": "Replay archive pending: " + error})

    def _start_fill(self):
        """One background analysis at a time; an import that brings new plays starts another."""
        if self._fill_thread is None or not self._fill_thread.is_alive():
            if self._first_learning():
                self._learning = "Checking which of your played charts still need analysing…"
            self._fill_thread = threading.Thread(target=self._fill, daemon=True, name="features")
            self._fill_thread.start()

    def _fill(self):
        """Features for the user's own charts that the public table lacks (custom rates, unranked maps).
        A first run imports thousands of plays after this started once with none (fresh install
        2026-10-02: 5,632 chart/rates never analysed, so PP suggested 4K Beginner maps to a 7K player)."""
        db = recdata.connect()
        try:
            fs = FeatureSource(db, self.rec.pub)
            need = fs.missing(db.execute("SELECT sha256, md5, beatmap_id, rate FROM scores WHERE sha256 IS NOT NULL"))
            need += fs.missing([{"sha256": i["sha256"], "md5": i["md5"], "beatmap_id": None, "rate": r}
                                for i in self.rec.newer_ranked() for r, _v in recdata.VARIANTS])
            if need:
                t0, refits, first = time.monotonic(), [0], self._first_learning()

                def progress(done, total):
                    rate = done / max(1e-3, time.monotonic() - t0)
                    left = (total - done) / rate if rate > 0 else 0
                    eta = f" · ~{left / 60:.0f} min left" if left >= 90 else f" · ~{left:.0f} s left" if left > 0 else ""
                    text = (f"Analysing the charts you have played, to learn your level · {done:,} / {total:,} charts{eta}"
                            if first else f"Analysing {total:,} new or updated maps in the background · {done:,} done{eta}")
                    self._learning = text if first else None
                    self.emit({"type": "status", "text": text, "progress": (done, total), "learning": first})
                    # Only while the level is unknown: each refit restarts the NPS/Skills build.
                    if first and done - refits[0] >= max(200, total // 5) and done < total:
                        refits[0] = done
                        self.put("refit")           # suggestions sharpen while the rest is analysed
                progress(0, len({sha for sha, _r in need}))
                import webmaps
                # First learning is all the player waits for: every core but two, paused while playing.
                n = webmaps._workers() if first else 1
                fs.fill(need, stop=self.busy, workers=n, chunk=max(12, 4 * n),
                        cancel=self._halt.is_set, progress=progress)
                self._learning = None
                self.put("refit")
                self.emit({"type": "status", "text": "", "learning": False})
        except Exception:
            traceback.print_exc()
            self.emit({"type": "status", "text": "Some score analyses could not be completed; available recommendations are kept",
                       "learning": False})
        finally:
            self._learning = None
            db.close()

    def _first_learning(self):
        """Nothing known about the player yet: analyse with more workers (it is all they wait for)."""
        return not self.rec or len(self.rec.rows) < 30

    def _daily_check(self):
        last = recdata.kv_get(self.db, "snapshot_check", {}).get("t", 0)
        if time.time() - last > 86400:
            try:
                info = recdata.check_index(self.db)
                if info["newer"]:
                    self.emit({"type": "status", "text": f"newer public snapshot {info['newest'][:10]} — "
                                                         "recdata.py refresh --download (personal analysis; see licence)"})
            except OSError:
                pass

    def _session(self):
        stamp = (self.db.execute("SELECT MAX(id) FROM events").fetchone()[0], int(time.time()/2))
        if stamp != self._session_cache[0]:
            self._session_cache = (stamp, Session(load_events(self.db)))
        session = self._session_cache[1]
        session.acc_target = self.acc_targets.get(self.mode, .94)    # NPS/Skills never build at once
        return session.bind_warmup(getattr(getattr(self.rec,'model',None),'warmup',None))

    def _publish(self, mode=None):
        if not self.rec or self._stats_visible:
            return
        if len(self.rec.rows) < 30 and (self._learning or self._first_import):
            return          # the player's level is still unknown: an empty list says why, not beginner maps
        for mode in (mode or self.mode,):
            phase, shown, note = self._pool(mode)
            shown = self._prepared_playlist(mode, shown)
            st = {"nps": self.nps_state, "skills": self.skills_state}.get(mode) or {}
            # Charts that cannot be analysed (converts, broken files) are done too, or the bar waits forever.
            done = st.get("analyzed", 0) + st.get("failed", 0)
            progress = (done, st["total"]) if st.get("total") and done < st["total"] else None
            self.emit({"type": "pool", "mode": mode, "revision": self.revision, "selection_id": self._selection_id,
                       "phase": phase, "shown": shown, "note": note, "summary": summary(self.rec),
                       "progress": progress})

    def _prepared_playlist(self, mode, fresh):
        """Keep the displayed order; update predictions, then append new maps."""
        import nps
        session = self._session()
        ready = {}
        if mode in LOCAL_MODES:
            cached = self._local_pool_cache.get(mode)
            ready = {event_key(c): c for c in (cached[2] if cached else fresh)}
        shown = []
        for c in self.playlists[mode]:
            if c['keys'] not in self.keys[mode] or (mode == 'skills' and
                    tuple(c.get('practice_skills', ())) != self.practice_skills):
                continue
            sha = c.get('sha') or c.get('sha256')
            if c.get('installed') and sha not in self.rec.local_shas:
                continue
            f = c['f']
            expected = self.rec.predictor.predict(f, c.get('md5') or sha, c.get('bid'), c['rate'], session,
                                                  accuracy_only=mode != 'pp')
            alternative = ready.get(event_key(c))
            if alternative and alternative['rate'] != c['rate']:
                target = nps.central_target(session, sha)
                if abs(alternative['acc_mid']-target) < abs(expected['acc_mid']-target):
                    c, f = alternative, alternative['f']
                    expected = self.rec.predictor.predict(f, c.get('md5') or sha, c.get('bid'), c['rate'], session,
                                                          accuracy_only=True)
            if mode in LOCAL_MODES and not nps.eligible(expected, session, f):
                continue
            updated = dict(c, **expected)
            if mode == 'pp' and c['purpose'] != 'warmup':
                updated.update(pp_estimates(expected, f, c.get('bid'), self.rec.ledger))
            shown.append(updated)
        seen = {event_key(c) for c in shown} | set(self._playlist_consumed)
        for c in fresh:
            if len(shown) >= 20:
                break
            key = event_key(c)
            if key not in seen:
                shown.append(c); seen.add(key)
        if not shown and fresh:
            # A small credible library can cycle after its prepared queue ends.
            last = self._playlist_consumed[-1] if self._playlist_consumed else None
            shown = [c for c in fresh if event_key(c) != last][:20] or fresh[:1]
        self.playlists[mode] = shown
        return shown

    def t_publish(self):
        self._publish()

    def t_web_updated(self):
        """A crawl published more website charts: rebuild NPS/Skills so they can be suggested now."""
        self._request_nps()

    def _pool(self, mode):
        session = self._session()
        if not self.keys[mode]:
            return session.phase(), [], "Select at least one keymode"
        if mode == "pp":
            phase, shown, _pick, note = self.rec.pool(session, keys=self.keys[mode], choose=False)
            return phase, shown, note
        import nps
        context = (self.generation, session.model_key())
        if context != self._nps_prediction_context:
            self._nps_prediction_context, self._nps_predictions = context, {}
        st = self.skills_state if mode == "skills" else self.nps_state
        prepared_key = (context, self.keys[mode], self.practice_skills if mode == "skills" else (),
                        self.acc_targets.get(mode))
        cached = self._local_pool_cache.get(mode)
        if cached is None or cached[0] != prepared_key or cached[1] is not st:
            candidates = self._prepare_local(mode, st, session)
            self._local_pool_cache[mode] = (prepared_key, st, candidates)
        else:
            candidates = cached[2]
        # Explicit skill preferences must not be erased by unrelated NPS taste.
        taste = None
        if mode == 'nps':
            event_id = self.db.execute("SELECT MAX(id) FROM events").fetchone()[0]
            if event_id != self._taste_cache[0]:
                self._taste_cache = (event_id, nps.taste(self.db))
            taste = self._taste_cache[1]
        weighted = self._local_weights.get(mode)
        web = nps.web_factor(self.web_bias[mode])
        if weighted is None or weighted[0] is not session or weighted[1] is not candidates or weighted[2] is not taste \
                or weighted[3] != web:
            choices = nps.weighted_candidates(candidates, session, self.keys[mode], taste, mode=mode, web=web)
            self._local_weights[mode] = (session, candidates, taste, web, choices)
        else:
            choices = weighted[4]
        consumed = set(self._playlist_consumed)
        available = [c for c in choices if event_key(c) not in consumed] or choices
        if mode == 'nps':
            import pp_playlist
            seed = max((e.get('id', e['t']) for e in session.all
                        if e['kind'] in ('offer', 'finish', 'abort', 'reset')), default=0)
            shown = pp_playlist.preview(available, 20, seed, focus=self.nps_focus)
        else:
            shown = available[:20]
        note = f"target ~{100 * self.acc_targets.get(mode, .94):.0f}% · 0.70–1.50×"
        if not st["total"]:
            note += " · no maps of these keymodes found yet"
        elif st["analyzed"] + st["failed"] < st["total"]:
            note += (f" · {st['analyzed']:,} of {st['total']:,} maps analysed so far; "
                     "the rest are analysed a few at a time while you use this playlist")
        else:
            note += f" · all {st['total']:,} maps analysed"
        missing = sum(not c.get("installed", True) for c in shown)
        if missing:
            note += f" · {missing} of {len(shown)} not installed"
        if mode == "skills":
            note = skill_practice.label(self.practice_skills) + " · " + note
        if not shown:
            note += (" · looking for maps that fit your target…" if st["analyzed"] + st["failed"] < st["total"]
                     else " · no map fits your accuracy target at 0.70–1.50× (try another target or keymode)")
        if st["failed"]:
            note += f" · {st['failed']} unavailable analyses"
        if st["error"]:
            note += " · " + st["error"]
        if st.get("restored"):
            note += " · showing the last list while it updates"
        return session.phase(), shown, note

    def _prepare_local(self, mode, st, session):
        """Only form/features/filter changes revisit whole-map predictions and
        skill membership. A Next click just updates cheap exposure weights."""
        import nps
        candidates = []
        recent = [a for a in session.attempts[-6:] if a.get("profile") and a.get("meaningful", True)]
        recent_profiles = [np.array(a['profile']) for a in recent]
        profile_key = session.model_key()
        key=(self.generation,profile_key,self.keys[mode],self.practice_skills if mode=='skills' else ())
        previous=self._local_prepared.get(mode)
        memo=previous[1] if previous and previous[0]==key else {}
        self._local_prepared[mode]=(key,memo)
        for old in st["candidates"]:
            f, sha = old["f"], old["sha"]
            item_key=(sha,old['rate'])
            prepared=memo.get(item_key)
            if prepared and prepared[0] is f:
                if prepared[1] is not None:candidates.append(prepared[1])
                continue
            memo[item_key]=(f,None)
            if (old.get("installed", True) and sha not in self.rec.local_shas) or old["keys"] not in self.keys[mode] or not nps.MIN_RATE <= old["rate"] <= nps.MAX_RATE:
                continue
            if mode == "skills" and not skill_practice.match(f, self.practice_skills)[0]:
                continue
            cache_key = (sha, old["rate"])
            e = self._nps_predictions.get(cache_key)
            if e is None:
                e = self.rec.predictor.predict(f, old["md5"] or sha, self.rec.feats.md5_bid.get(old["md5"]), old["rate"], session, accuracy_only=True)
                self._nps_predictions[cache_key] = e
            target = nps.central_target(session, sha)
            low, high = nps.band(session)
            if not nps.eligible(e, session, f):
                continue
            value = nps.practice_value(f,e,old['rate'],target,session)
            if mode == "skills":
                value = skill_practice.value(f, e, old["rate"], target, self.practice_skills)
            v = np.array(old["profile"])
            similar = max((float(v @ a) for a in recent_profiles if len(a) == len(v)), default=0.)
            candidates.append(dict(old, **e, base_value=value, mode=mode, purpose=mode,
                                   practice_skills=self.practice_skills if mode == "skills" else (),
                                   practice_description=skill_practice.describe_match(f,self.practice_skills) if mode == "skills" else "",
                                   _profile_penalty=(profile_key, .18*similar)))
            memo[item_key]=(f,candidates[-1])
        return candidates

    def _restore_local(self, mode):
        if not self.rec or mode not in LOCAL_MODES or mode in self._restored_modes:
            return
        self._restored_modes.add(mode)
        saved = recdata.kv_get(self.db, "playlist_cache:"+mode, {})
        if saved.get("version") != 1:
            return
        # Saved choices are only chart/rate references. Reuse them after a
        # calculator update, but load features exclusively under the CURRENT
        # calculator key below and recompute predictions for this session.
        # Rejecting the references needlessly forced a full library pass.
        import nps
        by_sha = {i["sha256"]: i for i in self.rec.installed.values() if i["keys"] in self.keys[mode]}
        catalog, analyses = {}, collections.defaultdict(dict)
        for sha, rate in saved.get("choices", ()):
            if sha not in by_sha:
                continue
            f = self.rec.feats.lookup(self.rec.feats.key(sha, rate))
            if f:
                catalog[sha] = by_sha[sha]
                analyses[sha][rate] = f
        choices, _ = nps.candidates_from(catalog, analyses, self.rec.predictor, self._session(),
                                       mode=mode, selected_skills=self.practice_skills)
        state = dict(candidates=choices, total=len(by_sha), analyzed=min(len(by_sha), saved.get("analyzed", 0)),
                     failed=0, error="", restored=True)
        setattr(self, mode+"_state", state)

    def _request_nps(self):
        if self.builder and self.rec:
            self._local_request_id += 1
            session = Session(load_events(self.db))
            session.acc_target = self.acc_targets.get(self.mode, .94)
            self.builder.configure(self.rec.predictor, session, self.keys["nps"],
                                   self._local_request_id, self.generation, self.keys["skills"], self.practice_skills,
                                   "stats" if self._stats_visible else self.mode)

    def t_stats(self, visible=True):
        self._stats_visible = bool(visible)
        self._request_nps()
        if not visible:
            self._publish()
            return
        if not self.rec:
            return
        import player_stats
        if self._stats_cache[0] != self.generation:
            self._stats_cache = (self.generation, player_stats.build(self.rec))
        session=self._session()
        display=getattr(self.rec,'accuracy_model',None)
        if display is not None:session=session.for_accuracy(display.warmup)
        self.emit({"type": "stats", "data": player_stats.current(self._stats_cache[1], session)})

    def t_nps_update(self, kind, data):
        if kind == "error":
            for mode in LOCAL_MODES:
                self.emit({"type": "status", "mode": mode, "text": "Local map analysis: " + data})
            return
        if data["generation"] != self.generation or data["revision"] != self._local_request_id:
            return
        if any(tuple(data["keys_by_mode"][m]) != self.keys[m] for m in LOCAL_MODES) \
                or tuple(data["selected_skills"]) != self.practice_skills:
            return
        # Exact proposed features are immediately available to selected-map and
        # start-event predictions, without recalculating them on a Next click.
        for mode, state in data["states"].items():
            # The restart cache holds installed charts only; the first real build replaces
            # that placeholder list instead of keeping it, or website charts never show up.
            if getattr(self, mode+"_state", {}).get("restored"):
                self.playlists[mode] = []
            setattr(self, mode+"_state", state)
            for c in state["candidates"]:
                self.rec.feats.cache[self.rec.feats.key(c["sha"], c["rate"])] = c["f"]
            # Small restart cache of references, never frozen predictions or a
            # duplicate copy of the feature database. Preserve broad coverage.
            import hashlib
            rows = state["candidates"]
            kept = sorted(rows, key=lambda c: -c["base_value"])[:512]
            kept += sorted(rows, key=lambda c: hashlib.sha256(c["sha"].encode()).digest())[:2048]
            refs = sorted({(c["sha"], c["rate"]) for c in kept})
            recdata.kv_set(self.db, "playlist_cache:"+mode, {"version": 1, "calc": self.rec.feats.calc,
                            "choices": refs, "analyzed": state["analyzed"]})
            if mode == self.mode:
                self._publish(mode)

    def t_configure(self, mode, keys_by_mode, action="auto", revision=0, practice_skills=None, nps_focus=None,
                    acc_targets=None, web_bias=None):
        # Fast clicks need one search for the final selection, not a full
        # catalogue pass for every intermediate checkbox state. Other queued
        # events (including scores and gameplay) keep their original ordering.
        if mode not in MODES or revision < max(self.revision, self._queued_config_revision):
            return
        old = (self.keys["nps"], self.keys["skills"], self.practice_skills)
        old_mode = self.mode
        self.mode, self.action, self.revision = mode, action if action in navigation.ACTIONS else "auto", revision
        self.keys = {m: recdata.normalize_keys(keys_by_mode.get(m, None if m == "pp" else [7])) for m in MODES}
        self.practice_skills = skill_practice.normalize(practice_skills)
        if nps_focus is not None:
            import nps
            focus = nps.normalize_focus(nps_focus)
            if focus != self.nps_focus:
                self.nps_focus = focus
                self.playlists['nps'] = []
        for m, value in (web_bias or {}).items():
            if m in self.web_bias and value != self.web_bias[m]:
                self.web_bias[m] = value
                self.playlists[m] = []
        retarget = False
        for m, value in (acc_targets or {}).items():
            import nps
            value = nps.normalize_target(value)
            if m in self.acc_targets and value != self.acc_targets[m]:
                self.acc_targets[m], retarget = value, True
                self.playlists[m] = []
                self._local_pool_cache.pop(m, None)
        self._restore_local(mode)
        if retarget or old != (self.keys["nps"], self.keys["skills"], self.practice_skills) or old_mode != mode:
            self._request_nps()
        self._publish()

    def t_reset_taste(self):
        recdata.log_event(self.db, "taste_reset")
        self.playlists = {m: [] for m in MODES}
        self._publish()

    def stop(self):
        self._halt.set()
        if self.builder:
            self.builder.stop()

    # -- tasks ------------------------------------------------------------
    def t_start(self, play):
        if not recdata.tracking_allowed(self.db, play["t"]):
            return
        exp = {}
        if self.rec and play.get("path") and recdata.predictable_context(play.get("mods"), play.get("rate")):
            try:
                exp = self.rec.expectation(play["sha"], play["path"], play["rate"], compute=False)
            except OSError:
                pass
        inst = self.db.execute("SELECT artist,title,version,md5,keys,length,beatmap_id FROM installed WHERE sha256=?", (play["sha"],)).fetchone()
        title = f"{inst['artist']} - {inst['title']} [{inst['version']}]" if inst else play["sha"]
        md = {}
        if not inst and play.get("path"):
            import lazer_index
            md = lazer_index._read_metadata(play["path"]) or {}
            if md.get("Title"):
                title = f"{md.get('Artist', '')} - {md['Title']} [{md.get('Version', '')}]"
        c = self.target
        bid = inst['beatmap_id'] if inst else md.get('BeatmapID')
        matched = navigation.same_map(c, sha=play['sha'], md5=exp.get('md5'), bid=bid)
        if not matched:
            # Restore the offer linkage after an application restart. Only the
            # latest preceding offer can match; an unrelated manual pick cannot.
            offer=self.db.execute("SELECT id,info FROM events WHERE kind='offer' AND t<=? AND t>=? "
                                  "ORDER BY t DESC,id DESC LIMIT 1",(play['t'],play['t']-1800)).fetchone()
            offered=json.loads(offer['info']) if offer else {}
            md5=exp.get('md5') or (inst['md5'] if inst else None)
            if navigation.same_map(offered, sha=play['sha'], md5=md5, bid=bid):
                c=dict(offered,offer_id=offer['id'],acc_mid=offered.get('accuracy'),
                       opening_acc=offered.get('opening_accuracy'))
                matched=True
        purpose = c["purpose"] if matched else "manual"
        f = exp.get("f")
        pattern, family = None, None
        if f and f.get("nps", {}).get("version"):
            import nps
            pattern, family = nps.profile(f).tolist(), f["nps"].get("family")
        exp.setdefault("md5", inst["md5"] if inst else None)
        exp.setdefault("keys", inst["keys"] if inst else None)
        exp.setdefault("length", (inst["length"] or 0) / 1000 / play["rate"] if inst and play.get("rate") else 0)
        key = event_key({"md5": exp.get("md5"), "sha": play["sha"]})
        for mode in MODES:
            self.playlists[mode] = [p for p in self.playlists[mode] if event_key(p) != key]
        recdata.log_event(self.db, "start", key, t=play["t"], sha=play["sha"], rate=play["rate"], purpose=purpose,
                          mode=c.get("mode") if matched else "manual", offer_id=c.get("offer_id") if matched else None,
                          offered_rate=c.get("rate") if matched else None,
                          offered_accuracy=c.get("acc_mid") if matched else None,
                          offered_opening_accuracy=c.get("opening_acc") if matched else None,
                          practice_skills=c.get("practice_skills") if matched else None,
                          choice=c.get("selection") if matched else None,
                          profile=pattern or (c.get("profile") if matched else None),
                          family=family or (c.get("family") if matched else None),
                          mods=[m.get("acronym") for m in play.get("mods") or []], mods_list=play.get("mods"),
                          title=title, predicted_at=time.time(), predictor=self.predictor_id,
                          calculator=self.rec.pub.get("calc") if self.rec else None,
                          snapshot=self.rec.pub.get("snapshot") if self.rec else None, features=exp.get("f"),
                          difficulty=score_units.displayed_feature(f) if f else None,
                          expected=exp.get("acc_mid"), expected_lo=exp.get("acc_lo"), expected_hi=exp.get("acc_hi"),
                          shown_mu=exp.get("mu"),
                          **{k: v for k, v in exp.items() if k in ("mu", "base_mu", "cold_penalty", "cold_scale", "activation_minutes", "opening_acc", "rate_slope", "sd", "sdm", "keys", "skill", "ranked", "length", "md5",
                              "display_mu","display_base_mu","display_cold_penalty","display_sdm","display_sda","display_sd")})
        self._current = key

    def t_abort(self, play):
        if not recdata.tracking_allowed(self.db, play.get("end", time.time()), play["t"]):
            return
        start = self.db.execute("SELECT id,beatmap FROM events WHERE kind='start' AND t=? ORDER BY id DESC LIMIT 1",
                                (play["t"],)).fetchone()
        recdata.log_event(self.db, "abort", start["beatmap"] if start else getattr(self, "_current", None), t=play.get("end"),
                          start_id=start["id"] if start else None, progress=round(play.get("progress", 0), 3),
                          played_seconds=play.get("played_seconds", 0.), partial_stats=play.get("partial_stats"))
        self._request_nps()
        self._publish()

    def t_void(self, play):
        recdata.log_event(self.db, "void", getattr(self, "_current", None))

    def t_score(self, score, path, fresh, play):
        import hashlib
        try:
            with open(path, "rb") as fh:
                md5 = hashlib.md5(fh.read()).hexdigest()
        except OSError:
            md5 = None                               # still stored; the next Realm import fills the ids
        inst = installed_by_md5(self.db, md5).get(md5, {}) if md5 else {}
        bid = inst.get("beatmap_id")
        before = (self.rec.ledger.total(), self.rec.ledger.best.get(bid, (0.0,))[0],
                  Session(load_events(self.db)).phase()) if self.rec else None
        key, _new = recdata.add_live_score(self.db, dict(score, md5=md5, beatmap_id=inst.get("beatmap_id"),
                                                         status=inst.get("status")), path, fresh, play)
        if key is None:
            self.emit({"type": "status", "text": "play excluded: tracking was paused"})
            return
        start = self.db.execute("SELECT * FROM events WHERE kind='start' AND t=? ORDER BY id DESC LIMIT 1",
                                (play["t"],)).fetchone() if play else None
        info = json.loads(start["info"]) if start else {}
        if info.get("sha") != score["sha256"] or info.get("rate") != score.get("rate"):
            start, info = None, {}
        if info.get("predicted_at", 0) > (_ts(score.get("played")) or time.time()):
            info = {}                       # a delayed startup must not masquerade as a pre-play prediction
        title = f"{inst['artist']} - {inst['title']} [{inst['version']}]" if inst else info.get("title") or score["sha256"]
        rate = f"{score['rate']:.2f}×" if score.get("rate") else "unknown rate"
        label = f"{title} · {rate} · {time.strftime('%H:%M', time.localtime(_ts(score.get('played')) or time.time()))}"
        if fresh:
            self._replay_due, self._replay_retries = time.monotonic() + 3, 0
            with self.db:
                saved = self.db.execute("SELECT acc,stats FROM scores WHERE key=?", (key,)).fetchone()
                acc=saved[0]
                display_accuracy=recdata.acc_lazer(json.loads(saved[1]))
            duplicate = self.db.execute("SELECT 1 FROM events WHERE kind='finish' AND json_extract(info,'$.key')=?", (key,)).fetchone()
            if duplicate:
                return
            recdata.log_event(self.db, "finish", start["beatmap"] if start else None, key=key,
                              start_id=start["id"] if start else None,
                              prediction_valid=info.get("mu") is not None,
                              y=math.log(max(1e-3, 1 - min(acc, 0.999))),
                              y_lazer=math.log(max(1e-3,1-min(display_accuracy,.999))))
            # Show the saved play even if subsequent feature calculation/refitting fails.
            self.emit({"type": "result", "title": label, "text": self._verdict(key, bid, None, info),
                       "targets": [m for m, c in self.targets.items()
                                   if c and navigation.same_map(c, sha=score["sha256"], md5=md5, bid=bid)]})
        if self.rec and (_new or fresh):
            if recdata.predictable_context(score.get("mods_list"), score.get("rate")):
                self._request_feature(score.get("sha256"), path, score.get("rate"), refit=True)
            self.rec.observe(key)
            if fresh and before is not None and start is not None and bid:
                recdata.log_event(self.db, "pp_result", start["beatmap"], start_id=start["id"],
                                  before=before[1], after=self.rec.ledger.best.get(bid, (0.,))[0])
            if not fresh and _new:
                # A newly viewed old score is historical evidence, unlike a
                # live current-session result already handled by form.
                self.t_refit()
            self._request_nps()
            self._publish()
            self._card()
            if self._stats_visible:
                self.t_stats(True)
            if fresh:
                self.emit({"type": "result", "title": label, "text": self._verdict(key, bid, before, info)})

    def _verdict(self, key, bid, before, info):
        """One line after a finished play: the result against what was shown, and what it did."""
        st = json.loads(self.db.execute("SELECT stats FROM scores WHERE key=?", (key,)).fetchone()[0])
        acc, _n = recdata.acc320(st)
        out = [f"{100 * recdata.acc_lazer(st):.2f}%"]
        if info.get("mu") is not None and info.get("sd"):
            shown_mu = info.get("shown_mu", info["mu"] + PESSIMISM * (info.get("sdm") or 0.0))
            z = (shown_mu - math.log(max(1e-3, 1 - min(acc, 0.999)))) / info["sd"]
            if info.get('display_mu') is not None and info.get('display_sd'):
                z=(info['display_mu']-math.log(max(1e-3,1-min(recdata.acc_lazer(st),.999))))/info['display_sd']
            expected = info.get("expected")
            out.append((f"expected ~{100 * expected:.2f}% · " if expected is not None else "")
                       + ("way above expected" if z > 1.5 else "better than expected" if z > 0.5 else
                          "way below expected" if z < -1.5 else "below expected" if z < -0.5 else "about as expected"))
        if before is None:
            return " · ".join(out)
        t0, b0, ph0 = before
        L = self.rec.ledger
        b1 = L.best.get(bid, (0.0,))[0] if bid else 0.0
        if b1 > b0 + 0.05:
            out.append(f"new best {b1:.0f}pp (was {b0:.0f}) · account {L.total() - t0:+.1f}pp" if b0
                       else f"new map {b1:.0f}pp · account {L.total() - t0:+.1f}pp")
        ph1 = Session(load_events(self.db)).phase()
        if ph1 != ph0:
            out.append({"build": "warmed up — building", "push": "in form — pushing", "recover": "easing off",
                        "warmup": "warming up"}[ph1])
        return " · ".join(out)

    def _card(self):
        path, rate, mods = getattr(self, "_sel", (None, None, None))
        if self.rec and path:
            try:
                c = None
                if rate is None or mods is None:
                    note = "Waiting for osu! to report the playback rate and mods"
                elif not recdata.predictable_context(mods, rate):
                    note = "Prediction unavailable for these mods"
                else:
                    import hashlib
                    with open(path, 'rb') as fh:
                        sha = hashlib.sha256(fh.read()).hexdigest()
                    c = self.rec.card(path, rate, compute=False)
                    failed = self._feature_errors.get((sha, round(rate,3), self.rec.feats.calc))
                    note = f"Prediction unavailable at {rate:.2f}×: {failed[1]}" if failed else f"Calculating prediction at {rate:.2f}×…"
                self.emit({"type": "card", "c": c, "note": None if c else note, "path":path, "rate":rate})
            except OSError:
                self.emit({"type": "card", "c": None, "note": "Selected chart file is unavailable", "path":path, "rate":rate})
        else:
            self.emit({"type":"card", "c":None, "note":(getattr(self, "_loading", None) or "Loading prediction model…") if path else None,
                       "path":path, "rate":rate})

    def t_next(self, skip=False, mode=None, revision=None, candidate=None, result=None, selected_at=None, selection_id=0):
        mode = mode or self.mode
        if mode not in self.keys or (result is None and
                ((revision is not None and revision != self.revision) or not self.keys[mode])):
            return
        if not self.rec:
            return
        if skip and self.targets[mode]:
            recdata.log_event(self.db, "skip", event_key(self.targets[mode]), mode=mode)
        pick = candidate or next(iter(self.playlists[mode]), None)
        if pick:
            recdata.log_event(self.db, "next", event_key(pick), t=selected_at, purpose=pick["purpose"], var=pick["var"], mode=mode)
            self._open(pick, mode, selection="next", result=result, selected_at=selected_at, selection_id=selection_id)
        else:
            self.emit({"type": "target", "mode": mode, "revision": self.revision, "selection_id": self._selection_id,
                       "c": None, "msg": "Preparing the next playlist…"})
        # Let pending selections log before refreshing the remainder.
        self.put("publish")

    def t_open(self, c, mode=None, revision=None, result=None, selected_at=None, selection_id=0):
        mode = mode or c.get("mode") or self.mode
        if mode not in self.keys or (result is None and
                ((revision is not None and revision != self.revision) or c["keys"] not in self.keys[mode])):
            return
        if result is None and mode == "skills" and (tuple(c.get("practice_skills", ())) != self.practice_skills
                                  or not skill_practice.match(c.get("f", {}), self.practice_skills)[0]):
            return
        self._open(c, mode, selection="list", result=result, selected_at=selected_at, selection_id=selection_id)
        self._publish()

    def _open(self, c, mode=None, selection="list", *, result=None, selected_at=None, selection_id=0):
        mode = mode or c.get("mode") or self.mode
        c = dict(c, mode=mode, selection=selection)
        performed = result is not None
        result = result if performed else navigation.perform(c, "copy" if self.busy() else self.action)
        self._selection_id = max(self._selection_id, selection_id)
        key = event_key(c)
        self._playlist_consumed.append(key)
        for m in MODES:
            self.playlists[m] = [p for p in self.playlists[m] if event_key(p) != key]
        if result.get('unavailable'):
            self.emit({"type": "status", "text": result['msg']})
            return
        c["offer_id"] = recdata.log_event(self.db, "offer", key, t=selected_at, mode=mode, keys=c["keys"],
                                         group=c.get("group"), bid=c.get('bid'), rate=c["rate"], selection=selection,
                                         length=c.get("length"), accuracy=c.get("acc_mid"), title=c.get("title"),
                                         opening_accuracy=c.get("opening_acc"), activation_minutes=c.get("activation_minutes"),
                                         purpose=c.get("purpose"), md5=c.get("md5"), sha=c.get("sha"),
                                         p_up=c.get("p_up"), gain=c.get("gain"), best_pp=c.get("best"), pp_mid=c.get("pp_mid"),
                                         practice_skills=c.get("practice_skills") if mode == "skills" else None,
                                         matched_skills=c.get("matched_skills") if mode == "skills" else None,
                                         predictor=self.predictor_id, calculator=self.rec.feats.calc)
        self.target = self.targets[mode] = c
        if result["sent"]:
            recdata.log_event(self.db, "open", key, t=selected_at, var=c["var"], mode=mode)
        self.emit({"type": "target", "mode": mode, "revision": self.revision, "selection_id": self._selection_id, "c": c,
                   "msg": result["msg"] + f" · set {c['var']} manually", "copy": None if performed else result["copy"]})

    def t_selected(self, path, rate, mods=None):
        import hashlib
        previous_path = getattr(self, '_sel', (None,))[0]
        self._sel = (path, rate, mods)
        if path:
            with open(path, "rb") as fh:
                data = fh.read()
            sha, md5 = hashlib.sha256(data).hexdigest(), hashlib.md5(data).hexdigest()
            if recdata.predictable_context(mods, rate):
                self._request_feature(sha, path, rate)
            if self.rec and path != previous_path and sha not in self.rec.local_shas:
                self._begin_import()
        self._card()
        if not self.target or not path:
            return
        import lazer_index
        md = lazer_index._read_metadata(path) or {}
        if navigation.same_map(self.target, sha=sha, md5=md5, bid=md.get('BeatmapID')):
            updated = md5 != self.target.get('md5')
            recdata.log_event(self.db, "selected", event_key({'md5':md5}),
                              offered_revision=event_key(self.target), updated=updated)
            self.emit({"type": "target", "mode": self.target.get("mode", "pp"), "revision": self.revision, "selection_id": self._selection_id,
                       "c": self.target, "msg": f"✓ {'updated ' if updated else ''}chart selected — intended {self.target['var']}; "
                       + (f"observed {rate:.2f}×" if rate else "actual rate unknown")})

    def _request_feature(self, sha, path, rate, refit=False):
        if not self.rec or not sha or not path or rate is None:
            return
        row = {"sha256": sha, "md5": None, "beatmap_id": None, "rate": rate}
        if self.rec.feats(row)[0]:
            return
        with self._feature_lock:
            failed = self._feature_errors.get((sha, round(rate,3), self.rec.feats.calc))
            if failed and failed[0] > time.monotonic():
                return
            self._feature_errors.pop((sha,round(rate,3),self.rec.feats.calc),None)
            self._feature_pending = (sha, path, rate, refit, self.rec.feats.calc)
            if self._feature_thread and self._feature_thread.is_alive():
                return
            self._feature_thread = threading.Thread(target=self._feature_loop, daemon=True, name="selected-feature")
            self._feature_thread.start()

    def _feature_loop(self):
        while not self._halt.is_set():
            with self._feature_lock:
                task, self._feature_pending = self._feature_pending, None
                if task is None:
                    self._feature_thread = None
                    return
            sha, path, rate, refit, calc = task
            while self.busy() and not self._halt.is_set():
                self._halt.wait(1)
            if self._halt.is_set():
                return
            try:
                import selected_analysis
                fs = selected_analysis.calculate(path, round(rate, 3), calc)
                with self._feature_lock:
                    self._feature_errors.pop((sha,round(rate,3),calc),None)
                if not self._halt.is_set():
                    self.put("feature_ready", sha, fs, refit, calc)
            except Exception as exc:
                traceback.print_exc()
                reason = "App updated — restart ManiaScope to finish the update" if 'Calculator changed' in str(exc) else str(exc)[:240]
                with self._feature_lock:
                    if len(self._feature_errors)>64:
                        self._feature_errors.clear()
                    self._feature_errors[sha,round(rate,3),calc]=(time.monotonic()+30.,reason)
                if not self._halt.is_set():
                    self.emit({"type": "status", "text": "Selected chart prediction unavailable: "+str(exc)[:160]})
                    self.put("feature_ready", sha, None, False, calc)

    def t_feature_ready(self, sha, fs, refit, calc):
        if self.rec and calc == self.rec.feats.calc:
            if fs:
                self.rec.feats._store(sha, fs)
            if fs and refit:
                self._request_nps()
                self._publish()
            self._card()

    def t_reset(self):
        recdata.log_event(self.db, "reset")
        self.playlists = {m: [] for m in MODES}
        self._request_nps()
        self._publish()

    def t_skip_warmup(self):
        recdata.log_event(self.db, "skip_warmup")
        self.playlists = {m: [c for c in self.playlists[m] if c['purpose'] != 'warmup'] for m in MODES}
        self._request_nps()
        self._publish()

    def t_lock(self, keys):
        self.keys[self.mode] = recdata.normalize_keys(None if keys is None else [keys])
        self._request_nps()
        self._publish()

    def t_import(self):
        self._begin_import()

    def _begin_import(self):
        """Restored recommendations are usable before Realm copy/index I/O.
        One import is coalesced; the live event/model owner remains this worker."""
        if self._import_thread is not None and self._import_thread.is_alive():
            return
        path=self.db.execute('PRAGMA database_list').fetchone()[2]
        self.emit({"type":"status","text":"Reading your osu!lazer scores and installed maps (first time: about a minute)…"
                   if self._first_import else "Using saved history · checking lazer for new plays in the background…"})
        def run():
            db=None
            try:
                while self.busy() and not self._halt.is_set():
                    self._halt.wait(1)
                if self._halt.is_set():return
                import hashlib
                db=recdata.connect(path)
                def signature():
                    digest=hashlib.sha256()
                    for row in db.execute('SELECT sha256,md5,status,keys FROM installed ORDER BY sha256'):
                        digest.update(repr(tuple(row)).encode())
                    return digest.digest()
                before=signature()
                result=recdata.import_realm(db)
                recovered=recover_results(db)
                changed=signature()!=before
                # Website top 100 for a named profile: bests from stable/other PCs (user 2026-10-01).
                name=recdata.kv_get(db,'website_username')
                if name and time.time()-(recdata.kv_get(db,'website_user',{}) or {}).get('t',0)>6*3600:
                    try:
                        recdata.import_website_best(db,name)
                        changed=True
                    except (OSError,ValueError) as exc:
                        self.emit({"type":"status","text":f"osu! profile scores unavailable: {exc}"[:160]})
                self.put('imported',result,recovered,changed,None)
            except Exception as exc:
                self.put('imported',None,0,False,str(exc).splitlines()[0][:160])
            finally:
                if db is not None:db.close()
        self._import_thread=threading.Thread(target=run,daemon=True,name='score-import')
        self._import_thread.start()

    def t_website_user(self, name):
        recdata.kv_set(self.db, "website_username", (name or "").strip() or None)
        recdata.kv_set(self.db, "website_user", {})
        self._begin_import()

    def t_imported(self, result, recovered, changed, error=None):
        first, self._first_import = self._first_import, False
        if error and first:
            self._stage(f"Could not read your osu!lazer scores: {error}", blocking=True)
            return
        if error:
            self.emit({"type":"status","text":"Saved recommendations retained; lazer import pending: "+error})
            return
        self.emit({"type": "status", "text": (f"Read {result['new']:,} new plays and {result['installed']:,} installed maps from osu!lazer"
                                               if result['new'] else f"Up to date with osu!lazer ({result['installed']:,} maps)")
                   + (f" · {recovered} predictions recovered" if recovered else "")})
        if result["new"] or recovered:
            rows = play_history(self.db, limit=1)
            if rows:
                self.emit({"type": "result", "title": rows[0]["title"], "text": history_text(rows)})
        if result['new'] and self.rec:
            self._start_fill()          # before the refit: its publish must already know the level is pending
        if result['new'] or recovered or changed:
            self.t_refit()

    def t_tracking(self):
        self.t_refit()

    def t_history(self, outliers=False):
        self.emit({"type": "history", "text": history_text(play_history(self.db, outliers=outliers))})

    def t_refit(self):
        if self.busy():
            self._refit_due=True
            return
        self._refit_due=False
        if self.rec:
            self.rec.refit()
            self._last_refit = time.time()
            self.generation += 1
            self._request_nps()
            self._publish()
            if self._stats_visible:
                self.t_stats(True)


def summary(rec):
    L = rec.ledger
    st = rec.pub["user"]["stats"] or {}
    return (f"ledger {L.total():.1f}pp over {len(L.P)} maps (snapshot {st.get('rank_score')}pp on {str(st.get('last_update'))[:10]}; "
            f"{rec.ledger_missing} lazer scores without a local chart) · {rec.model.n_rows} evidence rows · "
            f"levels {', '.join(f'{k}K {v:+.2f}' for k, v in rec.model.level.items())} · retry {rec.model.retry:.3f}")


def retro(cut="2026-06-01"):
    """Held-out check: fit on everything before `cut` (public rows of other players too), predict the
    user's own later lazer attempts. Reports y error vs simple baselines, interval coverage, and whether
    the predicted chance of beating the prior best matches how often it happened."""
    import pickle
    db = recdata.connect()
    pub = recdata.load_public()
    with open(os.path.join(recdata.FROZEN, "scores.pkl"), "rb") as fh:
        scores, maps = pickle.load(fh)
    fr = {b: {r: v for r, v in f.items() if isinstance(r, float)} for b, f in pub["feats"].items()}
    ymd = int(cut.replace("-", ""))
    pub_cut = dict(pub, pop=recdata.fit_population(scores, maps, fr, pub["user"]["id"], date_cut=ymd))
    fs = FeatureSource(db, pub_cut)
    rows, _sk = evidence_rows(db, pub_cut, fs)
    before = [r for r in rows if r["t"] < months(cut)]
    after = sorted((r for r in rows if r["t"] >= months(cut) and r["src"] != "public"), key=lambda r: r["ts"] or 0)
    levels = play_levels(before, months(cut))
    for r in before + after:
        r["base"] = base_of(pub_cut["pop"], r["f"], r["b"], round(r["rate"], 3), levels.get(r["keys"]))[0]
    m = Personal(pub_cut, before, now=months(cut), levels=levels)
    hist = collections.defaultdict(list)
    for r in before:
        hist[r["chart"]].append(r["y"])
    res = collections.defaultdict(list)
    bins = collections.defaultdict(lambda: [0.0, 0, 0])
    prior_best = {}
    for r in before:
        if r["b"] and round(r["rate"], 2) in (0.75, 1.0, 1.5):
            prior_best[(r["b"], round(r["rate"], 2))] = max(prior_best.get((r["b"], round(r["rate"], 2)), -1), 1 - math.exp(r["y"]))
    for r in after:
        mu, sdm, sda = m.predict(r["f"], r["chart"], r["b"], round(r["rate"], 3))
        kind = "refarm" if r["chart"] in hist else "unseen"
        res[(kind, "model")].append(r["y"] - mu)
        res[(kind, "level+rating")].append(r["y"] - (m.level_of(r["keys"])[0] + r["base"]))
        if kind == "refarm":
            res[(kind, "own history mean")].append(r["y"] - float(np.mean(hist[r["chart"]])))
        z = (r["y"] - mu - PESSIMISM * sdm) / math.sqrt(sda ** 2 + sdm ** 2)
        res[(kind, "cover80")].append(abs(z) < BAND_Z)
        key = (r["b"], round(r["rate"], 2))
        if key in prior_best and r["f"].get("stars"):
            acc_best = prior_best[key]
            p = float(p_beat(mu + PESSIMISM * sdm, sda, acc_best))
            bb = bins[min(4, int(p * 5))]
            bb[0] += p
            bb[1] += 1
            bb[2] += (1 - math.exp(r["y"])) > acc_best
        if key[1] is not None and r["b"]:
            prior_best[key] = max(prior_best.get(key, -1), 1 - math.exp(r["y"]))
        hist[r["chart"]].append(r["y"])
    print(f"cut {cut}: fit on {len(before)} rows, {len(after)} held-out lazer attempts "
          f"({sum(r['chart'] in dict.fromkeys(x['chart'] for x in before) for r in after)} on charts seen before)")
    for (kind, what), v in sorted(res.items()):
        v = np.array(v, float)
        print(f"  {kind:<7} {what:<17} " + (f"coverage {100 * v.mean():.0f}% (target 80)" if what == "cover80"
                                            else f"rmse {math.sqrt((v ** 2).mean()):.3f}  bias {v.mean():+.3f}  n={len(v)}"))
    print("  P(beat prior best at this map+rate): predicted vs happened")
    for k in sorted(bins):
        s, n, hit = bins[k]
        print(f"    {20 * k:>3}-{20 * k + 20}%  predicted {100 * s / n:5.1f}%  happened {100 * hit / n:5.1f}%  n={n}")


def main():
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else "list"
    if cmd == "retro":
        return retro(*sys.argv[2:3])
    db = recdata.connect()
    if cmd == "history":
        rows = play_history(db, limit=None if "--json" in sys.argv else 100, outliers="--outliers" in sys.argv)
        print(json.dumps(rows, ensure_ascii=False, indent=2) if "--json" in sys.argv else history_text(rows))
        return
    rec = Recommender(db)
    if cmd == "list":
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 25
        print(summary(rec))
        phase, shown, pick, note = rec.pool(n_show=n)
        print(f"phase {phase}{' · ' + note if note else ''}")
        for c in shown:
            print(f"{c['keys']:>2}K {c['var']:<3} {c['gain']:6.2f}pp  {c.get('pp_lo', 0):4.0f}–{c.get('pp_hi', 0):4.0f}  "
                  f"acc~{100 * c['acc_mid']:.2f}  {c['length'] / 60:4.1f}m  {'' if c['installed'] else '(dl) '}{c['title'][:70]}  · {c['why']}")
        if pick:
            print("Next →", pick["title"], pick["var"])
    elif cmd == "retro":
        retro(*sys.argv[2:3])


if __name__ == "__main__":
    main()
