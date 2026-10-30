"""Installed-map density recommendations: exact-rate features, bounded search and variety.

The personal performance model is shared with PP. This module owns only the
installed catalogue, recomputable structure and rate work, and the NPS objective.
"""
import collections
import hashlib
import heapq
import json
import math
import os
import random
import threading
import time

import numpy as np

import recdata
import skill_practice

from chart_structure import STRUCTURE_VERSION, structure
MIN_RATE, MAX_RATE = 0.70, 1.50
# An exact rate whose last finished result (14 days) missed the target by more than this is not
# offered again; the next rate comes from that result (user 2026-10-13: "not get picked exactly the
# same rate even though it's not accurate").
MISS_TOL = 0.005
PROFILE = ("stream", "delay", "jumpstream", "handstream", "chordstream", "chordjack",
           "jack", "jackspeed", "minijack", "longjack", "jumptrill", "splittrill", "bracket",
           "ln", "release", "inverse", "hybrid", "shield", "patterntech", "rhythmtech", "sv")



def normalize_focus(value):
    try:
        value = float(value)
        return min(1., max(0., value)) if math.isfinite(value) else .5
    except (TypeError, ValueError):
        return .5


def web_factor(value):
    """'Maps you don't have' slider → their odds against your own maps (see mix). Their list share rises
    linearly: 0 at the left end, half in the middle, 80 % at the right (an exponent jumped 0 → 22 %, user 2026-10-26)."""
    value = normalize_focus(value)
    share = value if value <= .5 else .5 + .6 * (value - .5)
    return share / (1. - share)


DEFAULT_TARGET = .94


def normalize_target(value):
    """The tab's displayed-accuracy target (Settings slider): 88–98%, default 94%."""
    try:
        value = float(value)
        return min(.98, max(.88, value)) if math.isfinite(value) else DEFAULT_TARGET
    except (TypeError, ValueError):
        return DEFAULT_TARGET


# Offsets scale with the error left to 100 % (user 2026-10-29: "at 97% the diff is 3% so it should increase
# by 1%"): the 94 %-default values are shares of its 6 % error. Too close to 100 % nothing qualifies.
MAX_TARGET, MAX_HIGH = .985, .99
LEAD = 300          # charts re-predicted first after a play or refit (the shown 20 come from the top ~60)


def _err(session):
    """Band width follows the slider's error, so jack maps keep the same tolerance."""
    return 1 - central_target(session)


def band(session, f=None):
    """±1.5 % at 94 %; ±0.75 % at 97 %."""
    target, err = central_target(session, f), _err(session)
    return target - .25 * err, min(MAX_HIGH, target + .25 * err)


def reach(pred, keys, target):
    """{keys: (lo, hi)} ratings that can land near the target: website charts outside are never loaded.
    The level + population curve alone put the target at ov; every NPS candidate of 2026-10-01
    (4,513 website chart-rates) lay within 0.64–1.00·ov, so 0.55–1.10 loses none."""
    import recommend
    model = (getattr(pred, "display", None) or pred).model
    out = {}
    for k in keys:
        level, y = model.level_of(k)[0], math.log(1 - target)
        lo, hi = math.log(.5), math.log(60.)
        for _ in range(40):
            mid = (lo + hi) / 2
            if level + recommend.base_of(model.pop, {"keys": k, "overall": math.exp(mid), "od": 8.})[0] < y:
                lo = mid
            else:
                hi = mid
        out[k] = (round(.55 * math.exp(lo), 1), round(1.1 * math.exp(lo), 1))
    return out


def central_target(session, f=None):
    """The active tab's accuracy target; difficulty is fitted to current form around it.
    Jack-led maps aim a third closer to 100 % (user 2026-10-26: "for jack maps ... like 96% instead of 94%")."""
    target = getattr(session, "acc_target", DEFAULT_TARGET)
    if f and f.get("sk") and group(f) in ("chordjack", "jack"):
        target = 1 - (1 - target) * 2 / 3          # 94 → 96, 97 → 98
    return min(MAX_TARGET, target)


def opening_floor(session, f, expected=None):
    low, _high = band(session, f)
    err = _err(session)
    if session.warm_flag:
        return low-.1*err
    model = getattr(session, '_warmup_model', None)
    tau = (expected or {}).get('warmup_tau') or (model.parameters(f['keys'])['tau'] if model is not None else 6.)
    minutes = (expected or {}).get('activation_minutes')
    if minutes is None:
        minutes = session.activation_for(f)
    cold = math.exp(-minutes/tau)
    # The first pick should meet the practice goal at its opening readiness,
    # not merely average the cold opening with an imagined warmed-up ending.
    return (low-.1*err) + cold*(central_target(session, f)-.05*err-(low-.1*err))


def eligible(expected, session, f):
    low, high = band(session, f)
    slack = .1 * _err(session)
    if not low-slack <= expected['acc_mid'] <= min(MAX_HIGH, high+slack):
        return False
    return (session.warm_flag or 'opening_acc' not in expected
            or expected['opening_acc'] >= opening_floor(session, f, expected))


def rate_shift(expected, session, f, target):
    """Log-error gap for rate search, respecting BOTH readiness constraints.

    Searching only for a 94% average would repeatedly propose a rate rejected
    by the opening guard. Use the more conservative constraint to find an
    actually eligible exact analysis, without inventing an opening score.
    """
    gap = math.log(1.-target)-math.log(max(1e-6, 1.-expected['acc_mid']))
    if not session.warm_flag and 'opening_acc' in expected:
        gap = min(gap, math.log(1.-opening_floor(session, f, expected))
                  - math.log(max(1e-6, 1.-expected['opening_acc'])))
    return gap


def rate_slope(pred, f):
    model = getattr(pred.model, 'warmup', None)
    if model is not None:
        return model.rate_slope(f)
    pop, k = pred.model.pop, f["keys"]
    slope = pop["slope"].get(k, pop["slope"][0])
    curve = pop.get("curve", {}).get(k, pop.get("curve", {}).get(0, 0.))
    slope += 2 * curve * max(0., math.log(max(.05, f["overall"]) / pop.get("curve_knee", 4.)))
    return max(1., .8 * slope)


def quality_penalty(length, structure, rate=1.):
    short = 1.5 * max(0., math.log(90. / max(10., length)))
    return short + 2.8 * structure.get("drill", 0.) + .18 * abs(math.log(rate))


def profile(f):
    sk, st = f["sk"], f.get("nps", {})
    v = np.array([sk.get(k, 0.) for k in PROFILE] + [
        .65 * min(2., st.get("chord", 1.) - 1), st.get("wide", 0), st.get("repeat", 0),
        st.get("overlap", 0), .5 * min(2., st.get("burst", 1.)),
        .6 * st.get("active", .5), .35 * min(2., st.get("rhythm", 0))], float)
    return v / max(.001, np.linalg.norm(v))


_GROUPS = {}


def group(f):
    # The library ranking asks each chart's group ~10 times per request. A replaced "sk" is a new
    # chart; a website chart's features are decoded from fixed bytes, so its token alone suffices.
    sk, value = recdata.per_chart(_GROUPS, f, lambda f: (f["sk"], _group(f)))
    return value if sk is f["sk"] or recdata.identity(f) is not f else _group(f)


def _group(f):
    sk = f["sk"]
    family = {"rice": max(sk.get(k, 0) for k in ("stream", "delay", "dump", "jumpstream", "handstream")),
              "chords": sk.get("chordstream", 0),
              "chordjack": sk.get("chordjack", 0),
              "jack": max(sk.get(k, 0) for k in ("jackspeed", "jack", "minijack", "longjack")),
              "ln": max(sk.get(k, 0) for k in ("ln", "release", "hybrid", "inverse", "shield")),
              "trill": max(sk.get(k, 0) for k in ("jumptrill", "splittrill", "bracket")),
              "sv": sk.get("sv", 0)}
    return max(family, key=family.get)


def description(f):
    if f.get("description"):
        return f["description"]
    import skill_calc
    names = sorted(((v, k) for k, v in f["sk"].items() if k not in ("technical", "patterntech", "rhythmtech")), reverse=True)
    return " / ".join(skill_calc.NAMES.get(k, k) for v, k in names[:2])


def webmaps_workers():
    import webmaps
    return webmaps._workers()


def installed(db):
    """A manifest is machine-local, even when portable feature caches are copied."""
    origin, here = recdata.kv_get(db, "manifest_origin"), recdata.local_origin()
    if origin and origin != here:
        return {}, "Library belongs to another location — import local lazer maps"
    out = {}
    # Exhaust the metadata cursor before file checks, which can wait on disk
    # while another app connection needs to commit a live event.
    for row in db.execute("SELECT i.*,l.online_md5,l.audio_sha,l.audio_required,l.creator "
                          "FROM installed i LEFT JOIN installed_local l USING(sha256)").fetchall():
        d = dict(row)
        path = recdata.local_file(d["sha256"], here["root"])
        if d["keys"] not in recdata.SUPPORTED_KEYS or not path or not os.path.isfile(path):
            continue
        audio = recdata.local_file(d["audio_sha"], here["root"])
        if d["audio_required"] and (not audio or not os.path.isfile(audio)):
            continue
        out[d["sha256"]] = d
    return out, recdata.kv_get(db, "manifest_error") or ""


class Builder(threading.Thread):
    """One coalescing, low-priority NPS/Skills builder. No second observer/model/ledger.

    Sparse work is persistent: feature batches enrich the existing cache and each
    publication uses exact analyses, never a scaling guess advertised as a result.
    """
    def __init__(self, emit, busy=lambda: False):
        super().__init__(daemon=True, name="nps-build")
        self.emit, self.busy = emit, busy
        self.cv = threading.Condition()
        self.request = None
        self.halt = threading.Event()
        self._catalog_key = None
        self._catalog = {}
        self._analyses = collections.defaultdict(dict)
        self._current = set()
        self._lead = {}
        self._loaded_rowid = 0

    def configure(self, predictor, session, keys, revision, generation, skills_keys=(), selected_skills=(), active_mode="nps"):
        with self.cv:
            self.request = (predictor, session, tuple(keys), revision, generation,
                            tuple(skills_keys), skill_practice.normalize(selected_skills), active_mode)
            self.cv.notify()

    def _note(self, sha):
        """Keep `current` (charts analysed with this STRUCTURE_VERSION) in step with an analyses change.
        Answering it from the features on every publish decoded every website chart each time."""
        if any(f.get("nps", {}).get("version") == STRUCTURE_VERSION for f in self._analyses[sha].values()):
            self._current.add(sha)
        else:
            self._current.discard(sha)

    def stop(self):
        self.halt.set()
        with self.cv:
            self.cv.notify_all()

    def run(self):
        db = None
        try:
            while not self.halt.is_set():
                with self.cv:
                    while self.request is None and not self.halt.is_set():
                        self.cv.wait(2)
                    request = self.request
                    self.request = None
                if request is None or request[-1] not in ("nps", "skills"):
                    continue
                try:
                    if db is None:
                        db = recdata.connect()
                    self._build(db, request)
                except Exception as exc:
                    import traceback
                    traceback.print_exc()
                    self._catalog_key = None     # an interrupted catalogue is not reusable
                    _pred, _session, keys, revision, generation, skills_keys, selected_skills, mode = request
                    self.emit("error", {"error": f"{type(exc).__name__}: {exc}", "mode": mode,
                                        "keys_by_mode": {"nps": keys, "skills": skills_keys},
                                        "selected_skills": selected_skills, "revision": revision,
                                        "generation": generation})
        finally:
            if db is not None:
                db.close()

    def _build(self, db, request):
        pred, session, keys, revision, generation, skills_keys, selected_skills, active_mode = request
        if active_mode not in ("nps", "skills"):
            return                          # PP never waits for an invisible tab
        keys_by_mode = {"nps": keys, "skills": skills_keys}
        allowed = tuple(keys_by_mode[active_mode])
        calc = pred.calc
        imported = recdata.kv_get(db, "realm_import", {}).get("t")
        import webmaps
        web_path = os.path.join(webmaps.WEB_DIR, "web.pkl")
        web_mtime = os.path.getmtime(web_path) if os.path.exists(web_path) else None
        window = reach(pred, allowed, central_target(session)) if web_mtime else None
        catalog_key = (calc, allowed, imported, web_mtime, window)
        if catalog_key != self._catalog_key:
            full, self._catalog_error = installed(db)
            self._catalog = {s: i for s, i in full.items() if i["keys"] in allowed}
            self._catalog_key = catalog_key
            self._analyses = collections.defaultdict(dict)
            self._current = set()
            self._loaded_rowid = 0
            # Website charts join NPS and Skills alike, marked not installed (Next offers the download).
            web, web_feats = webmaps.catalog(set(allowed), {i["md5"] for i in full.values()}, window)
            for sha, inst in web.items():
                if sha not in self._catalog:
                    self._catalog[sha] = inst
                    if sha in web_feats:
                        self._analyses[sha] = web_feats[sha]
                        b = pred.md5_bid.get(inst["md5"])
                        pub = pred.pub["feats"].get(b, {})
                        if (pub.get("calc") == calc and inst["md5"] and pub.get("md5") == inst["md5"]
                                and pred.pub["maps"].get(b, {}).get("md5") == inst["md5"]):
                            # Replace only offered web anchors; current SQLite below still wins.
                            for rate in self._analyses[sha]:
                                if rate in pub and pub[rate].get("keys") == inst["keys"]:
                                    self._analyses[sha][rate] = pub[rate]
                    self._note(sha)
        catalog, error, analyses, current = self._catalog, self._catalog_error, self._analyses, self._current
        catalogs = {active_mode: catalog}
        # Reuse decoded features across scores/filter changes. SQLite rowids are
        # monotone for the append/replace-only derived feature table. A fresh
        # import/calculator/catalogue resets this cursor.
        upper = db.execute("SELECT COALESCE(MAX(rowid),0) FROM feats").fetchone()[0]
        while self._loaded_rowid < upper:
            if self.halt.is_set():
                return
            with self.cv:
                if self.request is not None:
                    return
            if self.busy():
                self.halt.wait(1)               # the existing gameplay pause policy
                continue
            # Exhaust each bounded page before decoding: a live read cursor
            # would block event commits on another rollback-journal connection.
            # New append/replacement rows above this request's bound wait for
            # the next request, so concurrent cache filling cannot prolong it.
            rows = db.execute("SELECT rowid,key,data FROM feats WHERE rowid>? AND rowid<=? "
                              "AND key LIKE ? ORDER BY rowid LIMIT 256",
                              (self._loaded_rowid, upper, f"%|{calc}")).fetchall()
            if not rows:
                self._loaded_rowid = upper      # skip an irrelevant calculator tail
                break
            for row in rows:
                self._loaded_rowid = row["rowid"]
                try:
                    sha, rate = row["key"].split("@", 1)
                    if sha in catalog:
                        analyses[sha][float(rate.split("|")[0])] = json.loads(row["data"])
                        self._note(sha)
                except (ValueError, TypeError):
                    continue                   # recompute a damaged derived entry
        for sha, inst in catalog.items():
            b = pred.md5_bid.get(inst["md5"])
            pub = pred.pub["feats"].get(b, {})
            if (pub.get("calc") == calc and inst["md5"] and pub.get("md5") == inst["md5"]
                    and pred.pub["maps"].get(b, {}).get("md5") == inst["md5"]):
                for rate, f in pub.items():
                    if isinstance(rate, float) and f.get("keys") == inst["keys"] and rate not in analyses[sha]:
                        analyses[sha][rate] = f
                        self._note(sha)
        predictions = {}
        # A play or refit re-predicts the whole library (~10 s for 20k charts). The previous request's
        # leading charts go first, so the shown list follows the new session within a second.
        lead = [s for s in self._lead.get(active_mode, ()) if s in catalog]
        if lead:
            quick, _ = candidates_from({s: catalog[s] for s in lead}, analyses, pred, session, mode=active_mode,
                                       selected_skills=selected_skills, predictions=predictions)
            self.emit("ready", {"states": {active_mode: {
                              "candidates": quick, "total": len(catalog), "analyzed": sum(s in current for s in catalog),
                              "failed": 0, "error": error, "partial": True}},
                          "keys_by_mode": keys_by_mode, "selected_skills": selected_skills, "revision": revision,
                          "generation": generation})
        # Deterministic per-chart order spreads unplayed coverage across collections.
        unknown = sorted((s for s in catalog if not analyses[s]), key=lambda s: hashlib.sha256(s.encode()).digest())
        enrich = [s for s in catalog if analyses[s] and s not in current]
        # Broad known-chart reuse first; don't serially exhaust only personal favorites.
        first = {sha: min(analyses[sha], key=lambda r: abs(r - 1)) for sha in enrich}
        _predict_all(pred, catalog, session, predictions, [(sha, r, analyses[sha][r]) for sha, r in first.items()])

        def initial_value(sha):
            rate = first[sha]
            f = analyses[sha][rate]
            e = predictions[(sha, rate)]
            slope = e.get('rate_slope') or rate_slope(pred, f)
            need = rate * math.exp(rate_shift(e, session, f, central_target(session, f)) / slope)
            density = f.get("notes", 0) / max(1000., f.get("length", 1)) * min(MAX_RATE, need)
            return density * (.05 if need < MIN_RATE * .85 else 1.)
        enrich.sort(key=lambda s: -initial_value(s))
        # Don't analyse only the global density leaders before variety exists.
        # Interleave strong representatives of every supported pattern/mode;
        # all of the remaining installed library still receives coverage.
        groups = collections.defaultdict(collections.deque)
        for sha in enrich:
            f = analyses[sha][min(analyses[sha], key=lambda r: abs(r - 1))]
            groups[(catalog[sha]["keys"], group(f))].append(sha)
        diverse = []
        while groups:
            for g in list(groups):
                diverse.append(groups[g].popleft())
                if not groups[g]:
                    del groups[g]
        enrich = diverse
        bad, count, last_publish = {}, 0, 0.
        refined = set()
        refined_counts = collections.Counter()
        refinement_groups = collections.Counter()
        evaluated = {active_mode: {}}
        def publish(candidates):
            states = {}
            for mode, subset in catalogs.items():
                self._lead[mode] = [c["sha"] for c in sorted(candidates[mode], key=lambda c: -c["base_value"])[:LEAD]]
                states[mode] = {"candidates": candidates[mode], "total": len(subset),
                    "analyzed": sum(s in current for s in subset),
                    "failed": sum(s in bad for s in subset), "error": error}
            self.emit("ready", {"states": states, "keys_by_mode": keys_by_mode,
                                "selected_skills": selected_skills, "revision": revision, "generation": generation})
        pending_unknown = collections.deque(unknown)
        pending_enrich = collections.deque(enrich)
        while not self.halt.is_set():
            with self.cv:
                if self.request is not None:
                    return
            if self.busy():
                self.halt.wait(1)
                continue
            pass_start = time.monotonic()
            candidates, fronts = {}, {}
            for mode, subset in catalogs.items():
                candidates[mode], fronts[mode] = candidates_from(
                    subset, analyses, pred, session, memo=evaluated[mode], mode=mode,
                    selected_skills=selected_skills, predictions=predictions,
                    skip_proposal=lambda sha, rate: (sha, rate) in refined or refined_counts[sha] >= 6 or sha in bad)
            if time.monotonic() - last_publish > 10 or count == 0:
                publish(candidates)
                last_publish = time.monotonic()
            jobs = []
            # Useful exact rate refinements and new unplayed maps advance together.
            def proposal_group(p):
                fs = analyses[p[1]]
                return (catalog[p[1]]["keys"], group(fs[min(fs, key=lambda r: abs(r - p[2]))]))
            # Objectives have different units: interleave their frontiers instead
            # of letting NPS's log-density reward starve ordinary skill maps.
            priority = (active_mode,)
            for frontier in fronts.values():
                frontier.sort(key=lambda p: -(p[0] - .45 * math.log1p(refinement_groups[proposal_group(p)])))
            proposals = []
            for i in range(max((len(f) for f in fronts.values()), default=0)):
                for mode in priority:
                    if i < len(fronts[mode]):
                        proposals.append(fronts[mode][i])
            for _score, sha, rate in proposals if count < 160 else ():
                if (sha, rate) not in refined and sha not in bad and refined_counts[sha] < 6:
                    jobs.append((sha, rate, {}))
                    refined.add((sha, rate))
                    refined_counts[sha] += 1
                    refinement_groups[proposal_group((_score, sha, rate))] += 1
                    if len(jobs) >= 2:
                        break
            def take():
                """The next queued maps: analysed ones at a new rate (enrich) and never-analysed ones."""
                out = []
                for queue_, n in ((pending_enrich, 24), (pending_unknown, 40)):
                    for _ in range(min(n, len(queue_))):
                        sha = queue_.popleft()
                        if sha in bad:
                            continue
                        fs = analyses[sha]
                        rate = min(fs, key=lambda r: abs(r - 1)) if fs else 1.
                        out.append((sha, rate, {rate: fs[rate]} if fs else {}))
                return out
            jobs += take()
            # A useful exact shortlist does not need an endless all-library refinement job (160 per update),
            # but unanalysed maps keep going until the library is done: a calculator change left ~10k waiting
            # for two weeks at 160 per update (user 2026-10-25). busy() still pauses it during gameplay.
            if not jobs:
                publish(candidates)
                return
            # In-process native batches on every core but two (niced threads, the GIL released while the
            # calculator runs). Spawned workers and per-chart round trips used to cost more than the Rust
            # calculation itself (~20 charts/s; a 52k-chart library took 45 min after a calculator change).
            # A pass analyses for at least as long as its bookkeeping (candidates for every catalogue, seconds
            # on a large library) took: a fixed ~50 maps a pass left the tail of a re-analysis at ~10/s.
            bookkeeping, analysing = time.monotonic() - pass_start, time.monotonic()
            while jobs:
                if self.halt.is_set() or self.request is not None:
                    return
                while self.busy() and not self.halt.is_set():
                    self.halt.wait(1)
                if self.halt.is_set() or self.request is not None:
                    return
                done = recdata.feats_batch([(catalog[s].get("path") or recdata.local_file(s), [r], a)
                                            for s, r, a in jobs], threads=webmaps_workers())
                results = [(s, {} if isinstance(fs, Exception) else fs,
                            f"{type(fs).__name__}: {fs}" if isinstance(fs, Exception) else None)
                           for (s, _r, _a), fs in zip(jobs, done)]
                for sha, fs, failure in results:
                    if failure:
                        bad[sha] = failure
                        continue
                    with db:
                        for rate, f in fs.items():
                            analyses[sha][rate] = f
                            self._note(sha)
                            db.execute("INSERT OR REPLACE INTO feats VALUES (?,?)",
                                       (f"{sha}@{rate:.3f}|{calc}", json.dumps(f)))
                    for memo in evaluated.values():
                        memo.pop(sha, None)
                    for rate in fs:
                        predictions.pop((sha, rate), None)
                    count += 1
                jobs = take() if time.monotonic() - analysing < bookkeeping else []
                self.halt.wait(.02)


def missed_rates(session, inst, target):
    """{rate: displayed accuracy} of this chart's rates whose last finish (14 days) missed the target."""
    by = getattr(session, "_attempts_by_map", {})
    att = (*by.get("md5:" + str(inst.get("md5")), ()), *by.get("sha:" + str(inst.get("sha256")), ()))
    last = {}
    for a in sorted(att, key=lambda a: a.get("end") or 0.):
        y = a.get("y_lazer", a.get("y"))
        if a.get("kind") == "finish" and a.get("rate") and y is not None:
            last[round(a["rate"], 2)] = 1 - math.exp(y)
    return {r: acc for r, acc in last.items() if abs(acc - target) > MISS_TOL}


def _predict_all(pred, catalog, session, predictions, todo):
    """predictions[(sha, rate)] for [(sha, rate, features)] still missing, in one native batch."""
    todo = [t for t in todo if (t[0], t[1]) not in predictions]
    if todo:
        for (sha, rate, _f), e in zip(todo, pred.predict_many(
                [(f, catalog[sha]["md5"] or catalog[sha]["sha256"], pred.md5_bid.get(catalog[sha]["md5"]), rate)
                 for sha, rate, f in todo], session)):
            predictions[(sha, rate)] = e


def practice_value(f, expected, rate, target, session):
    """Keep the accuracy target; earn the density objective through readiness.

    A completely cold NPS list should not prefer the least-certain wide-chord
    chart merely because it has the biggest raw note count. Start with reliable
    whole-map predictions, then restore the normal density reward continuously
    with relevant completed play. Neither rates nor chart families are banned.
    """
    model=getattr(session,'_warmup_model',None)
    if session.warm_flag:
        ready=1.
    elif model is not None:
        minutes=expected.get('activation_minutes')
        if minutes is None:
            minutes=session.activation_for(f)
        ready=-math.expm1(-minutes/(expected.get('warmup_tau') or model.parameters(f['keys'])['tau']))
    else:
        ready=-math.expm1(-session.activation(f['keys'],group(f))/6.)
    st=f.get('nps',{})
    length=st.get('play_span',f.get('length',0.)/1000)/rate
    return (ready*math.log(max(1.,st.get('nps',1.)))
            -12*abs(expected['acc_mid']-target)
            -(.35+1.-ready)*expected['sd_model']-quality_penalty(length,st,rate))


def candidates_from(catalog, analyses, pred, session, memo=None, skip_proposal=lambda _s, _r: False,
                    mode="nps", selected_skills=(), predictions=None):
    """Return exact-rate choices and a diverse frontier of cheap rate proposals."""
    # The model/session are fixed for one Builder request. Re-evaluate only
    # charts whose analysis changed, rather than predict the whole library on
    # every small feature batch. No cached value crosses a model/session update.
    memo = {} if memo is None else memo
    predictions = {} if predictions is None else predictions
    choices, proposals = [], []
    # Every missing prediction in one native batch (recommend.Predictor.predict_many).
    rateable = {sha: list(_rateable(analyses[sha], mode, selected_skills)) for sha in catalog if sha not in memo}
    _predict_all(pred, catalog, session, predictions,
                 [(sha, rate, f) for sha, rfs in rateable.items() for rate, f in rfs])
    for sha, inst in catalog.items():
        if sha in memo:
            choice, proposal = memo[sha]
            if choice is not None:
                choices.append(choice)
            if proposal is not None and not skip_proposal(sha, proposal[2]):
                proposals.append(proposal)
            continue
        memo[sha] = [None, None]
        fs = analyses[sha]
        rated = [(rate, f, predictions[(sha, rate)]) for rate, f in rateable[sha]]
        if not rated:
            continue
        target = central_target(session, rated[0][1])
        missed = missed_rates(session, inst, target)
        anchor = min(rated, key=lambda r: abs(rate_shift(r[2], session, r[1], target)))
        r, f, e = anchor
        # y=log(1-accuracy) is near-linear in log-rate locally. This is ONLY a
        # proposal: the subprocess calculates the proposed rounded rate exactly.
        slope = e.get('rate_slope') or rate_slope(pred, f)
        if len(rated) > 1:
            nearest = sorted((v for v in rated if abs(v[0] - r) > .015), key=lambda v: abs(v[0] - r))
            if nearest:
                rr, _ff, ee = nearest[0]
                empirical = (math.log(max(1e-6, 1 - ee["acc_mid"])) - math.log(max(1e-6, 1 - e["acc_mid"]))) / math.log(rr / r)
                if empirical > .2:
                    slope = min(8., max(.5, empirical))
        shift = rate_shift(e, session, f, target) / slope
        proposal = round(min(MAX_RATE, max(MIN_RATE, r * math.exp(shift))), 2)
        if not eligible(e, session, f) and proposal in fs and shift:
            # A hundredth-rate rounding cell can straddle the opening guard.
            # Do not stop at an ineligible 1.35 because the root rounds back to
            # 1.35: inspect the next unanalysed grid point toward feasibility.
            direction = -1 if shift < 0 else 1
            step = math.floor(r*100-1e-8) if direction < 0 else math.ceil(r*100+1e-8)
            while MIN_RATE <= step/100 <= MAX_RATE and step/100 in fs:
                step += direction
            if MIN_RATE <= step/100 <= MAX_RATE:
                proposal = step/100
        if round(r, 2) in missed:
            # the anchor is a rate you just missed: move from your result, not the forecast
            shift = (math.log(1. - target) - math.log(max(1e-6, 1. - missed[round(r, 2)]))) / slope
            proposal = round(min(MAX_RATE, max(MIN_RATE, r * math.exp(shift))), 2)
            if proposal == round(r, 2):
                proposal = round(min(MAX_RATE, max(MIN_RATE, r + (.01 if shift > 0 else -.01))), 2)
        st = f["nps"]
        proposed=dict(f,nps=dict(st,nps=st['nps']*proposal/r))
        quality = practice_value(proposed,dict(e,acc_mid=target),proposal,target,session)
        if mode == "skills":
            quality = skill_practice.value(f, dict(e, acc_mid=target), proposal, target, selected_skills)
        # More than a few exact refinements per chart per request is not useful.
        if proposal not in fs and (not eligible(e,session,f) or abs(e['acc_mid']-target) > .003
                                   or round(r, 2) in missed):
            item = (quality, sha, proposal, (inst["keys"], group(f)))
            memo[sha][1] = item
            if not skip_proposal(sha, proposal):
                proposals.append(item)
        best = None
        for rate, f, e in rated:
            st = f["nps"]
            if round(rate, 2) in missed or not eligible(e, session, f) or (st.get("vibro_focused")
                    and not (mode == "skills" and skill_practice.wants_vibro(selected_skills))):
                continue
            value = practice_value(f,e,rate,target,session)
            if mode == "skills":
                value = skill_practice.value(f, e, rate, target, selected_skills)
                if value is None:
                    continue
            if best is None or value > best[0]:
                best = (value, rate, f, e)
        if best is None:
            continue
        value, rate, f, e = best
        st = f["nps"]
        bid = inst["beatmap_id"] if (inst["beatmap_id"] or 0) > 0 else None
        online_match = bool(bid and inst.get("online_md5") and inst["online_md5"] == inst["md5"])
        choices.append(dict(e, sha=sha, sha256=sha, md5=inst["md5"], bid=bid,
                            title=f'{inst["artist"]} - {inst["title"]} [{inst["version"]}]',
                            search_meta={"Artist": inst["artist"], "Title": inst["title"],
                                         "Version": inst["version"], "Creator": inst.get("creator")},
                            keys=inst["keys"], rate=rate, var=rate_label(rate), nps=st["nps"],
                            length=st["play_span"] / rate, skill=description(f), description=description(f),
                            family=st["family"], group=group(f), profile=profile(f).tolist(), f=f,
                            installed=inst.get("installed", True), playcount=inst.get("playcount"),
                            online_match=online_match, purpose=mode, base_value=value,
                            why="Local map at a personally suitable rate", mode=mode,
                            practice_skills=skill_practice.normalize(selected_skills) if mode == "skills" else (),
                            matched_skills=skill_practice.match(f, selected_skills)[1] if mode == "skills" else ()))
        memo[sha][0] = choices[-1]
    # Keep patterns before imposing a global limit. This is retention, not quotas.
    by = collections.defaultdict(list)
    for item in proposals:
        by[item[3]].append(item)
    kept = heapq.nlargest(24, proposals)
    for values in by.values():
        kept += heapq.nlargest(4, values)
    unique = {(s, r): (v, s, r) for v, s, r, _g in kept}
    # Keep the actual eligible population: an equal per-category frontier would
    # turn a rare SV category into the same sampling mass as thousands of rice maps.
    return choices, sorted(unique.values(), reverse=True)


def _rateable(fs, mode, selected_skills):
    """(rate, features) of one chart that can be offered: playable rate, long enough, current structure."""
    for rate, f in fs.items():
        if not MIN_RATE <= rate <= MAX_RATE or f.get("notes", 0) < 100 or f.get("length", 0) < 10000:
            continue
        st = f.get("nps", {})
        if st.get("version") != STRUCTURE_VERSION or not st.get("nps"):
            continue
        if mode == "skills" and not skill_practice.match(f, selected_skills)[0]:
            continue
        yield rate, f


def rate_label(rate):
    return ("NM" if rate == 1. else "HT" if rate < 1. else "DT") + f" {rate:.2f}×"


def taste(db, now=None):
    """Shrunk completion/offer rates. Next alone is never a positive vote."""
    now = now or time.time()
    reset = db.execute("SELECT MAX(t) FROM events WHERE kind='taste_reset'").fetchone()[0] or 0
    since = max(reset, now - 90 * 86400)
    rows = recdata.tracked_events(db, since, ('offer', 'start', 'finish'))
    offers, starts, completed = {}, {}, set()
    for row in rows:
        if row['t'] <= since:
            continue
        info = recdata.event_info(row["info"])
        if row["kind"] == "offer" and info.get("mode") == "nps":
            offers[row["id"]] = (info, .5 ** ((now - row["t"]) / (30 * 86400)))
        elif row["kind"] == "start":
            starts[row["id"]] = info.get("offer_id")
        elif row["kind"] == "finish" and info.get("start_id") in starts:
            completed.add(starts[info["start_id"]])
    counts = collections.defaultdict(lambda: [0., 0.])
    contexts = collections.defaultdict(lambda: [0., 0.])
    for oid, (info, weight) in offers.items():
        length = info.get("length") or 240
        context = (info.get("keys"), 0 if length < 240 else 1 if length < 600 else 2,
                   int((info.get("accuracy") or .95) * 100 // 3))
        g = (info.get("keys"), info.get("group"), context)
        counts[g][0] += weight
        counts[g][1] += weight * (oid in completed)
        contexts[context][0] += weight
        contexts[context][1] += weight * (oid in completed)
    result = collections.defaultdict(lambda: [0., 0.])
    for (keys, group_, context), (a, b) in counts.items():
        ca, cb = contexts[context]
        baseline = (cb + 2) / (ca + 4)
        result[(keys, group_)][0] += a * ((b + 6 * baseline) / (a + 6) - baseline)
        result[(keys, group_)][1] += a
    return {g: max(-.25, min(.25, score / max(1., count))) for g, (score, count) in result.items()}


def weighted_candidates(candidates, session, keys, taste_profile=None, mode="nps", web=1.):
    """Each eligible chart family contributes quality-weighted sampling mass.

    Equally good maps naturally reproduce library proportions. Density, poor
    structure and fatigue fit still matter across categories; no group receives
    a guaranteed slot or a protected mass of otherwise unsuitable maps.
    """
    from recommend import event_key
    allowed, taste_profile = set(recdata.normalize_keys(keys)), taste_profile or {}
    families = {}
    for c in candidates:
        if c["keys"] not in allowed or not MIN_RATE <= c.get("rate", 1.) <= MAX_RATE:
            continue
        if web <= 0. and not c.get("installed", True):
            continue
        key = (c["keys"], c["family"])
        if key not in families or c["base_value"] > families[key]["base_value"]:
            families[key] = c
    rows = [dict(c) for c in families.values()]
    if not rows:
        return []
    # Twenty NPS is ordinary 7K variety at this ability. Follow the eligible
    # pool down for lower ability/cold states; above the reference, keep the
    # existing quality weights. Below it, use a steep but continuous soft tail.
    density_floor = {}
    if mode == "nps":
        for k in allowed:
            ns = [c["nps"] for c in rows if c["keys"] == k and c.get("nps", 0) > 0]
            if ns:
                density_floor[k] = min(20. * k / 7., .85 * float(np.quantile(ns, .70)))
        # Far below the floor is not NPS practice at all (user 2026-10-22: a 6 NPS map "should never appear").
        rows = [c for c in rows if not 0 < c.get("nps", 0) < .75 * density_floor.get(c["keys"], 0.)]
        if not rows:
            return []
    counts = collections.Counter((c["keys"], c["group"]) for c in rows)
    w = np.exp(np.array([c["base_value"] for c in rows]) - max(c["base_value"] for c in rows))
    pool = float(w.sum() ** 2 / (w ** 2).sum())        # effective number of charts the playlist draws from
    fresh = [session.freshness(event_key(c), c["length"], mode=mode, pool=pool) for c in rows]
    relax = sum(fresh) < 3
    recent = [a for a in session.attempts[-6:] if a.get("profile") and a.get("meaningful", True)]
    recent_families = {a.get("family") for a in recent[-4:]}
    recent_profiles = [np.array(a["profile"]) for a in recent]      # once, not once per chart
    profile_key = session.model_key()
    grouped = collections.defaultdict(list)
    for c, fr in zip(rows, fresh):
        value = c["base_value"] + .65 * math.log(max(.001, fr ** (.3 if relax else 1.)))
        floor = density_floor.get(c["keys"], 0.)
        if floor and c.get("nps", 0) > 0:
            value -= 24. * max(0., math.log(floor / c["nps"]))
        if c["family"] in recent_families:
            value -= 1.2
        if not c.get("installed", True):
            # Mild popularity prior for website charts only: ×0.7 near 10 plays → ×1 from 10k.
            value += math.log(.6 + .1 * min(4., math.log10(1 + (c.get("playcount") or 0))))
        cached_penalty = c.get("_profile_penalty")
        if cached_penalty and cached_penalty[0] == profile_key:
            value -= cached_penalty[1]
        elif recent:
            v = np.array(c["profile"])
            compatible = [a for a in recent_profiles if len(a) == len(v)]
            if compatible:
                value -= .18 * max(float(v @ a) for a in compatible)
        if len(allowed) > 1 and session.current_keys() in allowed and c["keys"] != session.current_keys():
            value -= .12
        c["value"] = value
        headline = f"{c['nps']:.1f} raw NPS" if mode == "nps" else (c.get("practice_description") or
                                   skill_practice.describe_match(c["f"], c.get("practice_skills")))
        c["why"] = (f"{headline} · {c['description']} · expect ~{100*c['acc_mid']:.1f}%"
                    + (" · freshness relaxed" if relax else ""))
        grouped[(c["keys"], c["group"])].append(c)
    maximum = max(c["value"] for c in rows)
    for group_, values in grouped.items():
        preference = math.exp(.65 * taste_profile.get(group_, 0.))
        for c in values:
            c["weight"] = math.exp(max(-20., c["value"]-maximum)) * preference
            c["category_share"] = counts[group_] / len(rows)
    return sorted(rows, key=lambda c: -c["weight"])


def mix(ranked, web):
    """Interleaves installed and not-installed maps, each in its own ranked order, at share web/(1+web).
    Sampling both from one weighted pool let whichever side had more mass fill the whole list (user 2026-10-25)."""
    pools = [[c for c in ranked if c.get("installed", True)], [c for c in ranked if not c.get("installed", True)]]
    share, out, taken = web / (1. + web), [], 0
    for i in range(len(ranked)):
        # A website chart only once its share is due, so a list never opens with one.
        side = 1 if pools[1] and (not pools[0] or taken + 1 <= share * (i + 1) + 1e-9) else 0
        taken += side
        out.append(pools[side].pop(0))
    return out


def select(candidates, session, keys, taste_profile=None, n_show=20, rng=None):
    rows = weighted_candidates(candidates, session, keys, taste_profile)
    if n_show is None:
        return rows
    # Slight anti-monotony in presentation, never an equal-category rotation.
    shown, used = [], collections.Counter()
    while rows and len(shown) < n_show:
        c = max(rows, key=lambda c: c["weight"] / (1 + .12 * used[(c["keys"], c["group"])]))
        rows.remove(c)
        shown.append(c)
        used[(c["keys"], c["group"])] += 1
    return shown
