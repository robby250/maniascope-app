#!/usr/bin/env python3
"""
Saved maps and comparative judgments — local evidence for calibration.

Stored in ~/.config/maniascope/feedback.json (per machine; fetch over ssh). An entry is a chart identity
(sha256 of the file) AT A RATE; judgments compare two entries. Nothing here
retunes the model: run `feedback.py` to see where the current model agrees or
disagrees with the saved judgments.
"""
import hashlib
import json
import os
import time
from contextlib import closing

import paths  # noqa: E402
FILE = os.path.join(paths.CONFIG, "feedback.json")
LAZER_FILES = os.path.join(paths.lazer_data(), "files")
VERDICTS = ("A harder", "about equal", "B harder", "unsure")
EQUAL_BAND = 0.05   # |log ratio| under this counts as "about equal"


def _read(path):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            db = json.load(fh)
    except (OSError, ValueError):
        db = {}
    for key, empty in (("maps", {}), ("judgments", []), ("scores", [])):
        db.setdefault(key, empty)
    return db


def rate_from_mods(mods):
    """tosu `play.mods` → (constant rate | None, note). Unknown is never 1.0."""
    arr = mods.get("array") if isinstance(mods, dict) else mods
    if not isinstance(arr, list):
        return None, "rate not reported"
    rate = 1.0
    for m in arr:
        ac = str(m.get("acronym", "")).upper()
        if ac in ("WU", "WD", "AS"):
            return None, f"{ac} is variable speed — unsupported, set the rate manually"
        if ac in ("DT", "NC", "HT", "DC"):
            rate = (m.get("settings") or {}).get(
                "speed_change", 1.5 if ac in ("DT", "NC") else 0.75)
    if not isinstance(rate, (int, float)) or not 0.1 <= rate <= 5.0:
        return None, f"implausible rate {rate!r}"
    return float(rate), ""


def constant_speed(mods):
    """tosu `play.mods` include Constant Speed (scroll changes removed)."""
    arr = mods.get("array") if isinstance(mods, dict) else mods
    return isinstance(arr, list) and any(str(m.get("acronym", "")).upper() == "CS" for m in arr)


def load():
    return _read(FILE)


def chart_path(m):
    """Stored path if it exists here, else the same sha in this machine's lazer store."""
    p = m.get("path", "")
    if os.path.isfile(p):
        return p
    sha = m["sha256"]
    return os.path.join(LAZER_FILES, sha[0], sha[:2], sha)


def save(db):
    os.makedirs(os.path.dirname(FILE), exist_ok=True)
    tmp = FILE + ".part"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(db, fh, ensure_ascii=False, indent=1)
    os.replace(tmp, FILE)


def entry_id(sha, rate):
    return f"{sha}@{rate:.2f}"


def add_map(db, path, chart, rate):
    """Save a chart at a rate; returns its id."""
    with open(path, "rb") as fh:
        sha = hashlib.sha256(fh.read()).hexdigest()
    eid = entry_id(sha, rate)
    db["maps"][eid] = {"sha256": sha, "rate": round(rate, 2), "path": path,
                       "keys": chart.keys, "artist": chart.artist, "title": chart.title,
                       "version": chart.version, "creator": chart.creator}
    return eid


def add_score(db, score, path, store=None):
    """Record one of the user's own results (from tosu's results screen); False if known."""
    import recdata
    if store is None:
        with closing(recdata.connect()) as store:
            return add_score(db, score, path, store)
    if not recdata.score_allowed(store, score):
        return False
    same = ("sha256", "score", "accuracy", "max_combo", "played")
    if any(all(s.get(k) == score.get(k) for k in same) or
           (score.get("id") and s.get("id") == score["id"]) for s in db["scores"]):
        return False
    db["scores"].append(dict(score, path=path, seen=time.strftime("%F")))
    return True


def label(m):
    return f"{m['title']} [{m['version']}] · {m['keys']}K · {m['rate']:.2f}×"


def remove_map(db, eid):
    db["maps"].pop(eid, None)
    db["judgments"] = [j for j in db["judgments"] if eid not in (j["a"], j["b"])]


def judge(db, a, b, verdict, note, model):
    """Record (or correct) the judgment for the pair a/b."""
    db["judgments"] = [j for j in db["judgments"] if {j["a"], j["b"]} != {a, b}]
    db["judgments"].append({"a": a, "b": b, "verdict": verdict, "note": note,
                            "model_when_judged": model, "date": time.strftime("%F")})


def main():
    import math
    import skill_calc
    db = load()
    print(f"model {skill_calc.MODEL_VERSION} · {len(db['maps'])} saved · "
          f"{len(db['judgments'])} judgments")
    agree = total = 0
    for j in db["judgments"]:
        ma, mb = db["maps"][j["a"]], db["maps"][j["b"]]
        try:
            ra, rb = (skill_calc.compute(skill_calc.parse_osu(chart_path(m)), m["rate"])
                      for m in (ma, mb))
        except (OSError, skill_calc.ChartError) as exc:
            print(f"  ?  {label(ma)} vs {label(mb)}: {exc}")
            continue
        oa, ob = ra["scores"]["overall"], rb["scores"]["overall"]
        d = math.log(oa / ob) if oa > 0 and ob > 0 else 0.0
        model = "about equal" if abs(d) < EQUAL_BAND else "A harder" if d > 0 else "B harder"
        if j["verdict"] == "unsure":
            mark = " ~ "
        else:
            total += 1
            agree += model == j["verdict"]
            mark = " ok" if model == j["verdict"] else "BAD"
        print(f"{mark}  {oa:5.1f} {skill_calc.describe(ra)[0]:<22} {label(ma)}\n"
              f"     {ob:5.1f} {skill_calc.describe(rb)[0]:<22} {label(mb)}\n"
              f"     you: {j['verdict']}  model: {model}  {j.get('note', '')}")
    if total:
        print(f"agreement {agree}/{total}")
    report_scores(db)


def _od(path):
    """OverallDifficulty of a chart (its 300 window: 64 − 3·OD ms)."""
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if line.startswith("OverallDifficulty:"):
                return float(line.split(":", 1)[1])
            if line.startswith("[HitObjects]"):
                break
    return 5.0


def report_scores(db):
    """Your own results: fit log(1−acc) = a + b·log(rating) + c·log(hit window), then show
    the leftover per map character. + = you did worse than the rating predicts → UNDERrated."""
    import math
    import skill_calc
    # thousands of scores → cache the per-map@rate analysis on disk, keyed by model version
    cache_file = os.path.join(paths.CACHE, "score_analysis.json")
    try:
        with open(cache_file, "r", encoding="utf-8") as fh:
            cache = json.load(fh)
    except (OSError, ValueError):
        cache = {}
    if cache.get("model") != skill_calc.MODEL_VERSION:
        cache = {"model": skill_calc.MODEL_VERSION}
    rows, dirty = [], False
    # contemporaneous only: your level drifts, so compare scores within a year of your newest one;
    # Constant Speed / Invert / Hold Off change what is read or played and are left out; >99 % is
    # the timing floor
    newest = max((s.get("played", "") for s in db["scores"]), default="")
    since = str(int(newest[:4]) - 1) + newest[4:10] if newest[:4].isdigit() else ""
    skipped = {"old": 0, "cs": 0, ">99%": 0}
    for s in db["scores"]:
        if not s.get("rate") or s.get("rank") == "F" or s["accuracy"] / 100 < 0.80:
            continue
        mods = str(s.get("mods", ""))
        why = ("old" if s.get("played", "") < since
               else "cs" if {"CS", "IN", "HO"} & {mods[i:i + 2] for i in range(0, len(mods), 2)}
               else ">99%" if s["accuracy"] / 100 > 0.99 else None)
        if why:
            skipped[why] += 1
            continue
        key = entry_id(s["sha256"], s["rate"])
        if key not in cache:
            try:
                path = chart_path(s)
                chart = skill_calc.parse_osu(path)
                res = skill_calc.compute(chart, s["rate"])
                cache[key] = {"keys": chart.keys, "title": chart.title, "version": chart.version,
                              "od": _od(path), "notes": res["notes"], "ln_notes": res["ln_notes"],
                              "scores": res["scores"], "levels": res["levels"] if res["horizon"] else None,
                              "dom": skill_calc.describe(res)[0] if res["horizon"] else ""}
            except (OSError, skill_calc.ChartError):
                cache[key] = None
            dirty = True
        c = cache[key]
        if not c or c["notes"] < 300 or c["scores"]["overall"] < 1:
            continue
        rows.append((s, c, math.log(max(10.0, 64 - 3 * c["od"]) / s["rate"])))  # noqa
    if dirty:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        with open(cache_file, "w", encoding="utf-8") as fh:
            json.dump(cache, fh)
    if len(rows) < 30:
        if rows:
            print(f"\n{len(rows)} usable scores — need ~30+ for a diagnosis (python3 lazer_scores.py imports lazer's)")
        return
    # best score per map@rate only: a retry-until-pass session is one data point
    best = {}
    for r in rows:
        k = (r[0]["sha256"], r[0]["rate"])
        if k not in best or r[0]["accuracy"] > best[k][0]["accuracy"]:
            best[k] = r
    rows = list(best.values())
    # keymodes are separate skills: fit one level per keymode (>= 30 maps), maps within the
    # top 70 % of that keymode's range only (easy maps are capped by timing noise, not skill)
    by_k = {}
    for r in rows:
        by_k.setdefault(r[1]["keys"], []).append(r)
    rows = []
    for k, lst in by_k.items():
        if len(lst) < 30:
            continue
        top = sorted(x[1]["scores"]["overall"] for x in lst)[int(0.9 * (len(lst) - 1))]
        rows += [x for x in lst if x[1]["scores"]["overall"] >= 0.7 * top]
    if len(rows) < 30:
        return
    n = len(rows)
    y = [math.log(1 - r[0]["accuracy"] / 100) for r in rows]
    x1 = [math.log(r[1]["scores"]["overall"]) for r in rows]
    x2 = [r[2] for r in rows]
    keys = [r[1]["keys"] for r in rows]

    def demean(v):   # remove the per-keymode mean → keymode intercepts drop out
        m, c = {}, {}
        for k, a in zip(keys, v):
            m[k] = m.get(k, 0.0) + a
            c[k] = c.get(k, 0) + 1
        return [a - m[k] / c[k] for k, a in zip(keys, v)]
    y, x1, x2 = demean(y), demean(x1), demean(x2)
    # Your own maps are chosen to be playable (rate picked until they are), so accuracy vs
    # rating is flat by selection; use the public-score slope and hit-window term instead
    # (osu! data dump, top 1000 mania players — see README) and fit only the keymode levels.
    cw = -0.9
    slope = [{4: 3.4, 7: 2.95}.get(k, 2.5) for k in keys]
    own = sum(a * c for a, c in zip(x1, y)) / (sum(a * a for a in x1) or 1e-9)
    off = [(c - s * a - cw * b) / s for a, b, c, s in zip(x1, x2, y, slope)]
    print(f"\nYour scores: {n} maps (best per map@rate, top 70 % of each keymode's range; left out: "
          f"{skipped['old']} older than {since[:10]}, {skipped['cs']} CS/IN/HO, {skipped['>99%']} above 99 %); "
          f"own accuracy-vs-rating slope {own:.2f} (public players: 3.4 in 4K, 2.95 in 7K — below that "
          f"means you pick rates to keep maps playable, so the fixed public slope is used).")
    print("Leftover per map character (log ratio of rating; + = harder for you than rated, − = easier):")

    def table(name, key, order=None):
        g = {}
        for r, o in zip(rows, off):
            g.setdefault(key(r), []).append(o)
        for k in (order or sorted(g)):
            v = g.get(k)
            if v and len(v) >= 8 and k is not None:
                v.sort()
                print(f"  {name:<10} {str(k):<14} {sum(v) / len(v):+.3f}  (median {v[len(v) // 2]:+.3f}, n={len(v)})")

    def lead(r):
        sc = r[1]["scores"]
        return skill_calc.NAMES[max((k for k in skill_calc.skills(r[1]["keys"]) if k != "stamina"),
                                    key=lambda k: sc.get(k, 0))]
    table("skill", lead, list(dict.fromkeys(skill_calc.NAMES.values())))
    table("LN share", lambda r: ("0" if r[1]["ln_notes"] < .02 * r[1]["notes"] else "<25%" if r[1]["ln_notes"] < .25 * r[1]["notes"]
                                 else "<60%" if r[1]["ln_notes"] < .6 * r[1]["notes"] else ">=60%"), ("0", "<25%", "<60%", ">=60%"))
    table("keys", lambda r: f"{r[1]['keys']}K")
    table("rating", lambda r: f"{int(r[1]['scores']['overall'])}-{int(r[1]['scores']['overall']) + 1}")
    table("rate", lambda r: f"{r[0]['rate']:.2f}x")
    table("length", lambda r: "sustained" if r[1]["levels"]["120.0"] >= .85 * r[1]["scores"]["overall"] else "burst" if r[1]["levels"]["120.0"] < .7 * r[1]["scores"]["overall"] else "mixed", ("burst", "mixed", "sustained"))
    worst = sorted(zip(off, rows), key=lambda t: -abs(t[0]))[:12]
    print("Largest single disagreements:")
    for o, (s, c, _w) in worst:
        print(f"  {o:+.2f}  {c['scores']['overall']:5.1f} {s['accuracy']:6.2f}% {s['rate']:.2f}x "
              f"{c['dom']:<20} {c['title']} [{c['version']}] {c['keys']}K")


if __name__ == "__main__":
    main()
