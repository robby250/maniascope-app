"""Player ability in chart-difficulty units, using the existing personal model.

A skill estimate is the difficulty giving 94% displayed accuracy on a diverse
set of that player's representative chart shapes. It is not a raw regression
coefficient, an average of personal bests, or a claimed population percentile.
No chart analyses or model fits are started by this page.
"""
import collections
import math

import numpy as np

import recdata
import skill_practice
import score_units

TARGET = .94


def _quantile(values, weights, q=.5):
    pairs = sorted(zip(values, weights))
    total = sum(w for _, w in pairs)
    acc = 0.
    for value, weight in pairs:
        acc += weight
        if acc >= total*q:
            return float(value)
    return float(pairs[-1][0]) if pairs else None


def _capacity(model, f, target_y):
    # Shape, timing-window and sustained-demand context stay fixed; only the
    # calibrated difficulty coordinate is inverted. Unseen-chart predictions
    # omit an individual map's learned affinity/public residual.
    lo, hi = math.log(.1), math.log(60.)
    for _ in range(24):
        mid = (lo+hi)/2
        candidate = dict(f, overall=math.exp(mid))
        mu = model.predict(candidate, "profile:unseen", None, 1.)[0]
        if mu < target_y:
            lo = mid
        else:
            hi = mid
    value = math.exp((lo+hi)/2)
    lower = model.predict(dict(f, overall=value*.99), "profile:unseen", None, 1.)[0]
    upper = model.predict(dict(f, overall=value*1.01), "profile:unseen", None, 1.)[0]
    slope = max(.25, (upper-lower)/math.log(1.01/.99))
    return value, slope


def build(rec):
    from recommend import play_history
    a, b = rec._shown
    target_y = (math.log(1-TARGET)-a)/max(.1, b)
    model=getattr(rec,'accuracy_model',None) or rec.model
    if getattr(rec,'accuracy_model',None) is not None:target_y=math.log(1-TARGET)
    by_key = collections.defaultdict(list)
    for r in rec.rows:
        if r["keys"] in recdata.SUPPORTED_KEYS and not r.get("rd") and r["f"].get("overall", 0) > .1:
            by_key[r["keys"]].append(r)
    result = {}
    for keys in recdata.SUPPORTED_KEYS:
        rows = by_key[keys]
        # Many retries/rates of one chart are not independent coverage. Choose
        # the highest played difficulty as its representative, not its best acc.
        charts = {}
        for row in rows:
            if row["chart"] not in charts or row["f"]["overall"] > charts[row["chart"]]["f"]["overall"]:
                charts[row["chart"]] = row
        floor = float(np.percentile([r["f"]["overall"] for r in charts.values()], 65))*.6 if charts else 0.
        representatives = [r for r in charts.values() if r["f"]["overall"] >= floor]
        groups = {}
        capacities = {}
        for group in ("overall", *skill_practice.dimensions(keys)):
            relevant = [r for r in representatives if group == "overall" or
                        group in skill_practice.match(r["f"], [group])[1]]
            newest = max((r.get("ts") or 0 for r in relevant), default=0)
            count = len(relevant)
            if count < 3 or keys not in model.keys:
                groups[group] = {"value": None, "maps": count, "plays": 0, "newest": newest,
                                 "confidence": "Not enough evidence"}
                continue
            # Quantile-spaced shapes, not the most retried/favourite charts.
            relevant.sort(key=lambda r: (r["f"]["overall"], r["chart"]))
            chosen = [relevant[i] for i in np.linspace(0, count-1, min(32, count)).astype(int)]
            values, slopes, weights, display_factors = [], [], [], []
            for row in chosen:
                identity=(row['chart'],row['rate'])
                if identity not in capacities:
                    capacities[identity]=_capacity(model,row['f'],target_y)
                value, slope = capacities[identity]
                values.append(value); slopes.append(slope)
                # Ability has no single chart shape; use the same representative
                # sample's median display correction, leaving its fitted value intact.
                display_factors.append(score_units.feature_display_factor(row['f']))
                weights.append(.5**(max(0., model.now-row["t"])/12.))
            value = _quantile(values, weights)
            slope = _quantile(slopes, weights)
            shape_spread = (math.log(_quantile(values, weights, .85))-
                            math.log(_quantile(values, weights, .15)))/2
            level_var = model.level_var.get(keys, .25)
            # Conservative estimate spread includes shape ambiguity and sparse
            # coverage. This is not labelled as a calibrated 95% confidence CI.
            spread = math.sqrt(level_var/max(.25, slope*slope) + shape_spread**2 + .12**2/min(25, count))
            eff = sum(.5**(max(0., rec.model.now-r["t"])/12.) for r in relevant)
            confidence = "Strong history" if count >= 40 and eff >= 20 else "Moderate history" if count >= 10 else "Limited history"
            members = {r["chart"] for r in relevant}
            highest = max(r["f"]["overall"] for r in relevant)
            examples = []
            for r in relevant[-3:]:
                inst = rec.installed.get(r["chart"], {})
                title = (f"{inst.get('artist','')} - {inst['title']} [{inst.get('version','')}]" if inst.get('title')
                         else rec.pub["maps"].get(r.get("b"), {}).get("file", "Unavailable chart title"))
                examples.append({"title": title, "rate": r["rate"], "difficulty": score_units.displayed_feature(r["f"])})
            groups[group] = {"value": value, "low": value*math.exp(-spread), "high": value*math.exp(spread),
                             "slope": slope, "display_factor": _quantile(display_factors, weights),
                             "maps": count, "plays": sum(r["chart"] in members for r in rows),
                             "newest": newest, "confidence": confidence,
                             "references": len(chosen), "examples": examples,
                             "tested_to": max(score_units.displayed_feature(r['f']) for r in relevant)}
            if value > highest*1.35:
                # Easy/near-perfect historical scores do not identify the
                # player's limit. Do not present a huge extrapolated rating as
                # a measured 5K/9K strength merely because the solver found it.
                groups[group].update(value=None, confidence="Limit not measured", extrapolated=True)
        result[keys] = {"keys": keys, "groups": groups, "maps": len(charts), "plays": len(rows),
                        "special": {"mash_maps": sum(r["f"].get("sk", {}).get("mash", 0.) >= .3 for r in charts.values())}}
    # Only frozen pre-play predictions can support a model-error diagnosis.
    # Collapse retries by exact chart, and require several charts before a label.
    residuals = collections.defaultdict(lambda: collections.defaultdict(list))
    for row in play_history(rec.db, limit=300):
        p = row["prediction"]
        f = p.get("features")
        if row.get("expected") is None or not f or row["keys"] not in result:
            continue
        error = 100*(row["actual"]-row["expected"])
        for group in skill_practice.match(f)[1]:
            residuals[(row["keys"], group)][row["sha256"]].append(error)
    for keys, page in result.items():
        diagnostics = []
        for (k, group), charts in residuals.items():
            if k != keys or len(charts) < 4:
                continue
            errors = [float(np.mean(es)) for es in charts.values()]
            delta = float(np.mean(errors))
            se = float(np.std(errors))/math.sqrt(len(errors))
            if abs(delta) > max(.75, 1.5*se):
                diagnostics.append({"skill": group, "delta": delta, "maps": len(charts)})
        page["diagnostics"] = sorted(diagnostics, key=lambda d: -abs(d["delta"]))[:3]
    return result


def current(profile, session):
    """Cheap session overlay. This never alters the persistent skill estimate."""
    pages = {}
    for keys, old in profile.items():
        page = dict(old, groups={})
        for group, oldrow in old["groups"].items():
            row = dict(oldrow)
            skill = None if group == "overall" else skill_practice.session_skill(group, keys)
            row["activation"] = session.activation(keys, skill)
            if row["value"] is not None:
                correction = session.correction(keys, skill) + session.warmup_penalty(keys, skill)
                row["current"] = row["value"]*math.exp(-correction/row["slope"])
            for field in ('value', 'low', 'high', 'current'):
                if field in row:
                    row[field] = score_units.displayed_value(row[field], keys, factor=row.get('display_factor', 1.))
            page["groups"][group] = row
        pages[keys] = page
    return {"target": TARGET, "pages": pages, "phase": session.phase()}
