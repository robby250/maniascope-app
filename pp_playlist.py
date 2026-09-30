"""A broad PP playlist, with the player's own safe/peak scoring envelope.

This is selection, not a second accuracy model. Ratings, central accuracy and
the exact weighted-account PP gain all come from the shared recommender.
"""
import collections
import hashlib
import json
import math

import numpy as np

import recdata


def reference(db, ledger):
    """Empirical accuracy/PP endpoints, one counted best per map (not retry count)."""
    order = sorted(ledger.best, key=lambda b: -ledger.best[b][0])
    ranks = {b: i+1 for i, b in enumerate(order)}
    rows = collections.defaultdict(list)
    seen = set()
    by_bid = {r[0]: r[1] for r in db.execute("SELECT beatmap_id,keys FROM installed WHERE beatmap_id>0")}
    for r in db.execute("SELECT s.beatmap_id,s.pp,s.stats,i.keys FROM scores s "
                        "LEFT JOIN installed i ON i.sha256=s.sha256 WHERE s.pp>0 AND s.ranked=1 ORDER BY s.pp DESC"):
        bid = r["beatmap_id"]
        if bid in seen or bid not in ranks or abs(r["pp"]-ledger.best[bid][0]) > .01:
            continue
        seen.add(bid)
        try:
            acc = recdata.acc_lazer(json.loads(r["stats"]))
        except (TypeError, ValueError):
            continue
        keys = r["keys"] or by_bid.get(bid)
        if keys in recdata.SUPPORTED_KEYS and 0 < acc <= 1:
            rows[keys].append((r["pp"], acc))
    result = {}
    for k, values in rows.items():
        values = sorted(values, reverse=True)[:100]
        safe = values[len(values)//2:] or values
        peak = values[:max(1, len(values)//5)]
        result[k] = {"safe_pp": float(np.median([p for p, _ in safe])),
                     "peak_pp": float(np.median([p for p, _ in peak])),
                     "comfort": min(.99, max(.965, float(np.median([a for _, a in safe])))),
                     # The peak tier contains both precise LN plays and lower-
                     # accuracy speed/peak plays. Its lower quartile describes
                     # a demonstrated push, not an invented universal 94% floor.
                     "peak_acc": min(.98, max(.88, float(np.percentile([a for _, a in peak],25)))),
                     "maps": len(values)}
    return ranks, result


def push_share(steps):
    """0 → 1 along the push staircase, on the same half-sd steps as the challenge ceiling."""
    from recommend import PUSH_STEP
    return 1. - math.exp(-PUSH_STEP * steps)


def readiness(session, keys, skill):
    active = min(1., session.activation(keys, skill)/6.)
    # Warming up unlocks ordinary PP work; each beaten best moves further toward the
    # player's high-PP, lower-accuracy edge. A bad result pulls back at once.
    form = session.correction(keys, skill)
    return active * (.45 + .5 * push_share(session.push(keys, skill))) * math.exp(-3. * max(0., form))


def comfort(rec, keys):
    return getattr(rec, "pp_reference", {}).get(keys, {}).get("comfort", .975)


def preview(rows, count, seed, focus=0.):
    """Stable weighted shuffle. Merely repainting/restarting does not consume it."""
    def priority(c):
        identity = c.get("md5") or c.get("sha") or c.get("bid") or c.get("title")
        raw = hashlib.blake2b(f"{seed}:{identity}".encode(), digest_size=8).digest()
        u = (int.from_bytes(raw, "big") + 1) / (2**64 + 1)
        # Zero samples by weight; one keeps the highest weights in order.
        return (1-focus)*math.log(max(1e-300, -math.log(u))) - math.log(max(1e-12, c["weight"]))
    return sorted(rows, key=priority)[:count]


def prepare(rec, candidates, session):
    from recommend import event_key, reason
    rows = []
    k0 = session.current_keys() or rec.home_keys()
    ledger = getattr(rec, 'ledger', None)
    floor = float(ledger.P[99]) if ledger is not None and len(ledger.P)>=100 else 0.
    for source in candidates:
        c = dict(source)
        if not c["ranked"] or c["gain"] <= .001 or c["sd_model"] >= 1. or c.get("p_up", 0.) < .12:
            continue
        keys, skill = c["keys"], c.get("skill")
        acc = c.get("acc_mid", 1.)
        cold = session.activation(keys) < 1.5
        if cold and acc < comfort(rec, keys):
            continue
        contender = c.get('best_rank',101)<=100 or c.get('pp_hi',c.get('pp_mid',float('inf')))>=floor
        state = readiness(session, keys, skill)
        ref = getattr(rec, "pp_reference", {}).get(keys, {})
        target_acc = (1-state)*ref.get("comfort", .975) + state*ref.get("peak_acc", .93)
        # Not a hard accuracy floor for ordinary PP play: difficulty can increase
        # as readiness improves, including the user's low-acc top-play style.
        accuracy_fit = math.exp(-.5 * (max(0., target_acc-acc)/.025)**2)
        pp = c.get("pp_mid", c.get("best") or 1.)
        target_pp = (1-state)*ref.get("safe_pp", pp) + state*ref.get("peak_pp", pp)
        tier_fit = math.exp(-.5 * (math.log(max(1., pp)/max(1., target_pp))/.35)**2)
        # Compress, do not erase, utility. A 46pp outlier must not remove a
        # hundred credible 0.1–3pp improvements before freshness is considered.
        utility = .3 + math.log1p(c["gain"]/.05)**.65
        headroom = max(0., pp-(c.get("best") or pp)) / max(20., pp*.1)
        fr = session.freshness(event_key(c), c["length"], mode="pp")
        continuity = 1. if keys == k0 else .45
        c["fresh"] = fr
        c["purpose"] = "probe" if cold else "push" if c.get("push_only") or state > .8 else "farm"
        # Likely improvements come far more often than long shots (user 2026-09-30).
        c["weight"] = (utility * (.2+.8*tier_fit) * accuracy_fit * continuity * (1+.3*min(2., headroom)) * fr
                       * c.get("p_up", 1.))
        c["why"] = (reason(c) + f" · expected {100*acc:.1f}%"
                    + (" · comfortable keymode probe" if cold else " · peak opportunity" if state > .8 else " · current-form fit")
                    + (f" · current top #{c['best_rank']}" if c.get("best_rank", 101) <= 100 else
                       " · new/top-100 contender" if contender else " · smaller PP improvement"))
        rows.append(c)
    # Offer/skip never promises a repeat. Only relax when *all* credible choices
    # have been cycled; there is no utility-floor trap shrinking the pool first.
    fresh = [c for c in rows if c["fresh"] >= .08]
    relaxed = bool(rows) and not fresh
    if not relaxed:
        rows = fresh
    elif rows:
        for c in rows:
            c["weight"] /= max(1e-6, c["fresh"])**.5
    # Cycle the eligible charts before offering the same favourites again.
    # Persisted offers survive restarts; next/open are the same exposure.
    counts = collections.Counter(e['beatmap'] for e in session.all
                                 if e['kind']=='offer' and e['info'].get('mode', 'pp')=='pp')
    if rows:
        least = min(counts[event_key(c)] for c in rows)
        rows = [c for c in rows if counts[event_key(c)]==least]
    last = next((e["beatmap"] for e in reversed(session.all) if e["kind"] == "offer"), None)
    if len(rows) > 1:
        rows = [c for c in rows if event_key(c) != last]
    return rows, relaxed
