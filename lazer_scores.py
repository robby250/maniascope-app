#!/usr/bin/env python3
"""
Import every mania score from osu!lazer's local database into the feedback store.

Reads a COPY of client.realm with Realm JS (node + the `realm` package, installed by
setup.sh into ~/.local/share/maniascope/realmjs). Never touches lazer's own file.
Run any time; already-imported scores are skipped.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
from contextlib import closing

import feedback
import recdata

import paths  # noqa: E402
LAZER_DATA = paths.lazer_data()
# Packaged builds ship node + realm beside the executable; source installs use setup.sh's copy.
BUNDLE = getattr(sys, "_MEIPASS", None)
REALM_JS = next((d for d in (BUNDLE and os.path.join(BUNDLE, "realmjs", "node_modules"),
                             os.path.join(paths.DATA, "realmjs", "node_modules")) if d and os.path.isdir(d)),
                os.path.join(paths.DATA, "realmjs", "node_modules"))
HERE = os.path.dirname(os.path.abspath(__file__))


def dump(realm=None, maps=False, recent=None):
    """→ list of raw score dicts (+ installed-map dicts with maps=True) from a snapshot of client.realm."""
    if not os.path.isdir(os.path.join(REALM_JS, "realm")):
        raise RuntimeError(f"realm JS not installed under {REALM_JS} — run ./setup.sh")
    with tempfile.TemporaryDirectory() as tmp:
        snap = os.path.join(tmp, "client.realm")
        shutil.copyfile(realm or os.path.join(LAZER_DATA, "client.realm"), snap)
        env = dict(os.environ, NODE_PATH=REALM_JS,
                   PATH=os.pathsep.join((os.path.join(BUNDLE or paths.DATA, "node"), os.path.expanduser("~/.local/bin"),
                                         os.environ.get("PATH", ""))))
        out = subprocess.run([shutil.which("node", path=env["PATH"]) or "node", os.path.join(HERE, "lazer_scores.js"), snap]
                             + (["maps"] if maps else ["recent", recent] if recent else []),
                             env=env, capture_output=True, text=True,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    lines = out.stdout.splitlines()
    if not lines or lines[-1] != "END":
        raise RuntimeError(f"realm dump failed:\n{out.stderr[-2000:]}")
    return [json.loads(l) for l in lines[:-1]]


def to_entry(s):
    st = json.loads(s["stats"] or "{}")
    hits = {"300": st.get("great", 0), "geki": st.get("perfect", 0), "katu": st.get("good", 0),
            "100": st.get("ok", 0), "50": st.get("meh", 0), "0": st.get("miss", 0)}
    mods = json.loads(s["mods"] or "[]")
    return {"id": s["id"], "sha256": s["hash"], "rate": feedback.rate_from_mods(mods)[0],
            "mods": "".join(m.get("acronym", "") for m in mods),
            "accuracy": round(100 * s["acc"], 2), "hits": hits,
            "rank": ("F", "D", "C", "B", "A", "S", "SH", "X", "XH")[s["rank"] + 1]
            if isinstance(s["rank"], int) and -1 <= s["rank"] < 8 else str(s["rank"]),
            "max_combo": s["combo"], "score": s["total"], "played": s["date"] or "",
            "player": s["user"] or ""}


def main():
    raw = dump()
    users = {}
    for s in raw:
        users[s["user"]] = users.get(s["user"], 0) + 1
    me = max(users, key=users.get) if users else None
    db = feedback.load()
    added = 0
    with closing(recdata.connect()) as store:
        for s in raw:
            if s["user"] != me:
                continue                       # guests / other accounts on this install
            e = to_entry(s)
            sha = e["sha256"]
            added += feedback.add_score(db, e, os.path.join(feedback.LAZER_FILES, sha[0], sha[:2], sha), store)
    feedback.save(db)
    print(f"{len(raw)} mania scores in lazer, {users.get(me, 0)} by {me!r}; "
          f"{added} new, {len(db['scores'])} stored")


if __name__ == "__main__":
    main()
