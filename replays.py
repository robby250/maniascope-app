"""Durable evidence for tracked attempts. Lazer's .osr payload is already compressed.

Hashes identify files, not machine-specific absolute paths. Never writes to the
game's file store, and never archives scores from a tracking-disabled interval.
"""
import hashlib
import json
import os
import re
import tempfile

import recdata

SCHEMA = """
CREATE TABLE IF NOT EXISTS replay_evidence(
 score_key TEXT PRIMARY KEY, score_uuid TEXT, chart_sha256 TEXT, chart_md5 TEXT,
 replay_sha256 TEXT, archive_path TEXT, pauses TEXT, mods TEXT, rate REAL,
 played TEXT, status TEXT, error TEXT);
CREATE INDEX IF NOT EXISTS replay_score_uuid ON replay_evidence(score_uuid);
"""


def archive(db, raw, root=None):
    """Link imported/live score keys to files; repeated calls are idempotent.

    Only observed attempts are archived. Old imported bests remain ordinary
    historical evidence, not fictitious sessions. Missing files keep a pointer
    and a reason, so a subsequent import can finish archiving them.
    """
    db.executescript(SCHEMA)
    root = root or os.path.dirname(os.path.abspath(db.execute("PRAGMA database_list").fetchone()[2]))
    user = recdata.kv_get(db, "user_id")
    count = 0
    for item in raw:
        if "map" in item or item.get("user_id") != user:
            continue
        sha, played = item.get("hash"), item.get("date")
        t = recdata.timestamp(played)
        if not sha or t is None or not recdata.tracking_allowed(db, t):
            continue
        st = json.loads(item.get("stats") or "{}")
        stats = [int(st.get(k, 0)) for k in ("perfect", "great", "good", "ok", "meh", "miss")]
        key = recdata.score_key(sha, item.get("total") or 0, stats, played)
        score = db.execute("SELECT key,fresh FROM scores WHERE key=? OR local_id=? ORDER BY fresh DESC LIMIT 1",
                           (key, item.get("id"))).fetchone()
        if not score:
            continue
        key = score["key"]
        tracked = score["fresh"] or db.execute(
            "SELECT 1 FROM events WHERE kind='finish' AND json_extract(info,'$.key')=? LIMIT 1", (key,)).fetchone()
        if not tracked:
            continue
        replay_sha = item.get("replay_sha256") or item.get("replay")
        previous = db.execute("SELECT replay_sha256 FROM replay_evidence WHERE score_key=? AND status='archived'",
                              (key,)).fetchone()
        if not replay_sha and previous:
            replay_sha = previous[0]  # a cleaned game-side reference must not erase our archive
        path, status, error = None, "missing", None
        if isinstance(replay_sha, str) and re.fullmatch(r"[0-9a-f]{64}", replay_sha):
            relative = os.path.join("replays", replay_sha[:2], replay_sha + ".osr")
            target = os.path.join(root, relative)
            source = recdata.local_file(replay_sha)
            try:
                if not os.path.isfile(target):
                    with open(source, "rb") as fh:
                        data = fh.read()
                    if hashlib.sha256(data).hexdigest() != replay_sha:
                        raise ValueError("replay hash mismatch")
                    os.makedirs(os.path.dirname(target), exist_ok=True)
                    fd, tmp = tempfile.mkstemp(prefix=".replay-", dir=os.path.dirname(target))
                    try:
                        with os.fdopen(fd, "wb") as fh:
                            fh.write(data)
                            fh.flush()
                            os.fsync(fh.fileno())
                        os.replace(tmp, target)
                    finally:
                        if os.path.exists(tmp):
                            os.unlink(tmp)
                    count += 1
                path, status = relative, "archived"
            except (OSError, ValueError, TypeError) as exc:
                error = str(exc)[:240]
        else:
            error = "lazer has no replay file yet"
        pauses = item.get("pauses") or []
        with db:
            db.execute("INSERT OR REPLACE INTO replay_evidence VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                       (key, item.get("id"), sha, item.get("md5"), replay_sha, path,
                        json.dumps(pauses), item.get("mods") or "[]",
                        recdata.variant(json.loads(item.get("mods") or "[]"))[0], played, status, error))
    return count


def recent(db):
    """Small post-result catch-up: copy/read Realm, import recent scores, archive.

    The caller schedules this while gameplay is idle. No full library scan and
    no replay simulation happens in the observation/rendering thread.
    """
    import datetime
    import time
    import lazer_scores
    since = datetime.datetime.fromtimestamp(time.time() - 3 * 86400, datetime.timezone.utc).isoformat()
    raw = lazer_scores.dump(recent=since)
    rows, _maps = recdata.realm_rows(raw, recdata.kv_get(db, "user_id"))
    with db:
        for row in rows:
            recdata.upsert_score(db, row)
    # recover_results needs only the existing recorded start, never predicts retrospectively.
    import recommend
    recommend.recover_results(db)
    return archive(db, raw)
