"""Fetch dan course sets (every mania difficulty: courses and their single-song diffs) for calib/dan_table.py.

    python3 calib/dan_fetch.py [OUT_DIR]        # default ~/.cache/maniascope/dans/raw

Files are saved by SHA-256 under OUT_DIR/files; OUT_DIR/manifest.tsv lists set, beatmap and difficulty
name. Mappers' work stays out of the repository. osu.direct's search is the only listing without an
API key; it finds a set by a title query, so each set is named by (query, set id).
"""
import hashlib
import json
import os
import sys
import time
import urllib.parse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import webmaps  # noqa: E402

SETS = {
    # 4K REFORM: Thaumiel's re-uploads (with INTRO-1st and the later 6th) and the Zeta/Eta single songs
    "Thaumiel REFORM": (1079991, 1079998, 1188968),
    "Dan REFORM Pack": (593983, 598858, 608287, 611978),
    "DDMythical REFORM": (616461, 625751),
    "REFORM FINAL": (1156299,),
    "REFORM Alter": (1260676,),
    # 4K vibro: both course sets and the per-song map packs, and the evaluation courses
    "Vibro Dan Courses": (563163, 564238, 973783, 973787, 2494639, 2495024, 2495355, 2495648),
    "Vibro Evaluation Courses": (659362, 659846, 659867),
    # 4K LN: v2 courses and the stage song packs
    "4K LN Dan": (891143, 891152, 891157, 891164, 1116467, 2181524, 2181595, 2181627, 2181680),
    # 7K: Jinjin's phases and the per-dan practice packs
    "Dan Phase": (450069, 450649, 451788, 895138, 930218, 1220647, 1061136),
    "Insane Level": (455715, 456346, 525735),
    "Extra Level": (477431, 529168),
    "7K Regular Dan Practice": (1877617, 1877625, 1877636, 1877727),
    "7K LN Dan Practice": (1887981, 1888000, 1888009, 1888027),
    "dan(Regular) Practice": (1681464, 1681515, 1682188, 1682211),
}


def listing(query, wanted):
    url = f"https://osu.direct/api/v2/search?q={urllib.parse.quote(query)}&mode=3&amount=100"
    found = {s["id"]: s for s in json.loads(webmaps._get(url) or b"[]") if s["id"] in wanted}
    for sid in wanted:
        if sid not in found:
            # a set the title query misses is retried by its own title words
            print("not listed", sid, flush=True)
    return found


def main(argv):
    out = os.path.expanduser(argv[0] if argv else "~/.cache/maniascope/dans/raw")
    os.makedirs(os.path.join(out, "files"), exist_ok=True)
    path = os.path.join(out, "manifest.tsv")
    done = {line.split("\t")[1] for line in open(path, encoding="utf-8")} if os.path.exists(path) else set()
    with open(path, "a", encoding="utf-8") as man:
        for query, sids in SETS.items():
            for sid, s in listing(query, sids).items():
                for b in s["beatmaps"]:
                    if b.get("mode_int") != 3 or str(b["id"]) in done:
                        continue
                    data = b""
                    for template, pause in webmaps.FILES:
                        data = webmaps._get(template.format(bid=b["id"]))
                        time.sleep(pause)
                        if data.startswith(b"osu file format"):
                            break
                    if not data.startswith(b"osu file format"):
                        print("missing", sid, b["id"], b["version"], flush=True)
                        continue
                    sha = hashlib.sha256(data).hexdigest()
                    with open(os.path.join(out, "files", sha), "wb") as fh:
                        fh.write(data)
                    man.write("\t".join((str(sid), str(b["id"]), sha, str(int(b.get("cs", 0))),
                                         s["artist"], s["title"], b["version"], s["creator"])) + "\n")
                    man.flush()
                    print(sid, b["id"], b["version"], flush=True)
            time.sleep(webmaps.PAUSE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
