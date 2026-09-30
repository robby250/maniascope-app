"""Rate the community dan courses with this calculator → calib/dans.json (the dan display's anchors).

    python3 calib/dan_table.py [COURSE_DIR]

COURSE_DIR holds the course charts by SHA-256 (any layout; default ~/.cache/maniascope/dans).
Course files are mappers' work and stay out of the repository; calib/dan_courses.tsv names them.
Rerun after any calculator change: dans.py ignores a table from another calculator.
"""
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import native_backend  # noqa: E402
native_backend.activate()
import recdata  # noqa: E402
import skill_calc  # noqa: E402


def main(argv):
    root = os.path.expanduser(argv[0] if argv else "~/.cache/maniascope/dans")
    files = {f: os.path.join(d, f) for d, _, fs in os.walk(root) for f in fs}
    series = {}
    for row in csv.DictReader(open(os.path.join(HERE, "dan_courses.tsv")), delimiter="\t"):
        path = files.get(row["sha256"]) or files.get(row["sha256"] + ".osu")
        if not path:
            print("missing", row["course"])
            continue
        res = skill_calc.compute(skill_calc.parse_osu(path), 1.0)
        rating = res["scores"]["overall"]
        series.setdefault(row["series"], []).append(
            {"order": float(row["order"]), "tier": row["tier"], "rating": round(rating, 3), "course": row["course"]})
        print(f"{row['series']:11} {row['tier']:9} {rating:6.2f}  {row['course']}", flush=True)
    for s, tiers in series.items():
        tiers.sort(key=lambda t: t["order"])
        flips = [(a["tier"], b["tier"]) for a, b in zip(tiers, tiers[1:]) if b["rating"] <= a["rating"]]
        if flips:
            print(f"NOT MONOTONE {s}: {flips}")
    with open(os.path.join(HERE, "dans.json"), "w") as fh:
        json.dump({"calc": recdata.calc_id(), "series": series}, fh, indent=1, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
