"""Copy the releasable tree into DEST (a checkout of the public repo), without personal material.

    python3 packaging/export_public.py DEST

Excluded: docs/ (session reports with personal scores), AGENTS.md, calib/ tooling (only the runtime
tables ship), patches/. The public README is packaging/PUBLIC_README.md. Existing DEST files that are
no longer released are removed; DEST/.git is untouched.
"""
import os
import shutil
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
KEEP_CALIB = {"calib/difficulty.json", "calib/dans.json", "calib/dan_courses.tsv", "calib/dan_table.py",
              "calib/dan_fetch.py"}


def released():
    files = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.split("\n")
    for f in filter(None, files):
        top = f.split("/")[0]
        if top in ("docs", "patches") or f in ("AGENTS.md", "README.md") or \
                (top == "calib" and f not in KEEP_CALIB) or f == "packaging/PUBLIC_README.md":
            continue
        yield f


def main(dest):
    dest = os.path.abspath(dest)
    wanted = set(released())
    for d, dirs, files in os.walk(dest):
        dirs[:] = [x for x in dirs if x != ".git"]
        for f in files:
            rel = os.path.relpath(os.path.join(d, f), dest)
            if rel not in wanted and rel not in ("README.md", os.path.join(".github", "workflows", "windows.yml"),
                                                 os.path.join("packaging", "calc_id.expected")):
                os.remove(os.path.join(d, f))
    for rel in wanted:
        os.makedirs(os.path.dirname(os.path.join(dest, rel)) or dest, exist_ok=True)
        shutil.copy2(os.path.join(ROOT, rel), os.path.join(dest, rel))
    shutil.copy2(os.path.join(ROOT, "packaging", "PUBLIC_README.md"), os.path.join(dest, "README.md"))
    # Pushing workflow files needs the "workflow" token scope, so the private repo keeps it as a template.
    os.makedirs(os.path.join(dest, ".github", "workflows"), exist_ok=True)
    shutil.copy2(os.path.join(ROOT, "packaging", "windows-workflow.yml"), os.path.join(dest, ".github", "workflows", "windows.yml"))
    # CI checks the frozen identity against the one whose public-<calc>.pkl this machine publishes.
    sys.path.insert(0, ROOT)
    import recdata
    with open(os.path.join(dest, "packaging", "calc_id.expected"), "w") as fh:
        fh.write(recdata.calc_id() + "\n")
    print(f"{len(wanted) + 1} files → {dest}")


if __name__ == "__main__":
    main(sys.argv[1])
