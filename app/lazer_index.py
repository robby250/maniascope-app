#!/usr/bin/env python3
"""
Build/cache an index of osu!lazer's hash-named file store, mapping the human
string lazer prints in its log ("Artist - Title (Creator) [Version]") to the
on-disk .osu path.

lazer stores imported files by content hash with no extension, and the
metadata->file mapping lives in its realm DB. Rather than parse realm (binary,
schema-versioned, fragile), we scan the .osu files once and key each by the same
string lazer's BeatmapInfo.ToString() produces — for BOTH romanised and unicode
metadata, so it matches regardless of the user's "prefer unicode" setting.

The index is cached to ~/.cache/maniascope/index.json and rebuilt on demand
(a selected map that is missing from it). Only used when tosu is not running;
with tosu the selected file path comes straight from the game.
"""
import json
import os
from concurrent.futures import ThreadPoolExecutor

import paths  # noqa: E402
DEFAULT_STORE = os.path.expanduser(os.environ.get("LAZER_FILES", os.path.join(paths.lazer_data(), "files")))
CACHE_DIR = paths.CACHE
CACHE_FILE = os.path.join(CACHE_DIR, "index.json")

_META_KEYS = ("Title", "TitleUnicode", "Artist", "ArtistUnicode", "Creator", "Version", "BeatmapID", "BeatmapSetID")


def _read_metadata(path):
    """Return dict of metadata fields from an .osu, or None if not an .osu."""
    md = {}
    try:
        with open(path, "rb") as fh:
            if fh.read(17) != b"osu file format v":
                return None
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            in_meta = False
            for line in fh:
                line = line.rstrip("\n").rstrip("\r")
                if line.startswith("["):
                    if line == "[Metadata]":
                        in_meta = True
                        continue
                    # stop once we've passed metadata
                    if in_meta:
                        break
                    continue
                if in_meta and ":" in line:
                    k, v = line.split(":", 1)
                    if k in _META_KEYS:
                        md[k] = v.strip()
    except OSError:
        return None
    return md


def _compose_keys(md):
    """Compose the lazer-style display strings (romanised + unicode) for an .osu."""
    title_r = md.get("Title", "")
    title_u = md.get("TitleUnicode") or title_r
    artist_r = md.get("Artist", "")
    artist_u = md.get("ArtistUnicode") or artist_r
    creator = md.get("Creator", "")
    version = md.get("Version", "")
    keys = set()
    for artist, title in ((artist_r, title_r), (artist_u, title_u)):
        keys.add(f"{artist} - {title} ({creator}) [{version}]")
    return keys


def build_index(store=DEFAULT_STORE, progress=None):
    """Scan the store and return {display_string: path}. Calls progress(done,total)."""
    paths = []
    for dp, _dn, fns in os.walk(store):
        for fn in fns:
            paths.append(os.path.join(dp, fn))
    total = len(paths)
    index = {}
    done = 0

    def work(p):
        md = _read_metadata(p)
        return (p, md)

    with ThreadPoolExecutor(max_workers=32) as ex:
        for p, md in ex.map(work, paths, chunksize=64):
            done += 1
            if md:
                for k in _compose_keys(md):
                    index.setdefault(k, p)
            if progress and done % 5000 == 0:
                progress(done, total)
    if progress:
        progress(total, total)
    return index


def load_or_build(store=DEFAULT_STORE, progress=None, force=False):
    """Load the cached index, else (re)build and cache it.

    The cache is trusted as-is (no store walk on launch); callers pass force=True
    when a selected map is missing. A cache written under another alias of the
    same store (symlink / bind mount) is re-pointed instead of rebuilt.
    """
    store = os.path.realpath(store)
    if not force and os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as fh:
                cached = json.load(fh)
            old, index = cached["store"], cached["index"]
            if old != store:
                index = {k: store + p[len(old):] if p.startswith(old) else p
                         for k, p in index.items()}
            if any(os.path.exists(p) for p in list(index.values())[:20]):
                return index
        except (OSError, ValueError, KeyError):
            pass
    index = build_index(store, progress)
    os.makedirs(CACHE_DIR, exist_ok=True)
    tmp = CACHE_FILE + ".part"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump({"store": store, "index": index}, fh)
    os.replace(tmp, CACHE_FILE)
    return index


if __name__ == "__main__":
    import sys, time
    t0 = time.time()

    def prog(d, t):
        print(f"\r  indexing {d}/{t}…", end="", file=sys.stderr, flush=True)

    idx = load_or_build(progress=prog, force="--force" in sys.argv)
    print(f"\nindexed {len(idx)} display-keys in {time.time()-t0:.1f}s", file=sys.stderr)
    # optional lookup test
    q = next((a for a in sys.argv[1:] if not a.startswith("-")), None)
    if q:
        print(idx.get(q, "NOT FOUND"))
