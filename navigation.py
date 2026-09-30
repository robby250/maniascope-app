"""Explicit Next actions. Clipboard ownership stays on GTK's main thread.

Search syntax checked against ppy/osu FilterQueryParser.cs and FilterCriteria.cs:
quoted values followed by ! match the complete field; backslash-escaped quotes
are NOT supported. A local SHA/MD5 is an identity, not a song-search keyword.
"""
import os
import re
import subprocess

import recdata

ACTIONS = ("auto", "copy", "link")


def positive_id(value):
    try:
        n = int(value)
        return n if n > 0 and not isinstance(value, bool) else None
    except (ValueError, TypeError):
        return None


def same_map(c, *, sha=None, md5=None, bid=None):
    """A submitted chart can update its bytes without changing its map ID."""
    return bool(c and ((sha and sha == (c.get('sha') or c.get('sha256')))
                       or (md5 and md5 == c.get('md5'))
                       or (positive_id(bid) and positive_id(bid) == positive_id(c.get('bid')))))


def _field(name, value):
    value = str(value or "").strip()
    if not value:
        return ""
    # The lazer parser has no quoted-string escape convention. A literal safe
    # fragment is preferable to a syntactically broken 'escaped' exact match.
    fragments = re.split(r'["\r\n\t]', value)
    fragment = max(fragments, key=len).strip()
    if not fragment:
        return ""
    exact = len(fragments) == 1
    return f'{name}="{fragment}"' + ("!" if exact else "")


def song_search(c):
    """Search the local difficulty; intended playback rate is deliberately absent."""
    md = dict(c.get("search_meta") or {})
    if not md:
        md = {"Artist": c.get("artist"), "Title": c.get("song_title"),
              "Version": c.get("version"), "Creator": c.get("creator")}
    if not md.get("Version"):
        path = recdata.local_file(c.get("sha256") or c.get("sha"))
        if path and os.path.isfile(path):
            import lazer_index
            md = lazer_index._read_metadata(path) or md
    fields = [_field(k, md.get(v)) for k, v in
              (("artist", "Artist"), ("title", "Title"), ("diff", "Version"), ("creator", "Creator"))]
    query = " ".join(f for f in fields if f)
    if query:
        return query + (f" cs={int(c['keys'])}" if c.get("keys") else "")
    bid = positive_id(c.get("bid"))
    return str(bid) if bid else str(c.get("title") or "").strip()


def lazer_running():
    import paths
    if paths.WINDOWS:
        try:
            out = subprocess.run(["tasklist", "/FI", "IMAGENAME eq osu!.exe", "/NH"], capture_output=True,
                                 text=True, timeout=5, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            return "osu!.exe" in out.stdout
        except (OSError, subprocess.SubprocessError):
            return False
    import glob
    for path in glob.glob("/proc/[0-9]*/comm"):
        try:
            with open(path) as fh:
                if fh.read().strip() == "osu!":
                    return True
        except OSError:
            pass
    return False


def open_url(url):
    """Hand an osu:// link to the running lazer (a second lazer process only forwards it over IPC)."""
    import paths
    if paths.WINDOWS:
        os.startfile(url)
        return
    flatpak = os.path.isdir(os.path.expanduser("~/.var/app/sh.ppy.osu"))
    subprocess.Popen(["flatpak", "run", "sh.ppy.osu", url] if flatpak else ["xdg-open", url],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def perform(c, action="auto"):
    """Return copy/msg/sent, never start the game or automatically change mods."""
    if c.get('installed'):
        path = recdata.local_file(c.get('sha') or c.get('sha256'))
        if not path or not os.path.isfile(path):
            if c.get('mode') in ('nps', 'skills'):
                return dict(copy=None, msg="Chart is no longer installed — refresh the library", sent=False, unavailable=True)
            c['installed'] = False
    action = action if action in ACTIONS else "auto"
    bid = positive_id(c.get("bid"))
    installed = bool(c.get("installed"))
    # A known edited revision must never open the online original instead.
    trusted = bid is not None and c.get("online_match", True)
    copy = action == "copy" or (action == "auto" and installed) or not trusted
    if copy:
        query = song_search(c)
        if not query:
            return {"copy": None, "msg": "No searchable metadata for this chart", "sent": False}
        msg = "Copied song-select search — paste in osu! and select the difficulty"
        if not installed:
            msg = "Copied search; this map is not installed — download it before searching locally"
        elif action == "link" and not trusted:
            msg = "No matching online map — copied local song-select search instead"
        return {"copy": query, "msg": msg, "sent": False}
    url = f"osu://b/{bid}"
    if not lazer_running():
        return {"copy": url, "msg": "osu! is not running — copied beatmap link", "sent": False}
    try:
        open_url(url)
    except OSError as exc:
        return {"copy": url, "msg": f"Could not open beatmap panel ({exc}) — copied link", "sent": False}
    return {"copy": None, "msg": "Sent beatmap link to osu! — choose Go to beatmap; mods unchanged", "sent": True}
