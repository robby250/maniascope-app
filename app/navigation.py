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
    if recdata.client() == "stable":
        # stable has no quoted field values: its words match artist/title/difficulty/creator/tags, keys= filters
        # (osu! wiki, Beatmap search)
        words = " ".join(str(md.get(k) or "") for k in ("Artist", "Title", "Version"))
        query = " ".join(re.sub(r'["=<>!]', " ", words).split())
        return (query + (f" keys={int(c['keys'])}" if c.get("keys") else "")) if query else ""
    fields = [_field(k, md.get(v)) for k, v in
              (("artist", "Artist"), ("title", "Title"), ("diff", "Version"), ("creator", "Creator"))]
    query = " ".join(f for f in fields if f)
    if query:
        return query + (f" cs={int(c['keys'])}" if c.get("keys") else "")
    bid = positive_id(c.get("bid"))
    return str(bid) if bid else str(c.get("title") or "").strip()


def _osu_exes():
    """Full paths of the running osu!.exe processes (Windows). lazer and stable share that file name."""
    import ctypes
    from ctypes import wintypes as w
    k = ctypes.WinDLL("kernel32")
    k.OpenProcess.restype, k.OpenProcess.argtypes = w.HANDLE, (w.DWORD, w.BOOL, w.DWORD)
    k.QueryFullProcessImageNameW.argtypes = (w.HANDLE, w.DWORD, w.LPWSTR, w.PDWORD)
    k.CloseHandle.argtypes = (w.HANDLE,)
    pids, used = (w.DWORD * 16384)(), w.DWORD()
    if not k.K32EnumProcesses(pids, ctypes.sizeof(pids), ctypes.byref(used)):
        return []
    found = []
    for pid in pids[:used.value // ctypes.sizeof(w.DWORD)]:
        handle = k.OpenProcess(0x1000, False, pid)      # PROCESS_QUERY_LIMITED_INFORMATION
        if handle:
            name, size = ctypes.create_unicode_buffer(32768), w.DWORD(32768)
            if k.QueryFullProcessImageNameW(handle, 0, name, ctypes.byref(size)) \
                    and os.path.basename(name.value).lower() == "osu!.exe":
                found.append(name.value)
            k.CloseHandle(handle)
    return found


def client_exe():
    """The active client's running osu!.exe (Windows); stable's sits beside its osu!.db."""
    stable = recdata.client() == "stable"
    return next((e for e in _osu_exes() if os.path.isfile(os.path.join(os.path.dirname(e), "osu!.db")) == stable), None)


def lazer_running():
    """The active client is running (stable is osu!.exe under Wine)."""
    import paths
    if paths.WINDOWS:
        return client_exe() is not None
    import glob
    for path in glob.glob("/proc/[0-9]*/comm"):
        try:
            with open(path) as fh:
                if fh.read().strip() == ("osu!.exe" if recdata.client() == "stable" else "osu!"):
                    return True
        except OSError:
            pass
    return False


def open_url(url):
    """Hand an osu:// link to the running client: a second process of the same exe only forwards it over IPC.
    On Windows never through os.startfile: osu:// belongs to whichever client registered it, so a lazer player
    got stable launched (user 2026-10-31)."""
    import paths
    if paths.WINDOWS:
        exe = client_exe()
        if not exe:
            raise OSError("osu! has closed")
        command = [exe, url]
    elif recdata.client() == "lazer" and os.path.isdir(os.path.expanduser("~/.var/app/sh.ppy.osu")):
        command = ["flatpak", "run", "sh.ppy.osu", url]
    else:
        command = ["xdg-open", url]
    subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


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
        msg = "Search copied — press Ctrl+V in song select"
        if not installed:
            msg = "Search copied, but you don't have this map — download it, then Ctrl+V in song select"
        elif action == "link" and not trusted:
            msg = "No matching online map — search copied, press Ctrl+V in song select"
        return {"copy": query, "msg": msg, "sent": False}
    url = f"osu://b/{bid}"
    if not lazer_running():
        return {"copy": url, "msg": "osu! is not running — copied beatmap link", "sent": False}
    try:
        open_url(url)
    except OSError as exc:
        return {"copy": url, "msg": f"Could not open beatmap panel ({exc}) — copied link", "sent": False}
    # Auto still copies the search, so after downloading it can be pasted in song select (user 2026-10-09).
    query = song_search(c) if action == "auto" else None
    return {"copy": query or None, "msg": "Sent beatmap link to osu! — choose Go to beatmap; mods unchanged"
            + (" · search copied, Ctrl+V in song select" if query else ""), "sent": True}
