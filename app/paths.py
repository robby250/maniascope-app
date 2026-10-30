"""Per-platform locations. Linux paths are unchanged from earlier releases; Windows uses the
usual %APPDATA% / %LOCALAPPDATA% homes. Environment overrides win everywhere."""
import os
import sys

WINDOWS = sys.platform == "win32"
_home = os.path.expanduser("~")


def _win(var, *parts):
    return os.path.join(os.environ.get(var) or os.path.join(_home, "AppData", "Roaming" if var == "APPDATA" else "Local"),
                        *parts)


def lazer_data():
    """lazer's data directory (holds client.realm, files/, logs/)."""
    if os.environ.get("LAZER_DATA"):
        return os.path.expanduser(os.environ["LAZER_DATA"])
    if WINDOWS:
        return _win("APPDATA", "osu")
    if sys.platform == "darwin":
        return os.path.join(_home, "Library", "Application Support", "osu")
    flatpak = os.path.join(_home, ".var/app/sh.ppy.osu/data/osu")
    native = os.path.join(_home, ".local/share/osu")
    return flatpak if os.path.isdir(flatpak) or not os.path.isdir(native) else native


def stable_data():
    """osu!stable's folder (osu!.db, scores.db, Songs): STABLE_DATA, else the usual install. Linux runs it under
    Wine: osu-winello's saved path, its default, then the default Wine prefix."""
    if os.environ.get("STABLE_DATA"):
        return os.path.expanduser(os.environ["STABLE_DATA"])
    if WINDOWS:
        return _win("LOCALAPPDATA", "osu!")
    found = []
    try:
        with open(os.path.join(_home, ".local/share/osuconfig/osupath"), encoding="utf-8") as fh:
            found.append(fh.read().strip())
    except OSError:
        pass
    user = os.environ.get("USER") or os.path.basename(_home)
    found += [os.path.join(_home, ".local/share/osu-wine/osu!"),
              os.path.join(_home, ".wine/drive_c/users", user, "AppData/Local/osu!")]
    return next((d for d in found if d and os.path.isfile(os.path.join(d, "osu!.db"))), found[0])


DATA = _win("LOCALAPPDATA", "maniascope") if WINDOWS else os.path.join(_home, ".local/share/maniascope")
CACHE = _win("LOCALAPPDATA", "maniascope", "cache") if WINDOWS else os.path.join(_home, ".cache/maniascope")
CONFIG = _win("APPDATA", "maniascope") if WINDOWS else os.path.join(_home, ".config/maniascope")


def lower_priority(level=10):
    """Background analysis must not steal frames from the game."""
    if hasattr(os, "nice"):
        os.nice(level)
    elif WINDOWS:
        import ctypes
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)  # BELOW_NORMAL
