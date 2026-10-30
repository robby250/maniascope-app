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
