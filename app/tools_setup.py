"""One-click helpers for the optional external tools (same steps setup.sh does on Linux)."""
import io
import json
import os
import re
import shutil
import stat
import urllib.request
import zipfile

import paths

UA = {"User-Agent": "maniascope"}


def tosu_bin():
    return os.path.join(paths.DATA, "tosu", "tosu.exe" if paths.WINDOWS else "tosu")


def install_tosu():
    """Latest official tosu release for this platform → paths.DATA/tosu. Returns the executable path.
    tosu (LGPL-3.0) stays a separate program downloaded from its own releases, never bundled.
    The official build is right for Windows: patches/tosu-proc-read.patch fixes a crash in the
    Linux-only /proc reader, and tosu-loop-waits.patch only shortens song-select polling."""
    req = urllib.request.Request("https://api.github.com/repos/tosuapp/tosu/releases/latest", headers=UA)
    with urllib.request.urlopen(req, timeout=20) as resp:
        release = json.load(resp)
    want = "tosu-windows-" if paths.WINDOWS else "tosu-linux-"
    asset = next(a for a in release["assets"] if a["name"].startswith(want) and a["name"].endswith(".zip"))
    with urllib.request.urlopen(urllib.request.Request(asset["browser_download_url"], headers=UA), timeout=120) as resp:
        data = resp.read()
    target = os.path.dirname(tosu_bin())
    os.makedirs(target, exist_ok=True)
    zipfile.ZipFile(io.BytesIO(data)).extractall(target)
    exe = tosu_bin()
    if not paths.WINDOWS:
        os.chmod(exe, os.stat(exe).st_mode | stat.S_IXUSR)
    return exe


def set_tosu_env(path, values):
    """Set KEY=value lines in tosu.env, keeping the others; returns whether the file changed."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            old = fh.read()
    except OSError:
        old = ""
    new = old
    for key, value in values.items():
        line = f"{key}={value}"
        new, hits = re.subn(rf"(?m)^{key}=.*$", line, new)
        if not hits:
            new += ("" if not new or new.endswith("\n") else "\n") + line + "\n"
    if new != old:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(new)
    return new != old


def overlay_dir():
    """The ManiaScope page in tosu's static folder (its default on Windows: ./static beside tosu.exe)."""
    return os.path.join(os.path.dirname(tosu_bin()), "static", "ManiaScope")


def install_overlay():
    shutil.copytree(os.path.join(os.path.dirname(os.path.abspath(__file__)), "overlay", "ManiaScope"),
                    overlay_dir(), dirs_exist_ok=True)


def write_overlay_state(state):
    """state.json for the page; tmp + replace, so a fetch never reads half a file. A busy file skips one update."""
    path = os.path.join(overlay_dir(), "state.json")
    try:
        with open(path + ".tmp", "w", encoding="utf-8") as fh:
            json.dump(state, fh)
        os.replace(path + ".tmp", path)
    except OSError:
        pass
