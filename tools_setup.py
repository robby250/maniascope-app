"""One-click helpers for the optional external tools (same steps setup.sh does on Linux)."""
import io
import json
import os
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
