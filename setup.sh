#!/usr/bin/env bash
# One-time setup for ManiaScope on Linux; run it again after `git pull` (it skips what is done).
#
# Requirements (README step 1): python3 with PyGObject (GTK 3) + pycairo + numpy, and a C compiler
# with Python headers for the fast calculator.
# Everything else is installed for this user only, under ~/.local, with no sudo.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA="$HOME/.local/share/maniascope"
ask() {   # ask "question" default(Y|N) → 0 for yes
  local yn; read -r -p "$1 [$([ "$2" = Y ] && echo Y/n || echo y/N)] " yn || true
  [[ -z "$yn" && "$2" = Y || "$yn" =~ ^[Yy] ]]
}
pipi() {  # pip install --user, also on distros that mark the system Python externally managed
  python3 -m pip install --user -q --no-warn-script-location "$@" 2>/dev/null \
    || python3 -m pip install --user -q --no-warn-script-location --break-system-packages "$@"
}

echo "== 1/5 Checking Python packages"
missing=""
python3 -c "import gi; gi.require_version('Gtk', '3.0'); from gi.repository import Gtk" 2>/dev/null || missing+=" GTK/PyGObject"
python3 -c "import cairo" 2>/dev/null || missing+=" pycairo"
python3 -c "import numpy" 2>/dev/null || missing+=" numpy"
if [ -n "$missing" ]; then
  echo "Missing:$missing. Install the packages from README step 1 for your distro, then run ./setup.sh again."
  exit 1
fi
python3 -m pip --version >/dev/null 2>&1 || { echo "pip is missing: install python3-pip (README step 1)."; exit 1; }
if ! python3 -c "import rosu_pp_py" 2>/dev/null; then
  echo "Installing rosu-pp-py (pp calculation, ~3 MB)…"
  pipi rosu-pp-py
fi

echo "== 2/5 osu!lazer"
LAZER="$(cd "$HERE" && python3 -c 'import paths; print(paths.lazer_data())')"
if [ -f "$LAZER/client.realm" ]; then
  echo "Found osu!lazer data at $LAZER"
else
  echo "osu!lazer data not found at $LAZER."
  echo "Start lazer once, or launch ManiaScope as: LAZER_DATA=/path/to/osu ./run.sh"
fi

echo "== 3/5 Score reader (reads your lazer scores and installed maps; required)"
NODE_DIR="$DATA/node"                   # lazer_scores.py puts this on PATH
if [ ! -d "$DATA/realmjs/node_modules/realm" ]; then
  if ! command -v npm >/dev/null && [ ! -x "$NODE_DIR/npm" ]; then
    arch="$(uname -m)"; arch="${arch/x86_64/x64}"; arch="${arch/aarch64/arm64}"
    base="https://nodejs.org/dist/latest-v20.x"
    file="$(curl -fsSL "$base/SHASUMS256.txt" | grep -o "node-v[0-9.]*-linux-$arch.tar.xz" | head -1)"
    echo "Downloading Node.js ($file, ~25 MB) into $DATA…"
    mkdir -p "$DATA"
    curl -fL --progress-bar -o "$DATA/$file" "$base/$file"
    (cd "$DATA" && curl -fsSL "$base/SHASUMS256.txt" | grep " $file\$" | sha256sum -c --quiet)
    rm -rf "$DATA/nodejs" && mkdir "$DATA/nodejs"
    tar -xJf "$DATA/$file" -C "$DATA/nodejs" --strip-components=1 && rm "$DATA/$file"
    ln -sfn nodejs/bin "$NODE_DIR"
  fi
  echo "Installing Realm (~150 MB)…"
  mkdir -p "$DATA/realmjs"
  (cd "$DATA/realmjs" && PATH="$NODE_DIR:$PATH" && npm init -y >/dev/null && npm install --no-fund --no-audit realm@12)
fi

echo "== 4/5 Fast calculator (map switching ~2.5× faster)"
if ! python3 -c "import Cython, setuptools" 2>/dev/null; then
  echo "Installing Cython…"
  pipi cython setuptools || true
fi
if (cd "$HERE" && nice python3 native_backend.py >/tmp/maniascope-native.log 2>&1); then
  echo "Built."
else
  echo "Could not build it (log: /tmp/maniascope-native.log). ManiaScope still works, ~2.5× slower."
  echo "Usually the C compiler or Python headers from README step 1 are missing; install them and run ./setup.sh again."
fi

echo "== 5/5 tosu and the application menu"
TOSU_DIR="$DATA/tosu"
if [ ! -x "$TOSU_DIR/tosu" ] && ask "Download tosu (lets ManiaScope see the rate you pick in lazer, ~40 MB)?" Y; then
  mkdir -p "$TOSU_DIR"
  url="$(curl -fsSL https://api.github.com/repos/tosuapp/tosu/releases/latest \
         | grep -o 'https://[^"]*tosu-linux-[^"]*\.zip' | head -1)"
  curl -fL --progress-bar -o "$TOSU_DIR/tosu.zip" "$url" && unzip -o -q "$TOSU_DIR/tosu.zip" -d "$TOSU_DIR" \
    && rm "$TOSU_DIR/tosu.zip" && chmod +x "$TOSU_DIR/tosu"
fi
APPS="$HOME/.local/share/applications"
mkdir -p "$APPS"
sed "s#@HERE@#$HERE#g" "$HERE/maniascope.desktop.in" > "$APPS/maniascope.desktop"
chmod +x "$APPS/maniascope.desktop"
update-desktop-database "$APPS" 2>/dev/null || true

echo
echo "Done. Start ManiaScope from your application menu, or with ./run.sh"
